"""AutoResearcher — infinite-loop hypothesis-driven improvement agent.

The researcher runs continuously, generating and testing hypotheses that try to
improve a user-defined score. Each iteration:

  1. Reads memory for past hypothesis outcomes (to avoid repeating mistakes).
  2. Formulates ONE specific, novel hypothesis.
  3. Delegates implementation to specialist agents (each in an isolated git worktree).
  4. Calls ``evaluate_hypothesis`` — which merges all worktrees into the main
     workspace, then invokes the user-supplied ``evaluate()`` function.
  5. If the score improved, calls ``save_hypothesis`` (user-supplied ``save()``).
  6. If not, reverts the main workspace to the pre-iteration checkpoint.
  7. Records the outcome in shared memory and loops.

User tools interface (see workspace_tools_example.py)::

    def evaluate(workspace_path: str) -> dict | float:
        # return {"score": float, "details": str}  or just a float

    def save(workspace_path: str, hypothesis: str, score: float, iteration: int) -> None:
        # called once per improved hypothesis

Run::

    from coordinator.researcher import AutoResearcher

    researcher = AutoResearcher(
        workspace_dir="/path/to/project",
        user_tools_path="workspace_tools.py",
    )
    researcher.run("Maximise test coverage above 90%.")
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from agent.core import OllamaDeepAgent
from coordinator.config import AgentSpec, load_config
from coordinator.git_worktree import WorktreeManager
from deepagents import CompiledSubAgent
from storage.memory_store import SharedMemoryStore
from tools.web_tools import WebConfig

logger = logging.getLogger(__name__)


# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an autonomous research coordinator. Your purpose is to iteratively \
improve a software project by generating and testing hypotheses.

## Your specialists
{roster}

## Per-iteration workflow

1. Call ``list_past_hypotheses`` to review all previous attempts.
   Identify what has been tried, what scores were achieved, and why things failed.

2. Formulate ONE specific, novel hypothesis:
   - It MUST differ from every hypothesis in the history.
   - Be concrete: name the exact files, functions, or algorithms to change.
   - Explain briefly why this change should improve the score.

3. Delegate implementation to your specialists using the ``task`` tool.
   Give each specialist a precise, self-contained instruction including:
   the workspace path, which files to touch, and the expected output.
   Divide the work so specialists do not conflict.

4. After all specialists report back, call ``evaluate_hypothesis``.
   This merges their work and runs the evaluation suite.
   It returns the score and evaluation details.

5. If the score improved (see "Current best score" in the task prompt):
   call ``save_hypothesis`` with a clear description and the achieved score.

6. Return a concise summary:
   - The hypothesis you tested
   - Score achieved vs. the current best
   - Key insight: why it worked or failed
   - What you intend to try next

## Rules
- NEVER implement anything yourself — always delegate to specialists.
- ALWAYS call ``evaluate_hypothesis`` before reporting done.
- NEVER repeat a hypothesis. Learn from failures.
- Use ``list_past_hypotheses`` before every new hypothesis.
"""

_ROSTER_ENTRY = "- **{name}** — expertise: {expertise}\n  {description}"


def _build_roster(agents: list[AgentSpec]) -> str:
    return "\n".join(
        _ROSTER_ENTRY.format(
            name=a.name,
            expertise=a.expertise_line(),
            description=a.description.strip(),
        )
        for a in agents
    )


def _specialist_prompt(spec: AgentSpec) -> str:
    return (
        f"You are {spec.name}, a specialist AI agent.\n"
        f"Expertise: {spec.expertise_line()}.\n\n"
        f"Work autonomously to completion. Use your tools to read files, write files, "
        f"and run shell commands. Verify your own work (run tests, check output). "
        f"When done, provide a concise summary of what you changed and why.\n"
        f"Stay inside the workspace directory you were given."
    )


def _build_iteration_prompt(
    *,
    iteration: int,
    goal: str,
    best_score: float,
    history: list[dict],
    codebase_summary: str,
    workspace: str,
) -> str:
    if history:
        history_lines = "\n".join(
            f"  [{h['status']}] iter {h['iteration']:>3}: "
            f"score={h['score']:.4f}  — {h['summary'][:160]}"
            for h in history[-25:]          # cap to avoid prompt bloat
        )
    else:
        history_lines = "  (none yet)"

    return (
        f"## Iteration {iteration}\n\n"
        f"**Goal:** {goal}\n\n"
        f"**Workspace:** {workspace}\n\n"
        f"**Current best score:** {best_score:.4f}\n\n"
        f"**Hypothesis history (most recent 25):**\n{history_lines}\n\n"
        f"**Codebase summary:**\n{codebase_summary[:900]}\n\n"
        f"Generate a new hypothesis, implement it via your specialists, then call "
        f"``evaluate_hypothesis``. If score > {best_score:.4f}, call ``save_hypothesis``."
    )


# ─── Git helpers ──────────────────────────────────────────────────────────────

def _git(workspace: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=workspace, capture_output=True, text=True, check=check,
    )


def _git_head(workspace: Path) -> str | None:
    r = _git(workspace, "rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def _git_reset_hard(workspace: Path, commit: str) -> None:
    _git(workspace, "reset", "--hard", commit)
    _git(workspace, "clean", "-fd")


def _git_commit_checkpoint(workspace: Path, iteration: int, score: float) -> None:
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m",
         f"research(iter={iteration}): score={score:.4f}")


def _reset_worktrees_to(worktree_mgr: WorktreeManager, commit: str) -> None:
    """Reset all specialist worktrees to a specific commit so next iteration starts clean."""
    for _name, (_branch, wt_path) in worktree_mgr._worktrees.items():
        _git(wt_path, "reset", "--hard", commit)
        _git(wt_path, "clean", "-fd")


# ─── User tools loader ────────────────────────────────────────────────────────

def _load_user_tools(path: Path) -> tuple[Callable, Callable]:
    """Dynamically import evaluate() and save() from the user's tools file."""
    if not path.exists():
        raise FileNotFoundError(
            f"User tools file not found: {path}\n"
            "Create a Python file exporting:\n"
            "  evaluate(workspace_path: str) -> dict | float\n"
            "  save(workspace_path: str, hypothesis: str, score: float, iteration: int) -> None\n"
            "See workspace_tools_example.py for a template."
        )
    spec = importlib.util.spec_from_file_location("_researcher_user_tools", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    for fn_name in ("evaluate", "save"):
        if not hasattr(mod, fn_name):
            raise AttributeError(
                f"{path} must define '{fn_name}'. See workspace_tools_example.py."
            )
    return mod.evaluate, mod.save


# ─── AutoResearcher ───────────────────────────────────────────────────────────

class AutoResearcher:
    """Infinite-loop hypothesis-driven improvement coordinator.

    Parameters
    ----------
    workspace_dir:
        Path to the project being improved. Agents read and write here.
        Must be (or will be initialised as) a git repository.
    user_tools_path:
        Python file exporting ``evaluate(workspace_path)`` and
        ``save(workspace_path, hypothesis, score, iteration)``.
    config_path:
        Coordinator config YAML with model roster (default: coordinator/config.yml).
    storage_path:
        LanceDB directory for shared agent memory (default: data/lancedb).
    memory_store:
        Pre-built SharedMemoryStore to reuse across runs.
    debug:
        Enable verbose LangGraph logging.
    """

    def __init__(
        self,
        workspace_dir: str | Path,
        user_tools_path: str | Path,
        config_path: str | Path = "coordinator/config.yml",
        storage_path: str | Path = "data/lancedb",
        memory_store: SharedMemoryStore | None = None,
        web_config: WebConfig | None = None,
        debug: bool = False,
    ) -> None:
        self._workspace = Path(workspace_dir).resolve()
        self._debug = debug
        self._web_config = web_config or WebConfig()
        self._current_iteration: int = 0

        # Shared mutable state written by tool closures, read by the Python loop
        self._last_score: float = -float("inf")
        self._last_eval_details: str = ""
        self._save_called: bool = False

        # Load user-defined hooks
        self._eval_fn, self._save_fn = _load_user_tools(Path(user_tools_path))

        # Load coordinator config (model roster)
        cfg = load_config(config_path)

        # Shared memory store
        self._store = memory_store or SharedMemoryStore(
            data_path=storage_path,
            embedding_model="nomic-embed-text",
            ollama_base_url=cfg.coordinator.base_url,
        )

        # Git worktrees — one per specialist, inside the workspace repo
        self._worktree_mgr = WorktreeManager(self._workspace, base_branch="main")
        self._worktree_mgr.ensure_repo()

        # Build specialist CompiledSubAgents (each lives in its own worktree)
        compiled: list[CompiledSubAgent] = []
        for spec in cfg.agents:
            wt_path = self._worktree_mgr.create_worktree(spec.name)
            agent = OllamaDeepAgent(
                model_name=spec.model,
                base_url=spec.base_url,
                temperature=spec.temperature,
                num_ctx=spec.num_ctx,
                workspace_dir=wt_path,
                memory_store=self._store,
                name=spec.name,
                require_permission=False,
                persistent_memory=True,
                system_prompt=_specialist_prompt(spec),
                web_config=self._web_config,
                debug=debug,
            )
            compiled.append(CompiledSubAgent(
                name=spec.name,
                description=spec.description.strip(),
                runnable=agent.graph,
            ))

        # Researcher-specific extra tools
        extra_tools = self._make_tools()

        # Build the coordinator/researcher agent
        system_prompt = _SYSTEM_PROMPT.format(roster=_build_roster(cfg.agents))
        self._agent = OllamaDeepAgent(
            model_name=cfg.coordinator.model,
            base_url=cfg.coordinator.base_url,
            temperature=cfg.coordinator.temperature,
            num_ctx=cfg.coordinator.num_ctx,
            workspace_dir=self._workspace,
            memory_store=self._store,
            name="researcher",
            require_permission=False,
            persistent_memory=True,
            system_prompt=system_prompt,
            subagents=compiled,
            extra_tools=extra_tools,
            web_config=self._web_config,
            debug=debug,
        )
        self._cfg = cfg

    # ─── Public API ───────────────────────────────────────────────────────────

    def run(self, goal: str) -> None:
        """Start the infinite research loop.  Press Ctrl+C to stop.

        Args:
            goal: What the researcher should try to improve, e.g.
                  "Maximise test pass rate above 95%."
        """
        print(f"  Understanding codebase at {self._workspace} …")
        codebase_summary = self._understand_codebase(goal)
        print("  Codebase understood.  Starting research loop.\n")

        best_score: float = -float("inf")
        history: list[dict] = []

        try:
            while True:
                self._current_iteration += 1
                n = self._current_iteration

                # Reset shared state for this iteration
                self._last_score = -float("inf")
                self._last_eval_details = ""
                self._save_called = False

                # Checkpoint: record HEAD before agents touch anything
                checkpoint = _git_head(self._workspace)

                # Reset specialist worktrees to current HEAD so each
                # iteration starts from the best-so-far state.
                if checkpoint:
                    _reset_worktrees_to(self._worktree_mgr, checkpoint)

                # Build the per-iteration prompt with full context
                prompt = _build_iteration_prompt(
                    iteration=n,
                    goal=goal,
                    best_score=best_score,
                    history=history,
                    codebase_summary=codebase_summary,
                    workspace=str(self._workspace),
                )

                print(f"[iter {n}] Running …", flush=True)
                result = self._agent.run(prompt, thread_id=str(uuid.uuid4()))

                # If the LLM forgot to call evaluate_hypothesis, do it ourselves.
                if self._last_score == -float("inf"):
                    logger.warning("evaluate_hypothesis was not called — running eval now.")
                    self._force_eval()

                score = self._last_score
                details = self._last_eval_details

                if score > best_score:
                    best_score = score
                    # Commit the improvement to main (if workspace is a git repo)
                    if checkpoint:
                        _git_commit_checkpoint(self._workspace, n, score)
                    # Call save if LLM didn't
                    if not self._save_called:
                        try:
                            self._save_fn(str(self._workspace), result[:400], score, n)
                        except Exception as exc:
                            logger.warning("save_fn failed: %s", exc)
                    self._store.add_memory(
                        f"[IMPROVED] iter={n} score={score:.4f}: {result[:300]}",
                        agent_name="researcher",
                        tags=["hypothesis", "improved"],
                    )
                    history.append({
                        "iteration": n, "score": score,
                        "status": "IMPROVED",
                        "summary": result[:300],
                    })
                    print(
                        f"[iter {n}] ✓  IMPROVED  score={score:.4f}"
                        + (f"  ({details[:80]})" if details else "")
                    )
                else:
                    # Revert main workspace to pre-iteration checkpoint
                    if checkpoint:
                        _git_reset_hard(self._workspace, checkpoint)
                        _reset_worktrees_to(self._worktree_mgr, checkpoint)
                    self._store.add_memory(
                        f"[NO_IMPROVEMENT] iter={n} score={score:.4f}: {result[:300]}",
                        agent_name="researcher",
                        tags=["hypothesis", "failed"],
                    )
                    history.append({
                        "iteration": n, "score": score,
                        "status": "NO_IMPROVEMENT",
                        "summary": result[:300],
                    })
                    print(
                        f"[iter {n}] ✗  no improvement  score={score:.4f}"
                        + (f"  ({details[:80]})" if details else "")
                    )

        except KeyboardInterrupt:
            print(f"\n  Research stopped after {self._current_iteration} iterations.")
            print(f"  Best score: {best_score:.4f}")
            self.cleanup()

    def cleanup(self) -> None:
        """Remove all specialist git worktrees."""
        self._worktree_mgr.cleanup()

    @property
    def memory_store(self) -> SharedMemoryStore:
        return self._store

    # ─── Codebase understanding ───────────────────────────────────────────────

    def _understand_codebase(self, goal: str) -> str:
        # Check if we already have a cached summary
        try:
            note = self._store.get_note("researcher:codebase_summary")
            if note:
                return note["value"]
        except Exception:
            pass

        prompt = (
            f"Explore the codebase at {self._workspace}. Read important files "
            f"(README, key source files, tests, config). "
            f"The research goal is: {goal}\n\n"
            f"Produce a concise technical summary (≤500 words) covering: "
            f"project structure, key components, current approach, "
            f"and which parts are most relevant to the goal."
        )
        summary = self._agent.run(prompt, thread_id=str(uuid.uuid4()))
        try:
            self._store.save_note(
                "researcher:codebase_summary", summary[:1200], agent_name="researcher"
            )
        except Exception:
            pass
        return summary

    # ─── Force eval (fallback when LLM skips it) ─────────────────────────────

    def _force_eval(self) -> None:
        # First merge whatever the specialists produced
        try:
            self._worktree_mgr.merge_all()
        except Exception:
            pass
        try:
            raw = self._eval_fn(str(self._workspace))
            if isinstance(raw, dict):
                self._last_score = float(raw.get("score", -float("inf")))
                self._last_eval_details = raw.get("details", "")
            else:
                self._last_score = float(raw)
        except Exception as exc:
            logger.warning("Force eval failed: %s", exc)
            self._last_score = -float("inf")
            self._last_eval_details = str(exc)

    # ─── Extra tools for the researcher LLM ──────────────────────────────────

    def _make_tools(self) -> list[BaseTool]:
        # All closures capture self — they can write to _last_score etc.

        @tool
        def evaluate_hypothesis() -> str:
            """Merge specialist work into the main workspace and run the evaluation suite.

            Call this after ALL specialists have finished implementing the hypothesis.
            Returns a numeric score (higher = better) and optional details.
            The score is used to decide whether to keep or revert this iteration's work.
            """
            # Merge all specialist worktrees into main
            merge_status = self._worktree_mgr.merge_all()
            logger.debug("merge_all: %s", merge_status)

            # Run the user's evaluation function
            try:
                raw = self._eval_fn(str(self._workspace))
                if isinstance(raw, dict):
                    score = float(raw.get("score", -float("inf")))
                    details = raw.get("details", "")
                else:
                    score = float(raw)
                    details = ""
            except Exception as exc:
                score = -float("inf")
                details = f"Evaluation error: {exc}"

            self._last_score = score
            self._last_eval_details = details
            return f"Score: {score:.4f}\nDetails: {details}"

        @tool
        def save_hypothesis(hypothesis_description: str, achieved_score: float) -> str:
            """Save the current implementation as the new best result.

            Call this ONLY when evaluate_hypothesis returned a score that is better
            than the current best score shown in the iteration prompt.

            Args:
                hypothesis_description: What was changed and why it works.
                achieved_score: The exact score returned by evaluate_hypothesis.
            """
            try:
                self._save_fn(
                    str(self._workspace),
                    hypothesis_description,
                    achieved_score,
                    self._current_iteration,
                )
                self._save_called = True
                return f"Saved (iteration {self._current_iteration}, score {achieved_score:.4f})."
            except Exception as exc:
                return f"Save failed: {exc}"

        @tool
        def list_past_hypotheses(limit: int = 40) -> str:
            """Retrieve past hypothesis outcomes from memory.

            Call this before formulating a new hypothesis to avoid repeating
            approaches that have already been tried (and may have failed).

            Args:
                limit: Maximum number of entries to return (default 40).
            """
            try:
                mems = self._store.list_memories(agent_name="researcher", limit=limit)
                if mems.empty:
                    return "No past hypotheses recorded."
                rows = mems[["content", "created_at"]].to_dict("records")
                return "\n---\n".join(
                    f"{r['created_at']}: {r['content']}" for r in rows
                )
            except Exception as exc:
                return f"Could not retrieve memory: {exc}"

        return [evaluate_hypothesis, save_hypothesis, list_past_hypotheses]
