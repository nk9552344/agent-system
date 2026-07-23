"""Coordinator — manager, router, and merger for a fleet of specialist agents.

The coordinator is itself an OllamaDeepAgent with:
- deepagents' built-in SubAgent middleware so its LLM can call specialists
  via the ``task`` tool
- Three extra tools: ``judge_result``, ``store_agent_result``, ``merge_all``
- The shared LanceDB memory (same store as all specialists)
- Its own git-merge capability via WorktreeManager

Flow when ``coordinator.run(task)`` is called:
1. Coordinator LLM reads config-injected roster and decomposes the task.
2. It calls ``task(agent_name, instruction)`` for each specialist in parallel.
3. Specialists run autonomously in their own git worktrees with full tool access.
4. Coordinator calls ``judge_result`` on each result; retries up to N times.
5. On pass, calls ``store_agent_result`` to persist to LanceDB.
6. After all specialists finish, calls ``merge_all`` to git-merge their branches.

Example::

    from coordinator import Coordinator

    coord = Coordinator(workspace_dir="/tmp/myproject")
    result = coord.run("Build a FastAPI CRUD app with SQLite and full test coverage.")
    print(result)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from agent.core import OllamaDeepAgent
from coordinator.config import AgentSpec, CoordinatorConfig, load_config
from coordinator.git_worktree import WorktreeManager
from coordinator.judge import ResultJudge
from deepagents import CompiledSubAgent
from storage.memory_store import SharedMemoryStore
from tools.web_tools import WebConfig

logger = logging.getLogger(__name__)

# Maximum times the coordinator retries a specialist task after a FAIL verdict.
MAX_RETRIES = 2

# -------------------------------------------------------------------------
# Coordinator system prompt template — injected with the model roster
# -------------------------------------------------------------------------

_COORDINATOR_PROMPT = """You are a coordinator agent. Your role is to manage a team of \
specialist AI agents, gather external knowledge when useful, and deliver high-quality \
results by delegating, evaluating, and integrating their work.

## Your team

{agent_roster}

## Workflow — follow this exactly

1. **Decompose** the user's task into non-overlapping subtasks, one per specialist.
   Assign each subtask to the specialist whose expertise best matches.
   Never do implementation work yourself — always delegate via the ``task`` tool.

2. **Delegate** each subtask using the ``task`` tool with a clear, self-contained
   instruction. Be specific: tell the specialist what to build, what files to touch,
   and what the expected output looks like.

3. **Evaluate** each result with ``judge_result``. If the verdict is FAIL, rewrite
   the instruction incorporating the retry hint and call ``task`` again (max {max_retries} retries).

4. **Store** each accepted result with ``store_agent_result`` before moving on.

5. **[OPTIONAL] Web Research** — After all specialist results are in, decide whether
   ingesting external knowledge (GitHub repos, docs, research papers, articles) would
   meaningfully improve the project. Skip this step if the task is self-contained.

   If external knowledge would help:
   a. Call ``spawn_web_researchers`` with a list of specific search topics — one agent
      is spawned per topic, all running in parallel. Each agent searches, fetches, and
      stores relevant pages into the shared memory RAG automatically.
   b. Call ``search_web_resources`` to review what was ingested.
   c. Use insights from ingested content to enrich the next step.

   You can also call ``web_search`` and ``fetch_and_store_url`` directly at any point
   if you need a specific resource right now.

6. **Hypothesis** — Based on all specialist results and any ingested web resources,
   form a concrete improvement hypothesis for the user's workspace project:
   - What specific changes or additions would make this project better?
   - Which patterns, libraries, or approaches from the web research are relevant?
   - Rank improvements by expected impact. Name exact files, functions, or techniques.
   State this hypothesis clearly before the merge step so it is recorded.

7. **Merge** all specialist branches into the main branch with ``merge_all`` once
   every specialist has finished.

8. **Summarise** what was built, which files changed, the improvement hypothesis,
   and any caveats.

## Rules

- Never do a specialist's job yourself. Delegate everything.
- Each subtask must be independent — no specialist should depend on another's
  in-progress output (dependencies → sequence; independent work → parallel delegation).
- Write precise, unambiguous instructions to specialists. Vague prompts cause FAIL verdicts.
- If a task keeps failing after {max_retries} retries, report the blocker and stop.
- Web research is optional — only invoke it when external knowledge adds real value.
"""

_AGENT_ROSTER_ENTRY = "- **{name}** ({model}) — expertise: {expertise}\n  {description}"


def _build_roster(agents: list[AgentSpec]) -> str:
    return "\n".join(
        _AGENT_ROSTER_ENTRY.format(
            name=a.name,
            model=a.model,
            expertise=a.expertise_line(),
            description=a.description.strip(),
        )
        for a in agents
    )


# -------------------------------------------------------------------------
# Coordinator
# -------------------------------------------------------------------------

class Coordinator:
    """Manager-coordinator-router for multi-agent task execution.

    Parameters
    ----------
    config_path:
        Path to ``coordinator/config.yml``.
    workspace_dir:
        Root directory where specialist agents read and write files.
        A git repo is initialized here if one doesn't already exist.
    storage_path:
        Path to the shared LanceDB directory (``data/lancedb`` by default).
    memory_store:
        Pre-built SharedMemoryStore to share with all agents.
        If omitted, one is created pointing at ``storage_path``.
    debug:
        Enable LangGraph debug logging on all agents.
    """

    def __init__(
        self,
        config_path: str | Path = "coordinator/config.yml",
        workspace_dir: str | Path = ".",
        storage_path: str | Path = "data/lancedb",
        memory_store: SharedMemoryStore | None = None,
        web_config: WebConfig | None = None,
        debug: bool = False,
    ) -> None:
        cfg = load_config(config_path)
        workspace = Path(workspace_dir).resolve()
        self._workspace = workspace
        self._web_config = web_config or WebConfig()

        # --- Shared memory (all agents, including coordinator, share this) --------
        self._store = memory_store or SharedMemoryStore(
            data_path=storage_path,
            embedding_model="nomic-embed-text",
            ollama_base_url=cfg.coordinator.base_url,
        )

        # --- Git worktrees — one per specialist ----------------------------------
        self._worktree_mgr = WorktreeManager(workspace, base_branch="main")
        self._worktree_mgr.ensure_repo()

        # --- Judge (uses coordinator's model) ------------------------------------
        self._judge = ResultJudge(
            model_name=cfg.coordinator.model,
            base_url=cfg.coordinator.base_url,
        )

        # --- Build specialist agents + CompiledSubAgent specs --------------------
        compiled_subagents: list[CompiledSubAgent] = []
        self._specialists: dict[str, OllamaDeepAgent] = {}

        for spec in cfg.agents:
            worktree_path = self._worktree_mgr.create_worktree(spec.name)
            agent = OllamaDeepAgent(
                model_name=spec.model,
                base_url=spec.base_url,
                temperature=spec.temperature,
                num_ctx=spec.num_ctx,
                workspace_dir=worktree_path,
                memory_store=self._store,
                name=spec.name,
                # Specialists run autonomously — no HITL interrupts
                require_permission=False,
                persistent_memory=True,
                system_prompt=_specialist_prompt(spec),
                web_config=self._web_config,
                debug=debug,
            )
            self._specialists[spec.name] = agent
            compiled_subagents.append(
                CompiledSubAgent(
                    name=spec.name,
                    description=spec.description.strip(),
                    runnable=agent.graph,
                )
            )

        # --- Coordinator-specific tools ------------------------------------------
        coord_tools = self._make_coordinator_tools(cfg)

        # --- Build coordinator agent (is itself an OllamaDeepAgent) ---------------
        coord_prompt = _COORDINATOR_PROMPT.format(
            agent_roster=_build_roster(cfg.agents),
            max_retries=MAX_RETRIES,
        )
        self._agent = OllamaDeepAgent(
            model_name=cfg.coordinator.model,
            base_url=cfg.coordinator.base_url,
            temperature=cfg.coordinator.temperature,
            num_ctx=cfg.coordinator.num_ctx,
            workspace_dir=workspace,
            memory_store=self._store,
            name="coordinator",
            require_permission=False,
            persistent_memory=True,
            system_prompt=coord_prompt,
            subagents=compiled_subagents,
            extra_tools=coord_tools,
            web_config=self._web_config,
            debug=debug,
        )

        self._config = cfg

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(
        self,
        task: str,
        *,
        thread_id: str | None = None,
        auto_approve: bool = True,
    ) -> str:
        """Decompose, delegate, evaluate, and merge a task.

        The coordinator's LLM drives the entire flow — it decides which
        specialists to use, evaluates their output, and merges when done.

        Args:
            task: High-level description of what to build or accomplish.
            thread_id: Reuse for multi-turn follow-up on the same project.
            auto_approve: Approve any HITL interrupts automatically (default True).
        """
        return self._agent.run(task, thread_id=thread_id, auto_approve=auto_approve)

    def stream(self, task: str, **kwargs: Any) -> Iterator[str]:
        """Stream coordinator output token by token."""
        yield from self._agent.stream(task, **kwargs)

    def cleanup(self) -> None:
        """Remove all specialist git worktrees. Call after a project is done."""
        self._worktree_mgr.cleanup()

    @property
    def memory_store(self) -> SharedMemoryStore:
        """The shared memory store — pass to other Coordinator instances to share memory."""
        return self._store

    @property
    def specialists(self) -> dict[str, OllamaDeepAgent]:
        """Direct access to specialist agents, keyed by name."""
        return self._specialists

    # -------------------------------------------------------------------------
    # Coordinator-specific tools (injected into coordinator only)
    # -------------------------------------------------------------------------

    def _make_coordinator_tools(self, cfg: CoordinatorConfig) -> list[BaseTool]:
        store      = self._store
        judge      = self._judge
        worktree_mgr = self._worktree_mgr
        workspace  = self._workspace
        web_config = self._web_config

        @tool
        def judge_result(agent_name: str, task_given: str, agent_output: str) -> str:
            """Evaluate whether a specialist's output satisfactorily completed its task.

            Call this immediately after receiving a result from any specialist.
            Returns PASS or FAIL with reasoning. If FAIL, use the retry_hint to
            craft a better instruction and call ``task`` again.

            Args:
                agent_name: The specialist's name (matches config.yml).
                task_given: The exact instruction you sent to the specialist.
                agent_output: The specialist's final response or a summary of it.
            """
            passed, reason, retry_hint = judge.evaluate(task_given, agent_output)
            verdict = "PASS" if passed else "FAIL"
            lines = [f"VERDICT: {verdict}", f"REASON: {reason}"]
            if retry_hint:
                lines.append(f"RETRY_HINT: {retry_hint}")
            return "\n".join(lines)

        @tool
        def store_agent_result(agent_name: str, task_summary: str, result_summary: str) -> str:
            """Persist an accepted specialist result to the shared memory database.

            Call this after a specialist receives a PASS verdict, before moving on.

            Args:
                agent_name: The specialist's name.
                task_summary: Brief description of the task that was completed.
                result_summary: Key deliverables or outcomes from the agent.
            """
            mem_id = store.add_memory(
                f"[{agent_name}] {task_summary}: {result_summary}",
                agent_name=agent_name,
                tags=["agent_result", agent_name],
            )
            store.save_note(
                f"result:{agent_name}:{task_summary[:40]}",
                result_summary,
                agent_name="coordinator",
            )
            return f"Stored result for '{agent_name}' (memory id: {mem_id[:8]}…)."

        @tool
        def merge_all() -> str:
            """Merge all specialist agent branches into the main branch.

            Call this once every specialist has finished and their results have
            been stored. Returns a per-agent merge status summary.
            Each agent's git branch is ``agent/<agent_name>``.
            """
            return worktree_mgr.merge_all()

        @tool
        def list_specialists() -> str:
            """List available specialists with their expertise areas.

            Call this if you are unsure which specialist to use for a subtask.
            """
            lines = []
            for spec in cfg.agents:
                lines.append(f"- {spec.name}: {spec.expertise_line()}")
            return "\n".join(lines)

        @tool
        def spawn_web_researchers(topics: list[str]) -> str:
            """Spawn parallel web-research agents to find and ingest relevant external resources.

            Each topic gets a dedicated agent that:
            1. Searches the web for relevant articles, GitHub repos, or research papers
            2. Selects the most useful results
            3. Fetches and parses each resource
            4. Stores parsed content in the shared RAG memory (visible to all agents)

            Agents run in parallel — total wall-clock time equals the slowest single topic.
            Call search_web_resources afterwards to query what was ingested.

            Use this when external knowledge (best practices, reference implementations,
            papers, documentation) would meaningfully improve the project.
            Skip it for straightforward or self-contained tasks.

            Args:
                topics: List of search queries, one per parallel agent.
                        Be specific: e.g. ['FastAPI dependency injection patterns',
                        'pytest fixtures best practices', 'SQLite WAL mode Python'].
            """
            if not topics:
                return "No topics provided — skipping web research."

            _WEB_RESEARCHER_PROMPT = (
                "You are a web research agent. Your sole job is to find, fetch, and "
                "store relevant external resources for a software project.\n\n"
                "Workflow:\n"
                "1. Call web_search with your assigned topic to get candidate URLs.\n"
                "2. Review the results and pick the 2–3 most relevant ones.\n"
                "3. Call fetch_and_store_url for each chosen URL with a clear description.\n"
                "4. Report: what you ingested and the key insight from each resource.\n\n"
                "Be selective — quality over quantity. Prefer official docs, reputable "
                "GitHub repos, and peer-reviewed papers over blog posts."
            )

            def _research_one(topic: str) -> str:
                agent = OllamaDeepAgent(
                    model_name=cfg.coordinator.model,
                    base_url=cfg.coordinator.base_url,
                    temperature=cfg.coordinator.temperature,
                    num_ctx=cfg.coordinator.num_ctx,
                    workspace_dir=workspace,
                    memory_store=store,
                    name="web-researcher",
                    require_permission=False,
                    persistent_memory=False,
                    system_prompt=_WEB_RESEARCHER_PROMPT,
                    web_config=web_config,
                )
                return agent.run(
                    f"Research topic: {topic}\n\n"
                    f"Search for relevant resources, fetch the best ones, store them.",
                    auto_approve=True,
                )

            results: list[str] = []
            max_workers = min(len(topics), 6)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_topic = {pool.submit(_research_one, t): t for t in topics}
                for future in as_completed(future_to_topic):
                    topic = future_to_topic[future]
                    try:
                        summary = future.result()
                        results.append(f"[{topic}]\n{summary[:400]}")
                    except Exception as exc:
                        logger.warning("Web researcher for %r failed: %s", topic, exc)
                        results.append(f"[{topic}]\nERROR: {exc}")

            header = f"Web research complete — {len(topics)} topic(s) processed in parallel.\n"
            return header + "\n---\n".join(results)

        return [judge_result, store_agent_result, merge_all, list_specialists, spawn_web_researchers]


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _specialist_prompt(spec: AgentSpec) -> str:
    """System prompt injected into each specialist agent."""
    return (
        f"You are {spec.name}, a specialist AI agent.\n\n"
        f"Your expertise: {spec.expertise_line()}.\n\n"
        f"## Instructions\n"
        f"- Work autonomously to completion. Do not ask clarifying questions.\n"
        f"- Use your tools: read files, write files, run shell commands, search.\n"
        f"- Verify your own work: after writing code, check it runs; after writing\n"
        f"  a doc, review it for correctness.\n"
        f"- When you are done, provide a concise summary of what you delivered.\n"
        f"- Stay within your workspace directory. Do not modify files outside it.\n"
    )
