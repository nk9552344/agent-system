"""Git worktree management for per-agent isolated workspaces.

Each specialist agent gets its own git branch + worktree so changes are
isolated until the coordinator explicitly merges them. The coordinator
calls merge_all() after all agents finish.

Worktrees are created under ``<repo>/.agent-worktrees/<agent-name>/``.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class WorktreeManager:
    """Create, track, and merge per-agent git worktrees.

    Parameters
    ----------
    repo_dir:
        Root directory of the git repository. Initialized automatically
        if not already a git repo.
    base_branch:
        The main branch that agent branches diverge from and merge back into.
    """

    def __init__(self, repo_dir: str | Path, base_branch: str = "main") -> None:
        self._repo = Path(repo_dir).resolve()
        self._base_branch = base_branch
        # agent_name -> (branch_name, worktree_path)
        self._worktrees: dict[str, tuple[str, Path]] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def ensure_repo(self) -> None:
        """Initialize a git repo if the workspace isn't one already.

        Creates an initial commit so worktree branches have a parent.
        """
        self._repo.mkdir(parents=True, exist_ok=True)

        if (self._repo / ".git").exists():
            logger.debug("Git repo already exists at %s", self._repo)
            self._ensure_branch()
            return

        logger.info("Initializing git repo at %s", self._repo)
        self._run("init", "-b", self._base_branch)
        self._run("add", "-A")
        # Empty tree commit if nothing staged
        result = self._run_raw("diff", "--cached", "--quiet")
        if result.returncode != 0:
            self._run("commit", "-m", "chore: initial workspace commit")
        else:
            self._run("commit", "--allow-empty", "-m", "chore: initial workspace commit")

    def _ensure_branch(self) -> None:
        """Make sure we're on base_branch (not detached HEAD)."""
        current = self._run("branch", "--show-current").strip()
        if not current:
            # Detached HEAD — create/switch to base branch
            self._run("checkout", "-B", self._base_branch)

    # ------------------------------------------------------------------
    # Worktree lifecycle
    # ------------------------------------------------------------------

    def create_worktree(self, agent_name: str) -> Path:
        """Create a git branch and worktree for a specialist agent.

        The worktree is placed at ``<repo>/.agent-worktrees/<agent_name>/``.
        If the worktree already exists (from a previous run), it is removed
        and recreated cleanly.

        Args:
            agent_name: Identifier matching the agent's name in config.yml.

        Returns:
            Absolute path to the worktree directory.
        """
        branch = f"agent/{agent_name}"
        worktree_path = self._repo / ".agent-worktrees" / agent_name
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale worktree/branch from prior runs
        self._cleanup_one(agent_name, silent=True)

        # Create fresh branch from current HEAD of base branch
        self._run("branch", branch)
        self._run("worktree", "add", str(worktree_path), branch)

        self._worktrees[agent_name] = (branch, worktree_path)
        logger.info("Worktree created: %s → %s", branch, worktree_path)
        return worktree_path

    def worktree_path(self, agent_name: str) -> Path | None:
        """Return the worktree path for an agent, or None if not created."""
        entry = self._worktrees.get(agent_name)
        return entry[1] if entry else None

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_all(self) -> str:
        """Merge all agent branches into the base branch sequentially.

        Commits any staged changes in each worktree before merging, so
        agents don't need to commit manually.

        Returns:
            Human-readable summary of merge results.
        """
        if not self._worktrees:
            return "No agent worktrees to merge."

        # First, auto-commit any uncommitted changes in each worktree
        for agent_name, (branch, wt_path) in self._worktrees.items():
            self._auto_commit_worktree(agent_name, wt_path, branch)

        # Switch to base branch and merge each agent branch
        lines: list[str] = []
        for agent_name, (branch, _) in self._worktrees.items():
            try:
                out = self._run(
                    "merge", "--no-ff", branch,
                    "-m", f"Merge specialist/{agent_name} results",
                )
                lines.append(f"✓ {agent_name}: merged successfully.")
                logger.info("Merged branch %s", branch)
            except subprocess.CalledProcessError as exc:
                err = exc.stderr or exc.stdout or str(exc)
                lines.append(f"✗ {agent_name}: merge conflict — {err.strip()[:200]}")
                # Abort the conflicted merge so subsequent merges can proceed
                self._run_raw("merge", "--abort")
                logger.warning("Merge conflict for %s: %s", branch, err.strip())

        return "\n".join(lines)

    def _auto_commit_worktree(self, agent_name: str, wt_path: Path, branch: str) -> None:
        """Stage and commit all changes in a worktree if anything is modified."""
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=wt_path,
        )
        if not status.stdout.strip():
            return  # nothing to commit
        subprocess.run(["git", "add", "-A"], cwd=wt_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"feat({agent_name}): agent work output"],
            cwd=wt_path, check=True, capture_output=True,
        )
        logger.debug("Auto-committed changes in worktree for %s", agent_name)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove all agent worktrees and their branches."""
        for agent_name in list(self._worktrees):
            self._cleanup_one(agent_name)
        self._worktrees.clear()
        logger.info("All agent worktrees removed.")

    def _cleanup_one(self, agent_name: str, *, silent: bool = False) -> None:
        branch = f"agent/{agent_name}"
        wt_path = self._repo / ".agent-worktrees" / agent_name
        # Remove worktree
        self._run_raw("worktree", "remove", str(wt_path), "--force")
        # Delete branch
        self._run_raw("branch", "-D", branch)
        self._worktrees.pop(agent_name, None)
        if not silent:
            logger.info("Removed worktree and branch for %s", agent_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, cwd=self._repo, check=True,
        )
        return result.stdout

    def _run_raw(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            capture_output=True, text=True, cwd=self._repo,
        )
