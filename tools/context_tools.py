"""Context tools — git status only. Use execute() for env/package queries."""

from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import BaseTool, tool


def make_context_tools(workspace_dir: Path) -> list[BaseTool]:
    """Return context tools scoped to workspace_dir."""

    @tool
    def get_git_status() -> str:
        """Return git branch, staged/unstaged changes, and recent commits."""
        def run_git(*args: str) -> str:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=str(workspace_dir),
            )
            return result.stdout.strip() or result.stderr.strip()

        is_git = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=str(workspace_dir),
        )
        if is_git.returncode != 0:
            return "Not a git repository."

        parts = [
            f"Branch: {run_git('branch', '--show-current')}",
            f"\nStatus:\n{run_git('status', '--short')}",
            f"\nRecent commits:\n{run_git('log', '--oneline', '-5')}",
        ]
        return "\n".join(parts)

    return [get_git_status]
