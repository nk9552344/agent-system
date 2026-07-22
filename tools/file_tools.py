"""Extra filesystem tools beyond deepagents' built-in read/write/edit/grep/glob."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from langchain_core.tools import BaseTool, tool


def make_file_tools(workspace_dir: Path) -> list[BaseTool]:
    """Return file tools scoped to workspace_dir."""

    @tool
    def get_directory_tree(
        path: str = ".",
        max_depth: int = 3,
        show_hidden: bool = False,
    ) -> str:
        """Return a tree view of a directory. Useful for understanding project structure.

        Args:
            path: Directory to show (relative to workspace, or absolute).
            max_depth: Maximum depth to recurse (default 3).
            show_hidden: Include hidden files/dirs starting with dot (default False).
        """
        target = Path(path) if Path(path).is_absolute() else workspace_dir / path
        if not target.exists():
            return f"Path does not exist: {path}"
        if not target.is_dir():
            return f"Not a directory: {path}"

        lines: list[str] = [str(target)]

        def _walk(directory: Path, prefix: str, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                entries = sorted(directory.iterdir(), key=lambda e: (e.is_file(), e.name))
            except PermissionError:
                lines.append(f"{prefix}[permission denied]")
                return

            visible = [e for e in entries if show_hidden or not e.name.startswith(".")]
            for i, entry in enumerate(visible):
                connector = "└── " if i == len(visible) - 1 else "├── "
                lines.append(f"{prefix}{connector}{entry.name}")
                if entry.is_dir():
                    extension = "    " if i == len(visible) - 1 else "│   "
                    _walk(entry, prefix + extension, depth + 1)

        _walk(target, "", 1)
        return "\n".join(lines)

    @tool
    def diff_files(file_a: str, file_b: str) -> str:
        """Show unified diff between two text files.

        Args:
            file_a: First file path (relative to workspace or absolute).
            file_b: Second file path (relative to workspace or absolute).
        """
        path_a = Path(file_a) if Path(file_a).is_absolute() else workspace_dir / file_a
        path_b = Path(file_b) if Path(file_b).is_absolute() else workspace_dir / file_b

        for p, name in [(path_a, file_a), (path_b, file_b)]:
            if not p.exists():
                return f"File not found: {name}"
            if not p.is_file():
                return f"Not a file: {name}"

        result = subprocess.run(
            ["diff", "-u", str(path_a), str(path_b)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return "Files are identical."
        return result.stdout or result.stderr

    @tool
    def find_files(
        pattern: str,
        search_dir: str = ".",
        max_results: int = 50,
    ) -> str:
        """Find files matching a name pattern using glob, returning relative paths.

        Args:
            pattern: Glob pattern like '*.py', '**/*.json', 'src/*.ts'.
            search_dir: Directory to search in (relative or absolute).
            max_results: Maximum number of results to return.
        """
        target = Path(search_dir) if Path(search_dir).is_absolute() else workspace_dir / search_dir
        if not target.exists():
            return f"Directory not found: {search_dir}"

        matches = list(target.glob(pattern))[:max_results]
        if not matches:
            return f"No files matching '{pattern}' in '{search_dir}'."

        lines = [str(p.relative_to(workspace_dir)) if p.is_relative_to(workspace_dir) else str(p) for p in sorted(matches)]
        suffix = f"\n... (showing {max_results} of more)" if len(matches) == max_results else ""
        return "\n".join(lines) + suffix

    return [get_directory_tree, diff_files, find_files]
