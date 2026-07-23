"""Workspace-scoped file tools — replace deepagents' absolute-path-only built-ins.

These tools accept relative paths (resolved against workspace_dir) and use
parameter names that models naturally produce ('path', not 'file_path').
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import BaseTool, tool


def make_file_tools(workspace_dir: Path) -> list[BaseTool]:
    """Return file tools scoped to workspace_dir."""

    def _resolve(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else workspace_dir / p

    # ── Core file operations ────────────────────────────────────────────────────

    @tool
    def write_file(path: str, content: str = "") -> str:
        """Create or overwrite a file on disk (path relative to workspace)."""
        target = _resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"✓ Wrote {len(content)} chars to {path}"

    @tool
    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a file and return its text (optionally a line range)."""
        target = _resolve(path)
        if not target.exists():
            return f"File not found: {path}"
        if not target.is_file():
            return f"Not a file: {path}"
        text = target.read_text(encoding="utf-8", errors="replace")
        if start_line == 1 and end_line is None:
            return text or "(empty file)"
        lines = text.splitlines()
        start = max(0, start_line - 1)
        end   = end_line if end_line else len(lines)
        return "\n".join(lines[start:end]) or "(empty)"

    @tool
    def edit_file(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Replace old_string with new_string in a file (old_string must exist exactly)."""
        target = _resolve(path)
        if not target.exists():
            return f"File not found: {path}"
        content = target.read_text(encoding="utf-8")
        if old_string not in content:
            return f"String not found in {path}: {old_string!r}"
        count = content.count(old_string)
        if not replace_all and count > 1:
            return (
                f"String appears {count} times in {path}. "
                "Use replace_all=True or provide more surrounding context to make it unique."
            )
        updated = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        target.write_text(updated, encoding="utf-8")
        n = count if replace_all else 1
        return f"✓ Replaced {n} occurrence(s) in {path}"

    @tool
    def list_directory(path: str = ".") -> str:
        """List files and subdirectories at a path (default: workspace root)."""
        target = _resolve(path)
        if not target.exists():
            return f"Path does not exist: {path}"
        if not target.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(target.iterdir(), key=lambda e: (e.is_file(), e.name))
        if not entries:
            return f"(empty directory: {path})"
        lines = []
        for e in entries:
            prefix = "📄 " if e.is_file() else "📁 "
            lines.append(f"{prefix}{e.name}")
        return "\n".join(lines)

    # ── Directory tree / search tools ───────────────────────────────────────────

    @tool
    def get_directory_tree(
        path: str = ".",
        max_depth: int = 3,
        show_hidden: bool = False,
    ) -> str:
        """Show a recursive tree of the directory structure."""
        target = _resolve(path)
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
                    ext = "    " if i == len(visible) - 1 else "│   "
                    _walk(entry, prefix + ext, depth + 1)

        _walk(target, "", 1)
        return "\n".join(lines)

    @tool
    def find_files(
        pattern: str,
        search_dir: str = ".",
        max_results: int = 50,
    ) -> str:
        """Find files by glob pattern (e.g. '**/*.py', 'src/*.ts')."""
        target = _resolve(search_dir)
        if not target.exists():
            return f"Directory not found: {search_dir}"
        matches = list(target.glob(pattern))[:max_results]
        if not matches:
            return f"No files matching '{pattern}' in '{search_dir}'."
        lines = [
            str(p.relative_to(workspace_dir)) if p.is_relative_to(workspace_dir) else str(p)
            for p in sorted(matches)
        ]
        suffix = f"\n…showing first {max_results}" if len(matches) == max_results else ""
        return "\n".join(lines) + suffix

    @tool
    def diff_files(file_a: str, file_b: str) -> str:
        """Show a unified diff between two files."""
        path_a, path_b = _resolve(file_a), _resolve(file_b)
        for p, name in [(path_a, file_a), (path_b, file_b)]:
            if not p.exists():
                return f"File not found: {name}"
            if not p.is_file():
                return f"Not a file: {name}"
        result = subprocess.run(
            ["diff", "-u", str(path_a), str(path_b)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return "Files are identical."
        return result.stdout or result.stderr

    return [
        write_file,
        read_file,
        edit_file,
        list_directory,
        get_directory_tree,
        find_files,
        diff_files,
    ]
