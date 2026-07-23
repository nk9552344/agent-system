"""Chunk-safe file writing tools.

Ollama truncates any tool-call JSON argument that exceeds ~3000 chars (~50 lines).
These tools let the agent write large files in small chunks (15-20 lines each)
so no single tool-call argument ever overflows.

Workflow for a NEW file (>20 lines):
  write_file(file_path="/abs/path/file", content="...lines 1-15...")
  append_to_file(file_path="/abs/path/file", content="...lines 16-30...")
  append_to_file(file_path="/abs/path/file", content="...lines 31-45...")
  ...repeat until done, then verify with read_file...

Workflow for FULL REWRITE of an existing file:
  overwrite_file_start(file_path="/abs/path/file", content="...lines 1-15...")
  append_to_file(file_path="/abs/path/file", content="...lines 16-30...")
  ...repeat until done...

Workflow for a LARGE SECTION REPLACEMENT inside a file:
  1. read_file to find start/end line numbers
  2. Build replacement content in a temp file (same chunk pattern above)
  3. splice_file(file_path=..., start_line=N, end_line=M, content_file="/tmp/new_section")
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, tool


def make_file_tools() -> list[BaseTool]:
    """Return chunk-safe file writing tools."""

    @tool
    def append_to_file(file_path: str, content: str) -> str:
        """Append a chunk of content (15-20 lines max) to an existing file.

        Use this to write large files by breaking content into small chunks —
        Ollama truncates tool-call JSON above ~3000 chars (~50 lines), so never
        pass more than 15-20 lines per call.

        First-call pattern:
          - NEW file:      write_file(...)  then  append_to_file(...)  × N
          - EXISTING file: overwrite_file_start(...)  then  append_to_file(...)  × N
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return (
                    f"Error: '{file_path}' does not exist. "
                    "Create it first with write_file (new) or overwrite_file_start (existing)."
                )
            if content and not content.endswith("\n"):
                content += "\n"
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
            lines = content.count("\n")
            total = len(path.read_text(encoding="utf-8").splitlines())
            return f"Appended {lines} lines to '{file_path}' (file now has {total} lines total)."
        except Exception as exc:
            return f"Error: {exc}"

    @tool
    def overwrite_file_start(file_path: str, content: str) -> str:
        """Truncate an existing file and write the first chunk (15-20 lines max).

        Use this to start a FULL REWRITE of an existing file. The file is cleared
        and the first chunk is written. Then use append_to_file for the remaining chunks.

        For NEW files, use write_file instead (it creates the file).
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return (
                    f"Error: '{file_path}' does not exist. "
                    "For new files, use write_file instead."
                )
            if content and not content.endswith("\n"):
                content += "\n"
            path.write_text(content, encoding="utf-8")
            lines = content.count("\n")
            return (
                f"Truncated and wrote first {lines} lines to '{file_path}'. "
                "Use append_to_file for subsequent chunks."
            )
        except Exception as exc:
            return f"Error: {exc}"

    @tool
    def splice_file(
        file_path: str,
        start_line: int,
        end_line: int,
        content_file: str,
    ) -> str:
        """Replace lines start_line–end_line (1-indexed inclusive) with content from a temp file.

        Use this to replace a large section INSIDE a file without touching the rest.

        Full workflow:
          1. read_file to find the line numbers of the section to replace
          2. Build replacement content in a temp file using the chunk pattern:
               write_file(file_path="/tmp/new_section", content="...chunk 1...")
               append_to_file(file_path="/tmp/new_section", content="...chunk 2...")
          3. splice_file(file_path="/abs/path/target", start_line=N, end_line=M,
                         content_file="/tmp/new_section")
        """
        try:
            target = Path(file_path)
            temp   = Path(content_file)

            if not target.exists():
                return f"Error: '{file_path}' does not exist."
            if not temp.exists():
                return (
                    f"Error: content_file '{content_file}' does not exist. "
                    "Build it first with write_file + append_to_file chunks."
                )

            original_lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
            new_content    = temp.read_text(encoding="utf-8")

            if new_content and not new_content.endswith("\n"):
                new_content += "\n"

            total = len(original_lines)
            s = max(1, start_line)
            e = min(total, end_line)
            if s > total:
                return f"Error: start_line {start_line} exceeds file length ({total} lines)."

            result = original_lines[: s - 1] + [new_content] + original_lines[e:]
            target.write_text("".join(result), encoding="utf-8")

            replaced  = e - s + 1
            new_lines = new_content.count("\n")
            return (
                f"Replaced lines {s}–{e} ({replaced} old lines) with "
                f"{new_lines} new lines in '{file_path}'."
            )
        except Exception as exc:
            return f"Error: {exc}"

    return [append_to_file, overwrite_file_start, splice_file]
