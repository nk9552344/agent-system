"""Shell/command tools with explicit timeout and safe output capture."""

from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import BaseTool, tool


def make_shell_tools(workspace_dir: Path, default_timeout: int = 30) -> list[BaseTool]:
    """Return shell tools scoped to workspace_dir."""

    @tool
    def check_command_exists(command: str) -> str:
        """Check whether a CLI command is available on PATH.

        Args:
            command: Command name to check (e.g. 'git', 'python', 'node').
        """
        result = subprocess.run(
            ["which", command],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return f"'{command}' found at: {result.stdout.strip()}"
        return f"'{command}' is NOT available on PATH."

    @tool
    def run_python_snippet(code: str, timeout: int = default_timeout) -> str:
        """Run a short Python snippet and return stdout + stderr.

        Useful for quick calculations, data transformations, or verifying logic
        without touching any files.

        Args:
            code: Python source code to execute.
            timeout: Max seconds to wait (default: 30).
        """
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(workspace_dir),
            )
            output_parts = []
            if result.stdout:
                output_parts.append(f"stdout:\n{result.stdout.rstrip()}")
            if result.stderr:
                output_parts.append(f"stderr:\n{result.stderr.rstrip()}")
            if not output_parts:
                output_parts.append("(no output)")
            if result.returncode != 0:
                output_parts.append(f"exit code: {result.returncode}")
            return "\n".join(output_parts)
        except subprocess.TimeoutExpired:
            return f"Timed out after {timeout}s."

    @tool
    def get_process_info(process_name: str) -> str:
        """Check if a process is running by name. Useful for checking servers/services.

        Args:
            process_name: Process name to search for (e.g. 'ollama', 'uvicorn').
        """
        result = subprocess.run(
            ["pgrep", "-la", process_name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"Running processes matching '{process_name}':\n{result.stdout.strip()}"
        return f"No running process found matching '{process_name}'."

    return [check_command_exists, run_python_snippet, get_process_info]
