"""Context and environment awareness tools.

These help the agent understand the runtime environment, reducing
hallucinated assumptions about Python version, OS, installed packages, etc.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import BaseTool, tool


def make_context_tools(workspace_dir: Path) -> list[BaseTool]:
    """Return context/environment tools scoped to workspace_dir."""

    @tool
    def get_env_info() -> str:
        """Return a summary of the current runtime environment.

        Includes OS, Python version, CPU/memory, current directory, and key
        environment variables. Call this at the start of any task that depends
        on the environment (running code, installing packages, etc.) to avoid
        making wrong assumptions.
        """
        info: dict[str, str] = {
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "workspace_dir": str(workspace_dir.resolve()),
            "cwd": str(Path.cwd()),
            "home": str(Path.home()),
            "path_entries": str(len(os.environ.get("PATH", "").split(":"))),
        }

        # Check for virtual env / uv
        venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_DEFAULT_ENV")
        if venv:
            info["virtual_env"] = venv

        # Check package manager availability
        for mgr in ("uv", "pip", "pip3", "conda"):
            result = subprocess.run(["which", mgr], capture_output=True, text=True)
            if result.returncode == 0:
                info[f"{mgr}_available"] = result.stdout.strip()

        lines = [f"  {k}: {v}" for k, v in info.items()]
        return "Environment:\n" + "\n".join(lines)

    @tool
    def get_installed_packages(filter_prefix: str = "") -> str:
        """List installed Python packages, optionally filtered by name prefix.

        Args:
            filter_prefix: Only show packages starting with this string (e.g. 'lang').
                           Leave empty to list all packages.
        """
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=columns"],
            capture_output=True,
            text=True,
            cwd=str(workspace_dir),
        )
        if result.returncode != 0:
            # Try uv pip list as fallback
            result = subprocess.run(
                ["uv", "pip", "list"],
                capture_output=True,
                text=True,
                cwd=str(workspace_dir),
            )

        output = result.stdout or result.stderr
        if filter_prefix:
            lines = [ln for ln in output.splitlines() if ln.lower().startswith(filter_prefix.lower()) or "Package" in ln or "---" in ln]
            return "\n".join(lines) if lines else f"No packages found with prefix '{filter_prefix}'."
        return output.strip() or "Could not list packages."

    @tool
    def verify_import(module_name: str) -> str:
        """Verify that a Python module can be imported without running it.

        Use before assuming a library is available to avoid hallucinating that
        a package is installed when it isn't.

        Args:
            module_name: Module to import (e.g. 'langchain', 'numpy', 'fastapi').
        """
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}; print('ok')"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return f"'{module_name}' can be imported successfully."
        error = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        return f"'{module_name}' CANNOT be imported: {error}"

    @tool
    def get_git_status() -> str:
        """Return the current git status of the workspace.

        Shows branch, staged/unstaged changes, and recent commits.
        """
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

    return [get_env_info, get_installed_packages, verify_import, get_git_status]
