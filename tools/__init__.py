"""Custom tools for the OllamaDeepAgent.

All tools are created via factory functions so they can be parameterized
(workspace directory, shared store, etc.) per agent instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from tools.context_tools import make_context_tools
from tools.file_tools import make_file_tools
from tools.memory_tools import make_memory_tools
from tools.permission_tools import make_permission_tools
from tools.shell_tools import make_shell_tools
from tools.web_tools import WebConfig, make_web_tools

if TYPE_CHECKING:
    from storage.memory_store import SharedMemoryStore


def make_all_tools(
    workspace_dir: Path,
    memory_store: SharedMemoryStore,
    agent_name: str = "",
    session_id: str = "",
    shell_timeout: int = 30,
    web_config: WebConfig | None = None,
) -> list[BaseTool]:
    """Build the full tool suite for an agent instance.

    Args:
        workspace_dir: Root directory the agent operates in.
        memory_store:  Shared LanceDB store (all agents share one instance).
        agent_name:    Agent identity stamped on memory writes.
        session_id:    Current session ID stamped on memory writes.
        shell_timeout: Default timeout (seconds) for run_python_snippet.
        web_config:    Web tool settings (GitHub token, timeouts, chunk sizes).
                       Read from the ``web:`` section of config.yml via main.py.
                       Uses safe defaults when None.
    """
    return [
        *make_file_tools(workspace_dir),
        *make_shell_tools(workspace_dir, default_timeout=shell_timeout),
        *make_memory_tools(memory_store, agent_name=agent_name, session_id=session_id),
        *make_permission_tools(),
        *make_context_tools(workspace_dir),
        *make_web_tools(memory_store, agent_name=agent_name, config=web_config),
    ]


__all__ = [
    "make_all_tools",
    "make_context_tools",
    "make_file_tools",
    "make_memory_tools",
    "make_permission_tools",
    "make_shell_tools",
    "make_web_tools",
    "WebConfig",
]
