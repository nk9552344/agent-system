"""Custom tools for the OllamaDeepAgent.

Only tools that deepagents does NOT provide natively:
- splice_file: large-section edits via temp file (avoids JSON truncation)
- memory tools: save_note / recall_note
- permission tools: request_permission
- context tools: get_git_status
- web tools: web_search / fetch_and_store_url

deepagents provides: ls, read_file, write_file, edit_file, glob, grep, execute, write_todos
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from tools.context_tools import make_context_tools
from tools.file_tools import make_file_tools
from tools.memory_tools import make_memory_tools
from tools.permission_tools import make_permission_tools
from tools.web_tools import WebConfig, make_web_tools

if TYPE_CHECKING:
    from storage.memory_store import SharedMemoryStore


def make_all_tools(
    workspace_dir: Path,
    memory_store: SharedMemoryStore,
    agent_name: str = "",
    session_id: str = "",
    web_config: WebConfig | None = None,
) -> list[BaseTool]:
    """Build the custom tool suite for an agent instance."""
    return [
        *make_file_tools(),
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
    "make_web_tools",
    "WebConfig",
]
