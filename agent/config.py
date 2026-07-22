"""Configuration dataclass for OllamaDeepAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class AgentConfig:
    """All parameters for constructing an OllamaDeepAgent.

    Pass this to OllamaDeepAgent(**config.__dict__) or unpack it directly.
    Useful when you want to define agent configs in one place and instantiate
    many agents from them.

    Example::

        cfg = AgentConfig(model_name="qwen2.5-coder:7b", workspace_dir="/tmp/proj")
        agent = OllamaDeepAgent(**cfg.__dict__)
    """

    # --- Required ---
    model_name: str
    """Ollama model name, e.g. 'llama3.2', 'qwen2.5-coder:7b', 'mistral'."""

    # --- Ollama connection ---
    base_url: str = "http://localhost:11434"
    """Ollama server URL."""

    temperature: float = 0.0
    """Model temperature. 0 = deterministic, 1 = creative."""

    num_ctx: int = 8192
    """Context window size in tokens sent to Ollama."""

    # --- Identity ---
    name: str = "ollama-agent"
    """Display name for this agent instance (used in LangSmith traces)."""

    # --- Workspace ---
    workspace_dir: str | Path = "."
    """Root directory the agent reads/writes files in."""

    notes_file: str | Path | None = None
    """Path to the JSON file for persistent notes.
    Defaults to '<workspace_dir>/.agent_notes.json'."""

    # --- Prompting ---
    system_prompt: str | None = None
    """Custom instructions prepended to the base agent prompt."""

    # --- Permissions & safety ---
    require_permission: bool = True
    """If True, the agent must confirm before writing files or running shell commands."""

    allowed_operations: list[Literal["read", "write", "delete"]] = field(
        default_factory=lambda: ["read", "write", "delete"]
    )
    """File operations the agent is allowed to perform.
    Operations in this list but NOT in 'interrupt_operations' are silently allowed.
    Operations in 'interrupt_operations' require user approval via HITL."""

    interrupt_operations: list[Literal["read", "write", "delete"]] = field(
        default_factory=lambda: ["write", "delete"]
    )
    """File operations that trigger a human-in-the-loop interrupt for approval.
    Only meaningful when require_permission=True."""

    interrupt_on_execute: bool = True
    """If True and require_permission=True, the agent pauses before every shell command."""

    # --- Memory & context ---
    memory_files: list[str] | None = None
    """Paths to AGENTS.md files loaded as persistent context into the system prompt.
    E.g. ['~/.deepagents/AGENTS.md', './.deepagents/AGENTS.md']."""

    persistent_memory: bool = True
    """If True, uses MemorySaver + InMemoryStore so conversations persist within
    the same process. Pass your own checkpointer/store for cross-session persistence."""

    # --- Quality ---
    rubric: str | None = None
    """Optional quality rubric for self-evaluation.
    When set, a grader sub-agent checks the output after each completion and
    re-prompts the agent if the rubric isn't satisfied.
    Example: 'The response must include a concrete code example and no hallucinated APIs.'
    """

    # --- Shell ---
    execute_timeout: int = 120
    """Max seconds for shell command execution via the built-in 'execute' tool."""

    shell_snippet_timeout: int = 30
    """Max seconds for the run_python_snippet tool."""

    # --- Storage ---
    storage_path: str = "data/lancedb"
    """Path to the LanceDB directory used for shared memory.
    All agents pointing at the same path share notes and memories."""

    embedding_model: str | None = "nomic-embed-text"
    """Ollama model used for memory embeddings.
    Set to None to disable vector search (falls back to full-text / recency scan)."""

    # --- Extra ---
    debug: bool = False
    """If True, enables LangGraph debug logging."""

    extra_tools: list = field(default_factory=list)
    """Additional BaseTool instances to include alongside the built-in tool suite."""
