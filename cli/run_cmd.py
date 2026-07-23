"""Build the agent/coordinator and launch the TUI."""
from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from cli.config import CliConfig, ConfigError
from cli.theme import ORANGE, PURPLE, GREEN, RED, DIM, LOGO

console = Console()


# System prompt injected into the single agent.
# Tells it where the workspace is so it reads local files before answering.
_AGENT_SYSTEM_PROMPT = """\
You are a capable AI agent with full access to a local software project.

## Your workspace
{workspace}

## Workflow

For any question about the project:
1. Call `get_directory_tree(".")` or `list_directory(".")` to orient yourself
2. Read relevant files with `read_file` (README, pyproject.toml, source files)
3. Answer based on what you actually read

For tasks that require multiple agents or parallel work, you may use the `task`
tool to spawn sub-agents. For simple file reads and writes, use the filesystem
tools directly: `read_file`, `write_file`, `shell`, `grep`.

Never guess or assume file contents — always read them first.
"""

# Injected at the top of the coordinator's system prompt.
_COORDINATOR_WORKSPACE_PREFIX = """\
## Your project workspace
{workspace}

Always explore the project using `get_directory_tree(".")` before planning.
Call `get_directory_tree` with NO arguments or path="." to see the project layout.
Never look outside this directory for project files.

"""


def run_agent_mode(cfg: CliConfig, initial_prompt: str | None) -> None:
    """Start OllamaDeepAgent in the full-screen TUI."""
    _ensure_storage(cfg)

    from agent import OllamaDeepAgent
    from tools.web_tools import WebConfig

    store = _open_store(cfg)
    agent_cfg = cfg.agent
    workspace = Path(agent_cfg.workspace)   # already absolute from config.py

    ws_prompt = _AGENT_SYSTEM_PROMPT.format(workspace=workspace)
    system_prompt = (
        f"{agent_cfg.system_prompt}\n\n{ws_prompt}"
        if agent_cfg.system_prompt
        else ws_prompt
    )

    agent = OllamaDeepAgent(
        model_name=cfg.model.name,
        base_url=cfg.model.base_url,
        temperature=cfg.model.temperature,
        num_ctx=cfg.model.context_window,
        workspace_dir=workspace,
        memory_store=store,
        name=agent_cfg.name,
        system_prompt=system_prompt,
        require_permission=False,
        persistent_memory=True,
        web_config=WebConfig.from_dict(cfg.web),
        debug=cfg.debug,
    )

    _launch_tui(
        agent,
        mode="agent",
        model=cfg.model.name,
        workspace=workspace,
        initial_prompt=initial_prompt,
    )


def run_researcher_mode(cfg: CliConfig, initial_prompt: str | None) -> None:
    """Start the Coordinator in the full-screen TUI (researcher / multi-agent mode)."""
    _ensure_storage(cfg)

    from coordinator import Coordinator
    from tools.web_tools import WebConfig

    try:
        coord_cfg = cfg.coordinator_config
    except (ValueError, KeyError) as exc:
        _die(
            f"researcher section of agent_config.yml is incomplete:\n{exc}\n\n"
            f"Run  [bold {ORANGE}]agentx init[/bold {ORANGE}]  to create a valid template."
        )
        return

    store = _open_store(cfg)
    workspace = Path(cfg.researcher.workspace)   # already absolute from config.py

    coordinator = Coordinator(
        coordinator_config=coord_cfg,
        workspace_dir=workspace,
        workspace_prompt_prefix=_COORDINATOR_WORKSPACE_PREFIX.format(workspace=workspace),
        storage_path=cfg.storage.path,
        memory_store=store,
        web_config=WebConfig.from_dict(cfg.web),
        debug=cfg.debug,
    )

    _launch_tui(
        coordinator,
        mode="researcher",
        model=cfg.model.name,
        workspace=workspace,
        initial_prompt=initial_prompt,
        specialist_names=[spec.name for spec in coord_cfg.agents],
    )


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _ensure_storage(cfg: CliConfig) -> None:
    try:
        from storage.schema import bootstrap_tables
        bootstrap_tables(cfg.storage.path)
    except Exception as exc:
        console.print(
            f"  [{RED}]⚠[/{RED}]  Could not initialise storage at "
            f"'{cfg.storage.path}': {exc}\n  Agents will run without persistent memory."
        )


def _open_store(cfg: CliConfig):
    try:
        from storage import SharedMemoryStore
        return SharedMemoryStore(
            data_path=cfg.storage.path,
            embedding_model=cfg.storage.embedding_model,
            ollama_base_url=cfg.model.base_url,
        )
    except Exception as exc:
        console.print(f"  [{RED}]⚠[/{RED}]  Memory store unavailable: {exc}")
        return None


def _launch_tui(
    runner,
    *,
    mode: str,
    model: str,
    workspace: Path,
    initial_prompt: str | None,
    specialist_names: list[str] | None = None,
) -> None:
    from cli.tui import AgentTUI
    app = AgentTUI(
        runner=runner,
        mode=mode,
        model=model,
        workspace=workspace,
        initial_prompt=initial_prompt,
        specialist_names=specialist_names or [],
    )
    app.run()


def _die(message: str) -> None:
    console.print(Panel(message, border_style=RED, title="Error"))
    sys.exit(1)
