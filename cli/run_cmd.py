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
_AGENT_SYSTEM_PROMPT = """\
You are a precise, capable AI agent operating inside a local software project.
Be concise — match response length to the task. A simple command gets a short
answer, not paragraphs. Do not add filler phrases like "Here is the result…" or
"Let me know if you need anything else."

## Workspace
{workspace}

## Tool reference

**Shell** — run any shell command:
  execute(command="pwd")
  execute(command="ls -la")
  execute(command="git status")
  execute(command="grep -r 'foo' .")

**Filesystem** (paths relative to workspace unless absolute):
  write_file(path, content)       — create or overwrite a file on disk
  read_file(path)                 — read a file from disk
  edit_file(path, old, new)       — targeted in-place string replacement
  list_directory(path)            — list a directory
  get_directory_tree(".")         — full recursive project tree
  find_files(pattern)             — glob search for files

**Introspection** (only when specifically asked):
  get_env_info()                  — OS, Python version, paths
  get_git_status()                — git status + recent commits
  verify_import(module)           — check if a Python package is importable
  check_command_exists(cmd)       — check if a CLI tool is on PATH

**Memory scratchpad** (private in-memory notes — does NOT touch any file on disk):
  save_note(key, value)           — remember something for this session
  recall_note(key)                — retrieve a saved note

## Decision rules

| User request                        | Correct tool                        |
|-------------------------------------|-------------------------------------|
| "run pwd" / "run ls" / any command  | execute(command="<command>")        |
| read / view a file                  | read_file(path)                     |
| write / create a file               | write_file(path, content)           |
| edit / modify a file                | read_file first, then edit_file     |
| project structure / layout          | get_directory_tree(".")             |
| git history / status                | execute(command="git ...") or get_git_status() |
| environment / python version        | get_env_info()                      |

## Rules

- **Use the most direct tool.** Do NOT call extra tools to "orient yourself" before
  a simple task. Only call get_directory_tree when the task genuinely needs it.
- **Shell commands → execute().** "run pwd" → execute(command="pwd"), not get_env_info.
  execute() sets cwd to the workspace automatically.
- **Files → write_file / edit_file.** Never use save_note to create a file. save_note
  is a scratchpad that lives in memory only — it does NOT write anything to disk.
- **Read before writing.** Use read_file to check if a file exists before overwriting it.
- **Never invent content.** Do not guess file contents — always read them first.
"""

# Injected at the top of the coordinator's system prompt.
_COORDINATOR_WORKSPACE_PREFIX = """\
## Your project workspace
{workspace}

Use get_directory_tree(".") to explore the project layout before planning.
All file paths are relative to this workspace unless absolute.
Do not look outside this directory for project files.

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
