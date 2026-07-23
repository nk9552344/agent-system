"""AgentX CLI — main entry point.

Commands:
    agentx init                   Bootstrap workspace config and storage
    agentx run agent              Start single-agent TUI
    agentx run researcher         Start multi-agent research TUI
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from cli.theme import ORANGE, PURPLE, VIOLET, DIM, LOGO, RED

console = Console()

_VERSION = "0.1.0"

_BANNER = (
    f"\n  {LOGO}  [bold]AI agent system for your terminal[/bold]\n"
    f"  [{DIM}]v{_VERSION}[/{DIM}]\n"
)


# ── Root group ────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(_VERSION, "-V", "--version", prog_name="agentx")
@click.pass_context
def main(ctx: click.Context) -> None:
    """◆ AGENTX — AI agent system for your terminal.

    \b
    Quick start:
      agentx init                  # set up config in current directory
      agentx run agent             # start single-agent mode
      agentx run researcher        # start multi-agent researcher mode
    """
    if ctx.invoked_subcommand is None:
        console.print(_BANNER)
        console.print(ctx.get_help())


# ── init ──────────────────────────────────────────────────────────────────────

@main.command()
@click.option(
    "--force", "-f",
    is_flag=True,
    default=False,
    help="Overwrite existing config files.",
)
def init(force: bool) -> None:
    """Initialise agent_config.yml and agent_storage/ in the current directory.

    Safe to re-run — existing files are not overwritten unless --force is given.
    """
    from cli.init_cmd import run_init
    run_init(force=force)


# ── run ───────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("mode", type=click.Choice(["agent", "researcher"], case_sensitive=False))
@click.option(
    "--prompt", "-p",
    default=None,
    metavar="TEXT",
    help="Send this prompt immediately when the session starts.",
)
@click.option(
    "--prompt-file", "-f",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    metavar="FILE",
    help="Read the initial prompt from a file (overrides --prompt).",
)
@click.option(
    "--config", "-c",
    default=".agentx/agent_config.yml",
    show_default=True,
    type=click.Path(dir_okay=False),
    metavar="FILE",
    help="Path to agent_config.yml (default: .agentx/agent_config.yml).",
)
def run(mode: str, prompt: str | None, prompt_file: str | None, config: str) -> None:
    """Start the agent in the given MODE and open the interactive terminal UI.

    \b
    Modes:
      agent       Single AI agent — coding, file editing, shell commands
      researcher  Multi-agent coordinator — decomposes tasks across specialists

    \b
    Examples:
      agentx run agent
      agentx run agent -p "Refactor main.py to use dataclasses"
      agentx run agent -f task.txt
      agentx run researcher
      agentx run researcher -p "Improve test coverage to 80%%" -c my_config.yml
    """
    from cli.config import CliConfig, ConfigError

    # Resolve initial prompt
    initial_prompt: str | None = None
    if prompt_file:
        lines = [
            ln for ln in Path(prompt_file).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        initial_prompt = "\n".join(lines).strip() or None
    elif prompt:
        initial_prompt = prompt.strip() or None

    # Load config
    try:
        cfg = CliConfig.load(config)
    except ConfigError as exc:
        console.print(Panel(str(exc), border_style=RED, title=f"[{RED}]Config error[/{RED}]"))
        sys.exit(1)

    from cli.run_cmd import run_agent_mode, run_researcher_mode

    mode = mode.lower()
    if mode == "agent":
        run_agent_mode(cfg, initial_prompt)
    else:
        run_researcher_mode(cfg, initial_prompt)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
