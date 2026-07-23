"""agentx init — bootstrap agent_config.yml and agent_storage/ in CWD."""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cli.theme import ORANGE, PURPLE, VIOLET, GREEN, DIM, LOGO

console = Console()

# Template paths (relative to this file)
_TEMPLATES = Path(__file__).parent / "templates"


def run_init(force: bool = False) -> None:
    """Create agent_config.yml and agent_storage/ in the current directory."""
    cwd = Path.cwd()

    console.print()
    console.print(Panel(
        f"{LOGO}  [bold]Initializing workspace[/bold]\n"
        f"[{DIM}]{cwd}[/{DIM}]",
        border_style=PURPLE,
        padding=(0, 2),
    ))
    console.print()

    # ── agent_config.yml ────────────────────────────────────────────────────
    config_dst = cwd / "agent_config.yml"
    if config_dst.exists() and not force:
        console.print(f"  [{DIM}]~[/{DIM}]  agent_config.yml already exists — skipping  "
                      f"[{DIM}](use --force to overwrite)[/{DIM}]")
    else:
        config_src = _TEMPLATES / "agent_config.yml"
        config_dst.write_text(config_src.read_text(encoding="utf-8"), encoding="utf-8")
        _ok("agent_config.yml")

    # ── agent_storage/ ──────────────────────────────────────────────────────
    storage_dir = cwd / "agent_storage"
    storage_dir.mkdir(exist_ok=True)

    # .gitignore inside storage
    gi_dst = storage_dir / ".gitignore"
    if not gi_dst.exists():
        gi_src = _TEMPLATES / "storage_gitignore"
        gi_dst.write_text(gi_src.read_text(encoding="utf-8"), encoding="utf-8")

    # specialists.yml
    specialists_dst = storage_dir / "specialists.yml"
    if specialists_dst.exists() and not force:
        console.print(f"  [{DIM}]~[/{DIM}]  agent_storage/specialists.yml already exists — skipping")
    else:
        spec_src = _TEMPLATES / "specialists.yml"
        specialists_dst.write_text(spec_src.read_text(encoding="utf-8"), encoding="utf-8")
        _ok("agent_storage/specialists.yml")

    _ok("agent_storage/  (directory ready)")

    # ── Next steps ──────────────────────────────────────────────────────────
    console.print()
    steps = Text()
    steps.append("  Next steps\n\n", style=f"bold {VIOLET}")
    steps.append("  1. ", style=f"{DIM}")
    steps.append("Edit ", style="white")
    steps.append("agent_config.yml", style=f"bold {ORANGE}")
    steps.append(" — set your Ollama model and workspace path\n", style="white")

    steps.append("  2. ", style=f"{DIM}")
    steps.append("For researcher mode, also edit ", style="white")
    steps.append("agent_storage/specialists.yml", style=f"bold {ORANGE}")
    steps.append("\n\n", style="white")

    steps.append("  Run modes\n\n", style=f"bold {VIOLET}")
    steps.append("    agentx run agent       ", style=f"bold {ORANGE}")
    steps.append("  Single AI agent (chat / coding)\n", style="white")
    steps.append("    agentx run researcher  ", style=f"bold {ORANGE}")
    steps.append("  Multi-agent research coordinator\n", style="white")

    console.print(Panel(steps, border_style=PURPLE, padding=(0, 1)))
    console.print()


def _ok(label: str) -> None:
    console.print(f"  [{GREEN}]✓[/{GREEN}]  {label}")
