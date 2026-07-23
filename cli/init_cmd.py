"""agentx init — bootstrap .agentx/ in the current working directory."""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cli.theme import ORANGE, PURPLE, CYAN, GREEN, DIM, LOGO

console = Console()

_TEMPLATES  = Path(__file__).parent / "templates"
AGENTX_DIR  = ".agentx"          # hidden folder, consistent with .git / .github


def run_init(force: bool = False) -> None:
    """Create .agentx/agent_config.yml and .agentx/agent_storage/ in CWD."""
    cwd        = Path.cwd()
    agentx_dir = cwd / AGENTX_DIR

    console.print()
    console.print(Panel(
        f"{LOGO}  [bold]Initializing workspace[/bold]\n"
        f"[{DIM}]{cwd}[/{DIM}]",
        border_style=CYAN,
        padding=(0, 2),
    ))
    console.print()

    # ── .agentx/ root ────────────────────────────────────────────────────────
    agentx_dir.mkdir(exist_ok=True)
    _ok(f"{AGENTX_DIR}/")

    # ── agent_config.yml ─────────────────────────────────────────────────────
    config_dst = agentx_dir / "agent_config.yml"
    if config_dst.exists() and not force:
        console.print(
            f"  [{DIM}]~[/{DIM}]  {AGENTX_DIR}/agent_config.yml already exists — skipping  "
            f"[{DIM}](--force to overwrite)[/{DIM}]"
        )
    else:
        src = _TEMPLATES / "agent_config.yml"
        config_dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        _ok(f"{AGENTX_DIR}/agent_config.yml")

    # ── agent_storage/ (lancedb created at runtime) ──────────────────────────
    storage_dir = agentx_dir / "agent_storage"
    storage_dir.mkdir(exist_ok=True)
    _ok(f"{AGENTX_DIR}/agent_storage/  (lancedb DB created on first run)")

    # ── .agentx/.gitignore — keeps runtime data out of the repo ─────────────
    _write_agentx_gitignore(agentx_dir)

    # ── Next steps ────────────────────────────────────────────────────────────
    console.print()
    steps = Text()
    steps.append("  Next steps\n\n", style=f"bold {CYAN}")

    steps.append("  1. ", style=f"{DIM}")
    steps.append("Edit ", style="white")
    steps.append(f"{AGENTX_DIR}/agent_config.yml", style=f"bold {ORANGE}")
    steps.append("\n", style="white")
    steps.append(
        f"     [{DIM}]Set your Ollama model, workspace, specialists, and GitHub token[/{DIM}]\n\n",
        style="white",
    )

    steps.append("  2. ", style=f"{DIM}")
    steps.append("Pull required Ollama models  ", style="white")
    steps.append("ollama pull nomic-embed-text", style=f"bold {ORANGE}")
    steps.append("\n\n", style="white")

    steps.append("  Run modes\n\n", style=f"bold {CYAN}")
    steps.append("    agentx run agent       ", style=f"bold {ORANGE}")
    steps.append("  Single AI agent (chat / coding)\n", style="white")
    steps.append("    agentx run researcher  ", style=f"bold {ORANGE}")
    steps.append("  Multi-agent research coordinator\n", style="white")

    console.print(Panel(steps, border_style=CYAN, padding=(0, 1)))
    console.print()


def _write_agentx_gitignore(agentx_dir: Path) -> None:
    """Write .agentx/.gitignore keeping worktrees and storage out of git."""
    from coordinator.git_worktree import _AGENTX_GITIGNORE_CONTENT
    gi = agentx_dir / ".gitignore"
    gi.write_text(_AGENTX_GITIGNORE_CONTENT, encoding="utf-8")
    _ok(f"{AGENTX_DIR}/.gitignore")


def _ok(label: str) -> None:
    console.print(f"  [{GREEN}]✓[/{GREEN}]  {label}")
