"""Color constants — warm dark / orange theme (Claude Code inspired)."""

# ── Core palette ──────────────────────────────────────────────────────────────
ORANGE    = "#F97316"   # orange-500  — primary accent, commands
ORANGE_BR = "#FB923C"   # orange-400  — bright orange, highlights
AMBER     = "#FBBF24"   # amber-400   — warnings, tool calls
GREEN     = "#86EFAC"   # green-300   — success
RED       = "#FCA5A5"   # red-300     — errors

# ── Text ─────────────────────────────────────────────────────────────────────
TEXT   = "#F5F5F4"   # stone-100  — primary text
DIM    = "#78716C"   # stone-500  — muted / secondary
MUTED  = "#44403C"   # stone-700  — very muted, borders

# ── Backgrounds ───────────────────────────────────────────────────────────────
BG_DEEP   = "#1C1917"   # stone-900  — main terminal background
BG_HEADER = "#292524"   # stone-800  — header / footer bar

# ── Rich markup helpers ───────────────────────────────────────────────────────
def orange(s: str)  -> str: return f"[{ORANGE}]{s}[/{ORANGE}]"
def amber(s: str)   -> str: return f"[{AMBER}]{s}[/{AMBER}]"
def green(s: str)   -> str: return f"[{GREEN}]{s}[/{GREEN}]"
def red(s: str)     -> str: return f"[{RED}]{s}[/{RED}]"
def dim(s: str)     -> str: return f"[{DIM}]{s}[/{DIM}]"
def bold(s: str)    -> str: return f"[bold]{s}[/bold]"

# Compat aliases used by init_cmd / run_cmd
def purple(s: str)  -> str: return orange(s)
def violet(s: str)  -> str: return orange(s)
def cyan(s: str)    -> str: return orange(s)
def blue(s: str)    -> str: return amber(s)

CYAN   = ORANGE   # alias — used by init_cmd border_style
PURPLE = ORANGE   # alias
VIOLET = ORANGE_BR

LOGO = f"[bold {ORANGE}]◆ AGENTX[/bold {ORANGE}]"
SEP  = f"[{DIM}]─[/{DIM}]"
