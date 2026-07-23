"""Color constants for Rich terminal output (non-TUI commands)."""

# Core palette — Midnight Slate theme
CYAN    = "#06B6D4"   # primary accent
BLUE    = "#38BDF8"   # secondary accent / highlights
ORANGE  = "#F97316"   # commands, important callouts
AMBER   = "#FBBF24"   # warnings, tool calls
GREEN   = "#22C55E"   # success
RED     = "#F87171"   # errors

# Backward-compat aliases (primary / secondary accent)
PURPLE  = CYAN
VIOLET  = BLUE

# Text
TEXT    = "#F1F5F9"   # primary text (slate-100)
DIM     = "#64748B"   # muted text (slate-500)
MUTED   = "#1E293B"   # very muted / backgrounds

# Background tones
BG_DEEP   = "#0F172A"   # slate-900 main background
BG_HEADER = "#1E293B"   # slate-800 header / footer
BG_FOOTER = "#151F2E"   # slightly deeper footer

# Rich markup helpers (use inside [bracket] syntax)
def cyan(s: str)   -> str:  return f"[{CYAN}]{s}[/{CYAN}]"
def blue(s: str)   -> str:  return f"[{BLUE}]{s}[/{BLUE}]"
def orange(s: str) -> str:  return f"[{ORANGE}]{s}[/{ORANGE}]"
def purple(s: str) -> str:  return cyan(s)   # compat — maps to primary accent
def violet(s: str) -> str:  return blue(s)   # compat — maps to secondary accent
def amber(s: str)  -> str:  return f"[{AMBER}]{s}[/{AMBER}]"
def green(s: str)  -> str:  return f"[{GREEN}]{s}[/{GREEN}]"
def red(s: str)    -> str:  return f"[{RED}]{s}[/{RED}]"
def dim(s: str)    -> str:  return f"[{DIM}]{s}[/{DIM}]"
def bold(s: str)   -> str:  return f"[bold]{s}[/bold]"

LOGO = f"[bold {CYAN}]◆ AGENTX[/bold {CYAN}]"
SEP  = f"[{DIM}]─[/{DIM}]"
