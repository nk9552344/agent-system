"""Color constants for Rich terminal output (non-TUI commands)."""

# Core palette
ORANGE  = "#F97316"
PURPLE  = "#7C3AED"
VIOLET  = "#A78BFA"
AMBER   = "#F59E0B"
GREEN   = "#10B981"
RED     = "#EF4444"

# Text
TEXT    = "#E2E8F0"
DIM     = "#6B7280"
MUTED   = "#374151"

# Background tones
BG_DEEP   = "#0F0A1E"
BG_HEADER = "#4C1D95"
BG_FOOTER = "#1E1B4B"

# Rich markup helpers (use inside [bracket] syntax)
def orange(s: str) -> str:  return f"[{ORANGE}]{s}[/{ORANGE}]"
def purple(s: str) -> str:  return f"[{PURPLE}]{s}[/{PURPLE}]"
def violet(s: str) -> str:  return f"[{VIOLET}]{s}[/{VIOLET}]"
def amber(s: str)  -> str:  return f"[{AMBER}]{s}[/{AMBER}]"
def green(s: str)  -> str:  return f"[{GREEN}]{s}[/{GREEN}]"
def red(s: str)    -> str:  return f"[{RED}]{s}[/{RED}]"
def dim(s: str)    -> str:  return f"[{DIM}]{s}[/{DIM}]"
def bold(s: str)   -> str:  return f"[bold]{s}[/bold]"

LOGO = f"[bold {PURPLE}]◆ AGENTX[/bold {PURPLE}]"
SEP  = f"[{DIM}]─[/{DIM}]"
