#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  build_binary.sh — Build a standalone agentx binary for the current machine
#
#  Usage:
#    chmod +x build_binary.sh
#    ./build_binary.sh
#
#  Output:  dist/agentx-<os>-<arch>   (or .exe on Windows)
#  Copies a "agentx" symlink/copy into dist/ for convenience.
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
PURPLE='\033[0;35m'; ORANGE='\033[0;33m'
GREEN='\033[0;32m';  RED='\033[0;31m';  DIM='\033[2m'; NC='\033[0m'

log()  { echo -e "${PURPLE}◆${NC}  $*"; }
ok()   { echo -e "  ${GREEN}✓${NC}  $*"; }
warn() { echo -e "  ${ORANGE}⚠${NC}  $*"; }
die()  { echo -e "  ${RED}✗${NC}  $*" >&2; exit 1; }

echo
echo -e "${PURPLE}◆ AGENTX — Binary Build${NC}"
echo -e "${DIM}  Building a standalone executable for the current system${NC}"
echo

# ── 1. Detect OS ──────────────────────────────────────────────────────────────
RAW_OS=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$RAW_OS" in
  linux)   OS="linux"   ;;
  darwin)  OS="macos"   ;;
  msys*|cygwin*|mingw*) OS="windows" ;;
  *) die "Unsupported OS: $RAW_OS" ;;
esac

# ── 2. Detect architecture ────────────────────────────────────────────────────
RAW_ARCH=$(uname -m)
case "$RAW_ARCH" in
  x86_64|amd64)   ARCH="x86_64" ;;
  aarch64|arm64)  ARCH="arm64"  ;;
  armv7l)         ARCH="armv7"  ;;
  *) die "Unsupported architecture: $RAW_ARCH" ;;
esac

EXT=""
[[ "$OS" == "windows" ]] && EXT=".exe"

BINARY_LABEL="agentx-${OS}-${ARCH}${EXT}"
log "Target: ${ORANGE}${BINARY_LABEL}${NC}"

# ── 3. Check Python ───────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3.13 python3 python; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    MAJOR=$(echo "$VER" | cut -d. -f1)
    MINOR=$(echo "$VER" | cut -d. -f2)
    if [[ $MAJOR -ge 3 && $MINOR -ge 13 ]]; then
      PYTHON="$cmd"
      ok "Python $VER  ($cmd)"
      break
    fi
  fi
done

if [[ -z "$PYTHON" ]]; then
  die "Python 3.13+ is required. Install it from https://python.org or via your package manager."
fi

# ── 4. Install build dependencies ────────────────────────────────────────────
log "Installing build dependencies…"

# Prefer uv for speed, fall back to pip
if command -v uv &>/dev/null; then
  uv pip install --quiet pyinstaller 2>/dev/null \
    || uv pip install pyinstaller
  uv pip install --quiet -e . 2>/dev/null \
    || uv pip install -e .
elif command -v pip &>/dev/null || command -v pip3 &>/dev/null; then
  PIP=$(command -v pip3 || command -v pip)
  "$PIP" install --quiet pyinstaller
  "$PIP" install --quiet -e .
else
  die "Neither uv nor pip found. Install one of them first."
fi
ok "Dependencies installed"

# ── 5. Run PyInstaller ────────────────────────────────────────────────────────
log "Running PyInstaller…"
echo -e "  ${DIM}This may take 2–5 minutes on first run (downloads bootloader)${NC}"
echo

# Clean previous build artefacts for this target only
rm -rf "build/agentx" "dist/${BINARY_LABEL}" "dist/agentx${EXT}"

$PYTHON -m PyInstaller \
  --onefile \
  --name "${BINARY_LABEL}" \
  --collect-all deepagents \
  --collect-all langchain_core \
  --collect-all langchain_ollama \
  --collect-all langgraph \
  --collect-all lancedb \
  --collect-all textual \
  --collect-all rich \
  --collect-all click \
  --collect-data cli \
  --hidden-import yaml \
  --hidden-import pandas \
  --hidden-import pypdf \
  --hidden-import pdfminer \
  --noconfirm \
  --log-level WARN \
  cli/app.py

# ── 6. Also create a plain "agentx" copy ─────────────────────────────────────
cp "dist/${BINARY_LABEL}" "dist/agentx${EXT}"
chmod +x "dist/${BINARY_LABEL}" "dist/agentx${EXT}"

# ── 7. Print result ───────────────────────────────────────────────────────────
echo
echo -e "${PURPLE}◆ Build complete${NC}"
echo
SIZE=$(du -sh "dist/${BINARY_LABEL}" | cut -f1)
ok "${ORANGE}dist/${BINARY_LABEL}${NC}  (${SIZE})"
ok "${ORANGE}dist/agentx${EXT}${NC}  (convenience copy)"
echo
echo -e "  ${DIM}Install system-wide:${NC}"
if [[ "$OS" == "macos" || "$OS" == "linux" ]]; then
  echo -e "    ${ORANGE}sudo cp dist/agentx /usr/local/bin/agentx${NC}"
  echo -e "    ${ORANGE}agentx --version${NC}"
else
  echo -e "    Copy dist/agentx.exe to a directory on your PATH"
fi
echo
