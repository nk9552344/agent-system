#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  install.sh — One-line installer for agentx
#
#  curl -fsSL https://raw.githubusercontent.com/YOUR_USER/agentx/main/install.sh | bash
#
#  What this does:
#    1. Tries to install a pre-built binary from GitHub Releases (fastest)
#    2. Falls back to  uv tool install  (needs Python 3.13 + uv)
#    3. Falls back to  pip install      (needs Python 3.13 + pip)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

PURPLE='\033[0;35m'; ORANGE='\033[0;33m'
GREEN='\033[0;32m';  RED='\033[0;31m';  DIM='\033[2m'; NC='\033[0m'

REPO="YOUR_GITHUB_USER/agentx"   # ← replace with your GitHub user/repo
VERSION="latest"

log()  { echo -e "${PURPLE}◆${NC}  $*"; }
ok()   { echo -e "  ${GREEN}✓${NC}  $*"; }
warn() { echo -e "  ${ORANGE}⚠${NC}  $*" >&2; }
die()  { echo -e "  ${RED}✗${NC}  $*" >&2; exit 1; }

echo
echo -e "${PURPLE}◆ AGENTX — Installer${NC}"
echo

# ── Detect platform ───────────────────────────────────────────────────────────
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
  linux)  OS_LABEL="linux"  ;;
  darwin) OS_LABEL="macos"  ;;
  *) OS_LABEL="" ;;
esac

case "$ARCH" in
  x86_64|amd64) ARCH_LABEL="x86_64" ;;
  aarch64|arm64) ARCH_LABEL="arm64" ;;
  *) ARCH_LABEL="" ;;
esac

INSTALL_DIR="${HOME}/.local/bin"
mkdir -p "$INSTALL_DIR"

# ── Try GitHub Releases binary (if platform supported) ────────────────────────
install_from_release() {
  [[ -z "$OS_LABEL" || -z "$ARCH_LABEL" ]] && return 1

  local asset="agentx-${OS_LABEL}-${ARCH_LABEL}"
  local api_url="https://api.github.com/repos/${REPO}/releases/latest"
  local dl_url

  log "Checking GitHub Releases for a pre-built binary…"

  # Resolve "latest" to an actual download URL
  if command -v curl &>/dev/null; then
    dl_url=$(curl -fsSL "$api_url" 2>/dev/null \
      | grep "browser_download_url" \
      | grep "${asset}" \
      | head -1 \
      | sed 's/.*"browser_download_url": "\(.*\)".*/\1/')
  elif command -v wget &>/dev/null; then
    dl_url=$(wget -qO- "$api_url" 2>/dev/null \
      | grep "browser_download_url" \
      | grep "${asset}" \
      | head -1 \
      | sed 's/.*"browser_download_url": "\(.*\)".*/\1/')
  fi

  [[ -z "$dl_url" ]] && return 1

  log "Downloading ${asset}…"
  local tmp
  tmp=$(mktemp)

  if command -v curl &>/dev/null; then
    curl -fsSL -o "$tmp" "$dl_url"
  else
    wget -qO "$tmp" "$dl_url"
  fi

  chmod +x "$tmp"
  mv "$tmp" "${INSTALL_DIR}/agentx"
  ok "Installed binary → ${INSTALL_DIR}/agentx"
  return 0
}

# ── Fall back: uv tool install ────────────────────────────────────────────────
install_via_uv() {
  command -v uv &>/dev/null || return 1

  # Check Python 3.13
  if ! python3 -c "import sys; assert sys.version_info >= (3,13)" 2>/dev/null; then
    return 1
  fi

  log "Installing via uv tool install…"
  uv tool install "git+https://github.com/${REPO}.git"
  ok "Installed via uv — run: agentx --version"
  return 0
}

# ── Fall back: pip install ────────────────────────────────────────────────────
install_via_pip() {
  local pip_cmd
  for cmd in pip3.13 pip3 pip; do
    command -v "$cmd" &>/dev/null && pip_cmd="$cmd" && break
  done
  [[ -z "${pip_cmd:-}" ]] && return 1

  # Check Python 3.13
  if ! python3 -c "import sys; assert sys.version_info >= (3,13)" 2>/dev/null; then
    return 1
  fi

  log "Installing via pip…"
  "$pip_cmd" install --user "git+https://github.com/${REPO}.git"
  ok "Installed via pip — run: agentx --version"
  return 0
}

# ── Run install chain ─────────────────────────────────────────────────────────
if install_from_release; then
  :
elif install_via_uv; then
  :
elif install_via_pip; then
  :
else
  die "Could not install agentx automatically.

  Options:
    1. Install uv + Python 3.13, then run this script again.
    2. Download a binary from https://github.com/${REPO}/releases
    3. Clone the repo and run:  pip install .
"
fi

# ── PATH reminder ─────────────────────────────────────────────────────────────
if [[ ":$PATH:" != *":${INSTALL_DIR}:"* ]]; then
  echo
  warn "${INSTALL_DIR} is not in your PATH."
  warn "Add this to ~/.bashrc or ~/.zshrc:"
  echo
  echo -e "    ${ORANGE}export PATH=\"${INSTALL_DIR}:\$PATH\"${NC}"
  echo
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo
log "Quick start:"
echo -e "    ${ORANGE}agentx init${NC}          # set up config in any project directory"
echo -e "    ${ORANGE}agentx run agent${NC}     # start single-agent mode"
echo -e "    ${ORANGE}agentx run researcher${NC} # start multi-agent coordinator"
echo
