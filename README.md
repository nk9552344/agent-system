# ◆ AGENTX

**Local AI agent system for your terminal.**  
Runs on [Ollama](https://ollama.ai) — no cloud API keys, no data leaving your machine.

---

## Features

- **Agent mode** — single AI agent with file, shell, memory, and web tools
- **Researcher mode** — team of specialist agents coordinated by an LLM
- **Web research** — search, scrape, fetch GitHub repos and PDFs, store in RAG memory
- **Shared memory** — all agents read and write a central LanceDB vector store
- **Full-screen TUI** — dynamic terminal UI, interactive REPL, streaming output

---

## Requirements

- Python 3.13+  (only needed for install methods 2 and 3)
- [Ollama](https://ollama.ai) running locally

Pull the default models:
```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text     # for vector embeddings
```

---

## Installation

### Option 1 — Download a pre-built binary (no Python needed)

Go to the [**Releases page**](https://github.com/YOUR_USER/agentx/releases) and download the binary for your platform:

| File | Platform |
|---|---|
| `agentx-linux-x86_64` | Linux (Intel / AMD) |
| `agentx-linux-arm64` | Linux (ARM, Raspberry Pi, Ampere) |
| `agentx-macos-x86_64` | macOS (Intel) |
| `agentx-macos-arm64` | macOS (Apple Silicon M1/M2/M3) |
| `agentx-windows-x86_64.exe` | Windows |

Then install it:
```bash
# Linux / macOS
chmod +x agentx-linux-x86_64
sudo mv agentx-linux-x86_64 /usr/local/bin/agentx

# Verify
agentx --version
```

### Option 2 — One-line installer (auto-detects platform)

```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_USER/agentx/main/install.sh | bash
```

Tries the pre-built binary first; falls back to `uv tool install` or `pip install` if no binary exists for your platform.

### Option 3 — Install from source (needs Python 3.13 + uv)

```bash
git clone https://github.com/YOUR_USER/agentx
cd agentx
uv tool install .

agentx --version
```

For development (changes take effect immediately):
```bash
pip install -e .
```

### Option 4 — Build your own binary

```bash
git clone https://github.com/YOUR_USER/agentx
cd agentx
./build_binary.sh
# Binary: dist/agentx-<os>-<arch>

sudo mv dist/agentx-linux-arm64 /usr/local/bin/agentx
agentx --version
```

---

## Quick start

```bash
# 1. Go to your project directory
cd ~/my-project

# 2. Initialise — creates agent_config.yml and agent_storage/
agentx init

# 3. Edit config (set your model, workspace path, GitHub token, etc.)
$EDITOR agent_config.yml

# 4. Run
agentx run agent
```

---

## Commands

### `agentx init`

Bootstraps the current directory with:

```
agent_config.yml          ← main config (edit this)
agent_storage/
  specialists.yml         ← specialist roster for researcher mode
  .gitignore
```

```bash
agentx init               # safe to re-run; skips existing files
agentx init --force       # overwrite all existing config files
```

---

### `agentx run <mode>`

Opens the full-screen interactive terminal UI.

| Mode | Description |
|---|---|
| `agent` | Single AI agent — coding, file editing, shell, web search |
| `researcher` | Multi-agent coordinator — decomposes tasks across specialists |

**Options:**

| Flag | Short | Default | Description |
|---|---|---|---|
| `--prompt TEXT` | `-p` | — | Send this prompt immediately on startup |
| `--prompt-file FILE` | `-f` | — | Read initial prompt from a file |
| `--config FILE` | `-c` | `agent_config.yml` | Path to config file |

**Examples:**

```bash
# Interactive REPL — type tasks in the terminal UI
agentx run agent
agentx run researcher

# Send a task immediately, then drop into REPL
agentx run agent -p "Refactor main.py to use dataclasses"
agentx run researcher -p "Improve test coverage to 80%"

# Read task from a file
agentx run agent -f task.txt
agentx run researcher -f goal.txt

# Use a non-default config file
agentx run agent -c /path/to/my_config.yml
```

---

## Terminal UI

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ◆ AGENTX  ·  agent  ·  qwen2.5-coder:7b  ·  ● ready  thread #1   │  ← purple header
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │  ◆ AGENTX  │  agent mode  │  model: qwen2.5-coder:7b               │
  │  Thread #1 started. Ctrl+N for a new thread, /quit to exit.         │  ← scrollable output
  │                                                                      │
  │  You › Refactor main.py to use dataclasses                          │
  │                                                                      │
  │  I'll start by reading main.py to understand the current structure…  │
  │                                                                      │
  ├─────────────────────────────────────────────────────────────────────┤
  │  Type your message… (/help for commands)                             │  ← orange input
  ├─────────────────────────────────────────────────────────────────────┤
  │  Enter: send  ·  Ctrl+N: new thread  ·  /help: commands  ·  Ctrl+C  │  ← footer
  └─────────────────────────────────────────────────────────────────────┘
```

**In-session commands** (type in the input box):

| Command | Action |
|---|---|
| `/new` | Start a fresh conversation thread (forgets previous context) |
| `/clear` | Clear the output area |
| `/help` | Show available commands |
| `/quit` | Exit the session |
| `Ctrl+N` | New thread (keyboard shortcut for `/new`) |
| `Ctrl+C` | Quit immediately |

---

## Configuration — `agent_config.yml`

After `agentx init`, edit `agent_config.yml`. **This is the only config file the system uses.**

```yaml
# Ollama model — used by both modes (specialists can override per-entry)
model:
  name: qwen2.5-coder:7b
  base_url: http://localhost:11434
  temperature: 0.0
  context_window: 8192

# Agent mode
agent:
  name: agent
  workspace: .

# Researcher mode — coordinator + specialists defined right here
researcher:
  workspace: .
  eval_script: null         # optional — path to evaluate() + save() script

  coordinator:
    model: qwen2.5-coder:7b
    context_window: 16384

  specialists:
    - name: coder
      model: qwen2.5-coder:7b
      expertise: [python, typescript, debugging, testing]
      description: "Expert developer. Writes code, fixes bugs, adds tests."

    - name: reviewer
      model: qwen2.5-coder:7b
      expertise: [code review, security, documentation]
      description: "Code reviewer. Reviews, writes tests, produces docs."

# Storage
storage:
  path: agent_storage/lancedb
  embedding_model: nomic-embed-text

# Web tools (available to all agents)
web:
  github_token: ""          # STRONGLY recommended — 5000 req/hr vs 60 without
  timeout: 20
  max_pdf_pages: 60
  chunk_size: 2400
  chunk_overlap: 150
  max_chunks_per_resource: 50

debug: false
```

### Evaluation script (`eval_script`)

For autonomous hypothesis-driven improvement, point `eval_script` at a Python file that exports:

```python
def evaluate(workspace_path: str) -> float:
    """Return a score — higher is better."""
    ...

def save(workspace_path: str, hypothesis: str, score: float, iteration: int) -> None:
    """Persist the best result."""
    ...
```

See [`workspace_tools_example.py`](workspace_tools_example.py) for a full example.

---

## Web tools

All agents can use these tools at any time:

| Tool | Description |
|---|---|
| `web_search` | DuckDuckGo search, returns top results |
| `github_search` | Search GitHub repos and code |
| `fetch_and_store_url` | Download, parse, and store any URL in RAG memory |
| `http_request` | Raw HTTP/curl-style requests |
| `search_web_resources` | Semantic search over previously fetched content |
| `list_web_resources` | List all stored web resources |

The coordinator can also call `spawn_web_researchers` to run parallel web-research agents (one per topic).

Set your GitHub token in `agent_config.yml` to avoid rate limiting:
```yaml
web:
  github_token: "ghp_..."
```

---

## Project layout (after `agentx init`)

```
your-project/
├── agent_config.yml   # ← the only config file — edit this
├── agent_storage/
│   ├── lancedb/       # vector DB (created automatically on first run)
│   └── .gitignore
└── eval.py            # optional — autonomous research loop (evaluate + save)
```

---

## Advanced

### Using a different model for each specialist

Edit the `researcher.specialists` section of `agent_config.yml`:

```yaml
researcher:
  specialists:
    - name: coder
      model: qwen2.5-coder:14b   # bigger model for coding
    - name: reviewer
      model: llama3.2:latest      # different model for review
```

### Running in debug mode

```yaml
# agent_config.yml
debug: true
```

Or set the env var: `LANGCHAIN_VERBOSE=true`

### Binary distribution

```bash
# Build a single-file binary (includes Python interpreter)
pip install pyinstaller
pyinstaller --onefile --name agentx --add-data "cli/templates:cli/templates" cli/app.py

# Distribute: copy dist/agentx to any machine (no Python installation needed)
```
