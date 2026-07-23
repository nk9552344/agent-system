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

- Python 3.13+
- [Ollama](https://ollama.ai) running locally

Pull the default models:
```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text     # for vector embeddings
```

---

## Installation

### From source (development)

```bash
git clone <repo>
cd agent-system
pip install -e .
```

### Binary (single file, no Python needed)

```bash
pip install pyinstaller
pyinstaller \
  --onefile \
  --name agentx \
  --add-data "cli/templates:cli/templates" \
  cli/app.py
# Binary: dist/agentx
```

After building, copy `dist/agentx` anywhere on your `PATH`.

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

After `agentx init`, edit `agent_config.yml`:

```yaml
# Ollama model — used by both modes
model:
  name: qwen2.5-coder:7b
  base_url: http://localhost:11434
  temperature: 0.0
  context_window: 8192

# Agent mode settings
agent:
  name: agent
  workspace: .              # directory the agent reads/writes
  require_permission: true  # ask before writes/shell commands (non-TUI only)

# Researcher mode settings
researcher:
  workspace: .
  specialists_config: agent_storage/specialists.yml
  # Optional: path to an eval script for autonomous hypothesis testing
  eval_script: null

# Storage
storage:
  path: agent_storage/lancedb
  embedding_model: nomic-embed-text

# Web tools (available to all agents)
web:
  github_token: ""          # STRONGLY recommended — 5000 req/hr vs 60
  timeout: 20
  max_pdf_pages: 60
  chunk_size: 2400
  chunk_overlap: 150
  max_chunks_per_resource: 50

debug: false
```

### Researcher mode — specialist roster (`agent_storage/specialists.yml`)

Defines which models act as specialists under the coordinator:

```yaml
coordinator:
  model: "qwen2.5-coder:7b"
  num_ctx: 16384

agents:
  - name: "coder"
    model: "qwen2.5-coder:7b"
    expertise: [python, typescript, debugging, testing]
    description: "Expert developer. Delegate implementation work here."

  - name: "reviewer"
    model: "qwen2.5-coder:7b"
    expertise: [code review, security, documentation]
    description: "Code reviewer. Delegate quality and docs work here."
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
├── agent_config.yml          # ← edit this
├── agent_storage/
│   ├── specialists.yml       # ← specialist roster (researcher mode)
│   ├── lancedb/              # vector DB (created at runtime)
│   └── .gitignore
└── eval.py                   # optional — for autonomous research loop
```

---

## Advanced

### Using a different model for each specialist

Edit `agent_storage/specialists.yml` and set `model:` per agent:

```yaml
agents:
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
