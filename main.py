#!/usr/bin/env python3
"""
Multi-Agent System entry point.

Configuration is read from config.yml (or --config <path>).
The task / research goal is read from user_prompt.txt (or --prompt <path>).
If the prompt file is empty the system falls back to interactive input.

Usage:
    python main.py
    python main.py --config my_config.yml --prompt my_prompt.txt

Modes (set via  mode:  in config.yml):
    agent        — single OllamaDeepAgent, interactive or single-shot
    coordinator  — multi-specialist Coordinator, interactive or single-shot
    researcher   — infinite AutoResearcher loop (Ctrl+C to stop)
"""

import argparse
import logging
import sys
import uuid
from pathlib import Path

import yaml

# ─── ANSI colour helpers (no extra dep) ──────────────────────────────────────

_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_RESET  = "\033[0m"

def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}" if sys.stdout.isatty() else text

def _header(mode: str, detail: str) -> None:
    line = "─" * 60
    print(_c(_CYAN,  line))
    print(_c(_BOLD,  f"  Multi-Agent System  │  mode: {mode}"))
    print(_c(_DIM,   f"  {detail}"))
    print(_c(_CYAN,  line))

def _divider() -> None:
    print(_c(_DIM, "─" * 60))

def _info(msg: str)  -> None: print(_c(_GREEN,  f"  ✓  {msg}"))
def _warn(msg: str)  -> None: print(_c(_YELLOW, f"  ⚠  {msg}"), file=sys.stderr)
def _error(msg: str) -> None: print(_c(_RED,    f"  ✗  {msg}"), file=sys.stderr)

def _print_response(text: str, label: str = "Agent") -> None:
    _divider()
    print(_c(_BOLD, f"{label}:"))
    print(text.strip())
    _divider()

# ─── Config / prompt loading ──────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        _error(f"Config file not found: {path}")
        sys.exit(1)
    with p.open() as f:
        return yaml.safe_load(f) or {}

def _load_prompt(path: str) -> str | None:
    """Return the task text from the prompt file, or None if blank/comment-only."""
    p = Path(path)
    if not p.exists():
        _warn(f"Prompt file not found: {path}")
        return None
    lines = [
        ln for ln in p.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return "\n".join(lines).strip() or None

# ─── LanceDB bootstrap ────────────────────────────────────────────────────────

def _ensure_db(storage_path: str) -> bool:
    try:
        from storage.schema import bootstrap_tables
        bootstrap_tables(storage_path)
        return True
    except Exception as exc:
        _warn(f"Could not initialise LanceDB at '{storage_path}': {exc}")
        _warn("Agent will run without persistent memory.")
        return False

# ─── Interactive REPL (agent + coordinator modes) ─────────────────────────────

_REPL_HELP = "  Type a task and press Enter.  Commands: /new  /help  /quit"

def _repl(run_fn, label: str = "Agent") -> None:
    thread_id = str(uuid.uuid4())
    print(_c(_DIM, _REPL_HELP))
    print(_c(_DIM, f"  thread: {thread_id[:8]}"))
    print()

    while True:
        try:
            task = input(_c(_BOLD + _CYAN, "You> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not task:
            continue
        if task in ("/quit", "/exit", "/q"):
            break
        if task == "/new":
            thread_id = str(uuid.uuid4())
            _info(f"New conversation  (thread: {thread_id[:8]})")
            continue
        if task == "/help":
            print(_REPL_HELP)
            continue

        try:
            result = run_fn(task, thread_id=thread_id)
            _print_response(result, label=label)
        except KeyboardInterrupt:
            print()
            _warn("Interrupted.")
        except Exception as exc:
            _error(f"Error: {exc}")

# ─── Agent mode ───────────────────────────────────────────────────────────────

def run_agent(cfg: dict, task: str | None) -> None:
    a = cfg.get("agent", {})
    s = cfg.get("storage", {})

    model     = a.get("model") or ""
    base_url  = a.get("base_url", "http://localhost:11434")
    name      = a.get("name", "agent")
    workspace = a.get("workspace", ".")
    storage_path = s.get("path", "data/lancedb")

    if not model:
        _error("agent.model is not set in config.yml")
        sys.exit(1)

    _header("agent", f"model={model}  workspace={workspace}")

    db_ok = _ensure_db(storage_path)
    store = None
    if db_ok:
        try:
            from storage import SharedMemoryStore
            store = SharedMemoryStore(
                data_path=storage_path,
                embedding_model=s.get("embedding_model", "nomic-embed-text"),
                ollama_base_url=base_url,
            )
        except Exception as exc:
            _warn(f"Could not open memory store: {exc}")

    from agent import OllamaDeepAgent
    agent = OllamaDeepAgent(
        model_name=model,
        base_url=base_url,
        temperature=a.get("temperature", 0.0),
        num_ctx=a.get("num_ctx", 8192),
        workspace_dir=workspace,
        memory_store=store,
        name=name,
        system_prompt=a.get("system_prompt") or None,
        require_permission=a.get("require_permission", True),
        debug=cfg.get("debug", False),
    )
    _info(f"Agent ready  ({name} / {model})")

    if task:
        try:
            result = agent.run(task, thread_id=str(uuid.uuid4()))
            _print_response(result, label=name)
        except KeyboardInterrupt:
            _warn("Interrupted.")
        except Exception as exc:
            _error(f"Error: {exc}")
            sys.exit(1)
    else:
        _repl(agent.run, label=name)

# ─── Coordinator mode ─────────────────────────────────────────────────────────

def run_coordinator(cfg: dict, task: str | None) -> None:
    c = cfg.get("coordinator", {})
    s = cfg.get("storage", {})

    config_path  = c.get("config", "coordinator/config.yml")
    workspace    = c.get("workspace", ".")
    cleanup      = c.get("cleanup_worktrees", False)
    storage_path = s.get("path", "data/lancedb")

    if not Path(config_path).exists():
        _error(f"Coordinator config not found: {config_path}")
        _error("Set coordinator.config in config.yml.")
        sys.exit(1)

    _header("coordinator", f"config={config_path}  workspace={workspace}")

    db_ok = _ensure_db(storage_path)
    store = None
    if db_ok:
        try:
            from storage import SharedMemoryStore
            store = SharedMemoryStore(data_path=storage_path)
        except Exception as exc:
            _warn(f"Could not open memory store: {exc}")

    from coordinator import Coordinator
    coord = Coordinator(
        config_path=config_path,
        workspace_dir=workspace,
        storage_path=storage_path,
        memory_store=store,
        debug=cfg.get("debug", False),
    )

    specialists = list(coord.specialists)
    _info(f"Coordinator ready  ({len(specialists)} specialists: {', '.join(specialists)})")

    def _run(task_text: str, thread_id: str | None = None) -> str:
        return coord.run(task_text, thread_id=thread_id, auto_approve=True)

    try:
        if task:
            result = _run(task)
            _print_response(result, label="Coordinator")
        else:
            _repl(_run, label="Coordinator")
    except KeyboardInterrupt:
        print()
        _warn("Interrupted.")
    finally:
        if cleanup:
            coord.cleanup()
            _info("Worktrees cleaned up.")

# ─── Researcher mode ──────────────────────────────────────────────────────────

def run_researcher(cfg: dict, goal: str | None) -> None:
    r = cfg.get("researcher", {})
    s = cfg.get("storage", {})

    config_path  = r.get("config", "coordinator/config.yml")
    workspace    = r.get("workspace", "")
    user_tools   = r.get("user_tools", "")
    storage_path = s.get("path", "data/lancedb")

    # Validate required fields
    errors = []
    if not workspace or workspace == "/path/to/your/project":
        errors.append("researcher.workspace must be set to your project directory in config.yml")
    if not user_tools:
        errors.append("researcher.user_tools must point to a Python file with evaluate() + save()")
    if not Path(config_path).exists():
        errors.append(f"researcher.config not found: {config_path}")
    for e in errors:
        _error(e)
    if errors:
        sys.exit(1)

    # Prompt file → research goal
    if not goal:
        # Fall back to interactive input
        print(_c(_DIM, "  No goal found in user_prompt.txt."))
        try:
            goal = input(_c(_BOLD + _CYAN, "  Research goal> ")).strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
    if not goal:
        _error("No research goal provided.  Write it in user_prompt.txt.")
        sys.exit(1)

    _header(
        "researcher",
        f"workspace={workspace}  tools={user_tools}",
    )
    print(_c(_DIM, f"  Goal: {goal[:100]}"))
    print()

    _ensure_db(storage_path)

    from coordinator.researcher import AutoResearcher
    researcher = AutoResearcher(
        workspace_dir=workspace,
        user_tools_path=user_tools,
        config_path=config_path,
        storage_path=storage_path,
        debug=cfg.get("debug", False),
    )

    researcher.run(goal)

# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Multi-Agent System. Mode and settings come from config.yml. "
            "The task / research goal comes from user_prompt.txt."
        ),
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        default="config.yml",
        help="Root config file (default: config.yml).",
    )
    parser.add_argument(
        "--prompt",
        metavar="FILE",
        default="user_prompt.txt",
        help="File containing the task or research goal (default: user_prompt.txt).",
    )
    args = parser.parse_args()

    cfg  = _load_config(args.config)
    task = _load_prompt(args.prompt)

    mode = cfg.get("mode", "agent")
    if mode not in ("agent", "coordinator", "researcher"):
        _error(
            f"Unknown mode '{mode}' in {args.config}. "
            "Must be 'agent', 'coordinator', or 'researcher'."
        )
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if cfg.get("debug") else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if task:
        _info(f"{'Goal' if mode == 'researcher' else 'Task'} loaded from {args.prompt}")
    else:
        if mode == "researcher":
            pass  # run_researcher handles the interactive goal prompt
        else:
            _info("No task found — starting interactive mode")

    if mode == "agent":
        run_agent(cfg, task)
    elif mode == "coordinator":
        run_coordinator(cfg, task)
    else:
        run_researcher(cfg, task)


if __name__ == "__main__":
    main()
