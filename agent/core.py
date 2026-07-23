"""OllamaDeepAgent — a fully-featured deepagents agent backed by a local Ollama model.

Each instance is a self-contained, independently invokable agent. Create as many
instances as you need; they don't share any runtime state unless you pass the same
checkpointer/store explicitly.

Example usage::

    from agent import OllamaDeepAgent

    coder = OllamaDeepAgent(
        model_name="qwen2.5-coder:7b",
        system_prompt="You are an expert Python developer.",
        workspace_dir="/tmp/my-project",
    )

    result = coder.run("Refactor main.py to use dataclasses.")
    print(result)

    # Stream tokens as they arrive
    for chunk in coder.stream("Write unit tests for utils.py"):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from deepagents import (
    CompiledSubAgent,
    MemoryMiddleware,
    RubricMiddleware,
    SubAgent,
    create_deep_agent,
)
from deepagents.middleware.async_subagents import AsyncSubAgent
from deepagents.middleware._tool_exclusion import _ToolExclusionMiddleware
from deepagents.backends import FilesystemBackend, LocalShellBackend

# deepagents' built-in file tools require ABSOLUTE paths via a 'file_path' parameter.
# Models naturally use relative paths with 'path'.  We provide workspace-scoped
# replacements in make_file_tools that accept relative paths — these names must be
# excluded from the deepagents tool list so only our versions are visible.
_REPLACED_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "read_file",
    "edit_file",
    "list_directory",
})

from agent.config import AgentConfig
from storage.memory_store import SharedMemoryStore
from tools import WebConfig, make_all_tools

# Appended to every agent's system prompt to reduce hallucination.
_ACCURACY_SUFFIX = """

## Accuracy

- Never invent file contents, paths, or command output — always use a tool to verify.
- If a task cannot be completed as described, say why precisely instead of partially faking it.
"""


class OllamaDeepAgent:
    """A batteries-included agent running on a local Ollama model.

    Wraps deepagents' ``create_deep_agent`` with Ollama-specific defaults and
    a rich built-in tool suite (filesystem, shell, notes, permissions, context).

    Parameters
    ----------
    model_name:
        Ollama model name, e.g. ``'llama3.2'``, ``'qwen2.5-coder:7b'``.
    base_url:
        Ollama server URL (default: ``http://localhost:11434``).
    temperature:
        Model temperature 0–1 (default: ``0.0`` for deterministic outputs).
    num_ctx:
        Context window tokens sent to Ollama (default: ``8192``).
    system_prompt:
        Custom instructions prepended to the base agent prompt.
    workspace_dir:
        Root directory the agent reads/writes files in.
    notes_file:
        JSON file for persistent key-value notes.
        Defaults to ``<workspace_dir>/.agent_notes.json``.
    memory_files:
        Paths to AGENTS.md files loaded as persistent context.
    require_permission:
        Ask user before writing files or running shell commands.
    interrupt_operations:
        Which file operations need approval (default: ``["write", "delete"]``).
    interrupt_on_execute:
        Pause before shell ``execute`` calls (only when require_permission=True).
    persistent_memory:
        Use in-process MemorySaver + InMemoryStore for multi-turn conversations.
    rubric:
        Quality criterion passed to the RubricMiddleware grader.
        When set, a grader sub-agent re-runs after each completion and re-prompts
        the agent if the rubric criterion isn't met (up to 3 iterations).
    execute_timeout:
        Max seconds for the built-in ``execute`` shell tool.
    shell_snippet_timeout:
        Max seconds for ``run_python_snippet`` tool.
    extra_tools:
        Additional tools to include alongside the built-in suite.
    debug:
        Enable LangGraph debug logging.
    name:
        Agent display name used in traces.
    checkpointer:
        Custom LangGraph checkpointer. Overrides ``persistent_memory``.
    store:
        Custom LangGraph store. Overrides ``persistent_memory``.
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_ctx: int = 8192,
        system_prompt: str | None = None,
        workspace_dir: str | Path = ".",
        # --- Shared memory store ---
        memory_store: SharedMemoryStore | None = None,
        storage_path: str | Path = "data/lancedb",
        embedding_model: str | None = "nomic-embed-text",
        # --- AGENTS.md context ---
        memory_files: list[str] | None = None,
        # --- Permissions ---
        require_permission: bool = True,
        interrupt_operations: list[str] | None = None,
        interrupt_on_execute: bool = True,
        # --- Conversation state ---
        persistent_memory: bool = True,
        # --- Quality ---
        rubric: str | None = None,
        # --- Timeouts ---
        execute_timeout: int = 120,
        shell_snippet_timeout: int = 30,
        # --- Sub-agents (for coordinator role) ---
        subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
        # --- Extra ---
        extra_tools: Sequence[BaseTool | Callable] | None = None,
        web_config: WebConfig | None = None,
        debug: bool = False,
        name: str = "ollama-agent",
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
    ) -> None:
        self._model_name = model_name
        self._name = name
        self._thread_counter = 0

        workspace = Path(workspace_dir).resolve()

        # --- Shared memory store --------------------------------------------------
        # Accept an externally created store (for sharing across agents) or create
        # a new one pointing at storage_path. Multiple agents with the same path
        # share state automatically via LanceDB's concurrent-write support.
        if memory_store is not None:
            self._memory_store = memory_store
        else:
            self._memory_store = SharedMemoryStore(
                data_path=storage_path,
                embedding_model=embedding_model,
                ollama_base_url=base_url,
            )

        # --- Model ----------------------------------------------------------------
        model = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
        )

        # --- Backend --------------------------------------------------------------
        backend = LocalShellBackend(
            root_dir=workspace,
            timeout=execute_timeout,
            inherit_env=True,
            virtual_mode=False,
        )

        # --- Permissions ----------------------------------------------------------
        # LocalShellBackend (SandboxBackendProtocol) does not support FilesystemPermission
        # rules, so we use interrupt_on (HumanInTheLoopMiddleware) to gate specific
        # tool calls instead. This gives the same HITL effect without the restriction.
        interrupt_on: dict[str, Any] | None = None

        if require_permission:
            ops_to_interrupt = set(interrupt_operations or ["write", "delete"])

            # Map operation categories to deepagents tool names
            op_to_tools: dict[str, list[str]] = {
                "write":  ["write_file", "edit_file"],
                "delete": ["delete"],
                "read":   ["read_file"],  # usually not interrupted
            }

            hitl_tools: dict[str, bool] = {}
            for op, tool_names in op_to_tools.items():
                if op in ops_to_interrupt:
                    for t in tool_names:
                        hitl_tools[t] = True

            if interrupt_on_execute:
                hitl_tools["execute"] = True

            interrupt_on = hitl_tools if hitl_tools else None

        # --- Middleware -----------------------------------------------------------
        middleware: list[Any] = []

        # Exclude deepagents' built-in file tools that require absolute 'file_path'.
        # Our workspace-scoped replacements (in make_file_tools) use relative 'path'.
        middleware.append(_ToolExclusionMiddleware(excluded=_REPLACED_TOOLS))

        if memory_files:
            fs_backend = FilesystemBackend(root_dir="/")
            middleware.append(
                MemoryMiddleware(backend=fs_backend, sources=memory_files)
            )

        if rubric:
            # RubricMiddleware grader uses the same model; rubric becomes its system_prompt.
            rubric_system = (
                f"You are a grader evaluating an AI agent's response.\n\n"
                f"Rubric (all criteria must be met for 'satisfied'):\n{rubric}"
            )
            middleware.append(
                RubricMiddleware(
                    model=model,
                    system_prompt=rubric_system,
                    max_iterations=3,
                )
            )

        # --- Checkpointer & Store -------------------------------------------------
        if checkpointer is None and persistent_memory:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()

        if store is None and persistent_memory:
            from langgraph.store.memory import InMemoryStore
            store = InMemoryStore()

        # --- Tools ----------------------------------------------------------------
        custom_tools = make_all_tools(
            workspace_dir=workspace,
            memory_store=self._memory_store,
            agent_name=name,
            shell_timeout=shell_snippet_timeout,
            web_config=web_config,
        )
        all_tools: list[BaseTool | Callable] = [*custom_tools, *(extra_tools or [])]

        # --- System prompt --------------------------------------------------------
        combined_prompt = _build_system_prompt(system_prompt)

        # --- Assemble graph -------------------------------------------------------
        self._graph: CompiledStateGraph = create_deep_agent(
            model=model,
            tools=all_tools,
            system_prompt=combined_prompt,
            middleware=middleware,
            backend=backend,
            subagents=list(subagents) if subagents is not None else None,
            interrupt_on=interrupt_on,
            checkpointer=checkpointer,
            store=store,
            debug=debug,
            name=name,
        )

    # --------------------------------------------------------------------------
    # Public interface
    # --------------------------------------------------------------------------

    @property
    def memory_store(self) -> SharedMemoryStore:
        """The shared LanceDB store used by this agent.

        Pass this to another agent's ``memory_store=`` parameter to make
        two agents share the same central memory::

            store = agent1.memory_store
            agent2 = OllamaDeepAgent("llama3.2", memory_store=store)
        """
        return self._memory_store

    @property
    def graph(self) -> CompiledStateGraph:
        """Underlying LangGraph compiled graph.

        Use this for advanced control: streaming with custom stream_mode,
        sub-graphing, custom interrupt handling, or passing the agent as a
        sub-agent to another deepagents instance.
        """
        return self._graph

    def new_thread_id(self) -> str:
        """Generate a unique thread ID for a new independent conversation."""
        self._thread_counter += 1
        return f"{self._name}-thread-{self._thread_counter}-{uuid.uuid4().hex[:8]}"

    def run(
        self,
        task: str,
        *,
        thread_id: str | None = None,
        auto_approve: bool = False,
    ) -> str:
        """Run a task synchronously and return the agent's final text response.

        Handles human-in-the-loop interrupts automatically: if the agent asks
        for permission (file write, shell command), the user is prompted in the
        terminal. Set ``auto_approve=True`` to silently approve all interrupts
        (useful in automated pipelines with trusted tasks).

        Parameters
        ----------
        task:
            The instruction or question to send to the agent.
        thread_id:
            Conversation thread ID. Reuse the same ID to continue a prior
            conversation; pass a new one (or None) to start fresh.
        auto_approve:
            Automatically approve all HITL interrupts without prompting.
        """
        from langgraph.types import Command

        tid = thread_id or self.new_thread_id()
        config = {"configurable": {"thread_id": tid}}
        state = self._graph.invoke({"messages": task}, config=config)

        # Handle human-in-the-loop interrupts
        while state.get("__interrupt__"):
            interrupt = state["__interrupt__"][0]
            approved = self._handle_interrupt(interrupt, auto_approve=auto_approve)
            state = self._graph.invoke(Command(resume=approved), config=config)

        return self._extract_response(state)

    def stream(
        self,
        task: str,
        *,
        thread_id: str | None = None,
        auto_approve: bool = False,
    ) -> Iterator[str]:
        """Stream the agent's response token by token.

        Yields text chunks as they are generated. Interrupts are handled
        the same way as in ``run()``.

        Parameters
        ----------
        task:
            The instruction or question to send to the agent.
        thread_id:
            Conversation thread ID for continuing an existing conversation.
        auto_approve:
            Automatically approve all HITL interrupts.
        """
        from langgraph.types import Command

        tid = thread_id or self.new_thread_id()
        config = {"configurable": {"thread_id": tid}}
        seen: set[str] = set()

        def _stream_and_yield(inp: Any) -> dict[str, Any]:
            last: dict[str, Any] = {}
            for chunk in self._graph.stream(inp, config=config, stream_mode="values"):
                last = chunk
                for text in _extract_new_text(chunk, seen):
                    yield text
            return last  # type: ignore[return-value]

        # First call — yield initial stream
        last_state: dict[str, Any] = {}
        gen = _stream_and_yield({"messages": task})
        try:
            while True:
                text = next(gen)  # type: ignore[arg-type]
                yield text
        except StopIteration as exc:
            last_state = exc.value or {}

        # Re-fetch state for interrupt check (stream doesn't surface __interrupt__)
        try:
            snapshot = self._graph.get_state(config)
            state = snapshot.values if snapshot else last_state
        except Exception:
            state = last_state

        while state.get("__interrupt__"):
            interrupt = state["__interrupt__"][0]
            approved = self._handle_interrupt(interrupt, auto_approve=auto_approve)
            gen2 = _stream_and_yield(Command(resume=approved))
            try:
                while True:
                    text = next(gen2)  # type: ignore[arg-type]
                    yield text
            except StopIteration as exc:
                last_state = exc.value or {}
            try:
                snapshot = self._graph.get_state(config)
                state = snapshot.values if snapshot else {}
            except Exception:
                state = {}

    def stream_events(
        self,
        task: str,
        *,
        thread_id: str | None = None,
        auto_approve: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Stream all agent activity as structured events for real-time TUI display.

        Uses LangChain callbacks so every token fires immediately, regardless of
        LangGraph's internal streaming mode or middleware buffering.

        Yields dicts:
          {"kind": "token",       "text": str}
          {"kind": "tool_call",   "text": str, "tool": str, "args": dict}
          {"kind": "tool_result", "text": str, "tool": str}
          {"kind": "status",      "text": str}
          {"kind": "error",       "text": str}
        """
        import json
        import threading
        from queue import SimpleQueue

        from langchain_core.callbacks import BaseCallbackHandler
        from langgraph.types import Command

        tid    = thread_id or self.new_thread_id()
        config = {"configurable": {"thread_id": tid}}

        _DONE: object = object()
        q: SimpleQueue = SimpleQueue()

        class _CB(BaseCallbackHandler):
            def __init__(self) -> None:
                super().__init__()
                self._tool_names: dict[str, str] = {}

            # ── token streaming ────────────────────────────────────────────
            def on_llm_new_token(self, token: str, **_: Any) -> None:
                if token:
                    q.put({"kind": "token", "text": token})

            # ── tool lifecycle ─────────────────────────────────────────────
            def on_tool_start(
                self,
                serialized: dict,
                input_str: Any,
                *,
                run_id: Any = None,
                **_: Any,
            ) -> None:
                name = (
                    serialized.get("name", "") if isinstance(serialized, dict) else ""
                ) or ""
                if not name:
                    return
                if run_id:
                    self._tool_names[str(run_id)] = name
                if isinstance(input_str, dict):
                    args: dict = input_str
                elif isinstance(input_str, str):
                    try:
                        args = json.loads(input_str)
                    except Exception:
                        args = {}
                else:
                    args = {}
                q.put({"kind": "tool_call", "text": name, "tool": name, "args": args})

            def on_tool_end(
                self,
                output: Any,
                *,
                run_id: Any = None,
                **_: Any,
            ) -> None:
                name = self._tool_names.pop(str(run_id), "") if run_id else ""
                q.put({"kind": "tool_result", "text": str(output or ""), "tool": name})

            def on_tool_error(
                self,
                error: Any,
                *,
                run_id: Any = None,
                **_: Any,
            ) -> None:
                name = self._tool_names.pop(str(run_id), "") if run_id else ""
                q.put({"kind": "error", "text": f"[{name}] {error}"})

        cb = _CB()
        exc_box: list[BaseException | None] = [None]

        def _invoke(inp: Any) -> None:
            try:
                self._graph.invoke(inp, config={**config, "callbacks": [cb]})
            except Exception as exc:  # noqa: BLE001
                exc_box[0] = exc
            finally:
                q.put(_DONE)

        inp: Any = {"messages": task}
        while True:
            exc_box[0] = None
            t = threading.Thread(target=_invoke, args=(inp,), daemon=True)
            t.start()

            # Drain the queue until the invoke thread signals done
            while True:
                item = q.get()
                if item is _DONE:
                    break
                yield item  # type: ignore[misc]

            t.join()

            if exc_box[0] is not None:
                yield {"kind": "error", "text": str(exc_box[0])}
                break

            # Check for HITL interrupt
            try:
                snap  = self._graph.get_state(config)
                state = snap.values if snap else {}
            except Exception:
                break

            if not state.get("__interrupt__"):
                break

            interrupt = state["__interrupt__"][0]
            if auto_approve:
                yield {"kind": "status", "text": "Auto-approving action…"}
                inp = Command(resume=True)
            else:
                approved = self._handle_interrupt(interrupt, auto_approve=False)
                inp = Command(resume=approved)

    def invoke_raw(
        self,
        input_: dict[str, Any],
        *,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Pass input directly to the underlying LangGraph graph and return raw state.

        Use this when you need the full state dict (e.g. for tools output, metadata,
        or chaining with other LangGraph graphs).
        """
        config = {"configurable": {"thread_id": thread_id or self.new_thread_id()}}
        return self._graph.invoke(input_, config=config, **kwargs)

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    def _handle_interrupt(self, interrupt: Any, *, auto_approve: bool) -> bool:
        """Print interrupt info and ask the user whether to approve or reject."""
        if auto_approve:
            return True

        print("\n" + "=" * 60)
        print("[HITL] Agent wants to perform a privileged action:")

        value = getattr(interrupt, "value", interrupt)
        if isinstance(value, dict):
            tool_name = value.get("tool_name") or value.get("tool") or "unknown"
            args = value.get("args") or value.get("tool_input") or {}
            print(f"  Tool : {tool_name}")
            if args:
                for k, v in (args.items() if isinstance(args, dict) else [("input", args)]):
                    preview = str(v)[:200] + ("..." if len(str(v)) > 200 else "")
                    print(f"  {k:8}: {preview}")
        else:
            print(f"  {value}")
        print("=" * 60)

        try:
            response = input("Approve? [y/n] (default: n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[HITL] Interrupted — rejecting.")
            return False

        return response in ("y", "yes", "ok", "approve")

    @staticmethod
    def _extract_response(state: dict[str, Any]) -> str:
        """Extract the last AI message text from graph state."""
        messages = state.get("messages", [])
        if not messages:
            return ""
        last = messages[-1]
        content = getattr(last, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)

    def __repr__(self) -> str:
        return f"OllamaDeepAgent(model={self._model_name!r}, name={self._name!r})"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_system_prompt(user_prefix: str | None) -> str:
    """Combine user-supplied prefix with the accuracy suffix."""
    parts: list[str] = []
    if user_prefix:
        parts.append(user_prefix.strip())
    parts.append(_ACCURACY_SUFFIX.strip())
    return "\n\n".join(parts)


def _extract_new_text(chunk: dict[str, Any], seen: set[str]) -> list[str]:
    """Pull new AI message text out of a stream chunk, deduplicating by (id, content)."""
    results: list[str] = []
    for msg in chunk.get("messages", []):
        content = getattr(msg, "content", "")
        msg_id = getattr(msg, "id", None)
        key = f"{msg_id}:{id(content)}"
        if not content or key in seen:
            continue
        seen.add(key)
        if isinstance(content, str):
            results.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    results.append(block.get("text", ""))
    return results
