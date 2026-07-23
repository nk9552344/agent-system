"""Full-screen terminal UI for AgentX (built with Textual)."""
from __future__ import annotations

import uuid
from typing import Any

from rich.markup import escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Input, RichLog, Static


# ── CSS / Theme — warm dark, orange accents ────────────────────────────────────

_CSS = """
Screen {
    background: #1C1917;
    layers: base;
}

/* ── Header bar ── */
#header {
    height: 1;
    background: #292524;
    color: #F5F5F4;
    padding: 0 2;
    text-style: bold;
    content-align: left middle;
}

/* ── Activity bar — shows agents + active tools ── */
#activity {
    height: 1;
    background: #1C1917;
    color: #57534E;
    padding: 0 2;
    content-align: left middle;
}

/* ── Output log ── */
#output {
    height: 1fr;
    margin: 0 1;
    border: solid #44403C;
    background: #1C1917;
    color: #E7E5E4;
    scrollbar-color: #57534E #1C1917;
    scrollbar-size: 1 1;
}

/* ── Divider ── */
#divider {
    height: 1;
    background: #292524;
    color: #44403C;
    margin: 0 1;
}

/* ── Input area ── */
#input-row {
    height: auto;
    margin: 0 1 0 1;
}

#prompt {
    border: solid #57534E;
    background: #1C1917;
    color: #F5F5F4;
    height: 3;
}

#prompt:focus {
    border: solid #F97316;
}

/* ── Footer bar ── */
#footer {
    height: 1;
    background: #292524;
    color: #57534E;
    padding: 0 2;
    content-align: left middle;
}
"""

# ── Output line styles (Rich markup) ──────────────────────────────────────────

_USER_PREFIX  = "[bold #F97316]You[/bold #F97316] [#57534E]›[/#57534E] "
_AGENT_STYLE  = "#E7E5E4"        # warm off-white — agent response text
_TOOL_STYLE   = "#FBBF24"        # amber          — tool calls
_RESULT_STYLE = "#78716C"        # stone-500      — tool result preview
_MUTED_STYLE  = "#44403C"        # stone-700      — very dim (truncation marks)
_ERROR_STYLE  = "bold #FCA5A5"   # red-300        — errors
_OK_STYLE     = "#86EFAC"        # green-300      — success / ready
_DIM_STYLE    = "#78716C"        # stone-500      — secondary info

_FOOTER_TEXT = (
    "  Enter: send  ·  Ctrl+N: new thread  ·  /help: commands  ·  Ctrl+C: quit  "
)


def _fmt_args(args: dict) -> str:
    """Format tool call args into a short display string."""
    if not args:
        return "()"
    parts = []
    for k, v in list(args.items())[:3]:
        v_str = str(v)
        if len(v_str) > 40:
            v_str = v_str[:37] + "…"
        parts.append(f'{k}="{v_str}"')
    inner = ", ".join(parts)
    if len(args) > 3:
        inner += ", …"
    return f"({inner})"


class AgentTUI(App):
    """Full-screen agent chat interface."""

    CSS = _CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True, show=False),
        Binding("ctrl+n", "new_thread", "New thread", show=False),
    ]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __init__(
        self,
        runner: Any,                         # OllamaDeepAgent or Coordinator
        mode: str,
        model: str,
        initial_prompt: str | None = None,
        specialist_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._runner          = runner
        self._mode            = mode
        self._model           = model
        self._initial_prompt  = initial_prompt
        self._thread_id: str  = str(uuid.uuid4())
        self._thread_num: int = 1
        self._busy: bool      = False
        self._specialist_names: list[str] = specialist_names or []
        # Maps tool_name → active call count
        self._active_tools: dict[str, int] = {}

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), id="header")
        yield Static(self._activity_text(), id="activity")
        yield RichLog(
            id="output", wrap=True, markup=True, highlight=False, auto_scroll=True
        )
        yield Static("─" * 200, id="divider")
        with Horizontal(id="input-row"):
            yield Input(
                placeholder="  Type your message… (/help for commands)",
                id="prompt",
            )
        yield Static(_FOOTER_TEXT, id="footer")

    def on_mount(self) -> None:
        log = self._log()
        log.write(
            f"[bold #F97316]◆ AGENTX[/bold #F97316]"
            f"  [#44403C]│[/#44403C]  "
            f"[#FB923C]{self._mode}[/#FB923C] mode"
            f"  [#44403C]│[/#44403C]  "
            f"[{_DIM_STYLE}]model: {escape(self._model)}[/{_DIM_STYLE}]"
        )
        log.write(
            f"[{_DIM_STYLE}]Thread #1 started.  "
            f"Ctrl+N for a new thread, /quit to exit.[/{_DIM_STYLE}]"
        )
        log.write("")
        self.query_one("#prompt", Input).focus()

        if self._initial_prompt:
            self._submit(self._initial_prompt)

    # ── Events ────────────────────────────────────────────────────────────────

    @on(Input.Submitted, "#prompt")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text:
            return
        if text.startswith("/"):
            self._handle_command(text)
            return
        self._submit(text)

    # ── Commands ──────────────────────────────────────────────────────────────

    def _handle_command(self, text: str) -> None:
        cmd = text.lower().split()[0]
        log = self._log()

        if cmd in ("/quit", "/exit", "/q"):
            self.exit()
        elif cmd == "/new":
            self.action_new_thread()
        elif cmd == "/help":
            log.write(
                f"[{_DIM_STYLE}]Commands:  "
                "/new (new thread)  ·  "
                "/quit (exit)  ·  "
                f"/clear (clear output)[/{_DIM_STYLE}]"
            )
        elif cmd == "/clear":
            log.clear()
        else:
            log.write(
                f"[{_DIM_STYLE}]Unknown command: {escape(text)}  "
                f"(try /help)[/{_DIM_STYLE}]"
            )

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_quit(self) -> None:
        self.exit()

    def action_new_thread(self) -> None:
        self._thread_id = str(uuid.uuid4())
        self._thread_num += 1
        log = self._log()
        log.write("")
        log.write(
            f"[{_OK_STYLE}]✓ New thread #{self._thread_num} started.[/{_OK_STYLE}]"
            f"  [{_DIM_STYLE}](previous context cleared)[/{_DIM_STYLE}]"
        )
        log.write("")

    # ── Agent execution ───────────────────────────────────────────────────────

    def _submit(self, task: str) -> None:
        if self._busy:
            self._log().write(
                f"[{_TOOL_STYLE}]⚠  Agent is processing — please wait…[/{_TOOL_STYLE}]"
            )
            return
        log = self._log()
        log.write(f"{_USER_PREFIX}{escape(task)}")
        log.write("")
        self._run_agent(task)

    @work(thread=True, exclusive=False)
    def _run_agent(self, task: str) -> None:
        self._busy = True
        self._active_tools.clear()
        self.call_from_thread(self._set_status, "thinking…", "#FBBF24")
        self.call_from_thread(self._refresh_activity)

        log = self._log()
        token_buf: list[str] = []
        after_tool_result = False  # insert blank line before first response text

        # ── Token buffer helpers ───────────────────────────────────────────────

        def _write_response_line(line: str) -> None:
            nonlocal after_tool_result
            if after_tool_result:
                self.call_from_thread(log.write, "")
                after_tool_result = False
            if line.strip():
                self.call_from_thread(
                    log.write,
                    f"[{_AGENT_STYLE}]{escape(line)}[/{_AGENT_STYLE}]",
                )
            else:
                self.call_from_thread(log.write, "")

        def _process_tokens(text: str, *, flush: bool = False) -> None:
            token_buf.append(text)
            combined = "".join(token_buf)
            parts    = combined.split("\n")
            if flush:
                token_buf.clear()
                for ln in parts:
                    _write_response_line(ln)
            else:
                # Write all complete lines; keep the trailing partial in the buffer
                for ln in parts[:-1]:
                    _write_response_line(ln)
                token_buf.clear()
                if parts[-1]:
                    token_buf.append(parts[-1])

        # ── Main event loop ────────────────────────────────────────────────────

        try:
            for event in self._runner.stream_events(
                task, thread_id=self._thread_id, auto_approve=True
            ):
                kind = event.get("kind", "")
                text = event.get("text", "")

                if kind == "token":
                    _process_tokens(text)

                elif kind == "tool_call":
                    _process_tokens("", flush=True)
                    tool_name  = event.get("tool", text)
                    args_str   = _fmt_args(event.get("args") or {})
                    # Track active tool counts
                    self._active_tools[tool_name] = (
                        self._active_tools.get(tool_name, 0) + 1
                    )
                    self.call_from_thread(self._refresh_activity)
                    self.call_from_thread(
                        log.write,
                        f"[bold {_TOOL_STYLE}]⚙[/bold {_TOOL_STYLE}]"
                        f" [{_TOOL_STYLE}]{escape(tool_name)}[/{_TOOL_STYLE}]"
                        f"[{_MUTED_STYLE}]{escape(args_str)}[/{_MUTED_STYLE}]",
                    )
                    after_tool_result = False

                elif kind == "tool_result":
                    tool_name = event.get("tool", "")
                    # Decrement active count
                    cnt = self._active_tools.get(tool_name, 0)
                    if cnt > 1:
                        self._active_tools[tool_name] = cnt - 1
                    elif tool_name in self._active_tools:
                        del self._active_tools[tool_name]
                    self.call_from_thread(self._refresh_activity)
                    # Show result preview (up to 6 non-empty lines)
                    preview_lines = [
                        ln.strip() for ln in text.strip().split("\n") if ln.strip()
                    ][:6]
                    for ln in preview_lines:
                        self.call_from_thread(
                            log.write,
                            f"[{_RESULT_STYLE}]  ↳  {escape(ln)}[/{_RESULT_STYLE}]",
                        )
                    remaining = len(text.strip().split("\n")) - len(preview_lines)
                    if remaining > 0 or len(text) > 500:
                        self.call_from_thread(
                            log.write,
                            f"[{_MUTED_STYLE}]  ↳  …[/{_MUTED_STYLE}]",
                        )
                    after_tool_result = True

                elif kind == "status":
                    _process_tokens("", flush=True)
                    self.call_from_thread(
                        log.write,
                        f"[{_DIM_STYLE}]  ◎  {escape(text)}[/{_DIM_STYLE}]",
                    )

                elif kind == "error":
                    _process_tokens("", flush=True)
                    self.call_from_thread(
                        log.write,
                        f"[bold {_ERROR_STYLE}]Error:[/bold {_ERROR_STYLE}]"
                        f" [{_AGENT_STYLE}]{escape(text)}[/{_AGENT_STYLE}]",
                    )

        except Exception as exc:  # noqa: BLE001
            _process_tokens("", flush=True)
            self.call_from_thread(
                log.write,
                f"[bold {_ERROR_STYLE}]Error:[/bold {_ERROR_STYLE}]"
                f" [{_AGENT_STYLE}]{escape(str(exc))}[/{_AGENT_STYLE}]",
            )

        finally:
            _process_tokens("", flush=True)
            self.call_from_thread(log.write, "")
            self._active_tools.clear()
            self._busy = False
            self.call_from_thread(self._set_status, "ready", "#86EFAC")
            self.call_from_thread(self._refresh_activity)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self) -> RichLog:
        return self.query_one("#output", RichLog)

    def _header_text(self) -> str:
        return f"  ◆ AGENTX  ·  {self._mode}  ·  {self._model}"

    def _set_status(self, status: str, color: str) -> None:
        icon = "⟳" if status == "thinking…" else "●"
        self.query_one("#header", Static).update(
            f"  [bold #F97316]◆ AGENTX[/bold #F97316]"
            f"  [#44403C]·[/#44403C]  {self._mode}"
            f"  [#44403C]·[/#44403C]  [{_DIM_STYLE}]{escape(self._model)}[/{_DIM_STYLE}]"
            f"  [#44403C]·[/#44403C]  [{color}]{icon} {status}[/{color}]"
            f"  [{_DIM_STYLE}]thread #{self._thread_num}[/{_DIM_STYLE}]"
        )

    def _activity_text(self) -> str:
        """Build the activity bar markup: agent team + currently active tools."""
        # ── Agent team label ──────────────────────────────────────────────────
        if self._mode == "agent":
            team_mk = f"[{_DIM_STYLE}]◈ 1 agent[/{_DIM_STYLE}]"
        else:
            names = ["coordinator"] + self._specialist_names
            count = len(names)
            sep   = f" [{_MUTED_STYLE}]+[/{_MUTED_STYLE}] "
            team_mk = (
                f"[{_DIM_STYLE}]◈ {count} agents:[/{_DIM_STYLE}]  "
                + sep.join(
                    f"[{_DIM_STYLE}]{escape(n)}[/{_DIM_STYLE}]" for n in names
                )
            )

        # ── Active tools ──────────────────────────────────────────────────────
        if not self._active_tools:
            return f"  {team_mk}"

        tool_parts = []
        for name, cnt in self._active_tools.items():
            label = f"⚙ {escape(name)}"
            if cnt > 1:
                label += f" ×{cnt}"
            tool_parts.append(f"[{_TOOL_STYLE}]{label}[/{_TOOL_STYLE}]")

        return (
            f"  {team_mk}"
            f"  [{_MUTED_STYLE}]│[/{_MUTED_STYLE}]  "
            + "  ".join(tool_parts)
        )

    def _refresh_activity(self) -> None:
        self.query_one("#activity", Static).update(self._activity_text())
