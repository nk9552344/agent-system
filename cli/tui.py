"""Full-screen terminal UI for AgentX (built with Textual)."""
from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from rich.markup import escape
from rich.markdown import Markdown
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Input, RichLog, Static

# ── Spinner frames ─────────────────────────────────────────────────────────────
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = """
Screen { background: #1C1917; layers: base; }
#header {
    height: 1; background: #292524; color: #F5F5F4;
    padding: 0 2; content-align: left middle;
}
#activity {
    height: 1; background: #1C1917; color: #57534E;
    padding: 0 2; content-align: left middle;
}
#output {
    height: 1fr; margin: 0 1; border: solid #44403C;
    background: #1C1917;
    scrollbar-color: #57534E #1C1917; scrollbar-size: 1 1;
}
#live-strip {
    height: 1; background: #1C1917; color: #E7E5E4;
    padding: 0 3; content-align: left middle;
}
#divider { height: 1; background: #292524; color: #44403C; margin: 0 1; }
#input-row { height: auto; margin: 0 1 0 1; }
#prompt {
    border: solid #57534E; background: #1C1917; color: #F5F5F4; height: 3;
}
#prompt:focus { border: solid #F97316; }
#footer {
    height: 1; background: #292524; color: #57534E;
    padding: 0 2; content-align: left middle;
}
"""

# ── Styles ─────────────────────────────────────────────────────────────────────
_USER_PREFIX  = "[bold #F97316]You[/bold #F97316] [#57534E]›[/#57534E] "
_TOOL_STYLE   = "#FBBF24"
_RESULT_STYLE = "#78716C"
_MUTED_STYLE  = "#44403C"
_THINK_STYLE  = "#57534E"
_ERROR_STYLE  = "bold #FCA5A5"
_OK_STYLE     = "#86EFAC"
_DIM_STYLE    = "#78716C"

_THINK_OPEN  = "<think>"
_THINK_CLOSE = "</think>"

_FOOTER_IDLE = (
    "  Enter: send  ·  Ctrl+N: new thread  ·  /help: commands  ·  Ctrl+C: quit  "
)
_FOOTER_BUSY = (
    "  [#FBBF24]Ctrl+X: stop[/#FBBF24]  ·  "
    "Ctrl+N: new thread  ·  Ctrl+C: quit  "
)

# Conversation turns kept in context for the next message
_MAX_HISTORY_TURNS = 6
# Max chars per assistant response stored in history (avoids exploding context)
_MAX_RESP_IN_HISTORY = 600


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_args(args: dict) -> str:
    if not args:
        return "()"
    parts = []
    for k, v in list(args.items())[:3]:
        s = str(v)
        if len(s) > 40:
            s = s[:37] + "…"
        parts.append(f'{k}="{s}"')
    inner = ", ".join(parts)
    if len(args) > 3:
        inner += ", …"
    return f"({inner})"


def _partial_tag_suffix(text: str, tag: str) -> bool:
    for n in range(1, len(tag)):
        if text.endswith(tag[:n]):
            return True
    return False


# ── TUI ────────────────────────────────────────────────────────────────────────

class AgentTUI(App):
    CSS = _CSS

    BINDINGS = [
        Binding("ctrl+c", "quit",       "Quit", priority=True, show=False),
        Binding("ctrl+n", "new_thread", "New",  show=False),
        Binding("ctrl+x", "stop_agent", "Stop", show=False),
    ]

    def __init__(
        self,
        runner: Any,
        mode: str,
        model: str,
        workspace: Path | None = None,
        initial_prompt: str | None = None,
        specialist_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._runner           = runner
        self._mode             = mode
        self._model            = model
        self._workspace        = workspace
        self._initial_prompt   = initial_prompt
        self._thread_id: str   = str(uuid.uuid4())
        self._thread_num: int  = 1
        self._busy: bool       = False
        self._tool_wait: bool  = False
        self._spin_frame: int  = 0
        self._specialist_names: list[str]  = specialist_names or []
        self._active_tools: dict[str, int] = {}
        self._stop_event: threading.Event | None = None
        # Conversation history: (user_message, assistant_response)
        self._history: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), id="header")
        yield Static(self._activity_text(), id="activity")
        yield RichLog(
            id="output", wrap=True, markup=True, highlight=False, auto_scroll=True
        )
        yield Static("", id="live-strip")
        yield Static("─" * 200, id="divider")
        with Horizontal(id="input-row"):
            yield Input(
                placeholder="  Type your message… (/help for commands)",
                id="prompt",
            )
        yield Static(_FOOTER_IDLE, id="footer")

    def on_mount(self) -> None:
        log = self._log()
        log.write(
            f"[bold #F97316]◆ AGENTX[/bold #F97316]"
            f"  [#44403C]│[/#44403C]  [#FB923C]{self._mode}[/#FB923C] mode"
            f"  [#44403C]│[/#44403C]  [{_DIM_STYLE}]model: {escape(self._model)}[/{_DIM_STYLE}]"
        )
        if self._workspace:
            log.write(
                f"[{_DIM_STYLE}]Workspace: {escape(str(self._workspace))}[/{_DIM_STYLE}]"
            )
        log.write(
            f"[{_DIM_STYLE}]Thread #1 — Ctrl+N: new thread  ·  "
            f"Ctrl+X: stop  ·  /quit: exit[/{_DIM_STYLE}]"
        )
        log.write("")
        self.query_one("#prompt", Input).focus()
        self.set_interval(1 / 12, self._tick)

        if self._initial_prompt:
            self._submit(self._initial_prompt)

    # ── Spinner ────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._spin_frame = (self._spin_frame + 1) % len(_SPIN)
        if not self._busy:
            return
        frame = _SPIN[self._spin_frame]
        self.query_one("#header", Static).update(
            f"  [bold #F97316]◆ AGENTX[/bold #F97316]"
            f"  [#44403C]·[/#44403C]  {self._mode}"
            f"  [#44403C]·[/#44403C]  [{_DIM_STYLE}]{escape(self._model)}[/{_DIM_STYLE}]"
            f"  [#44403C]·[/#44403C]  [#FBBF24]{frame} thinking…[/#FBBF24]"
            f"  [{_DIM_STYLE}]thread #{self._thread_num}[/{_DIM_STYLE}]"
        )
        if self._tool_wait and self._active_tools:
            names = " · ".join(self._active_tools)
            self.query_one("#live-strip", Static).update(
                f"[{_TOOL_STYLE}]{frame} {escape(names)}…[/{_TOOL_STYLE}]"
            )

    # ── Input ──────────────────────────────────────────────────────────────────

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

    def _handle_command(self, text: str) -> None:
        cmd = text.lower().split()[0]
        log = self._log()
        if cmd in ("/quit", "/exit", "/q"):
            self.exit()
        elif cmd == "/new":
            self.action_new_thread()
        elif cmd == "/stop":
            self.action_stop_agent()
        elif cmd == "/help":
            log.write(
                f"[{_DIM_STYLE}]Commands:  "
                "/new (new thread)  ·  /stop (stop agent)  ·  "
                f"/quit (exit)  ·  /clear (clear screen)[/{_DIM_STYLE}]"
            )
        elif cmd == "/clear":
            log.clear()
        else:
            log.write(f"[{_DIM_STYLE}]Unknown: {escape(text)}  (try /help)[/{_DIM_STYLE}]")

    # ── Actions ────────────────────────────────────────────────────────────────

    def action_quit(self) -> None:
        self.exit()

    def action_new_thread(self) -> None:
        self._thread_id = str(uuid.uuid4())
        self._thread_num += 1
        self._history.clear()
        self._log().write("")
        self._log().write(
            f"[{_OK_STYLE}]✓ New thread #{self._thread_num} started[/{_OK_STYLE}]"
            f"  [{_DIM_STYLE}](context cleared)[/{_DIM_STYLE}]"
        )
        self._log().write("")

    def action_stop_agent(self) -> None:
        if not self._busy:
            return
        if self._stop_event:
            self._stop_event.set()
        self._log().write(
            f"[{_DIM_STYLE}]  ◎  Stop requested — finishing current action…[/{_DIM_STYLE}]"
        )

    # ── Conversation context ───────────────────────────────────────────────────

    def _build_task_with_context(self, task: str) -> str:
        """Prepend recent conversation history so the model has full context."""
        if not self._history:
            return task
        turns = self._history[-_MAX_HISTORY_TURNS:]
        lines = ["## Conversation so far\n"]
        for user_msg, asst_resp in turns:
            resp_preview = asst_resp.strip()
            if len(resp_preview) > _MAX_RESP_IN_HISTORY:
                resp_preview = resp_preview[:_MAX_RESP_IN_HISTORY] + "…"
            lines.append(f"**User:** {user_msg}")
            lines.append(f"**Assistant:** {resp_preview}")
            lines.append("")
        lines.append(f"## Current message\n\n{task}")
        return "\n".join(lines)

    # ── Submit ─────────────────────────────────────────────────────────────────

    def _submit(self, task: str) -> None:
        if self._busy:
            self._log().write(
                f"[{_TOOL_STYLE}]⚠  Agent is busy.  "
                f"Use [bold]Ctrl+X[/bold] or [bold]/stop[/bold] to cancel.[/{_TOOL_STYLE}]"
            )
            return
        self._log().write(f"{_USER_PREFIX}{escape(task)}")
        self._log().write("")
        self._stop_event = threading.Event()
        self._run_agent(task)

    # ── Agent worker ───────────────────────────────────────────────────────────

    @work(thread=True, exclusive=False)
    def _run_agent(self, task: str) -> None:  # noqa: C901
        self._busy = True
        self._tool_wait = False
        self._active_tools.clear()
        self.call_from_thread(self._set_footer, busy=True)
        self.call_from_thread(self._refresh_activity)

        log  = self._log()
        live = self.query_one("#live-strip", Static)
        stop = self._stop_event

        # ── Streaming line writer ───────────────────────────────────────────────
        # para_buf is used ONLY to accumulate code fence blocks for Markdown
        # rendering. Regular text lines are written to #output immediately so
        # content appears in chat line-by-line instead of jumping in all at once
        # after a full paragraph accumulates.
        para_buf: list[str] = []
        in_code_fence = [False]
        full_resp_buf: list[str] = []   # accumulates all response text for history

        def _flush_para() -> None:
            text = "".join(para_buf).strip()
            para_buf.clear()
            if text:
                self.call_from_thread(log.write, Markdown(text))

        def _commit_line(line: str) -> None:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_fence[0] = not in_code_fence[0]
                para_buf.append(line + "\n")
                if not in_code_fence[0]:
                    # Closing fence — render the buffered code block
                    _flush_para()
            elif in_code_fence[0]:
                para_buf.append(line + "\n")
            elif stripped == "":
                self.call_from_thread(log.write, "")
            else:
                # Regular text line — write immediately for live generating effect
                self.call_from_thread(log.write, Markdown(line))

        # ── <think>…</think> state machine ─────────────────────────────────────
        think_state  = "response"
        think_words  = [0]
        tag_buf      = [""]
        resp_line    = [""]
        after_tool   = [False]

        def _emit_resp(text: str) -> None:
            full_resp_buf.append(text)
            parts = text.split("\n")
            for i, part in enumerate(parts):
                resp_line[0] += part
                if i < len(parts) - 1:
                    line = resp_line[0]
                    resp_line[0] = ""
                    if after_tool[0]:
                        self.call_from_thread(log.write, "")
                        after_tool[0] = False
                    _commit_line(line)
            cur = resp_line[0]
            self.call_from_thread(
                live.update,
                f"[#E7E5E4]{escape(cur)}[/#E7E5E4]" if cur.strip() else "",
            )

        def _process_token(text: str) -> None:
            nonlocal think_state
            tag_buf[0] += text
            while tag_buf[0]:
                if think_state == "response":
                    if _THINK_OPEN in tag_buf[0]:
                        idx = tag_buf[0].find(_THINK_OPEN)
                        if idx:
                            _emit_resp(tag_buf[0][:idx])
                        tag_buf[0] = tag_buf[0][idx + len(_THINK_OPEN):]
                        think_state = "thinking"
                        think_words[0] = 0
                        self.call_from_thread(
                            live.update,
                            f"[{_THINK_STYLE} italic]  ◌  thinking…[/{_THINK_STYLE} italic]",
                        )
                    elif _partial_tag_suffix(tag_buf[0], _THINK_OPEN):
                        break
                    else:
                        _emit_resp(tag_buf[0])
                        tag_buf[0] = ""
                else:
                    if _THINK_CLOSE in tag_buf[0]:
                        idx     = tag_buf[0].find(_THINK_CLOSE)
                        content = tag_buf[0][:idx]
                        tag_buf[0] = tag_buf[0][idx + len(_THINK_CLOSE):]
                        if content:
                            think_words[0] += len(content.split())
                        wc = think_words[0]
                        self.call_from_thread(log.write, "")
                        self.call_from_thread(
                            log.write,
                            f"[{_THINK_STYLE}]  ▸ Thought "
                            f"({wc} word{'s' if wc != 1 else ''})[/{_THINK_STYLE}]",
                        )
                        self.call_from_thread(log.write, "")
                        self.call_from_thread(live.update, "")
                        think_state = "response"
                    elif _partial_tag_suffix(tag_buf[0], _THINK_CLOSE):
                        break
                    else:
                        think_words[0] += len(tag_buf[0].split())
                        preview = tag_buf[0].strip().replace("\n", " ")[-80:]
                        self.call_from_thread(
                            live.update,
                            f"[{_THINK_STYLE} italic]  ◌  {escape(preview)}"
                            f"[/{_THINK_STYLE} italic]",
                        )
                        tag_buf[0] = ""

        def _flush_all() -> None:
            if tag_buf[0]:
                _emit_resp(tag_buf[0])
                tag_buf[0] = ""
            if resp_line[0].strip():
                line = resp_line[0]
                resp_line[0] = ""
                if after_tool[0]:
                    self.call_from_thread(log.write, "")
                    after_tool[0] = False
                _commit_line(line)
            _flush_para()
            resp_line[0] = ""
            self.call_from_thread(live.update, "")

        # ── Event loop ─────────────────────────────────────────────────────────
        full_task = self._build_task_with_context(task)
        try:
            for event in self._runner.stream_events(
                full_task, thread_id=self._thread_id, auto_approve=True
            ):
                if stop and stop.is_set():
                    break

                kind = event.get("kind", "")
                text = event.get("text", "")

                if kind == "token":
                    self._tool_wait = False
                    _process_token(text)

                elif kind == "tool_call":
                    _flush_all()
                    tool = event.get("tool", text)
                    args_str = _fmt_args(event.get("args") or {})
                    self._active_tools[tool] = self._active_tools.get(tool, 0) + 1
                    self._tool_wait = True
                    self.call_from_thread(self._refresh_activity)
                    self.call_from_thread(
                        log.write,
                        f"[bold {_TOOL_STYLE}]⚙[/bold {_TOOL_STYLE}]"
                        f" [{_TOOL_STYLE}]{escape(tool)}[/{_TOOL_STYLE}]"
                        f"[{_MUTED_STYLE}]{escape(args_str)}[/{_MUTED_STYLE}]",
                    )
                    after_tool[0] = False

                elif kind == "tool_result":
                    tool = event.get("tool", "")
                    cnt  = self._active_tools.get(tool, 0)
                    if cnt > 1:
                        self._active_tools[tool] = cnt - 1
                    elif tool in self._active_tools:
                        del self._active_tools[tool]
                    if not self._active_tools:
                        self._tool_wait = False
                    self.call_from_thread(self._refresh_activity)
                    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
                    for ln in lines[:5]:
                        self.call_from_thread(
                            log.write,
                            f"[{_RESULT_STYLE}]  ↳  {escape(ln)}[/{_RESULT_STYLE}]",
                        )
                    if len(lines) > 5 or len(text) > 400:
                        self.call_from_thread(
                            log.write, f"[{_MUTED_STYLE}]  ↳  …[/{_MUTED_STYLE}]"
                        )
                    after_tool[0] = True

                elif kind == "status":
                    _flush_all()
                    self.call_from_thread(
                        log.write,
                        f"[{_DIM_STYLE}]  ◎  {escape(text)}[/{_DIM_STYLE}]",
                    )

                elif kind == "error":
                    _flush_all()
                    self.call_from_thread(
                        log.write,
                        f"[bold {_ERROR_STYLE}]Error:[/bold {_ERROR_STYLE}]"
                        f" [#E7E5E4]{escape(text)}[/#E7E5E4]",
                    )

        except Exception as exc:  # noqa: BLE001
            _flush_all()
            self.call_from_thread(
                log.write,
                f"[bold {_ERROR_STYLE}]Error:[/bold {_ERROR_STYLE}]"
                f" [#E7E5E4]{escape(str(exc))}[/#E7E5E4]",
            )

        finally:
            _flush_all()
            stopped = stop is not None and stop.is_set()
            if stopped:
                self.call_from_thread(
                    log.write,
                    f"[{_DIM_STYLE}]  ◎  Stopped.[/{_DIM_STYLE}]",
                )
            self.call_from_thread(log.write, "")

            # Commit this turn to conversation history (skip stopped runs)
            full_resp = "".join(full_resp_buf).strip()
            if full_resp and not stopped:
                self._history.append((task, full_resp))
                if len(self._history) > _MAX_HISTORY_TURNS:
                    self._history = self._history[-_MAX_HISTORY_TURNS:]

            self._tool_wait = False
            self._active_tools.clear()
            self._busy = False
            self.call_from_thread(self._set_status_idle)
            self.call_from_thread(self._set_footer, busy=False)
            self.call_from_thread(self._refresh_activity)

    # ── UI helpers (all called on main thread) ─────────────────────────────────

    def _log(self) -> RichLog:
        return self.query_one("#output", RichLog)

    def _header_text(self) -> str:
        return f"  ◆ AGENTX  ·  {self._mode}  ·  {self._model}"

    def _set_status_idle(self) -> None:
        self.query_one("#header", Static).update(
            f"  [bold #F97316]◆ AGENTX[/bold #F97316]"
            f"  [#44403C]·[/#44403C]  {self._mode}"
            f"  [#44403C]·[/#44403C]  [{_DIM_STYLE}]{escape(self._model)}[/{_DIM_STYLE}]"
            f"  [#44403C]·[/#44403C]  [{_OK_STYLE}]● ready[/{_OK_STYLE}]"
            f"  [{_DIM_STYLE}]thread #{self._thread_num}[/{_DIM_STYLE}]"
        )

    def _set_footer(self, *, busy: bool) -> None:
        self.query_one("#footer", Static).update(
            _FOOTER_BUSY if busy else _FOOTER_IDLE
        )

    def _activity_text(self) -> str:
        if self._mode == "agent":
            team = f"[{_DIM_STYLE}]◈ 1 agent[/{_DIM_STYLE}]"
        else:
            names = ["coordinator"] + self._specialist_names
            sep = f" [{_MUTED_STYLE}]+[/{_MUTED_STYLE}] "
            team = (
                f"[{_DIM_STYLE}]◈ {len(names)} agents:[/{_DIM_STYLE}]  "
                + sep.join(
                    f"[{_DIM_STYLE}]{escape(n)}[/{_DIM_STYLE}]" for n in names
                )
            )

        if not self._active_tools:
            return f"  {team}"

        parts = []
        for name, cnt in self._active_tools.items():
            label = f"⚙ {escape(name)}"
            if cnt > 1:
                label += f" ×{cnt}"
            parts.append(f"[{_TOOL_STYLE}]{label}[/{_TOOL_STYLE}]")

        return (
            f"  {team}"
            f"  [{_MUTED_STYLE}]│[/{_MUTED_STYLE}]  "
            + "  ".join(parts)
        )

    def _refresh_activity(self) -> None:
        self.query_one("#activity", Static).update(self._activity_text())
