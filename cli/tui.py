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

/* ── Activity bar — agent team + active tools ── */
#activity {
    height: 1;
    background: #1C1917;
    color: #57534E;
    padding: 0 2;
    content-align: left middle;
}

/* ── Scrollable output log ── */
#output {
    height: 1fr;
    margin: 0 1;
    border: solid #44403C;
    background: #1C1917;
    color: #E7E5E4;
    scrollbar-color: #57534E #1C1917;
    scrollbar-size: 1 1;
}

/* ── Live streaming strip — always 1 line, updated in place ── */
#live-strip {
    height: 1;
    background: #1C1917;
    color: #E7E5E4;
    padding: 0 3;
    content-align: left middle;
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
_AGENT_STYLE  = "#E7E5E4"        # warm off-white — response text
_TOOL_STYLE   = "#FBBF24"        # amber          — tool call names
_RESULT_STYLE = "#78716C"        # stone-500      — tool result preview
_MUTED_STYLE  = "#44403C"        # stone-700      — truncation markers
_THINK_STYLE  = "#57534E"        # stone-600      — thinking text (live)
_COLLAPSED    = "#57534E"        # collapsed thinking summary
_ERROR_STYLE  = "bold #FCA5A5"   # red-300        — errors
_OK_STYLE     = "#86EFAC"        # green-300      — success / ready
_DIM_STYLE    = "#78716C"        # stone-500      — secondary info

_FOOTER_TEXT = (
    "  Enter: send  ·  Ctrl+N: new thread  ·  /help: commands  ·  Ctrl+C: quit  "
)

# Tags that delimit LLM internal reasoning (e.g. QwQ, DeepSeek-R1, Qwen3)
_THINK_OPEN  = "<think>"
_THINK_CLOSE = "</think>"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_args(args: dict) -> str:
    """Short display string for tool call arguments."""
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


def _has_partial_tag_suffix(text: str, tag: str) -> bool:
    """Return True if *text* ends with a non-empty strict prefix of *tag*.

    Used to detect when a tag might be split across consecutive token chunks.
    """
    for n in range(1, len(tag)):
        if text.endswith(tag[:n]):
            return True
    return False


# ── Main TUI class ─────────────────────────────────────────────────────────────

class AgentTUI(App):
    """Full-screen agent chat interface."""

    CSS = _CSS

    BINDINGS = [
        Binding("ctrl+c", "quit",       "Quit",       priority=True, show=False),
        Binding("ctrl+n", "new_thread", "New thread", show=False),
    ]

    # ── Lifecycle ──────────────────────────────────────────────────────────────

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
        self._active_tools: dict[str, int] = {}   # tool_name → active call count

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

    # ── Input events ───────────────────────────────────────────────────────────

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

    # ── Slash commands ─────────────────────────────────────────────────────────

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
                "/new (new thread)  ·  /quit (exit)  ·  "
                f"/clear (clear output)[/{_DIM_STYLE}]"
            )
        elif cmd == "/clear":
            log.clear()
        else:
            log.write(
                f"[{_DIM_STYLE}]Unknown command: {escape(text)}  "
                f"(try /help)[/{_DIM_STYLE}]"
            )

    # ── Textual actions ────────────────────────────────────────────────────────

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

    # ── Agent execution ────────────────────────────────────────────────────────

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
    def _run_agent(self, task: str) -> None:  # noqa: C901 — long but readable
        self._busy = True
        self._active_tools.clear()
        self.call_from_thread(self._set_status, "thinking…", "#FBBF24")
        self.call_from_thread(self._refresh_activity)

        log  = self._log()
        live = self.query_one("#live-strip", Static)

        # ── State machine for <think>…</think> parsing ─────────────────────
        # Thinking models (QwQ, DeepSeek-R1, Qwen3) wrap reasoning in these
        # tags.  We stream thinking into the live-strip and collapse it to a
        # one-line summary when </think> is detected.
        think_state      = "response"   # "response" | "thinking"
        think_word_count = [0]
        tag_buf          = [""]         # partial tag accumulation
        resp_line        = [""]         # current partial response line
        after_tool       = [False]      # insert blank line before next response

        def _write_resp_line(line: str) -> None:
            if after_tool[0]:
                self.call_from_thread(log.write, "")
                after_tool[0] = False
            if line.strip():
                self.call_from_thread(
                    log.write,
                    f"[{_AGENT_STYLE}]{escape(line)}[/{_AGENT_STYLE}]",
                )
            else:
                self.call_from_thread(log.write, "")

        def _emit_resp(text: str) -> None:
            """Buffer response text and flush complete lines to the log."""
            parts = text.split("\n")
            for i, part in enumerate(parts):
                resp_line[0] += part
                if i < len(parts) - 1:          # there is a newline after this part
                    _write_resp_line(resp_line[0])
                    resp_line[0] = ""
            # Show the partial line in the live-strip
            cur = resp_line[0]
            self.call_from_thread(
                live.update,
                f"[{_AGENT_STYLE}]{escape(cur)}[/{_AGENT_STYLE}]" if cur.strip() else "",
            )

        def _process_token(text: str) -> None:
            """Route one token through the think/response state machine."""
            nonlocal think_state

            tag_buf[0] += text

            while tag_buf[0]:
                if think_state == "response":
                    if _THINK_OPEN in tag_buf[0]:
                        idx = tag_buf[0].find(_THINK_OPEN)
                        before = tag_buf[0][:idx]
                        tag_buf[0] = tag_buf[0][idx + len(_THINK_OPEN):]
                        if before:
                            _emit_resp(before)
                        think_state = "thinking"
                        think_word_count[0] = 0
                        self.call_from_thread(
                            live.update,
                            f"[{_THINK_STYLE} italic]  ◌  thinking…[/{_THINK_STYLE} italic]",
                        )
                    elif _has_partial_tag_suffix(tag_buf[0], _THINK_OPEN):
                        break   # might be start of tag — wait for next token
                    else:
                        _emit_resp(tag_buf[0])
                        tag_buf[0] = ""

                else:  # think_state == "thinking"
                    if _THINK_CLOSE in tag_buf[0]:
                        idx = tag_buf[0].find(_THINK_CLOSE)
                        content = tag_buf[0][:idx]
                        tag_buf[0] = tag_buf[0][idx + len(_THINK_CLOSE):]
                        if content:
                            think_word_count[0] += len(content.split())
                        # Collapse: write a one-line summary to the log
                        self.call_from_thread(log.write, "")
                        wc = think_word_count[0]
                        self.call_from_thread(
                            log.write,
                            f"[{_COLLAPSED}]  ▸ Thought ({wc} word{'s' if wc != 1 else ''})"
                            f"[/{_COLLAPSED}]",
                        )
                        self.call_from_thread(log.write, "")
                        self.call_from_thread(live.update, "")
                        think_state = "response"
                    elif _has_partial_tag_suffix(tag_buf[0], _THINK_CLOSE):
                        break   # partial closing tag — wait
                    else:
                        think_word_count[0] += len(tag_buf[0].split())
                        # Show latest thinking in the live-strip
                        preview = tag_buf[0].strip().replace("\n", " ")[-80:]
                        self.call_from_thread(
                            live.update,
                            f"[{_THINK_STYLE} italic]  ◌  {escape(preview)}"
                            f"[/{_THINK_STYLE} italic]",
                        )
                        tag_buf[0] = ""

        def _flush_all() -> None:
            """Flush all pending buffers at end-of-stream."""
            # Emit anything left in the tag buffer as response text
            if tag_buf[0]:
                _emit_resp(tag_buf[0])
                tag_buf[0] = ""
            # Commit the final partial response line
            if resp_line[0].strip():
                _write_resp_line(resp_line[0])
            resp_line[0] = ""
            # Clear the live-strip
            self.call_from_thread(live.update, "")

        # ── Event loop ─────────────────────────────────────────────────────────
        try:
            for event in self._runner.stream_events(
                task, thread_id=self._thread_id, auto_approve=True
            ):
                kind = event.get("kind", "")
                text = event.get("text", "")

                if kind == "token":
                    _process_token(text)

                elif kind == "tool_call":
                    _flush_all()
                    tool_name = event.get("tool", text)
                    args_str  = _fmt_args(event.get("args") or {})
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
                    after_tool[0] = False

                elif kind == "tool_result":
                    tool_name = event.get("tool", "")
                    cnt = self._active_tools.get(tool_name, 0)
                    if cnt > 1:
                        self._active_tools[tool_name] = cnt - 1
                    elif tool_name in self._active_tools:
                        del self._active_tools[tool_name]
                    self.call_from_thread(self._refresh_activity)
                    # Show up to 5 non-empty lines of the result
                    preview = [ln.strip() for ln in text.strip().split("\n") if ln.strip()][:5]
                    for ln in preview:
                        self.call_from_thread(
                            log.write,
                            f"[{_RESULT_STYLE}]  ↳  {escape(ln)}[/{_RESULT_STYLE}]",
                        )
                    total_lines = len([l for l in text.strip().split("\n") if l.strip()])
                    if total_lines > 5 or len(text) > 400:
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
                        f" [{_AGENT_STYLE}]{escape(text)}[/{_AGENT_STYLE}]",
                    )

        except Exception as exc:  # noqa: BLE001
            _flush_all()
            self.call_from_thread(
                log.write,
                f"[bold {_ERROR_STYLE}]Error:[/bold {_ERROR_STYLE}]"
                f" [{_AGENT_STYLE}]{escape(str(exc))}[/{_AGENT_STYLE}]",
            )

        finally:
            _flush_all()
            self.call_from_thread(log.write, "")
            self._active_tools.clear()
            self._busy = False
            self.call_from_thread(self._set_status, "ready", "#86EFAC")
            self.call_from_thread(self._refresh_activity)

    # ── Helpers ────────────────────────────────────────────────────────────────

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
        """Build activity bar: agent team + currently active tools."""
        # Agent team label
        if self._mode == "agent":
            team_mk = f"[{_DIM_STYLE}]◈ 1 agent[/{_DIM_STYLE}]"
        else:
            names = ["coordinator"] + self._specialist_names
            sep   = f" [{_MUTED_STYLE}]+[/{_MUTED_STYLE}] "
            team_mk = (
                f"[{_DIM_STYLE}]◈ {len(names)} agents:[/{_DIM_STYLE}]  "
                + sep.join(
                    f"[{_DIM_STYLE}]{escape(n)}[/{_DIM_STYLE}]" for n in names
                )
            )

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
