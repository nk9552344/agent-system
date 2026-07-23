"""Full-screen terminal UI for AgentX (built with Textual)."""
from __future__ import annotations

import uuid
from typing import Any

from rich.markup import escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, RichLog, Static


# ── CSS / Theme ───────────────────────────────────────────────────────────────

_CSS = """
Screen {
    background: #0F0A1E;
    layers: base;
}

/* ── Header bar ── */
#header {
    height: 1;
    background: #4C1D95;
    color: #F8FAFC;
    padding: 0 2;
    text-style: bold;
    content-align: left middle;
}

/* ── Output log ── */
#output {
    height: 1fr;
    margin: 0 1;
    border: solid #7C3AED;
    background: #0F0A1E;
    color: #E2E8F0;
    scrollbar-color: #7C3AED #0F0A1E;
    scrollbar-size: 1 1;
}

/* ── Divider ── */
#divider {
    height: 1;
    background: #1E1B4B;
    color: #4C1D95;
    margin: 0 1;
    padding: 0 0;
}

/* ── Input area ── */
#input-row {
    height: auto;
    margin: 0 1 0 1;
}

#prompt {
    border: solid #F97316;
    background: #0F0A1E;
    color: #F8FAFC;
    height: 3;
}

#prompt:focus {
    border: double #F97316;
}

/* ── Footer bar ── */
#footer {
    height: 1;
    background: #1E1B4B;
    color: #6B7280;
    padding: 0 2;
    content-align: left middle;
}
"""

# ── Output line styles (Rich markup) ──────────────────────────────────────────

_USER_PREFIX  = "[bold #A78BFA]You[/bold #A78BFA] [#6B7280]›[/#6B7280] "
_AGENT_STYLE  = "#CBD5E1"
_TOOL_STYLE   = "#F59E0B"
_ERROR_STYLE  = "bold #EF4444"
_OK_STYLE     = "#10B981"
_DIM_STYLE    = "#6B7280"

_FOOTER_TEXT = (
    "  Enter: send  ·  Ctrl+N: new thread  ·  /help: commands  ·  Ctrl+C: quit  "
)


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
        runner: Any,          # OllamaDeepAgent or Coordinator — must have .stream()
        mode: str,
        model: str,
        initial_prompt: str | None = None,
    ) -> None:
        super().__init__()
        self._runner = runner
        self._mode = mode
        self._model = model
        self._initial_prompt = initial_prompt
        self._thread_id: str = str(uuid.uuid4())
        self._thread_num: int = 1
        self._busy: bool = False

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), id="header")
        yield RichLog(id="output", wrap=True, markup=True, highlight=False, auto_scroll=True)
        yield Static("─" * 120, id="divider")
        with Horizontal(id="input-row"):
            yield Input(
                placeholder="  Type your message… (/help for commands)",
                id="prompt",
            )
        yield Static(_FOOTER_TEXT, id="footer")

    def on_mount(self) -> None:
        log = self._log()
        log.write(
            f"[bold #7C3AED]◆ AGENTX[/bold #7C3AED]"
            f"  [#6B7280]│[/#6B7280]  "
            f"[#F97316]{self._mode}[/#F97316] mode"
            f"  [#6B7280]│[/#6B7280]  "
            f"[#6B7280]model: {self._model}[/#6B7280]"
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
                f"[{_DIM_STYLE}]"
                "Commands:  "
                "/new  (new thread)  ·  "
                "/quit  (exit)  ·  "
                "/clear  (clear output)  "
                f"[/{_DIM_STYLE}]"
            )
        elif cmd == "/clear":
            log.clear()
        else:
            log.write(f"[{_DIM_STYLE}]Unknown command: {escape(text)}  (try /help)[/{_DIM_STYLE}]")

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
        self.call_from_thread(self._set_status, "thinking…", "#F97316")

        log = self._log()
        line_buf: list[str] = []

        def _flush(text: str, end: bool = False) -> None:
            """Write accumulated text line-by-line to the RichLog."""
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            parts = text.split("\n")
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    # Full line
                    line_buf.append(part)
                    full = "".join(line_buf)
                    self.call_from_thread(
                        log.write,
                        f"[{_AGENT_STYLE}]{escape(full)}[/{_AGENT_STYLE}]",
                    )
                    line_buf.clear()
                else:
                    # Partial line
                    line_buf.append(part)
            if end and line_buf:
                full = "".join(line_buf)
                if full.strip():
                    self.call_from_thread(
                        log.write,
                        f"[{_AGENT_STYLE}]{escape(full)}[/{_AGENT_STYLE}]",
                    )
                line_buf.clear()

        try:
            for chunk in self._runner.stream(
                task,
                thread_id=self._thread_id,
                auto_approve=True,
            ):
                if chunk:
                    _flush(chunk)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(
                log.write,
                f"[{_ERROR_STYLE}]Error:[/{_ERROR_STYLE}]"
                f" [{_AGENT_STYLE}]{escape(str(exc))}[/{_AGENT_STYLE}]",
            )
        finally:
            _flush("", end=True)
            self.call_from_thread(log.write, "")
            self._busy = False
            self.call_from_thread(self._set_status, "ready", "#10B981")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self) -> RichLog:
        return self.query_one("#output", RichLog)

    def _header_text(self) -> str:
        return (
            f"  ◆ AGENTX  ·  {self._mode}  ·  {self._model}"
        )

    def _set_status(self, status: str, color: str) -> None:
        icon = "⟳" if status == "thinking…" else "●"
        header = self.query_one("#header", Static)
        header.update(
            f"  [bold]◆ AGENTX[/bold]  ·  {self._mode}  ·  {self._model}"
            f"  [{_DIM_STYLE}]│[/{_DIM_STYLE}]  "
            f"[{color}]{icon} {status}[/{color}]  "
            f"  [{_DIM_STYLE}]thread #{self._thread_num}[/{_DIM_STYLE}]"
        )
