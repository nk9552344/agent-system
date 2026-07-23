"""Explicit permission-asking tools for human-in-the-loop confirmation.

The agent can call these tools when it needs the user to confirm an action
or provide clarification before proceeding. Unlike the automatic HITL
middleware (which intercepts file/shell tool calls), these tools let the
model initiate a conversation with the user at any point.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool


def make_permission_tools() -> list[BaseTool]:
    """Return permission/confirmation tools."""

    @tool
    def ask_user(question: str) -> str:
        """Ask the user a question and return their text response."""
        print(f"\n[Agent]: {question}")
        try:
            response = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "(no response — user interrupted)"
        return response if response else "(no response)"

    @tool
    def request_confirmation(action_description: str) -> str:
        """Ask the user to confirm an action. Returns 'confirmed' or 'rejected'."""
        print(f"\n[Agent wants to]: {action_description}")
        try:
            response = input("Confirm? [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "rejected (interrupted)"

        if response in ("y", "yes", "ok", "sure", "yep"):
            return "confirmed"
        return f"rejected (user said: '{response}')"

    @tool
    def choose_option(question: str, options: list[str]) -> str:
        """Present numbered choices to the user and return their selection."""
        if not options:
            return "No options provided."

        print(f"\n[Agent]: {question}")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")

        try:
            raw = input("Your choice (number or text): ").strip()
        except (EOFError, KeyboardInterrupt):
            return f"No choice made — defaulting to: {options[0]}"

        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
            return f"Invalid number '{raw}' — defaulting to: {options[0]}"

        # Try to match by prefix
        raw_lower = raw.lower()
        for opt in options:
            if opt.lower().startswith(raw_lower):
                return opt

        return raw if raw else options[0]

    return [ask_user, request_confirmation, choose_option]
