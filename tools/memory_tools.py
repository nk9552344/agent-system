"""Memory tools backed by the shared LanceDB store.

All agents sharing the same store instance see each other's notes and
memories. The tools are created per-agent so the agent_name label is
correctly stamped on every record.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

if TYPE_CHECKING:
    from storage.memory_store import SharedMemoryStore


def make_memory_tools(store: "SharedMemoryStore", agent_name: str = "", session_id: str = "") -> list[BaseTool]:  # noqa: F821
    """Return memory tools bound to a shared store.

    Args:
        store: Shared LanceDB store instance (from storage.SharedMemoryStore).
        agent_name: Agent identity stamped on every write.
        session_id: Current session/thread ID stamped on writes.
    """

    # --- Notes (key-value, exact lookup + semantic search) ---

    @tool
    def save_note(key: str, value: str) -> str:
        """Save or overwrite a note in the shared database.

        Notes are visible to ALL agents. Use for facts that should
        persist across sessions and be accessible everywhere, e.g.
        user preferences, project config, discovered API endpoints.

        Args:
            key: Unique identifier (e.g. 'user_email', 'db_host').
            value: Content to remember.
        """
        try:
            store.save_note(key, value, agent_name=agent_name, session_id=session_id)
            return f"Saved note '{key}'."
        except RuntimeError as exc:
            return f"[Memory Error] {exc}"

    @tool
    def recall_note(key: str) -> str:
        """Recall a note by exact key.

        Returns the stored value or 'not found' if the key doesn't exist.

        Args:
            key: The exact key used when the note was saved.
        """
        try:
            value = store.get_note(key)
            if value is not None:
                return f"{key}: {value}"
            notes = store.list_notes()
            available = ", ".join(n["key"] for n in notes[:10]) if notes else "none"
            return f"No note found for '{key}'. Available keys: {available}"
        except RuntimeError as exc:
            return f"[Memory Error] {exc}"

    @tool
    def search_notes(query: str, limit: int = 5) -> str:
        """Search notes by meaning, not just exact key.

        Useful when you don't remember the exact key but know what
        you're looking for. Uses vector search when available, falls
        back to text matching.

        Args:
            query: What you're looking for (natural language).
            limit: Max number of results to return.
        """
        try:
            results = store.search_notes(query, limit=limit)
            if not results:
                return "No matching notes found."
            lines = [f"  {r['key']}: {r['value']}" for r in results]
            return "Matching notes:\n" + "\n".join(lines)
        except RuntimeError as exc:
            return f"[Memory Error] {exc}"

    @tool
    def list_notes(prefix: str = "") -> str:
        """List all saved notes, optionally filtered by key prefix.

        Args:
            prefix: Only show notes whose key starts with this
                    (e.g. 'user_' shows 'user_email', 'user_name', ...).
                    Leave empty to list everything.
        """
        try:
            notes = store.list_notes(prefix=prefix)
            if not notes:
                label = f"starting with '{prefix}'" if prefix else "saved"
                return f"No notes {label}."
            lines = [f"  {n['key']}: {n['value']}" for n in notes]
            return f"Notes ({len(notes)}):\n" + "\n".join(lines)
        except RuntimeError as exc:
            return f"[Memory Error] {exc}"

    @tool
    def delete_note(key: str) -> str:
        """Delete a note from the shared store.

        This affects all agents — only delete notes that are no longer needed.

        Args:
            key: The exact key to delete.
        """
        try:
            deleted = store.delete_note(key)
            return f"Deleted note '{key}'." if deleted else f"No note found for '{key}'."
        except RuntimeError as exc:
            return f"[Memory Error] {exc}"

    # --- Memories (unstructured observations, semantic search) ---

    @tool
    def add_memory(content: str, tags: str = "") -> str:
        """Save a new memory observation to the shared store.

        Use for free-form things you learn: user feedback, discovered
        patterns, errors encountered, decisions made. Memories support
        semantic (similarity) search so you can retrieve relevant ones
        even without knowing exact keywords.

        Args:
            content: What you want to remember (plain text).
            tags: Optional comma-separated labels (e.g. 'python,bug,auth').
        """
        try:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            mem_id = store.add_memory(
                content,
                agent_name=agent_name,
                session_id=session_id,
                tags=tag_list,
            )
            return f"Memory saved (id: {mem_id[:8]}…)."
        except RuntimeError as exc:
            return f"[Memory Error] {exc}"

    @tool
    def search_memories(query: str, limit: int = 5, filter_own: bool = False) -> str:
        """Search past memories by semantic similarity.

        Returns the most relevant memories across all agents (or just
        yours if filter_own=True). Useful before starting a task to
        recall relevant past context.

        Args:
            query: What you want to recall (natural language).
            limit: Number of results (default 5, max 20).
            filter_own: If True, only return memories from this agent.
        """
        try:
            limit = min(limit, 20)
            agent_filter = agent_name if filter_own and agent_name else None
            results = store.search_memories(query, limit=limit, agent_name=agent_filter)
            if not results:
                return "No relevant memories found."
            lines = []
            for r in results:
                ts = time.strftime("%Y-%m-%d", time.localtime(r.get("created_at", 0)))
                who = r.get("agent_name", "?") or "?"
                tags = r.get("tags", "")
                tag_str = f" [{tags}]" if tags else ""
                lines.append(f"  [{ts} | {who}{tag_str}] {r.get('content', '')}")
            return f"Relevant memories ({len(results)}):\n" + "\n".join(lines)
        except RuntimeError as exc:
            return f"[Memory Error] {exc}"

    @tool
    def list_memories(limit: int = 10, filter_own: bool = False) -> str:
        """List recent memories, newest first.

        Args:
            limit: How many to show (default 10, max 50).
            filter_own: If True, only show this agent's memories.
        """
        try:
            limit = min(limit, 50)
            agent_filter = agent_name if filter_own and agent_name else None
            results = store.list_memories(agent_name=agent_filter, limit=limit)
            if not results:
                return "No memories stored yet."
            lines = []
            for r in results:
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.get("created_at", 0)))
                who = r.get("agent_name", "?") or "?"
                lines.append(f"  [{ts} | {who}] {r.get('content', '')[:120]}")
            return f"Recent memories ({len(results)}):\n" + "\n".join(lines)
        except RuntimeError as exc:
            return f"[Memory Error] {exc}"

    return [
        save_note,
        recall_note,
        search_notes,
        list_notes,
        delete_note,
        add_memory,
        search_memories,
        list_memories,
    ]
