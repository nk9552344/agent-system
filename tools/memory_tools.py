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
        """Save a key-value note to the in-memory store. Does NOT create or modify any file on disk."""
        try:
            store.save_note(key, value, agent_name=agent_name, session_id=session_id)
            return f"Saved note '{key}'."
        except RuntimeError as exc:
            return f"[Memory Error] {exc}"

    @tool
    def recall_note(key: str) -> str:
        """Retrieve a previously saved note by its exact key."""
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
        """Search saved notes by semantic similarity or keyword."""
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
        """List all saved notes, optionally filtered by key prefix."""
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
        """Delete a saved note by key."""
        try:
            deleted = store.delete_note(key)
            return f"Deleted note '{key}'." if deleted else f"No note found for '{key}'."
        except RuntimeError as exc:
            return f"[Memory Error] {exc}"

    # --- Memories (unstructured observations, semantic search) ---

    @tool
    def add_memory(content: str, tags: str = "") -> str:
        """Save a free-form observation to long-term memory (searchable later)."""
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
        """Search past memory observations by semantic similarity."""
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
        """List recent memory observations, newest first."""
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
