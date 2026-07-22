"""Shared persistent storage for agent memories and notes.

All agent instances that use the same ``data_path`` share the same
LanceDB state. Tables are managed externally — this package only reads
and writes records.

Quick start (dev)::

    # Create tables once (dev only — external service does this in production)
    from storage.schema import bootstrap_tables
    bootstrap_tables("data/lancedb")

    # Then use in agents
    from storage import SharedMemoryStore
    store = SharedMemoryStore("data/lancedb")

    store.add_memory("User prefers dark mode", agent_name="ui-agent")
    store.save_note("project_lang", "Python 3.13")

    results = store.search_memories("user interface preferences")
    value = store.get_note("project_lang")
"""

from storage.memory_store import SharedMemoryStore

__all__ = ["SharedMemoryStore"]
