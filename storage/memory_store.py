"""Shared LanceDB-backed memory store used by all agent instances.

The database at ``data/lancedb/`` is managed externally (tables already exist).
This class only reads and writes records — no schema creation.

Supports:
- Semantic search over memories via Ollama embeddings
- Exact key-value notes with merge_insert upsert
- Concurrent access from multiple agent instances (LanceDB handles isolation)

Tables expected: ``memories``, ``notes`` (see ``storage/schema.py`` for schema).
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SharedMemoryStore:
    """Shared LanceDB store for agent memories and notes.

    All agents pointing at the same ``data_path`` share state. LanceDB
    handles concurrent reads and writes safely at the file level.

    Parameters
    ----------
    data_path:
        Path to the LanceDB directory. Defaults to ``data/lancedb``.
    memories_table:
        Name of the memories table (default: ``"memories"``).
    notes_table:
        Name of the notes table (default: ``"notes"``).
    embedding_model:
        Ollama model used for text embeddings.
        Set to ``None`` to disable vector search (falls back to full scan).
        Default: ``"nomic-embed-text"``.
    ollama_base_url:
        Ollama server URL for embedding calls.
    """

    def __init__(
        self,
        data_path: str | Path = "data/lancedb",
        *,
        memories_table: str = "memories",
        notes_table: str = "notes",
        embedding_model: str | None = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        try:
            import lancedb
        except ImportError as exc:
            raise ImportError(
                "lancedb is required for SharedMemoryStore. "
                "Install it with: uv add lancedb"
            ) from exc

        self._data_path = Path(data_path).resolve()
        self._db = lancedb.connect(str(self._data_path))
        self._memories_table = memories_table
        self._notes_table = notes_table
        self._embedding_model = embedding_model
        self._ollama_base_url = ollama_base_url
        self._embed_available: bool | None = None  # None = untested yet

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float] | None:
        """Return an embedding vector for text, or None if unavailable."""
        if self._embedding_model is None or self._embed_available is False:
            return None
        try:
            import ollama

            client = ollama.Client(host=self._ollama_base_url)
            resp = client.embeddings(model=self._embedding_model, prompt=text)
            vector = resp.get("embedding") or []
            if vector:
                self._embed_available = True
                return [float(v) for v in vector]
            self._embed_available = False
            return None
        except Exception as exc:
            if self._embed_available is None:
                logger.warning(
                    "Ollama embedding unavailable (%s) — falling back to "
                    "full-text / recency scan for memory search.",
                    exc,
                )
            self._embed_available = False
            return None

    # ------------------------------------------------------------------
    # Internal table access
    # ------------------------------------------------------------------

    def _table(self, name: str) -> Any:
        """Open an existing LanceDB table; raises clearly if not found."""
        try:
            return self._db.open_table(name)
        except Exception as exc:
            raise RuntimeError(
                f"Table '{name}' not found in LanceDB at '{self._data_path}'. "
                f"Run `storage.schema.bootstrap_tables()` for dev, or ensure "
                f"the external service has created it. Error: {exc}"
            ) from exc

    @staticmethod
    def _sql_str(value: str) -> str:
        """Escape a string value for use in LanceDB SQL predicates."""
        return value.replace("'", "''")

    @staticmethod
    def _clean(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip vector and _distance columns — too noisy to return to agent."""
        return [
            {k: v for k, v in r.items() if k not in ("vector", "_distance")}
            for r in records
        ]

    # ------------------------------------------------------------------
    # Memories API
    # ------------------------------------------------------------------

    def add_memory(
        self,
        content: str,
        *,
        agent_name: str = "",
        session_id: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Persist a new memory observation. Returns the generated ID.

        Args:
            content: The memory text to store.
            agent_name: Which agent is writing (used for filtering).
            session_id: Conversation thread this came from.
            tags: Optional labels for categorization.
        """
        table = self._table(self._memories_table)
        record: dict[str, Any] = {
            "id":          str(uuid.uuid4()),
            "content":     content,
            "agent_name":  agent_name,
            "session_id":  session_id,
            "created_at":  time.time(),
            "tags":        ",".join(tags or []),
        }
        vector = self._embed(content)
        if vector is not None:
            record["vector"] = vector

        table.add([record])
        logger.debug("Memory saved id=%s agent=%s", record["id"], agent_name)
        return record["id"]

    def search_memories(
        self,
        query: str,
        *,
        limit: int = 5,
        agent_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant memories for a query.

        Uses vector similarity when an embedding model is available,
        otherwise returns the most recent records.

        Args:
            query: Natural-language query to match against stored memories.
            limit: Maximum results to return.
            agent_name: If set, restrict results to this agent's memories.
        """
        table = self._table(self._memories_table)
        where = f"agent_name = '{self._sql_str(agent_name)}'" if agent_name else None
        vector = self._embed(query)

        if vector is not None:
            search = table.search(vector)
            if where:
                search = search.where(where)
            results = search.limit(limit).to_list()
        else:
            # No embeddings — fall back to most-recent scan
            search = table.search()
            if where:
                search = search.where(where)
            df = search.limit(limit * 5).to_arrow().to_pydict()
            rows = [dict(zip(df.keys(), vals)) for vals in zip(*df.values())] if df else []
            rows.sort(key=lambda r: r.get("created_at", 0), reverse=True)
            results = rows[:limit]

        return self._clean(results)

    def list_memories(
        self,
        *,
        agent_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List recent memories, newest first.

        Args:
            agent_name: Filter to a specific agent's memories.
            limit: Maximum records to return.
        """
        table = self._table(self._memories_table)
        df = table.to_pandas()
        if agent_name:
            df = df[df["agent_name"] == agent_name]
        df = df.sort_values("created_at", ascending=False).head(limit)
        keep = [c for c in ("id", "content", "agent_name", "session_id", "created_at", "tags") if c in df.columns]
        return df[keep].to_dict("records")

    # ------------------------------------------------------------------
    # Notes API  (key-value, shared across all agents)
    # ------------------------------------------------------------------

    def save_note(
        self,
        key: str,
        value: str,
        *,
        agent_name: str = "",
        session_id: str = "",
    ) -> None:
        """Save or overwrite a note. All agents share the same key namespace.

        Uses merge_insert so concurrent writes on different keys are safe.

        Args:
            key: Unique identifier for the note.
            value: Content to store.
            agent_name: Agent performing the write (informational).
            session_id: Session that initiated the write.
        """
        table = self._table(self._notes_table)
        record: dict[str, Any] = {
            "key":        key,
            "value":      value,
            "agent_name": agent_name,
            "session_id": session_id,
            "updated_at": time.time(),
        }
        vector = self._embed(f"{key}: {value}")
        if vector is not None:
            record["vector"] = vector

        (
            table.merge_insert("key")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([record])
        )

    def get_note(self, key: str) -> str | None:
        """Get a note's value by exact key. Returns None if not found.

        Args:
            key: The note key to look up.
        """
        table = self._table(self._notes_table)
        try:
            results = (
                table.search()
                .where(f"key = '{self._sql_str(key)}'")
                .limit(1)
                .to_list()
            )
            return results[0]["value"] if results else None
        except Exception:
            # Fallback: pandas scan
            df = table.to_pandas()
            matches = df[df["key"] == key]
            return str(matches["value"].iloc[0]) if not matches.empty else None

    def search_notes(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search notes by semantic similarity or text match.

        Args:
            query: Natural-language query to match notes against.
            limit: Maximum results.
        """
        table = self._table(self._notes_table)
        vector = self._embed(query)

        if vector is not None:
            results = table.search(vector).limit(limit).to_list()
            return self._clean(results)

        # Fallback: substring filter via pandas
        df = table.to_pandas()
        q = query.lower()
        mask = (
            df["key"].str.lower().str.contains(q, na=False)
            | df["value"].str.lower().str.contains(q, na=False)
        )
        return df[mask].head(limit).to_dict("records")

    def list_notes(self, *, prefix: str = "") -> list[dict[str, Any]]:
        """List all notes, optionally filtered by key prefix, newest first.

        Args:
            prefix: Only return notes whose key starts with this string.
        """
        table = self._table(self._notes_table)
        df = table.to_pandas()
        if prefix:
            df = df[df["key"].str.startswith(prefix, na=False)]
        df = df.sort_values("updated_at", ascending=False)
        keep = [c for c in ("key", "value", "agent_name", "updated_at") if c in df.columns]
        return df[keep].to_dict("records")

    def delete_note(self, key: str) -> bool:
        """Delete a note by key. Returns True if deleted, False if not found.

        Args:
            key: The note key to delete.
        """
        table = self._table(self._notes_table)
        if self.get_note(key) is None:
            return False
        try:
            table.delete(f"key = '{self._sql_str(key)}'")
            return True
        except Exception as exc:
            logger.warning("delete_note failed for key=%r: %s", key, exc)
            return False

    def __repr__(self) -> str:
        return (
            f"SharedMemoryStore(path={str(self._data_path)!r}, "
            f"embedding_model={self._embedding_model!r})"
        )
