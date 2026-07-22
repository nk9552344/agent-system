"""Expected LanceDB table schemas for the shared memory store.

The external service that manages the database must create these tables
before any agent can use the store. This file documents the schema and
provides a ``bootstrap_tables()`` helper for development environments
where no external service is running.

Expected database path: ``data/lancedb/``

Tables
------

``memories``
    Unstructured observations an agent wants to persist. Supports
    semantic (vector) search via Ollama embeddings.

``notes``
    Structured key-value pairs with exact-key lookup and optional
    semantic search. Agents share a single notes namespace — later
    writes with the same key overwrite earlier ones.
"""

from __future__ import annotations

EMBEDDING_DIM = 768
"""Default embedding dimension (nomic-embed-text / mxbai-embed-large).
Adjust if a different Ollama embedding model is used."""

# ---------------------------------------------------------------------------
# Schema definitions (informational — used by bootstrap_tables)
# ---------------------------------------------------------------------------

MEMORIES_SCHEMA: dict = {
    "table_name": "memories",
    "description": "Agent observations / long-term memories.",
    "columns": {
        "id":           "string   — UUID, primary key",
        "content":      "string   — memory text",
        "agent_name":   "string   — which agent wrote this",
        "session_id":   "string   — conversation thread ID",
        "created_at":   "float64  — unix timestamp",
        "tags":         "string   — comma-separated labels",
        "vector":       f"list<float32>[{EMBEDDING_DIM}]  — optional embedding",
    },
}

NOTES_SCHEMA: dict = {
    "table_name": "notes",
    "description": "Key-value facts shared across all agents.",
    "columns": {
        "key":          "string   — unique note key",
        "value":        "string   — note content",
        "agent_name":   "string   — last writer",
        "session_id":   "string   — session that wrote it",
        "updated_at":   "float64  — unix timestamp",
        "vector":       f"list<float32>[{EMBEDDING_DIM}]  — optional embedding",
    },
}


# ---------------------------------------------------------------------------
# Development helper — NOT called by agents at runtime
# ---------------------------------------------------------------------------

def bootstrap_tables(
    db_path: str = "data/lancedb",
    embedding_dim: int = EMBEDDING_DIM,
    overwrite: bool = False,
) -> None:
    """Create the memories and notes tables in an existing LanceDB directory.

    Call this once during development setup. In production the external
    service creates the tables; this function is only for local dev.

    Args:
        db_path: Path to the LanceDB directory (created if absent).
        embedding_dim: Embedding vector dimension. Must match the Ollama
            embedding model you plan to use (default: 768 for nomic-embed-text).
        overwrite: If True, drop and recreate existing tables.
    """
    import pyarrow as pa
    import lancedb

    db = lancedb.connect(db_path)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        existing = set(db.table_names())

    memories_schema = pa.schema([
        pa.field("id",          pa.string()),
        pa.field("content",     pa.string()),
        pa.field("agent_name",  pa.string()),
        pa.field("session_id",  pa.string()),
        pa.field("created_at",  pa.float64()),
        pa.field("tags",        pa.string()),
        pa.field("vector",      pa.list_(pa.float32(), embedding_dim)),
    ])

    notes_schema = pa.schema([
        pa.field("key",         pa.string()),
        pa.field("value",       pa.string()),
        pa.field("agent_name",  pa.string()),
        pa.field("session_id",  pa.string()),
        pa.field("updated_at",  pa.float64()),
        pa.field("vector",      pa.list_(pa.float32(), embedding_dim)),
    ])

    for name, schema in [("memories", memories_schema), ("notes", notes_schema)]:
        if name in existing:
            if overwrite:
                db.drop_table(name)
            else:
                print(f"[schema] table '{name}' already exists — skipping.")
                continue
        db.create_table(name, schema=schema)
        print(f"[schema] created table '{name}'.")
