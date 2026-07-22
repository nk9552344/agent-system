"""Coordinator — manager, router, and merger for multi-agent task execution.

Quick start::

    from coordinator import Coordinator

    # Reads coordinator/config.yml for model roster
    coord = Coordinator(workspace_dir="/tmp/myproject")

    result = coord.run(
        "Build a FastAPI CRUD app with SQLite, Pydantic models, and pytest tests."
    )
    print(result)

    # Merge all agent branches when done
    coord.cleanup()

Multi-project / shared memory::

    from coordinator import Coordinator
    from storage import SharedMemoryStore
    from storage.schema import bootstrap_tables

    bootstrap_tables("data/lancedb")          # once, dev only
    store = SharedMemoryStore("data/lancedb")

    coord1 = Coordinator(memory_store=store, workspace_dir="/tmp/proj1")
    coord2 = Coordinator(memory_store=store, workspace_dir="/tmp/proj2")
    # Both coordinators and all their specialists share the same memory.
"""

from coordinator.config import AgentSpec, CoordinatorConfig, load_config
from coordinator.core import Coordinator
from coordinator.researcher import AutoResearcher

__all__ = ["AgentSpec", "AutoResearcher", "Coordinator", "CoordinatorConfig", "load_config"]
