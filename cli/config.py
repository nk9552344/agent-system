"""Load and validate agent_config.yml for the CLI."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


CONFIG_FILENAME = "agent_config.yml"


class ConfigError(Exception):
    pass


@dataclass
class ModelConfig:
    name: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    context_window: int = 8192


@dataclass
class AgentConfig:
    name: str = "agent"
    workspace: str = "."
    system_prompt: str | None = None
    require_permission: bool = True


@dataclass
class ResearcherConfig:
    workspace: str = "."
    specialists_config: str = "agent_storage/specialists.yml"
    eval_script: str | None = None


@dataclass
class StorageConfig:
    path: str = "agent_storage/lancedb"
    embedding_model: str = "nomic-embed-text"


@dataclass
class CliConfig:
    model: ModelConfig
    agent: AgentConfig
    researcher: ResearcherConfig
    storage: StorageConfig
    web: dict[str, Any] = field(default_factory=dict)
    debug: bool = False

    @classmethod
    def load(cls, path: str | Path = CONFIG_FILENAME) -> "CliConfig":
        p = Path(path)
        if not p.exists():
            raise ConfigError(
                f"Config file not found: {p}\n"
                f"Run  agentx init  to create it in the current directory."
            )

        raw: dict = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

        m = raw.get("model", {})
        if not m.get("name"):
            raise ConfigError("agent_config.yml: model.name is required.")

        model = ModelConfig(
            name=m["name"],
            base_url=m.get("base_url", "http://localhost:11434"),
            temperature=float(m.get("temperature", 0.0)),
            context_window=int(m.get("context_window", 8192)),
        )

        a = raw.get("agent", {})
        agent = AgentConfig(
            name=a.get("name", "agent"),
            workspace=a.get("workspace", "."),
            system_prompt=a.get("system_prompt") or None,
            require_permission=bool(a.get("require_permission", True)),
        )

        r = raw.get("researcher", {})
        researcher = ResearcherConfig(
            workspace=r.get("workspace", "."),
            specialists_config=r.get("specialists_config", "agent_storage/specialists.yml"),
            eval_script=r.get("eval_script") or None,
        )

        s = raw.get("storage", {})
        storage = StorageConfig(
            path=s.get("path", "agent_storage/lancedb"),
            embedding_model=s.get("embedding_model", "nomic-embed-text"),
        )

        return cls(
            model=model,
            agent=agent,
            researcher=researcher,
            storage=storage,
            web=raw.get("web", {}),
            debug=bool(raw.get("debug", False)),
        )
