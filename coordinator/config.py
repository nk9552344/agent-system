"""Load and validate coordinator/config.yml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AgentSpec:
    """One specialist agent entry from config.yml."""

    name: str
    model: str
    description: str
    expertise: list[str] = field(default_factory=list)
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_ctx: int = 8192

    def expertise_line(self) -> str:
        """Single-line expertise summary for prompt injection."""
        return ", ".join(self.expertise)


@dataclass
class CoordinatorSpec:
    """Coordinator LLM settings from config.yml."""

    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_ctx: int = 16384


@dataclass
class CoordinatorConfig:
    """Full parsed config."""

    coordinator: CoordinatorSpec
    agents: list[AgentSpec]

    def agent_by_name(self, name: str) -> AgentSpec | None:
        return next((a for a in self.agents if a.name == name), None)


def load_config(path: str | Path = "coordinator/config.yml") -> CoordinatorConfig:  # kept for legacy use
    """Parse config.yml and return a CoordinatorConfig.

    Args:
        path: Path to the YAML config file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required fields are missing.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Coordinator config not found at '{cfg_path}'. "
            "Copy coordinator/config.yml.example and edit it."
        )

    raw: dict = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    coord_raw = raw.get("coordinator", {})
    if not coord_raw.get("model"):
        raise ValueError("config.yml: coordinator.model is required.")

    coordinator = CoordinatorSpec(
        model=coord_raw["model"],
        base_url=coord_raw.get("base_url", "http://localhost:11434"),
        temperature=float(coord_raw.get("temperature", 0.0)),
        num_ctx=int(coord_raw.get("num_ctx", 16384)),
    )

    agents = []
    for entry in raw.get("agents", []):
        if not entry.get("name") or not entry.get("model"):
            raise ValueError(f"config.yml: each agent must have 'name' and 'model'. Got: {entry}")
        agents.append(
            AgentSpec(
                name=entry["name"],
                model=entry["model"],
                description=entry.get("description", ""),
                expertise=entry.get("expertise", []),
                base_url=entry.get("base_url", "http://localhost:11434"),
                temperature=float(entry.get("temperature", 0.0)),
                num_ctx=int(entry.get("num_ctx", 8192)),
            )
        )

    if not agents:
        raise ValueError("config.yml: at least one agent must be defined under 'agents'.")

    return CoordinatorConfig(coordinator=coordinator, agents=agents)


def from_agent_config(researcher_section: dict) -> CoordinatorConfig:
    """Build a CoordinatorConfig from the ``researcher:`` section of agent_config.yml.

    Expected shape::

        researcher:
          coordinator:
            model: qwen2.5-coder:7b
            base_url: http://localhost:11434
            temperature: 0
            context_window: 16384
          specialists:
            - name: coder
              model: qwen2.5-coder:7b
              ...
    """
    coord_raw = researcher_section.get("coordinator", {})
    if not coord_raw.get("model"):
        raise ValueError(
            "agent_config.yml: researcher.coordinator.model is required."
        )

    coordinator = CoordinatorSpec(
        model=coord_raw["model"],
        base_url=coord_raw.get("base_url", "http://localhost:11434"),
        temperature=float(coord_raw.get("temperature", 0.0)),
        # accept both context_window (new key) and num_ctx (old key)
        num_ctx=int(coord_raw.get("context_window", coord_raw.get("num_ctx", 16384))),
    )

    agents: list[AgentSpec] = []
    for entry in researcher_section.get("specialists", []):
        if not entry.get("name") or not entry.get("model"):
            raise ValueError(
                f"agent_config.yml: each specialist needs 'name' and 'model'. Got: {entry}"
            )
        agents.append(
            AgentSpec(
                name=entry["name"],
                model=entry["model"],
                description=entry.get("description", ""),
                expertise=entry.get("expertise", []),
                base_url=entry.get("base_url", "http://localhost:11434"),
                temperature=float(entry.get("temperature", 0.0)),
                num_ctx=int(entry.get("context_window", entry.get("num_ctx", 8192))),
            )
        )

    if not agents:
        raise ValueError(
            "agent_config.yml: researcher.specialists must have at least one entry."
        )

    return CoordinatorConfig(coordinator=coordinator, agents=agents)
