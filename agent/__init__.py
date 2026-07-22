"""OllamaDeepAgent package.

Quick start::

    from agent import OllamaDeepAgent, AgentConfig

    agent = OllamaDeepAgent(
        model_name="qwen2.5-coder:7b",
        system_prompt="You are an expert Python developer.",
        workspace_dir=".",
    )
    print(agent.run("List all Python files in this project."))

Multiple independent instances::

    researcher = OllamaDeepAgent("llama3.2", name="researcher", require_permission=False)
    coder = OllamaDeepAgent("qwen2.5-coder:7b", name="coder", workspace_dir="/tmp/proj")

From a config object::

    cfg = AgentConfig(model_name="mistral", workspace_dir="/tmp/myapp", debug=True)
    agent = OllamaDeepAgent(**{k: v for k, v in vars(cfg).items() if v is not None})
"""

from agent.config import AgentConfig
from agent.core import OllamaDeepAgent

__all__ = ["AgentConfig", "OllamaDeepAgent"]
