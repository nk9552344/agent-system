"""LLM-based result judge for the coordinator.

The judge asks the coordinator's own model to evaluate whether a specialist
agent's output satisfactorily completed the assigned subtask. This is a
separate LLM call — not a tool the agent calls on itself.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

_JUDGE_SYSTEM = """You are a strict quality reviewer for an AI coding and research pipeline.

Your job: decide whether an agent's output PASSES or FAILS the task it was given.

Evaluation criteria:
- PASS if the work is complete, correct, and directly addresses the task.
- FAIL if the work is incomplete, off-topic, contains obvious errors, or asks
  the user clarifying questions instead of doing the work.

Respond in this exact format:
VERDICT: PASS
REASON: <one sentence>

or

VERDICT: FAIL
REASON: <one sentence explaining what is missing or wrong>
RETRY_HINT: <one sentence telling the agent specifically what to fix>
"""

_JUDGE_TEMPLATE = """TASK GIVEN TO AGENT:
{task}

AGENT'S OUTPUT:
{result}

Evaluate the output. Be strict — a partial answer is a FAIL."""


class ResultJudge:
    """Uses the coordinator's LLM to evaluate specialist agent outputs.

    Parameters
    ----------
    model_name:
        Ollama model name for the judge (usually same as coordinator).
    base_url:
        Ollama server URL.
    temperature:
        Judge temperature — keep low (0) for consistent verdicts.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ) -> None:
        self._llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
        )

    def evaluate(
        self,
        task: str,
        result: str,
    ) -> tuple[bool, str, str]:
        """Evaluate a specialist agent's output.

        Args:
            task: The exact instruction that was sent to the agent.
            result: The agent's final output.

        Returns:
            Tuple of (passed: bool, reason: str, retry_hint: str).
            retry_hint is empty when passed=True.
        """
        prompt = _JUDGE_TEMPLATE.format(task=task.strip(), result=result.strip()[:4000])
        response = self._llm.invoke(
            [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=prompt)]
        )
        text: str = response.content if isinstance(response.content, str) else str(response.content)

        passed = "VERDICT: PASS" in text.upper()
        reason = _extract_field(text, "REASON")
        retry_hint = "" if passed else _extract_field(text, "RETRY_HINT")

        return passed, reason, retry_hint


def _extract_field(text: str, field: str) -> str:
    """Pull a labelled field value out of the judge's response."""
    for line in text.splitlines():
        if line.upper().startswith(f"{field}:"):
            return line[len(field) + 1:].strip()
    return ""
