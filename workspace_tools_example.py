"""
workspace_tools_example.py — template for AutoResearcher user-defined hooks.

Copy this file, implement evaluate() and save() for your project, then set:

    researcher:
      user_tools: path/to/your_workspace_tools.py

in config.yml.

The researcher calls evaluate() after each hypothesis is implemented to get a
score. It calls save() whenever a hypothesis improves the score.
"""


def evaluate(workspace_path: str) -> dict:
    """Evaluate the current implementation and return a score.

    Args:
        workspace_path: Absolute path to the project directory (after specialists
                        have applied the hypothesis).

    Returns:
        A dict with:
            "score"   (float) — the metric being optimised; higher = better.
            "details" (str)   — human-readable explanation shown in the loop output.

        Or just a float if you don't need details.

    Notes:
        - Keep this function fast when possible (it runs every iteration).
        - The score must be comparable across iterations (same units, same scale).
        - Raise an exception to signal a broken workspace — the researcher will
          treat this iteration as a failure and revert.
    """
    # ── Example: fraction of pytest tests that pass ───────────────────────────
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    report_file = Path(tempfile.mktemp(suffix=".json"))
    result = subprocess.run(
        [
            "pytest",
            "--tb=no",
            "-q",
            "--json-report",
            f"--json-report-file={report_file}",
        ],
        cwd=workspace_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    try:
        report = json.loads(report_file.read_text())
        summary = report.get("summary", {})
        passed = summary.get("passed", 0)
        total = max(summary.get("total", 1), 1)
        score = passed / total
        details = f"{passed}/{total} tests passing"
    except Exception:
        # Fallback: use returncode (0 = all pass)
        score = 1.0 if result.returncode == 0 else 0.0
        details = (result.stdout + result.stderr)[-300:]

    return {"score": score, "details": details}


def save(workspace_path: str, hypothesis: str, score: float, iteration: int) -> None:
    """Persist the improved implementation.

    Called once per iteration where the score beats the previous best.

    Args:
        workspace_path: Absolute path to the project directory.
        hypothesis:     Description of what was changed (from the researcher LLM).
        score:          The score that was achieved.
        iteration:      The iteration number (1-based).
    """
    # ── Example: create a git tag for the improved state ─────────────────────
    import subprocess

    tag = f"research/iter-{iteration:04d}-score-{score:.4f}"
    subprocess.run(
        ["git", "tag", "-f", tag, "-m", hypothesis[:200]],
        cwd=workspace_path,
        capture_output=True,
    )
    print(f"  Tagged as {tag}")

    # ── Example: also write a summary line to a results log ──────────────────
    from pathlib import Path

    log = Path(workspace_path) / "research_log.txt"
    with log.open("a") as f:
        f.write(f"iter={iteration:04d}  score={score:.4f}  {hypothesis[:200]}\n")
