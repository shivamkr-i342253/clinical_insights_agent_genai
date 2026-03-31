"""
Critique Node.
Reviews the regulated trial summary and triggers the agentic feedback loop
if the quality score or review count warrants revision.
"""

from __future__ import annotations

import json

from chains.chains import build_structured_chain
from graph.state import CritiqueOutput, GraphState
from prompts.templates import CRITIQUE_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


def critique_node(state: GraphState) -> dict:
    """
    Review the regulated trial summary.
    Produce structured critique feedback that drives the agentic re-analysis loop.

    Args:
        state: Current graph state with regulated_trial_summary.

    Returns:
        Partial state update with critique_feedback string.
    """
    logger.info("[Node] ── CRITIQUE ───────────────────────────────────────────")

    trial_summary = state.get("regulated_trial_summary") or {}
    review_count = state.get("review_count", 1)
    max_reviews = state.get("max_no_of_reviews", 2)

    chain = build_structured_chain(CRITIQUE_PROMPT, CritiqueOutput)

    logger.info(
        f"[Node:Critique] Invoking LLM | "
        f"review_count={review_count} / max={max_reviews}"
    )

    result: CritiqueOutput = chain.invoke(
        {"trial_summary": json.dumps(trial_summary, indent=2)}
    )

    # Serialise critique into a human-readable string for injection into prompts
    feedback_lines = [
        f"Quality Score: {result.overall_quality_score}/10",
        "",
        "Strengths:",
        *[f"  • {s}" for s in result.strengths],
        "",
        "Weaknesses:",
        *[f"  • {w}" for w in result.weaknesses],
        "",
        "Specific Revision Feedback:",
        *[f"  [{i+1}] {f}" for i, f in enumerate(result.specific_feedback)],
        "",
        f"Requires Revision: {'YES' if result.requires_revision else 'NO'}",
        f"Priority Areas: {result.revision_priority}",
    ]
    critique_feedback = "\n".join(feedback_lines)

    logger.info(
        f"[Node:Critique] Complete | "
        f"score={result.overall_quality_score}/10 | "
        f"requires_revision={result.requires_revision}"
    )

    return {"critique_feedback": critique_feedback}
