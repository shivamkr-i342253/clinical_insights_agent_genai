"""
Regulated Trial Summarisation Node.
Combines simulation, regulatory, and analysis outputs into a cohesive summary.
Routes to Critique or Final Report based on review_count vs max_no_of_reviews.
"""

from __future__ import annotations

import json

from chains.chains import build_structured_chain
from graph.state import GraphState, TrialSummaryOutput
from prompts.templates import TRIAL_SUMMARIZATION_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


def regulated_trial_summarization_node(state: GraphState) -> dict:
    """
    Generate the regulated trial summary by combining all upstream analysis.
    Increments the review_count on each pass.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with regulated_trial_summary and incremented review_count.
    """
    logger.info("[Node] ── REGULATED TRIAL SUMMARISATION ─────────────────────")

    clinical_simulation = state.get("clinical_simulation") or {}
    regulatory_summary = state.get("regulatory_summary") or {}
    trial_insights = state.get("trial_insights") or {}
    cohort_comparison = state.get("cohort_comparison") or {}
    critique_feedback = state.get("critique_feedback") or "No prior critique."
    review_count = state.get("review_count", 0)

    chain = build_structured_chain(TRIAL_SUMMARIZATION_PROMPT, TrialSummaryOutput)

    logger.info(
        f"[Node:TrialSummarisation] Invoking LLM | "
        f"review_count={review_count} | "
        f"incorporating_critique={bool(state.get('critique_feedback'))}"
    )

    result: TrialSummaryOutput = chain.invoke(
        {
            "clinical_simulation": json.dumps(clinical_simulation, indent=2),
            "regulatory_summary": json.dumps(regulatory_summary, indent=2),
            "trial_insights": json.dumps(trial_insights, indent=2),
            "cohort_comparison": json.dumps(cohort_comparison, indent=2),
            "critique_feedback": critique_feedback,
        }
    )

    new_review_count = review_count + 1
    logger.info(
        f"[Node:TrialSummarisation] Complete | "
        f"new review_count={new_review_count} | "
        f"exec_summary='{result.executive_summary[:80]}…'"
    )

    return {
        "regulated_trial_summary": result.model_dump(),
        "review_count": new_review_count,
        "messages": state.get("messages", [])
    }
