"""
Final Report Node.
Assembles and formats the approved trial summary into a complete,
regulatory-ready clinical trial report.
"""

from __future__ import annotations

import json
from datetime import date

from chains.chains import build_text_chain
from graph.state import GraphState
from prompts.templates import FINAL_REPORT_PROMPT
from utils.logger import get_logger
from chains.chains import summarize_conversation

logger = get_logger(__name__)


def final_report_node(state: GraphState) -> dict:
    """
    Produce the final human-readable clinical trial report.

    Args:
        state: Current graph state with regulated_trial_summary.

    Returns:
        Partial state update with final_report string.
    """
    logger.info("[Node] ── FINAL REPORT ───────────────────────────────────────")

    regulated_trial_summary = state.get("regulated_trial_summary") or {}
    critique_feedback = state.get("critique_feedback") or "No critique – single-pass review."
    review_count = state.get("review_count", 1)

    chain = build_text_chain(FINAL_REPORT_PROMPT)

    logger.info(
        f"[Node:FinalReport] Generating final report | "
        f"total_reviews_completed={review_count}"
    )

    conversation_summary = state.get("conversation_summary", "")

    print("Conversation summary for final report generation:")
    print(conversation_summary)

    report_text: str = chain.invoke(
        {
            "regulated_trial_summary": json.dumps(regulated_trial_summary, indent=2),
            "critique_feedback": critique_feedback,
            "conversation_summary": conversation_summary,
        }
    )

    # Prepend standard header metadata
    header = _build_header(review_count)
    final_report = header + "\n\n" + report_text

    logger.info(
        f"[Node:FinalReport] Report generated | "
        f"length={len(final_report)} chars"
    )
    return {"final_report": final_report}


def _build_header(review_count: int) -> str:
    """Build a standardised report header."""
    today = date.today().isoformat()
    return (
        "=" * 72 + "\n"
        "  CLINICAL TRIAL FINAL REPORT\n"
        "  GenAI-Powered Clinical Insights Agent | Azure OpenAI (GPT-4o)\n"
        "=" + "\n"
        f"  Report Date         : {today}\n"
        f"  Review Cycles       : {review_count}\n"
        f"  Document Status     : FINAL – Regulatory Review Ready\n"
        f"  Regulatory Standard : FDA 21 CFR Part 312 | ICH E6(R2) | ICH E9\n"
        "="
    )
