"""
Regulatory Summarisation Node.
Produces regulatory-compliant FDA-aligned summaries from trial insights.
"""

from __future__ import annotations

import json

from chains.chains import build_structured_chain
from graph.state import GraphState, RegulatorySummaryOutput
from prompts.templates import REGULATORY_SUMMARY_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


def regulatory_summarization_node(state: GraphState) -> dict:
    """
    Generate a regulatory-compliant summary aligned with FDA expectations
    based on the Trial Insights output.

    Args:
        state: Current graph state with trial_insights.

    Returns:
        Partial state update with regulatory_summary dict.
    """
    logger.info("[Node] ── REGULATORY SUMMARISATION ──────────────────────────")

    trial_insights = state.get("trial_insights") or {}

    chain = build_structured_chain(REGULATORY_SUMMARY_PROMPT, RegulatorySummaryOutput)

    logger.info("[Node:RegulatorySummarisation] Invoking LLM …")
    result: RegulatorySummaryOutput = chain.invoke(
        {"trial_insights": json.dumps(trial_insights, indent=2)}
    )

    logger.info(
        f"[Node:RegulatorySummarisation] Complete | "
        f"fda_status='{result.fda_compliance_status}' | "
        f"gaps={len(result.compliance_gaps)}"
    )
    return {"regulatory_summary": result.model_dump()}
