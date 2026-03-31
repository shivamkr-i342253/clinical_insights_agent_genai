"""
Cohort Comparison Subgraph.

Compares trial outcomes between patient cohorts using LLM reasoning
combined with statistical analysis techniques.
"""

from __future__ import annotations

import json

from langgraph.graph import StateGraph, START, END

from chains.chains import build_structured_chain
from graph.state import CohortComparisonOutput, CohortComparisonState
from prompts.templates import COHORT_COMPARISON_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Node: Cohort Comparison Analysis
# ---------------------------------------------------------------------------

def cohort_comparison_analysis_node(state: CohortComparisonState) -> dict:
    """
    Compare outcomes across patient cohorts using statistical and LLM analysis.

    Args:
        state: Cohort Comparison subgraph state.

    Returns:
        Partial state update with cohort_analysis dict.
    """
    logger.info("[SubNode] ── COHORT COMPARISON / ANALYSIS ───────────────────")

    data_stats = state.get("data_stats") or {}
    cleaned_data = state.get("cleaned_data") or "[]"
    critique_feedback = state.get("critique_feedback") or "No prior critique."

    # Build a readable sample
    try:
        records = json.loads(cleaned_data) if isinstance(cleaned_data, str) else cleaned_data
        sample_data = json.dumps(records[:20], indent=2)
    except Exception:
        sample_data = str(cleaned_data)[:3000]

    # Extract cohort list from data stats
    cohort_dist = data_stats.get("cohort_distribution", {})
    cohorts = list(cohort_dist.keys()) if cohort_dist else ["Treatment_A", "Treatment_B", "Control"]

    chain = build_structured_chain(COHORT_COMPARISON_PROMPT, CohortComparisonOutput)

    logger.info("[SubNode:CohortComparison] Invoking LLM …")
    result: CohortComparisonOutput = chain.invoke(
        {
            "data_stats": json.dumps(data_stats, indent=2),
            "sample_data": sample_data,
            "cohorts": ", ".join(cohorts),
            "critique_feedback": critique_feedback,
        }
    )

    logger.info(
        f"[SubNode:CohortComparison] Complete | "
        f"statistical_findings={len(result.statistical_findings)} | "
        f"recommendations={len(result.recommendations)}"
    )
    return {"cohort_analysis": result.model_dump()}


# ---------------------------------------------------------------------------
# Subgraph Builder
# ---------------------------------------------------------------------------

def create_cohort_comparison_subgraph():
    """
    Build, compile, and return the Cohort Comparison subgraph.

    Graph structure:
      START → cohort_comparison_analysis → END

    Returns:
        Compiled LangGraph subgraph (CompiledGraph).
    """
    graph = StateGraph(CohortComparisonState)
    graph.add_node("cohort_comparison_analysis", cohort_comparison_analysis_node)
    graph.add_edge(START, "cohort_comparison_analysis")
    graph.add_edge("cohort_comparison_analysis", END)

    compiled = graph.compile()
    logger.info("[Subgraph] Cohort Comparison compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Main graph adapter
# ---------------------------------------------------------------------------

def cohort_comparison_node(state) -> dict:
    """
    Adapter that runs the Cohort Comparison subgraph and merges
    its output into the main graph state.

    Args:
        state: Main GraphState.

    Returns:
        Partial main state update with cohort_comparison dict.
    """
    logger.info("[Node] ── COHORT COMPARISON (subgraph call) ─────────────────")

    subgraph = create_cohort_comparison_subgraph()
    sub_input: CohortComparisonState = {
        "cleaned_data": state.get("cleaned_data"),
        "data_stats": state.get("data_stats"),
        "cohort_analysis": None,
        "critique_feedback": state.get("critique_feedback"),
    }

    sub_output = subgraph.invoke(sub_input)

    cohort_comparison = sub_output.get("cohort_analysis", {})
    logger.info(
        f"[Node:CohortComparison] Subgraph complete | "
        f"cohort_summary='{str(cohort_comparison.get('cohort_summary', ''))[:80]}…'"
    )
    return {"cohort_comparison": cohort_comparison}
