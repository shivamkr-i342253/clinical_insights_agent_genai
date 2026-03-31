"""
Trial Insights Subgraph.

Contains two sequential nodes:
  1. data_analysis_node  – Analyse vitals, feedback, efficacy, and compliance.
  2. issue_detection_node – Detect non-compliance, adverse events, anomalies, ineffectiveness.

The subgraph is compiled and registered as a single node in the main workflow.
"""

from __future__ import annotations

import json

from langgraph.graph import StateGraph, START, END

from chains.chains import build_structured_chain
from graph.state import (
    DataAnalysisOutput,
    IssueDetectionOutput,
    TrialInsightsState,
)
from prompts.templates import DATA_ANALYSIS_PROMPT, ISSUE_DETECTION_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Node: Data Analysis
# ---------------------------------------------------------------------------

def data_analysis_node(state: TrialInsightsState) -> dict:
    """
    Analyse clinical trial data across vitals, doctor feedback,
    drug efficacy, and compliance dimensions.

    Args:
        state: Trial Insights subgraph state.

    Returns:
        Partial state update with data_analysis dict.
    """
    logger.info("[SubNode] ── TRIAL INSIGHTS / DATA ANALYSIS ─────────────────")

    data_stats = state.get("data_stats") or {}
    cleaned_data = state.get("cleaned_data") or "[]"
    critique_feedback = state.get("critique_feedback") or "No prior critique."

    # Build a readable sample for the prompt
    try:
        records = json.loads(cleaned_data) if isinstance(cleaned_data, str) else cleaned_data
        sample_data = json.dumps(records[:15], indent=2)
    except Exception:
        sample_data = str(cleaned_data)[:3000]

    chain = build_structured_chain(DATA_ANALYSIS_PROMPT, DataAnalysisOutput)

    logger.info("[SubNode:DataAnalysis] Invoking LLM …")
    result: DataAnalysisOutput = chain.invoke(
        {
            "data_stats": json.dumps(data_stats, indent=2),
            "sample_data": sample_data,
            "critique_feedback": critique_feedback,
        }
    )

    logger.info(
        f"[SubNode:DataAnalysis] Complete | "
        f"key_findings={len(result.key_findings)}"
    )
    return {"data_analysis": result.model_dump()}


# ---------------------------------------------------------------------------
# Node: Issue Detection
# ---------------------------------------------------------------------------

def issue_detection_node(state: TrialInsightsState) -> dict:
    """
    Detect non-compliance, adverse events, anomalies, and ineffectiveness
    using the data analysis results.

    Args:
        state: Trial Insights subgraph state.

    Returns:
        Partial state update with issue_detection dict.
    """
    logger.info("[SubNode] ── TRIAL INSIGHTS / ISSUE DETECTION ───────────────")

    data_analysis = state.get("data_analysis") or {}
    data_stats = state.get("data_stats") or {}
    critique_feedback = state.get("critique_feedback") or "No prior critique."

    chain = build_structured_chain(ISSUE_DETECTION_PROMPT, IssueDetectionOutput)

    logger.info("[SubNode:IssueDetection] Invoking LLM …")
    result: IssueDetectionOutput = chain.invoke(
        {
            "data_analysis": json.dumps(data_analysis, indent=2),
            "data_stats": json.dumps(data_stats, indent=2),
            "critique_feedback": critique_feedback,
        }
    )

    logger.info(
        f"[SubNode:IssueDetection] Complete | "
        f"adverse_events={len(result.adverse_events)} | "
        f"non_compliance={len(result.non_compliance_cases)}"
    )
    return {"issue_detection": result.model_dump()}


# ---------------------------------------------------------------------------
# Subgraph Builder
# ---------------------------------------------------------------------------

def create_trial_insights_subgraph():
    """
    Build, compile, and return the Trial Insights subgraph.

    Graph structure:
      START → data_analysis → issue_detection → END

    Returns:
        Compiled LangGraph subgraph (CompiledGraph).
    """
    graph = StateGraph(TrialInsightsState)

    graph.add_node("data_analysis", data_analysis_node)
    graph.add_node("issue_detection", issue_detection_node)

    graph.add_edge(START, "data_analysis")
    graph.add_edge("data_analysis", "issue_detection")
    graph.add_edge("issue_detection", END)

    compiled = graph.compile()
    logger.info("[Subgraph] Trial Insights compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Main graph adapter – wraps the subgraph as a callable node
# ---------------------------------------------------------------------------

def trial_insights_node(state) -> dict:
    """
    Adapter that runs the Trial Insights subgraph and merges
    its output into the main graph state.

    Args:
        state: Main GraphState.

    Returns:
        Partial main state update with trial_insights dict.
    """
    logger.info("[Node] ── TRIAL INSIGHTS (subgraph call) ────────────────────")

    subgraph = create_trial_insights_subgraph()
    sub_input: TrialInsightsState = {
        "cleaned_data": state.get("cleaned_data"),
        "data_stats": state.get("data_stats"),
        "data_analysis": None,
        "issue_detection": None,
        "critique_feedback": state.get("critique_feedback"),
    }

    sub_output = subgraph.invoke(sub_input)

    trial_insights = {
        "data_analysis": sub_output.get("data_analysis", {}),
        "issue_detection": sub_output.get("issue_detection", {}),
        "subgraph_summary": (
            f"Trial Insights completed with "
            f"{len(sub_output.get('issue_detection', {}).get('adverse_events', []))} "
            f"adverse events detected."
        ),
    }

    logger.info(f"[Node:TrialInsights] Subgraph complete | summary='{trial_insights['subgraph_summary']}'")
    return {"trial_insights": trial_insights}
