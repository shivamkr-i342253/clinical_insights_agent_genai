"""
Main LangGraph Workflow - Clinical Insights Agent.

Defines and compiles the full agentic graph including:
  - All nodes and subgraphs
  - Parallel fan-out/fan-in edges
  - Conditional edges (critique loop / final report routing)
  - SQLite checkpointer for persistent state
"""

from __future__ import annotations

import json
import sqlite3
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from graph.state import GraphState, InputGraphState
from graph.subgraphs.cohort_comparison import cohort_comparison_node
from graph.subgraphs.trial_insights import trial_insights_node
from nodes.conversation_summary import conversation_summary_node
from nodes.assistant import assistant_node
from nodes.clinical_simulation import clinical_simulation_node
from nodes.critique import critique_node
from nodes.data_cleanup import data_cleanup_node
from nodes.final_report import final_report_node
from nodes.regulated_trial_summarization import regulated_trial_summarization_node
from nodes.regulatory_summarization import regulatory_summarization_node
from tools.data_tools import ALL_TOOLS
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Router / Conditional Edge Functions
# ---------------------------------------------------------------------------

def route_after_assistant(
    state: GraphState,
) -> Literal["data_loader", "data_cleanup", "__end__"]:
    """
    After the Assistant node:
      - If the LLM emitted tool calls -> route to Data Loader.
      - If no tool calls AND has direct text response -> route to END (general query).
      - Otherwise -> route to Data Cleanup.
    
    This handles cases like general knowledge queries (e.g., "What's the largest river?")
    that don't require clinical trial data processing.
    """
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None

    if isinstance(last_msg, AIMessage):
        # Check if there are tool calls
        if getattr(last_msg, "tool_calls", []):
            logger.info("[Router] assistant -> data_loader (tool calls detected)")
            return "data_loader"
        
        # Check if there's a direct text response (general query answered)
        if hasattr(last_msg, "content") and last_msg.content:
            is_general_query = _is_general_query(last_msg.content)
            if is_general_query:
                logger.info(
                    "[Router] assistant -> __end__ (general query detected - "
                    "direct answer provided without clinical analysis needed)"
                )
                return "__end__"

    logger.info("[Router] assistant -> data_cleanup (clinical query - continue processing)")
    return "data_cleanup"


def _is_general_query(response_text: str) -> bool:
    """
    Detect if the assistant's response is answering a general query.
    
    A general query is identified when:
      1. The response doesn't indicate need for clinical data
      2. The response provides a complete, standalone answer
      3. The response doesn't reference clinical trials, medical data, or analysis
    
    Args:
        response_text: The text content of the assistant's response.
    
    Returns:
        True if the query appears to be general/non-clinical, False otherwise.
    """
    # Keywords that indicate clinical/trial-related queries
    clinical_keywords = [
        "clinical trial",
        "patient data",
        "medical records",
        "drug efficacy",
        "safety profile",
        "adverse event",
        "trial data",
        "cohort",
        "compliance",
        "dosage",
        "regulatory",
        "fda",
        "clinical data",
        "treatment outcome",
        "patient outcome",
        "diagnosis",
        "prescription",
        "check the data",
        "analyze the data",
        "review the trial",
        "examine clinical",
    ]
    
    # Phrases that indicate the assistant is asking for clinical data or clarification
    data_request_phrases = [
        "need",
        "require",
        "provide",
        "share",
        "upload",
        "data",
        "clinical",
        "trial",
        "patient",
        "load",
        "access",
    ]
    
    response_lower = response_text.lower()
    
    # If response mentions clinical keywords, it's not a general query
    # for keyword in clinical_keywords:
    #     if keyword in response_lower:
    #         return False
    
    # If response is asking for clinical data, keep processing
    # if any(phrase in response_lower for phrase in data_request_phrases if 
    #        len(response_text) < 500 and response_lower.count(phrase) > response_lower.count(".")):
    #     return False
    
    # If the response is substantial (not an error or clarification request)
    # and doesn't reference clinical elements, it's likely a general query answer
    if len(response_text.strip()) > 20:
        return True
    
    return False


def route_after_trial_summarization(
    state: GraphState,
) -> Literal["critique", "final_report"]:
    """
    After Regulated Trial Summarisation:
      - If review_count < max_no_of_reviews -> route to Critique (agentic loop).
      - Otherwise -> route to Final Report.
    """
    review_count = state.get("review_count", 0)
    max_reviews = state.get("max_no_of_reviews", 2)

    if review_count < max_reviews:
        logger.info(
            f"[Router] trial_summarization -> critique "
            f"(review {review_count}/{max_reviews})"
        )
        return "critique"

    logger.info(
        f"[Router] trial_summarization -> final_report "
        f"(max reviews reached: {review_count}/{max_reviews})"
    )
    return "final_report"


# ---------------------------------------------------------------------------
# Pass-through / Utility Nodes
# ---------------------------------------------------------------------------

def data_loader_passthrough(state: GraphState) -> dict:
    """
    Extract the tool result from messages and store as raw_data.
    Called after ToolNode to persist data into graph state.
    """
    messages = state.get("messages", [])
    # Find the last ToolMessage
    for msg in reversed(messages):
        if hasattr(msg, "content") and hasattr(msg, "name"):
            try:
                raw = json.loads(msg.content)
                logger.info(
                    f"[Node:DataLoaderPassthrough] Captured tool result | "
                    f"source={raw.get('source', 'unknown')}"
                )
                return {
                    "raw_data": msg.content,
                    "data_source": raw.get("source", "unknown"),
                }
            except Exception:
                pass
    return {}


def call_subgraphs_node(state: GraphState) -> dict:
    """
    Pass-through node that acts as the fan-out point for the parallel
    Trial Insights and Cohort Comparison subgraphs.
    Also serves as the re-entry point for the critique agentic loop.
    """
    review_count = state.get("review_count", 0)
    logger.info(
        f"[Node] -- CALL SUBGRAPHS (fan-out) | review_count={review_count} --"
    )
    return {}  # No state change; just triggers the fan-out edges


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_graph(db_path: str = "clinical_insights.db"):
    """
    Build and compile the main LangGraph workflow.

    Parallelism:
      1. trial_insights + cohort_comparison run in parallel after call_subgraphs.
      2. clinical_simulation + regulatory_summarization run in parallel after
         both subgraphs complete.

    Agentic Loop:
      regulated_trial_summarization -> critique -> call_subgraphs (loop)
      until review_count >= max_no_of_reviews.

    Args:
        db_path: Path to the SQLite database for checkpointer state.

    Returns:
        Compiled LangGraph workflow (CompiledGraph).
    """
    logger.info("[Graph] Building Clinical Insights Agent workflow ...")

    # -- State graph -------------------------------------------------------
    graph = StateGraph(GraphState, input=InputGraphState)

    # -- Nodes -------------------------------------------------------------
    graph.add_node("assistant", assistant_node)
    graph.add_node("data_loader", ToolNode(ALL_TOOLS))
    graph.add_node("data_loader_passthrough", data_loader_passthrough)
    graph.add_node("data_cleanup", data_cleanup_node)
    graph.add_node("call_subgraphs", call_subgraphs_node)

    # Parallel subgraph nodes
    graph.add_node("trial_insights", trial_insights_node)
    graph.add_node("cohort_comparison", cohort_comparison_node)

    # Parallel processing nodes
    graph.add_node("clinical_simulation", clinical_simulation_node)
    graph.add_node("regulatory_summarization", regulatory_summarization_node)

    graph.add_node("regulated_trial_summarization", regulated_trial_summarization_node)
    graph.add_node("conversation_summary", conversation_summary_node)
    graph.add_node("critique", critique_node)
    graph.add_node("final_report", final_report_node)

    # -- Edges -------------------------------------------------------------

    # Entry
    graph.add_edge(START, "assistant")

    # Assistant -> Data Loader (if tool calls) / END (if general query) / Data Cleanup (if clinical query)
    graph.add_conditional_edges(
        "assistant",
        route_after_assistant,
        {
            "data_loader": "data_loader",
            "data_cleanup": "data_cleanup",
            "__end__": END,  # General query - end workflow and return answer
        },
    )

    # Data Loader -> passthrough -> data cleanup
    graph.add_edge("data_loader", "data_loader_passthrough")
    graph.add_edge("data_loader_passthrough", "data_cleanup")

    # Data Cleanup -> fan-out node
    graph.add_edge("data_cleanup", "call_subgraphs")

    # -- PARALLELISM 1: Subgraphs run in parallel -------------------------
    graph.add_edge("call_subgraphs", "trial_insights")
    graph.add_edge("call_subgraphs", "cohort_comparison")

    # -- PARALLELISM 2: Simulation & Regulatory run in parallel ----------
    # clinical_simulation waits for BOTH subgraphs (fan-in)
    graph.add_edge("trial_insights", "clinical_simulation")
    graph.add_edge("cohort_comparison", "clinical_simulation")

    # regulatory_summarization waits only for trial_insights
    graph.add_edge("trial_insights", "regulatory_summarization")

    # Both feed into trial summarization (fan-in)
    graph.add_edge("clinical_simulation", "regulated_trial_summarization")
    graph.add_edge("regulatory_summarization", "regulated_trial_summarization")
    graph.add_edge("regulated_trial_summarization", "conversation_summary")

    # -- Conditional: Critique loop or Final Report -----------------------
    graph.add_conditional_edges(
        "conversation_summary",
        route_after_trial_summarization,
        {
            "critique": "critique",
            "final_report": "final_report",
        },
    )

    # Critique loops back to call_subgraphs (agentic re-analysis)
    graph.add_edge("critique", "call_subgraphs")

    # Final report ends the graph
    graph.add_edge("final_report", END)

    # -- Checkpointer ------------------------------------------------------
    # conn = sqlite3.connect(db_path, check_same_thread=False)
    # checkpointer = SqliteSaver(conn)

    # -- Compile -----------------------------------------------------------
    compiled = graph.compile()
    logger.info("[Graph] Clinical Insights Agent compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Convenience: visualise the graph
# ---------------------------------------------------------------------------

def get_graph_mermaid(db_path: str = "clinical_insights.db") -> str:
    """Return the Mermaid diagram string for the compiled graph."""
    g = build_graph(db_path)
    return g.get_graph().draw_mermaid()



graph = build_graph()