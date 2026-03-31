"""
Assistant Node – Entry point of the LangGraph workflow.
Parses user input, calls the LLM with bound tools, and decides next steps.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from chains.chains import get_llm_with_tools
from graph.state import GraphState
from prompts.templates import ASSISTANT_SYSTEM
from tools.data_tools import ALL_TOOLS
from utils.logger import get_logger

logger = get_logger(__name__)


def assistant_node(state: GraphState) -> dict:
    """
    Assistant node handler.

    Reads the current conversation messages, invokes the LLM with available
    data-loading tools, and returns updated messages.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with new messages.
    """
    logger.info("[Node] ── ASSISTANT ──────────────────────────────────────────")

    messages = state.get("messages", [])
    user_input = state.get("user_input", "")

    # Seed conversation if no messages yet
    if not messages:
        logger.info(f"[Node:Assistant] Initialising conversation | user_input='{user_input[:80]}...'")
        messages = [
            SystemMessage(content=ASSISTANT_SYSTEM),
            HumanMessage(content=user_input),
        ]

    # LLM with tools bound
    llm = get_llm_with_tools(ALL_TOOLS)

    logger.info("[Node:Assistant] Invoking LLM …")
    response = llm.invoke(messages)
    
    print("LLM response:", response)

    logger.info(
        f"[Node:Assistant] LLM response received | "
        f"has_tool_calls={bool(getattr(response, 'tool_calls', []))}"
    )

    if getattr(response, "tool_calls", []):
        for tc in response.tool_calls:
            logger.info(f"[Node:Assistant] Tool call → {tc['name']} | args={tc['args']}")

    return {"messages": [response]}
