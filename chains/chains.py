"""
LangChain Chains used across nodes in the Clinical Insights Agent.
Centralises LLM configuration and chain construction to avoid duplication.
"""

from __future__ import annotations

import os
from typing import Type, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from graph.state import GraphState


from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0.0) -> AzureChatOpenAI:
    """
    Instantiate and return an AzureChatOpenAI LLM.

    Args:
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        Configured AzureChatOpenAI instance.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ai-proxy.lab.epam.com")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-2024-07-18")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    logger.debug(
        f"[LLM] Instantiating AzureChatOpenAI | endpoint={azure_endpoint} | deployment={azure_deployment} | temp={temperature}"
    )

    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_key=azure_api_key,
        api_version=azure_api_version,
        temperature=temperature
    )


# ---------------------------------------------------------------------------
# Chain Builders
# ---------------------------------------------------------------------------

def build_structured_chain(prompt: PromptTemplate, output_schema: Type[BaseModel]):
    """
    Build a LangChain chain that:
      prompt → LLM (with structured JSON output) → Pydantic model instance.

    Args:
        prompt:        LangChain PromptTemplate.
        output_schema: Pydantic model class for structured output parsing.

    Returns:
        Runnable chain producing an instance of output_schema.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(output_schema)
    chain = prompt | structured_llm
    logger.debug(f"[Chain] Built structured chain → {output_schema.__name__}")
    return chain


def build_text_chain(prompt: PromptTemplate) -> object:
    """
    Build a LangChain chain that returns plain text output.

    Args:
        prompt: LangChain PromptTemplate.

    Returns:
        Runnable chain producing a plain string.
    """
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    logger.debug("[Chain] Built text chain")
    return chain


def get_llm_with_tools(tools: list) -> AzureChatOpenAI:
    """
    Return an LLM instance with tools bound (used by the Assistant node).

    Args:
        tools: List of LangChain tool objects.

    Returns:
        LLM with tools bound.
    """

    # This will look into the 'content' string and turn it into a tool call object
    parser = JsonOutputToolsParser()

    llm = get_llm()
    return llm.bind_tools(tools)

def summarize_conversation(state: GraphState) -> str:
   
    summary_message = """
    Create a summary of the conversation above. 
    Only include the conversation details and not the tool call response. 
    Exclude the clinical trial dataset from the summary.
    Take into account important details as mentioned in the conversation.
    """
 
    # Add prompt to our history
    llm = get_llm()
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    return response.content
