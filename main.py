"""
CLI Entry Point – Clinical Insights Agent.
Allows running the LangGraph workflow from the command line.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()


def main():
    """Run the Clinical Insights Agent from the command line."""
    from graph.workflow import build_graph
    from utils.logger import get_logger

    logger = get_logger("main")

    # Check for required Azure OpenAI environment variables
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not azure_api_key:
        print("\n[ERROR] AZURE_OPENAI_API_KEY is not set. Add it to your .env file.\n")
        sys.exit(1)
    
    if not azure_endpoint:
        print("\n[ERROR] AZURE_OPENAI_ENDPOINT is not set. Add it to your .env file.\n")
        sys.exit(1)

    # Default query
    user_input = (
        "Analyse the clinical trial dataset. "
        "Identify non-compliance, adverse events, and drug inefficacy. "
        "Compare cohort outcomes, simulate dosage changes, and generate "
        "an FDA-compliant final report."
    )

    data_path = Path(__file__).parent / "resources" / "clinical_trial_data.csv"
    data_instruction = f"Load clinical trial data from the CSV file at: {data_path}"
    full_input = f"{user_input}\n\n{data_instruction}"

    max_reviews = int(os.getenv("MAX_REVIEWS", "2"))
    thread_id = str(uuid.uuid4())

    logger.info(f"Starting Clinical Insights Agent | thread_id={thread_id}")
    logger.info(f"Max review cycles: {max_reviews}")

    graph = build_graph()
    graph_input = {
        "user_input": full_input,
        "max_no_of_reviews": max_reviews,
        "messages": [HumanMessage(content=full_input)],
        "review_count": 0,
    }
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 60)
    print("  Clinical Insights Agent – Starting Workflow")
    print("=" * 60 + "\n")

    for event in graph.stream(graph_input, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n✅ [{node_name.upper()}] completed")
            if node_output:
                keys = [k for k in node_output if node_output[k] is not None]
                print(f"   Updated keys: {keys}")

    # Retrieve final state
    final_state = graph.get_state(config)
    final_report = final_state.values.get("final_report", "")

    if final_report:
        print("\n" + "=" * 60)
        print("  FINAL REPORT")
        print("=" * 60)
        print(final_report)

        report_path = Path("clinical_trial_final_report.txt")
        report_path.write_text(final_report, encoding="utf-8")
        print(f"\n[INFO] Report saved to: {report_path.absolute()}")
    else:
        print("\n[WARNING] No final report generated.")


if __name__ == "__main__":
    main()
