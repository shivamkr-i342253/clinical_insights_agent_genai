"""
Clinical Simulation Node.
Simulates dosage and compliance changes and predicts patient outcome impact.
"""

from __future__ import annotations

import json

from chains.chains import build_structured_chain
from graph.state import ClinicalSimulationOutput, GraphState
from prompts.templates import CLINICAL_SIMULATION_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


def clinical_simulation_node(state: GraphState) -> dict:
    """
    Simulate clinical scenarios (dosage adjustments, compliance changes)
    and predict their impact on patient outcomes.

    Args:
        state: Current graph state with trial_insights and cohort_comparison.

    Returns:
        Partial state update with clinical_simulation dict.
    """
    logger.info("[Node] ── CLINICAL SIMULATION ───────────────────────────────")

    trial_insights = state.get("trial_insights") or {}
    cohort_comparison = state.get("cohort_comparison") or {}

    chain = build_structured_chain(CLINICAL_SIMULATION_PROMPT, ClinicalSimulationOutput)

    logger.info("[Node:ClinicalSimulation] Invoking LLM …")
    result: ClinicalSimulationOutput = chain.invoke(
        {
            "trial_insights": json.dumps(trial_insights, indent=2),
            "cohort_comparison": json.dumps(cohort_comparison, indent=2),
        }
    )

    logger.info(
        f"[Node:ClinicalSimulation] Complete | "
        f"simulation_summary='{result.simulation_summary[:80]}…'"
    )
    return {"clinical_simulation": result.model_dump()}
