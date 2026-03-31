"""
Graph State Schemas for the Clinical Insights Agent.
Defines input, output, and intermediate state schemas used across the LangGraph workflow.
"""

from __future__ import annotations
from typing import Annotated, Any, Optional, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Input / Output Schemas (Pydantic) – used for structured LLM output
# ---------------------------------------------------------------------------

class DataAnalysisOutput(BaseModel):
    """Structured output from the Data Analysis node inside Trial Insights subgraph."""
    vitals_analysis: str = Field(description="Analysis of patient vitals (BP, HR, temperature)")
    doctor_feedback_summary: str = Field(description="Synthesised summary of doctor notes and feedback")
    drug_efficacy_assessment: str = Field(description="Assessment of drug efficacy measurements")
    compliance_overview: str = Field(description="Overview of patient compliance logs")
    key_findings: List[str] = Field(description="Ordered list of key clinical findings")


class IssueDetectionOutput(BaseModel):
    """Structured output from the Issue Detection node inside Trial Insights subgraph."""
    non_compliance_cases: List[str] = Field(description="Detected patient non-compliance cases")
    adverse_events: List[str] = Field(description="Detected adverse events or side effects")
    anomalies: List[str] = Field(description="Statistical or clinical anomalies")
    ineffectiveness_indicators: List[str] = Field(description="Indicators of drug ineffectiveness")
    severity_assessment: str = Field(description="Overall severity assessment across all issues")


class TrialInsightsOutput(BaseModel):
    """Combined output from the Trial Insights subgraph."""
    data_analysis: DataAnalysisOutput
    issue_detection: IssueDetectionOutput
    subgraph_summary: str = Field(description="High-level summary of trial insights")


class CohortComparisonOutput(BaseModel):
    """Structured output from the Cohort Comparison subgraph."""
    cohort_summary: str = Field(description="Overview of compared cohorts")
    statistical_findings: List[str] = Field(description="Key statistical findings with p-values")
    outcome_differences: str = Field(description="Key differences in patient outcomes between cohorts")
    recommendations: List[str] = Field(description="Data-driven recommendations from cohort analysis")


class ClinicalSimulationOutput(BaseModel):
    """Structured output from the Clinical Simulation node."""
    dosage_adjustment_impact: str = Field(description="Predicted impact of dosage adjustments")
    compliance_change_impact: str = Field(description="Predicted impact of compliance improvements")
    projected_outcomes: str = Field(description="Projected patient outcomes under simulated scenarios")
    risk_assessment: str = Field(description="Risk assessment for simulated scenarios")
    simulation_summary: str = Field(description="Concise simulation summary")


class RegulatorySummaryOutput(BaseModel):
    """Structured output from the Regulatory Summarisation node."""
    fda_compliance_status: str = Field(description="Current FDA compliance status")
    regulatory_requirements: List[str] = Field(description="Applicable regulatory requirements met")
    compliance_gaps: List[str] = Field(description="Identified compliance gaps requiring action")
    regulatory_summary: str = Field(description="Full regulatory-compliant summary")


class TrialSummaryOutput(BaseModel):
    """Structured output from the Regulated Trial Summarisation node."""
    executive_summary: str = Field(description="Executive summary of the clinical trial")
    methodology_review: str = Field(description="Methodology review")
    results_summary: str = Field(description="Summary of trial results")
    safety_profile: str = Field(description="Drug safety and tolerability profile")
    regulatory_alignment: str = Field(description="Regulatory alignment statement")
    recommendations: List[str] = Field(description="Final recommendations")


class CritiqueOutput(BaseModel):
    """Structured output from the Critique node."""
    overall_quality_score: int = Field(description="Overall quality score from 1-10", ge=1, le=10)
    strengths: List[str] = Field(description="Identified strengths of the trial summary")
    weaknesses: List[str] = Field(description="Identified weaknesses or gaps")
    specific_feedback: List[str] = Field(description="Specific actionable revision feedback")
    requires_revision: bool = Field(description="Whether the summary requires further revision")
    revision_priority: str = Field(description="Priority areas for revision if required")


# ---------------------------------------------------------------------------
# TypedDict State Schemas – used as LangGraph node states
# ---------------------------------------------------------------------------

class InputGraphState(TypedDict):
    """
    Schema supplied when invoking (initialising) the graph.
    Controls user intent and review cycle depth.
    """
    user_input: str
    max_no_of_reviews: int
    messages: Annotated[list, add_messages]


class GraphState(TypedDict):
    """
    Full mutable state that flows through the entire LangGraph workflow.
    Each node reads from and writes to relevant keys only.
    """
    # ── Entry fields ────────────────────────────────────────────────────────
    user_input: str
    max_no_of_reviews: int
    messages: Annotated[list, add_messages]

    # ── Data pipeline fields ─────────────────────────────────────────────────
    raw_data: Optional[Any]           # Raw loaded data (JSON string from tools)
    cleaned_data: Optional[Any]       # Cleaned/transformed DataFrame as JSON
    data_source: Optional[str]        # Source identifier (filesystem/api/db)
    data_stats: Optional[dict]        # Descriptive stats of cleaned dataset

    # ── Analysis fields ───────────────────────────────────────────────────────
    trial_insights: Optional[dict]        # Output of Trial Insights subgraph
    cohort_comparison: Optional[dict]     # Output of Cohort Comparison subgraph

    # ── Simulation & regulation fields ───────────────────────────────────────
    clinical_simulation: Optional[dict]   # Output of Clinical Simulation node
    regulatory_summary: Optional[dict]    # Output of Regulatory Summarisation node

    # ── Report & review fields ────────────────────────────────────────────────
    regulated_trial_summary: Optional[dict]  # Output of Regulated Trial Summarisation
    critique_feedback: Optional[str]         # Consolidated critique feedback
    conversation_summary: Optional[str]     # Summary of the conversation history for final report context
    final_report: Optional[str]             # Final human-readable report

    # ── Control fields ────────────────────────────────────────────────────────
    review_count: int
    error: Optional[str]


# ---------------------------------------------------------------------------
# Sub-graph State Schemas
# ---------------------------------------------------------------------------

class TrialInsightsState(TypedDict):
    """State schema for the Trial Insights subgraph."""
    cleaned_data: Optional[Any]
    data_stats: Optional[dict]
    data_analysis: Optional[dict]
    issue_detection: Optional[dict]
    critique_feedback: Optional[str]


class CohortComparisonState(TypedDict):
    """State schema for the Cohort Comparison subgraph."""
    cleaned_data: Optional[Any]
    data_stats: Optional[dict]
    cohort_analysis: Optional[dict]
    critique_feedback: Optional[str]
