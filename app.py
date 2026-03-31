"""
Streamlit UI – Clinical Insights Agent.
Provides a professional web interface for running the LangGraph agentic workflow.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Session state init (MUST be before any widgets) ─────────────────────────────

if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "results" not in st.session_state:
    st.session_state.results = {}
if "assistant_response" not in st.session_state:
    st.session_state.assistant_response = None
if "completed_nodes" not in st.session_state:
    st.session_state.completed_nodes = set()
if "current_node" not in st.session_state:
    st.session_state.current_node = None
if "run_complete" not in st.session_state:
    st.session_state.run_complete = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Insights Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1a3a5c; margin-bottom: 0; }
    .sub-title  { font-size: 1rem;   color: #555;      margin-top: 0;    }
    .node-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.82rem; font-weight: 600; margin: 3px 2px;
    }
    .badge-done    { background: #d4edda; color: #155724; }
    .badge-running { background: #cce5ff; color: #004085; }
    .badge-pending { background: #f8f9fa; color: #6c757d; }
    .stExpander > summary { font-weight: 600; }
    .report-box {
        background: #f7f9fc; border: 1px solid #dee2e6;
        border-radius: 8px; padding: 1.5rem;
        font-family: 'Courier New', monospace; font-size: 0.85rem;
        white-space: pre-wrap; max-height: 600px; overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

NODE_ORDER = [
    "assistant",
    "data_loader",
    "data_loader_passthrough",
    "data_cleanup",
    "call_subgraphs",
    "trial_insights",
    "cohort_comparison",
    "clinical_simulation",
    "regulatory_summarization",
    "regulated_trial_summarization",
    "conversation_summary",
    "critique",
    "final_report",
]

NODE_LABELS = {
    "assistant":                       "🤖 Assistant",
    "data_loader":                     "📥 Data Loader",
    "data_loader_passthrough":         "🔗 Data Passthrough",
    "data_cleanup":                    "🧹 Data Cleanup",
    "call_subgraphs":                  "🔀 Call Subgraphs",
    "trial_insights":                  "🔬 Trial Insights",
    "cohort_comparison":               "📊 Cohort Comparison",
    "clinical_simulation":             "⚗️ Clinical Simulation",
    "regulatory_summarization":        "📋 Regulatory Summarisation",
    "regulated_trial_summarization":   "📝 Regulated Trial Summary",
    "conversation_summary":            "💬 Conversation Summary",
    "critique":                        "🔍 Critique",
    "final_report":                    "📄 Final Report",
}

SECTION_MAP = {
    "data_analysis":          "Data Analysis",
    "issue_detection":        "Issue Detection",
    "cohort_comparison":      "Cohort Comparison",
    "clinical_simulation":    "Clinical Simulation",
    "regulatory_summary":     "Regulatory Summary",
    "regulated_trial_summary":"Trial Summary",
    "conversation_summary":   "Conversation Summary",
    "critique_feedback":      "Critique Feedback",
    "final_report":           "Final Report",
}

RESOURCES_PATH = Path(__file__).parent / "resources" / "clinical_trial_data.csv"


def _badge(label: str, status: str) -> str:
    cls = {"done": "badge-done", "running": "badge-running", "pending": "badge-pending"}.get(
        status, "badge-pending"
    )
    return f'<span class="node-badge {cls}">{label}</span>'


def _render_progress(completed: set[str], current: str | None) -> None:
    html = ""
    for node in NODE_ORDER:
        label = NODE_LABELS.get(node, node)
        if node in completed:
            html += _badge(label, "done")
        elif node == current:
            html += _badge(f"⏳ {label}", "running")
        else:
            html += _badge(label, "pending")
    st.markdown(html, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/USDA_Biopesticides_Logo.svg/240px-USDA_Biopesticides_Logo.svg.png", width=60)
    st.markdown("## ⚙️ Configuration")

    st.markdown("### 🔑 Azure OpenAI Settings")
    
    azure_endpoint = st.text_input(
        "Azure Endpoint",
        value=os.getenv("AZURE_OPENAI_ENDPOINT", "https://ai-proxy.lab.epam.com"),
        help="Azure OpenAI endpoint URL",
    )
    if azure_endpoint:
        os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint

    azure_deployment = st.text_input(
        "Azure Deployment",
        value=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-2024-07-18"),
        help="Azure OpenAI deployment name",
    )
    if azure_deployment:
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = azure_deployment

    azure_api_key = st.text_input(
        "Azure API Key",
        value=os.getenv("AZURE_OPENAI_API_KEY", ""),
        type="password",
        help="Azure OpenAI API key (kept secret)",
    )
    if azure_api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key

    azure_api_version = st.text_input(
        "API Version",
        value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        help="Azure OpenAI API version (must be 2024-08-01-preview or later for structured outputs)",
    )
    if azure_api_version:
        os.environ["AZURE_OPENAI_API_VERSION"] = azure_api_version

    st.divider()
    st.markdown("### 📁 Data Source")
    data_source = st.selectbox(
        "Source type",
        ["General Query", "Filesystem (CSV)", "API Endpoint", "Database"],
    )

    if data_source == "General Query":
        data_instruction = ""

    elif data_source == "Filesystem (CSV)":
        file_path = st.text_input(
            "CSV File Path",
            value=str(RESOURCES_PATH),
            help="Absolute path to your clinical trial CSV file.",
        )
        data_instruction = f"Load clinical trial data from the CSV file at: {file_path}"

    elif data_source == "API Endpoint":
        api_url = st.text_input("API Endpoint URL", value="https://clinicaltrial.example.com/api/data")
        data_instruction = f"Load clinical trial data from the REST API at: {api_url}"

    else:
        db_conn = st.text_input("Connection String", value="sqlite:///clinical_trial.db")
        db_query = st.text_area("SQL Query", value="SELECT * FROM clinical_trials LIMIT 500")
        data_instruction = f"Load clinical trial data from the database using query: {db_query}"

    st.divider()
    st.markdown("### 🔄 Review Settings")
    max_reviews = st.slider(
        "Max Review Cycles",
        min_value=1, max_value=5, value=2,
        help="Number of Critique → Re-analysis cycles before generating the final report.",
    )

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "GenAI-Powered Clinical Insights Agent\n"
        "Built with LangGraph + Azure OpenAI (GPT-4o)\n"
        "Designed for FDA-aligned clinical trial analysis."
    )

# ── Main Layout ────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🧬 Clinical Insights Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">GenAI-powered agentic pipeline for pharmaceutical clinical trial analysis</p>',
    unsafe_allow_html=True,
)
st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    user_query = st.text_area(
        "Clinical Analysis Request",
        value=st.session_state.user_query,
        height=120,
        key="user_query_input",
        placeholder="e.g., Analyze the efficacy of the drug across different age groups...",
        help="Describe what you want the agent to analyse or investigate.",
    )

with col_right:
    st.markdown("**Workflow Summary**")
    st.markdown(
        "1. 🤖 Assistant parses your request\n"
        "2. 📥 Data Loader fetches trial data\n"
        "3. 🧹 Data Cleanup cleanses dataset\n"
        "4. 🔀 Parallel subgraphs (Trial Insights + Cohort Comparison)\n"
        "5. ⚗️ + 📋 Parallel: Simulation & Regulatory Summary\n"
        "6. 📝 Trial Summary generation\n"
        "7. 💬 Conversation Summary\n"
        "8. 🔍 Critique & agentic loop\n"
        "9. 📄 Final report\n"
    )

run_button = st.button("🚀 Run Clinical Analysis", type="primary", use_container_width=True)

# ── Run Graph ──────────────────────────────────────────────────────────────────

if run_button:
    if not user_query or user_query.strip() == "":
        st.error("⚠️  Please enter a Clinical Analysis Request before running.")
        st.stop()

    if not os.getenv("AZURE_OPENAI_API_KEY"):
        st.error("⚠️  Please set the AZURE_OPENAI_API_KEY environment variable or enter it in the sidebar.")
        st.stop()

    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        st.error("⚠️  Please set the AZURE_OPENAI_ENDPOINT environment variable or enter it in the sidebar.")
        st.stop()

    # Update session state with the current query
    st.session_state.user_query = user_query

    st.session_state.results = {}
    st.session_state.assistant_response = None
    st.session_state.completed_nodes = set()
    st.session_state.current_node = None
    st.session_state.run_complete = False

    # Construct full query with data instruction only if provided
    full_query = user_query
    if data_instruction.strip():
        full_query = f"{user_query}\n\n{data_instruction}"

    st.divider()
    st.markdown("### 🔄 Workflow Progress")
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    results_container = st.container()

    try:
        from graph.workflow import build_graph
        from langchain_core.messages import HumanMessage

        thread_id = str(uuid.uuid4())
        graph = build_graph()
        print("Full query passed to graph:", full_query)
        graph_input = {
            "user_input": full_query,
            "max_no_of_reviews": max_reviews,
            "messages": [HumanMessage(content=full_query)],
            "review_count": 0,
        }
        config = {"configurable": {"thread_id": thread_id}}

        with status_placeholder:
            st.info("🚀 Starting workflow …")

        for event in graph.stream(graph_input, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                st.session_state.current_node = node_name

                with progress_placeholder:
                    _render_progress(st.session_state.completed_nodes, node_name)

                with status_placeholder:
                    st.info(f"⏳ Running: **{NODE_LABELS.get(node_name, node_name)}**")

                # Store relevant outputs
                if node_output:
                    # Capture assistant response from messages
                    if "messages" in node_output and node_name == "assistant":
                        messages = node_output["messages"]
                        if messages:
                            # Get the last message (should be the assistant's response)
                            last_msg = messages[-1] if isinstance(messages, list) else messages
                            if hasattr(last_msg, "content"):
                                st.session_state.assistant_response = last_msg.content
                    
                    for key in SECTION_MAP:
                        if key in node_output:
                            st.session_state.results[key] = node_output[key]
                    if "data_stats" in node_output:
                        st.session_state.results["data_stats"] = node_output["data_stats"]

                time.sleep(0.3)
                st.session_state.completed_nodes.add(node_name)

        st.session_state.run_complete = True
        st.session_state.current_node = None

        with progress_placeholder:
            _render_progress(st.session_state.completed_nodes, None)
        with status_placeholder:
            st.success("✅ Workflow completed successfully!")

    except Exception as exc:
        with status_placeholder:
            st.error(f"❌ Workflow error: {exc}")
        st.exception(exc)

# ── Results Display ────────────────────────────────────────────────────────────

if st.session_state.results or st.session_state.assistant_response:
    st.divider()
    st.markdown("### 📊 Analysis Results")

    results = st.session_state.results

    # ── Assistant Response for General Queries ────────────────────────────────
    if st.session_state.assistant_response and not results:
        st.markdown("#### 💭 Assistant Response")
        st.info(st.session_state.assistant_response)

    # ── Data Stats ────────────────────────────────────────────────────────────
    if "data_stats" in results:
        with st.expander("🧹 Dataset Statistics", expanded=False):
            stats = results["data_stats"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", stats.get("row_count", "—"))
            m2.metric("Columns", stats.get("column_count", "—"))
            m3.metric("Withdrawals", stats.get("withdrawal_count", "—"))

            if "cohort_distribution" in stats:
                m4.metric("Cohorts", len(stats["cohort_distribution"]))

            col_a, col_b = st.columns(2)
            if "cohort_distribution" in stats:
                col_a.markdown("**Cohort Distribution**")
                col_a.json(stats["cohort_distribution"])
            if "response_distribution" in stats:
                col_b.markdown("**Response Distribution**")
                col_b.json(stats["response_distribution"])

    # ── Trial Insights ────────────────────────────────────────────────────────
    if "data_analysis" in results:
        with st.expander("🔬 Trial Insights – Data Analysis", expanded=True):
            da = results["data_analysis"]
            st.markdown("**Vitals Analysis**")
            st.write(da.get("vitals_analysis", "—"))
            st.markdown("**Drug Efficacy Assessment**")
            st.write(da.get("drug_efficacy_assessment", "—"))
            st.markdown("**Compliance Overview**")
            st.write(da.get("compliance_overview", "—"))
            st.markdown("**Key Findings**")
            for f in da.get("key_findings", []):
                st.markdown(f"- {f}")

    if "issue_detection" in results:
        with st.expander("⚠️ Trial Insights – Issue Detection", expanded=True):
            iss = results["issue_detection"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Non-Compliance Cases**")
                for item in iss.get("non_compliance_cases", []):
                    st.markdown(f"- {item}")
                st.markdown("**Adverse Events**")
                for item in iss.get("adverse_events", []):
                    st.markdown(f"- {item}")
            with c2:
                st.markdown("**Anomalies**")
                for item in iss.get("anomalies", []):
                    st.markdown(f"- {item}")
                st.markdown("**Ineffectiveness Indicators**")
                for item in iss.get("ineffectiveness_indicators", []):
                    st.markdown(f"- {item}")
            st.markdown(f"**Severity Assessment:** {iss.get('severity_assessment', '—')}")

    # ── Cohort Comparison ─────────────────────────────────────────────────────
    if "cohort_comparison" in results:
        with st.expander("📊 Cohort Comparison", expanded=False):
            cc = results["cohort_comparison"]
            st.write(cc.get("cohort_summary", "—"))
            st.markdown("**Statistical Findings**")
            for f in cc.get("statistical_findings", []):
                st.markdown(f"- {f}")
            st.markdown("**Recommendations**")
            for r in cc.get("recommendations", []):
                st.markdown(f"- {r}")

    # ── Clinical Simulation ────────────────────────────────────────────────────
    if "clinical_simulation" in results:
        with st.expander("⚗️ Clinical Simulation", expanded=False):
            sim = results["clinical_simulation"]
            st.markdown("**Dosage Adjustment Impact**")
            st.write(sim.get("dosage_adjustment_impact", "—"))
            st.markdown("**Compliance Change Impact**")
            st.write(sim.get("compliance_change_impact", "—"))
            st.markdown("**Projected Outcomes**")
            st.write(sim.get("projected_outcomes", "—"))
            st.markdown("**Risk Assessment**")
            st.write(sim.get("risk_assessment", "—"))

    # ── Regulatory Summary ─────────────────────────────────────────────────────
    if "regulatory_summary" in results:
        with st.expander("📋 Regulatory Summary", expanded=False):
            reg = results["regulatory_summary"]
            status = reg.get("fda_compliance_status", "—")
            colour = "🟢" if "compliant" in status.lower() else "🔴"
            st.markdown(f"**FDA Compliance Status:** {colour} {status}")
            gaps = reg.get("compliance_gaps", [])
            if gaps:
                st.markdown("**Compliance Gaps**")
                for g in gaps:
                    st.markdown(f"- ⚠️ {g}")
            st.markdown("**Full Regulatory Summary**")
            st.write(reg.get("regulatory_summary", "—"))

    # ── Conversation Summary ───────────────────────────────────────────────────
    if "conversation_summary" in results and results["conversation_summary"]:
        with st.expander("💬 Conversation Summary", expanded=False):
            st.markdown(results["conversation_summary"])

    # ── Critique Feedback ──────────────────────────────────────────────────────
    if "critique_feedback" in results and results["critique_feedback"]:
        with st.expander("🔍 Critique Feedback", expanded=False):
            st.code(results["critique_feedback"], language="text")

    # ── Final Report ───────────────────────────────────────────────────────────
    if "final_report" in results:
        st.divider()
        st.markdown("### 📄 Final Regulatory Report")
        st.markdown(
            f'<div class="report-box">{results["final_report"]}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            label="⬇️ Download Final Report (.txt)",
            data=results["final_report"],
            file_name="clinical_trial_final_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

elif not run_button:
    st.info(
        "👈 Configure your settings in the sidebar, then click **Run Clinical Analysis** to start."
    )
