# 🧬 GenAI-Powered Clinical Insights Agent

An autonomous agentic AI system for pharmaceutical clinical trial analysis, built with **LangGraph** and **Azure OpenAI (GPT-4o)**. The agent ingests clinical trial data, performs multi-dimensional analysis, detects issues, simulates scenarios, and generates FDA-compliant regulatory reports.

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Component Reference](#component-reference)
- [Agentic Workflow](#agentic-workflow)
- [Getting Started](#getting-started)
- [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Extending the System](#extending-the-system)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Clinical Insights Agent                           │
│                                                                      │
│  START → Assistant → Data Loader → Data Cleanup → Call Subgraphs   │
│                                                        │             │
│                              ┌─────────────────────────┤             │
│                              ▼                         ▼             │
│                       Trial Insights            Cohort Comparison   │
│                       (Subgraph)                (Subgraph)          │
│                         │    │                      │               │
│                  ┌──────┘    └──────────────────────┤               │
│                  ▼                                  │               │
│           Clinical Simulation ◄─────────────────────┘               │
│                  │                                                   │
│           Regulatory Summarisation (runs in parallel)               │
│                  │                                                   │
│           Regulated Trial Summarisation                              │
│                  │                                                   │
│          ┌───────┴───────┐                                           │
│          ▼               ▼                                           │
│       Critique       Final Report → END                             │
│          │                                                           │
│          └──────► Call Subgraphs (agentic loop)                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Feature | Implementation |
|---|---|
| LLM | EPAM DIAL Azure OpenAI (gpt-4o-mini-2024-07-18) |
| Orchestration | LangGraph StateGraph |
| State persistence | SQLite via `SqliteSaver` checkpointer |
| Structured output | Pydantic models + `.with_structured_output()` |
| Prompt management | LangChain `PromptTemplate` |
| Parallelism | LangGraph fan-out/fan-in edges |
| UI | Streamlit |
| Containerisation | Docker multi-stage build |

---

## Component Reference

### Nodes

| Node | File | Purpose |
|---|---|---|
| **Assistant** | `nodes/assistant.py` | Entry point; routes to tools or cleanup |
| **Data Loader** | `tools/data_tools.py` + ToolNode | Executes data-loading tools |
| **Data Cleanup** | `nodes/data_cleanup.py` | Cleanses and normalises dataset |
| **Trial Insights** | `graph/subgraphs/trial_insights.py` | Subgraph: analysis + issue detection |
| **Cohort Comparison** | `graph/subgraphs/cohort_comparison.py` | Subgraph: inter-cohort statistics |
| **Clinical Simulation** | `nodes/clinical_simulation.py` | Simulates dosage/compliance scenarios |
| **Regulatory Summarisation** | `nodes/regulatory_summarization.py` | FDA-aligned compliance summary |
| **Regulated Trial Summary** | `nodes/regulated_trial_summarization.py` | Integrated trial report draft |
| **Critique** | `nodes/critique.py` | Reviews summary; triggers feedback loop |
| **Final Report** | `nodes/final_report.py` | Formats submission-ready final report |

### Subgraphs

#### Trial Insights Subgraph
```
START → Data Analysis → Issue Detection → END
```
- **Data Analysis**: Analyses vitals, doctor notes, drug efficacy, compliance
- **Issue Detection**: Flags non-compliance, adverse events, anomalies, ineffectiveness

#### Cohort Comparison Subgraph
```
START → Cohort Comparison Analysis → END
```
- Compares outcomes between Treatment A, Treatment B, and Control cohorts
- Produces statistical findings, p-values, effect sizes

### Tools

| Tool | Source | Description |
|---|---|---|
| `load_from_filesystem` | `tools/data_tools.py` | Reads CSV from local/cloud path |
| `load_from_api` | `tools/data_tools.py` | Fetches from REST API (stub) |
| `load_from_database` | `tools/data_tools.py` | Queries SQL/NoSQL database (stub) |
| `web_search` | `tools/data_tools.py` | Searches regulatory information (stub) |

### State Schema

```python
class GraphState(TypedDict):
    user_input: str                     # User's analysis request
    max_no_of_reviews: int              # Max critique cycles
    messages: list                      # Conversation history (checkpointed)
    raw_data: Any                       # Tool-loaded raw data
    cleaned_data: Any                   # Cleansed dataset (JSON)
    data_stats: dict                    # Descriptive statistics
    trial_insights: dict                # Trial Insights subgraph output
    cohort_comparison: dict             # Cohort Comparison subgraph output
    clinical_simulation: dict           # Simulation predictions
    regulatory_summary: dict            # FDA regulatory summary
    regulated_trial_summary: dict       # Integrated trial summary
    critique_feedback: str              # Critique review feedback
    conversation_summary: Optional[str] # Summary of the conversation history for final report context
    final_report: str                   # Final regulatory report
    review_count: int                   # Current iteration count
    error: Optional[str]                # Error messages
```

---

## Agentic Workflow

### Parallelism

The graph exploits LangGraph's native parallel execution at two points:

**Fan-out 1** – After Data Cleanup:
```
Data Cleanup → Trial Insights   ─┐
             → Cohort Comparison ─┘ (run in parallel)
```

**Fan-out 2** – After subgraphs complete:
```
Trial Insights + Cohort Comparison → Clinical Simulation      ─┐
Trial Insights                     → Regulatory Summarisation ─┘ (run in parallel)
```

### Critique Agentic Loop

```
Regulated Trial Summary ─→ Critique ─→ Call Subgraphs ─→ (re-run)
        ↑                                                      │
        └──────────────────────────────────────────────────────┘
                     (until review_count >= max_no_of_reviews)
```

Each loop cycle:
1. The Critique node scores the summary (1–10) and produces actionable feedback
2. The feedback is stored in `critique_feedback` and injected into subsequent analysis prompts
3. After `max_no_of_reviews` cycles, the workflow routes to Final Report

### State Checkpointing

Every "superstep" (node execution) is automatically persisted to SQLite via LangGraph's `SqliteSaver`. This enables:
- Fault tolerance and workflow resumption
- Audit trail of all intermediate states
- Multi-session support (via unique `thread_id`)

---

## Getting Started

### Prerequisites

- Python 3.11+
- An Azure subscription with access to Azure OpenAI
- Azure OpenAI API key and endpoint URL

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd clinical_insights_agent

# 2. Create virtual environment
python -m venv trial_venv
source trial_venv/bin/activate  # Windows: trial_venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set Azure OpenAI credentials:
#   AZURE_OPENAI_API_KEY=<your-api-key>
#   AZURE_OPENAI_ENDPOINT=<your-endpoint-url>
#   AZURE_OPENAI_DEPLOYMENT_NAME=<your-deployment-name>
```

### Run the Streamlit UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Run using LangGraph Studio (LangSmith)
Ref: [https://docs.langchain.com/langsmith/cli#dev](https://docs.langchain.com/langsmith/cli#dev)

```bash
pip install -U "langgraph-cli[inmem]"
cd graph/
langgraph dev
```

The studio URL will be opening up automatically or can be clicked from the terminal. You should have an account created on the LangChain for looging into the console.

### Run from CLI

```bash
python main.py
```

The final report will be printed to the console and saved as `clinical_trial_final_report.txt`.

---

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t clinical-insights-agent .

# Run with Azure OpenAI
docker run -p 8501:8501 \
  -e AZURE_OPENAI_API_KEY=<your-api-key> \
  -e AZURE_OPENAI_ENDPOINT=<your-endpoint-url> \
  -e AZURE_OPENAI_DEPLOYMENT_NAME=<your-deployment-name> \
  -e MAX_REVIEWS=2 \
  clinical-insights-agent
```

### Docker Compose (optional)

```yaml
version: '3.8'
services:
  clinical-agent:
    build: .
    ports:
      - "8501:8501"
    environment:
      - AZURE_OPENAI_API_KEY=<your-api-key>
      - AZURE_OPENAI_ENDPOINT=<your-endpoint-url>
      - AZURE_OPENAI_DEPLOYMENT_NAME=<your-deployment-name>
      - MAX_REVIEWS=2
    volumes:
      - ./data:/app/data
      - ./resources:/app/resources
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | N/A | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | N/A | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | N/A | Azure OpenAI deployment name |
| `MAX_REVIEWS` | `2` | CLI-only max review cycles |
| `SQLITE_DB_PATH` | `clinical_insights.db` | Checkpointer database path |

---

## Project Structure

```
clinical_insights_agent/
├── app.py                          # Streamlit UI
├── main.py                         # CLI entry point
├── Dockerfile                      # Multi-stage Docker build
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── README.md                       # This file
│
├── graph/
│   ├── state.py                    # All state & Pydantic schemas
│   ├── workflow.py                 # Main LangGraph graph definition
│   └── subgraphs/
│       ├── trial_insights.py       # Trial Insights subgraph
│       └── cohort_comparison.py    # Cohort Comparison subgraph
│
├── nodes/
│   ├── assistant.py                # LLM + tool routing node
│   ├── data_cleanup.py             # Data cleansing (Pandas/PySpark)
│   ├── clinical_simulation.py      # Dosage/compliance simulation
│   ├── regulatory_summarization.py # FDA regulatory summary
│   ├── regulated_trial_summarization.py # Integrated trial summary
│   ├── conversation_summary.py          # Summarizes the conversation history to date
│   ├── critique.py                 # Quality review + feedback
│   └── final_report.py             # Final report formatter
│
├── tools/
│   └── data_tools.py               # Data loading tools (filesystem/api/db/web)
│
├── prompts/
│   └── templates.py                # All LangChain PromptTemplates
│
├── chains/
│   └── chains.py                   # LLM chain builders (structured/text)
│
├── utils/
│   └── logger.py                   # Coloured logging utility
│
└── resources/
    └── clinical_trial_data.csv     # Sample 40-patient trial dataset
```

---

## Extending the System

### Add a New Data Source Tool

```python
# tools/data_tools.py
@tool
def load_from_s3(bucket: str, key: str) -> str:
    """Load clinical data from AWS S3."""
    import boto3
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj["Body"])
    return json.dumps(_build_data_summary(df, "s3", f"s3://{bucket}/{key}"))
```

Register in `ALL_TOOLS` list and it will be available to the Assistant.

### Add a New Analysis Node

```python
# nodes/my_new_node.py
from chains.chains import build_structured_chain
from prompts.templates import MY_NEW_PROMPT
from graph.state import MyNewOutput, GraphState

def my_new_node(state: GraphState) -> dict:
    chain = build_structured_chain(MY_NEW_PROMPT, MyNewOutput)
    result = chain.invoke({...})
    return {"my_new_output": result.model_dump()}
```

Add to `graph/workflow.py` with appropriate edges.

### Replace Pandas with PySpark

Each `nodes/data_cleanup.py` function contains commented PySpark equivalents. On Databricks:

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("ClinicalInsights").getOrCreate()
df = spark.read.csv(file_path, header=True, inferSchema=True)
# ... transformations ...
pandas_df = df.toPandas()  # Convert for LLM consumption
```

---

## Regulatory Compliance

Reports generated by this agent are aligned with:

- **FDA 21 CFR Part 312** – IND Regulations
- **ICH E6(R2)** – Good Clinical Practice
- **ICH E9** – Statistical Principles for Clinical Trials
- **FDA 21 CFR Part 11** – Electronic Records

> ⚠️ **Disclaimer**: This system is a decision-support tool. All AI-generated outputs must be reviewed and validated by qualified clinical and regulatory professionals before submission.

---

## License

MIT License – See LICENSE file for details.
