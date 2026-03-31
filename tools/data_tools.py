"""
Data Loading Tools for the Clinical Insights Agent.
These tools are registered with the LangGraph ToolNode (Data Loader node).
Each tool implements a specific data source loading strategy.
In production, replace dummy logic with real connectors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from utils.logger import get_logger

logger = get_logger(__name__)

# Default fallback data path (relative to project root)
_DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "resources",
    "clinical_trial_data.csv",
)


# ---------------------------------------------------------------------------
# Tool: Load from Filesystem
# ---------------------------------------------------------------------------

@tool
def load_from_filesystem(file_path: str) -> str:
    """Load clinical trial data from a CSV file on the filesystem or cloud storage.

    Attempts to read the specified path; falls back to the bundled sample dataset
    if the file is not found.

    Args:
        file_path: Absolute or relative path to the CSV file.

    Returns:
        JSON string containing data shape, columns, sample records, null counts,
        and the full dataset serialised as JSON.
    """
    logger.info(f"[Tool] load_from_filesystem | path={file_path}")

    try:
        df = pd.read_csv(file_path)
        source_used = file_path
        logger.info(f"[Tool] Loaded {len(df)} records from '{file_path}'")
    except FileNotFoundError:
        logger.warning(
            f"[Tool] File not found at '{file_path}'. "
            f"Falling back to bundled sample data at '{_DEFAULT_DATA_PATH}'"
        )
        df = pd.read_csv(_DEFAULT_DATA_PATH)
        source_used = _DEFAULT_DATA_PATH

    result = _build_data_summary(df, source="filesystem", source_detail=source_used)
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool: Load from API
# ---------------------------------------------------------------------------

@tool
def load_from_api(api_endpoint: str, params: Optional[str] = None) -> str:
    """Load clinical trial data from a REST API endpoint.

    Args:
        api_endpoint: Full URL of the API endpoint.
        params:       Optional JSON string of query parameters.

    Returns:
        JSON string of loaded data (uses sample data in this demo).
    """
    logger.info(f"[Tool] load_from_api | endpoint={api_endpoint} | params={params}")

    # ── DUMMY IMPLEMENTATION ────────────────────────────────────────────────
    # Replace the block below with:
    #   import requests
    #   response = requests.get(api_endpoint, params=json.loads(params or "{}"))
    #   df = pd.DataFrame(response.json()["data"])
    # ────────────────────────────────────────────────────────────────────────
    df = _load_sample_data()
    result = _build_data_summary(df, source="api", source_detail=api_endpoint)
    result["note"] = "DEMO: replace with real API call in production."
    logger.info(f"[Tool] Returning {len(df)} records from API stub")
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool: Load from Database
# ---------------------------------------------------------------------------

@tool
def load_from_database(connection_string: str, query: str) -> str:
    """Load clinical trial data from a relational or NoSQL database.

    Args:
        connection_string: SQLAlchemy-compatible connection string.
        query:             SQL query to execute against the database.

    Returns:
        JSON string of query results (uses sample data in this demo).
    """
    logger.info(f"[Tool] load_from_database | query={query}")

    # ── DUMMY IMPLEMENTATION ────────────────────────────────────────────────
    # Replace the block below with:
    #   from sqlalchemy import create_engine, text
    #   engine = create_engine(connection_string)
    #   with engine.connect() as conn:
    #       df = pd.read_sql(text(query), conn)
    # For PySpark:
    #   df = spark.read.jdbc(url=connection_string, table=f"({query}) tmp")
    #   df = df.toPandas()
    # ────────────────────────────────────────────────────────────────────────
    df = _load_sample_data()
    result = _build_data_summary(df, source="database", source_detail=query)
    result["note"] = "DEMO: replace with real DB connection in production."
    logger.info(f"[Tool] Returning {len(df)} records from DB stub")
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool: Web Search
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search for relevant clinical trial and regulatory information on the web.

    Args:
        query: Natural language search query.

    Returns:
        JSON string of search results.
    """
    logger.info(f"[Tool] web_search | query={query}")

    # ── DUMMY IMPLEMENTATION ────────────────────────────────────────────────
    # Replace with a real search provider, e.g.:
    #   from langchain_community.tools import TavilySearchResults
    #   results = TavilySearchResults().run(query)
    # ────────────────────────────────────────────────────────────────────────
    results = {
        "query": query,
        "results": [
            {
                "title": "FDA Guidance for Industry – Clinical Trial Design",
                "snippet": (
                    "Comprehensive FDA guidelines covering adaptive trial design, "
                    "endpoint selection, and statistical analysis plans."
                ),
                "url": "https://www.fda.gov/science-research/clinical-trials-and-human-subject-protection",
            },
            {
                "title": "ICH E6(R2) – Good Clinical Practice Consolidated Guideline",
                "snippet": (
                    "International standard for the design, conduct, performance, "
                    "monitoring, auditing, recording, and reporting of clinical trials."
                ),
                "url": "https://www.ich.org/page/efficacy-guidelines",
            },
            {
                "title": "ICH E9 – Statistical Principles for Clinical Trials",
                "snippet": (
                    "Guidance on study design, conduct, and analysis, including "
                    "handling of missing data and multiplicity."
                ),
                "url": "https://www.ich.org/page/efficacy-guidelines",
            },
        ],
        "note": "DEMO: replace with real search API in production.",
    }
    return json.dumps(results)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _load_sample_data() -> pd.DataFrame:
    """Load the bundled sample clinical trial dataset."""
    return pd.read_csv(_DEFAULT_DATA_PATH)


def _build_data_summary(df: pd.DataFrame, source: str, source_detail: str) -> dict:
    """Serialise a DataFrame into a structured summary dict."""
    return {
        "source": source,
        "source_detail": source_detail,
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "sample_records": df.head(10).fillna("N/A").to_dict(orient="records"),
        "data_json": df.to_json(orient="records"),
    }


# Expose all tools as a list for ToolNode registration
ALL_TOOLS = [load_from_filesystem, load_from_api, load_from_database, web_search]
