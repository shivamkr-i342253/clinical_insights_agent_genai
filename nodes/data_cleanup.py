"""
Data Cleanup Node – Cleanses and normalises the raw clinical trial dataset.
Implements PySpark-style transformation logic using Pandas (with comments
showing the equivalent PySpark code for Databricks deployment).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from graph.state import GraphState
from utils.logger import get_logger

logger = get_logger(__name__)


def data_cleanup_node(state: GraphState) -> dict:
    """
    Data Cleanup node handler.

    Reads raw_data from state, performs comprehensive data cleansing, and
    writes cleaned_data + data_stats back to state.

    Cleansing steps:
      1. Parse raw JSON data.
      2. Drop fully duplicated rows.
      3. Handle missing values (imputation / removal by column type).
      4. Enforce correct dtypes.
      5. Remove out-of-range / physiologically impossible values.
      6. Normalise string columns (strip, lower-case categoricals).
      7. Compute descriptive statistics.

    Args:
        state: Current graph state containing raw_data.

    Returns:
        Partial state update with cleaned_data and data_stats.
    """
    logger.info("[Node] ── DATA CLEANUP ───────────────────────────────────────")

    raw_data = state.get("raw_data")
    if not raw_data:
        logger.warning("[Node:DataCleanup] No raw_data in state. Using empty fallback.")
        raw_data = "{}"

    # ── Step 1: Parse raw data ───────────────────────────────────────────────
    df = _parse_raw_data(raw_data)
    logger.info(f"[Node:DataCleanup] Parsed DataFrame | shape={df.shape}")

    # ── Step 2: Remove fully duplicate rows ──────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"[Node:DataCleanup] Duplicates removed: {before - len(df)} rows")

    # PySpark equivalent:
    # df = df.dropDuplicates()

    # ── Step 3: Enforce data types ───────────────────────────────────────────
    df = _enforce_dtypes(df)

    # ── Step 4: Handle missing values ────────────────────────────────────────
    df = _handle_missing_values(df)
    logger.info(f"[Node:DataCleanup] After missing-value handling | shape={df.shape}")

    # ── Step 5: Remove physiologically impossible values ────────────────────
    df = _remove_outliers(df)
    logger.info(f"[Node:DataCleanup] After outlier removal | shape={df.shape}")

    # ── Step 6: Normalise string/categorical columns ─────────────────────────
    df = _normalise_strings(df)

    # ── Step 7: Compute descriptive statistics ───────────────────────────────
    data_stats = _compute_stats(df)

    logger.info(
        f"[Node:DataCleanup] Cleanup complete | "
        f"final_shape={df.shape} | "
        f"null_total={df.isnull().sum().sum()}"
    )

    return {
        "cleaned_data": df.to_json(orient="records"),
        "data_stats": data_stats,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_raw_data(raw_data: Any) -> pd.DataFrame:
    """Parse raw_data (JSON string or dict) into a DataFrame."""
    try:
        if isinstance(raw_data, str):
            payload = json.loads(raw_data)
        else:
            payload = raw_data

        # Tool returns a dict with a "data_json" key
        data_json = payload.get("data_json") or payload.get("sample_records")
        if data_json:
            if isinstance(data_json, str):
                records = json.loads(data_json)
            else:
                records = data_json
            return pd.DataFrame(records)

        # Fallback: treat entire payload as records
        return pd.DataFrame([payload])

    except Exception as exc:
        logger.error(f"[Node:DataCleanup] Failed to parse raw_data: {exc}")
        # Return empty DataFrame with expected schema
        return pd.DataFrame()


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to their expected data types."""
    int_cols = [
        "age", "systolic_bp", "diastolic_bp", "heart_rate",
        "drug_dosage_mg", "missed_doses", "trial_week",
    ]
    float_cols = [
        "baseline_bmi", "temperature", "efficacy_score",
        "biomarker_level", "compliance_rate", "visit_adherence",
    ]
    bool_cols = ["withdrawal"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )

    # PySpark equivalent:
    # from pyspark.sql.functions import col
    # from pyspark.sql.types import IntegerType, FloatType, BooleanType
    # for c in int_cols:
    #     df = df.withColumn(c, col(c).cast(IntegerType()))

    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or remove missing values based on column semantics.
    - Numeric continuous: median imputation.
    - Categorical/string: 'Unknown' fill.
    - Critical ID columns: rows dropped.
    """
    if "patient_id" in df.columns:
        before = len(df)
        df = df.dropna(subset=["patient_id"])
        if before - len(df):
            logger.warning(f"[Node:DataCleanup] Dropped {before - len(df)} rows with null patient_id")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(f"[Node:DataCleanup] Imputed '{col}' with median={median_val:.2f}")

    string_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in string_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    # PySpark equivalent:
    # from pyspark.sql.functions import median, when, isnan
    # for c in numeric_cols:
    #     med = df.approxQuantile(c, [0.5], 0.01)[0]
    #     df = df.withColumn(c, when(col(c).isNull() | isnan(c), med).otherwise(col(c)))

    return df


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove physiologically impossible or clearly erroneous values."""
    physiological_bounds = {
        "systolic_bp":   (60, 260),
        "diastolic_bp":  (30, 180),
        "heart_rate":    (30, 220),
        "temperature":   (94.0, 108.0),
        "age":           (0, 120),
        "efficacy_score":(0.0, 10.0),
        "compliance_rate":(0.0, 1.0),
        "visit_adherence":(0.0, 1.0),
    }

    for col, (lo, hi) in physiological_bounds.items():
        if col not in df.columns:
            continue
        before = len(df)
        mask = (df[col] >= lo) & (df[col] <= hi)
        df = df[mask | df[col].isnull()]
        removed = before - len(df)
        if removed:
            logger.warning(
                f"[Node:DataCleanup] Removed {removed} rows with '{col}' outside [{lo}, {hi}]"
            )

    # PySpark equivalent:
    # df = df.filter((col("systolic_bp") >= 60) & (col("systolic_bp") <= 260))

    return df.reset_index(drop=True)


def _normalise_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalise known categoricals."""
    strip_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in strip_cols:
        df[col] = df[col].astype(str).str.strip()

    # Normalise known categoricals to title-case
    for col in ["cohort", "gender", "response_status", "adverse_event_severity"]:
        if col in df.columns:
            df[col] = df[col].str.title()

    return df


def _compute_stats(df: pd.DataFrame) -> dict:
    """Compute descriptive statistics for the cleaned dataset."""
    stats: dict = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "null_counts": df.isnull().sum().to_dict(),
        "sample_records": df.head(10).fillna("N/A").to_dict(orient="records"),
    }

    if "cohort" in df.columns:
        stats["cohort_distribution"] = df["cohort"].value_counts().to_dict()

    if "response_status" in df.columns:
        stats["response_distribution"] = df["response_status"].value_counts().to_dict()

    if "adverse_events" in df.columns:
        stats["adverse_event_distribution"] = df["adverse_events"].value_counts().to_dict()

    if "withdrawal" in df.columns:
        stats["withdrawal_count"] = int(df["withdrawal"].sum())

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        desc = numeric_df.describe().round(2)
        stats["numeric_summary"] = desc.to_dict()

    return stats
