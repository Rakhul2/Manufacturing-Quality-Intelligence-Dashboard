"""
preprocessing.py
----------------
Data loading and cleaning utilities for the SECOM manufacturing quality dataset.

These functions are intentionally kept pure (input -> output, no hidden state)
so they can be unit-tested and reused across notebooks, the dashboard build
script, and any future retraining job.
"""

import pandas as pd
import numpy as np

MISSING_THRESHOLD_PCT = 40.0  # sensors with more than this % missing are dropped
LABEL_COL = "Pass/Fail"
TIME_COL = "Time"


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw SECOM CSV, parsing the Time column."""
    df = pd.read_csv(path)
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df


def get_sensor_columns(df: pd.DataFrame) -> list:
    """Return the list of sensor feature columns (everything except Time/label)."""
    return [c for c in df.columns if c not in (TIME_COL, LABEL_COL)]


def missing_value_report(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """Return a per-sensor missing-value percentage report, sorted descending."""
    pct = df[sensor_cols].isna().mean() * 100
    return pct.sort_values(ascending=False).rename("missing_pct").to_frame()


def drop_high_missing(df: pd.DataFrame, sensor_cols: list,
                       threshold_pct: float = MISSING_THRESHOLD_PCT):
    """
    Drop sensor columns whose missing-value percentage exceeds threshold_pct.

    Returns (df_reduced, dropped_columns, kept_columns).
    """
    pct = df[sensor_cols].isna().mean() * 100
    dropped = pct[pct > threshold_pct].index.tolist()
    kept = [c for c in sensor_cols if c not in dropped]
    return df.drop(columns=dropped), dropped, kept


def drop_zero_variance(df: pd.DataFrame, sensor_cols: list):
    """
    Drop sensor columns with zero variance (constant, or constant aside from NaNs).
    These carry no discriminative information for pass/fail classification.

    Returns (df_reduced, dropped_columns, kept_columns).
    """
    var = df[sensor_cols].var(skipna=True)
    dropped = var[var == 0].index.tolist()
    kept = [c for c in sensor_cols if c not in dropped]
    return df.drop(columns=dropped), dropped, kept


def impute_median(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """Impute remaining missing values in sensor columns with the column median."""
    df = df.copy()
    for c in sensor_cols:
        median = df[c].median()
        df[c] = df[c].fillna(median)
    return df


def clean_pipeline(df: pd.DataFrame, threshold_pct: float = MISSING_THRESHOLD_PCT) -> dict:
    """
    Run the full cleaning pipeline and return a dict with:
      - 'data': cleaned DataFrame (Time, label, cleaned sensor columns, median-imputed)
      - 'summary': dict of counts at each stage
      - 'dropped_missing': list of columns dropped for missingness
      - 'dropped_variance': list of columns dropped for zero variance
    """
    sensor_cols = get_sensor_columns(df)
    raw_count = len(sensor_cols)

    df1, dropped_missing, kept1 = drop_high_missing(df, sensor_cols, threshold_pct)
    df2, dropped_variance, kept2 = drop_zero_variance(df1, kept1)
    df3 = impute_median(df2, kept2)

    summary = {
        "total_units": int(len(df3)),
        "pass_units": int((df3[LABEL_COL] == -1).sum()) if LABEL_COL in df3 else None,
        "fail_units": int((df3[LABEL_COL] == 1).sum()) if LABEL_COL in df3 else None,
        "raw_sensors": raw_count,
        "dropped_high_missing": len(dropped_missing),
        "dropped_zero_variance": len(dropped_variance),
        "cleaned_sensors": len(kept2),
        "missing_threshold_pct": threshold_pct,
    }
    if summary["pass_units"] is not None and summary["total_units"]:
        summary["fail_rate_pct"] = round(100 * summary["fail_units"] / summary["total_units"], 2)

    return {
        "data": df3,
        "summary": summary,
        "dropped_missing": dropped_missing,
        "dropped_variance": dropped_variance,
        "cleaned_sensor_cols": kept2,
    }
