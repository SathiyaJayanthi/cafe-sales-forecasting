"""
CoffeeSense AI - Feature Engineering Module
Creates lag / rolling features for the sales prediction model.

No data-leakage columns (transaction_count, cash_transactions, card_transactions)
are permitted in the feature set — use validate_no_leakage() to enforce this.
"""

import logging

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

LEAKAGE_COLUMNS = ["transaction_count", "cash_transactions", "card_transactions"]


def validate_no_leakage(feature_cols: list) -> None:
    """
    Raise ValueError if any leakage column is present in the feature list.

    Parameters
    ----------
    feature_cols : list of str
        Column names to check.
    """
    leaking = [c for c in feature_cols if c in LEAKAGE_COLUMNS]
    if leaking:
        raise ValueError(
            f"Data leakage detected! The following columns must NOT be used as "
            f"features because they are derived from the same day's transactions: "
            f"{leaking}"
        )


# ---------------------------------------------------------------------------
# Feature creation
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling-average features to the daily sales DataFrame.

    The DataFrame must:
    - Contain a 'daily_sales' column.
    - Be sorted ascending by date (earliest row first).

    Adds:
        previous_day_sales  — sales from the previous day (shift 1)
        previous_3day_avg   — 3-day rolling mean of previous sales
        previous_7day_avg   — 7-day rolling mean of previous sales

    Drops the first 7 rows which will have NaN values due to rolling windows.
    Returns a clean DataFrame with only the required columns.
    """
    if "daily_sales" not in df.columns:
        raise ValueError("DataFrame must contain a 'daily_sales' column.")

    df = df.copy()

    # Lag features (shift by 1 to avoid leaking today's value)
    df["previous_day_sales"] = df["daily_sales"].shift(1)
    df["previous_3day_avg"] = df["daily_sales"].shift(1).rolling(window=3).mean()
    df["previous_7day_avg"] = df["daily_sales"].shift(1).rolling(window=7).mean()

    # Keep only the columns needed for ML
    keep_cols = [
        "day",
        "month",
        "day_of_week",
        "is_weekend",
        "previous_day_sales",
        "previous_3day_avg",
        "previous_7day_avg",
        "daily_sales",
    ]
    # Only keep columns that exist in the DataFrame
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Drop rows with NaN (the first 7 rows due to 7-day rolling window)
    before = len(df)
    df = df.dropna()
    logger.info(
        "create_features: dropped %d rows with NaN (rolling window warm-up). "
        "Remaining: %d rows.",
        before - len(df), len(df),
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature name helper
# ---------------------------------------------------------------------------

def get_feature_names() -> list:
    """Return the ordered list of feature column names."""
    return [
        "day",
        "month",
        "day_of_week",
        "is_weekend",
        "previous_day_sales",
        "previous_3day_avg",
        "previous_7day_avg",
    ]


# ---------------------------------------------------------------------------
# X / y split
# ---------------------------------------------------------------------------

def split_features_target(df: pd.DataFrame):
    """
    Split a feature-engineered DataFrame into X (features) and y (target).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    validate_no_leakage(FEATURE_COLUMNS)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


# ---------------------------------------------------------------------------
# Chronological train / test split
# ---------------------------------------------------------------------------

def chronological_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """
    Split X and y chronologically (no shuffling) using the first 80 % for
    training and the remaining 20 % for testing.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    test_size : float, default 0.2

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    logger.info(
        "Chronological split — train: %d rows, test: %d rows (split index: %d).",
        len(X_train), len(X_test), split_idx,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Prediction input builder
# ---------------------------------------------------------------------------

def prepare_prediction_input(
    day: int,
    month: int,
    day_of_week: int,
    is_weekend: int,
    prev_day_sales: float,
    prev_3day_avg: float,
    prev_7day_avg: float,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame suitable for model.predict().

    Parameters
    ----------
    day         : int   — day of month (1–31)
    month       : int   — month number (1–12)
    day_of_week : int   — 0=Monday … 6=Sunday
    is_weekend  : int   — 1 if Saturday/Sunday, else 0
    prev_day_sales  : float — previous day's total sales
    prev_3day_avg   : float — rolling 3-day average of previous sales
    prev_7day_avg   : float — rolling 7-day average of previous sales

    Returns
    -------
    pd.DataFrame with FEATURE_COLUMNS as column names, one row.
    """
    data = {
        "day": [int(day)],
        "month": [int(month)],
        "day_of_week": [int(day_of_week)],
        "is_weekend": [int(is_weekend)],
        "previous_day_sales": [float(prev_day_sales)],
        "previous_3day_avg": [float(prev_3day_avg)],
        "previous_7day_avg": [float(prev_7day_avg)],
    }
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)
