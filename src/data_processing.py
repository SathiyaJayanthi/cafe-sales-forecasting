"""
CoffeeSense AI - Data Processing Module
Handles loading, validation, cleaning, and saving of both datasets.

Dataset notes
-------------
Daily dataset  : coffee_shop_sales_dataset_2025_daily.csv  (365 rows expected)
Transactions   : coffee_shop_transactions_raw_2025.csv      (6 592 rows expected)
Datetime format: "DD-MM-YYYY HH.MM"  (dots between hours and minutes, NOT colons)
Date column    : "DD-MM-YYYY" format
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_daily_dataset(filepath) -> pd.DataFrame:
    """Load the daily sales CSV and return a DataFrame."""
    filepath = Path(filepath)
    logger.info("Loading daily dataset from: %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Daily dataset loaded — shape: %s", df.shape)
    return df


def load_transactions_dataset(filepath) -> pd.DataFrame:
    """
    Load the transactions CSV, parse the datetime column, and extract an hour column.

    Datetime format in file: "DD-MM-YYYY HH.MM"
    """
    filepath = Path(filepath)
    logger.info("Loading transactions dataset from: %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Transactions raw shape: %s", df.shape)

    # Detect the datetime column (case-insensitive search)
    dt_col = None
    for col in df.columns:
        if "datetime" in col.lower() or "date_time" in col.lower():
            dt_col = col
            break

    if dt_col is not None:
        logger.info("Parsing datetime column: '%s' using format '%%d-%%m-%%Y %%H.%%M'", dt_col)
        df["datetime"] = pd.to_datetime(
            df[dt_col],
            format="%d-%m-%Y %H.%M",
            errors="coerce",
        )
        df["hour"] = df["datetime"].dt.hour
        logger.info(
            "Datetime parsed — unparseable rows: %d",
            df["datetime"].isna().sum(),
        )
    else:
        logger.warning("No datetime column found in transactions dataset.")

    logger.info("Transactions dataset loaded — shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_daily_dataset(df: pd.DataFrame) -> dict:
    """Validate the daily dataset and return a summary dict."""
    expected_rows = 365
    result = {
        "shape": df.shape,
        "missing": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "columns": list(df.columns),
        "row_count_ok": len(df) == expected_rows,
    }
    if not result["row_count_ok"]:
        logger.warning(
            "Daily dataset row count mismatch: expected %d, got %d",
            expected_rows, len(df),
        )
    return result


def validate_transactions_dataset(df: pd.DataFrame) -> dict:
    """Validate the transactions dataset and return a summary dict."""
    expected_rows = 6592
    result = {
        "shape": df.shape,
        "missing": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "columns": list(df.columns),
        "row_count_ok": len(df) == expected_rows,
    }
    if not result["row_count_ok"]:
        logger.warning(
            "Transactions dataset row count mismatch: expected %d, got %d",
            expected_rows, len(df),
        )
    return result


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_datasets(daily_df: pd.DataFrame, trans_df: pd.DataFrame) -> dict:
    """
    Aggregate transaction monetary values by date and compare with daily_sales.

    Returns a dict containing the merged comparison and correlation.
    """
    result = {"status": "ok", "details": "", "correlation": None, "merged": None}

    # Find the money / amount column in transactions
    money_col = None
    for col in trans_df.columns:
        lc = col.lower()
        if any(k in lc for k in ["amount", "money", "total", "price", "revenue", "sale"]):
            money_col = col
            break

    if money_col is None:
        result["status"] = "skipped"
        result["details"] = "No monetary column found in transactions dataset."
        logger.warning(result["details"])
        return result

    # Find date column in transactions
    date_col = None
    for col in trans_df.columns:
        lc = col.lower()
        if "date" in lc and "time" not in lc:
            date_col = col
            break

    if date_col is None and "datetime" in trans_df.columns:
        # Derive date from parsed datetime
        trans_df = trans_df.copy()
        trans_df["_date"] = trans_df["datetime"].dt.date
        date_col = "_date"

    if date_col is None:
        result["status"] = "skipped"
        result["details"] = "No date column found in transactions dataset."
        logger.warning(result["details"])
        return result

    # Aggregate
    agg = (
        trans_df.groupby(date_col)[money_col]
        .sum()
        .reset_index()
        .rename(columns={money_col: "trans_total"})
    )

    # Build a date column in daily_df if needed
    if "date" not in daily_df.columns:
        daily_df = create_date_column(daily_df.copy())

    daily_copy = daily_df.copy()
    daily_copy["_cmp_date"] = pd.to_datetime(daily_copy["date"]).dt.date

    merged = daily_copy.merge(
        agg.rename(columns={date_col: "_cmp_date"}),
        on="_cmp_date",
        how="left",
    )

    if "daily_sales" in merged.columns and "trans_total" in merged.columns:
        valid = merged[["daily_sales", "trans_total"]].dropna()
        if len(valid) > 1:
            corr = float(valid["daily_sales"].corr(valid["trans_total"]))
            result["correlation"] = corr
            logger.info("Cross-validation correlation (daily_sales vs trans_total): %.4f", corr)

    result["merged"] = merged
    result["details"] = f"Aggregated {len(agg)} unique transaction dates."
    return result


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_daily_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the daily sales dataset:
    - Fill missing values with column median (numeric) or mode (categorical)
    - Remove duplicates
    - Enforce correct dtypes
    """
    df = df.copy()

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows from daily dataset.", before - len(df))

    # Fill numeric missing values with median
    int_cols = []
    float_cols = []

    # Map expected columns to their target types
    int_col_names = ["day", "month", "day_of_week", "is_weekend", "transaction_count"]
    float_col_names = ["daily_sales"]

    for col in df.columns:
        if col in int_col_names:
            int_cols.append(col)
        elif col in float_col_names:
            float_cols.append(col)

    for col in int_cols + float_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Filled missing values in '%s' with median: %s", col, median_val)

    # Enforce dtypes
    for col in int_col_names:
        if col in df.columns:
            df[col] = df[col].astype(int)

    for col in float_col_names:
        if col in df.columns:
            df[col] = df[col].astype(float)

    logger.info("Daily dataset cleaned — final shape: %s", df.shape)
    return df


def clean_transactions_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the transactions dataset:
    - Drop rows where the datetime could not be parsed (NaT)
    """
    df = df.copy()
    if "datetime" in df.columns:
        before = len(df)
        df = df.dropna(subset=["datetime"])
        dropped = before - len(df)
        if dropped:
            logger.info(
                "Dropped %d rows with unparseable datetimes from transactions dataset.",
                dropped,
            )
    logger.info("Transactions dataset cleaned — final shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Date column creation
# ---------------------------------------------------------------------------

def create_date_column(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'date' column built from the 'day' and 'month' columns (year = 2025).
    Returns a copy of the DataFrame with the new column.
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(
        df["month"].astype(str) + "-" + df["day"].astype(str) + "-2025",
        format="%m-%d-%Y",
        errors="coerce",
    )
    invalid = df["date"].isna().sum()
    if invalid:
        logger.warning("create_date_column: %d rows produced NaT dates.", invalid)
    logger.info("'date' column created for daily dataset.")
    return df


# ---------------------------------------------------------------------------
# Preprocessing report
# ---------------------------------------------------------------------------

def get_preprocessing_report(daily_df: pd.DataFrame, trans_df: pd.DataFrame) -> dict:
    """Return a combined preprocessing summary for both datasets."""
    daily_validation = validate_daily_dataset(daily_df)
    trans_validation = validate_transactions_dataset(trans_df)
    return {
        "daily": {
            "validation": daily_validation,
            "total_sales": float(daily_df["daily_sales"].sum()) if "daily_sales" in daily_df.columns else None,
            "avg_daily_sales": float(daily_df["daily_sales"].mean()) if "daily_sales" in daily_df.columns else None,
            "date_range": {
                "min_month": int(daily_df["month"].min()) if "month" in daily_df.columns else None,
                "max_month": int(daily_df["month"].max()) if "month" in daily_df.columns else None,
            },
        },
        "transactions": {
            "validation": trans_validation,
            "parsed_datetime_ok": int((~trans_df["datetime"].isna()).sum()) if "datetime" in trans_df.columns else None,
        },
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_cleaned_dataset(df: pd.DataFrame, filepath) -> None:
    """Save a DataFrame to CSV at the given path."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info("Cleaned dataset saved to: %s  (shape: %s)", filepath, df.shape)
