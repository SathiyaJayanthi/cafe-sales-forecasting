"""
CoffeeSense AI - Prediction Service Module
Handles single predictions, batch predictions, and test-period evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_prediction_inputs(
    day: int,
    month: int,
    day_of_week: int,
    is_weekend: int,
    prev_day_sales: float,
    prev_3day_avg: float,
    prev_7day_avg: float,
) -> tuple[bool, str]:
    """
    Validate inputs for a single prediction.

    Returns
    -------
    (is_valid, error_message)
    """
    errors: list[str] = []

    # day: 1-31
    if not isinstance(day, (int, float)) or not (1 <= int(day) <= 31):
        errors.append(f"'day' must be between 1 and 31, got {day}.")

    # month: 1-12
    if not isinstance(month, (int, float)) or not (1 <= int(month) <= 12):
        errors.append(f"'month' must be between 1 and 12, got {month}.")

    # day_of_week: 0-6
    if not isinstance(day_of_week, (int, float)) or not (0 <= int(day_of_week) <= 6):
        errors.append(f"'day_of_week' must be between 0 and 6, got {day_of_week}.")

    # is_weekend: 0 or 1
    if int(is_weekend) not in (0, 1):
        errors.append(f"'is_weekend' must be 0 or 1, got {is_weekend}.")

    # sales values must be non-negative
    for name, val in [
        ("prev_day_sales", prev_day_sales),
        ("prev_3day_avg", prev_3day_avg),
        ("prev_7day_avg", prev_7day_avg),
    ]:
        try:
            if float(val) < 0:
                errors.append(f"'{name}' must be non-negative, got {val}.")
        except (TypeError, ValueError):
            errors.append(f"'{name}' must be a numeric value, got {val!r}.")

    if errors:
        return False, " | ".join(errors)
    return True, ""


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

def predict_single(
    model: Any,
    feature_names: list[str],
    day: int,
    month: int,
    day_of_week: int,
    is_weekend: int,
    prev_day_sales: float,
    prev_3day_avg: float,
    prev_7day_avg: float,
) -> dict:
    """
    Make a single revenue prediction.

    Parameters
    ----------
    model         : trained sklearn estimator
    feature_names : list of feature column names (must match training order)
    day, month, day_of_week, is_weekend, prev_day_sales, prev_3day_avg, prev_7day_avg

    Returns
    -------
    dict with keys: prediction, model_name, features_used, confidence_note
    """
    is_valid, error_msg = validate_prediction_inputs(
        day, month, day_of_week, is_weekend, prev_day_sales, prev_3day_avg, prev_7day_avg
    )
    if not is_valid:
        raise ValueError(f"Invalid prediction inputs: {error_msg}")

    features_dict = {
        "day": int(day),
        "month": int(month),
        "day_of_week": int(day_of_week),
        "is_weekend": int(is_weekend),
        "previous_day_sales": float(prev_day_sales),
        "previous_3day_avg": float(prev_3day_avg),
        "previous_7day_avg": float(prev_7day_avg),
    }

    # Build DataFrame aligned to training feature order
    input_df = pd.DataFrame([{k: features_dict.get(k, 0) for k in feature_names}])

    prediction_raw = model.predict(input_df)[0]
    prediction = max(0.0, float(prediction_raw))

    model_name = type(model).__name__

    # Confidence note based on training features vs provided values
    if prev_day_sales > 0 and prev_3day_avg > 0 and prev_7day_avg > 0:
        confidence_note = "High confidence — full lag history provided."
    elif prev_day_sales > 0:
        confidence_note = "Moderate confidence — only previous-day sales available."
    else:
        confidence_note = "Low confidence — no sales history provided; using averages."

    logger.info("predict_single: %s predicted ₹%.2f", model_name, prediction)

    return {
        "prediction": prediction,
        "model_name": model_name,
        "features_used": features_dict,
        "confidence_note": confidence_note,
    }


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_batch(model: Any, feature_names: list[str], feature_df: pd.DataFrame) -> np.ndarray:
    """
    Run batch predictions over a DataFrame of features.

    Parameters
    ----------
    model        : trained sklearn estimator
    feature_names: list of feature column names
    feature_df   : DataFrame with the required feature columns

    Returns
    -------
    np.ndarray of predicted values (clipped to >= 0)
    """
    missing = [c for c in feature_names if c not in feature_df.columns]
    if missing:
        raise ValueError(f"feature_df is missing columns: {missing}")

    input_df = feature_df[feature_names].copy()
    predictions = model.predict(input_df)
    predictions = np.clip(predictions, 0, None)
    logger.info("predict_batch: produced %d predictions.", len(predictions))
    return predictions


# ---------------------------------------------------------------------------
# Human-readable explanation
# ---------------------------------------------------------------------------

def get_prediction_explanation(
    prediction: float,
    features_dict: dict,
    avg_sales: float,
) -> str:
    """
    Generate a natural-language explanation for a prediction.

    Parameters
    ----------
    prediction    : predicted sales value
    features_dict : dict of features used (day, month, day_of_week, etc.)
    avg_sales     : historical average daily sales for context

    Returns
    -------
    Human-readable string.
    """
    _MONTH_NAMES = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }
    _DOW_NAMES = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
        4: "Friday", 5: "Saturday", 6: "Sunday",
    }

    month_name = _MONTH_NAMES.get(int(features_dict.get("month", 1)), "this month")
    dow_name = _DOW_NAMES.get(int(features_dict.get("day_of_week", 0)), "this day")
    prev_sales = features_dict.get("previous_day_sales", 0.0)
    is_weekend = int(features_dict.get("is_weekend", 0))

    day_type = "weekend day" if is_weekend else "weekday"

    # Relative to average
    if avg_sales and avg_sales > 0:
        pct_diff = ((prediction - avg_sales) / avg_sales) * 100
        if pct_diff > 5:
            relative = f" This is {pct_diff:.1f}% above the average daily revenue."
        elif pct_diff < -5:
            relative = f" This is {abs(pct_diff):.1f}% below the average daily revenue."
        else:
            relative = " This is close to the average daily revenue."
    else:
        relative = ""

    explanation = (
        f"Based on historical patterns for a {dow_name} in {month_name} "
        f"(a {day_type}) with previous day sales of ₹{prev_sales:,.0f}, "
        f"the model predicts daily revenue of ₹{prediction:,.0f}.{relative}"
    )
    return explanation


# ---------------------------------------------------------------------------
# Test-period evaluation
# ---------------------------------------------------------------------------

def predict_test_period(
    model: Any,
    feature_names: list[str],
    feature_df: pd.DataFrame,
    y_actual: np.ndarray | pd.Series,
) -> dict:
    """
    Generate predictions for a test period and compute evaluation metrics.

    Parameters
    ----------
    model        : trained sklearn estimator
    feature_names: list of feature column names
    feature_df   : DataFrame of test features
    y_actual     : actual target values (same length as feature_df)

    Returns
    -------
    dict with keys:
        predictions  – np.ndarray of predicted values
        actuals      – np.ndarray of actual values
        mae          – Mean Absolute Error
        mse          – Mean Squared Error
        rmse         – Root Mean Squared Error
        r2           – R² score
        mape         – Mean Absolute Percentage Error (%)
    """
    y_pred = predict_batch(model, feature_names, feature_df)
    y_actual_arr = np.array(y_actual)

    mae = float(mean_absolute_error(y_actual_arr, y_pred))
    mse = float(mean_squared_error(y_actual_arr, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_actual_arr, y_pred))

    # MAPE — guard against zero actuals
    nonzero_mask = y_actual_arr != 0
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs((y_actual_arr[nonzero_mask] - y_pred[nonzero_mask]) / y_actual_arr[nonzero_mask])) * 100
        )
    else:
        mape = float("nan")

    logger.info(
        "predict_test_period: MAE=%.2f  RMSE=%.2f  R²=%.4f  MAPE=%.2f%%",
        mae, rmse, r2, mape if not np.isnan(mape) else -1,
    )

    return {
        "predictions": y_pred,
        "actuals": y_actual_arr,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def evaluate_test_period(
    model: Any,
    feature_names: list[str],
    daily_df: pd.DataFrame,
) -> dict:
    """
    Convenience wrapper to evaluate the model on the chronological 20% holdout test set.
    """
    from src.feature_engineering import (
        create_features,
        split_features_target,
        chronological_train_test_split,
    )

    feat_df = create_features(daily_df)
    X, y = split_features_target(feat_df)
    _, X_test, _, y_test = chronological_train_test_split(X, y, test_size=0.2)

    eval_dict = predict_test_period(model, feature_names, X_test, y_test)

    if "date" in daily_df.columns:
        test_dates = daily_df["date"].iloc[-len(y_test):].values
    else:
        test_dates = range(len(y_test))

    predictions_df = pd.DataFrame({
        "date": test_dates,
        "actual_sales": eval_dict["actuals"],
        "predicted_sales": eval_dict["predictions"],
    })

    return {
        "metrics": {
            "MAE": eval_dict["mae"],
            "MSE": eval_dict["mse"],
            "RMSE": eval_dict["rmse"],
            "R2": eval_dict["r2"],
            "MAPE": eval_dict["mape"],
        },
        "predictions_df": predictions_df,
    }


def get_sample_prediction_inputs() -> dict:
    """Return default template feature inputs."""
    return {
        "day": 15,
        "month": 6,
        "day_of_week": 2,
        "is_weekend": 0,
        "previous_day_sales": 550.0,
        "previous_3day_avg": 540.0,
        "previous_7day_avg": 535.0,
    }
