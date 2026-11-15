"""
CoffeeSense AI - Multi-Step Recursive Forecasting Module
Generates recursive multi-step daily sales forecasts (e.g. 7-day, 14-day, 30-day).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_recursive_forecast(
    model: Any,
    feature_names: List[str],
    daily_df: pd.DataFrame,
    horizon_days: int = 7,
    last_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Generate recursive multi-step ahead sales forecasts.

    In recursive forecasting, each one-step prediction is fed back into
    the rolling history to compute lag features (prev_day, prev_3day_avg,
    prev_7day_avg) for future days.

    Parameters
    ----------
    model : trained scikit-learn regressor
    feature_names : list of feature columns expected by the model
    daily_df : DataFrame containing historical daily sales (must have 'daily_sales')
    horizon_days : number of future days to forecast (e.g., 7 or 30)
    last_date : optional starting date. If None, inferred from daily_df or assumed 2025-12-31

    Returns
    -------
    pd.DataFrame with columns:
        ['date', 'day', 'month', 'day_of_week', 'day_name', 'is_weekend',
         'predicted_sales', 'previous_day_sales', 'previous_3day_avg', 'previous_7day_avg',
         'lower_bound', 'upper_bound']
    """
    if daily_df is None or daily_df.empty:
        raise ValueError("daily_df cannot be empty for forecasting.")

    if "daily_sales" not in daily_df.columns:
        raise ValueError("daily_df must contain 'daily_sales' column.")

    # Determine baseline series of historical sales
    history_sales = daily_df["daily_sales"].astype(float).tolist()
    if len(history_sales) < 7:
        raise ValueError("Need at least 7 days of historical sales to compute rolling averages.")

    # Determine last date
    if last_date is None:
        if "date" in daily_df.columns:
            last_date = pd.to_datetime(daily_df["date"].iloc[-1])
        elif "month" in daily_df.columns and "day" in daily_df.columns:
            m = int(daily_df["month"].iloc[-1])
            d = int(daily_df["day"].iloc[-1])
            last_date = pd.Timestamp(year=2025, month=m, day=d)
        else:
            last_date = pd.Timestamp("2025-12-31")

    forecast_records = []
    current_date = last_date
    rolling_sales = list(history_sales)

    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Simple empirical uncertainty bound based on historical standard deviation
    hist_std = float(np.std(history_sales[-30:])) if len(history_sales) >= 30 else float(np.std(history_sales))

    for step in range(1, horizon_days + 1):
        current_date = current_date + timedelta(days=1)
        day = current_date.day
        month = current_date.month
        day_of_week = current_date.weekday()
        is_weekend = 1 if day_of_week in (5, 6) else 0

        prev_day = float(rolling_sales[-1])
        prev_3avg = float(np.mean(rolling_sales[-3:]))
        prev_7avg = float(np.mean(rolling_sales[-7:]))

        feature_dict = {
            "day": day,
            "month": month,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "previous_day_sales": prev_day,
            "previous_3day_avg": prev_3avg,
            "previous_7day_avg": prev_7avg,
        }

        # Build single-row DataFrame aligned to feature_names
        row_df = pd.DataFrame([{col: feature_dict.get(col, 0.0) for col in feature_names}])

        try:
            pred_val = float(model.predict(row_df)[0])
            pred_val = max(0.0, round(pred_val, 2))
        except Exception as e:
            logger.error("Forecast prediction failed at step %d: %s", step, e)
            pred_val = round(prev_7avg, 2)

        # Feed back prediction into rolling history
        rolling_sales.append(pred_val)

        # Confidence bounds expand with forecast horizon
        uncertainty_factor = 1.0 + (step * 0.04)
        margin = hist_std * 0.75 * uncertainty_factor
        lower = max(0.0, round(pred_val - margin, 2))
        upper = round(pred_val + margin, 2)

        forecast_records.append({
            "step": step,
            "date": current_date.strftime("%Y-%m-%d"),
            "day": day,
            "month": month,
            "day_of_week": day_of_week,
            "day_name": dow_names[day_of_week],
            "is_weekend": is_weekend,
            "predicted_sales": pred_val,
            "previous_day_sales": round(prev_day, 2),
            "previous_3day_avg": round(prev_3avg, 2),
            "previous_7day_avg": round(prev_7avg, 2),
            "lower_bound": lower,
            "upper_bound": upper,
        })

    forecast_df = pd.DataFrame(forecast_records)
    logger.info("Generated %d-day recursive forecast successfully.", horizon_days)
    return forecast_df


def get_forecast_summary(forecast_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics for a generated forecast.
    """
    if forecast_df is None or forecast_df.empty:
        return {}

    total_pred = float(forecast_df["predicted_sales"].sum())
    avg_daily_pred = float(forecast_df["predicted_sales"].mean())
    max_pred = float(forecast_df["predicted_sales"].max())
    min_pred = float(forecast_df["predicted_sales"].min())
    peak_row = forecast_df.loc[forecast_df["predicted_sales"].idxmax()]
    trough_row = forecast_df.loc[forecast_df["predicted_sales"].idxmin()]

    weekend_mask = forecast_df["is_weekend"] == 1
    weekend_avg = float(forecast_df.loc[weekend_mask, "predicted_sales"].mean()) if weekend_mask.any() else 0.0
    weekday_avg = float(forecast_df.loc[~weekend_mask, "predicted_sales"].mean()) if (~weekend_mask).any() else 0.0

    return {
        "horizon_days": len(forecast_df),
        "total_predicted_revenue": round(total_pred, 2),
        "avg_daily_sales": round(avg_daily_pred, 2),
        "max_sales": round(max_pred, 2),
        "peak_date": str(peak_row["date"]),
        "peak_day_name": str(peak_row["day_name"]),
        "min_sales": round(min_pred, 2),
        "trough_date": str(trough_row["date"]),
        "trough_day_name": str(trough_row["day_name"]),
        "weekend_avg": round(weekend_avg, 2),
        "weekday_avg": round(weekday_avg, 2),
    }


def prepare_forecast_chart_data(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """
    Combine recent historical actuals and future predictions into a single
    DataFrame ready for Plotly or Altair visualization.
    """
    hist_subset = historical_df.tail(lookback_days).copy()
    if "date" not in hist_subset.columns:
        if "month" in hist_subset.columns and "day" in hist_subset.columns:
            hist_subset["date"] = pd.to_datetime(
                hist_subset.apply(lambda r: f"2025-{int(r['month']):02d}-{int(r['day']):02d}", axis=1)
            )
        else:
            hist_subset["date"] = pd.date_range(end="2025-12-31", periods=len(hist_subset))

    hist_data = pd.DataFrame({
        "date": pd.to_datetime(hist_subset["date"]),
        "actual_sales": hist_subset["daily_sales"].astype(float),
        "predicted_sales": np.nan,
        "lower_bound": np.nan,
        "upper_bound": np.nan,
        "type": "Historical Actual",
    })

    forecast_data = pd.DataFrame({
        "date": pd.to_datetime(forecast_df["date"]),
        "actual_sales": np.nan,
        "predicted_sales": forecast_df["predicted_sales"].astype(float),
        "lower_bound": forecast_df["lower_bound"].astype(float),
        "upper_bound": forecast_df["upper_bound"].astype(float),
        "type": "AI Forecast",
    })

    # Join the last actual point to the forecast for visual continuity
    bridge_row = pd.DataFrame({
        "date": [hist_data["date"].iloc[-1]],
        "actual_sales": [hist_data["actual_sales"].iloc[-1]],
        "predicted_sales": [hist_data["actual_sales"].iloc[-1]],
        "lower_bound": [hist_data["actual_sales"].iloc[-1]],
        "upper_bound": [hist_data["actual_sales"].iloc[-1]],
        "type": "AI Forecast",
    })

    combined = pd.concat([hist_data, forecast_data], ignore_index=True)
    return combined
