"""
CoffeeSense AI – eda.py
Exploratory Data Analysis computation module.
All functions return plain Python objects (dicts, DataFrames, lists).
Chart rendering is handled by components/charts.py.
"""

from __future__ import annotations

import calendar
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(daily_df: pd.DataFrame, trans_df: pd.DataFrame) -> dict[str, Any]:
    """Compute high-level summary statistics across both datasets.

    Returns
    -------
    dict with keys:
        total_revenue, avg_daily_revenue, max_daily_sales, min_daily_sales,
        total_transactions, avg_transactions_per_day,
        revenue_std, revenue_median
    """
    money_col = _money_col(trans_df)

    total_revenue: float = float(daily_df["daily_sales"].sum())
    avg_daily_revenue: float = float(daily_df["daily_sales"].mean())
    max_daily_sales: float = float(daily_df["daily_sales"].max())
    min_daily_sales: float = float(daily_df["daily_sales"].min())
    revenue_std: float = float(daily_df["daily_sales"].std())
    revenue_median: float = float(daily_df["daily_sales"].median())

    total_transactions: int = int(len(trans_df))
    tx_per_day = trans_df.groupby(_date_col(trans_df)).size() if _date_col(trans_df) else pd.Series()
    avg_transactions_per_day: float = float(tx_per_day.mean()) if not tx_per_day.empty else float(
        total_transactions / max(len(daily_df), 1)
    )

    return {
        "total_revenue": total_revenue,
        "avg_daily_revenue": avg_daily_revenue,
        "max_daily_sales": max_daily_sales,
        "min_daily_sales": min_daily_sales,
        "total_transactions": total_transactions,
        "avg_transactions_per_day": avg_transactions_per_day,
        "revenue_std": revenue_std,
        "revenue_median": revenue_median,
    }


# ---------------------------------------------------------------------------
# Monthly stats
# ---------------------------------------------------------------------------

def compute_monthly_stats(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of monthly revenue totals and averages.

    Returns
    -------
    pd.DataFrame with columns:
        month, month_name, total_revenue, avg_daily_revenue, num_days,
        max_daily, min_daily
    """
    _df = daily_df.copy()
    if "month" not in _df.columns:
        raise ValueError("daily_df must contain a 'month' column.")

    grouped = _df.groupby("month")["daily_sales"].agg(
        total_revenue="sum",
        avg_daily_revenue="mean",
        num_days="count",
        max_daily="max",
        min_daily="min",
    ).reset_index()

    grouped["month_name"] = grouped["month"].apply(
        lambda m: calendar.month_name[int(m)]
    )

    return grouped[[
        "month", "month_name", "total_revenue", "avg_daily_revenue",
        "num_days", "max_daily", "min_daily",
    ]]


# ---------------------------------------------------------------------------
# Weekday stats
# ---------------------------------------------------------------------------

def compute_weekday_stats(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of sales aggregated by day of week.

    Returns
    -------
    pd.DataFrame with columns:
        day_of_week, weekday_name, avg_sales, total_sales,
        num_days, max_sales, min_sales
    """
    _df = daily_df.copy()
    if "day_of_week" not in _df.columns:
        raise ValueError("daily_df must contain a 'day_of_week' column (0=Mon … 6=Sun).")

    day_map = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
        4: "Friday", 5: "Saturday", 6: "Sunday",
    }

    grouped = _df.groupby("day_of_week")["daily_sales"].agg(
        avg_sales="mean",
        total_sales="sum",
        num_days="count",
        max_sales="max",
        min_sales="min",
    ).reset_index()

    grouped["weekday_name"] = grouped["day_of_week"].map(day_map)
    return grouped[[
        "day_of_week", "weekday_name", "avg_sales", "total_sales",
        "num_days", "max_sales", "min_sales",
    ]].sort_values("day_of_week")


# ---------------------------------------------------------------------------
# Weekend vs weekday comparison
# ---------------------------------------------------------------------------

def compute_weekend_stats(daily_df: pd.DataFrame) -> dict[str, Any]:
    """Compare weekend vs weekday sales performance.

    Returns
    -------
    dict with keys:
        weekend_avg, weekday_avg, weekend_total, weekday_total,
        weekend_days, weekday_days, weekend_premium_pct
    """
    _df = daily_df.copy()
    if "is_weekend" not in _df.columns:
        raise ValueError("daily_df must contain an 'is_weekend' column.")

    weekend = _df[_df["is_weekend"] == 1]["daily_sales"]
    weekday = _df[_df["is_weekend"] == 0]["daily_sales"]

    weekend_avg = float(weekend.mean()) if not weekend.empty else 0.0
    weekday_avg = float(weekday.mean()) if not weekday.empty else 0.0
    premium_pct = (
        ((weekend_avg - weekday_avg) / weekday_avg) * 100
        if weekday_avg != 0 else 0.0
    )

    return {
        "weekend_avg": weekend_avg,
        "weekday_avg": weekday_avg,
        "weekend_total": float(weekend.sum()),
        "weekday_total": float(weekday.sum()),
        "weekend_days": int(len(weekend)),
        "weekday_days": int(len(weekday)),
        "weekend_premium_pct": round(premium_pct, 2),
    }


# ---------------------------------------------------------------------------
# Product stats
# ---------------------------------------------------------------------------

def compute_product_stats(trans_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-product revenue, order count, and average price.

    Returns
    -------
    pd.DataFrame with columns:
        coffee_name, total_revenue, count, avg_price
    Sorted by total_revenue descending.
    """
    _df = trans_df.copy()
    name_col = "coffee_name" if "coffee_name" in _df.columns else _df.columns[0]
    money_col = _money_col(_df)

    grouped = _df.groupby(name_col)[money_col].agg(
        total_revenue="sum",
        count="count",
        avg_price="mean",
    ).reset_index().rename(columns={name_col: "coffee_name"})

    return grouped.sort_values("total_revenue", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Payment stats
# ---------------------------------------------------------------------------

def compute_payment_stats(trans_df: pd.DataFrame) -> dict[str, Any]:
    """Return cash vs card revenue totals and counts.

    Returns
    -------
    dict with per-method 'total' and 'count' sub-dicts, plus 'overall'.
    """
    _df = trans_df.copy()
    pay_col = next(
        (c for c in ["payment_method", "payment", "pay_method"] if c in _df.columns),
        None,
    )
    if pay_col is None:
        raise ValueError("No payment method column found.")

    money_col = _money_col(_df)
    grouped = _df.groupby(pay_col).agg(
        total=(money_col, "sum"),
        count=(money_col, "count"),
    )

    result: dict[str, Any] = {}
    for method, row in grouped.iterrows():
        result[str(method)] = {
            "total": float(row["total"]),
            "count": int(row["count"]),
        }
    result["overall"] = {
        "total": float(_df[money_col].sum()),
        "count": int(len(_df)),
    }
    return result


# ---------------------------------------------------------------------------
# Hourly stats
# ---------------------------------------------------------------------------

def compute_hourly_stats(trans_df: pd.DataFrame) -> pd.DataFrame:
    """Return transaction count per hour of day (0–23).

    Returns
    -------
    pd.DataFrame with columns: hour, count, revenue
    """
    _df = trans_df.copy()
    hour_col = "hour" if "hour" in _df.columns else None

    if hour_col is None:
        dt_col = next(
            (c for c in _df.columns if "datetime" in c.lower() or "time" in c.lower()),
            None,
        )
        if dt_col:
            _df["hour"] = pd.to_datetime(_df[dt_col], errors="coerce").dt.hour
            hour_col = "hour"
        else:
            raise ValueError("No 'hour' or datetime column found in trans_df.")

    money_col = _money_col(_df)
    grouped = _df.groupby(hour_col).agg(
        count=(money_col, "count"),
        revenue=(money_col, "sum"),
    ).reset_index().rename(columns={hour_col: "hour"})

    # Fill missing hours
    full = pd.DataFrame({"hour": range(24)})
    result = full.merge(grouped, on="hour", how="left").fillna(0)
    result["count"] = result["count"].astype(int)
    return result


# ---------------------------------------------------------------------------
# Top products
# ---------------------------------------------------------------------------

def get_top_products(trans_df: pd.DataFrame, n: int = 5) -> list[str]:
    """Return the top-N product names by total revenue."""
    stats = compute_product_stats(trans_df)
    return stats["coffee_name"].head(n).tolist()


# ---------------------------------------------------------------------------
# Best month
# ---------------------------------------------------------------------------

def get_best_month(daily_df: pd.DataFrame) -> str:
    """Return the name of the month with highest total revenue."""
    monthly = compute_monthly_stats(daily_df)
    best_row = monthly.loc[monthly["total_revenue"].idxmax()]
    return str(best_row["month_name"])


# ---------------------------------------------------------------------------
# Peak hour
# ---------------------------------------------------------------------------

def get_peak_hour(trans_df: pd.DataFrame) -> int:
    """Return the hour (0–23) with the most transactions."""
    hourly = compute_hourly_stats(trans_df)
    return int(hourly.loc[hourly["count"].idxmax(), "hour"])


# ---------------------------------------------------------------------------
# Auto-generated insights
# ---------------------------------------------------------------------------

def compute_insights(daily_df: pd.DataFrame, trans_df: pd.DataFrame) -> list[str]:
    """Generate 5–7 actionable business insight strings from the data.

    Returns
    -------
    list of str (each is one insight sentence suitable for display).
    """
    insights: list[str] = []

    # 1. Best month
    try:
        best_month = get_best_month(daily_df)
        monthly = compute_monthly_stats(daily_df)
        best_rev = float(monthly.loc[monthly["month_name"] == best_month, "total_revenue"].iloc[0])
        insights.append(
            f"☕ **{best_month}** is your strongest month with ₹{best_rev:,.0f} in total revenue."
        )
    except Exception:
        pass

    # 2. Weekend premium
    try:
        ws = compute_weekend_stats(daily_df)
        if ws["weekend_premium_pct"] > 0:
            insights.append(
                f"📈 Weekends generate **{ws['weekend_premium_pct']:.1f}% more revenue** on average "
                f"(₹{ws['weekend_avg']:,.0f}/day) compared to weekdays (₹{ws['weekday_avg']:,.0f}/day)."
            )
        else:
            insights.append(
                f"📉 Weekdays outperform weekends by **{abs(ws['weekend_premium_pct']):.1f}%** on average. "
                "Consider weekend promotions."
            )
    except Exception:
        pass

    # 3. Top product
    try:
        top = get_top_products(trans_df, n=1)
        if top:
            stats = compute_product_stats(trans_df)
            top_rev = float(stats.loc[stats["coffee_name"] == top[0], "total_revenue"].iloc[0])
            insights.append(
                f"🏆 **{top[0]}** is your best-selling product, generating ₹{top_rev:,.0f} in revenue."
            )
    except Exception:
        pass

    # 4. Peak hour
    try:
        peak_h = get_peak_hour(trans_df)
        hourly = compute_hourly_stats(trans_df)
        peak_count = int(hourly.loc[hourly["hour"] == peak_h, "count"].iloc[0])
        insights.append(
            f"⏰ Peak sales hour is **{peak_h}:00** with {peak_count:,} transactions — "
            "ensure adequate staffing during this window."
        )
    except Exception:
        pass

    # 5. Payment method
    try:
        ps = compute_payment_stats(trans_df)
        totals = {k: v["total"] for k, v in ps.items() if k != "overall"}
        if totals:
            dominant = max(totals, key=totals.get)
            dominant_pct = totals[dominant] / ps["overall"]["total"] * 100
            insights.append(
                f"💳 **{dominant}** is the dominant payment method accounting for "
                f"{dominant_pct:.1f}% of revenue."
            )
    except Exception:
        pass

    # 6. Revenue volatility
    try:
        cv = float(daily_df["daily_sales"].std() / daily_df["daily_sales"].mean() * 100)
        if cv > 20:
            insights.append(
                f"⚠️ Revenue shows high day-to-day variability (CV = {cv:.1f}%). "
                "Predictive models can help smooth planning."
            )
        else:
            insights.append(
                f"✅ Revenue is relatively stable with a coefficient of variation of {cv:.1f}%, "
                "indicating consistent customer demand."
            )
    except Exception:
        pass

    # 7. Avg transaction value
    try:
        money_col = _money_col(trans_df)
        avg_tx = float(trans_df[money_col].mean())
        insights.append(
            f"💰 Average transaction value is **₹{avg_tx:.2f}** — consider upsell opportunities "
            "to push this above the current benchmark."
        )
    except Exception:
        pass

    return insights[:7]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _money_col(df: pd.DataFrame) -> str:
    """Infer the revenue/money column name."""
    for candidate in ["money", "revenue", "total", "amount", "daily_sales"]:
        if candidate in df.columns:
            return candidate
    # Fall back to first numeric column
    nums = df.select_dtypes(include="number").columns
    if len(nums) > 0:
        return nums[0]
    raise ValueError("Cannot identify a money/revenue column in the DataFrame.")


def _date_col(df: pd.DataFrame) -> str | None:
    """Infer the date column name, or return None."""
    for candidate in ["date", "Date", "transaction_date", "datetime"]:
        if candidate in df.columns:
            return candidate
    return None
