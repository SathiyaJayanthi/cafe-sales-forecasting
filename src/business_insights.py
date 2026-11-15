"""
CoffeeSense AI - Business Insights & Recommendations Engine
Generates automated, actionable business intelligence and operational recommendations
from daily sales and raw transaction data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MONTH_MAP = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

DOW_MAP = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday",
}


def generate_insights(
    daily_df: pd.DataFrame,
    trans_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Generate structured analytical insights from daily and transaction data.

    Returns
    -------
    dict with metrics, trends, peak drivers, and operational indicators.
    """
    if daily_df is None or daily_df.empty:
        return {}

    sales = daily_df["daily_sales"].astype(float)
    total_revenue = float(sales.sum())
    avg_daily_sales = float(sales.mean())
    max_sales = float(sales.max())
    min_sales = float(sales.min())
    std_sales = float(sales.std())
    cv = (std_sales / avg_daily_sales * 100) if avg_daily_sales > 0 else 0.0

    # Best and worst days
    max_idx = sales.idxmax()
    min_idx = sales.idxmin()
    max_row = daily_df.loc[max_idx]
    min_row = daily_df.loc[min_idx]

    # Weekend vs Weekday
    weekend_mask = daily_df["is_weekend"] == 1
    weekend_avg = float(daily_df.loc[weekend_mask, "daily_sales"].mean()) if weekend_mask.any() else 0.0
    weekday_avg = float(daily_df.loc[~weekend_mask, "daily_sales"].mean()) if (~weekend_mask).any() else 0.0
    weekend_uplift_pct = (
        ((weekend_avg - weekday_avg) / weekday_avg * 100) if weekday_avg > 0 else 0.0
    )

    # Monthly performance
    monthly_sales = daily_df.groupby("month")["daily_sales"].sum()
    best_month_num = int(monthly_sales.idxmax())
    worst_month_num = int(monthly_sales.idxmin())
    best_month_name = MONTH_MAP.get(best_month_num, f"Month {best_month_num}")
    worst_month_name = MONTH_MAP.get(worst_month_num, f"Month {worst_month_num}")
    best_month_rev = float(monthly_sales.max())
    worst_month_rev = float(monthly_sales.min())

    # Day of week performance
    dow_sales = daily_df.groupby("day_of_week")["daily_sales"].mean()
    best_dow_num = int(dow_sales.idxmax())
    worst_dow_num = int(dow_sales.idxmin())
    best_dow_name = DOW_MAP.get(best_dow_num, f"Day {best_dow_num}")
    worst_dow_name = DOW_MAP.get(worst_dow_num, f"Day {worst_dow_num}")

    # Payment distribution (from daily or transaction)
    cash_total = float(daily_df["cash_transactions"].sum()) if "cash_transactions" in daily_df.columns else 0.0
    card_total = float(daily_df["card_transactions"].sum()) if "card_transactions" in daily_df.columns else 0.0
    total_tx = cash_total + card_total
    card_pct = (card_total / total_tx * 100) if total_tx > 0 else 0.0
    cash_pct = (cash_total / total_tx * 100) if total_tx > 0 else 0.0

    # Transaction-level metrics
    top_products_by_vol: List[Dict[str, Any]] = []
    top_products_by_rev: List[Dict[str, Any]] = []
    peak_hour = None
    peak_hour_tx = 0

    if trans_df is not None and not trans_df.empty:
        # Coffee products
        prod_col = next((c for c in trans_df.columns if "coffee" in c.lower() or "product" in c.lower()), None)
        money_col = next((c for c in trans_df.columns if "money" in c.lower() or "price" in c.lower() or "amount" in c.lower()), None)

        if prod_col:
            vol_counts = trans_df[prod_col].value_counts().head(5)
            for prod, count in vol_counts.items():
                top_products_by_vol.append({"product": str(prod), "volume": int(count)})

            if money_col:
                rev_series = trans_df.groupby(prod_col)[money_col].sum().sort_values(ascending=False).head(5)
                for prod, rev in rev_series.items():
                    top_products_by_rev.append({"product": str(prod), "revenue": round(float(rev), 2)})

        # Hourly distribution
        time_col = next((c for c in trans_df.columns if "datetime" in c.lower() or "time" in c.lower()), None)
        if time_col and pd.api.types.is_datetime64_any_dtype(trans_df[time_col]):
            hours = trans_df[time_col].dt.hour
            hour_counts = hours.value_counts()
            if not hour_counts.empty:
                peak_hour = int(hour_counts.idxmax())
                peak_hour_tx = int(hour_counts.max())

    insights = {
        "total_revenue": round(total_revenue, 2),
        "avg_daily_sales": round(avg_daily_sales, 2),
        "max_sales": round(max_sales, 2),
        "min_sales": round(min_sales, 2),
        "sales_volatility_cv": round(cv, 1),
        "best_month": best_month_name,
        "best_month_revenue": round(best_month_rev, 2),
        "worst_month": worst_month_name,
        "worst_month_revenue": round(worst_month_rev, 2),
        "best_day_of_week": best_dow_name,
        "worst_day_of_week": worst_dow_name,
        "weekend_avg": round(weekend_avg, 2),
        "weekday_avg": round(weekday_avg, 2),
        "weekend_uplift_pct": round(weekend_uplift_pct, 1),
        "card_share_pct": round(card_pct, 1),
        "cash_share_pct": round(cash_pct, 1),
        "total_transactions": int(total_tx) if total_tx > 0 else len(trans_df) if trans_df is not None else 0,
        "top_products_by_volume": top_products_by_vol,
        "top_products_by_revenue": top_products_by_rev,
        "peak_hour": peak_hour,
        "peak_hour_transactions": peak_hour_tx,
    }
    return insights


def get_recommendations(insights: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generate actionable business recommendations based on calculated insights.
    """
    recs: List[Dict[str, str]] = []

    # 1. Weekend vs Weekday staffing
    uplift = insights.get("weekend_uplift_pct", 0)
    best_dow = insights.get("best_day_of_week", "Saturday")
    worst_dow = insights.get("worst_day_of_week", "Monday")

    if uplift > 0:
        recs.append({
            "category": "Staffing & Labor Optimization",
            "icon": "👥",
            "title": f"Expand Barista Shift on {best_dow}s & Weekends",
            "detail": (
                f"Weekend daily sales are {abs(uplift):.1f}% higher than weekdays. "
                f"Schedule an extra barista and pre-calibrate grinders on Friday afternoon to avoid bottlenecking."
            ),
            "priority": "High",
        })
    else:
        recs.append({
            "category": "Staffing & Labor Optimization",
            "icon": "👥",
            "title": f"Rebalance Shift Hours for Weekday Demand",
            "detail": (
                f"Weekday demand remains robust relative to weekends. Maintain steady morning staffing through {best_dow}."
            ),
            "priority": "Medium",
        })

    # 2. Midweek / Slowest Day Campaign
    recs.append({
        "category": "Revenue Growth & Promotions",
        "icon": "🎯",
        "title": f"Launch Midweek 'Double Stamp' or Pastry Bundling on {worst_dow}s",
        "detail": (
            f"{worst_dow} generates the lowest average daily revenue. Introduce a loyalty double-points day "
            f"or a morning coffee + pastry discount to stimulate customer footfall."
        ),
        "priority": "High",
    })

    # 3. Peak hour prep
    peak_h = insights.get("peak_hour")
    if peak_h is not None:
        peak_formatted = f"{peak_h:02d}:00 - {peak_h+1:02d}:00"
        recs.append({
            "category": "Operational Speed",
            "icon": "⚡",
            "title": f"Batch Prep Ahead of Peak Window ({peak_formatted})",
            "detail": (
                f"The highest rush of the day occurs around {peak_formatted}. "
                f"Pre-fill milk jugs, prep cold brews, and stage cup lids by {peak_h-1:02d}:45 to maintain sub-90s ticket times."
            ),
            "priority": "High",
        })

    # 4. Product inventory focus
    top_vol = insights.get("top_products_by_volume", [])
    if top_vol:
        top_name = top_vol[0]["product"]
        recs.append({
            "category": "Supply Chain & Inventory",
            "icon": "☕",
            "title": f"Protect Bean & Milk Inventory for {top_name}",
            "detail": (
                f"'{top_name}' is your #1 bestseller by volume. Maintain minimum safety stock of single-origin and oat milk "
                f"to prevent 86'ing your top revenue driver."
            ),
            "priority": "Medium",
        })

    # 5. Payment terminal efficiency
    card_pct = insights.get("card_share_pct", 0)
    if card_pct > 60:
        recs.append({
            "category": "Point of Sale & Payments",
            "icon": "💳",
            "title": "Enable Tap-to-Pay & Dual Contactless Terminals",
            "detail": (
                f"{card_pct:.1f}% of transactions are card/digital. Add a secondary tap-to-pay terminal or QR kiosk "
                f"to halve checkout queues during morning surges."
            ),
            "priority": "Medium",
        })

    # 6. Seasonality & Holiday Planning
    best_m = insights.get("best_month", "December")
    worst_m = insights.get("worst_month", "January")
    recs.append({
        "category": "Seasonal Strategy",
        "icon": "📅",
        "title": f"Capitalize on Peak in {best_m} & Counteract Slump in {worst_m}",
        "detail": (
            f"Revenue peaked in {best_m} (₹{insights.get('best_month_revenue', 0):,.0f}) but dipped in {worst_m} "
            f"(₹{insights.get('worst_month_revenue', 0):,.0f}). Roll out subscription passes or winter warmers in {worst_m}."
        ),
        "priority": "Medium",
    })

    return recs
