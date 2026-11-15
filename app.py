"""
CoffeeSense AI – Main Application Entry Point
"Predict Smarter. Brew Better. Grow Faster."
Executive Sales Dashboard & Central Navigation Hub.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

# Set page config FIRST before any other streamlit commands
st.set_page_config(
    page_title="Cafe — Sales Forecasting & Analytics",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────────────
import importlib
import components.styles
import components.kpi_cards
import components.menu_showcase

importlib.reload(components.styles)
importlib.reload(components.kpi_cards)
importlib.reload(components.menu_showcase)

from components.styles import inject_css, PLOTLY_TEMPLATE
inject_css()

# ── Imports ──────────────────────────────────────────────────────────────────
from config.settings import (
    DAILY_DATASET_PATH,
    TRANSACTIONS_DATASET_PATH,
    MODEL_PATH,
    METADATA_PATH,
    FEATURE_NAMES_PATH,
    COLORS,
)
from src.data_processing import (
    load_daily_dataset,
    load_transactions_dataset,
    clean_daily_dataset,
    clean_transactions_dataset,
    create_date_column,
)
from src.business_insights import generate_insights, get_recommendations
from components.kpi_cards import render_kpi_card
from components.menu_showcase import render_signature_menu_showcase
from components.charts import (
    daily_sales_trend,
    monthly_revenue_bar,
    weekday_weekend_comparison,
    revenue_distribution_histogram,
    transaction_volume_trend,
)

# ── Cached Data Loaders ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading CoffeeSense AI data engine…")
def load_app_data():
    daily_raw = load_daily_dataset(DAILY_DATASET_PATH)
    daily_clean = clean_daily_dataset(daily_raw)
    daily_clean = create_date_column(daily_clean)
    daily_clean["date"] = pd.to_datetime(daily_clean["date"])

    trans_raw = load_transactions_dataset(TRANSACTIONS_DATASET_PATH)
    trans_clean = clean_transactions_dataset(trans_raw)

    return daily_clean, trans_clean


daily_df, trans_df = load_app_data()

# ── Sidebar Branding & Global Filters ────────────────────────────────────────
with st.sidebar:
    side_logo_html = (
        '<div style="text-align:center; padding: 0.8rem 0.5rem 1rem 0.5rem; background: rgba(35, 18, 12, 0.7); border: 1px solid rgba(200, 149, 108, 0.3); border-radius: 16px; margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(0,0,0,0.45);">'
        '<div style="font-size:2.8rem; margin-bottom: 0.2rem; filter: drop-shadow(0 4px 12px rgba(200,149,108,0.4));">☕</div>'
        '<div style="color: #FFF8F0; font-size: 1.8rem; font-weight: 800; letter-spacing: 1.5px; line-height: 1.2;">Cafe</div>'
        '<div style="color: #D4A96A; font-size: 0.76rem; font-style: italic; margin-top: 5px; letter-spacing: 0.4px;">'
        'Predict Smarter. Brew Better. Grow Faster.</div>'
        '</div>'
    )
    st.markdown(side_logo_html, unsafe_allow_html=True)

    st.markdown("### 🎛️ Dashboard Filters")

    MONTH_MAP = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }
    month_options = ["All Months"] + [MONTH_MAP[m] for m in sorted(daily_df["month"].unique())]
    sel_month = st.selectbox("📅 Filter by Month:", month_options, index=0)

    min_date = daily_df["date"].min().date()
    max_date = daily_df["date"].max().date()
    sel_date_range = st.date_input(
        "📆 Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    day_type_sel = st.radio(
        "🏖️ Day Type:",
        ["All Days", "Weekdays Only", "Weekends Only"],
        index=0,
        horizontal=True,
    )

    st.markdown("---")
    status_box_html = (
        '<div style="background: rgba(32, 17, 11, 0.75); padding: 0.85rem; border-radius: 12px; border: 1px solid rgba(200, 149, 108, 0.28); font-size: 0.75rem; color: #D4A96A; line-height: 1.6;">'
        '<b>System Status:</b> 🟢 Operational<br>'
        '<b>Currency:</b> Indian Rupee (₹)<br>'
        '<b>Dataset:</b> 365 Days (2025)<br>'
        '<b>Model:</b> ML Regressor Active'
        '</div>'
    )
    st.markdown(status_box_html, unsafe_allow_html=True)

# ── Filter Application ───────────────────────────────────────────────────────
filtered_df = daily_df.copy()

if sel_month != "All Months":
    rev_month_map = {v: k for k, v in MONTH_MAP.items()}
    filtered_df = filtered_df[filtered_df["month"] == rev_month_map[sel_month]]

if isinstance(sel_date_range, (list, tuple)) and len(sel_date_range) == 2:
    start_d, end_d = sel_date_range
    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= start_d) & (filtered_df["date"].dt.date <= end_d)
    ]

if day_type_sel == "Weekdays Only":
    filtered_df = filtered_df[filtered_df["is_weekend"] == 0]
elif day_type_sel == "Weekends Only":
    filtered_df = filtered_df[filtered_df["is_weekend"] == 1]

if filtered_df.empty:
    st.warning("⚠️ No records match your selected filter criteria. Please broaden your selection in the sidebar.")
    st.stop()

# ── Main Header ──────────────────────────────────────────────────────────────
main_header_html = (
    '<div style="margin-bottom:1.6rem; background: linear-gradient(135deg, rgba(38, 20, 13, 0.75) 0%, rgba(18, 9, 6, 0.85) 100%); '
    'backdrop-filter: blur(16px); padding: 1.4rem 1.8rem; border-radius: 18px; border: 1px solid rgba(212, 169, 106, 0.28); box-shadow: 0 10px 30px rgba(0,0,0,0.5);">'
    '<div style="display:flex; align-items:center; gap:12px; margin-bottom: 0.2rem;">'
    '<span style="font-size:2.2rem; filter: drop-shadow(0 2px 8px rgba(200,149,108,0.4));">☕</span>'
    '<h1 style="margin:0;font-size:2.3rem;font-weight:800;color:#FFF8F0;letter-spacing:-0.5px;">Cafe — Executive Sales Dashboard</h1>'
    '</div>'
    '<p style="color:#D4A96A;font-size:1.02rem;margin-top:0.4rem;margin-bottom:0;letter-spacing:0.2px;">'
    'Predict Smarter. Brew Better. Grow Faster. — <span style="color:#FFF8F0;font-weight:600;">2025 Sales & Analytics Intelligence</span>'
    '</p>'
    '</div>'
)
st.markdown(main_header_html, unsafe_allow_html=True)

# ── Dynamic KPI Calculation ──────────────────────────────────────────────────
total_revenue = float(filtered_df["daily_sales"].sum())
avg_daily_sales = float(filtered_df["daily_sales"].mean())
max_daily_sales = float(filtered_df["daily_sales"].max())
min_daily_sales = float(filtered_df["daily_sales"].min())
total_tx = int(filtered_df["transaction_count"].sum())
avg_tx_per_day = float(filtered_df["transaction_count"].mean())

# Render 6 Luxury Glassmorphism Executive KPI Cards with Indian Rupee (₹)
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    render_kpi_card(
        title="Total Revenue",
        value=total_revenue,
        delta=f"{len(filtered_df)} Days",
        icon="💰",
        prefix="₹",
    )
with kpi2:
    render_kpi_card(
        title="Avg Daily Sales",
        value=avg_daily_sales,
        delta=f"~₹{total_revenue/total_tx:.2f}/order" if total_tx > 0 else None,
        icon="📈",
        prefix="₹",
    )
with kpi3:
    render_kpi_card(
        title="Highest Sales",
        value=max_daily_sales,
        delta="Peak Day",
        icon="🏆",
        prefix="₹",
    )
with kpi4:
    render_kpi_card(
        title="Lowest Sales",
        value=min_daily_sales,
        delta="Trough Day",
        icon="📉",
        prefix="₹",
    )
with kpi5:
    render_kpi_card(
        title="Total Orders",
        value=total_tx,
        delta="Completed",
        icon="🧾",
        prefix="",
    )
with kpi6:
    render_kpi_card(
        title="Orders / Day",
        value=round(avg_tx_per_day, 1),
        delta="Velocity",
        icon="⚡",
        prefix="",
    )

st.markdown("---")

# ── Signature Menu Showcase (Reference UI) ───────────────────────────────────
render_signature_menu_showcase()

st.markdown("---")

# ── Visual Analytics Section ─────────────────────────────────────────────────
st.markdown("### 📈 Revenue Dynamics & Volume Patterns")

row1_col1, row1_col2 = st.columns([3, 2])

with row1_col1:
    st.markdown("#### 📅 Daily Sales Trend & 7-Day Moving Average")
    fig_daily = daily_sales_trend(filtered_df)
    st.plotly_chart(fig_daily, use_container_width=True)

with row1_col2:
    st.markdown("#### 📆 Monthly Revenue Breakdown")
    fig_month = monthly_revenue_bar(filtered_df)
    st.plotly_chart(fig_month, use_container_width=True)

row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])

with row2_col1:
    st.markdown("#### 🏖️ Weekend vs Weekday")
    fig_weekend = weekday_weekend_comparison(filtered_df)
    st.plotly_chart(fig_weekend, use_container_width=True)

with row2_col2:
    st.markdown("#### 📊 Sales Distribution")
    fig_dist = revenue_distribution_histogram(filtered_df)
    st.plotly_chart(fig_dist, use_container_width=True)

with row2_col3:
    st.markdown("#### 🧾 Transaction Volume Trend")
    fig_tx = transaction_volume_trend(filtered_df)
    st.plotly_chart(fig_tx, use_container_width=True)

st.markdown("---")

# ── Automated Executive Insights & Business Recommendations ──────────────────
st.markdown("### 💡 AI Executive Insights & Operational Recommendations")

insights = generate_insights(filtered_df, trans_df)
recommendations = get_recommendations(insights)

rec_col1, rec_col2 = st.columns([1, 1])

with rec_col1:
    st.markdown("#### 🔍 Key Empirical Observations")
    uplift = insights.get("weekend_uplift_pct", 0)
    best_m = insights.get("best_month", "N/A")
    worst_m = insights.get("worst_month", "N/A")
    best_dow = insights.get("best_day_of_week", "N/A")
    worst_dow = insights.get("worst_day_of_week", "N/A")
    card_share = insights.get("card_share_pct", 0)
    cash_share = insights.get("cash_share_pct", 0)
    cv = insights.get("sales_volatility_cv", 0)

    obs_html = (
        f'<div style="background: linear-gradient(135deg, rgba(38, 20, 13, 0.7) 0%, rgba(20, 10, 6, 0.85) 100%); backdrop-filter: blur(14px); padding: 1.3rem; border-radius: 16px; border: 1px solid rgba(212, 169, 106, 0.22); border-left: 4px solid #C8956C; color: #F5F0E8; line-height: 1.8;">'
        f'• <b>Weekend Demand Surge:</b> Weekends produce <b>{uplift:+.1f}%</b> higher daily sales than weekdays.<br>'
        f'• <b>Peak & Trough Days:</b> <b>{best_dow}</b> is the top-earning day of the week, while <b>{worst_dow}</b> sees the lowest foot traffic.<br>'
        f'• <b>Seasonal Peak:</b> <b>{best_m}</b> generated the highest monthly revenue (₹{insights.get("best_month_revenue", 0):,.2f}), while <b>{worst_m}</b> was slowest.<br>'
        f'• <b>Payment Habits:</b> Digital & card payments account for <b>{card_share:.1f}%</b> of volume, with cash at <b>{cash_share:.1f}%</b>.<br>'
        f'• <b>Revenue Stability:</b> Daily revenue demonstrates a coefficient of variation of <b>{cv:.1f}%</b>.'
        f'</div>'
    )
    st.markdown(obs_html, unsafe_allow_html=True)

with rec_col2:
    st.markdown("#### 🎯 Prioritized Action Plan")
    for rec in recommendations[:3]:
        rec_html = (
            f'<div style="background: linear-gradient(135deg, rgba(32, 17, 11, 0.7) 0%, rgba(18, 9, 6, 0.82) 100%); backdrop-filter: blur(14px); padding: 1rem; border-radius: 14px; border: 1px solid rgba(200, 149, 108, 0.25); margin-bottom: 0.65rem;">'
            f'<div style="color:#FFF8F0; font-weight:700; font-size:0.95rem;">'
            f'{rec["icon"]} {rec["title"]}'
            f'<span style="float:right; background: linear-gradient(135deg, #C8956C, #D4A96A); color:#120703; font-size:0.7rem; padding:2px 10px; border-radius:12px; font-weight:800;">{rec["priority"]}</span>'
            f'</div>'
            f'<div style="color:#D4A96A; font-size:0.83rem; margin-top:5px; line-height:1.5;">{rec["detail"]}</div>'
            f'</div>'
        )
        st.markdown(rec_html, unsafe_allow_html=True)

st.markdown("---")

# ── Module Quick Navigation Cards ────────────────────────────────────────────
st.markdown("### 🧭 Explore System Modules")
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)

def _make_nav_card(icon: str, title: str, subtitle: str) -> str:
    return (
        f'<div style="background: linear-gradient(135deg, rgba(38, 20, 13, 0.7) 0%, rgba(20, 10, 6, 0.85) 100%); '
        f'padding: 1.1rem; border-radius: 16px; border: 1px solid rgba(200,149,108,0.25); text-align: center; '
        f'box-shadow: 0 6px 20px rgba(0,0,0,0.4);">'
        f'<div style="font-size: 2rem;">{icon}</div>'
        f'<div style="font-weight: 700; color: #FFF8F0; margin: 6px 0 3px 0;">{title}</div>'
        f'<div style="font-size: 0.75rem; color: #D4A96A;">{subtitle}</div>'
        f'</div>'
    )

with nav_col1:
    st.markdown(_make_nav_card("📊", "Analytics", "Hourly rushes, products, and payment charts."), unsafe_allow_html=True)
with nav_col2:
    st.markdown(_make_nav_card("🤖", "AI Assistant", "Ask sales questions and get instant insights."), unsafe_allow_html=True)
with nav_col3:
    st.markdown(_make_nav_card("📂", "Dataset Explorer", "Audit 365 daily rows and 6,592 order lines."), unsafe_allow_html=True)
with nav_col4:
    st.markdown(_make_nav_card("🔮", "Prediction Workspace", "Simulate scenarios & 7/30-day recursive forecasts."), unsafe_allow_html=True)
with nav_col5:
    st.markdown(_make_nav_card("⚙️", "Model Center", "Inspect R², RMSE, weights & retrain locally."), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "☕ Cafe v1.0.0 | Currency: Indian Rupee (₹) | Synthetic Benchmark Dataset (2025) | Built with Streamlit, Scikit-Learn & Plotly."
)
