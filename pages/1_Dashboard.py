"""
CoffeeSense AI – 1_Dashboard.py
Executive Sales Dashboard: KPIs, trend charts, and key insights.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import importlib
import components.styles
import components.menu_showcase
importlib.reload(components.styles)
importlib.reload(components.menu_showcase)

from components.styles import inject_css, PLOTLY_TEMPLATE
from components.menu_showcase import render_signature_menu_showcase
from config.settings import (
    DAILY_DATASET_PATH,
    TRANSACTIONS_DATASET_PATH,
    COLORS,
)
from src.data_processing import (
    load_daily_dataset,
    load_transactions_dataset,
    clean_daily_dataset,
    clean_transactions_dataset,
    create_date_column,
)

# Optional imports – degrade gracefully
try:
    from src.model_training import load_model_artifacts
    _model_training_available = True
except ImportError:
    _model_training_available = False

try:
    from src import eda
    _eda_available = True
except ImportError:
    _eda_available = False

# ---------------------------------------------------------------------------
# Inject CSS
# ---------------------------------------------------------------------------
inject_css()

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading daily sales data…")
def get_daily_df() -> pd.DataFrame:
    df = load_daily_dataset(DAILY_DATASET_PATH)
    df = clean_daily_dataset(df)
    df = create_date_column(df)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner="Loading transactions data…")
def get_trans_df() -> pd.DataFrame:
    df = load_transactions_dataset(TRANSACTIONS_DATASET_PATH)
    df = clean_transactions_dataset(df)
    return df


@st.cache_resource(show_spinner="Loading model…")
def get_model_artifacts():
    if not _model_training_available:
        return None
    try:
        return load_model_artifacts()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("# ☕ Cafe — Executive Sales Dashboard")
st.markdown(
    "<p style='color:#C8956C;margin-top:-10px;'>Predict Smarter. Brew Better. Grow Faster. — 2025 Sales & Analytics Intelligence</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data_ok = True
try:
    daily_df = get_daily_df()
    trans_df = get_trans_df()
except Exception as exc:
    st.error(
        f"❌ **Could not load data.** Make sure both CSV files exist in `data/raw/` "
        f"and that the data processing pipeline has been run.\n\n*Details: {exc}*"
    )
    data_ok = False

if not data_ok:
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

with st.sidebar:
    st.markdown("## 🎛️ Filters")

    month_options = ["All"] + [MONTH_NAMES[m] for m in sorted(daily_df["month"].unique())]
    selected_month_name = st.selectbox("📅 Month", month_options, index=0)

    date_min = daily_df["date"].min().date()
    date_max = daily_df["date"].max().date()
    date_range = st.date_input(
        "📆 Date Range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    st.divider()
    st.markdown(
        "<div class='sidebar-footer'>Cafe v1.0.0</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
filtered = daily_df.copy()

if selected_month_name != "All":
    month_num = {v: k for k, v in MONTH_NAMES.items()}[selected_month_name]
    filtered = filtered[filtered["month"] == month_num]

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered["date"].dt.date >= start_date) &
        (filtered["date"].dt.date <= end_date)
    ]

if filtered.empty:
    st.warning("⚠️ No data matches the selected filters. Adjust the sidebar filters.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
st.markdown("## 📌 Key Performance Indicators")

total_revenue = filtered["daily_sales"].sum()
avg_daily_sales = filtered["daily_sales"].mean()
highest_day = filtered["daily_sales"].max()
lowest_day = filtered["daily_sales"].min()

has_txn = "transaction_count" in filtered.columns
total_txn = int(filtered["transaction_count"].sum()) if has_txn else "N/A"
avg_txn = round(filtered["transaction_count"].mean(), 1) if has_txn else "N/A"

# Period-over-period delta (compare to same-length prior period)
prior_days = len(filtered)
prior_df = daily_df[~daily_df.index.isin(filtered.index)].tail(prior_days)
rev_delta = None
if not prior_df.empty:
    prior_rev = prior_df["daily_sales"].sum()
    rev_delta = f"{'▲' if total_revenue >= prior_rev else '▼'} vs prior period"

kpi_cols = st.columns(6)
kpi_cols[0].metric("💰 Total Revenue", f"₹{total_revenue:,.0f}", delta=rev_delta)
kpi_cols[1].metric("📈 Avg Daily Sales", f"₹{avg_daily_sales:,.0f}")
kpi_cols[2].metric("🏆 Highest Day", f"₹{highest_day:,.0f}")
kpi_cols[3].metric("📉 Lowest Day", f"₹{lowest_day:,.0f}")
kpi_cols[4].metric("🧾 Total Orders", f"{total_txn:,}" if isinstance(total_txn, int) else total_txn)
kpi_cols[5].metric("📋 Avg Txn/Day", f"{avg_txn}" if isinstance(avg_txn, (int, float)) else avg_txn)

st.divider()

# ---------------------------------------------------------------------------
# Signature Café Items & Bestseller Showcase
# ---------------------------------------------------------------------------
render_signature_menu_showcase(trans_df)

# ---------------------------------------------------------------------------
# Charts – tabbed layout
# ---------------------------------------------------------------------------
st.markdown("## 📊 Charts & Analytics")

tab_trends, tab_revenue, tab_patterns, tab_dist = st.tabs(
    ["📈 Trends", "📊 Revenue", "📅 Patterns", "📉 Distribution"]
)

# ── Tab 1: Trends ────────────────────────────────────────────────────────────
with tab_trends:
    col1, col2 = st.columns(1), None  # single-col layout

    # Daily Sales Trend
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=filtered["date"],
        y=filtered["daily_sales"],
        mode="lines+markers",
        name="Daily Sales",
        line=dict(color=COLORS["caramel"], width=2.5),
        marker=dict(color=COLORS["latte"], size=5),
        hovertemplate="<b>%{x|%b %d}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
    ))
    # 7-day rolling average
    if len(filtered) >= 7:
        rolling = filtered["daily_sales"].rolling(7, center=True).mean()
        fig_trend.add_trace(go.Scatter(
            x=filtered["date"],
            y=rolling,
            mode="lines",
            name="7-Day Avg",
            line=dict(color=COLORS["accent"], width=2, dash="dot"),
            hovertemplate="7-Day Avg: ₹%{y:,.0f}<extra></extra>",
        ))
    fig_trend.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title="Daily Sales Revenue",
        xaxis_title="Date",
        yaxis_title="Revenue (₹)",
        height=380,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Transaction Volume Trend
    if has_txn:
        fig_txn = go.Figure()
        fig_txn.add_trace(go.Scatter(
            x=filtered["date"],
            y=filtered["transaction_count"],
            mode="lines+markers",
            name="Transactions",
            line=dict(color=COLORS["mocha"], width=2.5),
            marker=dict(color=COLORS["caramel"], size=5),
            fill="tozeroy",
            fillcolor="rgba(107,58,42,0.2)",
            hovertemplate="<b>%{x|%b %d}</b><br>Transactions: %{y}<extra></extra>",
        ))
        fig_txn.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title="Daily Transaction Volume",
            xaxis_title="Date",
            yaxis_title="Transactions",
            height=330,
        )
        st.plotly_chart(fig_txn, use_container_width=True)
    else:
        st.info("Transaction count column not available in dataset.")

# ── Tab 2: Revenue ───────────────────────────────────────────────────────────
with tab_revenue:
    # Monthly Revenue bar chart
    monthly = filtered.groupby("month")["daily_sales"].sum().reset_index()
    monthly["month_name"] = monthly["month"].map(MONTH_NAMES)

    fig_monthly = go.Figure(go.Bar(
        x=monthly["month_name"],
        y=monthly["daily_sales"],
        marker_color=COLORS["caramel"],
        marker_line_color=COLORS["latte"],
        marker_line_width=1,
        hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
    ))
    fig_monthly.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title="Monthly Revenue Breakdown",
        xaxis_title="Month",
        yaxis_title="Revenue (₹)",
        height=380,
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Top 5 best days
    st.markdown("### 🏆 Top 5 Best Sales Days")
    top5 = (
        filtered.nlargest(5, "daily_sales")[["date", "day_of_week", "daily_sales"]]
        .copy()
    )
    top5["date"] = top5["date"].dt.strftime("%B %d, %Y")
    top5["day_of_week"] = top5["day_of_week"].map(
        {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
         4: "Friday", 5: "Saturday", 6: "Sunday"}
    )
    top5["daily_sales"] = top5["daily_sales"].apply(lambda x: f"₹{x:,.2f}")
    top5.columns = ["Date", "Day", "Revenue"]
    top5 = top5.reset_index(drop=True)
    top5.index = top5.index + 1
    st.dataframe(top5, use_container_width=True)

# ── Tab 3: Patterns ──────────────────────────────────────────────────────────
with tab_patterns:
    if "is_weekend" in filtered.columns:
        wkd = filtered.groupby("is_weekend")["daily_sales"].mean().reset_index()
        wkd["label"] = wkd["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        wkd["color"] = wkd["is_weekend"].map({0: COLORS["caramel"], 1: COLORS["mocha"]})

        fig_wkd = go.Figure(go.Bar(
            x=wkd["label"],
            y=wkd["daily_sales"],
            marker_color=wkd["color"].tolist(),
            hovertemplate="<b>%{x}</b><br>Avg Revenue: ₹%{y:,.0f}<extra></extra>",
        ))
        fig_wkd.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title="Average Daily Revenue: Weekday vs Weekend",
            xaxis_title="Day Type",
            yaxis_title="Avg Revenue (₹)",
            height=380,
        )
        st.plotly_chart(fig_wkd, use_container_width=True)

        wkd_pct = wkd.set_index("label")["daily_sales"]
        if "Weekday" in wkd_pct and "Weekend" in wkd_pct:
            diff_pct = ((wkd_pct["Weekend"] - wkd_pct["Weekday"]) / wkd_pct["Weekday"]) * 100
            direction = "higher" if diff_pct >= 0 else "lower"
            st.info(
                f"💡 Weekend revenue is **{abs(diff_pct):.1f}%** {direction} than weekday average."
            )
    else:
        # Fallback: revenue by day of week
        dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        dow = filtered.groupby("day_of_week")["daily_sales"].mean().reset_index()
        dow["label"] = dow["day_of_week"].map(dow_map)
        fig_dow = go.Figure(go.Bar(
            x=dow["label"],
            y=dow["daily_sales"],
            marker_color=COLORS["caramel"],
        ))
        fig_dow.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title="Average Revenue by Day of Week",
            xaxis_title="Day",
            yaxis_title="Avg Revenue (₹)",
            height=380,
        )
        st.plotly_chart(fig_dow, use_container_width=True)

# ── Tab 4: Distribution ───────────────────────────────────────────────────────
with tab_dist:
    fig_hist = go.Figure(go.Histogram(
        x=filtered["daily_sales"],
        nbinsx=30,
        marker_color=COLORS["caramel"],
        marker_line_color=COLORS["latte"],
        marker_line_width=0.8,
        opacity=0.85,
        hovertemplate="Revenue bin: ₹%{x}<br>Count: %{y}<extra></extra>",
    ))
    fig_hist.add_vline(
        x=filtered["daily_sales"].mean(),
        line_dash="dash",
        line_color=COLORS["accent"],
        annotation_text=f"Mean: ₹{filtered['daily_sales'].mean():,.0f}",
        annotation_font_color=COLORS["accent"],
    )
    fig_hist.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title="Revenue Distribution (Daily Sales)",
        xaxis_title="Revenue (₹)",
        yaxis_title="Days (count)",
        height=380,
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        f"Sales range: ₹{filtered['daily_sales'].min():,.0f} – ₹{filtered['daily_sales'].max():,.0f} | "
        f"Std Dev: ₹{filtered['daily_sales'].std():,.0f}"
    )

# ---------------------------------------------------------------------------
# Key Insights
# ---------------------------------------------------------------------------
st.divider()
st.markdown("## 💡 Key Insights")

insights = []

if _eda_available:
    try:
        raw_insights = eda.compute_insights(daily_df)
        if isinstance(raw_insights, list):
            insights = raw_insights
        elif isinstance(raw_insights, dict):
            insights = list(raw_insights.values())
    except Exception:
        insights = []

# If EDA module unavailable or returned empty, compute manually
if not insights:
    best_month_num = daily_df.groupby("month")["daily_sales"].sum().idxmax()
    best_month_name = MONTH_NAMES.get(best_month_num, str(best_month_num))
    best_month_rev = daily_df.groupby("month")["daily_sales"].sum().max()
    insights.append(
        f"**Best revenue month:** {best_month_name} generated ₹{best_month_rev:,.0f} — the highest of any month in 2025."
    )

    peak_row = daily_df.loc[daily_df["daily_sales"].idxmax()]
    peak_date = peak_row["date"].strftime("%B %d") if "date" in peak_row else f"Day {int(peak_row['day'])}"
    insights.append(
        f"**Peak sales day:** {peak_date} achieved ₹{peak_row['daily_sales']:,.0f}, the single highest revenue day."
    )

    if "is_weekend" in daily_df.columns:
        wkd_avg = daily_df[daily_df["is_weekend"] == 0]["daily_sales"].mean()
        wke_avg = daily_df[daily_df["is_weekend"] == 1]["daily_sales"].mean()
        better = "weekends" if wke_avg > wkd_avg else "weekdays"
        diff = abs(wke_avg - wkd_avg)
        insights.append(
            f"**Day-type pattern:** {better.capitalize()} outperform with ₹{diff:,.0f} more revenue on average per day."
        )

    q1 = daily_df["daily_sales"].quantile(0.25)
    q3 = daily_df["daily_sales"].quantile(0.75)
    insights.append(
        f"**Sales consistency:** 50% of days land between ₹{q1:,.0f} and ₹{q3:,.0f} (IQR: ₹{q3-q1:,.0f})."
    )

    total = daily_df["daily_sales"].sum()
    insights.append(
        f"**Annual performance:** Total 2025 revenue stands at **₹{total:,.0f}** across {len(daily_df)} recorded days."
    )

for bullet in insights:
    st.markdown(
        f"<div class='insight-card'>• {bullet}</div>",
        unsafe_allow_html=True,
    )
