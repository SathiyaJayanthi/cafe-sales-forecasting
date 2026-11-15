"""
CoffeeSense AI – 2_Analytics.py
Data Visualization & Analytics page with 10 interactive charts across 5 tabs.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from components.styles import inject_css, PLOTLY_TEMPLATE
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

# Optional: model artifacts for actual vs predicted chart
try:
    from src.model_training import load_model_artifacts
    _model_training_available = True
except ImportError:
    _model_training_available = False

try:
    from src.feature_engineering import create_features
    _fe_available = True
except ImportError:
    _fe_available = False


# ---------------------------------------------------------------------------
# Inject CSS
# ---------------------------------------------------------------------------
inject_css()

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading daily sales data…")
def load_daily():
    df = load_daily_dataset(str(DAILY_DATASET_PATH))
    df = clean_daily_dataset(df)
    df = create_date_column(df)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner="Loading transaction data…")
def load_transactions():
    df = load_transactions_dataset(str(TRANSACTIONS_DATASET_PATH))
    df = clean_transactions_dataset(df)
    return df


@st.cache_resource(show_spinner="Loading model artifacts…")
def load_model():
    if not _model_training_available:
        return None, None, None
    try:
        artifacts = load_model_artifacts()
        return artifacts.get("model"), artifacts.get("feature_names"), artifacts
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# Helper: detect column names
# ---------------------------------------------------------------------------

def detect_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Return first column whose name contains any of the keywords (case-insensitive)."""
    for col in df.columns:
        lc = col.lower()
        if any(k in lc for k in keywords):
            return col
    return None


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown("## 📊 Data Visualization & Analytics")
st.markdown(
    '<p style="color:#C8956C;margin-top:-10px;">Explore sales trends, product performance, payment patterns, and model accuracy.</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data_ok = True

try:
    daily_df = load_daily()
except Exception as e:
    st.error(
        f"⚠️ Could not load daily sales data: {e}\n\n"
        "Please ensure the dataset exists at `data/raw/coffee_shop_sales_dataset_2025_daily.csv` "
        "and that the data pipeline has been run."
    )
    data_ok = False

try:
    trans_df = load_transactions()
except Exception as e:
    st.error(
        f"⚠️ Could not load transaction data: {e}\n\n"
        "Please ensure the dataset exists at `data/raw/coffee_shop_transactions_raw_2025.csv`."
    )
    data_ok = False

if not data_ok:
    st.stop()

model, feature_names, model_artifacts = load_model()

# ---------------------------------------------------------------------------
# Detect column names in transactions
# ---------------------------------------------------------------------------

PROD_COL = detect_col(trans_df, ["product", "coffee", "item", "menu"])
PAY_COL = detect_col(trans_df, ["payment", "method", "pay_type", "type"])
AMT_COL = detect_col(trans_df, ["amount", "total", "price", "revenue", "sale", "money"])
DATE_COL_T = detect_col(trans_df, ["date"])

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🔍 Filters")

# Date range
min_date = daily_df["date"].min().date() if not daily_df.empty else date(2025, 1, 1)
max_date = daily_df["date"].max().date() if not daily_df.empty else date(2025, 12, 31)

date_range = st.sidebar.date_input(
    "📅 Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Month multiselect
month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
selected_months = st.sidebar.multiselect(
    "📆 Month",
    options=month_names,
    default=month_names,
    help="Select one or more months to filter data.",
)
selected_month_nums = [month_names.index(m) + 1 for m in selected_months] if selected_months else list(range(1, 13))

# Product filter
if PROD_COL and PROD_COL in trans_df.columns:
    all_products = sorted(trans_df[PROD_COL].dropna().unique().tolist())
    selected_products = st.sidebar.multiselect(
        "☕ Coffee Products",
        options=all_products,
        default=all_products,
    )
else:
    selected_products = []
    all_products = []

# Payment method filter
if PAY_COL and PAY_COL in trans_df.columns:
    all_payments = sorted(trans_df[PAY_COL].dropna().unique().tolist())
    selected_payments = st.sidebar.multiselect(
        "💳 Payment Methods",
        options=all_payments,
        default=all_payments,
    )
else:
    selected_payments = []
    all_payments = []

# Day of week filter
dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_dows = st.sidebar.multiselect(
    "📅 Day of Week",
    options=dow_names,
    default=dow_names,
)
selected_dow_nums = [dow_names.index(d) for d in selected_dows] if selected_dows else list(range(7))

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="sidebar-footer">Cafe v1.0 · Analytics</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Apply filters to daily_df
# ---------------------------------------------------------------------------

filtered_daily = daily_df[
    (daily_df["date"].dt.date >= start_date)
    & (daily_df["date"].dt.date <= end_date)
    & (daily_df["month"].isin(selected_month_nums))
    & (daily_df["day_of_week"].isin(selected_dow_nums))
].copy()

# ---------------------------------------------------------------------------
# Apply filters to trans_df
# ---------------------------------------------------------------------------

filtered_trans = trans_df.copy()

# Date filter on transactions
if "datetime" in filtered_trans.columns:
    filtered_trans = filtered_trans[
        (filtered_trans["datetime"].dt.date >= start_date)
        & (filtered_trans["datetime"].dt.date <= end_date)
    ]
elif DATE_COL_T in filtered_trans.columns:
    try:
        filtered_trans["_dt"] = pd.to_datetime(
            filtered_trans[DATE_COL_T], dayfirst=True, errors="coerce"
        )
        filtered_trans = filtered_trans[
            (filtered_trans["_dt"].dt.date >= start_date)
            & (filtered_trans["_dt"].dt.date <= end_date)
        ]
    except Exception:
        pass

# Month filter on transactions
if "datetime" in filtered_trans.columns:
    filtered_trans = filtered_trans[filtered_trans["datetime"].dt.month.isin(selected_month_nums)]

# Product filter
if PROD_COL and selected_products:
    filtered_trans = filtered_trans[filtered_trans[PROD_COL].isin(selected_products)]

# Payment filter
if PAY_COL and selected_payments:
    filtered_trans = filtered_trans[filtered_trans[PAY_COL].isin(selected_payments)]

# Day-of-week filter
if "datetime" in filtered_trans.columns and selected_dows:
    filtered_trans = filtered_trans[filtered_trans["datetime"].dt.dayofweek.isin(selected_dow_nums)]

# ---------------------------------------------------------------------------
# Show filter summary
# ---------------------------------------------------------------------------

col_summary1, col_summary2 = st.columns(2)
with col_summary1:
    st.info(f"📊 Daily records after filtering: **{len(filtered_daily):,}**")
with col_summary2:
    st.info(f"🧾 Transactions after filtering: **{len(filtered_trans):,}**")

# ---------------------------------------------------------------------------
# Plotly helper
# ---------------------------------------------------------------------------

def apply_theme(fig: go.Figure) -> go.Figure:
    """Apply CoffeeSense Plotly theme to a figure."""
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Sales Trends", "☕ Products", "💳 Payments", "🕐 Patterns", "📊 Distribution"]
)

# ===========================================================================
# TAB 1 – Sales Trends
# ===========================================================================

with tab1:
    st.markdown("### 📈 Sales Trends")

    # ── Chart 1: Daily Revenue Line ──────────────────────────────────────────
    if not filtered_daily.empty:
        fig_daily = go.Figure()
        fig_daily.add_trace(
            go.Scatter(
                x=filtered_daily["date"],
                y=filtered_daily["daily_sales"],
                mode="lines",
                name="Daily Revenue",
                line=dict(color=COLORS["caramel"], width=2),
                fill="tozeroy",
                fillcolor="rgba(200,149,108,0.12)",
                hovertemplate="<b>%{x|%b %d}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
            )
        )
        fig_daily.update_layout(
            title="Daily Revenue Trend",
            xaxis_title="Date",
            yaxis_title="Revenue (₹)",
        )
        fig_daily = apply_theme(fig_daily)
        st.plotly_chart(fig_daily, use_container_width=True)

        avg = filtered_daily["daily_sales"].mean()
        peak = filtered_daily["daily_sales"].max()
        st.markdown(
            f"> 📌 **Interpretation:** The daily revenue trend shows how sales fluctuate "
            f"across the selected period. The average daily revenue is **₹{avg:,.0f}**, "
            f"with a peak of **₹{peak:,.0f}**. Look for recurring spikes on weekends or "
            f"holidays and dips mid-week to plan staffing and inventory."
        )
    else:
        st.warning("No daily data matches the current filters.")

    st.markdown("---")

    # ── Chart 2: Monthly Revenue Bar ─────────────────────────────────────────
    if not filtered_daily.empty:
        monthly_rev = (
            filtered_daily.groupby("month")["daily_sales"]
            .sum()
            .reset_index()
            .rename(columns={"daily_sales": "total_revenue"})
        )
        monthly_rev["month_name"] = monthly_rev["month"].apply(lambda m: month_names[m - 1])
        monthly_rev = monthly_rev.sort_values("month")

        fig_monthly = go.Figure(
            go.Bar(
                x=monthly_rev["month_name"],
                y=monthly_rev["total_revenue"],
                marker_color=COLORS["caramel"],
                hovertemplate="<b>%{x}</b><br>Total Revenue: ₹%{y:,.0f}<extra></extra>",
            )
        )
        fig_monthly.update_layout(
            title="Monthly Revenue",
            xaxis_title="Month",
            yaxis_title="Total Revenue (₹)",
        )
        fig_monthly = apply_theme(fig_monthly)
        st.plotly_chart(fig_monthly, use_container_width=True)

        if not monthly_rev.empty:
            best_month = monthly_rev.loc[monthly_rev["total_revenue"].idxmax(), "month_name"]
            best_rev = monthly_rev["total_revenue"].max()
            st.markdown(
                f"> 📌 **Interpretation:** Monthly revenue aggregates daily sales to show "
                f"seasonality across the year. **{best_month}** leads with **₹{best_rev:,.0f}** "
                f"in total revenue. Use this chart to identify seasonal peaks and plan "
                f"promotions accordingly."
            )
    else:
        st.warning("No daily data to plot monthly revenue.")

# ===========================================================================
# TAB 2 – Products
# ===========================================================================

with tab2:
    st.markdown("### ☕ Product Performance")

    if PROD_COL is None:
        st.info("ℹ️ No product column detected in the transactions dataset.")
    elif filtered_trans.empty:
        st.warning("No transaction data matches the current filters.")
    else:
        # ── Chart 3: Product Revenue Bar ─────────────────────────────────────
        if AMT_COL and AMT_COL in filtered_trans.columns:
            prod_rev = (
                filtered_trans.groupby(PROD_COL)[AMT_COL]
                .sum()
                .reset_index()
                .rename(columns={PROD_COL: "product", AMT_COL: "revenue"})
                .sort_values("revenue", ascending=True)
            )

            fig_prod_rev = go.Figure(
                go.Bar(
                    x=prod_rev["revenue"],
                    y=prod_rev["product"],
                    orientation="h",
                    marker_color=COLORS["caramel"],
                    hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>",
                )
            )
            fig_prod_rev.update_layout(
                title="Revenue by Product",
                xaxis_title="Total Revenue (₹)",
                yaxis_title="",
            )
            fig_prod_rev = apply_theme(fig_prod_rev)
            st.plotly_chart(fig_prod_rev, use_container_width=True)

            top_prod = prod_rev.iloc[-1]["product"] if not prod_rev.empty else "N/A"
            top_prod_rev = prod_rev.iloc[-1]["revenue"] if not prod_rev.empty else 0
            st.markdown(
                f"> 📌 **Interpretation:** This chart ranks products by total revenue. "
                f"**{top_prod}** is the highest earner at **₹{top_prod_rev:,.0f}**, "
                f"indicating it is the most profitable item on the menu. Consider promoting "
                f"lower-revenue products through bundles or discounts to balance the revenue mix."
            )
        else:
            st.info("ℹ️ No monetary amount column detected for product revenue chart.")

        st.markdown("---")

        # ── Chart 4: Product Popularity Donut ────────────────────────────────
        prod_counts = (
            filtered_trans.groupby(PROD_COL)
            .size()
            .reset_index(name="transaction_count")
            .sort_values("transaction_count", ascending=False)
        )

        fig_donut = go.Figure(
            go.Pie(
                labels=prod_counts[PROD_COL],
                values=prod_counts["transaction_count"],
                hole=0.4,
                marker=dict(
                    colors=[
                        COLORS["caramel"], COLORS["latte"], COLORS["accent"],
                        COLORS["mocha"], COLORS["cream"], "#a0634d", "#deb887", "#8B4513",
                    ]
                ),
                hovertemplate="<b>%{label}</b><br>Orders: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
        )
        fig_donut.update_layout(title="Product Popularity (Transaction Count)")
        fig_donut = apply_theme(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)

        if not prod_counts.empty:
            top_pop = prod_counts.iloc[0][PROD_COL]
            top_pct = prod_counts.iloc[0]["transaction_count"] / prod_counts["transaction_count"].sum() * 100
            st.markdown(
                f"> 📌 **Interpretation:** The donut chart shows order share by product. "
                f"**{top_pop}** accounts for **{top_pct:.1f}%** of all orders, making it "
                f"the most frequently ordered item. High-volume products should always be "
                f"well-stocked to prevent stockouts during peak hours."
            )

# ===========================================================================
# TAB 3 – Payments
# ===========================================================================

with tab3:
    st.markdown("### 💳 Payment Method Analysis")

    if PAY_COL is None:
        st.info("ℹ️ No payment method column detected in the transactions dataset.")
    elif filtered_trans.empty:
        st.warning("No transaction data matches the current filters.")
    else:
        pay_counts = (
            filtered_trans.groupby(PAY_COL)
            .size()
            .reset_index(name="count")
        )

        # ── Chart 5: Payment Methods Pie ─────────────────────────────────────
        fig_pay_pie = go.Figure(
            go.Pie(
                labels=pay_counts[PAY_COL],
                values=pay_counts["count"],
                marker=dict(
                    colors=[
                        COLORS["caramel"], COLORS["latte"], COLORS["mocha"],
                        COLORS["accent"], COLORS["cream"],
                    ]
                ),
                hovertemplate="<b>%{label}</b><br>Transactions: %{value:,}<br>%{percent}<extra></extra>",
            )
        )
        fig_pay_pie.update_layout(title="Payment Method Distribution")
        fig_pay_pie = apply_theme(fig_pay_pie)
        st.plotly_chart(fig_pay_pie, use_container_width=True)

        total_pay = pay_counts["count"].sum()
        if not pay_counts.empty:
            top_pay = pay_counts.loc[pay_counts["count"].idxmax(), PAY_COL]
            top_pay_pct = pay_counts["count"].max() / total_pay * 100
            st.markdown(
                f"> 📌 **Interpretation:** **{top_pay}** is the most popular payment method, "
                f"used in **{top_pay_pct:.1f}%** of transactions. Understanding payment "
                f"preferences helps the café plan for cash-handling needs and digital "
                f"payment infrastructure investments."
            )

        st.markdown("---")

        # ── Chart 6: Cash vs Card/Digital Bar ───────────────────────────────
        def classify_payment(method: str) -> str:
            if isinstance(method, str) and "cash" in method.lower():
                return "Cash"
            return "Card / Digital"

        pay_grouped = filtered_trans.copy()
        pay_grouped["pay_category"] = pay_grouped[PAY_COL].apply(classify_payment)
        cat_counts = (
            pay_grouped.groupby("pay_category")
            .size()
            .reset_index(name="count")
        )

        fig_pay_bar = go.Figure(
            go.Bar(
                x=cat_counts["pay_category"],
                y=cat_counts["count"],
                marker_color=[COLORS["caramel"], COLORS["latte"]],
                hovertemplate="<b>%{x}</b><br>Transactions: %{y:,}<extra></extra>",
                text=cat_counts["count"].apply(lambda v: f"{v:,}"),
                textposition="outside",
            )
        )
        fig_pay_bar.update_layout(
            title="Cash vs Card / Digital Payments",
            xaxis_title="Payment Category",
            yaxis_title="Transaction Count",
        )
        fig_pay_bar = apply_theme(fig_pay_bar)
        st.plotly_chart(fig_pay_bar, use_container_width=True)

        cash_rows = cat_counts[cat_counts["pay_category"] == "Cash"]
        card_rows = cat_counts[cat_counts["pay_category"] == "Card / Digital"]
        cash_n = int(cash_rows["count"].sum()) if not cash_rows.empty else 0
        card_n = int(card_rows["count"].sum()) if not card_rows.empty else 0
        st.markdown(
            f"> 📌 **Interpretation:** Cash payments total **{cash_n:,}** transactions "
            f"vs **{card_n:,}** card/digital transactions. A higher digital ratio suggests "
            f"customers prefer contactless payments, which can speed up service and reduce "
            f"cash-handling overhead."
        )

# ===========================================================================
# TAB 4 – Patterns
# ===========================================================================

with tab4:
    st.markdown("### 🕐 Time & Day Patterns")

    # ── Chart 7: Hourly Distribution Bar ────────────────────────────────────
    if "hour" in filtered_trans.columns and not filtered_trans.empty:
        hourly = (
            filtered_trans.groupby("hour")
            .size()
            .reset_index(name="transaction_count")
        )

        fig_hourly = go.Figure(
            go.Bar(
                x=hourly["hour"],
                y=hourly["transaction_count"],
                marker_color=COLORS["caramel"],
                hovertemplate="<b>Hour: %{x}:00</b><br>Transactions: %{y:,}<extra></extra>",
            )
        )
        fig_hourly.update_layout(
            title="Transactions by Hour of Day",
            xaxis_title="Hour of Day",
            yaxis_title="Transaction Count",
            xaxis=dict(tickmode="array", tickvals=list(range(0, 24))),
        )
        fig_hourly = apply_theme(fig_hourly)
        st.plotly_chart(fig_hourly, use_container_width=True)

        if not hourly.empty:
            peak_hour = hourly.loc[hourly["transaction_count"].idxmax(), "hour"]
            peak_txn = hourly["transaction_count"].max()
            st.markdown(
                f"> 📌 **Interpretation:** Transaction volume peaks at **{peak_hour}:00** "
                f"with **{peak_txn:,}** transactions. This is likely the morning rush or lunch "
                f"hour. Ensuring adequate staffing during peak hours can reduce wait times "
                f"and improve customer satisfaction."
            )
    else:
        st.info("ℹ️ Hourly data requires a parsed `datetime` / `hour` column in the transactions dataset.")

    st.markdown("---")

    # ── Chart 8: Daily Sales Box Plot by Day of Week ─────────────────────────
    if not filtered_daily.empty:
        dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        filtered_daily["dow_name"] = filtered_daily["day_of_week"].map(dow_map)

        fig_box = go.Figure()
        for dow_num in sorted(filtered_daily["day_of_week"].unique()):
            subset = filtered_daily[filtered_daily["day_of_week"] == dow_num]
            fig_box.add_trace(
                go.Box(
                    y=subset["daily_sales"],
                    name=dow_map.get(dow_num, str(dow_num)),
                    marker_color=COLORS["caramel"],
                    line_color=COLORS["latte"],
                    hovertemplate="<b>%{x}</b><br>Sales: ₹%{y:,.0f}<extra></extra>",
                )
            )
        fig_box.update_layout(
            title="Daily Sales Distribution by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Daily Sales (₹)",
            showlegend=False,
        )
        fig_box = apply_theme(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

        dow_avg = filtered_daily.groupby("dow_name")["daily_sales"].mean()
        best_dow = dow_avg.idxmax() if not dow_avg.empty else "N/A"
        best_dow_val = dow_avg.max() if not dow_avg.empty else 0
        st.markdown(
            f"> 📌 **Interpretation:** The box plot reveals the spread and median of sales "
            f"for each weekday. **{best_dow}** shows the highest median revenue at "
            f"**₹{best_dow_val:,.0f}** on average. The whiskers expose high-variability days "
            f"where promotional events may be driving outlier peaks."
        )
    else:
        st.warning("No daily data to plot day-of-week pattern.")

# ===========================================================================
# TAB 5 – Distribution
# ===========================================================================

with tab5:
    st.markdown("### 📊 Sales Distribution & Model Comparison")

    # ── Chart 9: Sales Histogram ─────────────────────────────────────────────
    if not filtered_daily.empty:
        fig_hist = go.Figure(
            go.Histogram(
                x=filtered_daily["daily_sales"],
                nbinsx=30,
                marker_color=COLORS["caramel"],
                opacity=0.85,
                hovertemplate="Revenue: ₹%{x:,.0f}<br>Days: %{y}<extra></extra>",
            )
        )
        fig_hist.update_layout(
            title="Daily Sales Revenue Distribution",
            xaxis_title="Daily Revenue (₹)",
            yaxis_title="Number of Days",
            bargap=0.05,
        )
        fig_hist = apply_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

        mean_sal = filtered_daily["daily_sales"].mean()
        std_sal = filtered_daily["daily_sales"].std()
        skew = filtered_daily["daily_sales"].skew()
        skew_desc = "right-skewed (a few high-sales days pull the average up)" if skew > 0.5 else (
            "left-skewed (a few low-sales days drag the average down)" if skew < -0.5 else "approximately normal"
        )
        st.markdown(
            f"> 📌 **Interpretation:** The distribution of daily revenue has a mean of "
            f"**₹{mean_sal:,.0f}** and a standard deviation of **₹{std_sal:,.0f}**. "
            f"The distribution is **{skew_desc}**. Understanding this helps set realistic "
            f"revenue targets and define what constitutes a 'bad' vs 'exceptional' day."
        )
    else:
        st.warning("No daily data to show histogram.")

    st.markdown("---")

    # ── Chart 10: Actual vs Predicted Scatter ────────────────────────────────
    st.markdown("#### 🤖 Actual vs Predicted Revenue")

    if model is None or not _model_training_available or not _fe_available:
        st.info(
            "ℹ️ The **Actual vs Predicted** chart requires a trained model. "
            "Please train the model from the **Model Information** page and return here."
        )
        st.markdown(
            "> 📌 **Interpretation:** Once the Random Forest model is trained, this scatter "
            "plot will compare each day's predicted revenue against the actual revenue. "
            "Points close to the diagonal line indicate high prediction accuracy. "
            "Clusters of outlier points reveal conditions the model struggles with."
        )
    else:
        try:
            feat_df = create_features(daily_df.copy())
            X = feat_df[feature_names]
            y_actual = feat_df["daily_sales"].values
            y_pred = model.predict(X)

            fig_scatter = go.Figure()
            fig_scatter.add_trace(
                go.Scatter(
                    x=y_actual,
                    y=y_pred,
                    mode="markers",
                    marker=dict(
                        color=COLORS["caramel"],
                        size=6,
                        opacity=0.75,
                        line=dict(color=COLORS["mocha"], width=0.5),
                    ),
                    hovertemplate="Actual: ₹%{x:,.0f}<br>Predicted: ₹%{y:,.0f}<extra></extra>",
                    name="Predictions",
                )
            )
            # Perfect prediction line
            min_val = min(y_actual.min(), y_pred.min())
            max_val = max(y_actual.max(), y_pred.max())
            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color=COLORS["latte"], dash="dash", width=1.5),
                    name="Perfect Prediction",
                    hoverinfo="skip",
                )
            )
            fig_scatter.update_layout(
                title="Actual vs Predicted Daily Revenue",
                xaxis_title="Actual Revenue (₹)",
                yaxis_title="Predicted Revenue (₹)",
            )
            fig_scatter = apply_theme(fig_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True)

            residuals = y_actual - y_pred
            mae = np.mean(np.abs(residuals))
            st.markdown(
                f"> 📌 **Interpretation:** Each point represents one day — the closer it is "
                f"to the dashed diagonal, the more accurate the model. "
                f"The model achieves a Mean Absolute Error of **₹{mae:,.0f}** per day, "
                f"meaning predictions deviate from actual sales by that amount on average. "
                f"Points far above the line represent underestimates; points far below are overestimates."
            )
        except Exception as exc:
            st.warning(f"Could not generate actual vs predicted chart: {exc}")
            st.markdown(
                "> 📌 **Interpretation:** This scatter plot compares actual vs predicted revenue. "
                "Train the model and ensure feature engineering is complete to enable this chart."
            )
