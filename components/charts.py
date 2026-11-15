"""
CoffeeSense AI – charts.py
Plotly chart factory using Indian Rupee (₹) formatting and luxury coffee UI styling.
All functions return a plotly Figure object.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objects import Figure

# ---------------------------------------------------------------------------
# Shared layout defaults (Luxury Coffee Palette)
# ---------------------------------------------------------------------------

_PAPER_BG = "rgba(0,0,0,0)"
_PLOT_BG  = "rgba(22, 11, 7, 0.75)"
_FONT_COLOR = "#F5F0E8"
_GRID_COLOR = "rgba(200, 149, 108, 0.12)"
_CARAMEL    = "#C8956C"
_LATTE      = "#D4A96A"
_ACCENT     = "#E8C49A"
_MOCHA      = "#6B3A2A"
_CREAM      = "#F5F0E8"
_FOAM       = "#FFF8F0"

_COLORWAY = [_CARAMEL, _LATTE, _ACCENT, _MOCHA, _CREAM, "#A0634D", "#8B4513", "#DEB887"]

_MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]


def _base_layout(**kwargs) -> dict:
    """Return a base layout dict merged with caller kwargs."""
    base = dict(
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(color=_FONT_COLOR, family="Plus Jakarta Sans, Segoe UI, sans-serif", size=13),
        legend=dict(
            bgcolor="rgba(28, 14, 9, 0.85)",
            bordercolor="rgba(200, 149, 108, 0.3)",
            borderwidth=1,
            font=dict(color=_FONT_COLOR),
        ),
        margin=dict(l=50, r=30, t=60, b=50),
        colorway=_COLORWAY,
        hoverlabel=dict(
            bgcolor="#1E100A",
            bordercolor=_CARAMEL,
            font=dict(color=_FOAM, size=13),
        ),
    )
    base.update(kwargs)
    return base


def _axis_style(**kwargs) -> dict:
    base = dict(
        gridcolor=_GRID_COLOR,
        linecolor="rgba(200, 149, 108, 0.25)",
        tickcolor=_CARAMEL,
        tickfont=dict(color=_FONT_COLOR),
        title_font=dict(color=_LATTE),
        zerolinecolor=_GRID_COLOR,
    )
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Daily trend
# ---------------------------------------------------------------------------

def daily_sales_trend(
    df: pd.DataFrame,
    date_col: str = "date",
    sales_col: str = "daily_sales",
) -> Figure:
    """Line chart with 7-day rolling average."""
    fig = go.Figure()

    # Raw daily line with subtle gradient area fill
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[sales_col],
        mode="lines",
        name="Daily Sales",
        line=dict(color=_CARAMEL, width=2),
        fill="tozeroy",
        fillcolor="rgba(200,149,108,0.12)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Sales: ₹%{y:,.2f}<extra></extra>",
    ))

    # 7-day rolling average
    if len(df) >= 7:
        rolling = df[sales_col].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=rolling,
            mode="lines",
            name="7-Day Avg",
            line=dict(color=_ACCENT, width=2.5, dash="dash"),
            hovertemplate="<b>%{x|%b %d}</b><br>7-Day Avg: ₹%{y:,.2f}<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(title="Daily Revenue Trend"),
        xaxis=_axis_style(title="Date"),
        yaxis=_axis_style(title="Revenue (₹)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Monthly revenue bar
# ---------------------------------------------------------------------------

def monthly_revenue_bar(df: pd.DataFrame) -> Figure:
    """Monthly total revenue bar chart."""
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }
    work = df.copy()
    if "month_name" not in work.columns:
        work["month_name"] = work["month"].map(month_names)

    monthly = (
        work.groupby(["month", "month_name"], as_index=False)["daily_sales"]
        .sum()
        .sort_values("month")
    )

    fig = go.Figure(go.Bar(
        x=monthly["month_name"],
        y=monthly["daily_sales"],
        marker=dict(
            color=monthly["daily_sales"],
            colorscale=[[0, "#8B4513"], [0.5, _CARAMEL], [1.0, _ACCENT]],
            line=dict(color="rgba(200,149,108,0.4)", width=1),
        ),
        hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(title="Monthly Revenue Comparison"),
        xaxis=_axis_style(title="Month", categoryorder="array", categoryarray=_MONTH_ORDER),
        yaxis=_axis_style(title="Revenue (₹)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Weekday vs Weekend comparison
# ---------------------------------------------------------------------------

def weekday_weekend_comparison(df: pd.DataFrame) -> Figure:
    """Bar chart comparing average daily sales: Weekday vs Weekend."""
    work = df.copy()
    work["day_type"] = work["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    avg_sales = work.groupby("day_type", as_index=False)["daily_sales"].mean()

    fig = go.Figure(go.Bar(
        x=avg_sales["day_type"],
        y=avg_sales["daily_sales"],
        marker=dict(
            color=[_CARAMEL, _LATTE],
            line=dict(color="rgba(200,149,108,0.5)", width=1.5),
        ),
        text=[f"₹{v:,.2f}" for v in avg_sales["daily_sales"]],
        textposition="outside",
        textfont=dict(color=_FOAM),
        hovertemplate="<b>%{x}</b><br>Avg Sales: ₹%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(title="Weekday vs Weekend Average Daily Sales"),
        xaxis=_axis_style(title=""),
        yaxis=_axis_style(title="Avg Revenue (₹)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Revenue distribution histogram
# ---------------------------------------------------------------------------

def revenue_distribution_histogram(
    df: pd.DataFrame,
    sales_col: str = "daily_sales",
    nbins: int = 30,
) -> Figure:
    """Histogram with KDE-like distribution curve."""
    fig = px.histogram(
        df,
        x=sales_col,
        nbins=nbins,
        color_discrete_sequence=[_CARAMEL],
        opacity=0.85,
    )
    fig.update_traces(
        marker=dict(line=dict(color="rgba(200,149,108,0.4)", width=1)),
        hovertemplate="Revenue Bin: ₹%{x}<br>Count: %{y}<extra></extra>",
    )
    fig.update_layout(
        **_base_layout(title="Daily Revenue Distribution"),
        xaxis=_axis_style(title="Daily Revenue (₹)"),
        yaxis=_axis_style(title="Frequency"),
    )
    return fig


# ---------------------------------------------------------------------------
# Transaction volume trend
# ---------------------------------------------------------------------------

def transaction_volume_trend(df: pd.DataFrame) -> Figure:
    """Line / area chart of transaction count over time."""
    date_col = "date" if "date" in df.columns else "day"
    tx_col = "transaction_count" if "transaction_count" in df.columns else df.columns[0]

    fig = go.Figure(go.Scatter(
        x=df[date_col],
        y=df[tx_col],
        mode="lines",
        name="Transactions",
        line=dict(color=_LATTE, width=2),
        fill="tozeroy",
        fillcolor="rgba(212,169,106,0.12)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Transactions: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(title="Transaction Volume Trend"),
        xaxis=_axis_style(title="Date"),
        yaxis=_axis_style(title="Transactions"),
    )
    return fig


# ---------------------------------------------------------------------------
# Product sales bar
# ---------------------------------------------------------------------------

def product_sales_bar(trans_df: pd.DataFrame, top_n: int = 10) -> Figure:
    """Horizontal bar chart of top coffee products by total revenue."""
    prod_col = next((c for c in trans_df.columns if "coffee" in c.lower() or "product" in c.lower()), None)
    money_col = next((c for c in trans_df.columns if "money" in c.lower() or "price" in c.lower() or "amount" in c.lower()), None)

    if prod_col is None or money_col is None:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title="Product Sales (columns not found)"))
        return fig

    grouped = (
        trans_df.groupby(prod_col, as_index=False)[money_col]
        .sum()
        .sort_values(money_col, ascending=True)
        .tail(top_n)
    )

    fig = go.Figure(go.Bar(
        x=grouped[money_col],
        y=grouped[prod_col],
        orientation="h",
        marker=dict(
            color=grouped[money_col],
            colorscale=[[0, "#6B3A2A"], [0.6, _CARAMEL], [1.0, _ACCENT]],
            line=dict(color="rgba(200,149,108,0.4)", width=1),
        ),
        text=[f"₹{v:,.2f}" for v in grouped[money_col]],
        textposition="outside",
        textfont=dict(color=_FOAM),
        hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(title=f"Top {top_n} Coffee Products by Revenue"),
        xaxis=_axis_style(title="Revenue (₹)"),
        yaxis=_axis_style(title=""),
        height=max(320, top_n * 38),
    )
    return fig


# ---------------------------------------------------------------------------
# Product popularity pie / donut
# ---------------------------------------------------------------------------

def product_popularity_pie(trans_df: pd.DataFrame) -> Figure:
    """Donut chart of coffee product transaction share."""
    prod_col = next((c for c in trans_df.columns if "coffee" in c.lower() or "product" in c.lower()), None)
    if prod_col is None:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title="Product Popularity (column not found)"))
        return fig

    counts = trans_df[prod_col].value_counts().reset_index()
    counts.columns = ["product", "count"]

    fig = go.Figure(go.Pie(
        labels=counts["product"],
        values=counts["count"],
        hole=0.45,
        marker=dict(colors=_COLORWAY, line=dict(color="#1a0f0a", width=2)),
        hovertemplate="<b>%{label}</b><br>Orders: %{value:,}<br>Share: %{percent}<extra></extra>",
        textfont=dict(color=_FOAM),
    ))
    fig.update_layout(
        **_base_layout(title="Coffee Product Popularity (Share of Orders)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Payment method pie
# ---------------------------------------------------------------------------

def payment_method_pie(trans_df: pd.DataFrame) -> Figure:
    """Donut chart of payment method distribution."""
    pay_col = next((c for c in trans_df.columns if "cash" in c.lower() or "pay" in c.lower()), None)
    if pay_col is None:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title="Payment Methods (column not found)"))
        return fig

    counts = trans_df[pay_col].value_counts().reset_index()
    counts.columns = ["method", "count"]

    colors = [_CARAMEL, _MOCHA, _LATTE, _ACCENT]
    fig = go.Figure(go.Pie(
        labels=counts["method"],
        values=counts["count"],
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#1a0f0a", width=2)),
        hovertemplate="<b>%{label}</b><br>Transactions: %{value:,}<br>Share: %{percent}<extra></extra>",
        textfont=dict(color=_FOAM),
    ))
    fig.update_layout(
        **_base_layout(title="Payment Method Breakdown"),
    )
    return fig


# ---------------------------------------------------------------------------
# Payment method bar (cash vs card revenue)
# ---------------------------------------------------------------------------

def payment_method_bar(trans_df: pd.DataFrame) -> Figure:
    """Bar chart comparing cash vs card total revenue."""
    pay_col = next((c for c in trans_df.columns if "cash" in c.lower() or "pay" in c.lower()), None)
    money_col = next((c for c in trans_df.columns if "money" in c.lower() or "price" in c.lower()), None)

    if pay_col is None or money_col is None:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title="Payment Revenue (columns not found)"))
        return fig

    grouped = trans_df.groupby(pay_col, as_index=False)[money_col].sum()
    colors = [_CARAMEL if "card" in str(m).lower() else _MOCHA for m in grouped[pay_col]]

    fig = go.Figure(go.Bar(
        x=grouped[pay_col],
        y=grouped[money_col],
        marker=dict(color=colors, line=dict(color="rgba(200,149,108,0.5)", width=1.5)),
        text=[f"₹{v:,.2f}" for v in grouped[money_col]],
        textposition="outside",
        textfont=dict(color=_FOAM),
        hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(title="Cash vs Card Revenue Comparison"),
        xaxis=_axis_style(title="Payment Method"),
        yaxis=_axis_style(title="Revenue (₹)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Hourly transaction distribution
# ---------------------------------------------------------------------------

def hourly_distribution(trans_df: pd.DataFrame) -> Figure:
    """Bar chart of transaction count by hour of day."""
    time_col = next((c for c in trans_df.columns if "datetime" in c.lower() or "time" in c.lower()), None)
    if time_col is None:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title="Hourly Distribution (column not found)"))
        return fig

    work = trans_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(work[time_col]):
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")

    work["hour"] = work[time_col].dt.hour
    hourly = work.groupby("hour", as_index=False).size()
    hourly.columns = ["hour", "count"]
    hourly["hour_label"] = hourly["hour"].apply(lambda h: f"{h:02d}:00")

    fig = go.Figure(go.Bar(
        x=hourly["hour_label"],
        y=hourly["count"],
        marker=dict(
            color=hourly["count"],
            colorscale=[[0, "#6B3A2A"], [0.5, _CARAMEL], [1.0, _ACCENT]],
            line=dict(color="rgba(200,149,108,0.4)", width=1),
        ),
        hovertemplate="<b>%{x}</b><br>Transactions: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(title="Peak Business Hours (Hourly Transaction Volume)"),
        xaxis=_axis_style(title="Hour of Day"),
        yaxis=_axis_style(title="Transactions"),
    )
    return fig


# ---------------------------------------------------------------------------
# Weekday box plot
# ---------------------------------------------------------------------------

def weekday_boxplot(df: pd.DataFrame) -> Figure:
    """Box plot of daily revenue by day of the week."""
    dow_map = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
        4: "Friday", 5: "Saturday", 6: "Sunday",
    }
    work = df.copy()
    if "day_name" not in work.columns:
        work["day_name"] = work["day_of_week"].map(dow_map)

    fig = px.box(
        work,
        x="day_name",
        y="daily_sales",
        category_orders={"day_name": _WEEKDAY_ORDER},
        color="day_name",
        color_discrete_sequence=_COLORWAY,
    )
    fig.update_traces(
        marker=dict(color=_ACCENT, size=4),
        line=dict(width=1.5),
        hovertemplate="<b>%{x}</b><br>Sales: ₹%{y:,.2f}<extra></extra>",
    )
    fig.update_layout(
        **_base_layout(title="Sales Variance by Day of Week"),
        xaxis=_axis_style(title=""),
        yaxis=_axis_style(title="Daily Sales (₹)"),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Actual vs Predicted line / scatter
# ---------------------------------------------------------------------------

def actual_vs_predicted(
    y_actual,
    y_pred,
    dates=None,
) -> Figure:
    """Comparison line chart: Actual vs Predicted sales."""
    x_vals = dates if dates is not None else list(range(len(y_actual)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_actual,
        mode="lines",
        name="Actual Sales",
        line=dict(color=_FOAM, width=2),
        hovertemplate="<b>%{x}</b><br>Actual: ₹%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_pred,
        mode="lines",
        name="Predicted Sales",
        line=dict(color=_CARAMEL, width=2, dash="dash"),
        hovertemplate="<b>%{x}</b><br>Predicted: ₹%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(title="Actual vs Predicted Sales (Chronological Test Set)"),
        xaxis=_axis_style(title="Date"),
        yaxis=_axis_style(title="Revenue (₹)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Residuals histogram
# ---------------------------------------------------------------------------

def residuals_histogram(residuals) -> Figure:
    """Histogram of prediction residuals with zero line."""
    fig = px.histogram(
        x=residuals,
        nbins=25,
        color_discrete_sequence=[_ACCENT],
        opacity=0.85,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#E8C49A", line_width=1.5)
    fig.update_traces(marker=dict(line=dict(color="rgba(200,149,108,0.4)", width=1)))
    fig.update_layout(
        **_base_layout(title="Prediction Residuals Distribution"),
        xaxis=_axis_style(title="Residual (₹)"),
        yaxis=_axis_style(title="Frequency"),
    )
    return fig


# ---------------------------------------------------------------------------
# Feature importance horizontal bar
# ---------------------------------------------------------------------------

def feature_importance_bar(importance_df: pd.DataFrame) -> Figure:
    """Horizontal bar chart of feature importances."""
    sorted_df = importance_df.sort_values("importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=sorted_df["importance"],
        y=sorted_df["feature"],
        orientation="h",
        marker=dict(
            color=sorted_df["importance"],
            colorscale=[[0, "#6B3A2A"], [0.6, _CARAMEL], [1.0, _ACCENT]],
            line=dict(color="rgba(200,149,108,0.4)", width=1),
        ),
        text=[f"{v:.3f}" for v in sorted_df["importance"]],
        textposition="outside",
        textfont=dict(color=_FOAM),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(title="Feature Importance"),
        xaxis=_axis_style(title="Importance Score"),
        yaxis=_axis_style(title="Feature"),
        height=max(280, len(importance_df) * 36),
    )
    return fig


# ---------------------------------------------------------------------------
# Multi-day forecast chart with uncertainty band
# ---------------------------------------------------------------------------

def forecast_chart(historical_df: pd.DataFrame, forecast_df: pd.DataFrame) -> Figure:
    """Combined historical + forecast line chart with upper/lower bounds."""
    date_col = "date" if "date" in historical_df.columns else historical_df.columns[0]
    sales_col = "daily_sales" if "daily_sales" in historical_df.columns else historical_df.columns[1]

    hist_tail = historical_df.tail(21)

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_tail[date_col],
        y=hist_tail[sales_col],
        mode="lines+markers",
        name="Actual Sales (Past 3 Weeks)",
        line=dict(color=_FOAM, width=2),
        marker=dict(size=5, color=_FOAM),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Sales: ₹%{y:,.2f}<extra></extra>",
    ))

    # Forecast upper bound
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["upper_bound"],
        mode="lines",
        line=dict(color="rgba(200,149,108,0.15)", width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    # Forecast lower bound (filled to upper)
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["lower_bound"],
        mode="lines",
        line=dict(color="rgba(200,149,108,0.15)", width=0),
        fill="tonexty",
        fillcolor="rgba(200,149,108,0.18)",
        name="Uncertainty Range",
        hoverinfo="skip",
    ))
    # Forecast mean line
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["predicted_sales"],
        mode="lines+markers",
        name="AI Forecast",
        line=dict(color=_CARAMEL, width=2.5, dash="dash"),
        marker=dict(size=6, color=_ACCENT),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: ₹%{y:,.2f}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title="Sales Forecast with Confidence Band"),
        xaxis=_axis_style(title="Date"),
        yaxis=_axis_style(title="Revenue (₹)"),
    )
    return fig
