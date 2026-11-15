"""
CoffeeSense AI – kpi_cards.py
Luxury KPI metric card rendering components using Indian Rupee (₹) formatting.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_currency(value: float | int, decimals: int = 2) -> str:
    """Format a numeric value as an Indian Rupee currency string."""
    try:
        val = float(value)
        return f"₹{val:,.{decimals}f}"
    except (TypeError, ValueError):
        return f"₹{value}"


def format_number(value: float | int, decimals: int = 0) -> str:
    """Format a numeric value with comma separators."""
    try:
        if decimals == 0:
            return f"{float(value):,.0f}"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Single KPI card
# ---------------------------------------------------------------------------

def render_kpi_card(
    title: str,
    value: float | int | str,
    delta: float | int | str | None = None,
    icon: str = "☕",
    prefix: str = "₹",
    suffix: str = "",
    help_text: str = "",
) -> None:
    """Render a single styled luxury glassmorphism KPI metric card.

    Parameters
    ----------
    title:     Card label shown above the value.
    value:     The main KPI value (numeric or pre-formatted string).
    delta:     Optional delta value shown below the main value.
    icon:      Emoji/icon prepended to the title.
    prefix:    String prepended to the displayed value (defaults to "₹").
    suffix:    String appended to the displayed value (e.g. "%").
    help_text: Optional tooltip text shown on the card.
    """
    # Build display strings
    if isinstance(value, (int, float)):
        display_value = f"{prefix}{format_number(value, decimals=2 if prefix == '₹' and value < 1000 else 0)}{suffix}"
    else:
        display_value = f"{prefix}{value}{suffix}" if not str(value).startswith("₹") else f"{value}{suffix}"

    # Build delta string cleanly as a styled inline block span (zero markdown code block triggers)
    delta_html = ""
    if delta is not None:
        delta_str = str(delta)
        is_positive = not delta_str.startswith("-") and "Trough" not in delta_str
        color = "#5AC56A" if is_positive else "#FF6B6B"
        arrow = "▲ " if (is_positive and not delta_str.startswith("+")) else "▼ " if ("Trough" in delta_str or delta_str.startswith("-")) else ""
        delta_html = f'<span style="color:{color};font-size:0.82rem;font-weight:600;margin-top:6px;display:block;">{arrow}{delta_str}</span>'

    help_html = f'<span style="color:#D4A96A;font-size:0.72rem;margin-top:4px;display:block;">{help_text}</span>' if help_text else ""

    # Luxury Glassmorphic Card container with ZERO newlines and pure semantic spans
    card_html = (
        f'<div style="background:linear-gradient(135deg, rgba(38, 20, 13, 0.85) 0%, rgba(20, 10, 6, 0.94) 100%);'
        f'backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);border:1px solid rgba(212, 169, 106, 0.28);'
        f'border-radius:16px;padding:1rem 1.15rem;box-shadow:0 10px 30px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.08);'
        f'margin-bottom:0.6rem;transition:all 0.25s ease;">'
        f'<span style="color:#C8956C;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;display:block;">'
        f'{icon} {title}</span>'
        f'<span style="color:#FFF8F0;font-size:1.52rem;font-weight:800;line-height:1.15;letter-spacing:-0.3px;white-space:nowrap;display:block;">'
        f'{display_value}</span>'
        f'{delta_html}'
        f'{help_html}'
        f'</div>'
    )
    clean_html = "".join(card_html.splitlines())
    st.markdown(clean_html, unsafe_allow_html=True)


def _delta_html(delta_str: str) -> str:
    """Return HTML fragment for the delta line."""
    is_positive = not delta_str.startswith("-")
    color = "#5AC56A" if is_positive else "#FF6B6B"
    arrow = "▲" if is_positive else "▼"
    return f'<span style="color:{color};font-size:0.82rem;font-weight:600;display:block;">{arrow} {delta_str}</span>'


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

def render_kpi_row(kpis_list: list[dict]) -> None:
    """Render a horizontal row of KPI cards using st.columns."""
    if not kpis_list:
        return

    cols = st.columns(len(kpis_list))
    for col, kpi in zip(cols, kpis_list):
        with col:
            render_kpi_card(
                title=kpi.get("title", ""),
                value=kpi.get("value", 0),
                delta=kpi.get("delta", None),
                icon=kpi.get("icon", "☕"),
                prefix=kpi.get("prefix", "₹"),
                suffix=kpi.get("suffix", ""),
                help_text=kpi.get("help_text", ""),
            )
