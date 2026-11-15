"""
CoffeeSense AI – sidebar.py
Sidebar navigation component.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAGES = [
    "🏠 Dashboard",
    "📊 Analytics",
    "🤖 AI Assistant",
    "📂 Dataset Explorer",
    "🔮 Prediction Workspace",
    "⚙️ Model Information",
]

_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Sidebar renderer
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    """Render the CoffeeSense AI sidebar and return the selected page name.

    Returns
    -------
    str
        The selected page name, e.g. "🏠 Dashboard".
    """
    with st.sidebar:
        # ── Logo / branding ──────────────────────────────────────────────
        brand_html = (
            '<div style="text-align:center; padding: 1rem 0 0.5rem 0;">'
            '<div style="font-size:2.8rem; filter: drop-shadow(0 4px 12px rgba(200,149,108,0.4));">☕</div>'
            '<div style="color: #FFF8F0; font-size: 1.7rem; font-weight: 800; letter-spacing: 1.5px; line-height: 1.2;">Cafe</div>'
            '<div style="color: #D4A96A; font-size: 0.75rem; font-style: italic; margin-top: 4px; letter-spacing: 0.5px;">'
            'Predict Smarter. Brew Better. Grow Faster.</div>'
            '</div>'
        )
        st.markdown(brand_html, unsafe_allow_html=True)

        st.markdown(
            '<hr style="border-color:rgba(200,149,108,0.3);margin:0.6rem 0 1rem 0;">',
            unsafe_allow_html=True,
        )

        # ── Navigation ───────────────────────────────────────────────────
        st.markdown(
            '<div style="color:#C8956C;font-size:0.72rem;font-weight:700;'
            'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">'
            'Navigation</div>',
            unsafe_allow_html=True,
        )

        selected_page: str = st.radio(
            label="Navigate",
            options=_PAGES,
            label_visibility="collapsed",
        )

        st.markdown(
            '<hr style="border-color:rgba(200,149,108,0.3);margin:1rem 0 0.8rem 0;">',
            unsafe_allow_html=True,
        )

        # ── Quick stats / helper links ───────────────────────────────────
        tips_html = (
            '<div style="font-size:0.78rem;color:#D4A96A;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">Quick Tips</div>'
            '<div style="font-size:0.76rem;color:#F5F0E8;line-height:1.6;">'
            '• Upload your own data in <b>Dataset Explorer</b><br>'
            '• Train models in <b>Prediction Workspace</b><br>'
            '• Chat with ☕ <b>AI Assistant</b>'
            '</div>'
        )
        st.markdown(tips_html, unsafe_allow_html=True)

        # ── Spacer to push footer to bottom ─────────────────────────────
        st.markdown("<div style='height:3rem;'></div>", unsafe_allow_html=True)

        # ── Footer ───────────────────────────────────────────────────────
        footer_html = (
            f'<div style="text-align:center;color:#C8956C;font-size:0.75rem;border-top:1px solid rgba(200,149,108,0.25);padding-top:0.7rem;line-height:1.6;">'
            f'☕ Cafe &nbsp;|&nbsp; v{_VERSION}<br>'
            f'<span style="color:#A67C52;">© 2025 All Rights Reserved</span>'
            f'</div>'
        )
        st.markdown(footer_html, unsafe_allow_html=True)

    return selected_page
