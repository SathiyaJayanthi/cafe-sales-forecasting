"""
Cafe – styles.py
CSS & Plotly theming inspired by luxury coffee UI/UX design:
Obsidian-espresso ambient radial background with colorful cafe item imagery,
glassmorphic frosted cards, warm caramel/golden bronze accents, and clean icon styling.
"""

import base64
from pathlib import Path
import streamlit as st

def _get_bg_image_url() -> str:
    """Return base64 data URL for subtle dark luxury cafe ambient wallpaper."""
    bg_path = Path(__file__).resolve().parent.parent / "assets" / "images" / "cafe_ambient_bg.jpg"
    if bg_path.exists():
        try:
            with open(bg_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"url('data:image/jpeg;base64,{b64}')"
        except Exception:
            return "none"
    return "none"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def get_custom_css() -> str:
    """Return the full CSS string for the luxury Cafe dashboard."""
    bg_img = _get_bg_image_url()
    raw_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap');

/* ===== ROOT PALETTE (LUXURY COFFEE DESIGN SYSTEM) ===== */
:root {
    --bg-darkest:    #090503;
    --bg-espresso:   #120906;
    --bg-card:       rgba(34, 18, 12, 0.72);
    --bg-card-hover: rgba(48, 25, 16, 0.82);
    --sidebar-bg:    linear-gradient(180deg, #180D08 0%, #0E0704 100%);
    
    --caramel:       #C8956C;
    --caramel-glow:  rgba(200, 149, 108, 0.35);
    --latte:         #D4A96A;
    --cream:         #F5F0E8;
    --foam:          #FFF8F0;
    --accent-gold:   #E8C49A;
    --mocha:         #6B3A2A;
    
    --card-border:       rgba(212, 169, 106, 0.22);
    --card-border-hover: rgba(212, 169, 106, 0.55);
    --card-shadow:       0 12px 36px 0 rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    --card-shadow-hover: 0 16px 42px 0 rgba(200, 149, 108, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    
    --font-sans: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-serif: "Playfair Display", Georgia, serif;
}

/* ===== APP BACKGROUND (RICH CAFE ITEM IMAGERY & AMBIENT COFFEE LIGHTING) ===== */
html, body, [data-testid="stAppViewContainer"] {
    background: 
        linear-gradient(180deg, rgba(10, 5, 3, 0.86) 0%, rgba(18, 9, 6, 0.88) 40%, rgba(8, 4, 2, 0.94) 100%),
        radial-gradient(circle at 12% 15%, rgba(184, 112, 62, 0.22) 0%, transparent 45%),
        radial-gradient(circle at 88% 20%, rgba(212, 169, 106, 0.18) 0%, transparent 45%),
        radial-gradient(ellipse at 50% 0%, #201008 0%, #110804 50%, #080402 100%),
        __BG_IMAGE_URL__ center center / cover no-repeat fixed !important;
    background-attachment: fixed !important;
    color: var(--cream) !important;
    font-family: var(--font-sans) !important;
    -webkit-font-smoothing: antialiased;
}

[data-testid="stApp"] {
    background: transparent !important;
}

[data-testid="stMain"] {
    background: transparent !important;
}

.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1400px;
}

/* ===== FIX SIDEBAR COLLAPSE "keyboard_double" ISSUE & REPLACE WITH SLEEK ICONS ===== */
[data-testid="stSidebarCollapseButton"] button {
    background: rgba(45, 24, 15, 0.75) !important;
    border: 1px solid rgba(200, 149, 108, 0.35) !important;
    border-radius: 10px !important;
    color: #D4A96A !important;
    padding: 4px 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

[data-testid="stSidebarCollapseButton"] button:hover {
    background: rgba(200, 149, 108, 0.25) !important;
    border-color: #E8C49A !important;
    transform: scale(1.05);
}

/* Hide raw text ligature string (keyboard_double_arrow_left) and show clean icon */
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="stSidebarCollapseButton"] button p {
    font-size: 0 !important;
    line-height: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 22px !important;
    height: 22px !important;
}

[data-testid="stSidebarCollapseButton"] button span::before,
[data-testid="stSidebarCollapseButton"] button p::before {
    content: "◀" !important;
    font-size: 0.95rem !important;
    line-height: 1 !important;
    color: #D4A96A !important;
    display: block !important;
}

/* Expand sidebar control when collapsed */
[data-testid="collapsedControl"] button {
    background: rgba(32, 16, 10, 0.88) !important;
    border: 1px solid rgba(200, 149, 108, 0.45) !important;
    border-radius: 12px !important;
    color: #D4A96A !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5) !important;
    padding: 6px 10px !important;
}

[data-testid="collapsedControl"] button span,
[data-testid="collapsedControl"] button p {
    font-size: 0 !important;
    line-height: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 26px !important;
    height: 26px !important;
}

[data-testid="collapsedControl"] button span::before,
[data-testid="collapsedControl"] button p::before {
    content: "☕ ▶" !important;
    font-size: 0.95rem !important;
    line-height: 1 !important;
    color: #D4A96A !important;
    display: block !important;
}

/* ===== LUXURY SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid rgba(212, 169, 106, 0.18) !important;
    box-shadow: 6px 0 24px rgba(0, 0, 0, 0.5) !important;
}

[data-testid="stSidebarNav"] {
    background: transparent !important;
    padding-top: 0.5rem;
}

[data-testid="stSidebarNav"] a {
    border-radius: 12px !important;
    margin: 2px 8px !important;
    padding: 8px 14px !important;
    transition: all 0.2s ease !important;
    color: #F5F0E8 !important;
}

[data-testid="stSidebarNav"] a:hover {
    background: rgba(200, 149, 108, 0.15) !important;
    color: #FFF8F0 !important;
    transform: translateX(3px);
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: linear-gradient(90deg, rgba(200, 149, 108, 0.3) 0%, rgba(45, 24, 15, 0.6) 100%) !important;
    border-left: 3px solid var(--caramel) !important;
    color: #FFF8F0 !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--latte) !important;
}

/* ===== HEADERS & TYPOGRAPHY ===== */
h1 {
    color: var(--foam) !important;
    font-size: 2.15rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.6);
}

h2 {
    color: var(--cream) !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
    border-bottom: 1px solid rgba(200, 149, 108, 0.25);
    padding-bottom: 8px;
    margin-bottom: 14px;
}

h3 {
    color: var(--latte) !important;
    font-size: 1.18rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.2px !important;
}

h4 {
    color: var(--accent-gold) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

/* ===== KPI METRIC CARDS (GLASSMORPHIC LUXURY) ===== */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 16px !important;
    padding: 1.1rem 1.2rem !important;
    box-shadow: var(--card-shadow) !important;
    transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="metric-container"]:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--card-border-hover) !important;
    box-shadow: var(--card-shadow-hover) !important;
    transform: translateY(-2px);
}

[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: var(--caramel) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 2px;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--foam) !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.3px !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

/* ===== LUXURY PILL BUTTONS (MATCHING REFERENCE UI) ===== */
.stButton > button {
    background: linear-gradient(135deg, #B8703E 0%, #C8956C 50%, #E2B282 100%) !important;
    color: #0E0704 !important;
    border: none !important;
    border-radius: 28px !important; /* Elegant pill shape */
    padding: 0.55rem 1.6rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.4px !important;
    box-shadow: 0 8px 24px rgba(184, 112, 62, 0.4) !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #C8956C 0%, #E2B282 50%, #FFF0D6 100%) !important;
    box-shadow: 0 10px 30px rgba(200, 149, 108, 0.6) !important;
    transform: translateY(-2px) !important;
    color: #000000 !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

.stDownloadButton > button {
    background: rgba(45, 24, 15, 0.85) !important;
    color: var(--latte) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 24px !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton > button:hover {
    background: rgba(200, 149, 108, 0.2) !important;
    border-color: var(--latte) !important;
    color: var(--foam) !important;
    transform: translateY(-1px);
}

/* ===== INPUTS, SELECTBOXES & DROPDOWNS ===== */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stDateInput > div > div > input,
.stTextArea textarea {
    background-color: rgba(26, 14, 9, 0.88) !important;
    color: var(--foam) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
    padding: 0.4rem 0.8rem !important;
}

.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--latte) !important;
    box-shadow: 0 0 0 2px rgba(200, 149, 108, 0.3) !important;
}

.stSelectbox label,
.stMultiSelect label,
.stTextInput label,
.stNumberInput label,
.stSlider label,
.stTextArea label,
.stDateInput label {
    color: var(--accent-gold) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* ===== TABS (PILL BAR) ===== */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(28, 14, 9, 0.75) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 16px !important;
    padding: 4px 6px !important;
    gap: 6px;
    margin-bottom: 1.2rem;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--caramel) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.3rem !important;
    transition: all 0.2s ease !important;
    border-bottom: none !important;
}

[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    background: rgba(200, 149, 108, 0.15) !important;
    color: var(--foam) !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(200, 149, 108, 0.35) 0%, rgba(138, 75, 42, 0.35) 100%) !important;
    border: 1px solid rgba(212, 169, 106, 0.5) !important;
    color: var(--foam) !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}

/* ===== EXPANDERS & CONTAINERS ===== */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 14px !important;
    box-shadow: var(--card-shadow) !important;
}

[data-testid="stExpander"] summary {
    color: var(--foam) !important;
    font-weight: 600 !important;
}

/* ===== DATA TABLES ===== */
[data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    backdrop-filter: blur(14px) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 14px !important;
    overflow: hidden;
}

/* ===== CHAT MESSAGES ===== */
[data-testid="stChatMessage"] {
    background: rgba(30, 16, 10, 0.75) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 16px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.8rem !important;
}

[data-testid="stChatInput"] textarea {
    background-color: rgba(24, 13, 8, 0.95) !important;
    color: var(--foam) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 20px !important;
}

/* ===== SCROLLBARS ===== */
::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}

::-webkit-scrollbar-track {
    background: rgba(14, 7, 4, 0.6);
}

::-webkit-scrollbar-thumb {
    background: rgba(200, 149, 108, 0.35);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(200, 149, 108, 0.65);
}

/* ===== DIVIDERS ===== */
hr {
    border-color: rgba(200, 149, 108, 0.2) !important;
    margin: 1.4rem 0 !important;
}
</style>
"""
    return raw_css.replace("__BG_IMAGE_URL__", bg_img)


def inject_css() -> None:
    """Inject the custom CSS into the Streamlit app."""
    st.markdown(get_custom_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly theme (Luxury Coffee Palette with Indian Rupee formatting)
# ---------------------------------------------------------------------------

PLOTLY_TEMPLATE: dict = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(20,11,7,0.7)",
        "font": {
            "family": "Plus Jakarta Sans, Segoe UI, sans-serif",
            "color": "#F5F0E8",
            "size": 13,
        },
        "title_font": {"color": "#FFF8F0", "size": 16, "family": "Plus Jakarta Sans, sans-serif"},
        "title_x": 0.03,
        "xaxis": {
            "gridcolor": "rgba(200,149,108,0.12)",
            "linecolor": "rgba(200,149,108,0.25)",
            "tickcolor": "#C8956C",
            "tickfont": {"color": "#F5F0E8"},
            "title_font": {"color": "#D4A96A"},
            "zerolinecolor": "rgba(200,149,108,0.15)",
        },
        "yaxis": {
            "gridcolor": "rgba(200,149,108,0.12)",
            "linecolor": "rgba(200,149,108,0.25)",
            "tickcolor": "#C8956C",
            "tickfont": {"color": "#F5F0E8"},
            "title_font": {"color": "#D4A96A"},
            "zerolinecolor": "rgba(200,149,108,0.15)",
        },
        "legend": {
            "bgcolor": "rgba(28,14,9,0.85)",
            "bordercolor": "rgba(200,149,108,0.3)",
            "borderwidth": 1,
            "font": {"color": "#F5F0E8"},
        },
        "colorway": [
            "#C8956C",   # caramel
            "#D4A96A",   # latte
            "#E8C49A",   # accent gold
            "#8B4513",   # warm amber
            "#F5F0E8",   # cream
            "#A0634D",   # mid-tone roast
            "#DEB887",   # burlywood
            "#E5A872",   # light caramel
        ],
        "hoverlabel": {
            "bgcolor": "#1D0F09",
            "bordercolor": "#C8956C",
            "font": {"color": "#FFF8F0", "size": 13},
        },
        "margin": {"l": 50, "r": 30, "t": 60, "b": 50},
    }
}
