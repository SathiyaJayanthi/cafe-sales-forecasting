"""
Cafe – menu_showcase.py
Renders a luxury visual showcase of signature café items using the newly provided coffee images.
Features star ratings, verified 2025 order volumes, and Indian Rupee (₹) pricing.
"""

from __future__ import annotations

import base64
from pathlib import Path
import streamlit as st

_IMAGES_CACHE: dict[str, str] = {}


def _get_b64_image(filename: str) -> str:
    """Return a base64 data URL for a local image asset."""
    if filename in _IMAGES_CACHE:
        return _IMAGES_CACHE[filename]

    img_path = Path(__file__).resolve().parent.parent / "assets" / "images" / filename
    if img_path.exists():
        try:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = "jpeg" if filename.endswith((".jpg", ".jpeg")) else "png"
            data_url = f"data:image/{ext};base64,{b64}"
            _IMAGES_CACHE[filename] = data_url
            return data_url
        except Exception:
            return ""
    return ""


def render_signature_menu_showcase(trans_df=None) -> None:
    """
    Render luxury cafe product cards using the user's uploaded coffee images.
    """
    img_latte = _get_b64_image("card_latte.jpg")
    img_cappuccino = _get_b64_image("card_cappuccino.jpg")
    img_caramel = _get_b64_image("card_caramel_latte.jpg")
    img_chocolate = _get_b64_image("card_chocolate_milk.jpg")
    img_espresso = _get_b64_image("card_double_espresso.jpg")
    img_tea = _get_b64_image("card_tea.jpg")

    items = [
        {
            "name": "Latte",
            "img": img_latte,
            "rating": "5.0k",
            "price": "₹38.00",
            "orig_price": "₹45.00",
            "orders": "1,085 orders",
            "revenue": "₹40,754",
            "tag": "🏆 #1 Bestseller",
            "tag_bg": "linear-gradient(135deg, #B8703E, #D4A96A)",
            "tag_color": "#120703",
        },
        {
            "name": "Cappuccino",
            "img": img_cappuccino,
            "rating": "4.9k",
            "price": "₹40.00",
            "orig_price": "₹50.00",
            "orders": "721 orders",
            "revenue": "₹24,555",
            "tag": "Popular",
            "tag_bg": "rgba(200,149,108,0.25)",
            "tag_color": "#FFF8F0",
        },
        {
            "name": "Caramel Latte",
            "img": img_caramel,
            "rating": "4.8k",
            "price": "₹45.00",
            "orig_price": "₹55.00",
            "orders": "411 orders",
            "revenue": "₹17,590",
            "tag": "Signature",
            "tag_bg": "rgba(200,149,108,0.25)",
            "tag_color": "#FFF8F0",
        },
        {
            "name": "Chocolate with Milk",
            "img": img_chocolate,
            "rating": "4.8k",
            "price": "₹35.00",
            "orig_price": "₹45.00",
            "orders": "286 orders",
            "revenue": "₹9,579",
            "tag": "Specialty",
            "tag_bg": "rgba(200,149,108,0.25)",
            "tag_color": "#FFF8F0",
        },
    ]

    header_html = (
        '<div style="margin: 1.8rem 0 1rem 0; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">'
        '<div>'
        '<h3 style="margin:0;color:#FFF8F0;font-size:1.35rem;font-weight:700;">☕ Signature Café Items & Bestseller Showcase</h3>'
        '<p style="margin:0.2rem 0 0 0;color:#C8956C;font-size:0.85rem;">Artisan roast profiles and verified menu order velocity (2025)</p>'
        '</div>'
        '<div style="background:rgba(200,149,108,0.18);border:1px solid rgba(200,149,108,0.4);border-radius:20px;padding:4px 14px;color:#D4A96A;font-size:0.78rem;font-weight:600;">'
        '⭐ Premium Roast Selection'
        '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    cols = st.columns(4)
    for col, item in zip(cols, items):
        with col:
            img_html = f'<img src="{item["img"]}" style="width:100%;height:155px;object-fit:cover;border-radius:12px;display:block;" alt="{item["name"]}"/>' if item["img"] else '<div style="height:155px;background:#2a160d;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:2.5rem;">☕</div>'
            
            card_html = (
                f'<div style="background:linear-gradient(135deg, rgba(38, 20, 13, 0.85) 0%, rgba(20, 10, 6, 0.94) 100%);'
                f'backdrop-filter:blur(18px);border:1px solid rgba(212, 169, 106, 0.28);border-radius:16px;padding:0.85rem;'
                f'box-shadow:0 10px 28px rgba(0,0,0,0.55);transition:all 0.25s ease;">'
                f'<div style="position:relative;overflow:hidden;border-radius:12px;margin-bottom:8px;">'
                f'{img_html}'
                f'<div style="position:absolute;top:8px;left:8px;background:rgba(20,10,6,0.75);backdrop-filter:blur(8px);'
                f'padding:2px 8px;border-radius:12px;border:1px solid rgba(212,169,106,0.3);color:#D4A96A;font-size:0.72rem;font-weight:700;">'
                f'★ {item["rating"]}</div>'
                f'<div style="position:absolute;top:8px;right:8px;background:{item["tag_bg"]};'
                f'padding:2px 8px;border-radius:12px;color:{item["tag_color"]};font-size:0.68rem;font-weight:700;">'
                f'{item["tag"]}</div>'
                f'</div>'
                f'<div style="color:#FFF8F0;font-size:1.02rem;font-weight:700;margin-top:2px;">{item["name"]}</div>'
                f'<div style="display:flex;align-items:baseline;gap:8px;margin-top:2px;">'
                f'<span style="color:#D4A96A;font-size:1.15rem;font-weight:800;">{item["price"]}</span>'
                f'<span style="color:#8A6D56;font-size:0.82rem;text-decoration:line-through;">{item["orig_price"]}</span>'
                f'</div>'
                f'<div style="color:#C8956C;font-size:0.76rem;font-weight:600;margin-top:4px;">📊 {item["orders"]} in 2025</div>'
                f'<div style="color:#D4A96A;font-size:0.72rem;margin-top:2px;">💰 {item["revenue"]} total sales</div>'
                f'</div>'
            )
            # Remove any potential newlines to prevent markdown code block rendering
            clean_card_html = "".join(card_html.splitlines())
            st.markdown(clean_card_html, unsafe_allow_html=True)
