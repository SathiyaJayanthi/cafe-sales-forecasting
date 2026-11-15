"""
CoffeeSense AI – 5_Prediction_Workspace.py
Manual Sales Prediction & Model Testing Workspace.
Includes interactive feature input, autofill presets, single prediction explanation,
session prediction history, historical test period evaluator, and 7/30-day recursive forecasting.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from datetime import date, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Styling & Imports ────────────────────────────────────────────────────────
try:
    from components.styles import inject_css
    inject_css()
except Exception:
    pass

from config.settings import (
    DAILY_DATASET_PATH,
    MODEL_PATH,
    METADATA_PATH,
    FEATURE_NAMES_PATH,
    COLORS,
)
from src.data_processing import (
    load_daily_dataset,
    clean_daily_dataset,
    create_date_column,
)
from src.prediction import (
    predict_single,
    get_prediction_explanation,
    evaluate_test_period,
    get_sample_prediction_inputs,
)
from src.forecasting import (
    generate_recursive_forecast,
    get_forecast_summary,
    prepare_forecast_chart_data,
)
from src.model_training import load_model_artifacts
from components.charts import actual_vs_predicted, forecast_chart


@st.cache_data(show_spinner=False)
def load_data_and_model():
    daily_raw = load_daily_dataset(DAILY_DATASET_PATH)
    daily_df = clean_daily_dataset(daily_raw)
    daily_df = create_date_column(daily_df)
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    model, feature_names, metadata = load_model_artifacts(
        model_path=MODEL_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        metadata_path=METADATA_PATH,
    )
    return daily_df, model, feature_names, metadata


def main():
    st.markdown(
        "<h1 style='margin-bottom:0'>🔮 Manual Sales Prediction & Model Testing Workspace</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#C8956C;margin-top:0;margin-bottom:1.5rem'>"
        "Simulate customized operational scenarios, evaluate historical out-of-sample accuracy, and generate recursive forecasts."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading prediction engine and datasets…"):
        daily_df, model, feature_names, metadata = load_data_and_model()

    if model is None or feature_names is None:
        st.error(
            "⚠️ Model artifact not found. Please run the model training script or train the model in the Model Information page."
        )
        return

    # Initialize session state for prediction history
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    # ── Workspace Tabs ───────────────────────────────────────────────────────
    tab_single, tab_history, tab_eval, tab_forecast = st.tabs([
        "🎯 Single-Day Scenario Prediction",
        "📜 Session Prediction History",
        "⚖️ Historical Test Period Evaluator",
        "📈 7 & 30-Day Recursive Forecast",
    ])

    avg_historical_sales = float(daily_df["daily_sales"].mean())
    min_historical_sales = float(daily_df["daily_sales"].min())
    max_historical_sales = float(daily_df["daily_sales"].max())

    # =========================================================================
    # TAB 1: SINGLE-DAY SCENARIO PREDICTION
    # =========================================================================
    with tab_single:
        st.markdown("### 🎛️ Configure Prediction Scenario")

        # Quick autofill buttons
        st.markdown("**⚡ Quick Autofill Presets:**")
        preset_cols = st.columns(4)
        with preset_cols[0]:
            if st.button("📊 Historical Average", use_container_width=True):
                st.session_state.p_prev_day = round(avg_historical_sales, 2)
                st.session_state.p_prev_3d = round(avg_historical_sales, 2)
                st.session_state.p_prev_7d = round(avg_historical_sales, 2)
                st.rerun()
        with preset_cols[1]:
            if st.button("⏪ Yesterday's Actual (Dec 31)", use_container_width=True):
                st.session_state.p_prev_day = float(daily_df["daily_sales"].iloc[-1])
                st.session_state.p_prev_3d = float(daily_df["daily_sales"].iloc[-3:].mean())
                st.session_state.p_prev_7d = float(daily_df["daily_sales"].iloc[-7:].mean())
                st.rerun()
        with preset_cols[2]:
            if st.button("📉 Slow Trough Day", use_container_width=True):
                st.session_state.p_prev_day = round(min_historical_sales, 2)
                st.session_state.p_prev_3d = round(min_historical_sales * 1.05, 2)
                st.session_state.p_prev_7d = round(min_historical_sales * 1.1, 2)
                st.rerun()
        with preset_cols[3]:
            if st.button("🚀 Peak Rush Day", use_container_width=True):
                st.session_state.p_prev_day = round(max_historical_sales, 2)
                st.session_state.p_prev_3d = round(max_historical_sales * 0.95, 2)
                st.session_state.p_prev_7d = round(max_historical_sales * 0.9, 2)
                st.rerun()

        # Input form
        with st.form("manual_prediction_form"):
            st.markdown("#### 1. Calendar Inputs")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                target_date = st.date_input("Target Date:", value=date(2026, 1, 1))
            with c2:
                day_val = target_date.day
                st.text_input("Day of Month:", value=str(day_val), disabled=True)
            with c3:
                month_val = target_date.month
                st.text_input("Month:", value=str(month_val), disabled=True)
            with c4:
                dow_val = target_date.weekday()
                dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                is_we_val = 1 if dow_val in (5, 6) else 0
                st.text_input("Day of Week:", value=f"{dow_names[dow_val]} ({'Weekend' if is_we_val else 'Weekday'})", disabled=True)

            st.markdown("#### 2. Historical Lag Sales Features (₹)")
            l1, l2, l3 = st.columns(3)
            with l1:
                prev_day_default = st.session_state.get("p_prev_day", round(float(daily_df["daily_sales"].iloc[-1]), 2))
                prev_day = st.number_input(
                    "Previous Day Sales (₹):",
                    min_value=0.0,
                    max_value=10000.0,
                    value=float(prev_day_default),
                    step=10.0,
                )
            with l2:
                prev_3d_default = st.session_state.get("p_prev_3d", round(float(daily_df["daily_sales"].iloc[-3:].mean()), 2))
                prev_3d = st.number_input(
                    "Previous 3-Day Average (₹):",
                    min_value=0.0,
                    max_value=10000.0,
                    value=float(prev_3d_default),
                    step=10.0,
                )
            with l3:
                prev_7d_default = st.session_state.get("p_prev_7d", round(float(daily_df["daily_sales"].iloc[-7:].mean()), 2))
                prev_7d = st.number_input(
                    "Previous 7-Day Average (₹):",
                    min_value=0.0,
                    max_value=10000.0,
                    value=float(prev_7d_default),
                    step=10.0,
                )

            st.caption("🔒 Zero Data Leakage: Advance forecasting strictly uses past observations only.")

            submit_pred = st.form_submit_button("🔮 Predict Daily Revenue", use_container_width=True)

        if submit_pred:
            result = predict_single(
                model=model,
                feature_names=feature_names,
                day=day_val,
                month=month_val,
                day_of_week=dow_val,
                is_weekend=is_we_val,
                prev_day_sales=prev_day,
                prev_3day_avg=prev_3d,
                prev_7day_avg=prev_7d,
            )

            pred_val = result["prediction"]
            model_used = result["model_name"]
            conf_note = result["confidence_note"]
            explanation = get_prediction_explanation(pred_val, result["features_used"], avg_historical_sales)

            # Store in session history
            st.session_state.prediction_history.append({
                "Timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                "Target Date": str(target_date),
                "Day": day_val,
                "Month": month_val,
                "Day of Week": dow_names[dow_val],
                "Is Weekend": "Yes" if is_we_val else "No",
                "Prev Day (₹)": prev_day,
                "Prev 3D Avg (₹)": prev_3d,
                "Prev 7D Avg (₹)": prev_7d,
                "Predicted Sales (₹)": pred_val,
                "Model": model_used,
            })

            # Display prediction KPI
            st.markdown("---")
            st.markdown("### 🎯 Forecast Result")
            r1, r2, r3 = st.columns([2, 1, 1])
            with r1:
                st.markdown(
                    f"<div style='background:linear-gradient(135deg, rgba(44,24,16,0.9), rgba(62,32,20,0.9));backdrop-filter:blur(16px);padding:1.5rem;border-radius:14px;border:1.5px solid rgba(200,149,108,0.5);box-shadow:0 12px 36px rgba(0,0,0,0.5);text-align:center;'>"
                    f"<div style='color:#D4A96A;font-size:1.1rem;font-weight:600;'>FORECASTED REVENUE FOR {target_date}</div>"
                    f"<div style='color:#FFF8F0;font-size:3.2rem;font-weight:800;margin:0.5rem 0;'>₹{pred_val:,.2f}</div>"
                    f"<div style='color:#C8956C;font-size:0.9rem;'>Model: <b>{model_used}</b> | {conf_note}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with r2:
                diff_from_avg = pred_val - avg_historical_sales
                diff_pct = (diff_from_avg / avg_historical_sales * 100) if avg_historical_sales > 0 else 0
                st.metric(
                    "vs. 2025 Average (₹)",
                    f"₹{pred_val:,.2f}",
                    delta=f"{diff_pct:+.1f}% (₹{diff_from_avg:+.2f})",
                )
            with r3:
                # Approximate cups calculation (₹35 average ticket)
                est_cups = int(pred_val / 35.0) if pred_val > 0 else 0
                st.metric("Estimated Drink Volume", f"~{est_cups:,} cups", help="Assuming ~₹35 avg beverage price")

            # Explanation card
            st.markdown("#### 💡 AI Interpretation & Feature Rationale")
            st.markdown(
                f"<div style='background:rgba(26,15,10,0.85);backdrop-filter:blur(12px);padding:1.2rem;border-radius:10px;border-left:4px solid #C8956C;color:#F5F0E8;line-height:1.6;'>"
                f"{explanation}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 2: SESSION PREDICTION HISTORY
    # =========================================================================
    with tab_history:
        st.markdown("### 📜 Session Prediction History")
        if not st.session_state.prediction_history:
            st.info("No manual predictions made yet in this session. Configure a scenario in the first tab to populate history.")
        else:
            hist_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(
                hist_df.style.format({
                    "Prev Day (₹)": "₹{:,.2f}",
                    "Prev 3D Avg (₹)": "₹{:,.2f}",
                    "Prev 7D Avg (₹)": "₹{:,.2f}",
                    "Predicted Sales (₹)": "₹{:,.2f}",
                }),
                use_container_width=True,
            )

            col_h1, col_h2 = st.columns([1, 4])
            with col_h1:
                csv_data = hist_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export History as CSV",
                    data=csv_data,
                    file_name="coffeesense_prediction_history.csv",
                    mime="text/csv",
                )
            with col_h2:
                if st.button("🗑️ Clear History"):
                    st.session_state.prediction_history = []
                    st.rerun()

    # =========================================================================
    # TAB 3: HISTORICAL TEST PERIOD EVALUATOR
    # =========================================================================
    with tab_eval:
        st.markdown("### ⚖️ Historical Test Period Evaluator (Out-of-Sample)")
        st.markdown(
            "Validate how closely the trained model's forecasts matched ground-truth sales during the 20% holdout test window."
        )

        test_eval = evaluate_test_period(model, feature_names, daily_df)
        if test_eval is None or test_eval.get("metrics") is None:
            st.warning("Could not calculate test period evaluations. Ensure dataset contains sufficient daily rows.")
        else:
            m = test_eval["metrics"]
            pred_df = test_eval["predictions_df"]

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Test MAE", f"₹{m.get('MAE', 0):,.2f}", help="Mean Absolute Error on unseen test period")
            with m2:
                st.metric("Test RMSE", f"₹{m.get('RMSE', 0):,.2f}", help="Root Mean Squared Error")
            with m3:
                st.metric("Test R²", f"{m.get('R2', 0):.4f}", help="Coefficient of Determination")
            with m4:
                st.metric("Test MAPE", f"{m.get('MAPE', 0):.2f}%", help="Mean Absolute Percentage Error")

            st.markdown("---")
            st.markdown("#### 📅 Single-Day Test Error Inspector")
            pred_df["date_str"] = pd.to_datetime(pred_df["date"]).dt.strftime("%Y-%m-%d")
            dates_available = pred_df["date_str"].tolist()

            sel_date_str = st.selectbox("Select Test Period Date:", options=dates_available, index=len(dates_available)-1)
            row_match = pred_df[pred_df["date_str"] == sel_date_str].iloc[0]

            act_val = float(row_match["actual_sales"])
            pred_val = float(row_match["predicted_sales"])
            err_val = pred_val - act_val
            err_pct = (abs(err_val) / act_val * 100) if act_val > 0 else 0

            ec1, ec2, ec3, ec4 = st.columns(4)
            with ec1:
                st.metric("Actual Sales", f"₹{act_val:,.2f}")
            with ec2:
                st.metric("Predicted Sales", f"₹{pred_val:,.2f}")
            with ec3:
                st.metric("Error (₹)", f"₹{err_val:+,.2f}", delta=f"₹{err_val:+,.2f}")
            with ec4:
                st.metric("Abs Error %", f"{err_pct:.1f}%")

            st.markdown("---")
            st.markdown("#### 📉 Actual vs Predicted Sales (Chronological Test Holdout)")
            fig_test = actual_vs_predicted(pred_df["actual_sales"], pred_df["predicted_sales"], pred_df["date"])
            st.plotly_chart(fig_test, use_container_width=True)

    # =========================================================================
    # TAB 4: RECURSIVE MULTI-DAY FORECAST
    # =========================================================================
    with tab_forecast:
        st.markdown("### 📈 Multi-Step Ahead Recursive Forecasting")
        st.markdown(
            "In recursive forecasting, each single-step forecast updates the rolling 3-day and 7-day windows "
            "to simulate progressive future trajectory."
        )

        fc_col1, fc_col2 = st.columns([1, 3])
        with fc_col1:
            horizon_choice = st.radio("Forecast Horizon:", options=[7, 14, 30], index=0, format_func=lambda x: f"{x}-Day Forecast")
            start_date_choice = st.date_input("Forecast Starting Date:", value=date(2026, 1, 1))

        forecast_df = generate_recursive_forecast(
            model=model,
            feature_names=feature_names,
            daily_df=daily_df,
            horizon_days=horizon_choice,
            last_date=pd.Timestamp(start_date_choice) - timedelta(days=1),
        )

        summary = get_forecast_summary(forecast_df)

        with fc_col2:
            st.markdown("#### 📊 Forecast Overview")
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                st.metric("Total Expected Revenue", f"₹{summary.get('total_predicted_revenue', 0):,.2f}")
            with f2:
                st.metric("Daily Average", f"₹{summary.get('avg_daily_sales', 0):,.2f}")
            with f3:
                st.metric("Peak Day", f"₹{summary.get('max_sales', 0):,.2f}", help=f"{summary.get('peak_date')} ({summary.get('peak_day_name')})")
            with f4:
                st.metric("Weekend Avg", f"₹{summary.get('weekend_avg', 0):,.2f}", delta=f"₹{summary.get('weekend_avg', 0) - summary.get('weekday_avg', 0):+,.2f} vs Weekday")

        st.markdown("---")
        st.markdown("#### 🔮 Forecast Trajectory with Uncertainty Bounds")

        # Combine historical and forecast for visual
        chart_data = prepare_forecast_chart_data(daily_df, forecast_df, lookback_days=21)

        fig_fc = go.Figure()

        # Historical line
        hist_part = chart_data[chart_data["type"] == "Historical Actual"]
        fig_fc.add_trace(go.Scatter(
            x=hist_part["date"],
            y=hist_part["actual_sales"],
            name="Actual Sales (Past 3 Weeks)",
            mode="lines+markers",
            line=dict(color="#F5F0E8", width=2.5),
        ))

        # Forecast upper & lower bounds (fill)
        fc_part = chart_data[chart_data["type"] == "AI Forecast"]
        fig_fc.add_trace(go.Scatter(
            x=fc_part["date"],
            y=fc_part["upper_bound"],
            mode="lines",
            line=dict(color="rgba(200,149,108,0.2)", width=0),
            showlegend=False,
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_part["date"],
            y=fc_part["lower_bound"],
            mode="lines",
            line=dict(color="rgba(200,149,108,0.2)", width=0),
            fill="tonexty",
            fillcolor="rgba(200,149,108,0.18)",
            name="Forecast Uncertainty Range",
        ))

        # Forecast center line
        fig_fc.add_trace(go.Scatter(
            x=fc_part["date"],
            y=fc_part["predicted_sales"],
            name=f"{horizon_choice}-Day AI Forecast",
            mode="lines+markers",
            line=dict(color="#C8956C", width=3, dash="dash"),
            marker=dict(size=7, color="#E8C49A"),
        ))

        fig_fc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,15,10,0.8)",
            font=dict(color="#F5F0E8"),
            xaxis=dict(title="Date", gridcolor="#3a2a20"),
            yaxis=dict(title="Daily Sales (₹)", gridcolor="#3a2a20"),
            legend=dict(bgcolor="rgba(44,24,16,0.8)", bordercolor="rgba(200,149,108,0.4)"),
            margin=dict(l=40, r=30, t=40, b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown(f"#### 📋 {horizon_choice}-Day Daily Breakdown Table")
        display_fc = forecast_df[["step", "date", "day_name", "predicted_sales", "lower_bound", "upper_bound", "previous_day_sales", "previous_7day_avg"]].copy()
        st.dataframe(
            display_fc.style.format({
                "predicted_sales": "₹{:,.2f}",
                "lower_bound": "₹{:,.2f}",
                "upper_bound": "₹{:,.2f}",
                "previous_day_sales": "₹{:,.2f}",
                "previous_7day_avg": "₹{:,.2f}",
            }),
            use_container_width=True,
        )

        csv_fc = display_fc.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"📥 Download {horizon_choice}-Day Forecast (CSV)",
            data=csv_fc,
            file_name=f"coffeesense_{horizon_choice}day_forecast.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
