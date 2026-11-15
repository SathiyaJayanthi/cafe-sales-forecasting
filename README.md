# ☕ Cafe — Intelligent Coffee Shop Sales Forecasting & Analytics System

> **"Predict Smarter. Brew Better. Grow Faster."**

Cafe is an enterprise-grade, locally running Machine Learning and Analytics web application built specifically for specialty coffee shops and café chains. It combines historical time-series econometric modeling, supervised machine learning, dynamic sales dashboards, an intelligent natural language chatbot assistant, and multi-step recursive forecasting into a unified, warm coffee-themed SaaS interface.

---

## 🌟 Key Application Modules

1. **Executive Sales Dashboard (`app.py` & `pages/1_Dashboard.py`)**
   - Real-time dynamic KPI cards (Total Revenue, Average Daily Sales, Peak Day, Trough Day, Total Orders, Order Velocity).
   - Date range, month, and day-type filters.
   - Interactive Plotly visualizations: daily sales trend with 7-day rolling moving average, monthly revenue comparisons, weekend vs. weekday distributions, and transaction volume trends.
   - Automated AI Executive Observations and Prioritized Operational Recommendations.

2. **Data Visualization & Analytics (`pages/2_Analytics.py`)**
   - 10 interactive Plotly charts organized across 5 analytical tabs:
     - Revenue Dynamics (line trends & monthly bar).
     - Product Breakdown (bestsellers by volume and revenue contribution).
     - Customer Payments (cash vs. card share, digital penetration).
     - Operational Timing (hourly rush distributions & peak windows).
     - Statistical Diagnostics (weekday box plots, sales histograms, and model residuals).

3. **CoffeeSense AI Assistant (`pages/3_AI_Assistant.py` & `src/chatbot.py`)**
   - Natural language conversational advisor backed by actual café data and model metadata.
   - Context-aware intent matching for financial inquiries, product bestsellers, weekday vs. weekend uplift, R² / RMSE interpretations, tomorrow's forecast, and strategic revenue growth tips.
   - Dual-mode architecture: comprehensive offline rule-based engine + optional OpenAI API integration.

4. **Dataset Explorer & Quality Inspector (`pages/4_Dataset_Explorer.py`)**
   - Inspection workspace for both the 365-day Daily Sales dataset and the 6,592-order Raw Transactions dataset.
   - Data schema, missingness checks, search filters, and one-click CSV data export.
   - Full Cross-Validation report verifying mathematical consistency between daily summaries and raw itemized transactions.

5. **Manual Sales Prediction & Testing Workspace (`pages/5_Prediction_Workspace.py`)**
   - Interactive scenario simulation: adjust calendar properties and historical lag sales.
   - 4 Quick-autofill presets (Historical Average, Yesterday's Actuals, Slow Trough, Peak Rush).
   - Instant single-day prediction with AI feature explanation and confidence rating.
   - Session-level prediction audit log with CSV download.
   - Chronological out-of-sample test holdout evaluator (day-by-day error inspector).
   - 7-day, 14-day, and 30-day recursive multi-step forecasting with expanding uncertainty bounds.

6. **ML Model Architecture & Performance Center (`pages/6_Model_Information.py`)**
   - Detailed machine learning specifications and feature engineering methodology.
   - Benchmark comparison table comparing Random Forest, Decision Tree, Linear Regression, Gradient Boosting, and a Naive Previous-Day baseline.
   - Feature importance and coefficient impact breakdown.
   - Self-contained, safe local retraining pipeline with automatic artifact persistence.

---

## 🏛️ System Architecture & Zero-Leakage Policy

```
CoffeeSense_AI/
├── app.py                      # Main landing page & Executive Dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
├── train_model.py              # Standalone local ML training script
│
├── config/
│   ├── __init__.py
│   └── settings.py             # Paths, hyperparameters, visual color palette
│
├── components/
│   ├── __init__.py
│   ├── styles.py               # Custom warm coffee CSS styling (espresso, caramel, cream)
│   ├── kpi_cards.py            # Reusable metric card renderer
│   ├── charts.py               # Production Plotly chart factory
│   └── sidebar.py              # Brand header & navigation
│
├── data/
│   ├── raw/                    # Original benchmark CSV datasets (365 daily, 6,592 raw)
│   └── processed/              # Cleaned & feature-engineered datasets
│
├── models/
│   ├── random_forest_model.pkl # Serialized trained regression model (Joblib)
│   ├── feature_names.json      # Ordered feature schema
│   └── model_metadata.json     # Training metrics, hyperparameters, evaluation results
│
├── pages/
│   ├── 1_Dashboard.py          # Executive Sales Dashboard
│   ├── 2_Analytics.py          # Deep Data Visualization & Diagnostics
│   ├── 3_AI_Assistant.py       # Conversational AI Assistant
│   ├── 4_Dataset_Explorer.py   # Dataset Quality & Schema Explorer
│   ├── 5_Prediction_Workspace.py # Manual Testing & Recursive Forecasting
│   └── 6_Model_Information.py  # Model Center & Retraining Pipeline
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Cleaning, parsing, integrity checks
│   ├── feature_engineering.py  # Lag/rolling calculation & zero-leakage enforcement
│   ├── model_training.py       # Multi-model training, baseline, artifact persistence
│   ├── model_evaluation.py     # MAE, RMSE, R2, MAPE calculations
│   ├── prediction.py           # Single & batch prediction services, explanations
│   ├── forecasting.py          # Multi-step recursive forecasting engine
│   ├── eda.py                  # Exploratory data analysis helpers
│   ├── chatbot.py              # CoffeeSenseBot NLP engine & fallback
│   └── business_insights.py    # Automated retail insights & recommendations
│
└── tests/
    ├── test_data_processing.py
    ├── test_feature_engineering.py
    ├── test_model.py
    └── test_prediction.py
```

### 🔒 Strict Data Leakage Prevention Guarantee
In retail sales forecasting, models must only use features that are strictly known **before** the trading day begins:
- ❌ **Disallowed**: `transaction_count`, `cash_transactions`, `card_transactions` (these are post-hoc daily outcomes).
- ✅ **Calendar Features**: `day`, `month`, `day_of_week`, `is_weekend` (known deterministically).
- ✅ **Historical Lags**:
  - `previous_day_sales` ($Y_{t-1}$)
  - `previous_3day_avg` ($\frac{1}{3}\sum_{i=1}^3 Y_{t-i}$)
  - `previous_7day_avg` ($\frac{1}{7}\sum_{i=1}^7 Y_{t-i}$)
- ⏱️ **Partitioning**: Strictly chronological (first 80% train = 286 days, last 20% test = 72 days). Time-series data is **never** randomly shuffled.

---

## 📊 Benchmark Model Performance

All models were evaluated on the identical 20% chronological holdout test period (72 days):

| Algorithm | RMSE (₹) | MAE (₹) | R² Score | MAPE (%) | vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **₹164.84** | **₹135.59** | **0.0893** | **22.25%** | **+9.9%** |
| **Random Forest Regressor** | ₹173.81 | ₹140.19 | -0.0125 | 22.47% | +5.0% |
| **Gradient Boosting** | ₹182.49 | ₹146.18 | -0.1161 | 23.56% | +0.3% |
| **Naive Previous-Day Baseline** | ₹182.96 | ₹148.35 | -0.1218 | 26.94% | 0.0% (Ref) |
| **Decision Tree Regressor** | ₹217.02 | ₹179.64 | -0.5785 | 29.55% | -18.6% |

*Note: All algorithms beat the naive previous-day baseline.*

---

## 🚀 Local Installation & Execution (Windows)

### 1. Requirements
- Python 3.10+ (Recommended: Python 3.12 or 3.13)
- Windows PowerShell or Command Prompt

### 2. Setup Environment & Dependencies
```powershell
# Navigate to project directory
cd E:\cafe_sales\CoffeeSense_AI

# Install required dependencies
pip install -r requirements.txt
```

### 3. Train the Model Artifacts (Optional — Pre-trained artifacts are included)
```powershell
python train_model.py
```

### 4. Run the Unit Test Suite
```powershell
python -m unittest discover tests
```

### 5. Launch the Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the full CoffeeSense AI platform.

---

## ⚠️ Benchmark Dataset Limitation Notice

The provided datasets (`coffee_shop_sales_dataset_2025_daily.csv` and `coffee_shop_transactions_raw_2025.csv`) represent a synthetic retail benchmark designed for sales forecasting research and system evaluation. The predicted values, customer traffic metrics, and simulated patterns reflect benchmark distributions.
