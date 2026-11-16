"""
CoffeeSense AI - Project Settings
Centralised configuration for paths, model parameters, and app constants.
"""

import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # e:\cafe_sales\CoffeeSense_AI

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
SRC_DIR = BASE_DIR / "src"

# Ensure critical directories exist at import time
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset file paths
# ---------------------------------------------------------------------------
DAILY_DATASET_PATH = RAW_DATA_DIR / "coffee_shop_sales_dataset_2025_daily.csv"
TRANSACTIONS_DATASET_PATH = RAW_DATA_DIR / "coffee_shop_transactions_raw_2025.csv"
CLEANED_DAILY_PATH = PROCESSED_DATA_DIR / "cleaned_daily_sales.csv"

# ---------------------------------------------------------------------------
# Model artifact paths
# ---------------------------------------------------------------------------
MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"

# ---------------------------------------------------------------------------
# ML hyperparameters
# ---------------------------------------------------------------------------
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "min_samples_leaf": 4,
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_NAME = "Cafe"
TAGLINE = "Predict Smarter. Brew Better. Grow Faster."
VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Color palette (coffee-themed)
# ---------------------------------------------------------------------------
COLORS = {
    "espresso": "#2C1810",
    "cream": "#F5F0E8",
    "caramel": "#C8956C",
    "latte": "#D4A96A",
    "mocha": "#6B3A2A",
    "foam": "#FFF8F0",
    "dark_bg": "#1a0f0a",
    "accent": "#E8C49A",
}

# ---------------------------------------------------------------------------
# Feature / target definitions
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "previous_day_sales",
    "previous_3day_avg",
    "previous_7day_avg",
]

TARGET_COLUMN = "daily_sales"
