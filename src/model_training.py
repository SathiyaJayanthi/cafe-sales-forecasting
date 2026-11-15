"""
CoffeeSense AI - Model Training Module
Trains multiple regression models, selects the best one, and persists artifacts.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    CLEANED_DAILY_PATH,
    DAILY_DATASET_PATH,
    FEATURE_NAMES_PATH,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RF_PARAMS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

def get_model_configs() -> dict:
    """Return a dict of {model_name: unfitted estimator}."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=RF_PARAMS["n_estimators"],
            max_depth=RF_PARAMS["max_depth"],
            min_samples_leaf=RF_PARAMS["min_samples_leaf"],
            random_state=RF_PARAMS["random_state"],
        ),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8,
            random_state=42,
        ),
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42,
        ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Fit every model returned by get_model_configs() on (X_train, y_train).

    Returns
    -------
    dict of {model_name: fitted_model}
    """
    configs = get_model_configs()
    fitted = {}
    for name, model in configs.items():
        logger.info("Training model: %s …", name)
        model.fit(X_train, y_train)
        fitted[name] = model
        logger.info("  %s — training complete.", name)
    return fitted


# ---------------------------------------------------------------------------
# Naive baseline
# ---------------------------------------------------------------------------

def naive_baseline_predict(y_train: pd.Series, n_test: int) -> np.ndarray:
    """
    Naive baseline: repeat the last training value for all test predictions.

    Parameters
    ----------
    y_train : pd.Series   — training target values
    n_test  : int         — number of predictions to generate

    Returns
    -------
    np.ndarray of shape (n_test,)
    """
    last_value = float(y_train.iloc[-1])
    return np.full(n_test, last_value)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_best_model(evaluation_results: dict) -> str:
    """
    Return the name of the model with the lowest RMSE.

    Parameters
    ----------
    evaluation_results : dict
        {model_name: {RMSE: float, ...}, ...}

    Returns
    -------
    str — best model name
    """
    best_name = min(
        evaluation_results,
        key=lambda name: evaluation_results[name].get("RMSE", float("inf")),
    )
    logger.info(
        "Best model selected: '%s'  (RMSE=%.4f)",
        best_name,
        evaluation_results[best_name].get("RMSE"),
    )
    return best_name


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_model_artifacts(
    model,
    model_name: str,
    feature_names: list,
    metrics: dict,
    training_metadata: dict,
    models_dir,
) -> None:
    """
    Persist model, metadata JSON, and feature names JSON to disk.

    The primary model is always saved as 'random_forest_model.pkl' regardless
    of the actual model name, so downstream pages can load it by a fixed path.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Model binary (fixed filename)
    model_path = models_dir / "random_forest_model.pkl"
    joblib.dump(model, model_path)
    logger.info("Model saved to: %s", model_path)

    # Feature names
    fn_path = models_dir / "feature_names.json"
    with open(fn_path, "w", encoding="utf-8") as fh:
        json.dump(feature_names, fh, indent=2)
    logger.info("Feature names saved to: %s", fn_path)

    # Metadata
    metadata = {
        "model_name": model_name,
        "saved_as": "random_forest_model.pkl",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "feature_names": feature_names,
        **training_metadata,
    }
    md_path = models_dir / "model_metadata.json"
    with open(md_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Model metadata saved to: %s", md_path)


def load_model_artifacts(
    models_dir=None,
    model_path=None,
    feature_names_path=None,
    metadata_path=None,
):
    """
    Load model, feature names, and metadata from disk.

    Returns
    -------
    (model, feature_names, metadata)

    Raises
    ------
    FileNotFoundError if any artifact is missing.
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    models_dir = Path(models_dir)
    m_path = Path(model_path) if model_path else (models_dir / "random_forest_model.pkl")
    fn_path = Path(feature_names_path) if feature_names_path else (models_dir / "feature_names.json")
    md_path = Path(metadata_path) if metadata_path else (models_dir / "model_metadata.json")

    for path in [m_path, fn_path, md_path]:
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

    model = joblib.load(m_path)
    with open(fn_path, "r", encoding="utf-8") as fh:
        feature_names = json.load(fh)
    with open(md_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    logger.info("Model artifacts loaded successfully from: %s", models_dir)
    return model, feature_names, metadata


# ---------------------------------------------------------------------------
# Full retrain pipeline
# ---------------------------------------------------------------------------

def retrain_pipeline(daily_df_path, models_dir, processed_dir) -> dict:
    """
    End-to-end retraining pipeline.

    Steps
    -----
    1. Load & clean daily dataset
    2. Create features
    3. Train / evaluate all models
    4. Select best model
    5. Save artifacts (only if training succeeded)

    Returns
    -------
    dict with keys: status ("success" | "error"), metrics, best_model, message
    """
    # Import here to avoid circular imports
    from src.data_processing import (
        clean_daily_dataset,
        create_date_column,
        load_daily_dataset,
        save_cleaned_dataset,
    )
    from src.feature_engineering import (
        chronological_train_test_split,
        create_features,
        get_feature_names,
        split_features_target,
    )
    from src.model_evaluation import evaluate_all_models, naive_baseline_rmse

    try:
        # 1. Load & clean
        daily_df = load_daily_dataset(daily_df_path)
        daily_df = clean_daily_dataset(daily_df)
        daily_df = create_date_column(daily_df)
        daily_df = daily_df.sort_values("date").reset_index(drop=True)

        # Save cleaned
        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        save_cleaned_dataset(daily_df, processed_dir / "cleaned_daily_sales.csv")

        # 2. Features
        featured_df = create_features(daily_df)
        X, y = split_features_target(featured_df)
        X_train, X_test, y_train, y_test = chronological_train_test_split(X, y)

        # 3. Train
        fitted_models = train_all_models(X_train, y_train)

        # 4. Evaluate
        from src.model_evaluation import evaluate_all_models
        eval_results = evaluate_all_models(fitted_models, X_test, y_test)

        # 5. Select best
        best_name = select_best_model(eval_results)
        best_model = fitted_models[best_name]
        best_metrics = eval_results[best_name]

        # 6. Save
        training_metadata = {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "all_model_metrics": eval_results,
        }
        save_model_artifacts(
            model=best_model,
            model_name=best_name,
            feature_names=get_feature_names(),
            metrics=best_metrics,
            training_metadata=training_metadata,
            models_dir=models_dir,
        )

        return {
            "status": "success",
            "best_model": best_name,
            "metrics": best_metrics,
            "all_metrics": eval_results,
            "message": f"Model '{best_name}' trained and saved successfully.",
        }

    except Exception as exc:
        logger.error("retrain_pipeline failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "best_model": None,
            "metrics": {},
            "all_metrics": {},
            "message": str(exc),
        }
