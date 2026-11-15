"""
CoffeeSense AI - Model Evaluation Module
Computes regression metrics, compares models, and extracts feature importances.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-model metrics
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, model_name: str = "") -> dict:
    """
    Calculate regression evaluation metrics.

    MAPE is computed on non-zero elements only to avoid division by zero.

    Parameters
    ----------
    y_true      : array-like — ground-truth target values
    y_pred      : array-like — model predictions
    model_name  : str        — optional label for the results dict

    Returns
    -------
    dict with keys: model_name, MAE, MSE, RMSE, R2, MAPE
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE — skip zero actuals to avoid division by zero
    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        )
    else:
        mape = float("nan")

    metrics = {
        "model_name": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }
    logger.info(
        "Metrics [%s] — MAE: %.2f | RMSE: %.2f | R²: %.4f | MAPE: %.2f%%",
        model_name or "unnamed", mae, rmse, r2, mape if not np.isnan(mape) else -1,
    )
    return metrics


# ---------------------------------------------------------------------------
# Multi-model evaluation
# ---------------------------------------------------------------------------

def evaluate_all_models(models_dict: dict, X_test, y_test) -> dict:
    """
    Run calculate_metrics for every fitted model in models_dict.

    Parameters
    ----------
    models_dict : dict  — {model_name: fitted_estimator}
    X_test      : pd.DataFrame or array
    y_test      : array-like

    Returns
    -------
    dict of {model_name: metrics_dict}
    """
    results = {}
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        results[name] = calculate_metrics(y_test, y_pred, model_name=name)
    return results


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def compare_with_baseline(evaluation_results: dict, baseline_rmse: float) -> dict:
    """
    Rank models by RMSE and show improvement over naive baseline.

    Parameters
    ----------
    evaluation_results : dict  — {model_name: metrics_dict}
    baseline_rmse      : float — RMSE of the naive baseline predictor

    Returns
    -------
    dict with keys:
        ranking          : list of dicts sorted by RMSE ascending
        baseline_rmse    : float
        best_model       : str
        best_improvement : float  (% improvement over baseline)
    """
    # Build ranked list
    ranked = sorted(
        [
            {
                "model_name": name,
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "R2": m["R2"],
                "MAPE": m["MAPE"],
                "improvement_vs_baseline_pct": (
                    (baseline_rmse - m["RMSE"]) / baseline_rmse * 100
                    if baseline_rmse > 0 else 0.0
                ),
            }
            for name, m in evaluation_results.items()
        ],
        key=lambda x: x["RMSE"],
    )

    best = ranked[0] if ranked else {}
    return {
        "ranking": ranked,
        "baseline_rmse": baseline_rmse,
        "best_model": best.get("model_name", ""),
        "best_improvement": best.get("improvement_vs_baseline_pct", 0.0),
    }


# ---------------------------------------------------------------------------
# Convenience helper used by retrain_pipeline
# ---------------------------------------------------------------------------

def naive_baseline_rmse(y_train, y_test) -> float:
    """
    Compute RMSE for the naive baseline (predict last training value for all test points).
    """
    from src.model_training import naive_baseline_predict
    y_pred = naive_baseline_predict(y_train, len(y_test))
    metrics = calculate_metrics(y_test, y_pred, model_name="Naive Baseline")
    return metrics["RMSE"]


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from a tree-based model.

    Returns a DataFrame with columns ['feature', 'importance'] sorted
    by importance descending. Returns an empty DataFrame if the model
    does not expose feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning(
            "Model %s does not have feature_importances_. Returning empty DataFrame.",
            type(model).__name__,
        )
        return pd.DataFrame(columns=["feature", "importance"])

    importances = model.feature_importances_
    df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info("Feature importances computed for %d features.", len(df))
    return df


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def get_residuals(y_true, y_pred) -> np.ndarray:
    """
    Compute residuals (y_true - y_pred).

    Returns
    -------
    np.ndarray
    """
    return np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
