"""
CoffeeSense AI - Standalone ML Training Script
Runs the full training pipeline and saves the best model artifacts.
"""

import os
import sys

# 1. Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 2. Import from config and src modules
from config.settings import (
    DAILY_DATASET_PATH,
    TRANSACTIONS_DATASET_PATH,
    CLEANED_DAILY_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.data_processing import (
    load_daily_dataset,
    load_transactions_dataset,
    create_date_column,
    clean_daily_dataset,
    clean_transactions_dataset,
    save_cleaned_dataset,
)
from src.feature_engineering import (
    create_features,
    get_feature_names,
    split_features_target,
    chronological_train_test_split,
    validate_no_leakage,
)
from src.model_training import (
    train_all_models,
    naive_baseline_predict,
    select_best_model,
    save_model_artifacts,
    load_model_artifacts,
)
from src.model_evaluation import (
    calculate_metrics,
    evaluate_all_models,
    compare_with_baseline,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_model")


def print_separator(char="=", width=70):
    print(char * width)


def print_comparison_table(eval_results: dict, baseline_rmse: float) -> None:
    """Print a formatted model comparison table."""
    print_separator()
    print(f"{'Model':<22} {'RMSE':>10} {'MAE':>10} {'R2':>8} {'MAPE%':>8}")
    print_separator("-")

    # Sort by RMSE ascending
    sorted_models = sorted(eval_results.items(), key=lambda x: x[1].get("RMSE", float("inf")))

    for name, metrics in sorted_models:
        rmse = metrics.get("RMSE", float("nan"))
        mae = metrics.get("MAE", float("nan"))
        r2 = metrics.get("R2", float("nan"))
        mape = metrics.get("MAPE", float("nan"))
        print(f"{name:<22} {rmse:>10.2f} {mae:>10.2f} {r2:>8.4f} {mape:>7.2f}%")

    # Baseline row
    print(f"{'Naive Baseline':<22} {baseline_rmse:>10.2f} {'N/A':>10} {'N/A':>8} {'N/A':>8}")
    print_separator()


def main():
    print_separator()
    print("  CoffeeSense AI — ML Training Pipeline")
    print_separator()

    # ------------------------------------------------------------------
    # 3. Load both datasets
    # ------------------------------------------------------------------
    print("\n[STEP 1] Loading datasets...")
    daily_df = load_daily_dataset(DAILY_DATASET_PATH)
    trans_df = load_transactions_dataset(TRANSACTIONS_DATASET_PATH)
    print(f"  Daily dataset:        {daily_df.shape[0]} rows × {daily_df.shape[1]} cols")
    print(f"  Transactions dataset: {trans_df.shape[0]} rows × {trans_df.shape[1]} cols")

    # ------------------------------------------------------------------
    # 4. Create date column on daily dataset
    # ------------------------------------------------------------------
    print("\n[STEP 2] Creating date column...")
    daily_df = create_date_column(daily_df)
    print(f"  Date range: {daily_df['date'].min().date()} → {daily_df['date'].max().date()}")

    # Sort by date ascending (required for lag features)
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Clean both datasets
    # ------------------------------------------------------------------
    print("\n[STEP 3] Cleaning datasets...")
    daily_df = clean_daily_dataset(daily_df)
    trans_df = clean_transactions_dataset(trans_df)
    print(f"  Daily cleaned shape:        {daily_df.shape}")
    print(f"  Transactions cleaned shape: {trans_df.shape}")

    # ------------------------------------------------------------------
    # 6. Create lag/rolling features
    # ------------------------------------------------------------------
    print("\n[STEP 4] Engineering features...")
    featured_df = create_features(daily_df)
    print(f"  Feature-engineered rows: {len(featured_df)}")
    print(f"  Feature columns: {list(featured_df.columns)}")

    # ------------------------------------------------------------------
    # 7. Split chronologically (80/20)
    # ------------------------------------------------------------------
    print("\n[STEP 5] Splitting data chronologically (80% train / 20% test)...")
    X, y = split_features_target(featured_df)
    X_train, X_test, y_train, y_test = chronological_train_test_split(X, y, test_size=0.2)
    print(f"  Training set:  {len(X_train)} samples")
    print(f"  Test set:      {len(X_test)} samples")

    # ------------------------------------------------------------------
    # 8. Validate no data leakage
    # ------------------------------------------------------------------
    print("\n[STEP 6] Validating no data leakage...")
    validate_no_leakage(list(X_train.columns))
    print("  ✓ No leakage columns detected.")

    # ------------------------------------------------------------------
    # 9. Train all 4 models
    # ------------------------------------------------------------------
    print("\n[STEP 7] Training all models...")
    fitted_models = train_all_models(X_train, y_train)
    print(f"  Trained models: {list(fitted_models.keys())}")

    # ------------------------------------------------------------------
    # 10. Evaluate all models + baseline
    # ------------------------------------------------------------------
    print("\n[STEP 8] Evaluating models...")
    eval_results = evaluate_all_models(fitted_models, X_test, y_test)

    # Naive baseline
    baseline_preds = naive_baseline_predict(y_train, len(y_test))
    baseline_metrics = calculate_metrics(y_test, baseline_preds, model_name="Naive Baseline")
    baseline_rmse = baseline_metrics["RMSE"]
    print(f"  Naive baseline RMSE: {baseline_rmse:.2f}")

    # ------------------------------------------------------------------
    # 11. Print full comparison table
    # ------------------------------------------------------------------
    print("\n[STEP 9] Model Comparison Table:")
    print_comparison_table(eval_results, baseline_rmse)

    # Detailed comparison with improvement %
    comparison = compare_with_baseline(eval_results, baseline_rmse)
    print("\nRanking (best to worst RMSE):")
    for i, entry in enumerate(comparison["ranking"], 1):
        improvement = entry.get("improvement_vs_baseline_pct", 0)
        print(f"  {i}. {entry['model_name']:<22} RMSE={entry['RMSE']:.2f}  "
              f"({'+' if improvement >= 0 else ''}{improvement:.1f}% vs baseline)")

    # ------------------------------------------------------------------
    # 12. Save best model artifacts
    # ------------------------------------------------------------------
    print("\n[STEP 10] Selecting & saving best model...")
    best_model_name = select_best_model(eval_results)
    best_model = fitted_models[best_model_name]
    best_metrics = eval_results[best_model_name]

    print(f"  Best model: {best_model_name}")
    print(f"    RMSE: {best_metrics['RMSE']:.2f}")
    print(f"    MAE:  {best_metrics['MAE']:.2f}")
    print(f"    R²:   {best_metrics['R2']:.4f}")
    print(f"    MAPE: {best_metrics['MAPE']:.2f}%")

    training_metadata = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "total_rows_after_feature_engineering": len(featured_df),
        "all_model_metrics": {
            name: {k: v for k, v in m.items() if k != "model_name"}
            for name, m in eval_results.items()
        },
        "baseline_rmse": baseline_rmse,
    }

    save_model_artifacts(
        model=best_model,
        model_name=best_model_name,
        feature_names=get_feature_names(),
        metrics={k: v for k, v in best_metrics.items() if k != "model_name"},
        training_metadata=training_metadata,
        models_dir=MODELS_DIR,
    )
    print(f"  ✓ Model artifacts saved to: {MODELS_DIR}")

    # ------------------------------------------------------------------
    # 13. Save cleaned daily dataset
    # ------------------------------------------------------------------
    print("\n[STEP 11] Saving cleaned daily dataset...")
    save_cleaned_dataset(daily_df, CLEANED_DAILY_PATH)
    print(f"  ✓ Cleaned daily dataset saved to: {CLEANED_DAILY_PATH}")

    # ------------------------------------------------------------------
    # 14. Training summary
    # ------------------------------------------------------------------
    print_separator()
    print("  TRAINING SUMMARY")
    print_separator()
    print(f"  Best model:      {best_model_name}")
    print(f"  RMSE:            {best_metrics['RMSE']:.2f}")
    print(f"  MAE:             {best_metrics['MAE']:.2f}")
    print(f"  R²:              {best_metrics['R2']:.4f}")
    print(f"  MAPE:            {best_metrics['MAPE']:.2f}%")
    print(f"  Baseline RMSE:   {baseline_rmse:.2f}")
    improvement = (baseline_rmse - best_metrics['RMSE']) / baseline_rmse * 100
    print(f"  Improvement:     {improvement:.1f}% over naive baseline")
    print(f"  Training rows:   {len(X_train)}")
    print(f"  Test rows:       {len(X_test)}")
    print_separator()
    print("  ✓ Training pipeline complete!")
    print_separator()

    return fitted_models, eval_results, baseline_rmse, best_model_name


if __name__ == "__main__":
    main()
