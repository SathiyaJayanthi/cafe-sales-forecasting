"""
Unit tests for model training and evaluation.
"""

import unittest
import numpy as np
import pandas as pd

from config.settings import (
    DAILY_DATASET_PATH,
    MODEL_PATH,
    FEATURE_NAMES_PATH,
    METADATA_PATH,
)
from src.data_processing import load_daily_dataset, clean_daily_dataset, create_date_column
from src.feature_engineering import create_features, split_features_target, chronological_train_test_split
from src.model_training import train_all_models, load_model_artifacts
from src.model_evaluation import calculate_metrics, evaluate_all_models


class TestModel(unittest.TestCase):

    def test_load_trained_artifacts(self):
        model, feature_names, metadata = load_model_artifacts(
            model_path=MODEL_PATH,
            feature_names_path=FEATURE_NAMES_PATH,
            metadata_path=METADATA_PATH,
        )
        self.assertIsNotNone(model)
        self.assertEqual(len(feature_names), 7)
        self.assertIn("metrics", metadata)
        self.assertIn("RMSE", metadata["metrics"])
        self.assertIn("MAE", metadata["metrics"])
        self.assertIn("R2", metadata["metrics"])
        self.assertIn("MAPE", metadata["metrics"])

    def test_calculate_metrics(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        metrics = calculate_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["MAE"], 10.0)
        self.assertAlmostEqual(metrics["RMSE"], 10.0)
        self.assertGreater(metrics["R2"], 0.9)
        self.assertAlmostEqual(metrics["MAPE"], 6.11, places=1)


if __name__ == "__main__":
    unittest.main()
