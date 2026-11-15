"""
Unit tests for prediction and recursive forecasting services.
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
from src.model_training import load_model_artifacts
from src.prediction import validate_prediction_inputs, predict_single, get_prediction_explanation
from src.forecasting import generate_recursive_forecast, get_forecast_summary


class TestPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.daily_df = create_date_column(clean_daily_dataset(load_daily_dataset(DAILY_DATASET_PATH)))
        cls.model, cls.feature_names, cls.metadata = load_model_artifacts(
            model_path=MODEL_PATH,
            feature_names_path=FEATURE_NAMES_PATH,
            metadata_path=METADATA_PATH,
        )

    def test_validate_inputs(self):
        # Valid inputs
        valid, msg = validate_prediction_inputs(15, 6, 2, 0, 500.0, 520.0, 510.0)
        self.assertTrue(valid)
        self.assertEqual(msg, "")

        # Invalid day
        valid, msg = validate_prediction_inputs(35, 6, 2, 0, 500.0, 520.0, 510.0)
        self.assertFalse(valid)

        # Negative sales
        valid, msg = validate_prediction_inputs(15, 6, 2, 0, -50.0, 520.0, 510.0)
        self.assertFalse(valid)

    def test_predict_single(self):
        result = predict_single(
            model=self.model,
            feature_names=self.feature_names,
            day=15,
            month=6,
            day_of_week=2,
            is_weekend=0,
            prev_day_sales=600.0,
            prev_3day_avg=620.0,
            prev_7day_avg=610.0,
        )
        self.assertIn("prediction", result)
        self.assertGreater(result["prediction"], 0)
        self.assertIn("model_name", result)

        explanation = get_prediction_explanation(
            result["prediction"],
            result["features_used"],
            avg_sales=650.0,
        )
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 20)

    def test_recursive_forecast(self):
        forecast_df = generate_recursive_forecast(
            model=self.model,
            feature_names=self.feature_names,
            daily_df=self.daily_df,
            horizon_days=7,
        )
        self.assertEqual(len(forecast_df), 7)
        self.assertIn("predicted_sales", forecast_df.columns)
        self.assertIn("lower_bound", forecast_df.columns)
        self.assertIn("upper_bound", forecast_df.columns)
        self.assertTrue((forecast_df["predicted_sales"] >= 0).all())

        summary = get_forecast_summary(forecast_df)
        self.assertIn("total_predicted_revenue", summary)
        self.assertEqual(summary["horizon_days"], 7)


if __name__ == "__main__":
    unittest.main()
