"""
Unit tests for feature engineering and data leakage validation.
"""

import unittest
import pandas as pd
import numpy as np

from config.settings import DAILY_DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN
from src.data_processing import load_daily_dataset, clean_daily_dataset, create_date_column
from src.feature_engineering import (
    create_features,
    split_features_target,
    chronological_train_test_split,
    validate_no_leakage,
)


class TestFeatureEngineering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df_raw = load_daily_dataset(DAILY_DATASET_PATH)
        df_clean = clean_daily_dataset(df_raw)
        cls.daily_df = create_date_column(df_clean)

    def test_feature_creation(self):
        feat_df = create_features(self.daily_df)
        # Because of 7-day rolling window, first 7 rows are dropped
        self.assertEqual(len(feat_df), 358)

        expected_cols = [
            "day",
            "month",
            "day_of_week",
            "is_weekend",
            "previous_day_sales",
            "previous_3day_avg",
            "previous_7day_avg",
            "daily_sales",
        ]
        for col in expected_cols:
            self.assertIn(col, feat_df.columns)

    def test_no_data_leakage(self):
        feat_df = create_features(self.daily_df)
        X, y = split_features_target(feat_df)

        # Valid feature set should not raise
        validate_no_leakage(list(X.columns))

        # Leaking columns should raise ValueError
        with self.assertRaises(ValueError):
            validate_no_leakage(["day", "transaction_count"])

        with self.assertRaises(ValueError):
            validate_no_leakage(["day", "cash_transactions"])

        # Ensure future/same-day counts are not in features
        for forbidden in ["transaction_count", "cash_transactions", "card_transactions"]:
            self.assertNotIn(forbidden, X.columns)

    def test_chronological_split(self):
        feat_df = create_features(self.daily_df)
        X, y = split_features_target(feat_df)
        X_train, X_test, y_train, y_test = chronological_train_test_split(X, y, test_size=0.2)

        self.assertEqual(len(X_train), 286)
        self.assertEqual(len(X_test), 72)
        self.assertEqual(len(y_train), 286)
        self.assertEqual(len(y_test), 72)

        # Verify no overlap and strict chronological ordering
        self.assertEqual(X_train.index[-1] + 1, X_test.index[0])


if __name__ == "__main__":
    unittest.main()
