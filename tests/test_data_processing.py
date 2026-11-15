"""
Unit tests for data processing module.
"""

import unittest
import pandas as pd
import numpy as np

from config.settings import DAILY_DATASET_PATH, TRANSACTIONS_DATASET_PATH
from src.data_processing import (
    load_daily_dataset,
    load_transactions_dataset,
    clean_daily_dataset,
    clean_transactions_dataset,
    create_date_column,
    validate_daily_dataset,
    validate_transactions_dataset,
    cross_validate_datasets,
)


class TestDataProcessing(unittest.TestCase):

    def test_load_and_clean_daily(self):
        df_raw = load_daily_dataset(DAILY_DATASET_PATH)
        self.assertFalse(df_raw.empty)
        self.assertEqual(len(df_raw), 365)

        df_clean = clean_daily_dataset(df_raw)
        self.assertEqual(len(df_clean), 365)
        self.assertFalse(df_clean.isna().any().any())

        df_with_date = create_date_column(df_clean)
        self.assertIn("date", df_with_date.columns)
        self.assertEqual(str(df_with_date["date"].iloc[0])[:10], "2025-01-01")
        self.assertEqual(str(df_with_date["date"].iloc[-1])[:10], "2025-12-31")

    def test_load_and_clean_transactions(self):
        df_trans = load_transactions_dataset(TRANSACTIONS_DATASET_PATH)
        self.assertFalse(df_trans.empty)
        self.assertEqual(len(df_trans), 6592)

        df_trans_clean = clean_transactions_dataset(df_trans)
        self.assertIn("datetime", df_trans_clean.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_trans_clean["datetime"]))

    def test_integrity_validation(self):
        daily_df = load_daily_dataset(DAILY_DATASET_PATH)
        trans_df = load_transactions_dataset(TRANSACTIONS_DATASET_PATH)
        daily_val = validate_daily_dataset(daily_df)
        trans_val = validate_transactions_dataset(trans_df)
        cross_val = cross_validate_datasets(daily_df, trans_df)

        self.assertIsInstance(daily_val, dict)
        self.assertIsInstance(trans_val, dict)
        self.assertIsInstance(cross_val, dict)
        self.assertEqual(daily_val["shape"][0], 365)
        self.assertEqual(trans_val["shape"][0], 6592)
        self.assertTrue(daily_val["row_count_ok"])
        self.assertTrue(trans_val["row_count_ok"])


if __name__ == "__main__":
    unittest.main()
