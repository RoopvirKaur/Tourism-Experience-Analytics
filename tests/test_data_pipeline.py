"""
Tests for Data Loader and Preprocessor modules.
"""

import unittest
from pathlib import Path
import pandas as pd

from src.data.loader import load_raw_data, get_default_data_dir
from src.data.preprocessor import build_consolidated_dataset


class TestDataIngestionAndCleaning(unittest.TestCase):

    def test_default_data_dir_exists(self):
        data_dir = get_default_data_dir()
        self.assertTrue(data_dir.exists(), f"Default data directory does not exist: {data_dir}")

    def test_load_raw_data(self):
        raw_data = load_raw_data()
        expected_keys = [
            "transaction", "user", "city", "continent",
            "country", "region", "type", "mode", "item"
        ]
        for key in expected_keys:
            self.assertIn(key, raw_data, f"Missing key '{key}' in raw_data")
            self.assertIsInstance(raw_data[key], pd.DataFrame, f"Value for '{key}' should be a DataFrame")
            self.assertFalse(raw_data[key].empty, f"DataFrame for '{key}' is empty")

    def test_build_consolidated_dataset(self):
        master_df, users_df, items_df = build_consolidated_dataset()

        # Check DataFrame instances and non-empty status
        self.assertIsInstance(master_df, pd.DataFrame)
        self.assertIsInstance(users_df, pd.DataFrame)
        self.assertIsInstance(items_df, pd.DataFrame)

        self.assertGreater(len(master_df), 0, "Master DataFrame should not be empty")
        self.assertGreater(len(users_df), 0, "Users DataFrame should not be empty")
        self.assertGreater(len(items_df), 0, "Items DataFrame should not be empty")

        # Check key columns in master DataFrame
        expected_columns = [
            "TransactionId", "UserId", "VisitYear", "VisitMonth",
            "AttractionId", "Rating", "Continent", "Country",
            "AttractionType", "Attraction"
        ]
        for col in expected_columns:
            self.assertIn(col, master_df.columns, f"Column '{col}' missing from master DataFrame")

        # Check Rating values are within valid range [1.0, 5.0]
        self.assertGreaterEqual(master_df["Rating"].min(), 1.0)
        self.assertLessEqual(master_df["Rating"].max(), 5.0)


if __name__ == "__main__":
    unittest.main()
