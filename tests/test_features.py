"""
Unit tests for Feature Engineering Pipeline.
"""

import unittest
import numpy as np
import pandas as pd

from src.features.build_features import build_all_features, FeaturePipeline


class TestFeatureEngineering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.features = build_all_features()

    def test_feature_matrix_shapes(self):
        X_reg = self.features["X_reg"]
        y_reg = self.features["y_reg"]
        X_cls = self.features["X_cls"]
        y_cls = self.features["y_cls"]

        self.assertEqual(len(X_reg), 52930)
        self.assertEqual(len(y_reg), 52930)
        self.assertEqual(len(X_cls), 52930)
        self.assertEqual(len(y_cls), 52930)

        # Check no NaN values in prepared feature matrices
        self.assertFalse(X_reg.isnull().any().any(), "X_reg contains NaN values")
        self.assertFalse(X_cls.isnull().any().any(), "X_cls contains NaN values")

    def test_tfidf_matrix_shape(self):
        tfidf_matrix = self.features["tfidf_matrix"]
        self.assertEqual(tfidf_matrix.shape[0], 1698)
        self.assertGreater(tfidf_matrix.shape[1], 0)

    def test_user_item_matrix_shape(self):
        user_item_matrix = self.features["user_item_matrix"]
        self.assertEqual(user_item_matrix.shape[0], 33530)
        self.assertEqual(user_item_matrix.shape[1], 30)
        self.assertFalse(user_item_matrix.isnull().any().any())

    def test_aggregates(self):
        user_stats = self.features["user_stats"]
        item_stats = self.features["item_stats"]

        self.assertIn("User_Avg_Rating", user_stats.columns)
        self.assertIn("User_Total_Visits", user_stats.columns)
        self.assertIn("Attraction_Avg_Rating", item_stats.columns)
        self.assertIn("Attraction_Total_Reviews", item_stats.columns)


if __name__ == "__main__":
    unittest.main()
