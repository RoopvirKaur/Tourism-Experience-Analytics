"""
Unit tests for Streamlit Web Application modules and cached loaders.
"""

import unittest
from pathlib import Path
import os
import joblib
import pandas as pd

from src.data.preprocessor import build_consolidated_dataset


class TestStreamlitAppModules(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.app_dir = cls.project_root / "src" / "app"

    def test_app_files_exist(self):
        """Test that all required Streamlit app pages exist."""
        required_pages = [
            "main.py",
            "pages/01_EDA_Dashboard.py",
            "pages/02_Visit_Mode_Predictor.py",
            "pages/03_Rating_Predictor.py",
            "pages/04_Attraction_Recommender.py"
        ]
        for rel_path in required_pages:
            filepath = self.app_dir / rel_path
            self.assertTrue(filepath.exists(), f"Streamlit app page missing: {filepath}")

    def test_data_and_artifact_loading(self):
        """Verify that preprocessor and serialized artifacts load correctly for the app."""
        master_df, users_df, items_df = build_consolidated_dataset()
        self.assertGreater(len(master_df), 0)
        self.assertGreater(len(users_df), 0)
        self.assertGreater(len(items_df), 0)

        artifacts_dir = self.project_root / "artifacts"
        reg_bundle = joblib.load(artifacts_dir / "regression_best_model.pkl")
        cls_bundle = joblib.load(artifacts_dir / "classification_best_model.pkl")
        rec_bundle = joblib.load(artifacts_dir / "recommender.pkl")

        self.assertIn("model", reg_bundle)
        self.assertIn("model", cls_bundle)
        self.assertIn("hybrid_recommender", rec_bundle)


if __name__ == "__main__":
    unittest.main()
