"""
Unit tests for Phase 4 Machine Learning Models, Recommendation System, and Target Leakage Prevention.
"""

import unittest
from pathlib import Path
import os
import joblib
import pandas as pd
import numpy as np

from src.data.preprocessor import build_consolidated_dataset


class TestPhase4Models(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts"
        cls.master_df, cls.users_df, cls.items_df = build_consolidated_dataset()

        # Load saved artifacts
        cls.reg_bundle_path = cls.artifacts_dir / "regression_best_model.pkl"
        cls.cls_bundle_path = cls.artifacts_dir / "classification_best_model.pkl"
        cls.label_enc_path = cls.artifacts_dir / "classification_label_encoder.pkl"
        cls.rec_bundle_path = cls.artifacts_dir / "recommender.pkl"

    def test_saved_artifacts_exist(self):
        """Test that all required Phase 4 artifact files exist."""
        required_files = [
            "regression_best_model.pkl",
            "regression_results.csv",
            "classification_best_model.pkl",
            "classification_results.csv",
            "classification_label_encoder.pkl",
            "recommender.pkl",
            "model_comparison.csv",
            "phase4_model_summary.txt"
        ]
        for filename in required_files:
            filepath = self.artifacts_dir / filename
            self.assertTrue(filepath.exists(), f"Missing required artifact file: {filepath}")

    def test_no_target_leakage_in_features(self):
        """Verify strict absence of target leakage in predictor feature lists."""
        reg_bundle = joblib.load(self.reg_bundle_path)
        cls_bundle = joblib.load(self.cls_bundle_path)

        reg_features = reg_bundle["feature_names"]
        cls_features = cls_bundle["feature_names"]

        # Regression: target 'Rating' must NOT be in feature list
        self.assertNotIn("Rating", reg_features, "Target 'Rating' found in regression feature list!")

        # Classification: VisitMode related terms must NOT be in predictor features
        forbidden_cls_terms = ["VisitMode", "VisitModeName", "VisitModeId", "User_Favorite_VisitMode"]
        for feat in cls_features:
            for forbidden in forbidden_cls_terms:
                self.assertNotIn(
                    forbidden, feat,
                    f"Target leakage! Classification feature '{feat}' contains forbidden target term '{forbidden}'"
                )

    def test_regression_prediction_pipeline(self):
        """Test regression model inference on sample dataframe rows."""
        reg_bundle = joblib.load(self.reg_bundle_path)
        model = reg_bundle["model"]
        preprocessor = reg_bundle["preprocessor"]

        sample_df = self.master_df.head(10).copy()
        X_sample = preprocessor.transform(sample_df)
        preds = model.predict(X_sample)

        self.assertEqual(len(preds), 10)
        self.assertTrue(np.all(np.isfinite(preds)))
        # Check predictions are clipped/within plausible rating bounds [1.0, 5.0]
        preds_clipped = np.clip(preds, 1.0, 5.0)
        self.assertGreaterEqual(preds_clipped.min(), 1.0)
        self.assertLessEqual(preds_clipped.max(), 5.0)

    def test_classification_prediction_pipeline(self):
        """Test classification model inference and label decoding."""
        cls_bundle = joblib.load(self.cls_bundle_path)
        model = cls_bundle["model"]
        preprocessor = cls_bundle["preprocessor"]
        target_encoder = cls_bundle["target_encoder"]

        sample_df = self.master_df.head(10).copy()
        X_sample, _ = preprocessor.transform(sample_df)
        preds_class_idx = model.predict(X_sample)
        pred_labels = target_encoder.inverse_transform(preds_class_idx)

        self.assertEqual(len(preds_class_idx), 10)
        self.assertEqual(len(pred_labels), 10)
        for label in pred_labels:
            self.assertIn(label, target_encoder.classes_)

    def test_recommendation_system_existing_user(self):
        """Test recommendation generation for an existing user with history."""
        rec_bundle = joblib.load(self.rec_bundle_path)
        hybrid_rec = rec_bundle["hybrid_recommender"]

        # Get an active user ID
        existing_uid = list(hybrid_rec.user_history.keys())[0]
        visited_ids = set(aid for aid, r in hybrid_rec.user_history[existing_uid])

        recs = hybrid_rec.recommend(user_id=existing_uid, top_k=5)

        self.assertIsInstance(recs, pd.DataFrame)
        self.assertEqual(len(recs), 5)
        self.assertIn("Attraction", recs.columns)
        self.assertIn("Score", recs.columns)

        # Confirm NO recommended attraction is in user's visited list
        rec_ids = set(recs["AttractionId"])
        intersection = rec_ids.intersection(visited_ids)
        self.assertEqual(
            len(intersection), 0,
            f"Recommender suggested already visited attractions {intersection} for user {existing_uid}"
        )

    def test_recommendation_system_cold_start_new_user(self):
        """Test cold-start recommendation generation for a new/unknown user."""
        rec_bundle = joblib.load(self.rec_bundle_path)
        hybrid_rec = rec_bundle["hybrid_recommender"]

        unknown_uid = 999999999
        recs = hybrid_rec.recommend(user_id=unknown_uid, top_k=5)

        self.assertIsInstance(recs, pd.DataFrame)
        self.assertEqual(len(recs), 5)
        self.assertIn("Attraction", recs.columns)
        self.assertIn("Popularity Fallback", recs["Recommendation_Type"].iloc[0])


if __name__ == "__main__":
    unittest.main()
