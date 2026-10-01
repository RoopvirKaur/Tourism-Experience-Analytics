"""
Classification Training Module for Tourism Experience Analytics.
Predicts user VisitMode (e.g., Business, Family, Couples, Friends, Solo) using Logistic Regression,
Random Forest, and XGBoost Classifiers. Includes class balancing, stratified splitting, and evaluation.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import build_consolidated_dataset


class ClassificationFeaturePreprocessor:
    """
    Preprocessor for VisitMode Classification task.
    Strictly excludes VisitMode, VisitModeName, VisitModeId, and User_Favorite_VisitMode from predictor features.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        self.cat_cols = [
            "ContinentId", "RegionId", "CountryId",
            "UserCityId", "AttractionTypeId"
        ]
        self.num_cols = [
            "VisitYear", "VisitMonth",
            "User_Avg_Rating", "User_Total_Visits",
            "Attraction_Avg_Rating", "Attraction_Total_Reviews"
        ]
        self.feature_names = []

    def _compute_aggregates(self, train_df: pd.DataFrame):
        """Computes user and attraction aggregate statistics strictly on training data."""
        user_grp = train_df.groupby("UserId")["Rating"]
        self.user_avg = user_grp.mean().to_dict()
        self.user_cnt = user_grp.count().to_dict()
        self.global_user_avg = float(train_df["Rating"].mean())

        item_grp = train_df.groupby("AttractionId")["Rating"]
        self.item_avg = item_grp.mean().to_dict()
        self.item_cnt = item_grp.count().to_dict()
        self.global_item_avg = self.global_user_avg

    def fit_transform(self, train_df: pd.DataFrame):
        df = train_df.copy()
        self._compute_aggregates(df)

        df["User_Avg_Rating"] = df["UserId"].map(self.user_avg).fillna(self.global_user_avg)
        df["User_Total_Visits"] = df["UserId"].map(self.user_cnt).fillna(0)
        df["Attraction_Avg_Rating"] = df["AttractionId"].map(self.item_avg).fillna(self.global_item_avg)
        df["Attraction_Total_Reviews"] = df["AttractionId"].map(self.item_cnt).fillna(0)

        # Categorical features
        encoded_cat_cols = []
        for col in self.cat_cols:
            enc_col = f"{col}_encoded"
            encoded_cat_cols.append(enc_col)
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        self.feature_names = encoded_cat_cols + self.num_cols

        # Scale numerical features
        X_num = self.scaler.fit_transform(df[self.num_cols].fillna(0))
        for idx, col in enumerate(self.num_cols):
            df[col] = X_num[:, idx]

        # Target variable
        target_series = df["VisitModeName"].astype(str).str.strip()
        y = self.target_encoder.fit_transform(target_series)

        return df[self.feature_names], y

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        df["User_Avg_Rating"] = df["UserId"].map(self.user_avg).fillna(self.global_user_avg)
        df["User_Total_Visits"] = df["UserId"].map(self.user_cnt).fillna(0)
        df["Attraction_Avg_Rating"] = df["AttractionId"].map(self.item_avg).fillna(self.global_item_avg)
        df["Attraction_Total_Reviews"] = df["AttractionId"].map(self.item_cnt).fillna(0)

        encoded_cat_cols = []
        for col in self.cat_cols:
            enc_col = f"{col}_encoded"
            encoded_cat_cols.append(enc_col)
            le = self.label_encoders[col]
            vals = df[col].astype(str)
            df[enc_col] = vals.map(
                lambda s: le.transform([s])[0] if s in le.classes_ else -1
            )

        X_num = self.scaler.transform(df[self.num_cols].fillna(0))
        for idx, col in enumerate(self.num_cols):
            df[col] = X_num[:, idx]

        if "VisitModeName" in df.columns:
            target_series = df["VisitModeName"].astype(str).str.strip()
            y = target_series.map(
                lambda s: self.target_encoder.transform([s])[0] if s in self.target_encoder.classes_ else -1
            ).values
        else:
            y = None

        return df[self.feature_names], y


def train_and_evaluate_classification(artifacts_dir: Path = None):
    """
    Main training workflow for classification models.
    """
    if artifacts_dir is None:
        artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading consolidated dataset for classification...")
    master_df, _, _ = build_consolidated_dataset()

    # Stratified Train (70%), Validation (15%), Test (15%) split
    target_str = master_df["VisitModeName"].astype(str).str.strip()
    train_df, temp_df = train_test_split(
        master_df, test_size=0.30, random_state=42, stratify=target_str
    )
    temp_target = temp_df["VisitModeName"].astype(str).str.strip()
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_target
    )

    print(f"Classification dataset splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    preprocessor = ClassificationFeaturePreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)

    # Confirm predictor feature list excludes VisitMode
    forbidden_terms = ["VisitMode", "VisitModeName", "VisitModeId", "User_Favorite_VisitMode"]
    for feat in preprocessor.feature_names:
        for forbidden in forbidden_terms:
            assert forbidden not in feat, f"Target leakage detected! Feature '{feat}' contains '{forbidden}'"

    print("Confirmed: Predictor feature list contains NO VisitMode target leakage.")
    print("Features used:", preprocessor.feature_names)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="mlogloss"
        )
    }

    results = []
    trained_models = {}
    reports = {}

    # Sample weights for XGBoost class balancing
    sample_weights_train = compute_sample_weight("balanced", y_train)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == "XGBoost Classifier":
            model.fit(X_train, y_train, sample_weight=sample_weights_train)
        else:
            model.fit(X_train, y_train)

        trained_models[name] = model

        preds_val = model.predict(X_val)
        preds_test = model.predict(X_test)

        val_acc = accuracy_score(y_val, preds_val)
        val_prec_macro = precision_score(y_val, preds_val, average="macro", zero_division=0)
        val_rec_macro = recall_score(y_val, preds_val, average="macro", zero_division=0)
        val_f1_macro = f1_score(y_val, preds_val, average="macro", zero_division=0)
        val_f1_weighted = f1_score(y_val, preds_val, average="weighted", zero_division=0)

        test_acc = accuracy_score(y_test, preds_test)
        test_f1_macro = f1_score(y_test, preds_test, average="macro", zero_division=0)
        test_f1_weighted = f1_score(y_test, preds_test, average="weighted", zero_division=0)

        results.append({
            "Model": name,
            "Val_Accuracy": val_acc,
            "Val_Precision_Macro": val_prec_macro,
            "Val_Recall_Macro": val_rec_macro,
            "Val_F1_Macro": val_f1_macro,
            "Val_F1_Weighted": val_f1_weighted,
            "Test_Accuracy": test_acc,
            "Test_F1_Macro": test_f1_macro,
            "Test_F1_Weighted": test_f1_weighted
        })

        target_names = preprocessor.target_encoder.classes_
        reports[name] = classification_report(
            y_val, preds_val, target_names=target_names, zero_division=0
        )

    results_df = pd.DataFrame(results)
    results_path = artifacts_dir / "classification_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved classification results to {results_path}")
    print(results_df[["Model", "Val_Accuracy", "Val_F1_Macro", "Val_F1_Weighted"]])

    # Select best model based on validation Macro F1
    best_row = results_df.loc[results_df["Val_F1_Macro"].idxmax()]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]

    print(f"\nBest Classification Model: {best_name} (Val Macro F1: {best_row['Val_F1_Macro']:.4f})")
    print(f"\nPer-class performance report for {best_name}:\n{reports[best_name]}")

    # Save label encoder artifact separately
    label_encoder_path = artifacts_dir / "classification_label_encoder.pkl"
    joblib.dump(preprocessor.target_encoder, label_encoder_path)
    print(f"Saved classification label encoder to {label_encoder_path}")

    # Save best model bundle artifact
    artifact_bundle = {
        "model_name": best_name,
        "model": best_model,
        "preprocessor": preprocessor,
        "target_encoder": preprocessor.target_encoder,
        "feature_names": preprocessor.feature_names,
        "val_accuracy": best_row["Val_Accuracy"],
        "val_f1_macro": best_row["Val_F1_Macro"],
        "val_f1_weighted": best_row["Val_F1_Weighted"]
    }
    model_path = artifacts_dir / "classification_best_model.pkl"
    joblib.dump(artifact_bundle, model_path)
    print(f"Saved best classification model artifact to {model_path}")

    return artifact_bundle, results_df


if __name__ == "__main__":
    train_and_evaluate_classification()
