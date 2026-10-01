"""
Feature Engineering Module for Tourism Experience Analytics.
Handles categorical encoding, profile aggregations, attraction TF-IDF feature vectorization,
and scaling for Machine Learning tasks (Regression, Classification, Recommendation).
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.preprocessor import build_consolidated_dataset


class FeaturePipeline:
    """
    Feature engineering pipeline that builds aggregated user/item profiles,
    encodes categorical variables, scales numerical metrics, and builds TF-IDF matrices.
    """

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, Any] = {}
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.user_profiles: Optional[pd.DataFrame] = None
        self.item_profiles: Optional[pd.DataFrame] = None

    def compute_user_aggregates(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes aggregated behavioral profiles per user.
        """
        user_stats = master_df.groupby("UserId").agg(
            User_Avg_Rating=("Rating", "mean"),
            User_Total_Visits=("Rating", "count"),
            User_Favorite_VisitMode=("VisitModeId", lambda x: x.mode()[0] if not x.empty else 0)
        ).reset_index()

        self.user_profiles = user_stats
        return user_stats

    def compute_item_aggregates(self, master_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes aggregated review metrics per attraction item.
        """
        item_stats = master_df.groupby("AttractionId").agg(
            Attraction_Avg_Rating=("Rating", "mean"),
            Attraction_Total_Reviews=("Rating", "count")
        ).reset_index()

        # Merge with master items list
        item_profiles = items_df.merge(item_stats, on="AttractionId", how="left")
        item_profiles["Attraction_Avg_Rating"] = item_profiles["Attraction_Avg_Rating"].fillna(master_df["Rating"].mean())
        item_profiles["Attraction_Total_Reviews"] = item_profiles["Attraction_Total_Reviews"].fillna(0).astype(int)

        self.item_profiles = item_profiles
        return item_profiles

    def build_attraction_tfidf_matrix(self, items_df: pd.DataFrame) -> Tuple[np.ndarray, TfidfVectorizer]:
        """
        Builds a TF-IDF text feature representation of attractions for Content-Based Filtering.
        Combines Attraction Name, Attraction Type, City Name, and Address.
        """
        items_copy = items_df.copy()

        # Combine text fields into a rich metadata document per attraction
        combined_text = (
            items_copy["Attraction"].astype(str) + " " +
            items_copy["AttractionType"].astype(str) + " " +
            items_copy["AttractionCityName"].astype(str) + " " +
            items_copy["AttractionAddress"].astype(str)
        )

        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text).toarray()

        return tfidf_matrix, self.tfidf_vectorizer

    def prepare_regression_features(
        self, master_df: pd.DataFrame, is_train: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares feature matrix X and target y for the Rating Prediction Regression task.
        """
        df = master_df.copy()

        # Ensure aggregations are present
        if "User_Avg_Rating" not in df.columns or "Attraction_Avg_Rating" not in df.columns:
            if self.user_profiles is None:
                self.compute_user_aggregates(df)
            if self.item_profiles is None:
                items_df = df[["AttractionId", "Attraction", "AttractionType", "AttractionCityName", "AttractionAddress"]].drop_duplicates("AttractionId")
                self.compute_item_aggregates(df, items_df)

            df = df.merge(self.user_profiles, on="UserId", how="left")
            df = df.merge(self.item_profiles[["AttractionId", "Attraction_Avg_Rating", "Attraction_Total_Reviews"]], on="AttractionId", how="left")

        categorical_cols = ["ContinentId", "RegionId", "CountryId", "UserCityId", "AttractionTypeId", "VisitModeId"]
        num_cols = ["VisitYear", "VisitMonth", "User_Avg_Rating", "User_Total_Visits", "Attraction_Avg_Rating", "Attraction_Total_Reviews"]

        for col in categorical_cols:
            if col in df.columns:
                if is_train or col not in self.label_encoders:
                    le = LabelEncoder()
                    df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    df[f"{col}_encoded"] = df[col].astype(str).map(
                        lambda s: le.transform([s])[0] if s in le.classes_ else -1
                    )

        feature_cols = [f"{col}_encoded" for col in categorical_cols if f"{col}_encoded" in df.columns] + num_cols
        X = df[feature_cols].copy()
        y = df["Rating"].copy()

        if is_train:
            scaler = StandardScaler()
            X[num_cols] = scaler.fit_transform(X[num_cols].fillna(0))
            self.scalers["regression"] = scaler
        else:
            scaler = self.scalers.get("regression")
            if scaler:
                X[num_cols] = scaler.transform(X[num_cols].fillna(0))

        return X, y

    def prepare_classification_features(
        self, master_df: pd.DataFrame, is_train: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares feature matrix X and target y for the Visit Mode Classification task.
        """
        df = master_df.copy()

        if "User_Avg_Rating" not in df.columns or "Attraction_Avg_Rating" not in df.columns:
            if self.user_profiles is None:
                self.compute_user_aggregates(df)
            if self.item_profiles is None:
                items_df = df[["AttractionId", "Attraction", "AttractionType", "AttractionCityName", "AttractionAddress"]].drop_duplicates("AttractionId")
                self.compute_item_aggregates(df, items_df)

            df = df.merge(self.user_profiles, on="UserId", how="left")
            df = df.merge(self.item_profiles[["AttractionId", "Attraction_Avg_Rating", "Attraction_Total_Reviews"]], on="AttractionId", how="left")

        categorical_cols = ["ContinentId", "RegionId", "CountryId", "UserCityId", "AttractionTypeId"]
        num_cols = ["VisitYear", "VisitMonth", "User_Avg_Rating", "User_Total_Visits", "Attraction_Avg_Rating", "Attraction_Total_Reviews"]

        for col in categorical_cols:
            if col in df.columns:
                if is_train or f"cls_{col}" not in self.label_encoders:
                    le = LabelEncoder()
                    df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[f"cls_{col}"] = le
                else:
                    le = self.label_encoders[f"cls_{col}"]
                    df[f"{col}_encoded"] = df[col].astype(str).map(
                        lambda s: le.transform([s])[0] if s in le.classes_ else -1
                    )

        feature_cols = [f"{col}_encoded" for col in categorical_cols if f"{col}_encoded" in df.columns] + num_cols
        X = df[feature_cols].copy()
        y = df["VisitModeId"].copy()

        if is_train:
            scaler = StandardScaler()
            X[num_cols] = scaler.fit_transform(X[num_cols].fillna(0))
            self.scalers["classification"] = scaler
        else:
            scaler = self.scalers.get("classification")
            if scaler:
                X[num_cols] = scaler.transform(X[num_cols].fillna(0))

        return X, y

    def build_user_item_matrix(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs the User-Item interaction rating pivot matrix for Collaborative Filtering.
        """
        pivot_df = master_df.pivot_table(
            index="UserId",
            columns="AttractionId",
            values="Rating",
            aggfunc="mean"
        ).fillna(0.0)

        return pivot_df


def build_all_features(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to execute complete feature engineering pipeline.

    Returns:
        Dict[str, Any]: Container of features, matrices, and metadata.
    """
    master_df, users_df, items_df = build_consolidated_dataset(data_dir=data_dir)

    pipeline = FeaturePipeline()
    user_stats = pipeline.compute_user_aggregates(master_df)
    item_stats = pipeline.compute_item_aggregates(master_df, items_df)

    tfidf_matrix, tfidf_vec = pipeline.build_attraction_tfidf_matrix(items_df)

    X_reg, y_reg = pipeline.prepare_regression_features(master_df, is_train=True)
    X_cls, y_cls = pipeline.prepare_classification_features(master_df, is_train=True)

    user_item_matrix = pipeline.build_user_item_matrix(master_df)

    return {
        "pipeline": pipeline,
        "master_df": master_df,
        "users_df": users_df,
        "items_df": items_df,
        "user_stats": user_stats,
        "item_stats": item_stats,
        "tfidf_matrix": tfidf_matrix,
        "X_reg": X_reg,
        "y_reg": y_reg,
        "X_cls": X_cls,
        "y_cls": y_cls,
        "user_item_matrix": user_item_matrix
    }


if __name__ == "__main__":
    print("Executing Feature Engineering Pipeline...")
    features = build_all_features()
    print(f"[OK] Regression feature matrix X shape: {features['X_reg'].shape}")
    print(f"[OK] Classification feature matrix X shape: {features['X_cls'].shape}")
    print(f"[OK] TF-IDF Attraction vector matrix shape: {features['tfidf_matrix'].shape}")
    print(f"[OK] User-Item Pivot matrix shape: {features['user_item_matrix'].shape}")
