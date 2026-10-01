"""
Regression Training Module for Tourism Experience Analytics.
Predicts attraction Rating (1.0 to 5.0) using Linear Regression, Random Forest, and XGBoost Regressors.
Includes strict leakage-free feature engineering and evaluation.
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.data.preprocessor import build_consolidated_dataset


class LeakageFreeRegressionPreprocessor:
    """
    Preprocessor for regression target rating prediction.
    Calculates user and item rating statistics strictly on training data using leave-one-out
    aggregations for training rows and training-set lookups for validation/test/inference rows.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.global_train_mean = 3.5
        self.user_stats = {}
        self.item_stats = {}
        self.cat_cols = [
            "ContinentId", "RegionId", "CountryId",
            "UserCityId", "AttractionTypeId", "VisitModeId"
        ]
        self.num_cols = [
            "VisitYear", "VisitMonth",
            "User_Avg_Rating", "User_Total_Visits",
            "Attraction_Avg_Rating", "Attraction_Total_Reviews"
        ]
        self.feature_names = []

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        df = train_df.copy()

        # Compute global mean rating on training data
        self.global_train_mean = float(df["Rating"].mean())

        # Compute user-level aggregates on training data
        user_grp = df.groupby("UserId")["Rating"]
        user_sums = user_grp.sum().to_dict()
        user_counts = user_grp.count().to_dict()
        for u in user_counts:
            self.user_stats[u] = {
                "sum": user_sums[u],
                "count": user_counts[u],
                "mean": user_sums[u] / user_counts[u]
            }

        # Compute item-level aggregates on training data
        item_grp = df.groupby("AttractionId")["Rating"]
        item_sums = item_grp.sum().to_dict()
        item_counts = item_grp.count().to_dict()
        for i in item_counts:
            self.item_stats[i] = {
                "sum": item_sums[i],
                "count": item_counts[i],
                "mean": item_sums[i] / item_counts[i]
            }

        # Leave-one-out calculation for train set rows
        user_avg = []
        user_visits = []
        for u, r in zip(df["UserId"], df["Rating"]):
            st = self.user_stats.get(u)
            if st and st["count"] > 1:
                user_avg.append((st["sum"] - r) / (st["count"] - 1))
                user_visits.append(st["count"] - 1)
            else:
                user_avg.append(self.global_train_mean)
                user_visits.append(0)

        item_avg = []
        item_reviews = []
        for i, r in zip(df["AttractionId"], df["Rating"]):
            st = self.item_stats.get(i)
            if st and st["count"] > 1:
                item_avg.append((st["sum"] - r) / (st["count"] - 1))
                item_reviews.append(st["count"] - 1)
            else:
                item_avg.append(self.global_train_mean)
                item_reviews.append(0)

        df["User_Avg_Rating"] = user_avg
        df["User_Total_Visits"] = user_visits
        df["Attraction_Avg_Rating"] = item_avg
        df["Attraction_Total_Reviews"] = item_reviews

        # Encode categorical columns
        encoded_cat_cols = []
        for col in self.cat_cols:
            enc_col = f"{col}_encoded"
            encoded_cat_cols.append(enc_col)
            le = LabelEncoder()
            # Handle potential nulls or string types
            vals = df[col].astype(str)
            df[enc_col] = le.fit_transform(vals)
            self.label_encoders[col] = le

        self.feature_names = encoded_cat_cols + self.num_cols

        # Scale numerical features
        X_num = self.scaler.fit_transform(df[self.num_cols].fillna(0))
        for idx, col in enumerate(self.num_cols):
            df[col] = X_num[:, idx]

        return df[self.feature_names]

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()

        # Map user stats from training lookup
        user_avg = []
        user_visits = []
        for u in df["UserId"]:
            st = self.user_stats.get(u)
            if st:
                user_avg.append(st["mean"])
                user_visits.append(st["count"])
            else:
                user_avg.append(self.global_train_mean)
                user_visits.append(0)

        # Map item stats from training lookup
        item_avg = []
        item_reviews = []
        for i in df["AttractionId"]:
            st = self.item_stats.get(i)
            if st:
                item_avg.append(st["mean"])
                item_reviews.append(st["count"])
            else:
                item_avg.append(self.global_train_mean)
                item_reviews.append(0)

        df["User_Avg_Rating"] = user_avg
        df["User_Total_Visits"] = user_visits
        df["Attraction_Avg_Rating"] = item_avg
        df["Attraction_Total_Reviews"] = item_reviews

        # Encode categorical columns
        encoded_cat_cols = []
        for col in self.cat_cols:
            enc_col = f"{col}_encoded"
            encoded_cat_cols.append(enc_col)
            le = self.label_encoders[col]
            vals = df[col].astype(str)
            df[enc_col] = vals.map(
                lambda s: le.transform([s])[0] if s in le.classes_ else -1
            )

        # Scale numerical features
        X_num = self.scaler.transform(df[self.num_cols].fillna(0))
        for idx, col in enumerate(self.num_cols):
            df[col] = X_num[:, idx]

        return df[self.feature_names]


def train_and_evaluate_regression(artifacts_dir: Path = None):
    """
    Main training workflow for regression models.
    """
    if artifacts_dir is None:
        artifacts_dir = Path(__file__).resolve().parent.parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading consolidated dataset...")
    master_df, _, _ = build_consolidated_dataset()

    # Split into Train (70%), Validation (15%), Test (15%) with reproducible seed
    train_df, temp_df = train_test_split(master_df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    print(f"Dataset splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Leakage-free preprocessor
    preprocessor = LeakageFreeRegressionPreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    y_train = train_df["Rating"].values

    X_val = preprocessor.transform(val_df)
    y_val = val_df["Rating"].values

    X_test = preprocessor.transform(test_df)
    y_test = test_df["Rating"].values

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Evaluate on Train, Val, Test
        preds_train = np.clip(model.predict(X_train), 1.0, 5.0)
        preds_val = np.clip(model.predict(X_val), 1.0, 5.0)
        preds_test = np.clip(model.predict(X_test), 1.0, 5.0)

        train_mse = mean_squared_error(y_train, preds_train)
        train_rmse = float(np.sqrt(train_mse))
        train_r2 = float(r2_score(y_train, preds_train))

        val_mse = mean_squared_error(y_val, preds_val)
        val_rmse = float(np.sqrt(val_mse))
        val_r2 = float(r2_score(y_val, preds_val))

        test_mse = mean_squared_error(y_test, preds_test)
        test_rmse = float(np.sqrt(test_mse))
        test_r2 = float(r2_score(y_test, preds_test))

        results.append({
            "Model": name,
            "Train_MSE": train_mse,
            "Train_RMSE": train_rmse,
            "Train_R2": train_r2,
            "Val_MSE": val_mse,
            "Val_RMSE": val_rmse,
            "Val_R2": val_r2,
            "Test_MSE": test_mse,
            "Test_RMSE": test_rmse,
            "Test_R2": test_r2,
        })

    results_df = pd.DataFrame(results)
    results_path = artifacts_dir / "regression_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved regression results to {results_path}")
    print(results_df[["Model", "Val_RMSE", "Val_R2", "Val_MSE"]])

    # Select best model based on validation RMSE
    best_row = results_df.loc[results_df["Val_RMSE"].idxmin()]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]

    print(f"\nBest Regression Model: {best_name} (Val RMSE: {best_row['Val_RMSE']:.4f})")

    # Save best model bundle artifact
    artifact_bundle = {
        "model_name": best_name,
        "model": best_model,
        "preprocessor": preprocessor,
        "feature_names": preprocessor.feature_names,
        "val_rmse": best_row["Val_RMSE"],
        "val_r2": best_row["Val_R2"],
        "val_mse": best_row["Val_MSE"]
    }
    model_path = artifacts_dir / "regression_best_model.pkl"
    joblib.dump(artifact_bundle, model_path)
    print(f"Saved best regression model artifact to {model_path}")

    return artifact_bundle, results_df


if __name__ == "__main__":
    train_and_evaluate_regression()
