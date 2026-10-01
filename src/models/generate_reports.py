"""
Master script to execute Phase 4 ML workflows, generate model comparison table,
and create the comprehensive phase4_model_summary.txt report.
"""

import sys
from pathlib import Path
import pandas as pd
import joblib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_regression import train_and_evaluate_regression
from src.models.train_classification import train_and_evaluate_classification
from src.models.recommender import train_and_evaluate_recommender


def generate_all_reports(artifacts_dir: Path = None):
    if artifacts_dir is None:
        artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("      RUNNING PHASE 4 MACHINE LEARNING PIPELINE   ")
    print("==================================================")

    # 1. Regression
    print("\n--- 1. REGRESSION TASK ---")
    reg_bundle, reg_df = train_and_evaluate_regression(artifacts_dir)

    # 2. Classification
    print("\n--- 2. CLASSIFICATION TASK ---")
    cls_bundle, cls_df = train_and_evaluate_classification(artifacts_dir)

    # 3. Recommendation System
    print("\n--- 3. RECOMMENDATION SYSTEM ---")
    rec_bundle, rec_metrics = train_and_evaluate_recommender(artifacts_dir)

    # 4. Generate model_comparison.csv
    print("\n--- GENERATING MODEL COMPARISON TABLE ---")
    comparison_rows = []

    # Add regression models
    for _, row in reg_df.iterrows():
        comparison_rows.append({
            "Model": row["Model"],
            "Task": "Regression",
            "Primary_Metric_Name": "Val_RMSE",
            "Primary_Metric_Value": f"{row['Val_RMSE']:.4f}",
            "Secondary_Metric_Name": "Val_R2",
            "Secondary_Metric_Value": f"{row['Val_R2']:.4f}",
            "Tertiary_Metric_Name": "Val_MSE",
            "Tertiary_Metric_Value": f"{row['Val_MSE']:.4f}",
            "Is_Best": "Yes" if row["Model"] == reg_bundle["model_name"] else "No"
        })

    # Add classification models
    for _, row in cls_df.iterrows():
        comparison_rows.append({
            "Model": row["Model"],
            "Task": "Classification",
            "Primary_Metric_Name": "Val_Macro_F1",
            "Primary_Metric_Value": f"{row['Val_F1_Macro']:.4f}",
            "Secondary_Metric_Name": "Val_Accuracy",
            "Secondary_Metric_Value": f"{row['Val_Accuracy']:.4f}",
            "Tertiary_Metric_Name": "Val_Weighted_F1",
            "Tertiary_Metric_Value": f"{row['Val_F1_Weighted']:.4f}",
            "Is_Best": "Yes" if row["Model"] == cls_bundle["model_name"] else "No"
        })

    # Add recommender
    comparison_rows.append({
        "Model": "Hybrid Recommender (CF + Content)",
        "Task": "Recommendation",
        "Primary_Metric_Name": "MAP@5",
        "Primary_Metric_Value": f"{rec_metrics['MAP@K']:.4f}",
        "Secondary_Metric_Name": "Precision@5",
        "Secondary_Metric_Value": f"{rec_metrics['Precision@K']:.4f}",
        "Tertiary_Metric_Name": "Recall@5",
        "Tertiary_Metric_Value": f"{rec_metrics['Recall@K']:.4f}",
        "Is_Best": "Yes"
    })

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = artifacts_dir / "model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"Saved model comparison table to {comp_path}")

    # 5. Generate phase4_model_summary.txt
    print("\n--- GENERATING PHASE 4 SUMMARY REPORT ---")
    best_reg_name = reg_bundle["model_name"]
    best_cls_name = cls_bundle["model_name"]

    summary_text = f"""================================================================================
PHASE 4: MACHINE LEARNING MODEL DEVELOPMENT SUMMARY REPORT
Tourism Experience Analytics Project
================================================================================

1. OBJECTIVE & OVERVIEW
   This phase implemented three core Machine Learning subsystems:
   - Regression: Predict attraction rating (1.0 - 5.0)
   - Classification: Predict user VisitMode (Business, Family, Couples, Friends, Solo)
   - Recommendation System: Hybrid engine (Content-Based + Collaborative Filtering)

--------------------------------------------------------------------------------
2. REGRESSION TASK SUMMARY (Target: Rating)
--------------------------------------------------------------------------------
Models Tested:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Model Performance Comparison:
{reg_df[['Model', 'Val_RMSE', 'Val_R2', 'Val_MSE', 'Test_RMSE', 'Test_R2']].to_string(index=False)}

Best Regression Model: {best_reg_name}
- Validation RMSE : {reg_bundle['val_rmse']:.4f}
- Validation R^2  : {reg_bundle['val_r2']:.4f}
- Validation MSE  : {reg_bundle['val_mse']:.4f}

Target Leakage Prevention Mechanism (Regression):
- Rating target is strictly excluded from predictor features.
- User_Avg_Rating and Attraction_Avg_Rating are computed strictly on the training set using leave-one-out calculation for training rows.
- Validation and Test sets use lookup statistics calculated strictly from training split, preventing target leakage across data splits.

Final Regression Features Used ({len(reg_bundle['feature_names'])} features):
- Categorical: ContinentId_encoded, RegionId_encoded, CountryId_encoded, UserCityId_encoded, AttractionTypeId_encoded, VisitModeId_encoded
- Numerical: VisitYear, VisitMonth, User_Avg_Rating, User_Total_Visits, Attraction_Avg_Rating, Attraction_Total_Reviews

--------------------------------------------------------------------------------
3. CLASSIFICATION TASK SUMMARY (Target: VisitMode)
--------------------------------------------------------------------------------
Models Tested:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Model Performance Comparison:
{cls_df[['Model', 'Val_Accuracy', 'Val_F1_Macro', 'Val_F1_Weighted', 'Test_Accuracy', 'Test_F1_Macro']].to_string(index=False)}

Best Classification Model: {best_cls_name}
- Validation Macro F1    : {cls_bundle['val_f1_macro']:.4f}
- Validation Accuracy    : {cls_bundle['val_accuracy']:.4f}
- Validation Weighted F1: {cls_bundle['val_f1_weighted']:.4f}

Target Exclusion & Class Balancing (Classification):
- VisitMode, VisitModeName, VisitModeId, and User_Favorite_VisitMode are STRICTLY EXCLUDED from predictor features.
- Stratified train/val/test splitting ensures identical class proportions across splits.
- Class balancing applied using 'balanced' class weights and sample weighting.

Final Classification Features Used ({len(cls_bundle['feature_names'])} features):
- Categorical: ContinentId_encoded, RegionId_encoded, CountryId_encoded, UserCityId_encoded, AttractionTypeId_encoded
- Numerical: VisitYear, VisitMonth, User_Avg_Rating, User_Total_Visits, Attraction_Avg_Rating, Attraction_Total_Reviews

--------------------------------------------------------------------------------
4. RECOMMENDATION SYSTEM SUMMARY
--------------------------------------------------------------------------------
Engine Architecture:
1. Content-Based Filtering: TF-IDF feature matrix of attraction metadata + Cosine Similarity.
2. Collaborative Filtering: Sparse User-Item interaction rating pivot matrix + User Cosine Similarity.
3. Hybrid Blending: Min-Max normalized weighted score combination (alpha * CF + (1 - alpha) * Content).

Cold-Start & Interaction Handling:
- Automatically filters out attractions the user has already visited/rated.
- Unknown / New Users (Cold-Start): Fallback to top-rated popular attractions ranked by composite popularity score.

Offline Evaluation Results (K=5):
- MAP@5        : {rec_metrics['MAP@K']:.4f}
- Precision@5  : {rec_metrics['Precision@K']:.4f}
- Recall@5     : {rec_metrics['Recall@K']:.4f}
- Test Users   : {rec_metrics['Evaluated_Users']}

--------------------------------------------------------------------------------
5. SAVED ARTIFACTS
--------------------------------------------------------------------------------
- artifacts/regression_best_model.pkl
- artifacts/regression_results.csv
- artifacts/classification_best_model.pkl
- artifacts/classification_results.csv
- artifacts/classification_label_encoder.pkl
- artifacts/recommender.pkl
- artifacts/model_comparison.csv
- artifacts/phase4_model_summary.txt

--------------------------------------------------------------------------------
6. LIMITATIONS & FUTURE ENHANCEMENTS
--------------------------------------------------------------------------------
- User-Item matrix sparsity can be mitigated in future iterations with Deep Learning matrix factorization (e.g. NCF / Autoencoders).
- Hyperparameters tuned with standard grids; further Bayesian Optimization (Optuna) can be performed.
- Model latency is low and suitable for interactive real-time Streamlit deployment in Phase 5.
================================================================================
"""
    summary_path = artifacts_dir / "phase4_model_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"Saved phase 4 summary text report to {summary_path}")
    print("\nPhase 4 ML pipeline execution completed successfully!")


if __name__ == "__main__":
    generate_all_reports()
