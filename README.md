# Tourism Experience Analytics: Classification, Prediction, and Recommendation System

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification%20%7C%20Regression%20%7C%20Recommender-green.svg)](https://scikit-learn.org/)

## 📌 Project Overview
**Tourism Experience Analytics** is an end-to-end Machine Learning and Interactive Web Platform designed to enhance travel experiences, optimize visitor engagement, and provide actionable analytics for tourism agencies, destination management organizations, and travel platforms.

The platform processes multi-relational visitor demographic data, transaction logs, attraction features, and travel modes across 9 relational datasets to deliver:
1. **Exploratory Data Analytics (EDA) Dashboard**: Visual insights into spatial visitor demographics, attraction popularity, temporal trends, and group travel dynamics.
2. **Visit Mode Classifier**: Predicts traveler group dynamics (`Business`, `Couples`, `Family`, `Friends`, `Solo`) using Random Forest classification.
3. **Attraction Rating Predictor**: Estimates expected user rating for specific tourist attractions ($1.0 - 5.0$) using regression models.
4. **Hybrid Recommendation Engine**: Generates personalized attraction recommendations by blending Content-Based vector similarity and Collaborative Filtering matrix factorization.

---

## 🏗️ Project Architecture & Pipeline

```
Tourism Experience/
├── dataset/                                 # Raw relational input Excel datasets (.xlsx)
├── docs/                                    # Project architecture, problem statement, and context
│   ├── problemstatement.txt
│   ├── context.md
│   ├── architecture.md
│   └── implementation_plan.md
├── src/                                     # Source Code
│   ├── data/                                # Data Ingestion & Preprocessing Pipeline
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── features/                            # Feature Engineering & Vectorization
│   │   └── build_features.py
│   ├── models/                              # ML Subsystems (Regression, Classification, Recommender)
│   │   ├── train_regression.py
│   │   ├── train_classification.py
│   │   ├── recommender.py
│   │   └── generate_reports.py
│   └── app/                                 # Streamlit Web Application
│       ├── main.py                          # App Entry Point & Sidebar Navigation
│       └── pages/                           # Interactive Multi-Page Dashboards & Interfaces
│           ├── 01_EDA_Dashboard.py
│           ├── 02_Visit_Mode_Predictor.py
│           ├── 03_Rating_Predictor.py
│           └── 04_Attraction_Recommender.py
├── artifacts/                               # Serialized Models & Evaluation Artifacts
│   ├── regression_best_model.pkl
│   ├── classification_best_model.pkl
│   ├── recommender.pkl
│   ├── classification_results.csv
│   ├── regression_results.csv
│   └── phase4_model_summary.txt
├── notebooks/                               # Jupyter Analytics Notebooks
│   └── 01_eda_and_insights.ipynb
├── tests/                                   # Unit & Integration Tests (15 test suites)
│   ├── test_data_pipeline.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_app.py
├── requirements.txt                         # Project Dependencies
└── README.md                                # Project Documentation
```

---

## 🚀 Status of Implementation Phases

| Phase | Description | Status | Verification & Deliverables |
| :--- | :--- | :---: | :--- |
| **Phase 1** | **Environment & Scaffolding** | ✅ **Completed** | Dependencies configured (`requirements.txt`), modular package setup (`src/`). |
| **Phase 2** | **Data Engineering & Preprocessing** | ✅ **Completed** | Modular loader (`loader.py`), null treatment, text normalization, ratings clamping $[1, 5]$, relational table merging (`preprocessor.py`). |
| **Phase 3** | **Feature Engineering & EDA** | ✅ **Completed** | Demographics encoding, user aggregations, TF-IDF attraction vectorization (`build_features.py`), EDA notebook (`01_eda_and_insights.ipynb`). |
| **Phase 4** | **Model Building & Serialization** | ✅ **Completed** | Rating Regression (`train_regression.py`), Visit Mode Classification (`train_classification.py`), Hybrid Recommender (`recommender.py`). Saved in `artifacts/`. |
| **Phase 5** | **Streamlit Web Application** | ✅ **Completed** | Interactive 4-page Streamlit web app (`main.py`, `01_EDA_Dashboard.py`, `02_Visit_Mode_Predictor.py`, `03_Rating_Predictor.py`, `04_Attraction_Recommender.py`). |
| **Phase 6** | **System Verification & Documentation** | ✅ **Completed** | Unit & Integration Test Suite (`pytest` - 15/15 passing), model summary reports (`artifacts/phase4_model_summary.txt`), architecture & user documentation. |

---

## 📊 Model Performance Highlights

### 1. Regression Subsystem (Target: Rating $1.0 - 5.0$)
- **Best Model**: Linear Regression
- **Validation RMSE**: `0.9181` | **Validation $R^2$**: `0.1261` | **Validation MSE**: `0.8429`
- *Target Leakage Prevention*: Computed out-of-fold leave-one-out statistics for user & attraction ratings.

### 2. Classification Subsystem (Target: Visit Mode)
- **Best Model**: Random Forest Classifier
- **Validation Accuracy**: `50.30%` | **Macro F1**: `0.3883` | **Weighted F1**: `0.4878`
- *Handling Imbalance*: Stratified splits and balanced sample weights.

### 3. Recommendation System (Hybrid Collaborative + Content-Based)
- **Offline Recommendation Evaluation (K=5)**:
  - **MAP@5**: `0.5201`
  - **Recall@5**: `0.5460`
  - **Precision@5**: `0.1092`
- *Cold Start Handling*: Fallback to composite attraction popularity ranking for new users.

---

## 💻 Installation & Usage Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Automated Tests
Execute the comprehensive test suite:
```bash
pytest
```

### 3. Launch the Streamlit Web Application
Start the application server:
```bash
streamlit run src/app/main.py
```
The app will open automatically at `http://localhost:8501`.

---

## 📬 Contact & License
Developed as part of the Tourism Experience Analytics Solution.
