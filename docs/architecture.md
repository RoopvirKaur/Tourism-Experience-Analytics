# System Architecture: Tourism Experience Analytics

## 1. High-Level System Architecture

The Tourism Experience Analytics system is designed as an end-to-end Machine Learning and Analytics platform. It ingests multi-relational tourism data, performs automated data cleaning and feature engineering, trains specialized machine learning models (Regression, Classification, Recommendation), and exposes interactive predictions and insights through a Streamlit web application.

```mermaid
graph TD
    subgraph Data Layer ["Data Layer"]
        RAW["Raw Excel Datasets (dataset/)"]
        VAL["Data Validation & Ingestion"]
    end

    subgraph Pipeline ["Processing & Feature Pipeline"]
        CLEAN["Data Cleaning & Discrepancy Resolution"]
        MERGE["Relational Merge & Preprocessing"]
        FEAT["Feature Engineering & Vectorization"]
    end

    subgraph ML ["Machine Learning Engine"]
        REG["Regression Model (Rating Predictor)"]
        CLASS["Classification Model (Visit Mode Predictor)"]
        REC["Recommendation Engine (Collaborative + Content-Based)"]
    end

    subgraph Artifacts ["Artifact Storage"]
        MOD_STORE["Model & Scaler Artifacts (.joblib / .pkl)"]
        MAT_STORE["User-Item Interaction Matrix & Vectors"]
    end

    subgraph Presentation ["Presentation Layer"]
        ST["Streamlit Multi-Page Web App"]
        DASH["Analytics & EDA Dashboards"]
        INF["Interactive Inference Interface"]
    end

    RAW --> VAL
    VAL --> CLEAN
    CLEAN --> MERGE
    MERGE --> FEAT
    FEAT --> REG
    FEAT --> CLASS
    FEAT --> REC

    REG --> MOD_STORE
    CLASS --> MOD_STORE
    REC --> MAT_STORE

    MOD_STORE --> INF
    MAT_STORE --> INF
    FEAT --> DASH

    INF --> ST
    DASH --> ST
```

---

## 2. Data Architecture & Relational Entity Model

The domain architecture relies on 9 interconnected relational tables. The Entity-Relationship Diagram below illustrates the relational mapping used during data consolidation:

```mermaid
erDiagram
    USER {
        int UserId PK
        int ContinentId FK
        int RegionId FK
        int CountryId FK
        int CityId FK
    }

    TRANSACTION {
        int TransactionId PK
        int UserId FK
        int AttractionId FK
        int VisitYear
        int VisitMonth
        string VisitMode
        float Rating
    }

    ITEM {
        int AttractionId PK
        int AttractionCityId FK
        int AttractionTypeId FK
        string Attraction
        string AttractionAddress
    }

    CITY {
        int CityId PK
        string CityName
        int CountryId FK
    }

    COUNTRY {
        int CountryId PK
        string Country
        int RegionId FK
    }

    REGION {
        int RegionId PK
        string Region
        int ContinentId FK
    }

    CONTINENT {
        int ContinentId PK
        string Continent
    }

    TYPE {
        int AttractionTypeId PK
        string AttractionType
    }

    MODE {
        int VisitModeId PK
        string VisitMode
    }

    USER ||--o{ TRANSACTION : "makes"
    ITEM ||--o{ TRANSACTION : "rated in"
    CITY ||--o{ USER : "resides in"
    CITY ||--o{ ITEM : "located in"
    COUNTRY ||--|{ CITY : "contains"
    REGION ||--|{ COUNTRY : "contains"
    CONTINENT ||--|{ REGION : "contains"
    TYPE ||--o{ ITEM : "categorizes"
    MODE ||--o{ TRANSACTION : "defines"
```

---

## 3. Data Processing & Pipeline Architecture

```mermaid
flowchart LR
    A["Raw Input Workbooks"] --> B["Ingestion Engine"]
    B --> C["Type Standardization & Null Treatment"]
    C --> D["Table Joiner (Consolidated DataFrame)"]
    D --> E["Encoding & Feature Scaling"]
    E --> F["Train / Test Splits"]
    F --> G["Model Trainers"]
```

### Data Pipeline Components

1. **Ingestion & Data Validation Layer**:
   - Reads `.xlsx` files from `dataset/`.
   - Validates schema compliance, data types, and primary key integrity.

2. **Cleaning & Transformation Module**:
   - **Missing Value Handling**: Imputes missing values based on demographic aggregations or category modes.
   - **Category Normalization**: Resolves duplicate or inconsistent strings in `VisitMode`, `AttractionType`, and `CityName`.
   - **Date Normalization**: Standardizes `VisitYear` and `VisitMonth` fields.
   - **Outlier Filtering**: Validates `Rating` range (1.0 to 5.0).

3. **Feature Engineering Store**:
   - **Demographic Features**: Encodes `ContinentId`, `CountryId`, `RegionId`, `CityId` using One-Hot / Label Encoding.
   - **Item Profile Aggregates**: Computes average ratings, total review counts, and visitor demographic distributions per attraction.
   - **User Behavior Vectors**: Derives user average rating per `VisitMode` and frequency of visits.

---

## 4. Machine Learning Subsystems

### Subsystem 1: Regression Architecture (Attraction Rating Predictor)
- **Target**: Continuous rating value $\in [1.0, 5.0]$.
- **Features**: User demographic encodings + Attraction type & location features + Temporal attributes (`VisitYear`, `VisitMonth`).
- **Algorithms**: Linear Regression, Random Forest Regressor, XGBoost Regressor, LightGBM Regressor.
- **Evaluation**: $R^2$, Mean Squared Error (MSE), Root Mean Squared Error (RMSE).

### Subsystem 2: Classification Architecture (Visit Mode Classifier)
- **Target**: Multi-class Visit Mode (`Business`, `Family`, `Couples`, `Friends`, `Solo`).
- **Features**: User origin demographics + Attraction features + Historical travel patterns.
- **Algorithms**: Random Forest Classifier, LightGBM Classifier, XGBoost Classifier.
- **Evaluation**: Accuracy, Precision, Recall, Macro/Weighted F1-Score.

### Subsystem 3: Hybrid Recommendation Engine

```mermaid
graph TD
    subgraph Input ["User Request"]
        UID["User ID / Demographic Input"]
    end

    subgraph Engine1 ["Collaborative Filtering Branch"]
        UIM["User-Item Interaction Matrix"]
        COS_U["Cosine / Pearson User Similarity"]
        CF_REC["Collaborative Top-K Candidates"]
    end

    subgraph Engine2 ["Content-Based Filtering Branch"]
        ITEM_VEC["Attraction Feature Vectors (Type, Location, Address)"]
        COS_I["Cosine Feature Similarity"]
        CB_REC["Content-Based Top-K Candidates"]
    end

    subgraph Hybrid ["Ensemble Merger"]
        WEIGHT["Weighted Score Combination"]
        RANK["Ranker & Filter"]
    end

    subgraph Output ["Recommendations"]
        FINAL["Ranked Attraction List"]
    end

    UID --> UIM
    UID --> ITEM_VEC

    UIM --> COS_U --> CF_REC
    ITEM_VEC --> COS_I --> CB_REC

    CF_REC --> WEIGHT
    CB_REC --> WEIGHT

    WEIGHT --> RANK --> FINAL
```

- **Collaborative Filtering**: Computes user-user or item-item similarity matrices on the sparse rating interaction matrix.
- **Content-Based Filtering**: Constructs TF-IDF / One-Hot feature vectors of attractions using `AttractionType`, `CityName`, and geographic parameters.
- **Hybrid Blending**: Combines normalized collaborative scores ($S_{CF}$) and content scores ($S_{CB}$) via weighted ensemble:
  $$S_{Hybrid} = \alpha \cdot S_{CF} + (1 - \alpha) \cdot S_{CB}$$

---

## 5. Streamlit Application Architecture

The user-facing presentation layer is structured as an interactive multi-page Streamlit app:

```mermaid
graph TD
    subgraph WebApp ["Streamlit Web Application (src/app/main.py)"]
        NAV["Sidebar Navigation"]

        subgraph Pages ["Application Pages"]
            P1["Page 1: Exploratory Data Analytics"]
            P2["Page 2: Visit Mode Predictor"]
            P3["Page 3: Rating Predictor"]
            P4["Page 4: Personalized Recommender"]
        end
    end

    subgraph AppBackend ["App Service Backend"]
        CACHE["Streamlit Caching Layer (@st.cache_data / @st.cache_resource)"]
        LOADER["Model & Artifact Loader"]
        INFER_ENG["Inference Engine"]
    end

    NAV --> P1
    NAV --> P2
    NAV --> P3
    NAV --> P4

    P1 --> CACHE
    P2 --> INFER_ENG
    P3 --> INFER_ENG
    P4 --> INFER_ENG

    LOADER --> CACHE
    CACHE --> INFER_ENG
```

### Key Application Components
- **State Management**: Uses `st.session_state` to store user inputs, selected filters, and active recommendations.
- **Caching Layer**: Uses `@st.cache_data` for heavy dataframes and `@st.cache_resource` for loading machine learning models to ensure low latency.
- **Interactive Controls**: Inputs for User demographic parameters, target visit modes, and desired attraction categories.

---

## 6. Recommended Repository Structure

```text
Tourism Experience/
├── dataset/                                 # Raw input data files (.xlsx)
│   ├── Transaction.xlsx
│   ├── User.xlsx
│   ├── City.xlsx
│   ├── Type.xlsx
│   ├── Mode.xlsx
│   ├── Continent.xlsx
│   ├── Country.xlsx
│   ├── Region.xlsx
│   ├── Item.xlsx
│   └── Additional_Data_for_Attraction_Sites/
│       └── Updated_Item.xlsx
├── docs/                                    # Project documentation
│   ├── problemstatement.txt
│   ├── context.md
│   └── architecture.md
├── src/                                     # Core application source code
│   ├── data/                                # Data ingestion and preprocessing scripts
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── features/                            # Feature engineering and vectorization
│   │   └── build_features.py
│   ├── models/                              # ML model definitions and training logic
│   │   ├── train_regression.py
│   │   ├── train_classification.py
│   │   └── recommendation.py
│   └── app/                                 # Streamlit UI application
│       ├── main.py                          # App entry point
│       └── pages/                           # Sub-pages (EDA, Models, Recommendations)
├── artifacts/                               # Serialized models and feature matrices (.joblib)
│   ├── regression_model.joblib
│   ├── classification_model.joblib
│   └── similarity_matrices.joblib
├── requirements.txt                         # Dependencies
└── README.md                                # Project introduction & setup guide
```
