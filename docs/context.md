# Project Context: Tourism Experience Analytics

## 1. Project Overview & Business Goal

- **Project Title**: Tourism Experience Analytics: Classification, Prediction, and Recommendation System
- **Domain**: Tourism & Travel Tech
- **Core Goal**: Leverage user travel data, attraction features, and transaction logs to build predictive models and personalized recommendation engines that enhance user satisfaction, guide targeted marketing, and enable data-driven tourism analytics.

### Business Use Cases
1. **Personalized Recommendations**: Suggest attractions tailored to individual user profiles and past behaviors, improving user experience and engagement.
2. **Tourism Analytics**: Deliver actionable insights regarding popular attractions, regional trends, and visitor demographics for tourism stakeholders.
3. **Customer Segmentation**: Classify users into distinct travel segments (e.g., family, solo, business) for targeted marketing campaigns and personalized promotional packages.
4. **Retention & Loyalty**: Increase user retention on travel platforms by continuously providing relevant and accurate attraction suggestions.

---

## 2. Core Machine Learning Tasks & Objectives

### 1. Regression Task: Predicting Attraction Ratings
- **Objective**: Develop a regression model to estimate the rating (1 to 5 scale) a user is likely to give to a specific tourist attraction.
- **Use Case**: Enables travel platforms to estimate user satisfaction, identify underperforming attractions, and set accurate user expectations.
- **Features**: User demographics (Continent, Region, Country, City), Visit details (Year, Month, Visit Mode), Attraction features (Type, Location, Previous Average Ratings).
- **Target**: Predicted Rating (`Rating`).

### 2. Classification Task: User Visit Mode Prediction
- **Objective**: Build a classification model to predict the visitor's mode of travel (e.g., Business, Family, Couples, Friends).
- **Use Case**: Allows travel agencies, hotels, and attraction organizers to tailor marketing strategies and resource allocations based on predicted visitor group dynamics.
- **Features**: User demographics, Attraction characteristics, Historical visit patterns (Month, Year, previous visit modes).
- **Target**: Visit Mode (`VisitMode`).

### 3. Recommendation System: Personalized Attraction Suggestions
- **Objective**: Create a recommendation engine to generate a ranked list of suggested attractions for users.
- **Approaches**:
  - **Collaborative Filtering**: Recommends items based on preference matrix and rating patterns of similar users.
  - **Content-Based Filtering**: Suggests attractions similar to a user's past visits based on attraction type, location, and features.
  - **Hybrid Recommendation System**: Combines collaborative and content-based approaches for optimal recommendation accuracy.
- **Output**: Ranked list of recommended tourist attractions.

---

## 3. Dataset Architecture & Schema

The dataset resides in the `dataset/` directory and consists of relational Excel workbooks:

| Dataset File | Primary Purpose / Contents | Key Columns |
| :--- | :--- | :--- |
| **`Transaction.xlsx`** | Transactional history of user visits and ratings | `TransactionId`, `UserId`, `VisitYear`, `VisitMonth`, `VisitMode`, `AttractionId`, `Rating` |
| **`User.xlsx`** | Geographical and demographic information of users | `UserId`, `ContinentId`, `RegionId`, `CountryId`, `CityId` |
| **`City.xlsx`** | City lookup table linking to countries | `CityId`, `CityName`, `CountryId` |
| **`Type.xlsx`** | Attraction type classifications | `AttractionTypeId`, `AttractionType` |
| **`Mode.xlsx`** | Visit mode classifications | `VisitModeId`, `VisitMode` |
| **`Continent.xlsx`** | Continent lookup mapping | `ContinentId`, `Continent` |
| **`Country.xlsx`** | Country lookup mapping | `CountryId`, `Country`, `RegionId` |
| **`Region.xlsx`** | Region lookup mapping | `RegionId`, `Region`, `ContinentId` |
| **`Item.xlsx`** | Master details of tourist attractions | `AttractionId`, `AttractionCityId`, `AttractionTypeId`, `Attraction`, `AttractionAddress` |
| **`Additional_Data_for_Attraction_Sites/Updated_Item.xlsx`** | Supplementary item features and details | Extended attraction attributes |

---

## 4. End-to-End Execution Pipeline

1. **Data Cleaning & Integration**
   - Resolve missing values across transaction, user, city, and item tables.
   - Standardize categorical names (`VisitMode`, `AttractionTypeId`, city names) and fix formatting inconsistencies.
   - Normalize/Standardize date and time attributes.
   - Identify and treat ratings/numerical outliers.

2. **Feature Engineering & Preprocessing**
   - Encode categorical attributes (`VisitMode`, `Continent`, `Country`, `AttractionTypeId`).
   - Merge transactional, demographic, location, and item datasets into a consolidated dataframe.
   - Aggregate user profile metrics (e.g., user average rating per visit mode).
   - Scale numerical features for model training.

3. **Exploratory Data Analysis (EDA) & Visualization**
   - Visualize demographic distributions across continents, countries, and cities.
   - Analyze attraction popularity vs. average ratings.
   - Examine relationship between `VisitMode` and demographic indicators.
   - Explore temporal patterns (visit year/month trends).

4. **Model Building & Algorithms**
   - **Regression**: Predict attraction ratings.
   - **Classification**: Train classifiers (e.g., Random Forest, LightGBM, XGBoost) to classify `VisitMode`.
   - **Recommendation System**: Build User-Item matrix for Collaborative Filtering and feature-vector matching for Content-Based Filtering.

5. **Evaluation Metrics**
   - **Classification**: Accuracy, Precision, Recall, F1-Score (with model performance comparison).
   - **Regression**: R², Mean Squared Error (MSE), Root Mean Squared Error (RMSE).
   - **Recommendation**: Mean Average Precision (MAP), RMSE.

6. **Deployment & Deliverables**
   - Interactive **Streamlit Application** allowing users to enter demographic/travel inputs to receive predicted visit modes and personalized attraction recommendations.
   - Visual dashboards displaying tourism trends, popular attractions, and user segmentation within Streamlit.

---

## 5. Technology Stack & Key Skills

- **Languages & Frameworks**: Python, Streamlit, SQL
- **Libraries**: Pandas, NumPy, Scikit-Learn, LightGBM, XGBoost, Matplotlib/Seaborn/Plotly
- **Core Skills**: Data Cleaning, Preprocessing, EDA, Classification, Regression, Collaborative & Content-Based Recommendation Systems, Web App Deployment.
