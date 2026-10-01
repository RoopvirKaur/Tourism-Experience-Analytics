"""
Main Entry Point for Tourism Experience Analytics Streamlit Application.
Sets up global styling, sidebar configuration, overview dashboards, and cached asset management.
"""

import sys
from pathlib import Path
import streamlit as st
import joblib
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import build_consolidated_dataset


# ------------------------------------------------------------------------------
# Page Configuration & Theme
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark/Glassmorphic Styling
STYLING_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }

    .gradient-title {
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 8px;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #38BDF8;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38BDF8;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .feature-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px;
        height: 100%;
        transition: border-color 0.2s ease;
    }

    .feature-card:hover {
        border-color: #818CF8;
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 12px;
    }

    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 8px;
    }

    .feature-desc {
        font-size: 0.92rem;
        color: #94A3B8;
        line-height: 1.5;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
"""
st.markdown(STYLING_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Cached Asset Loaders
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading master tourism dataset...")
def load_cached_data():
    """Loads and caches the master consolidated tourism dataframes."""
    master_df, users_df, items_df = build_consolidated_dataset()
    return master_df, users_df, items_df


@st.cache_resource(show_spinner="Loading trained machine learning models...")
def load_cached_artifacts():
    """Loads and caches serialized Phase 4 ML artifacts."""
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts = {}

    reg_path = artifacts_dir / "regression_best_model.pkl"
    if reg_path.exists():
        artifacts["regression"] = joblib.load(reg_path)

    cls_path = artifacts_dir / "classification_best_model.pkl"
    if cls_path.exists():
        artifacts["classification"] = joblib.load(cls_path)

    rec_path = artifacts_dir / "recommender.pkl"
    if rec_path.exists():
        artifacts["recommender"] = joblib.load(rec_path)

    comp_path = artifacts_dir / "model_comparison.csv"
    if comp_path.exists():
        artifacts["model_comparison"] = pd.read_csv(comp_path)

    return artifacts


# ------------------------------------------------------------------------------
# Application Entry Point
# ------------------------------------------------------------------------------
def main():
    # Load assets
    try:
        master_df, users_df, items_df = load_cached_data()
        artifacts = load_cached_artifacts()
    except Exception as e:
        st.error(f"Error loading system datasets or ML artifacts: {e}")
        st.stop()

    # Hero Banner
    st.markdown(
        """
        <div class="main-header">
            <div class="gradient-title">Tourism Experience Analytics</div>
            <p style="color: #94A3B8; font-size: 1.15rem; margin-bottom: 0;">
                End-to-End Travel Intelligence, Visit Mode Classification, Rating Prediction & Hybrid Recommendation Engine
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Key Statistics Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(master_df):,}</div>
                <div class="metric-label">Total Visits Analyzed</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(users_df):,}</div>
                <div class="metric-label">Unique Travelers</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(items_df):,}</div>
                <div class="metric-label">Attraction Sites</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        avg_rating = master_df["Rating"].mean()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{avg_rating:.2f} ⭐</div>
                <div class="metric-label">Average Satisfaction</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Platform Capabilities Section
    st.markdown("### 🚀 Analytics & Prediction Modules")
    st.write("Navigate using the sidebar menu on the left to access all analytical modules:")

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">1. Exploratory Data Analytics (EDA)</div>
                <div class="feature-desc">
                    Interactive dashboards uncovering global visitor demographics, regional tourist flows, 
                    attraction rating heatmaps, and temporal travel distributions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">⭐</div>
                <div class="feature-title">3. Attraction Rating Predictor</div>
                <div class="feature-desc">
                    Machine Learning regression engine estimating expected user rating (1.0 to 5.0) 
                    for specific attractions using demographic and historical interaction features.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m_col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🧳</div>
                <div class="feature-title">2. Visit Mode Classifier</div>
                <div class="feature-desc">
                    Multi-class machine learning classifier predicting travel group dynamics 
                    (Business, Family, Couples, Friends, Solo) with class confidence scores.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">4. Hybrid Recommendation Engine</div>
                <div class="feature-desc">
                    Personalized attraction recommender combining Collaborative Filtering and Content-Based TF-IDF 
                    vectors with cold-start fallback for new travelers.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Sidebar info
    st.sidebar.markdown("## 📌 System Status")
    st.sidebar.success("✓ Data Ready")
    st.sidebar.success("✓ ML Models Loaded")

    if "model_comparison" in artifacts:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🏆 Active ML Models")
        comp_df = artifacts["model_comparison"]
        best_models = comp_df[comp_df["Is_Best"] == "Yes"]
        for _, row in best_models.iterrows():
            task = row["Task"]
            model_name = row["Model"]
            if task == "Recommendation":
                model_name = "Hybrid CF + Content-Based"
            st.sidebar.markdown(f"**{task}**: {model_name}")
    else:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🏆 Active ML Models")
        st.sidebar.markdown("**Regression**: Linear Regression")
        st.sidebar.markdown("**Classification**: Random Forest Classifier")
        st.sidebar.markdown("**Recommendation**: Hybrid CF + Content-Based")

    st.sidebar.markdown("---")
    st.sidebar.caption("Tourism Experience Analytics v1.0.0 | Python & Streamlit")


if __name__ == "__main__":
    pages = [
        st.Page(main, title="Home", icon="🏠", default=True),
        st.Page("pages/01_EDA_Dashboard.py", title="EDA Dashboard", icon="📊"),
        st.Page("pages/02_Visit_Mode_Predictor.py", title="Visit Mode Predictor", icon="🔮"),
        st.Page("pages/03_Rating_Predictor.py", title="Rating Predictor", icon="⭐"),
        st.Page("pages/04_Attraction_Recommender.py", title="Attraction Recommender", icon="🗺️"),
    ]
    pg = st.navigation(pages)
    pg.run()

