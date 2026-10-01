import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def home():
    st.title("Home")

pg = st.navigation([
    st.Page(home, title="Home", icon="🏠", default=True),
    st.Page("pages/01_EDA_Dashboard.py", title="EDA Dashboard", icon="📊"),
    st.Page("pages/02_Visit_Mode_Predictor.py", title="Visit Mode Predictor", icon="🔮"),
    st.Page("pages/03_Rating_Predictor.py", title="Rating Predictor", icon="⭐"),
    st.Page("pages/04_Attraction_Recommender.py", title="Attraction Recommender", icon="🗺️"),
])
pg.run()
