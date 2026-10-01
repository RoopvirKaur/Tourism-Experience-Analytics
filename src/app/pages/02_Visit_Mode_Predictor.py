"""
Page 2: Visit Mode Predictor.
Uses pre-trained classification models (Random Forest / XGBoost / Logistic Regression) to predict
the traveler group dynamics (Family, Couples, Friends, Solo, Business) based on origin demographics and visit context.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import build_consolidated_dataset


st.set_page_config(page_title="Visit Mode Predictor | Tourism Analytics", page_icon="🧳", layout="wide")


@st.cache_resource
def load_classification_model():
    model_path = PROJECT_ROOT / "artifacts" / "classification_best_model.pkl"
    if not model_path.exists():
        st.error("Classification model artifact not found! Please ensure Phase 4 training has executed.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data
def get_master_data():
    return build_consolidated_dataset()


def main():
    st.markdown("## 🧳 Visit Mode Classification Engine")
    st.markdown(
        "Predict traveler group dynamics (e.g., Family, Couples, Friends, Solo, Business) "
        "using traveler origin demographics and visit parameters."
    )

    bundle = load_classification_model()
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    target_encoder = bundle["target_encoder"]
    model_name = bundle["model_name"]
    val_macro_f1 = bundle["val_f1_macro"]

    master_df, _, items_df = get_master_data()

    col1, col2 = st.columns([1, 1.2])

    pred_mode = None

    with col1:
        st.markdown("### 📋 Traveler Context Inputs")

        # Select Continent
        continents = sorted([c for c in master_df["Continent"].dropna().unique() if c != "Unknown"])
        selected_cont = st.selectbox("Traveler Origin Continent:", continents)

        # Filter countries
        cont_countries = master_df[master_df["Continent"] == selected_cont]["Country"].dropna().unique()
        countries = sorted([c for c in cont_countries if c != "Unknown"])
        selected_country = st.selectbox("Traveler Origin Country:", countries if countries else ["Unknown"])

        # Filter regions
        country_regions = master_df[master_df["Country"] == selected_country]["Region"].dropna().unique()
        regions = sorted([r for r in country_regions if r != "Unknown"])
        selected_region = st.selectbox("Traveler Origin Region:", regions if regions else ["Unknown"])

        # Attraction Type Selection
        att_types = sorted(items_df["AttractionType"].dropna().unique().tolist())
        selected_att_type = st.selectbox("Target Attraction Type:", att_types)

        # Timing inputs
        col_m, col_y = st.columns(2)
        with col_m:
            selected_month = st.slider("Visit Month:", 1, 12, 6)
        with col_y:
            selected_year = st.number_input("Visit Year:", min_value=2015, max_value=2030, value=2023)

        predict_btn = st.button("🚀 Predict Visit Mode", type="primary", width="stretch")

    with col2:
        st.markdown("### 📊 Classification Output")

        if predict_btn:
            # Map selected text inputs back to IDs
            cont_row = master_df[master_df["Continent"] == selected_cont].iloc[0]
            ctr_row = master_df[master_df["Country"] == selected_country].iloc[0] if not master_df[master_df["Country"] == selected_country].empty else cont_row
            reg_row = master_df[master_df["Region"] == selected_region].iloc[0] if not master_df[master_df["Region"] == selected_region].empty else cont_row
            type_row = items_df[items_df["AttractionType"] == selected_att_type].iloc[0]

            input_sample = pd.DataFrame([{
                "UserId": 0,
                "AttractionId": type_row["AttractionId"],
                "ContinentId": cont_row["ContinentId"],
                "RegionId": reg_row["RegionId"],
                "CountryId": ctr_row["CountryId"],
                "UserCityId": ctr_row["UserCityId"] if "UserCityId" in ctr_row else 0,
                "AttractionTypeId": type_row["AttractionTypeId"],
                "VisitYear": selected_year,
                "VisitMonth": selected_month
            }])

            # Transform features
            X_input, _ = preprocessor.transform(input_sample)

            # Predict probabilities
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_input)[0]
                pred_idx = np.argmax(probs)
                pred_mode = target_encoder.inverse_transform([pred_idx])[0]
                confidence = probs[pred_idx] * 100
            else:
                pred_idx = model.predict(X_input)[0]
                pred_mode = target_encoder.inverse_transform([pred_idx])[0]
                probs = None
                confidence = 100.0

            # Mode Icons mapping
            mode_icons = {
                "Couples": "💑",
                "Family": "👨‍👩‍👧‍👦",
                "Friends": "👫",
                "Business": "💼",
                "Solo": "🎒"
            }
            icon = mode_icons.get(pred_mode, "🧳")

            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.8); border: 2px solid #818CF8; border-radius: 16px; padding: 25px; text-align: center; margin-bottom: 20px;">
                    <div style="font-size: 3rem;">{icon}</div>
                    <div style="font-size: 1.1rem; color: #94A3B8; text-transform: uppercase;">Predicted Travel Segment</div>
                    <div style="font-size: 2.5rem; font-weight: 700; color: #38BDF8;">{pred_mode}</div>
                    <div style="font-size: 1rem; color: #34D399; font-weight: 600; margin-top: 5px;">
                        Top Prediction Probability: {confidence:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Probability Distribution Chart
            if probs is not None:
                st.markdown("#### Confidence Probabilities Across Segments")
                prob_df = pd.DataFrame({
                    "VisitMode": target_encoder.classes_,
                    "Probability": probs * 100
                }).sort_values(by="Probability", ascending=True)

                fig_prob = px.bar(
                    prob_df,
                    x="Probability",
                    y="VisitMode",
                    orientation="h",
                    color="Probability",
                    color_continuous_scale="Purples",
                    text_auto=".1f"
                )
                fig_prob.update_layout(
                    template="plotly_dark",
                    height=280,
                    margin=dict(t=20, b=20, l=20, r=20),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_prob, width="stretch")

        else:
            st.info("👈 Select traveler origin and visit context on the left and click **Predict Visit Mode**.")

    # Full-width Actionable Strategy Insights (positioned below two-column main section)
    if pred_mode:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💼 Tailored Business & Marketing Strategy")
        strategies = {
            "Family": "Promote family-friendly tour packages, discounted group tickets, and kid-friendly amenities.",
            "Couples": "Highlight romantic dining packages, scenic view spots, luxury stays, and private tours.",
            "Friends": "Target group activity discounts, nightlife passes, adventure tours, and social media photo spots.",
            "Business": "Offer express entry tickets, fast WiFi lounges, business transport passes, and premium concierge service.",
            "Solo": "Suggest solo traveler meetups, audio guides, budget-friendly hostels, and self-guided walking routes."
        }
        st.info(strategies.get(pred_mode, "Provide customized tourism recommendations."))

    # Model Info Card
    st.markdown("---")
    st.caption(f"Active Model: **{model_name}** | Validation Macro F1: `{val_macro_f1:.4f}`")


if __name__ == "__main__":
    main()
