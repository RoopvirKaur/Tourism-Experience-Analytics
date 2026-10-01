"""
Page 3: Rating Predictor.
Uses pre-trained regression models (Linear Regression / Random Forest / XGBoost) to estimate the expected
rating (1.0 to 5.0) a traveler is likely to give to a specific tourist attraction.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import build_consolidated_dataset


st.set_page_config(page_title="Rating Predictor | Tourism Analytics", page_icon="⭐", layout="wide")


@st.cache_resource
def load_regression_model():
    model_path = PROJECT_ROOT / "artifacts" / "regression_best_model.pkl"
    if not model_path.exists():
        st.error("Regression model artifact not found! Please ensure Phase 4 training has executed.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data
def get_master_data():
    return build_consolidated_dataset()


def get_star_rating(rating: float) -> str:
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    return "★" * full_stars + ("½" if half_star else "") + "☆" * empty_stars


def main():
    st.markdown("## ⭐ Attraction Rating Prediction Engine")
    st.markdown("Estimate expected traveler satisfaction rating (1.0 to 5.0) for tourist attraction visits.")

    bundle = load_regression_model()
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    model_name = bundle["model_name"]
    val_rmse = bundle["val_rmse"]

    master_df, _, items_df = get_master_data()

    # Top Two-Column Section: Inputs on Left, Result on Right
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🎯 Prediction Inputs")

        # Attraction Selection with default placeholder
        placeholder = "Select an attraction..."
        raw_attractions = sorted(items_df["Attraction"].dropna().unique().tolist())
        attractions_list = [placeholder] + [a for a in raw_attractions if a != placeholder]
        selected_attraction = st.selectbox("Select Target Attraction:", attractions_list, index=0)

        # Display location and category ONLY if an actual attraction is selected
        if selected_attraction != placeholder:
            item_row = items_df[items_df["Attraction"] == selected_attraction].iloc[0]
            att_id = item_row["AttractionId"]
            att_type = item_row["AttractionType"]
            att_city = item_row["AttractionCityName"]
            st.markdown(
                f"""
                <div style="margin-top: 2px; margin-bottom: 8px; font-size: 0.85rem; color: #94A3B8;">
                    📍 Location: <strong style="color: #E2E8F0;">{att_city}</strong> &nbsp;|&nbsp; Category: <strong style="color: #E2E8F0;">{att_type}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            item_row = None
            att_id = None

        # Visit Mode selection (compact spacing directly below metadata)
        visit_modes = sorted(master_df["VisitModeName"].dropna().unique().tolist())
        selected_mode = st.selectbox("Select Visit Mode:", visit_modes)
        mode_row = master_df[master_df["VisitModeName"] == selected_mode].iloc[0]

        # Continent Selection
        continents = sorted([c for c in master_df["Continent"].dropna().unique() if c != "Unknown"])
        selected_cont = st.selectbox("Traveler Origin Continent:", continents)
        cont_row = master_df[master_df["Continent"] == selected_cont].iloc[0]

        # Timing Selection
        col_m, col_y = st.columns(2)
        with col_m:
            selected_month = st.slider("Visit Month:", 1, 12, 7)
        with col_y:
            selected_year = st.number_input("Visit Year:", min_value=2015, max_value=2030, value=2023)

        predict_btn = st.button("🔮 Estimate Rating", type="primary", width="stretch")

    predicted_rating = None
    hist_avg = float(master_df["Rating"].mean()) if not master_df.empty else 4.16

    # Calculate prediction if form submitted
    if predict_btn:
        if selected_attraction == placeholder or att_id is None:
            st.warning("⚠️ Please select an attraction to estimate rating.")
        else:
            input_sample = pd.DataFrame([{
                "UserId": 0,
                "AttractionId": att_id,
                "ContinentId": cont_row["ContinentId"],
                "RegionId": cont_row["RegionId"],
                "CountryId": cont_row["CountryId"],
                "UserCityId": cont_row["UserCityId"] if "UserCityId" in cont_row else 0,
                "AttractionTypeId": item_row["AttractionTypeId"],
                "VisitModeId": mode_row["VisitModeId"],
                "VisitYear": selected_year,
                "VisitMonth": selected_month
            }])

            X_input = preprocessor.transform(input_sample)
            raw_pred = model.predict(X_input)[0]
            predicted_rating = float(np.clip(raw_pred, 1.0, 5.0))
            st.session_state["predicted_rating"] = predicted_rating

    if selected_attraction == placeholder:
        if "predicted_rating" in st.session_state:
            del st.session_state["predicted_rating"]
        predicted_rating = None
    elif "predicted_rating" in st.session_state:
        predicted_rating = st.session_state["predicted_rating"]

    with col2:
        st.markdown("### 📊 Prediction Result")

        if predicted_rating is not None:
            stars = get_star_rating(predicted_rating)
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #38BDF8; border-radius: 14px; padding: 24px 20px; min-height: 420px; display: flex; flex-direction: column; justify-content: space-between; align-items: center; text-align: center; box-sizing: border-box;">
                    <div>
                        <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.8px; margin-bottom: 10px;">EXPECTED SATISFACTION RATING</div>
                        <div style="font-size: 3.5rem; font-weight: 800; color: #38BDF8; margin: 6px 0; line-height: 1.0;">{predicted_rating:.2f} <span style="font-size: 1.8rem; font-weight: 500; color: #94A3B8;">/ 5.0</span></div>
                        <div style="font-size: 1.8rem; color: #FACC15; margin: 6px 0;">{stars}</div>
                    </div>
                    <div style="margin: 12px 0;">
                        <div style="font-size: 0.95rem; font-weight: 600; color: #E2E8F0;">Expected satisfaction rating</div>
                        <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px; font-style: italic;">
                            Higher values indicate greater expected traveler satisfaction.
                        </div>
                    </div>
                    <div style="width: 100%; border-top: 1px solid rgba(148, 163, 184, 0.2); padding-top: 12px; font-size: 0.82rem; color: #94A3B8;">
                        Model: <strong style="color: #F8FAFC;">{model_name}</strong> &nbsp;|&nbsp; Validation RMSE: <code style="color: #38BDF8; background: transparent;">{val_rmse:.4f}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.4); border: 1px dashed #475569; border-radius: 14px; padding: 24px 20px; min-height: 420px; display: flex; flex-direction: column; justify-content: space-between; align-items: center; text-align: center; box-sizing: border-box;">
                    <div></div>
                    <div style="margin: 16px 0;">
                        <div style="font-size: 2.2rem; margin-bottom: 8px;">🔮</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #E2E8F0; margin-bottom: 6px;">Ready to Estimate Rating</div>
                        <div style="font-size: 0.88rem; color: #94A3B8; max-width: 320px;">Select an attraction and visit parameters on the left, then click <strong>Estimate Rating</strong> to generate predictions.</div>
                    </div>
                    <div style="width: 100%; border-top: 1px solid rgba(148, 163, 184, 0.2); padding-top: 12px; font-size: 0.82rem; color: #94A3B8;">
                        Model: <strong style="color: #F8FAFC;">{model_name}</strong> &nbsp;|&nbsp; Validation RMSE: <code style="color: #38BDF8; background: transparent;">{val_rmse:.4f}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Lower Section: Benchmark & Insights
    if predicted_rating is not None:
        diff = predicted_rating - hist_avg
        diff_str = f"{'+' if diff >= 0 else ''}{diff:.2f}"

        st.markdown("---")
        st.markdown("### 📈 Rating Benchmark")

        # Two-Column Layout: Left = 3 Vertically Stacked Metric Cards, Right = Large Gauge
        bench_col1, bench_col2 = st.columns([5, 7])

        with bench_col1:
            st.markdown(
                f"""
                <div style="display: flex; flex-direction: column; gap: 10px; height: 100%; justify-content: space-between;">
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid #334155; border-radius: 10px; padding: 12px 16px;">
                        <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; margin-bottom: 2px;">Historical Average</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #F8FAFC;">{hist_avg:.2f} <span style="font-size: 0.85rem; color: #94A3B8;">/ 5.0</span></div>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid #38BDF8; border-radius: 10px; padding: 12px 16px;">
                        <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; margin-bottom: 2px;">Predicted Satisfaction</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #38BDF8;">{predicted_rating:.2f} <span style="font-size: 0.85rem; color: #94A3B8;">/ 5.0</span></div>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid #334155; border-radius: 10px; padding: 12px 16px;">
                        <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; margin-bottom: 2px;">Difference From Average</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: {'#34D399' if diff >= 0 else '#F87171'};">{diff_str}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with bench_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_rating,
                title={'text': "Predicted Satisfaction", 'font': {'size': 16, 'color': "#F8FAFC"}, 'align': 'center'},
                number={'suffix': " / 5.0", 'font': {'size': 22, 'color': "#38BDF8"}, 'valueformat': ".2f"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [1, 5], 'tickwidth': 1, 'tickcolor': "#94A3B8", 'dtick': 1, 'tickfont': {'size': 13}},
                    'bar': {'color': "#38BDF8", 'thickness': 0.5},
                    'steps': [
                        {'range': [1, 2.5], 'color': "rgba(239, 68, 68, 0.25)"},
                        {'range': [2.5, 3.8], 'color': "rgba(234, 179, 8, 0.25)"},
                        {'range': [3.8, 5.0], 'color': "rgba(34, 197, 94, 0.25)"}
                    ],
                    'threshold': {
                        'line': {'color': "#EF4444", 'width': 4},
                        'thickness': 0.75,
                        'value': hist_avg
                    }
                }
            ))
            fig_gauge.update_layout(
                template="plotly_dark",
                height=240,
                margin=dict(t=50, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_gauge, width="stretch")

            st.markdown(
                "<div style='text-align: center; color: #94A3B8; font-size: 0.82rem; margin-top: -5px;'>"
                "Rating scale: 1 = lower satisfaction &middot; 5 = higher satisfaction"
                "</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### 💡 Satisfaction Optimization Insights")

        diff_status = "above" if diff >= 0 else "below"
        diff_mag = f"{abs(diff):.2f} points"

        if predicted_rating >= 4.0:
            badge_color = "#34D399"
            bg_badge = "rgba(52, 211, 153, 0.15)"
            status_title = "High Expected Satisfaction"
            summary_text = f"The regression model predicts strong traveler satisfaction ({predicted_rating:.2f} / 5.0), performing {diff_mag} {diff_status} the historical baseline ({hist_avg:.2f})."
            points = [
                "<strong>Visitor Appeal:</strong> High expected satisfaction under the selected traveler profile and visit timing.",
                "<strong>Marketing Strategy:</strong> Well-suited for flagship promotional campaigns, featured travel listings, and premium tour packages.",
                "<strong>Operational Focus:</strong> Maintain service consistency, facility quality, and crowd management to sustain high satisfaction ratings."
            ]
        elif predicted_rating >= 3.0:
            badge_color = "#FBBF24"
            bg_badge = "rgba(251, 191, 36, 0.15)"
            status_title = "Moderate Expected Satisfaction"
            summary_text = f"The regression model predicts moderate traveler satisfaction ({predicted_rating:.2f} / 5.0), performing {diff_mag} {diff_status} the historical baseline ({hist_avg:.2f})."
            points = [
                "<strong>Visitor Appeal:</strong> Satisfactory overall traveler experience expected, with key opportunities for service enhancement.",
                "<strong>Operational Guidance:</strong> Focus on targeted improvements in visitor amenities, queue management, and guided tour quality.",
                "<strong>Feedback Alignment:</strong> Review seasonal visitor feedback to address specific friction points during peak travel months."
            ]
        else:
            badge_color = "#F87171"
            bg_badge = "rgba(248, 113, 113, 0.15)"
            status_title = "Lower Expected Satisfaction"
            summary_text = f"The regression model predicts lower expected satisfaction ({predicted_rating:.2f} / 5.0), performing {diff_mag} {diff_status} the historical baseline ({hist_avg:.2f})."
            points = [
                "<strong>Visitor Experience Gap:</strong> Potential traveler dissatisfaction indicated under the selected travel parameters.",
                "<strong>Operational Review:</strong> Conduct an operational audit covering facility maintenance, staff training, and visitor amenities.",
                "<strong>Value Alignment:</strong> Evaluate ticket pricing structure and service delivery prior to major marketing campaigns."
            ]

        points_html = "".join([f"<li style='margin-bottom: 6px; color: #CBD5E1; font-size: 0.92rem;'>{pt}</li>" for pt in points])

        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {badge_color}; border-radius: 12px; padding: 20px 24px; margin-top: 8px;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; border-bottom: 1px solid rgba(148, 163, 184, 0.15); padding-bottom: 10px;">
                    <div style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC;">Executive Summary & Operational Guidance</div>
                    <span style="background: {bg_badge}; color: {badge_color}; border: 1px solid {badge_color}; padding: 4px 12px; border-radius: 6px; font-weight: 600; font-size: 0.8rem; text-transform: uppercase;">
                        {status_title}
                    </span>
                </div>
                <p style="color: #E2E8F0; font-size: 0.95rem; margin-bottom: 12px; line-height: 1.5;">
                    {summary_text}
                </p>
                <ul style="margin: 0; padding-left: 20px; line-height: 1.6;">
                    {points_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
