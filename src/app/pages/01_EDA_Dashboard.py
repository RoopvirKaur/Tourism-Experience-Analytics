"""
Page 1: Exploratory Data Analytics (EDA) Dashboard.
Provides interactive visual insights into visitor demographics, top attractions, temporal trends, and visit modes.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as gg

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import build_consolidated_dataset


st.set_page_config(page_title="EDA Dashboard | Tourism Experience Analytics", page_icon="📊", layout="wide")


@st.cache_data
def get_dashboard_data():
    master_df, users_df, items_df = build_consolidated_dataset()
    return master_df, users_df, items_df


def main():
    master_df, users_df, items_df = get_dashboard_data()

    # Header & Filters Section (Filters on the right side)
    col_header, col_filter = st.columns([1.6, 1.4], gap="medium")

    with col_header:
        st.markdown("## 📊 Exploratory Data Analytics Dashboard")
        st.markdown(
            "Explore spatial visitor demographics, attraction popularity, "
            "temporal trends, and group travel behavior."
        )

    with col_filter:
        with st.container(border=True):
            st.markdown("#### 🔎 Dashboard Filters")

            f_col1, f_col2 = st.columns(2)

            all_continents = ["All"] + sorted([c for c in master_df["Continent"].dropna().unique() if c != "Unknown"])
            with f_col1:
                selected_continent = st.selectbox("Filter by Traveler Continent:", all_continents)

            all_years = ["All"] + sorted(master_df["VisitYear"].unique().tolist())
            with f_col2:
                selected_year = st.selectbox("Filter by Visit Year:", all_years)

            # Filter application
            filtered_df = master_df.copy()
            if selected_continent != "All":
                filtered_df = filtered_df[filtered_df["Continent"] == selected_continent]
            if selected_year != "All":
                filtered_df = filtered_df[filtered_df["VisitYear"] == selected_year]

            st.caption(f"ℹ️ Showing **{len(filtered_df):,}** visit logs")

    # Tabbed Interface for Organization
    tab1, tab2, tab3 = st.tabs([
        "🌍 Demographic & Geographical Distribution",
        "🏰 Attraction Site Performance",
        "📅 Temporal Trends & Visit Modes"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: Demographics
    # --------------------------------------------------------------------------
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Visitor Share by Continent")
            cont_counts = filtered_df["Continent"].value_counts().reset_index()
            cont_counts.columns = ["Continent", "Count"]

            fig_cont = px.pie(
                cont_counts,
                names="Continent",
                values="Count",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_cont.update_traces(
                textfont=dict(size=14, weight="bold"),
                insidetextfont=dict(size=14, weight="bold")
            )
            fig_cont.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_cont, width="stretch")

        with col2:
            st.markdown("#### Top 10 Visitor Origin Countries")
            country_counts = filtered_df["Country"].value_counts().head(10).reset_index()
            country_counts.columns = ["Country", "Visits"]

            fig_country = px.bar(
                country_counts,
                x="Visits",
                y="Country",
                orientation="h",
                color="Visits",
                color_continuous_scale="Viridis"
            )
            fig_country.update_layout(
                template="plotly_dark",
                yaxis=dict(autorange="reversed"),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_country, width="stretch")

    # --------------------------------------------------------------------------
    # TAB 2: Attraction Site Performance
    # --------------------------------------------------------------------------
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top 10 Most Visited Attractions")
            top_attractions = (
                filtered_df.groupby("Attraction")
                .agg(Visits=("Rating", "count"), Avg_Rating=("Rating", "mean"))
                .reset_index()
                .sort_values(by="Visits", ascending=False)
                .head(10)
            )

            fig_att = px.bar(
                top_attractions,
                x="Visits",
                y="Attraction",
                orientation="h",
                color="Avg_Rating",
                color_continuous_scale="Tealgrn",
                hover_data=["Avg_Rating"]
            )
            fig_att.update_layout(
                template="plotly_dark",
                yaxis=dict(autorange="reversed"),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_att, width="stretch")

        with col2:
            st.markdown("#### Attraction Type Satisfaction Overview")
            type_ratings = (
                filtered_df.groupby("AttractionType")
                .agg(Avg_Rating=("Rating", "mean"), Total_Reviews=("Rating", "count"))
                .reset_index()
                .sort_values(by="Avg_Rating", ascending=False)
            )

            fig_type = px.bar(
                type_ratings,
                x="AttractionType",
                y="Avg_Rating",
                color="Total_Reviews",
                color_continuous_scale="Plasma",
                hover_data=["Total_Reviews"]
            )
            fig_type.update_layout(
                template="plotly_dark",
                xaxis=dict(tickangle=-45),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_type, width="stretch")

    # --------------------------------------------------------------------------
    # TAB 3: Temporal Trends & Visit Modes
    # --------------------------------------------------------------------------
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Monthly Visit Volume Distribution")
            monthly_visits = (
                filtered_df.groupby(["VisitMonth", "VisitYear"])
                .size()
                .reset_index(name="Visits")
            )
            month_names = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
            }
            monthly_visits["Month"] = monthly_visits["VisitMonth"].map(month_names)

            fig_monthly = px.line(
                monthly_visits,
                x="Month",
                y="Visits",
                color="VisitYear",
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig_monthly.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_monthly, width="stretch")

        with col2:
            st.markdown("#### Visit Mode Distribution (Group Dynamics)")
            mode_counts = (
                filtered_df["VisitModeName"]
                .value_counts()
                .reset_index()
            )
            mode_counts.columns = ["VisitMode", "Count"]

            fig_mode = px.bar(
                mode_counts,
                x="VisitMode",
                y="Count",
                color="VisitMode",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_mode.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_mode, width="stretch")


if __name__ == "__main__":
    main()
