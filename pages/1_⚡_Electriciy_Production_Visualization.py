import streamlit as st
import pandas as pd
from utils.load_data import load_energy_production_data
from utils.visualization_electricity_production import create_pie_chart, create_lineplot_production


st.set_page_config(
    page_title="Electricity Production Visualization",
    page_icon="⚡",
    layout="wide"
)
st.title("⚡ Electricity Production Visualization")

st.write("Here you can explore enery production data from Norway for the year 2021")

color_map_production = {
    'hydro':   '#0072B2',  # Blue
    'thermal': "#C75702",  # Vermilion
    'wind':    '#009E73',  # Bluish green
    'solar':   '#E69F00',  # Orange (dunkler als Gelb, besser auf Weiß)
    'other':   '#6C757D'   # Neutral gray
}


year_options = [2021, 2022, 2023, 2024]
selected_years = st.pills(
    "Select year(s)",
    year_options,
    selection_mode="multi",
    default=[year_options[0]],
)

for year in selected_years:
    if year not in st.session_state.loaded_years:
        new_df = load_energy_production_data(year=year)
        new_df["starttime"] = pd.to_datetime(new_df["starttime"], errors="coerce", utc=True)
        st.session_state.production_data = (
            new_df if st.session_state.production_data.empty
            else pd.concat([st.session_state.production_data, new_df], ignore_index=True)
        )
        st.session_state.loaded_years.add(year)

production_data = st.session_state.production_data[
    st.session_state.production_data["starttime"].dt.year.isin(selected_years)
].copy()

col1, col2 = st.columns([1,2]) #splitting the slide in two

with col1: # ---------------Price area and pie chart------------------------
    
    options = ["NO1", "NO2", "NO3", "NO4", "NO5"]
    default_area = st.session_state.get("price_area")  # set your desired default here
    price_area = st.radio(
        "Select Price Area",
        options=options,
        index=options.index(default_area))
    st.session_state["price_area"] = price_area 
    #filtering the data for the selected price area
    filtered_data_area = production_data[production_data['pricearea'] == price_area]
    st.plotly_chart(create_pie_chart(filtered_data_area, price_area, color_map_production=color_map_production), use_container_width=True)
    st.info("Production Distribution: 'other' include all categories below 2% of total (e.g. thermal, wind, solar, other).", icon="ℹ️")

with col2: #------------------Production groups and line plot------------------------
    options = ["hydro", "solar", "thermal", "wind", "other"]
    production_groups = st.pills("Production Groups", options, selection_mode="multi", default=options)

    month_prod = st.slider("Select month", 1, 12, (1, 12))
    # filter data after months selected, options selected and price area 
    filtered_data_groups = production_data[
        (production_data['starttime'].dt.month >= month_prod[0]) &
        (production_data['starttime'].dt.month <= month_prod[1]) &
        (production_data['productiongroup'].isin(production_groups)) &
        (production_data['pricearea'] == price_area)
    ]
    st.plotly_chart(create_lineplot_production(filtered_data_groups, price_area, color_map_production=color_map_production), use_container_width=True)
    

#source expander
expander = st.expander("Sources")
expander.write("Elhub API data © 2025 Elhub AS, used under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: https://api.elhub.no/energy-data-api/price-areas. Title: “PRODUCTION_PER_GROUP_MBA_HOUR”. Changes: None. No endorsement by Elhub AS.")