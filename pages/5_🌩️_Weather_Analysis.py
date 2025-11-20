import streamlit as st
import utils.analysis_weather as weather_analysis
from utils.load_data import load_weather_data

st.set_page_config(
    page_title="Weather Analysis",
    page_icon="🌩️",
    layout="wide"
)
st.title("🌩️ Weather Analysis")

price_area = st.session_state.get("price_area")
weather_data = load_weather_data(price_area, year=2021)

#link to change price area
st.page_link(
    "pages/1_⚡_Electriciy_Production_Visualization.py",
    label="⬅️ Go to ⚡ Electricity Production Visualization to change Price Area"
)

st.write(f"Here you can analyse the weather data in the selected price area {price_area}.")

variable = st.selectbox(
"What data would you like to analyze?",
("temperature", "precipitation", "wind speed", "wind gusts", "wind direction"),
index=0, 
placeholder="Select an option",
help="Choose the weather variable you want to see plotted"
)

#dict for selectbox and dataframe column mapping
variable_column_mapping = {
    "temperature": "temperature_2m",
    "precipitation": "precipitation",
    "wind speed": "wind_speed_10m",
    "wind gusts": "wind_gusts_10m",
    "wind direction": "wind_direction_10m",
}

tab1, tab2 = st.tabs(["SPC", "LOF"])

colors = {
    "temperature_2m": "#C4611A",
    "precipitation": "#3173EE",
    "wind_speed_10m": "#AD4DE0",
    "wind_gusts_10m": "#3C1053",
    "wind_direction_10m": "#075E50",
}

with tab1:
    st.write("SPC")
    #weather_analysis.plot_temperature_with_spc(weather_data)
    fig = weather_analysis.dct_outliers(weather_data, target_col=variable_column_mapping[variable], colors=colors)
    st.plotly_chart(fig)

with tab2:
    st.write("LOF")
    #silder for n_neighbors and contamination
    n_neighbors = st.slider("Number of Neighbors (n_neighbors):", min_value=5, max_value=50, value=20, step=1, help="Number of neighbors to use for LOF.")
    contamination = st.slider("Contamination (proportion of outliers):", min_value=0.01, max_value=0.1, value=0.01, step=0.01, help="Proportion of outliers in the data set.")
    
    #filtered_data = weather_data[[variable_column_mapping[variable]]].copy()

    fig = weather_analysis.apply_lof_time_series(weather_data, target_col=variable_column_mapping[variable], n_neighbors=n_neighbors, contamination=contamination)
    st.pyplot(fig)
