import streamlit as st 
import pandas as pd
from utils.load_data import load_weather_data
from utils import visualization_weather_data as vwd

st.set_page_config(
    page_title="Weather Visualization",
    page_icon="🌦️",  # sun behind small cloud icon
    layout="centered",
)



st.title("🌦️ Weather Visualization")

# Link to change price area
st.page_link(
    "pages/1_⚡_Electriciy_Production_Visualization.py",
    label="⬅️ Go to ⚡ Electricity Production Visualization to change Price Area"
)

variable = st.selectbox(
"What data would you like to visualize?",
("temperature", "precipitation", "wind speed", "wind gusts", "wind direction", "All variables"),
index=0, 
placeholder="Select an option",
help="Choose the weather variable you want to see plotted"
)

year_options = [2021, 2022, 2023, 2024]
selected_years = st.multiselect(
    "Select year(s)",
    year_options,
    default=[year_options[0]],
    help="Weather data will be combined across the selected years.",
)

price_area = st.session_state.get("price_area")
if st.session_state.get("clicked_points") is None: 
    weather_data = pd.concat([load_weather_data(price_area, year=year) for year in selected_years])
else:
    points = st.session_state.get("clicked_points")
    weather_df = pd.concat([load_weather_data(
    latitude=points[0],
    longitude=points[1],
    price_area=price_area,
    year=year,
    )]for year in selected_years)

#defining colors
colors = {
    "temperature": "#C4611A",
    "precipitation": "#3173EE",
    "wind speed": "#AD4DE0",
    "wind gusts": "#3C1053",
    "wind direction": "#075E50",
}

#Plotting based on user input 
if variable == "temperature":
    st.plotly_chart(vwd.plot_temp(weather_data, colors=colors))
elif variable == "precipitation":
    st.plotly_chart(vwd.plot_precipitation(weather_data, colors=colors))
elif variable == "wind speed":
    st.plotly_chart(vwd.plot_wind_speed(weather_data, colors=colors))  
elif variable == "wind gusts":
    st.plotly_chart(vwd.plot_wind_gusts(weather_data, colors=colors))
elif variable == "wind direction":
    st.plotly_chart(vwd.plot_wind_direction_plotly(weather_data, colors=colors))
elif variable == "All variables":
    st.plotly_chart(vwd.plot_all(weather_data, colors=colors, series_colors=colors))  