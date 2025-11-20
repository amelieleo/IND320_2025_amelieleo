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

price_area = st.session_state.get("price_area")
weather_data = load_weather_data(price_area, year=2021)

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

#Slider to choose which months to visualize
month = st.slider("Select month", 1, 12, (1, 12))
filtered_data = weather_data[(weather_data.index.month >= month[0]) & (weather_data.index.month <= month[1])]


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
    st.plotly_chart(vwd.plot_temp(filtered_data, colors=colors))
elif variable == "precipitation":
    st.plotly_chart(vwd.plot_precipitation(filtered_data, colors=colors))
elif variable == "wind speed":
    st.plotly_chart(vwd.plot_wind_speed(filtered_data, colors=colors))  
elif variable == "wind gusts":
    st.plotly_chart(vwd.plot_wind_gusts(filtered_data, colors=colors))
elif variable == "wind direction":
    st.plotly_chart(vwd.plot_wind_direction_plotly(filtered_data, colors=colors))
elif variable == "All variables":
    st.plotly_chart(vwd.plot_all(filtered_data, colors=colors))  