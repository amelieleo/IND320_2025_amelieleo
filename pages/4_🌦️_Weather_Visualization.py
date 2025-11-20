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

#Plotting based on user input 
if variable == "temperature":
    st.plotly_chart(vwd.plot_temp(filtered_data))
elif variable == "precipitation":
    st.pyplot(vwd.plot_percipitation(filtered_data))
elif variable == "wind speed":
    st.pyplot(vwd.plot_wind_speed(filtered_data))  
elif variable == "wind gusts":
    st.pyplot(vwd.plot_wind_gusts(filtered_data))
elif variable == "wind direction":
    st.pyplot(vwd.plot_wind_direction(filtered_data))
elif variable == "All variables":
    st.pyplot(vwd.plot_all(filtered_data))  