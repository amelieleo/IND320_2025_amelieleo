import streamlit as st
import pandas as pd
import datetime 
from utils.load_data import load_weather_data



st.set_page_config(
    page_title="Weather Data",
    page_icon="📈",  # sun behind cloud icon
    layout="wide",
)

st.title("📈 Weather Data")
price_area = st.session_state.get("price_area")
weather_data = load_weather_data(price_area, year=2021)
st.session_state.weather_data = weather_data

#link to change price area
st.page_link(
    "pages/1_⚡_Electriciy_Production_Visualization.py",
    label="⬅️ Go to ⚡ Electricity Production Visualization to change Price Area"
)

st.text("Here is the weather data for the first month in the dataset:")
# Filter the first month
first_month = st.session_state.weather_data.index[0].month
first_year = st.session_state.weather_data.index[0].year
first_month_data = st.session_state.weather_data[(st.session_state.weather_data.index.month == first_month) & (st.session_state.weather_data.index.year == first_year)]
# Transpose so each variable is a row, each time is a column
first_month_data = first_month_data.T
# Add a column with the time series as a list for each variable
first_month_data['values'] = first_month_data.values.tolist()
# Reset index so variable names are a column
first_month_data = first_month_data.reset_index().rename(columns={'index': 'Variable'})
# Show in data_editor with a row-wise LineChartColumn
st.data_editor(
    first_month_data[['Variable', 'values']],
    column_config={
        "values": st.column_config.LineChartColumn(
            "Time Series (First Month)",
            width="medium",
            help="Time series for the first month"
        ),
    },
    hide_index=True,
)
