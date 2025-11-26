import streamlit as st
import pandas as pd
import datetime as dt

st.set_page_config(
    page_title="SARIMAX Forecasting",
    page_icon="📈",
    layout="wide"
)
st.title("📈 SARIMAX Forecasting for Energy Metrics")
st.caption(
    "Interactively train SARIMAX models on energy production or consumption data, configure seasonal parameters, "
    "and generate dynamic forecasts with confidence intervals."
)

price_area = st.session_state.get("price_area", "NO1")

options = ["production", "consumption"]
  # set your desired default here
pred_data_option = st.radio(
    "Select Data to Predict",
    options=options,
    index=0 )

min_dt = dt.datetime("2021-12-31 23:00:00+01:00")
max_dt = dt.datemtime("2024-12-31 23:00:00+01:00")


start_dt, end_dt = st.slider(
    "Select time range",
    min_value=min_dt,
    max_value=max_dt,
    value=(min_dt, max_dt),
    step=dt.timedelta(days=1),  # change to days if you want coarser control
)
start_ts = pd.Timestamp(start_dt)
end_ts = pd.Timestamp(end_dt)

if pred_data_option == "production":
    st.write(f"You are predicting **{price_area}** production data.")

else:
    st.write(f"You are predicting **{price_area}** consumption data.")
