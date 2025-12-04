import datetime as dt
from typing import Dict, List, Optional
from pandas import Timestamp, Timedelta

import pandas as pd
import streamlit as st

from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils.load_data import load_energy_consumption_data, load_energy_production_data, load_weather_data


st.set_page_config(page_title="SARIMAX Forecasting", page_icon="🔮", layout="wide")
st.title("🔮 SARIMAX Forecasting for Energy Metrics")

TIMEZONE = "Europe/Oslo"
TARGET_COLUMN_KEY = "quantitykwh"
PRODUCTION_GROUP = ["hydro", "wind", "solar", "thermal", "other"]
CONSUMPTION_GROUP = ["cabin", "household", "primary", "secondary", "tertiary"]

st.session_state.setdefault("production_data", pd.DataFrame())
st.session_state.setdefault("consumption_data", pd.DataFrame())
st.session_state.setdefault("loaded_prod_years", set())
st.session_state.setdefault("loaded_cons_years", set())
st.session_state.setdefault("price_area", "NO1")

# Inputs ----------------------------------------------------------------------------------------
st.info("Data is available from 2021 to 2024. Select years and parameters in the sidebar, then configure the model below.")
col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    train_start = st.date_input("Training start", pd.to_datetime("2021-01-01"))
    dataset_label = st.selectbox("Dataset", ["Energy Production", "Energy Consumption"], index=0)
with col_t2:
    train_end = st.date_input("Training end", pd.to_datetime("2021-12-31"))
    price_area = st.selectbox("Price Area", ["NO1", "NO2", "NO3", "NO4", "NO5"], index=["NO1", "NO2", "NO3", "NO4", "NO5"].index(st.session_state.price_area))
with col_t3:
    horizon_days = st.number_input(
        "Forecast horizon (days)",
        min_value=1,
        max_value=60,
        value=7,
        step=1,
    )
    #if  dataset_label == "Energy Production": then choos from list [hydro, wind, solar, thermal, other] when "consumption" then from [cabin, household, primary, secondary, tertiary]
    group = st.selectbox(
        "Choose group",
        (PRODUCTION_GROUP if dataset_label == "Energy Production" else CONSUMPTION_GROUP),
        index=0,
    )
if train_end < train_start:
    st.error("Training end date must be on or after start date.")

start_ts = pd.to_datetime(train_start).tz_localize(TIMEZONE)

end_ts = (pd.to_datetime(train_end) + pd.Timedelta(days=1)).tz_localize(TIMEZONE)  # make end exclusive
steps = int(horizon_days) * 24     



selected_years = list(range(start_ts.year, end_ts.year + 1))

# Load and preprocess data ------------------------------------------------------------------------

raw_df = pd.DataFrame()
if dataset_label == "Energy Production":
    for year in selected_years:
        if year not in st.session_state.loaded_prod_years:
            prod_df = load_energy_production_data(year).copy()
            prod_df.drop(columns=["_id"], inplace=True, errors="ignore")
            if "starttime" in prod_df.columns:
                prod_df["starttime"] = pd.to_datetime(prod_df["starttime"], errors="coerce", utc=True)
            prod_df["source_year"] = year
            prod_df = prod_df.dropna(subset=["starttime"])
            st.session_state.production_data = (
                prod_df
                if st.session_state.production_data.empty
                else pd.concat([st.session_state.production_data, prod_df], ignore_index=True)
            )
            st.session_state.loaded_prod_years.add(year)
    raw_df = st.session_state.production_data[
        st.session_state.production_data["source_year"].isin(selected_years)
    ].copy()
    #data from start_ts to end_ts and price area
    data = raw_df[(raw_df["starttime"] >= start_ts) & (raw_df["starttime"] < end_ts) & (raw_df["pricearea"].astype(str) == price_area) & (raw_df["productiongroup"].astype(str) == group)] 
else:
    for year in selected_years:
        if year not in st.session_state.loaded_cons_years:
            cons_df = load_energy_consumption_data(year).copy()
            cons_df.drop(columns=["_id"], inplace=True, errors="ignore")
            if "starttime" in cons_df.columns:
                cons_df["starttime"] = pd.to_datetime(cons_df["starttime"], errors="coerce", utc=True)
            cons_df["source_year"] = year
            cons_df = cons_df.dropna(subset=["starttime"])
            st.session_state.consumption_data = (
                cons_df
                if st.session_state.consumption_data.empty
                else pd.concat([st.session_state.consumption_data, cons_df], ignore_index=True)
            )
            st.session_state.loaded_cons_years.add(year)
    raw_df = st.session_state.consumption_data[
        st.session_state.consumption_data["source_year"].isin(selected_years)
    ].copy()
    data = raw_df[(raw_df["starttime"] >= start_ts) & (raw_df["starttime"] < end_ts) & (raw_df["pricearea"].astype(str) == price_area) & (raw_df["consumptiongroup"].astype(str) == group)]


if raw_df.empty:
    st.error("No data loaded for the selected years.")
    st.stop()
else: 
    st.success(f"Loaded {len(raw_df)} records for selected years.")

# SARIMAX parameters input --------------------------------------------------------------------------------
st.subheader("SARIMAX parameters")

c_p, c_d, c_q = st.columns(3)
with c_p:
    p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1, step=1)
with c_d:
    d = st.number_input("Differencing (d)", min_value=0, max_value=2, value=0, step=1)
with c_q:
    q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1, step=1)

c_P, c_D, c_Q, c_s = st.columns(4)
with c_P:
    P = st.number_input("Seasonal AR (P)", min_value=0, max_value=3, value=0, step=1)
with c_D:
    D = st.number_input("Seasonal diff (D)", min_value=0, max_value=1, value=0, step=1)
with c_Q:
    Q = st.number_input("Seasonal MA (Q)", min_value=0, max_value=3, value=0, step=1)
with c_s:
    seasonal_period = st.number_input(
        "Seasonal period (s, hours)",
        min_value=0,
        max_value=24 * 14,
        value=24,
        step=24,
        help="0 disables seasonality; 24 = daily, 24*7 = weekly for hourly data.",
    )

# Prepare exogenous variables (if any) ----------------------------------------------------------------
exog_train = None
exog_forecast = None
# here we would prepare exogenous variables if needed

exog_include = st.checkbox("Including exogenous variables.", value=False)
if exog_include:
    # importing weather data for the exog here
    forecast_end = Timestamp(end_ts) + Timedelta(days=int(horizon_days))
    st.write(forecast_end)
    years = list(range(start_ts.year, (forecast_end).year))
    st.write(years)
    weather_data = pd.concat([load_weather_data(price_area, year=year) for year in years])
    #option = colum names to choose from weather data
    weather_options = {col for col in weather_data.columns}
    weather_select = st.multiselect("Select weather variables as exogenous:", weather_options, default=[])
    if weather_select:
        weather_data.index = pd.to_datetime(weather_data.index, errors="coerce", utc=True)
        weather_data = weather_data.sort_index()
        
        exog_full = weather_data[weather_select].copy()
        exog_full = exog_full.tz_convert(TIMEZONE)

        exog_train = exog_full[(exog_full.index >= start_ts) & (exog_full.index < end_ts)].copy()
        exog_forecast = exog_full[(exog_full.index >= end_ts) & (exog_full.index < forecast_end)].copy()

        if exog_train.empty or exog_forecast.empty:
            st.error("Exogenous variables data is missing for the training or forecast period.")
            st.write(exog_train)
            st.write(exog_forecast)
            st.stop()

# Run SARIMAX forecast -------------------------------------------------------------------------------------
if st.button("Run SARIMAX Forecast"):
    data["starttime"] = pd.to_datetime(data["starttime"], errors="coerce", utc=True)
    data = data.sort_values("starttime")

    y = data.set_index("starttime")["quantitykwh"]

    if y.size < 48:
            st.warning("Not enough data in the training window (need at least 48 points).")
            

    last_time = y.index[-1]
    freq = pd.Timedelta(hours=1)
    future_index = pd.date_range(last_time + freq, periods=steps, freq=freq)

    order = (int(p), int(d), int(q))
    if seasonal_period > 0:
        seasonal_order = (int(P), int(D), int(Q), int(seasonal_period))
    else:
        seasonal_order = (0, 0, 0, 0)

    try:
        model = SARIMAX(
            endog=y,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False)
    except Exception as e:
        st.error("SARIMAX failed to fit with the chosen parameters above.")
        with st.expander("Show error details"):
            st.exception(e)

    try:
        forecast_res = results.get_forecast(steps=steps, exog=exog_forecast)
    except Exception as e:
        st.error("Forecasting failed (exogenous mismatch or model issue).")
        with st.expander("Show error details"):
            st.exception(e)
       
    future_index = pd.date_range(last_time + freq, periods=steps, freq=freq)

    forecast_res = results.get_forecast(steps=steps, exog=exog_forecast)
    fc_mean = forecast_res.predicted_mean.copy()
    conf_int = forecast_res.conf_int(alpha=0.05).copy()

    if isinstance(fc_mean.index, pd.RangeIndex):
        fc_mean.index = future_index
        conf_int.index = future_index

    # Visualization
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y, mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean, mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(
        x=conf_int.index.tolist() + conf_int.index[::-1].tolist(),
        y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='95% Confidence Interval'
    ))
    st.plotly_chart(fig)




    col1, col2, col3 = st.columns(3)
    col1.metric("Training RMSE", f"{results['rmse']:.2f}")
    col2.metric("AIC", f"{results['aic']:.2f}")
    col3.metric("BIC", f"{results['bic']:.2f}")

    with st.expander("Model diagnostics"):
        st.markdown("**SARIMAX summary**")
        st.text(results["model"].summary())
        st.markdown("**Residual sample**")
        st.dataframe(results["residuals"].to_frame(name="residuals").tail(20))