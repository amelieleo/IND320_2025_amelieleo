from __future__ import annotations

from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.correlation import compute_sliding_correlation
from utils.load_data import (
    load_energy_consumption_data,
    load_energy_production_data,
    load_weather_data,
)


st.set_page_config(page_title="Sliding Window Correlation", page_icon="🔄", layout="wide")
st.title("🔄 Sliding Window Correlation: Meteorology vs. Energy Metrics")

price_area = st.session_state.get("price_area", "NO1")
st.caption(f"Active price area: **{price_area}**")


@lru_cache(maxsize=32)
def get_production(year: int) -> pd.DataFrame:
    df = load_energy_production_data(year)
    if df.empty:
        return df
    df = df.copy()
    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce", utc=True)
    return df.dropna(subset=["starttime"])


@lru_cache(maxsize=32)
def get_consumption(year: int) -> pd.DataFrame:
    df = load_energy_consumption_data(year)
    if df.empty:
        return df
    df = df.copy()
    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce", utc=True)
    return df.dropna(subset=["starttime"])


@lru_cache(maxsize=32)
def get_weather(year: int, area: str) -> pd.DataFrame:
    df = load_weather_data(price_area=area, year=year, latitude=None, longitude=None)
    df = df.reset_index().rename(columns={"date": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df.rename(
        columns={
            "precipitation": "precipitation (mm)",
            "temperature_2m": "temperature_2m (°C)",
            "wind_speed_10m": "wind_speed_10m (m/s)",
            "wind_direction_10m": "wind_direction_10m (°)",
            "wind_gusts_10m": "wind_gusts_10m (m/s)",
        }
    )
    return df


year_options = [2021, 2022, 2023, 2024]
selected_years = st.multiselect(
    "Select year(s)",
    year_options,
    default=[year_options[0]],
    help="Data will be concatenated across the chosen years.",
)

if not selected_years:
    st.warning("Select at least one year to proceed.")
    st.stop()

with st.spinner("Loading datasets…"):
    prod_frames = [get_production(year) for year in selected_years]
    cons_frames = [get_consumption(year) for year in selected_years]
    met_frames = [get_weather(year, price_area) for year in selected_years]

production_df = pd.concat(prod_frames, ignore_index=True) if prod_frames else pd.DataFrame()
consumption_df = pd.concat(cons_frames, ignore_index=True) if cons_frames else pd.DataFrame()
weather_df = pd.concat(met_frames, ignore_index=True) if met_frames else pd.DataFrame()

if production_df.empty or consumption_df.empty or weather_df.empty:
    st.error("One or more datasets are empty for the chosen years. Please adjust your selection.")
    st.stop()

meteorology_choices = [
    col
    for col in ["precipitation (mm)", "temperature_2m (°C)", "wind_speed_10m (m/s)", "wind_direction_10m (°)", "wind_gusts_10m (m/s)"]
    if col in weather_df.columns
]
energy_numeric_cols = sorted(
    {col for col in production_df.columns if col not in {"starttime"} and pd.api.types.is_numeric_dtype(production_df[col])}
    | {
        col
        for col in consumption_df.columns
        if col not in {"starttime"} and pd.api.types.is_numeric_dtype(consumption_df[col])
    }
)

col_select_met, col_select_energy = st.columns(2)
with col_select_met:
    met_col = st.selectbox(
        "Meteorological variable",
        meteorology_choices,
        index=0,
    )

with col_select_energy:
    dataset_choice = st.radio(
        "Energy dataset",
        ("Production", "Consumption"),
        horizontal=True,
    )
    if dataset_choice == "Production":
        dataset_columns = [col for col in energy_numeric_cols if col in production_df.columns]
    else:
        dataset_columns = [col for col in energy_numeric_cols if col in consumption_df.columns]

    if not dataset_columns:
        st.warning(f"No numeric columns found in {dataset_choice.lower()} data.")
        st.stop()

    energy_col = st.selectbox(
        f"{dataset_choice} variable",
        dataset_columns,
        index=0,
    )

lag_hours = st.slider(
    "Lag energy data (hours)",
    min_value=-168,
    max_value=168,
    value=0,
    step=1,
    help="Positive values lag energy behind meteorology; negative values lead energy.",
)

window_hours = st.slider(
    "Window length (hours)",
    min_value=24,
    max_value=24 * 21,
    value=24 * 7,
    step=24,
    help="Rolling window size for correlation.",
)

frequency = st.radio(
    "Aggregation frequency",
    options=("Hourly", "Daily"),
    index=0,
    horizontal=True,
)
freq_code = "H" if frequency == "Hourly" else "D"

if dataset_choice == "Production":
    energy_source = production_df
    time_col_energy = "starttime"
else:
    energy_source = consumption_df
    time_col_energy = "starttime"

result = compute_sliding_correlation(
    met_df=weather_df,
    energy_df=energy_source,
    met_col=met_col,
    energy_col=energy_col,
    time_col_met="time",
    time_col_energy=time_col_energy,
    window_hours=window_hours if freq_code == "H" else max(1, window_hours // 24),
    lag_hours=lag_hours if freq_code == "H" else lag_hours // 24,
    freq=freq_code,
)

if result.correlation.empty:
    st.error("No overlapping data after processing. Try different settings.")
    st.stop()

corr_fig = px.line(
    result.correlation,
    x="time",
    y="correlation",
    title="Sliding Window Correlation",
    labels={"time": "Time", "correlation": "Pearson r"},
)
corr_fig.update_layout(yaxis_range=[-1, 1])
st.plotly_chart(corr_fig, use_container_width=True)

aligned = result.aligned.rename(columns={"meteorology": met_col, "energy": energy_col})
scatter_fig = px.scatter(
    aligned,
    x=met_col,
    y=energy_col,
    trendline="ols",
    title="Aligned Values (after lag)",
    labels={met_col: met_col, energy_col: energy_col},
)
st.plotly_chart(scatter_fig, use_container_width=True)

with st.expander("Download correlation data"):
    st.download_button(
        label="Download CSV",
        data=result.correlation.to_csv(index=False).encode("utf-8"),
        file_name=f"sliding_correlation_{met_col}_{energy_col}.csv",
        mime="text/csv",
    )

st.caption(
    f"Computed using a {window_hours}-hour window and {lag_hours}-hour lag at {frequency.lower()} resolution."
)