import datetime as dt
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from utils.load_data import load_energy_consumption_data, load_energy_production_data
from utils.sarimax_interface import (
    build_forecast_plot,
    ensure_datetime_index,
    filter_dimensions,
    list_categorical_columns,
    list_numeric_columns,
    prepare_model_frames,
    run_sarimax_forecast,
)

st.set_page_config(page_title="SARIMAX Forecasting", page_icon="📈", layout="wide")
st.title("📈 SARIMAX Forecasting for Energy Metrics")

TIMEZONE = "Europe/Oslo"
TARGET_COLUMN_KEY = "quantitykwh"
TIMESTAMP_COLUMN_KEY = "starttime"
DEFAULT_YEARS = list(range(2020, 2024))
PRICE_AREA_KEYS = ["pricearea", "price_area", "pricezone", "price_zone"]
PRODUCTION_GROUP_KEYS = ["production_group", "productiongroup", "technology", "fuel", "generator", "production"]
CONSUMPTION_GROUP_KEYS = ["consumption_group", "consumptiongroup", "sector", "customer", "category", "usage"]

st.session_state.setdefault("production_data", pd.DataFrame())
st.session_state.setdefault("consumption_data", pd.DataFrame())
st.session_state.setdefault("loaded_prod_years", set())
st.session_state.setdefault("loaded_cons_years", set())


def find_column_by_keywords(columns: List[str], keywords: List[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in columns}
    for keyword in keywords:
        for key, original in lowered.items():
            if keyword in key:
                return original
    return None


def resolve_column_case(columns: List[str], name: str) -> Optional[str]:
    name_lower = name.lower()
    for col in columns:
        if col.lower() == name_lower:
            return col
    return None


with st.sidebar:
    st.header("MongoDB settings")
    dataset_label = st.selectbox("Dataset", ["Energy Production", "Energy Consumption"], index=0)
    selected_years = st.multiselect(
        "Years",
        options=DEFAULT_YEARS,
        default=[2021] if 2021 in DEFAULT_YEARS else [DEFAULT_YEARS[-1]],
    )
    if st.button("Clear cached years"):
        st.session_state.production_data = pd.DataFrame()
        st.session_state.consumption_data = pd.DataFrame()
        st.session_state.loaded_prod_years = set()
        st.session_state.loaded_cons_years = set()
        st.experimental_rerun()

if not selected_years:
    st.info("Select at least one year to load data.")
    st.stop()

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

if raw_df.empty:
    st.error("No data loaded for the selected years.")
    st.stop()

price_col = find_column_by_keywords(list(raw_df.columns), PRICE_AREA_KEYS)
if price_col:
    price_options = sorted(raw_df[price_col].dropna().astype(str).unique())
    if price_options:
        selected_price = st.selectbox("Select price area", price_options)
        raw_df = raw_df[raw_df[price_col].astype(str) == selected_price]

group_keys = PRODUCTION_GROUP_KEYS if dataset_label == "Energy Production" else CONSUMPTION_GROUP_KEYS
group_col = find_column_by_keywords(list(raw_df.columns), group_keys)
if group_col is None:
    st.error("Required production/consumption group column not found.")
    st.stop()

group_options = sorted(raw_df[group_col].dropna().astype(str).unique())
if not group_options:
    st.error("No production/consumption group values available.")
    st.stop()

group_label = "Select production group" if dataset_label == "Energy Production" else "Select consumption group"
selected_group = st.selectbox(group_label, group_options)
raw_df = raw_df[raw_df[group_col].astype(str) == selected_group]

if raw_df.empty:
    st.error("No data left after applying scope filters.")
    st.stop()

timestamp_col = resolve_column_case(list(raw_df.columns), TIMESTAMP_COLUMN_KEY)
if timestamp_col is None:
    st.error(f"Timestamp column '{TIMESTAMP_COLUMN_KEY}' not found.")
    st.stop()

target_col = resolve_column_case(list(raw_df.columns), TARGET_COLUMN_KEY)
if target_col is None:
    st.error(f"Target column '{TARGET_COLUMN_KEY}' not found.")
    st.stop()

raw_df.drop(columns=["_id"], inplace=True, errors="ignore")

try:
    indexed_df = ensure_datetime_index(raw_df, timestamp_col, TIMEZONE)
except Exception as exc:
    st.error(f"Failed to parse timestamps: {exc}")
    st.stop()

exclude_dims = [col for col in [price_col, group_col, target_col] if col and col in indexed_df.columns]
dimension_filters: Dict[str, List[str]] = {}
other_categoricals = list_categorical_columns(indexed_df, exclude=exclude_dims)
if other_categoricals:
    with st.expander("Optional dimension locks", expanded=False):
        for col in other_categoricals:
            options = ["All"] + sorted(indexed_df[col].dropna().astype(str).unique())
            choice = st.selectbox(f"{col}", options=options, index=0)
            if choice != "All":
                dimension_filters[col] = [choice]

filtered_df = filter_dimensions(indexed_df, dimension_filters) if dimension_filters else indexed_df
if filtered_df.empty:
    st.error("No data left after filtering. Adjust the filters.")
    st.stop()

model_df = filtered_df.select_dtypes(include="number")
if model_df.empty:
    st.error("No numeric columns available for modelling.")
    st.stop()

if model_df.index.has_duplicates:
    st.warning("Duplicate timestamps detected; aggregating numeric columns by hourly mean.")
    model_df = model_df.groupby(level=0).mean()

model_df = model_df.sort_index()

if target_col not in model_df.columns:
    st.error(f"Target column '{target_col}' is not numeric after processing.")
    st.stop()

numeric_cols = list_numeric_columns(model_df, exclude=[target_col])
exog_cols = st.multiselect("Exogenous regressors", options=numeric_cols, default=[])

min_ts = model_df.index.min()
max_ts = model_df.index.max()
if pd.isna(min_ts) or pd.isna(max_ts):
    st.error("Invalid timestamp range.")
    st.stop()

default_start = min_ts.to_pydatetime()
default_end = (
    (max_ts - pd.Timedelta(hours=24)).to_pydatetime()
    if (max_ts - min_ts) > pd.Timedelta(hours=24)
    else max_ts.to_pydatetime()
)

train_start_dt, train_end_dt = st.slider(
    "Training timeframe",
    min_value=min_ts.to_pydatetime(),
    max_value=max_ts.to_pydatetime(),
    value=(default_start, default_end),
    step=dt.timedelta(hours=1),
)

forecast_steps = st.number_input("Forecast horizon (steps)", min_value=0, max_value=1000, value=48, step=1)
confidence_level = st.slider("Confidence level", min_value=0.80, max_value=0.99, value=0.95, step=0.01)

use_dynamic = st.checkbox("Enable dynamic in-sample predictions", value=True)
dynamic_start = None
if use_dynamic:
    dyn_default = (
        train_end_dt - dt.timedelta(days=7)
        if (train_end_dt - train_start_dt) >= dt.timedelta(days=7)
        else train_start_dt
    )
    dynamic_start = st.slider(
        "Dynamic start",
        min_value=train_start_dt,
        max_value=train_end_dt,
        value=dyn_default,
        step=dt.timedelta(hours=1),
    )

order_col1, order_col2, order_col3 = st.columns(3)
p = order_col1.number_input("AR order (p)", min_value=0, max_value=10, value=1)
d = order_col2.number_input("Differencing (d)", min_value=0, max_value=2, value=1)
q = order_col3.number_input("MA order (q)", min_value=0, max_value=10, value=1)

seasonal_enabled = st.checkbox("Use seasonal components", value=False)
P = D = Q = 0
m = 24
if seasonal_enabled:
    seas_col1, seas_col2, seas_col3, seas_col4 = st.columns(4)
    P = seas_col1.number_input("Seasonal AR (P)", min_value=0, max_value=10, value=1)
    D = seas_col2.number_input("Seasonal differencing (D)", min_value=0, max_value=2, value=0)
    Q = seas_col3.number_input("Seasonal MA (Q)", min_value=0, max_value=10, value=1)
    m = seas_col4.number_input("Seasonal period (m)", min_value=1, max_value=8760, value=24)

train_start = pd.Timestamp(train_start_dt)
train_end = pd.Timestamp(train_end_dt)
dynamic_start_ts = pd.Timestamp(dynamic_start) if dynamic_start else None

try:
    y_train, exog_train, y_history, forecast_index, _ = prepare_model_frames(
        model_df,
        target_col=target_col,
        exog_cols=exog_cols,
        train_start=train_start,
        train_end=train_end,
        forecast_steps=int(forecast_steps),
    )
except Exception as exc:
    st.error(f"Failed to prepare model data: {exc}")
    st.stop()

exog_forecast = None
if exog_cols and not forecast_index.empty:
    exog_forecast = model_df[exog_cols].reindex(forecast_index)
    if exog_forecast.isnull().values.any():
        exog_forecast = exog_forecast.ffill().bfill()

seasonal_params = (int(P), int(D), int(Q), int(m) if seasonal_enabled else 0)

try:
    result = run_sarimax_forecast(
        y_train=y_train,
        exog_train=exog_train,
        exog_forecast=exog_forecast,
        forecast_steps=int(forecast_steps),
        order=(int(p), int(d), int(q)),
        seasonal_order=seasonal_params,
        dynamic_start=dynamic_start_ts,
        alpha=1.0 - confidence_level,
    )
except Exception as exc:
    st.error(f"Model fitting failed: {exc}")
    st.stop()

forecast_fig = build_forecast_plot(
    actual_series=y_history,
    train_end=train_end,
    dynamic_mean=result["dynamic_mean"],
    dynamic_ci=result["dynamic_ci"],
    forecast_mean=result["forecast_mean"],
    forecast_ci=result["forecast_ci"],
    confidence_level=confidence_level,
)
st.plotly_chart(forecast_fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Training RMSE", f"{result['rmse']:.2f}")
col2.metric("AIC", f"{result['aic']:.2f}")
col3.metric("BIC", f"{result['bic']:.2f}")

with st.expander("Model diagnostics"):
    st.markdown("**SARIMAX summary**")
    st.text(result["model"].summary())
    st.markdown("**Residual sample**")
    st.dataframe(result["residuals"].to_frame(name="residuals").tail(20))