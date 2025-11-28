import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.sarimax_interface import (
    build_forecast_plot,
    ensure_datetime_index,
    filter_dimensions,
    list_categorical_columns,
    list_numeric_columns,
    load_dataset_from_bytes,
    load_dataset_from_path,
    prepare_model_frames,
    run_sarimax_forecast,
)

st.set_page_config(page_title="SARIMAX Forecasting", page_icon="📈", layout="wide")
st.title("📈 SARIMAX Forecasting for Energy Metrics")

BASE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET_MAP = {
    "Energy Production": BASE_DATA_DIR / "energy_production_sample.csv",
    "Energy Consumption": BASE_DATA_DIR / "energy_consumption_sample.csv",
}

with st.sidebar:
    st.header("Data source")
    dataset_key = st.selectbox("Preset dataset", options=list(DATASET_MAP.keys()))
    default_path = DATASET_MAP[dataset_key]
    data_path_input = st.text_input("Dataset file path", value=str(default_path))
    uploaded_file = st.file_uploader("Or upload a file", type=["csv", "parquet", "pq", "feather", "ft"])

raw_df = pd.DataFrame()
try:
    if uploaded_file is not None:
        raw_df = load_dataset_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
    else:
        raw_df = load_dataset_from_path(data_path_input)
except Exception as exc:
    st.error(f"Failed to load dataset: {exc}")

if raw_df.empty:
    st.info("Load a dataset to continue.")
    st.stop()

st.subheader("Feature selection")
timestamp_col = st.selectbox("Timestamp column", options=list(raw_df.columns), index=0 if "starttime" not in raw_df.columns else list(raw_df.columns).index("starttime"))
timezone = st.selectbox("Timestamp timezone", options=["UTC", "Europe/Oslo", "Europe/Berlin", "CET"], index=0)

try:
    indexed_df = ensure_datetime_index(raw_df, timestamp_col, timezone)
except Exception as exc:
    st.error(f"Failed to parse timestamps: {exc}")
    st.stop()

available_dimensions = list_categorical_columns(indexed_df, exclude=[])
dimension_filters = {}
if available_dimensions:
    with st.expander("Filter dimensions", expanded=False):
        for dim in available_dimensions:
            options = sorted(indexed_df[dim].dropna().astype(str).unique())
            selection = st.multiselect(f"{dim}", options=options, default=options)
            if selection and len(selection) < len(options):
                dimension_filters[dim] = selection

filtered_df = filter_dimensions(indexed_df, dimension_filters) if dimension_filters else indexed_df
if filtered_df.empty:
    st.error("No data left after filtering. Adjust the filters.")
    st.stop()

numeric_columns = list_numeric_columns(filtered_df)
if not numeric_columns:
    st.error("No numeric columns found. Add numeric features to proceed.")
    st.stop()

target_col = st.selectbox("Target column", options=numeric_columns, index=0)
exog_candidates = [col for col in numeric_columns if col != target_col]
selected_exog = st.multiselect("Exogenous regressors", options=exog_candidates, default=[])

min_ts = filtered_df.index.min()
max_ts = filtered_df.index.max()
if pd.isna(min_ts) or pd.isna(max_ts):
    st.error("Timestamp range is invalid.")
    st.stop()

default_start = min_ts.to_pydatetime()
default_end = (max_ts - pd.Timedelta(hours=24)).to_pydatetime() if max_ts - min_ts > pd.Timedelta(hours=24) else max_ts.to_pydatetime()

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
dynamic_anchor_dt = None
if use_dynamic:
    dyn_default = (train_end_dt - dt.timedelta(days=7)) if (train_end_dt - train_start_dt) >= dt.timedelta(days=7) else train_start_dt
    dynamic_anchor_dt = st.slider(
        "Dynamic start",
        min_value=train_start_dt,
        max_value=train_end_dt,
        value=dyn_default,
        step=dt.timedelta(hours=1),
    )

order_col1, order_col2, order_col3 = st.columns(3)
p = order_col1.number_input("AR order (p)", min_value=0, max_value=10, value=1, step=1)
d = order_col2.number_input("Differencing (d)", min_value=0, max_value=2, value=1, step=1)
q = order_col3.number_input("MA order (q)", min_value=0, max_value=10, value=1, step=1)

seasonal_enabled = st.checkbox("Use seasonal components", value=False)
P = D = Q = m = 0
if seasonal_enabled:
    seas_col1, seas_col2, seas_col3, seas_col4 = st.columns(4)
    P = seas_col1.number_input("Seasonal AR (P)", min_value=0, max_value=10, value=1, step=1)
    D = seas_col2.number_input("Seasonal differencing (D)", min_value=0, max_value=2, value=0, step=1)
    Q = seas_col3.number_input("Seasonal MA (Q)", min_value=0, max_value=10, value=1, step=1)
    m = seas_col4.number_input("Seasonal period (m)", min_value=1, max_value=8760, value=24, step=1)

train_start = pd.Timestamp(train_start_dt)
train_end = pd.Timestamp(train_end_dt)
dynamic_start = pd.Timestamp(dynamic_anchor_dt) if dynamic_anchor_dt else None

try:
    y_train, exog_train, y_history, _, forecast_index, freq_delta = prepare_model_frames(
        filtered_df,
        target_col=target_col,
        exog_cols=selected_exog,
        train_start=train_start,
        train_end=train_end,
        forecast_steps=int(forecast_steps),
    )
except Exception as exc:
    st.error(f"Failed to prepare model data: {exc}")
    st.stop()

try:
    result = run_sarimax_forecast(
        y_train=y_train,
        exog_train=exog_train,
        exog_forecast=filtered_df[selected_exog].reindex(forecast_index).ffill().bfill() if selected_exog and len(forecast_index) else None,
        forecast_steps=int(forecast_steps),
        order=(int(p), int(d), int(q)),
        seasonal_order=(int(P), int(D), int(Q), int(m)) if seasonal_enabled else (0, 0, 0, 0),
        dynamic_start=dynamic_start,
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

metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
metrics_col1.metric("Training RMSE", f"{result['rmse']:.2f}")
metrics_col2.metric("AIC", f"{result['aic']:.2f}")
metrics_col3.metric("BIC", f"{result['bic']:.2f}")

with st.expander("Model diagnostics"):
    st.markdown("**SARIMAX summary**")
    st.text(result["model"].summary())
    st.markdown("**Residual sample**")
    st.dataframe(result["residuals"].to_frame(name="residuals").tail(20))
