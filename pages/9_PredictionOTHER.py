import streamlit as st
import pandas as pd
import datetime as dt
from utils.load_data import load_energy_production_data, load_energy_consumption_data
from utils.prediction_SMIRAX import sarimax_forecast  # keep your existing function
from utils.sarimax_utils import (
OSLO, UTC,
to_utc_from_oslo,
detect_cols_long_safe,
make_hourly_wide,
build_model_df,
effective_steps,
plot_forecast_plotly,
get_data_for_years,
)

st.set_page_config(page_title="SARIMAX Forecasting", page_icon="📈", layout="wide")
st.title("📈 SARIMAX Forecasting for Energy Metrics")

#UI: dataset

pred_data_option = st.radio("Select Data to Predict", options=["production", "consumption"], index=0)
#UI: training window (Europe/Oslo)

start_local_default = dt.datetime(2021, 1, 1, 0, 0)
end_local_default = dt.datetime(2024, 12, 31, 23, 0)
start_local, end_local = st.slider("Training data window (Europe/Oslo)", min_value=start_local_default, max_value=end_local_default, value=(start_local_default, end_local_default), step=dt.timedelta(days=1),)
start_ts = to_utc_from_oslo(start_local)
end_ts = to_utc_from_oslo(end_local)

selected_years = list(range(start_local.year, end_local.year + 1))
dataset = "production" if pred_data_option == "production" else "consumption"

data = get_data_for_years(dataset, selected_years)
if data.empty:
    st.error(f"No {dataset} data loaded for years {selected_years}.")
    data["starttime"] = pd.to_datetime(data["starttime"], errors="coerce", utc=True)

data["starttime"] = pd.to_datetime(data["starttime"], errors="coerce", utc=True)
data = data.dropna(subset=["starttime"])

#UI: forecast steps
forecast_steps_ui = st.number_input("Forecast horizon (steps)", min_value=1, max_value=1000, value=48)
# Load data for all years between start and end (inclusive)

years = tuple(range(start_ts.year, end_ts.year + 1))
data = data[data["starttime"] >= start_ts].copy()
if data.empty:
    st.error("No data available in the selected window. Try widening the date range.")
    #st.stop()
#Detect columns and build wide hourly

time_col, val_col, price_col, type_col = detect_cols_long_safe(data)
wide = make_hourly_wide(data, time_col, val_col, price_col, type_col)
#Target selection

areas = sorted({str(a) for a in data[price_col].dropna().unique()})
types = sorted({str(t) for t in data[type_col].dropna().unique()})
c1, c2 = st.columns(2)
sel_area = c1.selectbox("Target price area", options=areas, index=0)
default_type_idx = types.index("wind") if "wind" in types else 0
sel_type = c2.selectbox("Target type", options=types, index=default_type_idx)

target_key = (sel_area, sel_type)
if target_key not in wide.columns:
    st.error(f"No series found for area={sel_area}, type={sel_type}.")
    #st.stop()
#Exogenous selection

st.markdown("Exogenous selection")
same_area_other_types = st.checkbox("Include same area, other types", value=True)
same_type_other_areas = st.checkbox("Include same type, other areas", value=False)
all_other = st.checkbox("Include all other area-type combos", value=False)

exog_candidates = []
if same_area_other_types:
    exog_candidates += [(sel_area, t) for t in types if t != sel_type]
if same_type_other_areas:
    exog_candidates += [(a, sel_type) for a in areas if a != sel_area]
if all_other:
    exog_candidates += [(a, t) for a in areas for t in types if (a, t) != target_key]
# dedupe and ensure availability

exog_candidates = [c for c in dict.fromkeys(exog_candidates) if c in wide.columns]
# Allow custom add/remove

all_keys = [(a, t) for a in areas for t in types if (a, t) in wide.columns and (a, t) != target_key]
labels = [f"{a}|{t}" for (a, t) in all_keys]
preselect_labels = [f"{a}|{t}" for (a, t) in exog_candidates]
custom_labels = st.multiselect("Custom exog (add/remove)", options=labels, default=preselect_labels)
exog_keys = [tuple(lbl.split("|")) for lbl in custom_labels]
# Build modeling frame

model_df, exog_cols = build_model_df(wide, target_key, exog_keys)
model_df["starttime"] = pd.to_datetime(model_df["starttime"], errors="coerce", utc=True)
model_df = model_df.dropna(subset=["starttime"])


def ensure_utc(ts): 
    if ts is None: return None 
    t = pd.Timestamp(ts) 
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")

train_start = ensure_utc(start_ts)
train_end = ensure_utc(end_ts)

# Dynamic forecasting start (UI in Oslo -> UTC)

use_dynamic = st.checkbox("Use dynamic in-sample predictions", value=True)
dynamic_anchor = None
if use_dynamic:
    default_dyn_local = end_local - dt.timedelta(days=7)
    dyn_local = st.slider(
    "Dynamic start (Europe/Oslo)",
    min_value=start_local,
    max_value=end_local,
    value=default_dyn_local,
    step=dt.timedelta(hours=1),
    )
    dynamic_anchor = to_utc_from_oslo(dyn_local)
# SARIMAX parameters
dynamic_start = ensure_utc(dynamic_anchor) if dynamic_anchor is not None else None

st.markdown("SARIMAX parameters")
c3, c4, c5 = st.columns(3)
p = c3.number_input("AR order (p)", min_value=0, max_value=10, value=1)
d = c4.number_input("Differencing (d)", min_value=0, max_value=2, value=1)
q = c5.number_input("MA order (q)", min_value=0, max_value=10, value=1)

seasonal = st.checkbox("Seasonal parameters", value=False)
P = D = Q = m = 0
if seasonal:
    s1, s2, s3, s4 = st.columns(4)
    P = s1.number_input("Seasonal AR (P)", min_value=0, max_value=10, value=1)
    D = s2.number_input("Seasonal differencing (D)", min_value=0, max_value=2, value=0)
    Q = s3.number_input("Seasonal MA (Q)", min_value=0, max_value=10, value=1)
    m = s4.number_input("Seasonal period (m)", min_value=1, max_value=8760, value=24)
# Determine effective forecast steps (clip to available data)

freq = pd.infer_freq(model_df.index) or "H"
steps_effective = effective_steps(model_df, end_ts, int(forecast_steps_ui), freq=freq)
if steps_effective < int(forecast_steps_ui):
    last_ts = pd.DatetimeIndex(model_df["starttime"]).max().tz_convert(OSLO)
    st.info(
        f"Clipping forecast horizon from {forecast_steps_ui} to {steps_effective} "
        f"because data ends at {last_ts:%Y-%m-%d %H:%M} (Europe/Oslo)."
    )
# Fit + forecast (use full model_df, limit training by start/end, and steps by steps_effective)

if train_start is not None and train_start.tzinfo is not None:
    train_start = train_start.tz_convert(None)
if train_end is not None and train_end.tzinfo is not None:
    train_end = train_end.tz_convert(None)
if dynamic_start is not None and dynamic_start.tzinfo is not None:
    dynamic_start = dynamic_start.tz_convert(None)

try:
    result = sarimax_forecast(
        data=model_df,
        datetime_column="starttime",
        target_column="kwh",
        exog_columns=exog_cols if exog_cols else None,
        train_start=train_start,
        train_end=train_end,
        dynamic_start=dynamic_start,
        order=(int(p), int(d), int(q)),
        seasonal_order=(int(P), int(D), int(Q), int(m)) if seasonal else (0, 0, 0, 0),
        forecast_steps=int(steps_effective),
        )
except Exception as exc:
    st.error(f"Forecast failed: {exc}")
    #st.stop()

try: 
    series = result.get("data")
    dynamic_mean = result.get("dynamic_mean")
    dynamic_ci = result.get("dynamic_ci")
    forecast_mean = result.get("forecast_mean")
    forecast_ci = result.get("forecast_ci")
    #Plot

    fig = plot_forecast_plotly(series, dynamic_mean, dynamic_ci, forecast_mean, forecast_ci, title="SARIMAX forecast (Europe/Oslo)")
    st.plotly_chart(fig, use_container_width=True)
except Exception as exc:
    st.error(f"Plotting failed: {exc}")