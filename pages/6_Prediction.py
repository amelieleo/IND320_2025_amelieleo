import streamlit as st
import pandas as pd
import datetime as dt
from utils.load_data import load_energy_production_data, load_energy_consumption_data
from utils.preditcion_SMIRAX import sarimax_forecast, detect_cols_long
from plotly import graph_objects as go 


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

#price_area = st.session_state.get("price_area", "NO1")
st.session_state.setdefault("production_data", pd.DataFrame())
st.session_state.setdefault("consumption_data", pd.DataFrame())
st.session_state.setdefault("loaded_prod_years", set())
st.session_state.setdefault("loaded_cons_years", set())


options = ["production", "consumption"]
pred_data_option = st.radio(
    "Select Data to Predict",
    options=options,
    index=0 )

start_dt = dt.datetime(2021, 1, 1, 0, 0)
end_dt = dt.datetime(2024, 12, 31, 23, 0)


start_dt, end_dt = st.slider(
    "Select time range used for training the SARIMAX model",
    min_value=start_dt,
    max_value=end_dt,
    value=(start_dt, end_dt),
    step=dt.timedelta(days=1),  # change to days if you want coarser control
)

#make timezone oslo
start_ts = pd.Timestamp(start_dt, tz="Europe/Oslo") 
end_ts = pd.Timestamp(end_dt, tz="Europe/Oslo")

years = list(range(start_dt.year, end_dt.year + 1))

if pred_data_option == "production":
    st.write(f"You are predicting production data.")
    for year in years:
        if year not in st.session_state.loaded_prod_years:
            prod_df = load_energy_production_data(year)
            prod_df["starttime"] = pd.to_datetime(prod_df["starttime"], errors="coerce", utc=True)
            st.session_state.production_data = (
                prod_df if st.session_state.production_data.empty
                else pd.concat([st.session_state.production_data, prod_df], ignore_index=True)
            )
            st.session_state.loaded_prod_years.add(year)
    prod_df = st.session_state.production_data.copy()
    data = prod_df[(prod_df["starttime"] >= start_ts) & (prod_df["starttime"] <= end_ts)]
else:
    st.write(f"You are predicting consumption data.")
    for year in years:
        if year not in st.session_state.loaded_cons_years:
            cons_df = load_energy_consumption_data(year)
            cons_df["starttime"] = pd.to_datetime(cons_df["starttime"], errors="coerce", utc=True)
            cons_df = cons_df.dropna(subset=["starttime"])
            st.session_state.consumption_data = pd.concat(
                [st.session_state.consumption_data, cons_df], ignore_index=True
            )
            st.session_state.loaded_cons_years.add(year)
    cons_df = st.session_state.consumption_data.copy()
    data = cons_df[(cons_df["starttime"] >= start_dt) & (cons_df["starttime"] <= end_dt)]

time_col, val_col, price_col, type_col = detect_cols_long(data)

dfw = data.copy()
dfw[time_col] = pd.to_datetime(dfw[time_col], utc=True, errors="coerce")
dfw = dfw.dropna(subset=[time_col])
dfw = dfw.sort_values(time_col)

wide = dfw.pivot_table(index=time_col, columns=[price_col, type_col], values=val_col, aggfunc="sum")
wide = wide.sort_index()

# Make hourly regular and lightly fill tiny gaps
if wide.index.inferred_freq is None:
    wide = wide.resample("H").sum()
wide = wide.ffill(limit=6).bfill(limit=1)

areas = sorted({str(a) for a in dfw[price_col].unique()})
types = sorted({str(t) for t in dfw[type_col].unique()})

c_tgt1, c_tgt2 = st.columns(2)
sel_area = c_tgt1.selectbox("Target price area", options=areas, index=0)
sel_type = c_tgt2.selectbox("Target production type", options=types, index=types.index("wind") if "wind" in types else 0)

# Target series key and data
target_key = (sel_area, sel_type)
if target_key not in wide.columns:
    st.error(f"No series found for area={sel_area}, type={sel_type}.")
    st.stop()

y = wide[target_key].astype(float)

# Preset exog options
st.markdown("#### Exogenous selection")
same_area_other_types = st.checkbox("Include same area, other types", value=True)
same_type_other_areas = st.checkbox("Include same type, other areas", value=False)
all_other = st.checkbox("Include all other area-type combos", value=False)

# Build candidate exog list
exog_candidates = []
if same_area_other_types:
    exog_candidates += [(sel_area, t) for t in types if t != sel_type]
if same_type_other_areas:
    exog_candidates += [(a, sel_type) for a in areas if a != sel_area]
if all_other:
    exog_candidates += [(a, t) for a in areas for t in types if (a, t) != target_key]

# Remove duplicates and ensure existence
exog_candidates = [c for c in dict.fromkeys(exog_candidates) if c in wide.columns]

# Allow custom additions/removals
all_keys = [(a, t) for a in areas for t in types if (a, t) in wide.columns and (a, t) != target_key]
labels = [f"{a}|{t}" for (a, t) in all_keys]
preselect_labels = [f"{a}|{t}" for (a, t) in exog_candidates]
custom_labels = st.multiselect("Custom exog (add/remove)", options=labels, default=preselect_labels)
exog_keys = [tuple(lbl.split("|")) for lbl in custom_labels]

# Flatten to a modeling DataFrame with target + exogs (long-to-wide columns)
model_df = pd.DataFrame(index=wide.index)
model_df["starttime"] = model_df.index
model_df["kwh"] = y

# Create named exog columns
exog_cols = []
for (a, t) in exog_keys:
    col_name = f"exog_{a}_{t}"
    model_df[col_name] = wide[(a, t)].astype(float)
    exog_cols.append(col_name)

# Training window based on your slider (already defined: start_dt, end_dt)
train_mask = (model_df["starttime"] >= pd.Timestamp(start_dt, tz="UTC")) & (model_df["starttime"] <= pd.Timestamp(end_dt, tz="UTC"))
train_df = model_df.loc[train_mask].copy()

# Dynamic anchor selection (in-sample)
use_dynamic = st.checkbox("Use dynamic in-sample predictions", value=True)
dynamic_anchor = None
if use_dynamic:
    # Default: last 7 days of training switch to dynamic
    default_dyn = (pd.Timestamp(end_dt, tz="UTC") - pd.Timedelta(days=7)).to_pydatetime()
    dyn = st.slider(
        "Dynamic start",
        min_value=pd.Timestamp(start_dt, tz="UTC").to_pydatetime(),
        max_value=pd.Timestamp(end_dt, tz="UTC").to_pydatetime(),
        value=default_dyn,
        step=dt.timedelta(hours=1),
    )
    ts = pd.Timestamp(dyn)
    if ts.tz is None:
        dynamic_anchor = ts.tz_localize("Europe/Oslo", ambiguous="NaT", nonexistent="shift_forward")
    else:
        dynamic_anchor = ts.tz_convert("Europe/Oslo")

forecast_steps = st.number_input("Forecast horizon (steps)", min_value=1, max_value=1000, value=48)
# Ensure future exog are available for the chosen horizon
freq = pd.infer_freq(model_df.index) or "H"
future_index = pd.date_range(start=pd.Timestamp(end_dt, tz="Europe/Oslo") + pd.tseries.frequencies.to_offset(freq),
                             periods=int(forecast_steps), freq=freq)

if exog_cols and forecast_steps > 0:
    missing = [ts for ts in future_index if ts not in model_df.index]
    if missing:
        st.warning("Future exogenous values are not available past the loaded data range. "
                   "Increase the loaded window or reduce the forecast horizon.")

# Now call your existing SARIMAX function with the prepared long-lik

st.markdown("### SARIMAX parameters")
c1, c2, c3 = st.columns(3)
p = c1.number_input("AR order (p)", min_value=0, max_value=10, value=1)
d = c2.number_input("Differencing (d)", min_value=0, max_value=2, value=1)
q = c3.number_input("MA order (q)", min_value=0, max_value=10, value=1)

seasonal = st.checkbox("Seasonal parameters", value=False)
P = D = Q = m = 0
if seasonal:
    s1, s2, s3, s4 = st.columns(4)
    P = s1.number_input("Seasonal AR (P)", min_value=0, max_value=10, value=1)
    D = s2.number_input("Seasonal differencing (D)", min_value=0, max_value=2, value=0)
    Q = s3.number_input("Seasonal MA (Q)", min_value=0, max_value=10, value=1)
    m = s4.number_input("Seasonal period (m)", min_value=1, max_value=8760, value=24)

try:
    result = sarimax_forecast(
        data=model_df,
        datetime_column="starttime",
        target_column="kwh",
        exog_columns=exog_cols,
        train_start=start_ts,
        train_end=end_ts,
        dynamic_start=dynamic_anchor,
        order=(int(p), int(d), int(q)),
        seasonal_order=(int(P), int(D), int(Q), int(m)) if seasonal else (0, 0, 0, 0),
        forecast_steps=int(forecast_steps),
    )
except Exception as exc:
    st.error(f"Forecast failed: {exc}")
    st.stop()

series = result["data"]
dynamic_mean = result["dynamic_mean"]
dynamic_ci = result["dynamic_ci"]
forecast_mean = result["forecast_mean"]
forecast_ci = result["forecast_ci"]
model_results = result["model_results"]

def pick_ci_cols(ci_df):
    cols = list(ci_df.columns)
    lo = next((c for c in cols if "lower" in c.lower()), cols[0])
    hi = next((c for c in cols if "upper" in c.lower()), cols[-1] if cols[-1] != lo else cols[min(1, len(cols)-1)])
    return lo, hi

fig = go.Figure()

if series is not None and len(series) > 0:
    y_obs = series["kwh"] if isinstance(series, pd.DataFrame) and "kwh" in series.columns else series.squeeze()
    fig.add_trace(go.Scatter(
        x=series.index, y=y_obs,
        name="Observed", line=dict(color="gray"), opacity=0.6
    ))

if dynamic_mean is not None and len(dynamic_mean) > 0 and dynamic_ci is not None and len(dynamic_ci) > 0:
    lo, hi = pick_ci_cols(dynamic_ci)
    fig.add_trace(go.Scatter(
        x=dynamic_ci.index, y=dynamic_ci[hi],
        line=dict(color="royalblue", width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=dynamic_ci.index, y=dynamic_ci[lo],
        fill="tonexty", fillcolor="rgba(65,105,225,0.20)",
        line=dict(color="royalblue", width=0),
        name="Dynamic 95% CI"
    ))
fig.add_trace(go.Scatter(
x=dynamic_mean.index, y=dynamic_mean.squeeze(),
name="Dynamic mean", line=dict(color="royalblue")
))


if forecast_ci is not None and len(forecast_ci) > 0:
    lo, hi = pick_ci_cols(forecast_ci)
    fig.add_trace(go.Scatter(
        x=forecast_ci.index, y=forecast_ci[hi],
        line=dict(color="orangered", width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
    x=forecast_ci.index, y=forecast_ci[lo],
    fill="tonexty", fillcolor="rgba(255,69,0,0.20)",
    line=dict(color="orangered", width=0),
    name="Forecast 95% CI"
    ))
if forecast_mean is not None and len(forecast_mean) > 0:
    fig.add_trace(go.Scatter(
    x=forecast_mean.index, y=forecast_mean.squeeze(),
    name="Forecast mean", line=dict(color="orangered")
    ))

fig.update_layout(
title="SARIMAX forecast (Europe/Oslo)",
xaxis_title="Time (Europe/Oslo)",
yaxis_title="kWh",
template="plotly_white",
hovermode="x unified",
legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
)

st.plotly_chart(fig, use_container_width=True)