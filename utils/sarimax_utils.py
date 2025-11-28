from zoneinfo import ZoneInfo
import pandas as pd
from pandas.tseries.frequencies import to_offset
from plotly import graph_objs as go

OSLO = ZoneInfo("Europe/Oslo")
UTC = ZoneInfo("UTC")

def to_oslo(ts_like):
    ts = pd.Timestamp(ts_like)
    if ts.tz is None:
        return ts.tz_localize(OSLO, ambiguous="NaT", nonexistent="shift_forward")

    return ts.tz_convert(OSLO)

def to_utc_from_oslo(ts_like):
    return to_oslo(ts_like).tz_convert(UTC)

def detect_cols_long_safe(df: pd.DataFrame):
    def first_match(cands):
        lower = {c.lower(): c for c in df.columns}
        for key in cands:
            if key in lower:
                return lower[key]
        return None
    
    time_col = first_match(["starttime", "time", "datetime", "timestamp", "date"])
    val_col = first_match(["quantitykwh", "kwh", "value", "quantity"])
    price_col = first_match(["pricearea", "price_area", "area", "biddingzone"])
    type_candidates = [c for c in df.columns if c.lower() in {"productiongroup", "consumptiongroup"}]
    if not type_candidates:
        raise ValueError(f"Missing required type column (productiongroup/consumptiongroup). Found: {list(df.columns)}")
    # choose the one with more non-nulls if both exist
    type_col = max(type_candidates, key=lambda c: df[c].notna().sum())
    for name, col in [("time", time_col), ("value", val_col), ("price", price_col), ("type", type_col)]:
        if col is None:
            raise ValueError(f"Missing required {name} column. Found: {list(df.columns)}")
    return time_col, val_col, price_col, type_col

def make_hourly_wide(df: pd.DataFrame, time_col: str, val_col: str, price_col: str, type_col: str) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    wide = df.pivot_table(index=time_col, columns=[price_col, type_col], values=val_col, aggfunc="sum")
    wide = wide.sort_index()
    if wide.index.inferred_freq is None:
        wide = wide.resample("H").sum()
        # light fill small gaps
        wide = wide.ffill(limit=6).bfill(limit=1)
    return wide

def build_model_df(wide: pd.DataFrame, target_key: tuple, exog_keys: list[tuple]):
    if target_key not in wide.columns:
        raise KeyError(f"Target series {target_key} not found in wide columns.")
    y = wide[target_key].astype(float)
    model_df = pd.DataFrame(index=wide.index.copy())
    model_df.index.name = None
    model_df["starttime"] = wide.index
    model_df["kwh"] = y.values
    exog_cols = []
    for (a, t) in exog_keys:
        if (a, t) in wide.columns:
            col_name = f"exog_{a}_{t}"
            model_df[col_name] = wide[(a, t)].astype(float).values
            exog_cols.append(col_name)
    return model_df, exog_cols

def effective_steps(model_df: pd.DataFrame, end_ts: pd.Timestamp, steps_requested: int, freq: str | None = None) -> int:
    freq = freq or (pd.infer_freq(model_df.index) or "H")
    offset = to_offset(freq)
    last_data_ts = pd.DatetimeIndex(model_df["starttime"]).max()
    future_nominal = pd.date_range(
    start=end_ts + offset,
    periods=int(steps_requested),
    freq=freq,
    tz="UTC",
    )
    return int((future_nominal <= last_data_ts).sum())

def plot_forecast_plotly(series, dynamic_mean, dynamic_ci, forecast_mean, forecast_ci, title="SARIMAX forecast (Europe/Oslo)"):
    def to_oslo_index(obj):
        if obj is None:
            return None
        obj = obj.copy()
        if hasattr(obj, "index") and getattr(obj.index, "tz", None) is not None:
            obj.index = obj.index.tz_convert(OSLO)
        return obj
def pick_ci_cols(ci_df):
    cols = list(ci_df.columns)
    lo = next((c for c in cols if "lower" in c.lower()), cols[0])
    hi = next((c for c in cols if "upper" in c.lower()), cols[-1] if cols[-1] != lo else cols[min(1, len(cols)-1)])
    return lo, hi

def plot_forecast_plotly(series, dynamic_mean, dynamic_ci, forecast_mean, forecast_ci, title="SARIMAX forecast (Europe/Oslo)"):
    # assumes OSLO = ZoneInfo("Europe/Oslo") is defined at module level
    from plotly import graph_objs as go
    def to_oslo_index(obj):
        if obj is None:
            return None
        obj = obj.copy()
        if hasattr(obj, "index") and getattr(obj.index, "tz", None) is not None:
            obj.index = obj.index.tz_convert(OSLO)
        return obj

def pick_ci_cols(ci_df):
    cols = list(ci_df.columns)
    lo = next((c for c in cols if "lower" in c.lower()), cols[0])
    hi = next((c for c in cols if "upper" in c.lower()), cols[-1] if cols[-1] != lo else cols[min(1, len(cols)-1)])
    return lo, hi


def plot_forecast_plotly(series, dynamic_mean, dynamic_ci, forecast_mean, forecast_ci, title="SARIMAX forecast (Europe/Oslo)"):
    # assumes OSLO = ZoneInfo("Europe/Oslo") is defined at module level
    from plotly import graph_objs as go   

    def to_oslo_index(obj):
        if obj is None:
            return None
        obj = obj.copy()
        if hasattr(obj, "index") and getattr(obj.index, "tz", None) is not None:
            obj.index = obj.index.tz_convert(OSLO)
        return obj

    def pick_ci_cols(ci_df):
        cols = list(ci_df.columns)
        lo = next((c for c in cols if "lower" in c.lower()), cols[0])
        hi = next((c for c in cols if "upper" in c.lower()), cols[-1] if cols[-1] != lo else cols[min(1, len(cols)-1)])
        return lo, hi

    series_oslo = to_oslo_index(series)
    dynamic_mean_oslo = to_oslo_index(dynamic_mean)
    dynamic_ci_oslo = to_oslo_index(dynamic_ci)
    forecast_mean_oslo = to_oslo_index(forecast_mean)
    forecast_ci_oslo = to_oslo_index(forecast_ci)

    fig = go.Figure()
    if series_oslo is not None and len(series_oslo) > 0:
        y_obs = series_oslo["kwh"] if isinstance(series_oslo, pd.DataFrame) and "kwh" in series_oslo.columns else series_oslo.squeeze()
        fig.add_trace(go.Scatter(x=series_oslo.index, y=y_obs, name="Observed", line=dict(color="gray"), opacity=0.6))
    if dynamic_ci_oslo is not None and len(dynamic_ci_oslo) > 0:
        lo, hi = pick_ci_cols(dynamic_ci_oslo)
        fig.add_trace(go.Scatter(x=dynamic_ci_oslo.index, y=dynamic_ci_oslo[hi], line=dict(width=0, color="royalblue"), showlegend=False))
        fig.add_trace(go.Scatter(x=dynamic_ci_oslo.index, y=dynamic_ci_oslo[lo], fill="tonexty", fillcolor="rgba(65,105,225,0.20)", line=dict(width=0, color="royalblue"), name="Dynamic 95% CI"))
    if dynamic_mean_oslo is not None and len(dynamic_mean_oslo) > 0:
        fig.add_trace(go.Scatter(x=dynamic_mean_oslo.index, y=dynamic_mean_oslo.squeeze(), name="Dynamic mean", line=dict(color="royalblue")))
    if forecast_ci_oslo is not None and len(forecast_ci_oslo) > 0:
        lo, hi = pick_ci_cols(forecast_ci_oslo)
        fig.add_trace(go.Scatter(x=forecast_ci_oslo.index, y=forecast_ci_oslo[hi], line=dict(width=0, color="orangered"), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_ci_oslo.index, y=forecast_ci_oslo[lo], fill="tonexty", fillcolor="rgba(255,69,0,0.20)", line=dict(width=0, color="orangered"), name="Forecast 95% CI"))
    if forecast_mean_oslo is not None and len(forecast_mean_oslo) > 0:
        fig.add_trace(go.Scatter(x=forecast_mean_oslo.index, y=forecast_mean_oslo.squeeze(), name="Forecast mean", line=dict(color="orangered")))
        fig.update_layout(
        title=title,
        xaxis_title="Time (Europe/Oslo)",
        yaxis_title="kWh",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig