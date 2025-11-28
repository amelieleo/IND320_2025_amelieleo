from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pandas.tseries.frequencies import to_offset
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

@st.cache_data(show_spinner=False)
def ensure_datetime_index(
    df: pd.DataFrame,
    timestamp_column: str,
    timezone: str = "UTC",
) -> pd.DataFrame:
    working = df.copy()
    working[timestamp_column] = pd.to_datetime(
        working[timestamp_column],
        errors="coerce",
        utc=True,
    )
    working = working.dropna(subset=[timestamp_column])
    working = working.sort_values(timestamp_column)
    working = working.set_index(timestamp_column)
    if timezone and timezone != "UTC":
        try:
            working.index = working.index.tz_convert(timezone)
        except Exception as exc:
            raise ValueError(f"Failed to convert timezone: {exc}") from exc
    working.index = working.index.tz_localize(None)
    working = working.sort_index()
    return working

@st.cache_data(show_spinner=False)
def list_numeric_columns(df: pd.DataFrame, exclude: Optional[Iterable[str]] = None) -> List[str]:
    excluded = set(exclude or [])
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


@st.cache_data(show_spinner=False)
def list_categorical_columns(df: pd.DataFrame, exclude: Optional[Iterable[str]] = None) -> List[str]:
    excluded = set(exclude or [])
    return [
        col
        for col in df.columns
        if col not in excluded
        and (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]))
    ]


@st.cache_data(show_spinner=False)
def filter_dimensions(df: pd.DataFrame, selections: Dict[str, List[str]]) -> pd.DataFrame:
    filtered = df.copy()
    for col, allowed_values in selections.items():
        if allowed_values and col in filtered.columns:
            allowed = {str(v) for v in allowed_values}
            filtered = filtered[filtered[col].astype(str).isin(allowed)]
    return filtered


def _infer_frequency(index: pd.DatetimeIndex):
    if len(index) < 2:
        return to_offset("H")
    inferred = pd.infer_freq(index)
    if inferred:
        return to_offset(inferred)
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return to_offset("H")
    mode_delta = diffs.mode().iat[0]
    return to_offset(mode_delta)


def _align_to_index(index: pd.DatetimeIndex, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    if ts in index:
        return ts
    loc = index.get_indexer([ts], method="nearest")
    if loc.size and loc[0] >= 0:
        return index[loc[0]]
    return None


def prepare_model_frames(
    df: pd.DataFrame,
    target_col: str,
    exog_cols: List[str],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    forecast_steps: int,
) -> Tuple[pd.Series, Optional[pd.DataFrame], pd.Series, pd.DatetimeIndex, pd.tseries.offsets.BaseOffset]:
    sorted_df = df.sort_index()
    y_full = sorted_df[target_col].astype(float)
    exog_full = sorted_df[exog_cols].astype(float) if exog_cols else None

    y_train = y_full.loc[train_start:train_end]
    if y_train.empty:
        raise ValueError("No target values found inside the selected training window.")

    exog_train = exog_full.loc[train_start:train_end] if exog_full is not None else None
    offset = _infer_frequency(y_train.index)

    forecast_index = pd.DatetimeIndex([])
    if forecast_steps > 0:
        start_next = y_train.index[-1] + offset
        forecast_index = pd.date_range(start=start_next, periods=forecast_steps, freq=offset)

    y_history = y_full.copy()
    return y_train, exog_train, y_history, forecast_index, offset


# ...existing code...
def _sanitize_exog(exog: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if exog is None:
        return None
    if exog.empty or exog.shape[1] == 0:
        return None
    cleaned = exog.astype(float)
    cleaned.columns = cleaned.columns.astype(str)
    cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()]
    return cleaned


def run_sarimax_forecast(
    y_train: pd.Series,
    exog_train: Optional[pd.DataFrame],
    exog_forecast: Optional[pd.DataFrame],
    forecast_steps: int,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    dynamic_start: Optional[pd.Timestamp] = None,
    alpha: float = 0.05,
) -> Dict[str, object]:
    P, D, Q, m = seasonal_order
    if m <= 0:
        seasonal_order = (P, D, Q, 1)

    y_train = y_train.astype(float)
    exog_train = _sanitize_exog(exog_train)
    exog_forecast = _sanitize_exog(exog_forecast)

    if forecast_steps > 0 and exog_train is not None:
        if exog_forecast is None:
            raise ValueError("Exogenous values for the forecast horizon are required.")
        if exog_forecast.shape[0] < forecast_steps:
            raise ValueError("Not enough exogenous rows supplied for the requested forecast steps.")
        exog_future = exog_forecast.iloc[:forecast_steps]
    else:
        exog_future = None

    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results: SARIMAXResultsWrapper = model.fit(disp=False)

    dynamic_label = None
    if dynamic_start is not None:
        aligned = _align_to_index(y_train.index, dynamic_start)
        if aligned is not None:
            dynamic_label = aligned

    in_sample_pred = results.get_prediction(
        start=y_train.index[0],
        end=y_train.index[-1],
        dynamic=dynamic_label,
        exog=exog_train,
    )
    dynamic_mean = in_sample_pred.predicted_mean
    dynamic_ci = in_sample_pred.conf_int(alpha=alpha)

    forecast_mean = None
    forecast_ci = None
    if forecast_steps > 0:
        future_pred = results.get_forecast(steps=forecast_steps, exog=exog_future)
        forecast_mean = future_pred.predicted_mean
        forecast_ci = future_pred.conf_int(alpha=alpha)
# ...existing code...


def build_forecast_plot(
    actual_series: pd.Series,
    train_end: pd.Timestamp,
    dynamic_mean: pd.Series,
    dynamic_ci: pd.DataFrame,
    forecast_mean: Optional[pd.Series],
    forecast_ci: Optional[pd.DataFrame],
    confidence_level: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual_series.index,
            y=actual_series.values,
            name="Actual",
            mode="lines",
            line=dict(color="#1f77b4"),
        )
    )

    if dynamic_mean is not None and not dynamic_mean.empty:
        fig.add_trace(
            go.Scatter(
                x=dynamic_mean.index,
                y=dynamic_mean.values,
                name="Dynamic prediction",
                mode="lines",
                line=dict(color="#ff7f0e"),
            )
        )
        lower_dyn, upper_dyn = _extract_bounds(dynamic_ci)
        fig.add_trace(
            go.Scatter(
                x=dynamic_mean.index,
                y=upper_dyn,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dynamic_mean.index,
                y=lower_dyn,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255, 127, 14, 0.2)",
                name=f"Dynamic CI ({int(confidence_level * 100)}%)",
            )
        )

    if forecast_mean is not None and not forecast_mean.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_mean.index,
                y=forecast_mean.values,
                name="Forecast",
                mode="lines",
                line=dict(color="#2ca02c", dash="dash"),
            )
        )
        if forecast_ci is not None and not forecast_ci.empty:
            lower_fc, upper_fc = _extract_bounds(forecast_ci)
            fig.add_trace(
                go.Scatter(
                    x=forecast_mean.index,
                    y=upper_fc,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_mean.index,
                    y=lower_fc,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(44, 160, 44, 0.2)",
                    name=f"Forecast CI ({int(confidence_level * 100)}%)",
                )
            )

    fig.add_vline(
        x=train_end,
        line_width=1,
        line_dash="dot",
        line_color="#444",
        annotation_text="Train end",
        annotation_position="top right",
    )

    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, l=40, r=40, b=40),
        xaxis_title="Timestamp",
        yaxis_title=actual_series.name or "Value",
    )
    return fig