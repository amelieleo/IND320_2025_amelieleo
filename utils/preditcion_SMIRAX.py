import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ValueWarning)


def _ensure_datetime_index(data: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = data.copy()
    frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = frame.dropna(subset=[column]).sort_values(column).set_index(column)
    frame.index = frame.index.tz_localize(None)
    return frame


def _prepare_training_slice(
    data: pd.DataFrame,
    target: str,
    exog: Optional[Sequence[str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    cols = [target] + list(exog or [])
    subset = data.loc[start:end, cols].copy()
    subset[target] = pd.to_numeric(subset[target], errors="coerce")
    subset = subset.dropna(subset=[target])
    if subset.empty:
        raise ValueError("No valid observations inside the selected training window.")
    if exog:
        for col in exog:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")
        subset[list(exog)] = subset[list(exog)].ffill().bfill()
    return subset[target], subset[list(exog)] if exog else None


def _infer_offset(index: pd.DatetimeIndex) -> pd.DateOffset:
    freq = pd.infer_freq(index)
    if freq:
        try:
            return to_offset(freq)
        except ValueError:
            pass
    if len(index) >= 3:
        diffs = index.to_series().diff().dropna()
        if not diffs.empty:
            try:
                return to_offset(diffs.mode().iloc[0])
            except (ValueError, IndexError):
                try:
                    return to_offset(diffs.median())
                except ValueError:
                    pass
    return to_offset("D")


def _repeat_last_values(frame: pd.DataFrame, steps: int, new_index: pd.DatetimeIndex) -> pd.DataFrame:
    repeated = pd.concat([frame.iloc[[-1]].copy()] * steps, ignore_index=True)
    repeated.index = new_index
    return repeated


def _align_to_index(index: pd.DatetimeIndex, anchor: pd.Timestamp) -> pd.Timestamp:
    if anchor in index:
        return anchor
    nearest = index.get_indexer([anchor], method="nearest")
    return index[nearest[0]] if nearest.size and nearest[0] >= 0 else index[-1]


def sarimax_forecast(
    data: pd.DataFrame,
    datetime_column: str,
    target_column: str,
    exog_columns: Optional[Sequence[str]],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    dynamic_start: pd.Timestamp,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    forecast_steps: int,
    future_exog_strategy: str = "repeat_last",
) -> Dict[str, Any]:
    if forecast_steps < 1:
        raise ValueError("Forecast steps must be at least 1.")

    indexed = _ensure_datetime_index(data, datetime_column)
    target_series, exog_frame = _prepare_training_slice(
        indexed, target_column, exog_columns, train_start, train_end
    )

    model = SARIMAX(
        target_series,
        exog=exog_frame,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)

    anchor_idx = _align_to_index(target_series.index, dynamic_start)
    dynamic_pred = results.get_prediction(start=anchor_idx, dynamic=anchor_idx, full_results=True)
    dynamic_mean = dynamic_pred.predicted_mean.loc[anchor_idx:]
    dynamic_ci = dynamic_pred.conf_int().loc[anchor_idx:]

    offset = _infer_offset(target_series.index)
    future_index = pd.date_range(target_series.index[-1] + offset, periods=forecast_steps, freq=offset)

    future_exog = None
    if exog_frame is not None and future_exog_strategy == "repeat_last":
        future_exog = _repeat_last_values(exog_frame, forecast_steps, future_index)

    forecast = results.get_forecast(steps=forecast_steps, exog=future_exog)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    if not isinstance(forecast_mean.index, pd.DatetimeIndex):
        forecast_mean.index = future_index
        forecast_ci.index = future_index

    return {
        "data": target_series,
        "dynamic_mean": dynamic_mean,
        "dynamic_ci": dynamic_ci,
        "forecast_mean": forecast_mean,
        "forecast_ci": forecast_ci,
        "future_exog": future_exog,
        "model_results": results,
    }

def detect_cols_long(df):
    time_col = "starttime"
    val_col = "quantitykwh"
    price_col = "pricearea"
    type_col = [c for c in df.columns if c.lower() in ["productiongroup", "consumptiongroup"]]
    prequired = [time_col, val_col, price_col] + type_col
    
    missing = [c for c in prequired if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")    
    return time_col, val_col, price_col, type_col