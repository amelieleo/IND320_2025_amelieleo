from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SlidingCorrelationResult:
    correlation: pd.DataFrame
    aligned: pd.DataFrame


def _prepare_series(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    freq: str = "H",
) -> pd.Series:
    if time_col not in df or value_col not in df:
        raise KeyError(f"Columns '{time_col}' and '{value_col}' must exist in the dataframe.")

    series = (
        df[[time_col, value_col]]
        .dropna(subset=[time_col, value_col])
        .assign(**{time_col: lambda frame: pd.to_datetime(frame[time_col], utc=True, errors="coerce")})
        .dropna(subset=[time_col])
        .set_index(time_col)
        .sort_index()[value_col]
    )
    if series.empty:
        return series

    series = series.groupby(series.index).mean()
    series = (
        series.tz_localize(None)
        if getattr(series.index, "tz", None) is not None
        else series
    )
    series = series.resample(freq).mean().interpolate(limit=3)
    return series


def compute_sliding_correlation(
    met_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    met_col: str,
    energy_col: str,
    time_col_met: str,
    time_col_energy: str,
    window_hours: int,
    lag_hours: int,
    freq: Literal["H", "D"] = "H",
) -> SlidingCorrelationResult:
    if window_hours <= 0:
        raise ValueError("window_hours must be a positive integer.")

    met_series = _prepare_series(met_df, time_col_met, met_col, freq=freq)
    energy_series = _prepare_series(energy_df, time_col_energy, energy_col, freq=freq)

    if lag_hours != 0:
        energy_series = energy_series.shift(lag_hours, freq=freq)

    combined = pd.concat(
        {"meteorology": met_series, "energy": energy_series},
        axis=1,
        join="inner",
    ).dropna()

    if combined.empty:
        return SlidingCorrelationResult(
            correlation=pd.DataFrame(columns=["time", "correlation"]),
            aligned=combined,
        )

    window = f"{window_hours}{'H' if freq == 'H' else 'D'}"
    rolling_corr = (
        combined["meteorology"]
        .rolling(window=window, min_periods=max(3, window_hours // 4))
        .corr(combined["energy"])
        .dropna()
    )

    corr_df = (
        rolling_corr.to_frame("correlation")
        .reset_index()
        .rename(columns={rolling_corr.index.name or "index": "time"})
    )

    return SlidingCorrelationResult(correlation=corr_df, aligned=combined.reset_index())