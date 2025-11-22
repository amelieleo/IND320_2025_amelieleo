import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import streamlit as st
import numpy as np
from scipy.signal import spectrogram
import plotly.graph_objects as go


def LOESS_energy_production(
    df: pd.DataFrame,
    price_area: str = "NO1",
    production_group: str = "hydro",
    period_length: int = 24 * 7,
    seasonal_smoothing: int = 25,
    trend_smoothing: int = 24 * 7 * 4 + 1, 
    robust: bool = True,
) -> plt.Figure:
    #adding a loader while processing
   
    # mask = (df["pricearea"] == price_area) & (df["productiongroup"] == production_group)
    # df_filtered = df.loc[mask].copy()
    # if df_filtered.empty:
    #     raise ValueError("No rows for given price_area/production_group")

    # # ensure datetime and ordering
    # df_filtered["starttime"] = pd.to_datetime(df_filtered["starttime"])
    # df_filtered.sort_values("starttime", inplace=True)
    # df_filtered.set_index("starttime", inplace=True)

    # # STL decomposition and plot
    # stl = STL(df_filtered["quantitykwh"].astype(float), period=period_length, robust=robust, seasonal=seasonal_smoothing, trend=trend_smoothing)
    # res = stl.fit()
    # fig = res.plot()
    # fig.suptitle(f"STL decomposition — {production_group} in {price_area}")
    # plt.tight_layout()

    # #Improve datetime ticks: locator + concise formatter + rotated labels
    # import matplotlib.dates as mdates
    # locator = mdates.AutoDateLocator()
    # formatter = mdates.ConciseDateFormatter(locator)
    # for ax in fig.axes:
    #     ax.xaxis.set_major_locator(locator)
    #     ax.xaxis.set_major_formatter(formatter)
    #     plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # fig.tight_layout()
    # plt.subplots_adjust(bottom=0.22)  # give room for rotated labels
    # return fig
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    def _ensure_odd(n: int) -> int:
        return n if n % 2 == 1 else n + 1

    mask = (df["pricearea"] == price_area) & (df["productiongroup"] == production_group)
    df_filtered = df.loc[mask].copy()
    if df_filtered.empty:
        raise ValueError("No rows for given price_area/production_group")

    # ensure datetime and ordering
    df_filtered["starttime"] = pd.to_datetime(df_filtered["starttime"])
    df_filtered.sort_values("starttime", inplace=True)
    df_filtered.set_index("starttime", inplace=True)

    s = df_filtered["quantitykwh"].astype(float).dropna()
    if s.empty:
        raise ValueError("No valid quantitykwh data after filtering/dropping NA")

    stl = STL(
        s,
        period=int(period_length),
        seasonal=_ensure_odd(int(seasonal_smoothing)),
        trend=_ensure_odd(int(trend_smoothing)),
        robust=robust,
    )
    res = stl.fit()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.28, 0.24, 0.24, 0.24],
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
    )

    # Observed
    fig.add_trace(
        go.Scatter(x=s.index, y=res.observed, mode="lines", name="Observed",
                   line=dict(color="#1f77b4"),
                   hovertemplate="%{x}<br>Observed: %{y:.2f}<extra></extra>"),
        row=1, col=1
    )
    # Trend
    fig.add_trace(
        go.Scatter(x=s.index, y=res.trend, mode="lines", name="Trend",
                   line=dict(color="#ff7f0e"),
                   hovertemplate="%{x}<br>Trend: %{y:.2f}<extra></extra>"),
        row=2, col=1
    )
    # Seasonal
    fig.add_trace(
        go.Scatter(x=s.index, y=res.seasonal, mode="lines", name="Seasonal",
                   line=dict(color="#2ca02c"),
                   hovertemplate="%{x}<br>Seasonal: %{y:.2f}<extra></extra>"),
        row=3, col=1
    )
    # Residual
    fig.add_trace(
        go.Scatter(x=s.index, y=res.resid, mode="lines", name="Residual",
                   line=dict(color="#d62728"),
                   hovertemplate="%{x}<br>Residual: %{y:.2f}<extra></extra>"),
        row=4, col=1
    )

    fig.update_layout(
        title=f"STL decomposition — {production_group} in {price_area}",
        template="plotly_white",
        height=900,
        hovermode="x unified",
        margin=dict(t=80, r=20, l=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showspikes=True, spikedash="dot", spikemode="across", spikesnap="cursor")
    fig.update_yaxes(matches=None, zeroline=False)

    return fig



def spectrogram_energy_production(
    df: pd.DataFrame, 
    price_area: str = "NO1",
    production_group: str = "hydro",
    NFFT: int = 24*7*4,
    Fs: int = 1,
    noverlap: int = 24*7*4 // 2  
) -> go.Figure:
    mask = (df["pricearea"] == price_area) & (df["productiongroup"] == production_group)
    df_filtered = df.loc[mask].copy()
    if df_filtered.empty:
        raise ValueError("No rows for given price_area/production_group")

    df_filtered["starttime"] = pd.to_datetime(df_filtered["starttime"])
    df_filtered.sort_values("starttime", inplace=True)

    x = df_filtered["quantitykwh"].astype(float).to_numpy()
    freqs, bins, Sxx = spectrogram(x, fs=Fs, nperseg=NFFT, noverlap=noverlap, scaling="density", mode="psd")

    start_time = df_filtered["starttime"].iloc[0]
    time_axis = start_time + pd.to_timedelta(bins, unit="h")

    intensity_db = 10 * np.log10(Sxx + np.finfo(float).eps)

    import plotly.graph_objects as go
    fig = go.Figure(data=go.Heatmap(
        x=time_axis,
        y=freqs,
        z=intensity_db,
        colorscale="Viridis",
        colorbar=dict(title="Intensity [dB]")
    ))
    fig.update_layout(
        title=f"Spectrogram of {production_group} production in {price_area}",
        xaxis_title="Time",
        yaxis_title="Frequency (cycles/hour)",
        template="plotly_white"
    )

    return fig, Sxx, freqs, bins