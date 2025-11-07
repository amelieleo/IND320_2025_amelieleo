import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import streamlit as st


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
   
    mask = (df["pricearea"] == price_area) & (df["productiongroup"] == production_group)
    df_filtered = df.loc[mask].copy()
    if df_filtered.empty:
        raise ValueError("No rows for given price_area/production_group")

    # ensure datetime and ordering
    df_filtered["starttime"] = pd.to_datetime(df_filtered["starttime"])
    df_filtered.sort_values("starttime", inplace=True)
    df_filtered.set_index("starttime", inplace=True)

    # STL decomposition and plot
    stl = STL(df_filtered["quantitykwh"].astype(float), period=period_length, robust=robust, seasonal=seasonal_smoothing, trend=trend_smoothing)
    res = stl.fit()
    fig = res.plot()
    fig.suptitle(f"STL decomposition — {production_group} in {price_area}")
    plt.tight_layout()

    #Improve datetime ticks: locator + concise formatter + rotated labels
    import matplotlib.dates as mdates
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in fig.axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.22)  # give room for rotated labels
    return fig



def spectrogram_energy_production(
    df: pd.DataFrame, 
    price_area: str = "NO1",
    production_group: str = "hydro",
    NFFT: int = 24*7*4,
    Fs: int = 1,
    noverlap: int = 24*7*4 // 2  
) -> plt.Figure:
        mask = (df["pricearea"] == price_area) & (df["productiongroup"] == production_group)
        df_filtered = df.loc[mask].sort_values("starttime").copy()
        x = df_filtered['quantitykwh'].to_numpy()
        fig, ax = plt.subplots(figsize=(12, 6))
        Pxx, freqs, bins, im = ax.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Spectrogram of {production_group} production in {price_area}')
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.22)  # give room for rotated labels
        plt.colorbar(im, ax=ax).set_label('Intensity [dB]') 
        return fig, Pxx, freqs, bins