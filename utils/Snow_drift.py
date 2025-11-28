from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_Qupot(hourly_wind_speeds, dt: int = 3600) -> float:
    total = sum((u ** 3.8) * dt for u in hourly_wind_speeds) / 233847
    return total


def sector_index(direction: float) -> int:
    return int(((direction + 11.25) % 360) // 22.5)


def compute_sector_transport(
    hourly_wind_speeds,
    hourly_wind_dirs,
    dt: int = 3600,
) -> list[float]:
    sectors = [0.0] * 16
    for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
        if pd.isna(u) or pd.isna(d):
            continue
        idx = sector_index(d)
        sectors[idx] += ((u ** 3.8) * dt) / 233847
    return sectors


def compute_snow_transport(T, F, theta, Swe, hourly_wind_speeds, dt: int = 3600):
    Qupot = compute_Qupot(hourly_wind_speeds, dt)
    Qspot = 0.5 * T * Swe
    Srwe = theta * Swe

    if Qupot > Qspot:
        Qinf = 0.5 * T * Srwe
        control = "Snowfall controlled"
    else:
        Qinf = Qupot
        control = "Wind controlled"

    Qt = Qinf * (1 - 0.14 ** (F / T))

    return {
        "Qupot (kg/m)": Qupot,
        "Qspot (kg/m)": Qspot,
        "Srwe (mm)": Srwe,
        "Qinf (kg/m)": Qinf,
        "Qt (kg/m)": Qt,
        "Control": control,
    }


def compute_yearly_results(df: pd.DataFrame, T: float, F: float, theta: float) -> pd.DataFrame:
    seasons = sorted(df["season"].unique())
    results_list: list[dict[str, float]] = []
    for s in seasons:
        season_start = pd.Timestamp(year=s, month=7, day=1)
        season_end = pd.Timestamp(year=s + 1, month=6, day=30, hour=23, minute=59, second=59)
        df_season = df[(df["time"] >= season_start) & (df["time"] <= season_end)]
        if df_season.empty:
            continue
        df_season = df_season.copy()
        df_season["Swe_hourly"] = df_season.apply(
            lambda row: row["precipitation (mm)"] if row["temperature_2m (°C)"] < 1 else 0,
            axis=1,
        )
        total_Swe = df_season["Swe_hourly"].sum()
        wind_speeds = df_season["wind_speed_10m (m/s)"].tolist()
        result = compute_snow_transport(T, F, theta, total_Swe, wind_speeds)
        result["season"] = f"{s}-{s + 1}"
        results_list.append(result)
    return pd.DataFrame(results_list)


def compute_average_sector(df: pd.DataFrame) -> np.ndarray | None:
    sectors_list: list[list[float]] = []
    for _, group in df.groupby("season"):
        group = group.copy()
        group["Swe_hourly"] = group.apply(
            lambda row: row["precipitation (mm)"] if row["temperature_2m (°C)"] < 1 else 0,
            axis=1,
        )
        ws = group["wind_speed_10m (m/s)"].tolist()
        wdir = group["wind_direction_10m (°)"].tolist()
        sectors = compute_sector_transport(ws, wdir)
        sectors_list.append(sectors)
    if not sectors_list:
        return None
    return np.mean(sectors_list, axis=0)


def plot_rose(avg_sector_values, overall_avg: float) -> plt.Figure:
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    num_sectors = 16
    angles = np.deg2rad(np.arange(0, 360, 360 / num_sectors))
    avg_sector_values_tonnes = np.array(avg_sector_values) / 1000.0

    ax.bar(
        angles,
        avg_sector_values_tonnes,
        width=np.deg2rad(360 / num_sectors),
        align="center",
        edgecolor="black",
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    directions = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    ax.set_xticks(angles)
    ax.set_xticklabels(directions)

    overall_tonnes = overall_avg / 1000.0 if np.isfinite(overall_avg) else 0.0
    ax.set_title(
        f"Average Directional Distribution of Snow Transport\n"
        f"Overall Average Qt: {overall_tonnes:,.1f} tonnes/m",
        va="bottom",
    )
    fig.tight_layout()
    return fig


def compute_fence_height(Qt: float, fence_type: str) -> float:
    Qt_tonnes = Qt / 1000.0
    if fence_type.lower() == "wyoming":
        factor = 8.5
    elif fence_type.lower() in ["slat-and-wire", "slat and wire"]:
        factor = 7.7
    elif fence_type.lower() == "solid":
        factor = 2.9
    else:
        raise ValueError("Unsupported fence type. Choose 'Wyoming', 'Slat-and-wire', or 'Solid'.")
    H = (Qt_tonnes / factor) ** (1 / 2.2)
    return H