from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils.load_data import load_weather_data
from utils.Snow_drift import (
    compute_average_sector,
    compute_fence_height,
    compute_yearly_results,
    plot_rose,
)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_map_selection() -> Optional[Dict[str, Any]]:
    points = st.session_state.get("clicked_points")
    if not points:
        return None
    latest = points[-1]
    if not isinstance(latest, (list, tuple)) or len(latest) != 2:
        return None
    lat = _to_float(latest[0])
    lon = _to_float(latest[1])
    if lat is None or lon is None:
        return None
    price_area = st.session_state.get("price_area")
    if isinstance(price_area, (list, tuple)):
        price_area = price_area[0] if price_area else None
    price_area = str(price_area).strip() if price_area else None
    return {"latitude": lat, "longitude": lon, "price_area": price_area}


@st.cache_data(show_spinner=False)
def load_seasonal_weather_data(
    latitude: float,
    longitude: float,
    price_area: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 2):
        df_year = load_weather_data(
            price_area=price_area,
            year=year,
            latitude=latitude,
            longitude=longitude,
        )
        if df_year is None or df_year.empty:
            continue
        df_year = df_year.reset_index().rename(columns={"date": "time"})
        if pd.api.types.is_datetime64tz_dtype(df_year["time"]):
            df_year["time"] = df_year["time"].dt.tz_localize(None)
        rename_map = {
            "precipitation": "precipitation (mm)",
            "temperature_2m": "temperature_2m (°C)",
            "wind_speed_10m": "wind_speed_10m (m/s)",
            "wind_direction_10m": "wind_direction_10m (°)",
        }
        df_year = df_year.rename(columns=rename_map)
        frames.append(df_year)
    if not frames:
        return pd.DataFrame()
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="time")
        .sort_values("time")
        .reset_index(drop=True)
    )
    start_ts = pd.Timestamp(year=start_year, month=7, day=1)
    end_ts = pd.Timestamp(year=end_year + 1, month=6, day=30, hour=23, minute=59, second=59)
    combined = combined[(combined["time"] >= start_ts) & (combined["time"] <= end_ts)]
    if combined.empty:
        return combined
    combined["season"] = combined["time"].apply(lambda dt: dt.year if dt.month >= 7 else dt.year - 1)
    required = [
        "precipitation (mm)",
        "temperature_2m (°C)",
        "wind_speed_10m (m/s)",
        "wind_direction_10m (°)",
    ]
    combined[required] = combined[required].apply(pd.to_numeric, errors="coerce")
    combined = combined.dropna(subset=["wind_speed_10m (m/s)", "wind_direction_10m (°)"])
    return combined.reset_index(drop=True)


def main() -> None:
    st.title("Snow Drift and Wind Rose")
    st.caption("Seasons span 1 July to 30 June using ERA5 archives via Open-Meteo.")

    selection = get_map_selection()
    if selection is None:
        st.info("Select a location on the map page first.")
        st.stop()

    latitude = selection["latitude"]
    longitude = selection["longitude"]
    price_area = selection.get("price_area") or "NO1"

    st.markdown(
        f"**Selected location:** lat {latitude:.4f}, lon {longitude:.4f}"
        + (f" • price area {price_area}" if selection.get("price_area") else "")
    )
    if not selection.get("price_area"):
        st.info("No price area stored; defaulting to NO1 for weather retrieval.")

    min_year = 1979
    current_year = pd.Timestamp.now().year
    max_season_year = max(min_year + 1, current_year - 1)
    default_start = max(min_year, max_season_year - 4)
    default_end = min(max_season_year, default_start + 2)

    start_year, end_year = st.slider(
        "Season range (July–June)",
        min_value=min_year,
        max_value=max_season_year,
        value=(default_start, default_end),
        help="A season covers 1 July of the selected year to 30 June of the next year.",
    )

    with st.spinner("Loading weather data and computing snow drift…"):
        weather_df = load_seasonal_weather_data(
            latitude=latitude,
            longitude=longitude,
            price_area=price_area,
            start_year=start_year,
            end_year=end_year,
        )

    if weather_df.empty:
        st.error("No weather data available for this location and range.")
        st.stop()

    expected_seasons = list(range(start_year, end_year + 1))
    available_seasons = sorted(weather_df["season"].unique())
    missing = [season for season in expected_seasons if season not in available_seasons]
    if missing:
        st.warning("Missing data for: " + ", ".join(f"{year}-{year + 1}" for year in missing))

    filtered_df = weather_df[weather_df["season"].isin(expected_seasons)]
    if filtered_df.empty:
        st.error("No overlapping data for the chosen seasons.")
        st.stop()

    T = 3000
    F = 30000
    theta = 0.5

    yearly_df = compute_yearly_results(filtered_df, T, F, theta)
    if yearly_df.empty:
        st.warning("Snow drift calculation failed for the selected seasons.")
        st.stop()

    yearly_df = yearly_df.assign(season_year=yearly_df["season"].str.split("-").str[0].astype(int))
    yearly_df = yearly_df.sort_values("season_year").drop(columns="season_year")

    overall_avg = yearly_df["Qt (kg/m)"].mean()
    summary_df = yearly_df.copy()
    summary_df["Qt (tonnes/m)"] = (summary_df["Qt (kg/m)"] / 1000).round(1)
    summary_df = summary_df[["season", "Qt (tonnes/m)", "Control"]]

    st.subheader("Seasonal snow drift")
    st.dataframe(summary_df.rename(columns={"season": "Season"}), use_container_width=True)
    if np.isfinite(overall_avg):
        st.metric("Overall average Qt (tonnes/m)", f"{overall_avg / 1000:.1f}")

    st.subheader("Wind rose")
    avg_sectors = compute_average_sector(filtered_df)
    if avg_sectors is None:
        st.info("Wind-direction data unavailable for these seasons.")
    else:
        fig = plot_rose(avg_sectors, overall_avg if np.isfinite(overall_avg) else 0.0)
        st.pyplot(fig, use_container_width=True)

    fence_types = ["Wyoming", "Slat-and-wire", "Solid"]
    fence_rows = []
    for _, row in yearly_df.iterrows():
        entry = {"Season": row["season"]}
        for fence in fence_types:
            entry[fence] = round(compute_fence_height(row["Qt (kg/m)"], fence), 1)
        fence_rows.append(entry)

    if fence_rows:
        fence_df = pd.DataFrame(fence_rows)
        st.subheader("Required fence heights (m)")
        st.dataframe(fence_df, use_container_width=True)


if __name__ == "__main__":
    main()