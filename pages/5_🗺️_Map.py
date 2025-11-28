from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from utils.map_utils import display_map, load_json
from utils.load_data import (
    load_energy_consumption_data,
    load_energy_production_data,
)

st.set_page_config(
    page_title="Price Area Map",
    page_icon="🗺️",
    layout="wide",
)
st.title("🗺️ Price Area Map")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "norway_price_areas.geojson"

if not DATA_PATH.exists():
    st.error(
        "GeoJSON not found. Download it from NVE (ElSpot_omraade) and place it at data/norway_price_areas.geojson."
    )
    st.stop()

try:
    areas = load_json(DATA_PATH)
    st.success("GeoJSON loaded successfully.")
except Exception as exc:
    st.error(f"Failed to load GeoJSON: {exc}")
    st.stop()

price_area_col = next(
    (c for c in areas.columns if str(c).lower().replace(" ", "_") == "price_area"), None
)
price_area_options = (
    sorted({str(val) for val in areas[price_area_col].dropna()})
    if price_area_col
    else []
)

st.session_state.setdefault("selected_price_area", price_area_options[0] if price_area_options else None)
st.session_state.setdefault("clicked_points", [])

DATASET_CONFIG: dict[
    Literal["Energy Production", "Energy Consumption"], dict[str, object]
] = {
    "Energy Production": {
        "state_key": "production_data",
        "loaded_years_key": "loaded_production_years",
        "loader": load_energy_production_data,
        "group_column": "productiongroup",
        "value_column": "quantitykwh",
    },
    "Energy Consumption": {
        "state_key": "consumption_data",
        "loaded_years_key": "loaded_consumption_years",
        "loader": load_energy_consumption_data,
        "group_column": "consumptiongroup",
        "value_column": "quantitykwh",
    },
}
for cfg in DATASET_CONFIG.values():
    st.session_state.setdefault(cfg["state_key"], pd.DataFrame())
    st.session_state.setdefault(cfg["loaded_years_key"], set())

value_map: dict[str, float] = {}
value_caption: str | None = None

controls_col, map_col = st.columns([1.8, 4])

with controls_col:
    st.subheader("Controls")

    if price_area_options:
        st.selectbox(
            "Highlight price area",
            price_area_options,
            key="selected_price_area",
        )

    dataset_label: Literal["Energy Production", "Energy Consumption"] = st.radio(
        "Dataset",
        list(DATASET_CONFIG.keys()),
        horizontal=True,
    )
    config = DATASET_CONFIG[dataset_label]

    year_options = [2021, 2022, 2023, 2024]
    selected_years = st.multiselect(
        "Years",
        options=year_options,
        default=[year_options[-1]],
    )
    if not selected_years:
        st.warning("Select at least one year to continue.")
        selected_years = [year_options[-1]]

    loader = config["loader"]
    state_key = config["state_key"]
    loaded_years_key = config["loaded_years_key"]

    for year in selected_years:
        if year not in st.session_state[loaded_years_key]:
            df_new = loader(year)
            df_new["starttime"] = pd.to_datetime(df_new["starttime"], errors="coerce", utc=True)
            current = st.session_state[state_key]
            st.session_state[state_key] = df_new if current.empty else pd.concat([current, df_new], ignore_index=True)
            st.session_state[loaded_years_key].add(year)

    data = st.session_state[state_key]
    data = data[data["starttime"].dt.year.isin(selected_years)].dropna(subset=["starttime"])

    if data.empty:
        st.info("No data available for the selected combination.")
    else:
        group_col = config["group_column"]
        value_col = config["value_column"]

        groups = sorted(data[group_col].dropna().astype(str).unique())
        if not groups:
            st.info("No groups available for the selected dataset.")
        else:
            selected_group = st.selectbox("Group", groups)
            group_data = data[data[group_col].astype(str) == selected_group]

            if group_data.empty:
                st.info("No rows match the selected group.")
            else:
                time_index = pd.DatetimeIndex(group_data["starttime"])
                min_date = time_index.min().tz_convert("UTC").date()
                max_date = time_index.max().tz_convert("UTC").date()

                start_date = st.date_input(
                    "Interval start date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                )

                max_days = max(1, min(90, (max_date - min_date).days + 1))
                interval_days = st.slider(
                    "Interval length (days)",
                    min_value=1,
                    max_value=max_days,
                    value=min(7, max_days),
                )

                start_ts = pd.Timestamp(start_date, tz="UTC")
                end_ts = start_ts + pd.Timedelta(days=interval_days)

                window = group_data[
                    (group_data["starttime"] >= start_ts)
                    & (group_data["starttime"] < end_ts)
                ]

                if window.empty:
                    st.warning("No records within the selected interval.")
                else:
                    summary = (
                        window.groupby("pricearea", as_index=False)[value_col]
                        .mean()
                        .rename(columns={value_col: "mean_quantitykwh"})
                        .sort_values("mean_quantitykwh", ascending=False)
                    )

                    st.dataframe(
                        summary.assign(mean_quantitykwh=lambda df: df["mean_quantitykwh"].round(2)),
                        use_container_width=True,
                        hide_index=True,
                    )

                    value_map = {
                        str(row["pricearea"]).upper(): float(row["mean_quantitykwh"])
                        for _, row in summary.iterrows()
                        if pd.notna(row["pricearea"])
                    }
                    value_caption = f"Mean {dataset_label.lower()} ({selected_group}) in kWh"

    if st.button("Clear markers"):
        st.session_state["clicked_points"] = []
        st.rerun()

with map_col:
    map_object = display_map(
        areas,
        selected_price_area=st.session_state.get("selected_price_area"),
        clicked_points=st.session_state["clicked_points"],
        value_map=value_map,
        value_caption=value_caption,
    )
    map_event = st_folium(map_object, use_container_width=True, key="price_area_map")

if map_event:
    if map_event.get("last_clicked"):
        click = map_event["last_clicked"]
        coords = [click["lat"], click["lng"]]
        if not st.session_state["clicked_points"] or st.session_state["clicked_points"][-1] != coords:
            st.session_state["clicked_points"].append(coords)
        st.rerun()

    obj = map_event.get("last_object_clicked")
    if obj and obj.get("properties"):
        area = (
            obj["properties"].get("Price area")
            or obj["properties"].get("price_area")
            or obj["properties"].get("Price_area")
        )
        if area and area in price_area_options and area != st.session_state.get("selected_price_area"):
            st.session_state["selected_price_area"] = area
            st.rerun()

st.caption(f"Clicked coordinates: {st.session_state['clicked_points'] or 'None'}")