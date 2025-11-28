from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from shapely.geometry import Point

from utils.map_utils import display_map, load_json, normalize_price_area
from utils.load_data import (
    load_energy_consumption_data,
    load_energy_production_data,
)
st.session_state.setdefault("price_area", "NO1")

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
price_area_options = ["NO1", "NO2", "NO3", "NO4", "NO5"]

areas["_area_norm"] = areas[price_area_col].map(normalize_price_area) if price_area_col else None
area_centroids = {
    norm: (geom.representative_point().y, geom.representative_point().x)
    for norm, geom in zip(areas["_area_norm"], areas.geometry)
    if norm
}
normalized_to_original = {
    normalize_price_area(opt): opt
    for opt in price_area_options
    if normalize_price_area(opt)
}

default_area = st.session_state.get("price_area")
st.session_state.setdefault("clicked_points", [])

def find_price_area_for_point(lat: float, lon: float) -> str | None:
    if price_area_col is None:
        return None
    point = Point(lon, lat)
    matches = areas[areas.geometry.contains(point)]
    if matches.empty:
        return None
    return str(matches.iloc[0][price_area_col])

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
        current_area = st.session_state.get("price_area", price_area_options[0])
        if current_area not in price_area_options:
            current_area = price_area_options[0]
            st.session_state["price_area"] = current_area
        selected_area = st.selectbox(
            "Highlight price area",
            price_area_options,
            index=price_area_options.index(current_area),
        )
        st.session_state["price_area"] = selected_area
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
    data = data[data["starttime"].dt.year.isin(selected_years)].dropna(subset=["starttime", "pricearea"])

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
                summary = (
                    group_data.groupby("pricearea", as_index=False)[value_col]
                    .mean()
                    .rename(columns={value_col: "mean_quantitykwh"})
                    .sort_values("mean_quantitykwh", ascending=False)
                )

                st.dataframe(
                    summary.assign(mean_quantitykwh=lambda df: df["mean_quantitykwh"].round(2)),
                    width="stretch",
                    hide_index=True,
                )

                value_map = {
                    str(row["pricearea"]): float(row["mean_quantitykwh"])
                    for _, row in summary.iterrows()
                    if pd.notna(row["pricearea"])
                }
                year_label = ", ".join(map(str, selected_years))
                value_caption = f"Mean {dataset_label.lower()} ({selected_group}, {year_label}) in kWh"

    if st.button("Clear markers"):
        st.session_state["clicked_points"] = []
        st.rerun()

with map_col:
    folium_map = display_map(
        areas,
        selected_price_area=st.session_state.get("price_area"),
        clicked_points=st.session_state["clicked_points"],
        value_map=value_map,
        value_caption=value_caption,
    )
    map_event = st_folium(
        folium_map,
        width="stretch",
        feature_group_to_add=None,
        key="price_area_map",
    )

if map_event:
    should_rerun = False

    last_clicked = map_event.get("last_clicked")
    if last_clicked:
        coords = [float(last_clicked["lat"]), float(last_clicked["lng"])]
        st.session_state["clicked_points"] = [coords]

        inferred = find_price_area_for_point(coords[0], coords[1])
        if inferred and inferred != st.session_state.get("price_area"):
            st.session_state["price_area"] = inferred

        should_rerun = True

    obj = map_event.get("last_object_clicked")
    if obj and obj.get("properties"):
        area = (
            obj["properties"].get("Price area")
            or obj["properties"].get("price_area")
            or obj["properties"].get("Price_area")
        )
        if area and area in price_area_options and area != st.session_state.get("price_area"):
            st.session_state["price_area"] = area
            should_rerun = True

    if should_rerun:
        st.rerun()

st.caption(f"Clicked coordinates: {st.session_state['clicked_points'] or 'None'}")