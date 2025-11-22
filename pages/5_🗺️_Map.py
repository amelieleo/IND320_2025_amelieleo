# streamlit_app.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
from streamlit_folium import st_folium

from utils.map_utils import load_price_areas, make_folium_map

st.set_page_config(
    page_title="Price Area Map",
    page_icon="🗺️",
    layout="wide"
)
st.title("🗺️ Price Area Map")

DATA_PATH = Path("data/shape_price_areas.geojson")

# Load local GeoJSON
if not DATA_PATH.exists():
    st.error("GeoJSON not found. Place the file at ddata/shape_price_areas.geojson")
    st.stop()

try:
    areas_geojson, area_codes = load_price_areas(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load GeoJSON: {e}")
    st.stop()

# Session state
if "selected_area" not in st.session_state:
    st.session_state.selected_area = None
if "clicked_coord" not in st.session_state:
    st.session_state.clicked_coord = None

# Build map
m = make_folium_map(
    areas_geojson=areas_geojson,
    selected_area=st.session_state.selected_area,
    clicked_coord=st.session_state.clicked_coord,
)

col_map, col_side = st.columns([4, 1], vertical_alignment="top")

with col_map:
    map_state = st_folium(
        m,
        height=700,
        width=None,
        returned_objects=["last_clicked", "last_object_clicked"]
    )

with col_side:
    st.subheader("State")
    st.write("Selected area:", st.session_state.selected_area or "—")
    st.write("Clicked coord:", st.session_state.clicked_coord or "—")
    if st.button("Clear selection/marker"):
        st.session_state.selected_area = None
        st.session_state.clicked_coord = None
        st.experimental_rerun()

# Handle clicks
changed = False

# Click anywhere -> store coordinate
if map_state and map_state.get("last_clicked"):
    lat = map_state["last_clicked"].get("lat")
    lng = map_state["last_clicked"].get("lng")
    if lat is not None and lng is not None:
        st.session_state.clicked_coord = {"lat": float(lat), "lon": float(lng)}
        changed = True

# Click polygon -> store selected AREA_CODE
if map_state and map_state.get("last_object_clicked"):
    props = (map_state["last_object_clicked"].get("properties") or {})
    code = props.get("AREA_CODE")
    if code in (area_codes or []):
        st.session_state.selected_area = code
        changed = True

if changed:
    st.experimental_rerun()