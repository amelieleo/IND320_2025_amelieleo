# streamlit_app.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
from streamlit_folium import st_folium

from utils.map_utils import display_map, load_json
st.set_page_config(
    page_title="Price Area Map",
    page_icon="🗺️",
    layout="wide"
)
st.title("🗺️ Price Area Map")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "norway_price_areas.geojson"

# Load local GeoJSON
if not DATA_PATH.exists():
    st.error("GeoJSON not found. Place the file at data/shape_price_area.geojson")
    st.stop()

try:
   areas = load_json(DATA_PATH)
   st.success("GeoJSON loaded successfully.")
except Exception as e:
    st.error(f"Failed to load GeoJSON: {e}")
    st.stop()

    
if "clicked_points" not in st.session_state:
    st.session_state["clicked_points"] = []

price_area_col = next((c for c in areas.columns if c.lower().replace(" ", "_") == "price_area"), None)
price_area_options = sorted({str(val) for val in areas[price_area_col].dropna()}) if price_area_col else []
selected_area = st.selectbox("Select price area", price_area_options, index=0 if price_area_options else None)

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Clear markers"):
        st.session_state["clicked_points"].clear()
        st.experimental_rerun()

m = display_map(
    areas,
    selected_price_area=selected_area,
    clicked_points=st.session_state["clicked_points"],
)

map_event = st_folium(m, use_container_width=True)
if map_event and map_event.get("last_clicked"):
    click = map_event["last_clicked"]
    st.session_state["clicked_points"].append([click["lat"], click["lng"]])
    st.experimental_rerun()

st.caption(f"Stored clicks: {len(st.session_state['clicked_points'])}")