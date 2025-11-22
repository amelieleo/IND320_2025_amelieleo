# streamlit_app.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
from streamlit_folium import st_folium

from utils.map_utils import load_json, display_choropleth

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

fig = display_choropleth(areas)
st.plotly_chart(fig, use_container_width=True)

selected_area = st.session_state.get("price_area", None)