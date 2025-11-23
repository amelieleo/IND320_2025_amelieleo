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


# ...existing code...
if "clicked_point" not in st.session_state:
    st.session_state["clicked_point"] = None

price_area_col = next((c for c in areas.columns if c.lower().replace(" ", "_") == "price_area"), None)
price_area_options = sorted({str(val) for val in areas[price_area_col].dropna()}) if price_area_col else []

if price_area_options:
    if "selected_price_area" not in st.session_state or st.session_state["selected_price_area"] not in price_area_options:
        st.session_state["selected_price_area"] = price_area_options[0]
    selected_area = st.selectbox("Select price area", price_area_options, key="selected_price_area")
else:
    st.session_state["selected_price_area"] = None
    selected_area = None

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Clear marker"):
        st.session_state["clicked_point"] = None
        st.rerun()

m = display_map(
    areas,
    selected_price_area=selected_area,
    clicked_points=[st.session_state["clicked_point"]] if st.session_state["clicked_point"] else None,
)

map_event = st_folium(m, use_container_width=True, key="price_area_map")
if map_event:
    if map_event.get("last_clicked"):
        click = map_event["last_clicked"]
        st.session_state["clicked_point"] = [click["lat"], click["lng"]]
        st.rerun()
    obj = map_event.get("last_object_clicked")
    if obj and obj.get("properties"):
        area = obj["properties"].get("Price area") or obj["properties"].get("price_area") or obj["properties"].get("Price_area")
        if area and area in price_area_options and area != st.session_state["selected_price_area"]:
            st.session_state["selected_price_area"] = area
            st.rerun()

st.caption(f"Current click: {st.session_state['clicked_point']}")
# ...existing code...