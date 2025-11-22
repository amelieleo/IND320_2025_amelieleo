from pathlib import Path
import streamlit as st
from streamlit_plotly_events import plotly_events

from utils import (
    extract_price_areas_from_geojson,
    load_price_areas,
    make_price_areas_figure,
)



st.set_page_config(
    page_title="Price Area Map",
    page_icon="🗺️",
    layout="wide"
)
st.title("🗺️ Price Area Map")

default_path = Path("data/shape_price_area.geojson")

# Load areas
try:
    if default_path.exists():
        areas_geojson, area_codes = load_price_areas(default_path)
    else:
        st.info("Upload the exported GeoJSON or place it at data/elspot_omraade.geojson")
        st.stop()
except Exception as e:
    st.error(f"Failed to load/parse GeoJSON: {e}")
    st.stop()

# Session state
if "selected_area" not in st.session_state:
    st.session_state.selected_area = None
if "clicked_coord" not in st.session_state:
    st.session_state.clicked_coord = None

# Build and show figure
fig = make_price_areas_figure(
    areas_geojson=areas_geojson,
    area_codes=area_codes,
    selected_area=st.session_state.selected_area,
    clicked_coord=st.session_state.clicked_coord,
)

col_map, col_side = st.columns([4, 1], vertical_alignment="top")

with col_map:
    events = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=700,
        override_width="100%",
        key="price-areas-plot",
    )

with col_side:
    st.subheader("State")
    st.write("Selected area:", st.session_state.selected_area or "—")
    st.write("Clicked coord:", st.session_state.clicked_coord or "—")
    if st.button("Clear selection/marker"):
        st.session_state.selected_area = None
        st.session_state.clicked_coord = None
        st.experimental_rerun()

# Handle clicks (on polygons/markers)
if events:
    pt = events[0]
    code = pt.get("customdata")
    if code in area_codes:
        st.session_state.selected_area = code
    lat = pt.get("lat")
    lon = pt.get("lon")
    if lat is not None and lon is not None:
        st.session_state.clicked_coord = {"lat": float(lat), "lon": float(lon)}
    st.experimental_rerun()