from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import geopandas as gpd
from streamlit_folium import st_folium
import folium

def load_json(filepath: Path):
    """Load GeoJSON file and extract area codes."""
    with open(filepath, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    return geojson_data

def display_map(geojson_data):
    # Ensure WGS84 for folium and pick a safe tooltip field
    gdf = gdf.to_crs(epsg=4326)
    tooltip_field = "name" if "name" in gdf.columns else next((c for c in gdf.columns if c != "geometry"), None)

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lat = (miny + maxy) / 2.0
    center_lon = (minx + maxx) / 2.0
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4.5)

    folium.GeoJson(
        data=gdf.__geo_interface__,
        name="Norwegian Bidding Zones",
        style_function=lambda x: {"fillColor": "blue", "color": "black", "weight": 1, "fillOpacity": 0.5},
        tooltip=folium.GeoJsonTooltip(fields=[tooltip_field] if tooltip_field else [], aliases=["Zone"])
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m