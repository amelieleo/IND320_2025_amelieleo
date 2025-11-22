import streamlit as st
import json
import re
from pathlib import Path

WANTED_CODES = {"NO1", "NO2", "NO3", "NO4", "NO5"}

def _detect_area_code(props: dict) -> str | None:
    if not props:
        return None
    for v in props.values():
        if isinstance(v, str):
            m = re.search(r"\bNO[1-5]\b", v.upper())
            if m:
                return m.group(0)
    return None

def extract_price_areas_from_geojson(gj: dict) -> dict:
    """
    Filter a GeoJSON FeatureCollection to NO1–NO5 and normalize AREA_CODE in properties.
    Returns a new FeatureCollection.
    """
    feats = []
    for ft in gj.get("features", []):
        props = ft.get("properties", {}) or {}
        code = _detect_area_code(props)
        if code in WANTED_CODES:
            nft = dict(ft)
            nprops = dict(props)
            nprops["AREA_CODE"] = code
            nft["properties"] = nprops
            feats.append(nft)
    if not feats:
        raise ValueError("No price area features NO1–NO5 found in the GeoJSON.")
    return {"type": "FeatureCollection", "features": feats}

def area_codes_from_geojson(areas_geojson: dict) -> list[str]:
    return [ft["properties"]["AREA_CODE"] for ft in areas_geojson.get("features", [])]

def load_price_areas(path: str | Path) -> tuple[dict, list[str]]:
    """
    Load and extract price areas from a GeoJSON file.
    Returns (areas_geojson, area_codes).
    """
    path = Path(path)
    gj = json.loads(path.read_text(encoding="utf-8"))
    areas_geojson = extract_price_areas_from_geojson(gj)
    area_codes = area_codes_from_geojson(areas_geojson)
    return areas_geojson, area_codes

def make_price_areas_figure(
    areas_geojson: dict,
    area_codes: list[str],
    selected_area: str | None = None,
    clicked_coord: dict | None = None,
    *,
    center_lat: float = 65.0,
    center_lon: float = 12.0,
    zoom: float = 4,
    mapbox_style: str = "open-street-map",
    title: str = "NVE Elspot Price Areas (NO1–NO5) — Click a polygon to select and drop a marker",
) -> go.Figure:
    """
    Build a Plotly Mapbox figure with:
    - outlines for all price areas
    - optional selected area highlighted
    - optional clicked coordinate marker
    """
    base = go.Choroplethmapbox(
        geojson=areas_geojson,
        featureidkey="properties.AREA_CODE",
        locations=area_codes,
        z=[0] * len(area_codes),
        showscale=False,
        marker=dict(opacity=0, line=dict(color="black", width=2)),
        hovertemplate="Area: %{location}<extra></extra>",
        customdata=area_codes,
        name="Price areas",
    )

    fig = go.Figure([base])

    if selected_area and selected_area in area_codes:
        sel = go.Choroplethmapbox(
            geojson=areas_geojson,
            featureidkey="properties.AREA_CODE",
            locations=[selected_area],
            z=[1],
            showscale=False,
            marker=dict(opacity=0, line=dict(color="red", width=4)),
            hoverinfo="skip",
            customdata=[selected_area],
            name="Selected area",
        )
        fig.add_trace(sel)

    if clicked_coord and "lat" in clicked_coord and "lon" in clicked_coord:
        fig.add_trace(
            go.Scattermapbox(
                lat=[clicked_coord["lat"]],
                lon=[clicked_coord["lon"]],
                mode="markers",
                marker=dict(size=12, color="red"),
                name="Clicked point",
                hovertemplate="Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>",
            )
        )

    fig.update_layout(
        mapbox_style=mapbox_style,
        mapbox_zoom=zoom,
        mapbox_center=dict(lat=center_lat, lon=center_lon),
        margin=dict(l=0, r=0, t=35, b=0),
        title=title,
    )
    return fig