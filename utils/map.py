from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import folium
from folium.plugins import MousePosition

WANTED_CODES = {"NO1", "NO2", "NO3", "NO4", "NO5"}

def _detect_area_code(props: Dict[str, Any]) -> Optional[str]:
    if not props:
        return None
    for v in props.values():
        if isinstance(v, str):
            m = re.search(r"\bNO[1-5]\b", v.upper())
            if m:
                return m.group(0)
    return None

def extract_price_areas_from_geojson(gj: Dict[str, Any]) -> Dict[str, Any]:
    """Filter to NO1–NO5 and add AREA_CODE to properties."""
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

def load_price_areas(path: str | Path) -> Tuple[Dict[str, Any], List[str]]:
    """Load local GeoJSON file and return (filtered FeatureCollection, area_codes)."""
    path = Path(path)
    gj = json.loads(path.read_text(encoding="utf-8"))
    areas_geojson = extract_price_areas_from_geojson(gj)
    area_codes = [ft["properties"]["AREA_CODE"] for ft in areas_geojson["features"]]
    return areas_geojson, area_codes

def make_folium_map(
    areas_geojson: Dict[str, Any],
    selected_area: Optional[str] = None,
    clicked_coord: Optional[Dict[str, float]] = None,
    *,
    center_lat: float = 65.0,
    center_lon: float = 12.0,
    zoom_start: int = 4,
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """Folium map with outlines, hover highlight, persistent selection, and optional marker."""
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=tiles)

    def style_function(feature):
        code = (feature.get("properties") or {}).get("AREA_CODE")
        is_selected = (selected_area is not None) and (code == selected_area)
        return dict(color="red" if is_selected else "black",
                    weight=4 if is_selected else 2,
                    fillOpacity=0)

    def highlight_function(_):
        return dict(color="#666", weight=4, fillOpacity=0)

    gj_layer = folium.GeoJson(
        areas_geojson,
        name="Price areas",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(fields=["AREA_CODE"], aliases=["Area:"]),
    )
    gj_layer.add_to(m)

    try:
        m.fit_bounds(gj_layer.get_bounds())
    except Exception:
        pass

    if clicked_coord and "lat" in clicked_coord and "lon" in clicked_coord:
        folium.Marker(
            location=[clicked_coord["lat"], clicked_coord["lon"]],
            tooltip=f"{clicked_coord['lat']:.5f}, {clicked_coord['lon']:.5f}",
            icon=folium.Icon(color="red"),
        ).add_to(m)

    MousePosition(position="bottomright", separator=" | ", num_digits=5).add_to(m)
    folium.LayerControl().add_to(m)
    return m