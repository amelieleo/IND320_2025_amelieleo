from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd


def normalize_price_area(area: object) -> str | None:
    if area is None or (isinstance(area, float) and pd.isna(area)):
        return None
    text = str(area).strip()
    if not text:
        return None
    return text.replace(" ", "").replace("-", "").upper()


def load_json(path: Path | str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    return gdf.to_crs(epsg=4326)


def display_map(
    geojson_data: gpd.GeoDataFrame,
    selected_price_area: str | None = None,
    clicked_points: Iterable[Sequence[float]] | None = None,
    value_map: Mapping[str, float] | None = None,
    value_caption: str | None = None,
) -> folium.Map:
    gdf = geojson_data.to_crs(epsg=4326).copy()
    area_field = next(
        (c for c in gdf.columns if str(c).lower().replace(" ", "_") == "price_area"),
        None,
    )
    if area_field is None:
        raise ValueError("GeoJSON is missing a 'price_area' column required for mapping.")

    gdf["_area_norm"] = gdf[area_field].map(normalize_price_area)
    gdf = gdf[~gdf["_area_norm"].isna()].copy()
    selected_area_norm = normalize_price_area(selected_price_area)

    value_lookup = {
        norm_key: val
        for key, val in (value_map or {}).items()
        if (norm_key := normalize_price_area(key)) is not None
    }
    valid_values = [val for val in value_lookup.values() if pd.notna(val)]

    colormap = None
    if valid_values:
        vmin, vmax = min(valid_values), max(valid_values)
        if vmin == vmax:
            vmax = vmin + 1e-9
        colormap = cm.LinearColormap(
            colors=["#e0ecf4", "#9ebcda", "#8856a7"],
            vmin=vmin,
            vmax=vmax,
        )
        colormap.caption = value_caption or "Mean value"
        gdf["_value"] = gdf["_area_norm"].map(value_lookup.get)
    else:
        gdf["_value"] = None

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lat = (miny + maxy) / 2.0
    center_lon = (minx + maxx) / 2.0
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4.6, tiles="CartoDB positron")

    def style_function(feature: dict) -> dict:
        props = feature.get("properties", {})
        price_area_norm = normalize_price_area(
            props.get("Price area")
            or props.get("price_area")
            or props.get("Price_area")
            or props.get("_area_norm")
        )
        style = {"color": "#1f2937", "weight": 1, "fillColor": "#2563eb", "fillOpacity": 0.80}
        value = value_lookup.get(price_area_norm)
        if colormap and pd.notna(value):
            style["fillColor"] = colormap(value)
            style["fillOpacity"] = 0.8
        if selected_area_norm and price_area_norm == selected_area_norm:
            style.update({"color": "#f97316", "weight": 3, "fillOpacity": max(style.get("fillOpacity", 0.8), 0.8)})
        return style

    tooltip_fields: list[str] = []
    tooltip_aliases: list[str] = []
    tooltip_field = "name" if "name" in gdf.columns else area_field
    if tooltip_field:
        tooltip_fields.append(tooltip_field)
        tooltip_aliases.append("Zone")
    if colormap:
        tooltip_fields.append("_value")
        tooltip_aliases.append(value_caption or "Mean value")

    folium.GeoJson(
        data=gdf.__geo_interface__,
        name="Norwegian Price Areas",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, localize=True, labels=True),
        highlight_function=lambda _: {"weight": 3, "color": "#f97316"},
    ).add_to(m)

    for point in clicked_points or []:
        if isinstance(point, Sequence) and len(point) == 2:
            folium.CircleMarker(
                location=[float(point[0]), float(point[1])],
                radius=6,
                color="#ef4444",
                fill=True,
                fill_color="#ef4444",
                fill_opacity=0.9,
            ).add_to(m)

    folium.LayerControl().add_to(m)
    if colormap:
        colormap.add_to(m)

    return m