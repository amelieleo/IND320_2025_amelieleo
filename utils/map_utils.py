from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm


def load_json(path: Path | str) -> gpd.GeoDataFrame:
    """
    Load a GeoJSON file into a GeoDataFrame with a standard CRS.
    """
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
    """
    Render a folium map with Norwegian price area overlays, optional highlights,
    choropleth colouring, and persisted click markers.
    """
    gdf = geojson_data.to_crs(epsg=4326).copy()

    tooltip_field = "name" if "name" in gdf.columns else next(
        (c for c in gdf.columns if c != "geometry"),
        None,
    )
    selected_area_lc = selected_price_area.lower() if selected_price_area else None
    area_field = next(
        (c for c in gdf.columns if str(c).lower().replace(" ", "_") == "price_area"),
        None,
    )

    value_lookup = {str(k).upper(): v for k, v in (value_map or {}).items()}
    valid_values = [v for v in value_lookup.values() if pd.notna(v)]
    colormap: cm.LinearColormap | None = None

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
        if area_field:
            gdf["_value"] = gdf[area_field].map(lambda x: value_lookup.get(str(x).upper()))

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lat = (miny + maxy) / 2.0
    center_lon = (minx + maxx) / 2.0
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4.5)

    def style_function(feature: dict) -> dict:
        props = feature.get("properties", {})
        price_area_raw = str(
            props.get("Price area")
            or props.get("price_area")
            or props.get("Price_area")
            or ""
        )
        price_area_lc = price_area_raw.lower()
        price_area_uc = price_area_raw.upper()

        style = {
            "fillColor": "#2563eb",
            "color": "#1f2937",
            "weight": 1,
            "fillOpacity": 0.15,
        }

        value = value_lookup.get(price_area_uc)
        if colormap and pd.notna(value):
            style["fillColor"] = colormap(value)
            style["fillOpacity"] = 0.6

        if selected_area_lc and price_area_lc == selected_area_lc:
            style.update(
                {
                    "color": "#f97316",
                    "weight": 3,
                    "fillOpacity": max(style.get("fillOpacity", 0.6), 0.6),
                }
            )

        return style

    tooltip_fields: list[str] = []
    tooltip_aliases: list[str] = []

    if tooltip_field:
        tooltip_fields.append(tooltip_field)
        tooltip_aliases.append("Zone")

    if colormap and "_value" in gdf.columns:
        tooltip_fields.append("_value")
        tooltip_aliases.append(value_caption or "Mean value")

    folium.GeoJson(
        data=gdf.__geo_interface__,
        name="Norwegian Bidding Zones",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            labels=True,
        ),
    ).add_to(m)

    for point in clicked_points or []:
        if len(point) == 2:
            folium.CircleMarker(
                location=[point[0], point[1]],
                radius=4,
                color="#ef4444",
                fill=True,
                fill_opacity=0.9,
            ).add_to(m)

    folium.LayerControl().add_to(m)

    if colormap:
        colormap.add_to(m)

    return m