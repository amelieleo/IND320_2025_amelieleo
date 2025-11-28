from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon


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


def _geometry_line_coordinates(geometry) -> list[list[tuple[float, float]]]:
    coords: list[list[tuple[float, float]]] = []

    def extract_polygon(poly: Polygon) -> None:
        coords.append(list(poly.exterior.coords))
        for interior in poly.interiors:
            coords.append(list(interior.coords))

    if isinstance(geometry, Polygon):
        extract_polygon(geometry)
    elif isinstance(geometry, MultiPolygon):
        for geom in geometry.geoms:
            extract_polygon(geom)
    elif isinstance(geometry, LineString):
        coords.append(list(geometry.coords))
    elif isinstance(geometry, MultiLineString):
        for geom in geometry.geoms:
            coords.append(list(geom.coords))

    return coords


def _extract_event_coordinates(event: dict) -> tuple[float | None, float | None]:
    lat = event.get("lat")
    lon = event.get("lon")
    if lat is not None and lon is not None:
        return float(lat), float(lon)

    point_data = event.get("pointData")
    if isinstance(point_data, dict):
        lat = point_data.get("lat") or point_data.get("latitude")
        lon = point_data.get("lon") or point_data.get("longitude")
        if lat is not None and lon is not None:
            return float(lat), float(lon)

    points = event.get("points")
    if isinstance(points, list) and points:
        point = points[0]
        lat = point.get("lat")
        lon = point.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)

    return None, None


def _extract_event_location(event: dict) -> str | None:
    location = event.get("location")
    if location:
        return normalize_price_area(location)

    point_data = event.get("pointData")
    if isinstance(point_data, dict):
        loc = point_data.get("location")
        if loc:
            return normalize_price_area(loc)

    points = event.get("points")
    if isinstance(points, list) and points:
        loc = points[0].get("location")
        if loc:
            return normalize_price_area(loc)
    return None

def display_map(
    geojson_data: gpd.GeoDataFrame,
    selected_price_area: str | None = None,
    clicked_points: Iterable[Sequence[float]] | None = None,
    value_map: Mapping[str, float] | None = None,
    value_caption: str | None = None,
) -> go.Figure:
    gdf = geojson_data.to_crs(epsg=4326).copy()
    area_field = next(
        (c for c in gdf.columns if str(c).lower().replace(" ", "_") == "price_area"),
        None,
    )
    if area_field is None:
        raise ValueError("GeoJSON is missing a 'price_area' column required for mapping.")

    gdf["_area_norm"] = gdf[area_field].map(normalize_price_area)
    gdf = gdf[~gdf["_area_norm"].isna()].copy()

    value_lookup = {
        norm_key: val
        for key, val in (value_map or {}).items()
        if (norm_key := normalize_price_area(key)) is not None
    }
    z_values = [
        value_lookup.get(norm) if norm in value_lookup else np.nan
        for norm in gdf["_area_norm"]
    ]
    valid_values = [val for val in z_values if pd.notna(val)]

    tooltip_field = "name" if "name" in gdf.columns else area_field
    hover_names = gdf[tooltip_field].fillna("Unknown").astype(str)

    hover_template = "<b>%{customdata[0]}</b>"
    if valid_values:
        hover_template += f"<br>{value_caption or 'Mean value'}: %{{z:.2f}}"
    hover_template += "<extra></extra>"

    fig = go.Figure()

    colorbar = dict(title=value_caption or "Mean value (kWh)") if valid_values else None
    fig.add_choroplethmapbox(
        geojson=gdf.__geo_interface__,
        locations=gdf["_area_norm"],
        z=z_values,
        colorscale=[
            [0.0, "#e0ecf4"],
            [0.5, "#9ebcda"],
            [1.0, "#8856a7"],
        ],
        zmin=min(valid_values) if valid_values else None,
        zmax=max(valid_values) if valid_values else None,
        marker={"line": {"color": "#1f2937", "width": 1}},
        colorbar=colorbar,
        hovertemplate=hover_template,
        customdata=np.stack([hover_names], axis=-1),
        showscale=bool(valid_values),
        featureidkey="properties._area_norm",
        name=value_caption or "Mean value",
    )

    selected_area_norm = normalize_price_area(selected_price_area)
    if selected_area_norm:
        selected_geoms = gdf.loc[gdf["_area_norm"] == selected_area_norm, "geometry"]
        for geom in selected_geoms:
            for ring in _geometry_line_coordinates(geom):
                lons = [pt[0] for pt in ring]
                lats = [pt[1] for pt in ring]
                fig.add_scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode="lines",
                    line={"color": "#f97316", "width": 4},
                    hoverinfo="skip",
                    showlegend=False,
                )

    if clicked_points:
        lats, lons = zip(
            *[
                (pt[0], pt[1])
                for pt in clicked_points
                if isinstance(pt, Sequence) and len(pt) == 2
            ]
        ) if clicked_points else ([], [])
        if lats and lons:
            fig.add_scattermapbox(
                lat=lats,
                lon=lons,
                mode="markers",
                marker={"size": 10, "color": "#ef4444"},
                name="Clicked points",
                hovertemplate="Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
            )

    minx, miny, maxx, maxy = gdf.total_bounds
    fig.update_layout(
        mapbox={
            "style": "carto-positron",
            "zoom": 2.5,
            "center": {"lat": (miny + maxy) / 2.0, "lon": (minx + maxx) / 2.0},
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.01, "x": 0, "xanchor": "left"},
        clickmode="event",
    )

    return fig