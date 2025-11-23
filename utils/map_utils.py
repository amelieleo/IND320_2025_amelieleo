from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import numpy as np

def load_json(filepath: Path):
    """Load GeoJSON file and extract area codes."""
    with open(filepath, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    return geojson_data

# def display_choropleth(geojson: dict):
#     features = geojson.get("features", [])
#     print("GeoJSON features:", len(features))
#     # Ensure each feature has an 'id' for mapping
#     for i, f in enumerate(features):
#         if not f.get("id"):
#             f["id"] = str(i)

#     ids = [f["id"] for f in features]
#     df = pd.DataFrame({"id": ids, "z": np.ones(len(ids))})

#     fig = px.choropleth(
#         df,
#         geojson=geojson,
#         locations="id",
#         color="z",
#         featureidkey="id",
#         projection="mercator",
#         range_color=(1, 1),
#     )
#     fig.update_traces(showscale=False)
#     fig.update_geos(fitbounds="locations", visible=False)
#     fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

#     return fig

# ...existing code...
def display_choropleth(geojson: dict):
    features = geojson.get("features", [])
    print("GeoJSON features:", len(features))

    preferred_props = ["id", "ID", "Id", "code", "Code", "OBJECTID", "Price area"]
    rows: list[dict[str, str | float]] = []
    used_ids: set[str] = set()

    for idx, feature in enumerate(features):
        props = feature.setdefault("properties", {})
        key = next((k for k in preferred_props if k in props), next(iter(props), None))
        label = str(props[key]) if key else f"feature_{idx}"

        feature_id = str(feature.get("id") or props.get("id") or label)
        if feature_id in used_ids:
            feature_id = f"{feature_id}_{idx}"
        feature["id"] = feature_id
        used_ids.add(feature_id)

        rows.append({"feature_id": feature_id, "label": label, "value": 1.0})

    df = pd.DataFrame(rows)

    fig = px.choropleth(
        df,
        geojson=geojson,
        locations="feature_id",
        color="label",
        featureidkey="id",
        projection="mercator",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(showscale=False)
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig
# ...existing code...