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

def display_choropleth(geojson: dict):
    features = geojson.get("features", [])
    print("GeoJSON features:", len(features))

    # infer the key that identifies each feature
    featureidkey = "id"
    candidate = next((f for f in features if f.get("properties")), None)
    if candidate:
        preferred_props = ["id", "ID", "Id", "code", "Code", "OBJECTID"]
        key = next(
            (k for k in preferred_props if k in candidate["properties"]),
            next(iter(candidate["properties"]), None),
        )
        if key:
            featureidkey = f"properties.{key}"
            ids = [f["properties"][key] for f in features]
        else:
            ids = [f.setdefault("id", str(i)) for i, f in enumerate(features)]
    else:
        ids = [f.setdefault("id", str(i)) for i, f in enumerate(features)]

    df = pd.DataFrame({"id": ids, "z": np.ones(len(ids))})

    fig = px.choropleth(
        df,
        geojson=geojson,
        locations="id",
        color="z",
        featureidkey=featureidkey,
        projection="mercator",
        range_color=(1, 1),
    )
    fig.update_traces(showscale=False)
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig
