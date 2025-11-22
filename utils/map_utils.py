from __future__ import annotations
import json
from pathlib import Path
from turtle import pd

import plotly.express as px
import numpy as np


def load_json(filepath: Path):
    """Load GeoJSON file and extract area codes."""
    with open(filepath, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    return geojson_data

def display_choropleth(geojson: dict):
    features = geojson.get("features", [])
    # Ensure each feature has an 'id' for mapping
    for i, f in enumerate(features):
        if not f.get("id"):
            f["id"] = str(i)

    df = pd.DataFrame({"id": [f["id"] for f in features], "z": np.ones(len(features))})

    fig = px.choropleth(
        df,
        geojson=geojson,
        locations="id",
        color="z",
        featureidkey="id",
        projection="mercator",
        range_color=(1, 1)  # single color
    )
    fig.update_traces(showscale=False)
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig
