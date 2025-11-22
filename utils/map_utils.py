from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import folium
from folium.plugins import MousePosition

def load_json(filepath: Path):
    """Load GeoJSON file and extract area codes."""
    with open(filepath, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    return geojson_data
