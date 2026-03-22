"""Build a 3-panel synchronized Leaflet map comparing ALS, Meta, and Fine-tuned CHM at 1km.

Each panel shows colored grid cells over an ESRI satellite basemap.
Pan/zoom is synced across all three panels.

Usage:
    python -m src.data_processing.build_3panel_map
"""

import json
import logging

import numpy as np
import pandas as pd

from src.config import DATA_DIR, RES_DEG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

VALIDATION_PATH = DATA_DIR / "validation_grid_1km_all.parquet"
OUTPUT_PATH = DATA_DIR / "chm_comparison_3panel.html"


def value_to_color(val, vmin=0, vmax=30):
    """Map value to YlGn hex color."""
    if np.isnan(val):
        return "#888888"
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    # YlGn colormap approximation (5 stops)
    stops = [
        (0.0, (255, 255, 229)),
        (0.25, (194, 230, 153)),
        (0.5, (120, 198, 121)),
        (0.75, (49, 163, 84)),
        (1.0, (0, 104, 55)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            r = int(c0[0] + f * (c1[0] - c0[0]))
            g = int(c0[1] + f * (c1[1] - c0[1]))
            b = int(c0[2] + f * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return f"#{stops[-1][1][0]:02x}{stops[-1][1][1]:02x}{stops[-1][1][2]:02x}"


def main():
    df = pd.read_parquet(str(VALIDATION_PATH))
    logger.info("Loaded %d rows", len(df))

    # Prepare data for each panel
    half = RES_DEG / 2
    panels = [
        ("ALS CHM (reference)", "canopy_height_ref"),
        ("Meta CHM", "canopy_height_meta"),
        ("Fine-tuned (XGBoost)", "canopy_height_finetuned"),
    ]

    # Build JS data arrays for each panel
    panel_data = {}
    for title, col in panels:
        records = []
        for _, row in df.iterrows():
            val = row[col]
            if pd.isna(val):
                continue
            records.append({
                "s": round(row["lat"] - half, 6),
                "n": round(row["lat"] + half, 6),
                "w": round(row["lon"] - half, 6),
                "e": round(row["lon"] + half, 6),
                "v": round(float(val), 1),
                "c": value_to_color(float(val)),
            })
        panel_data[col] = records
        logger.info("  %s: %d cells", title, len(records))

    # Center map on data centroid
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>CHM 3-Panel Comparison (1km)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: Arial, sans-serif; }}
  #container {{ display: flex; width: 100vw; height: 100vh; }}
  .panel {{ flex: 1; position: relative; border-right: 2px solid #333; }}
  .panel:last-child {{ border-right: none; }}
  .map {{ width: 100%; height: 100%; }}
  .panel-label {{
    position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
    z-index: 1000; background: rgba(0,0,0,0.8); color: white;
    padding: 6px 16px; border-radius: 4px; font: bold 14px Arial;
    pointer-events: none; white-space: nowrap;
  }}
  #colorbar {{
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    z-index: 1000; background: rgba(255,255,255,0.92); padding: 8px 16px;
    border-radius: 6px; display: flex; align-items: center; gap: 6px;
    font: 12px Arial; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
  }}
  #colorbar .gradient {{
    width: 200px; height: 14px; border-radius: 3px;
    background: linear-gradient(to right, #ffffe5, #c2e699, #78c679, #31a354, #006837);
  }}
</style>
</head>
<body>
<div id="container">
  <div class="panel">
    <div class="panel-label">{panels[0][0]}</div>
    <div id="map0" class="map"></div>
  </div>
  <div class="panel">
    <div class="panel-label">{panels[1][0]}</div>
    <div id="map1" class="map"></div>
  </div>
  <div class="panel">
    <div class="panel-label">{panels[2][0]}</div>
    <div id="map2" class="map"></div>
  </div>
</div>
<div id="colorbar">
  <span>0 m</span>
  <div class="gradient"></div>
  <span>30 m</span>
  <span style="margin-left:8px; font-weight:bold;">Canopy Height</span>
</div>

<script>
var data0 = {json.dumps(panel_data[panels[0][1]])};
var data1 = {json.dumps(panel_data[panels[1][1]])};
var data2 = {json.dumps(panel_data[panels[2][1]])};

var esri = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}';
var esriAttr = 'Tiles &copy; Esri';

var map0 = L.map('map0', {{renderer: L.canvas()}}).setView([{center_lat}, {center_lon}], 6);
var map1 = L.map('map1', {{renderer: L.canvas()}}).setView([{center_lat}, {center_lon}], 6);
var map2 = L.map('map2', {{renderer: L.canvas()}}).setView([{center_lat}, {center_lon}], 6);

L.tileLayer(esri, {{attribution: esriAttr, maxZoom: 18}}).addTo(map0);
L.tileLayer(esri, {{attribution: esriAttr, maxZoom: 18}}).addTo(map1);
L.tileLayer(esri, {{attribution: esriAttr, maxZoom: 18}}).addTo(map2);

function addRects(map, data) {{
  var group = L.layerGroup();
  for (var i = 0; i < data.length; i++) {{
    var d = data[i];
    L.rectangle([[d.s, d.w], [d.n, d.e]], {{
      color: d.c, fillColor: d.c, fillOpacity: 0.7,
      weight: 0.3, opacity: 0.5
    }}).bindPopup('CHM: ' + d.v + ' m').addTo(group);
  }}
  group.addTo(map);
}}

addRects(map0, data0);
addRects(map1, data1);
addRects(map2, data2);

// Sync pan/zoom across all 3 maps
var syncing = false;
function syncMaps(source, targets) {{
  source.on('moveend', function() {{
    if (syncing) return;
    syncing = true;
    var c = source.getCenter();
    var z = source.getZoom();
    for (var i = 0; i < targets.length; i++) {{
      targets[i].setView(c, z, {{animate: false}});
    }}
    syncing = false;
  }});
}}
syncMaps(map0, [map1, map2]);
syncMaps(map1, [map0, map2]);
syncMaps(map2, [map0, map1]);
</script>
</body>
</html>"""

    with open(str(OUTPUT_PATH), "w") as f:
        f.write(html)
    logger.info("Saved: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
