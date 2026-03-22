"""
Utilities for accessing geospatial datasets via Google Earth Engine.
Includes functions for GEDI canopy height and Meta CHM retrieval.
"""

import ee
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import GEE_PROJECT


def init_gee():
    """Initialize GEE with the project credentials."""
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)


def load_aoi(geojson_path):
    """Load AOI from GeoJSON and return as ee.Geometry and GeoDataFrame."""
    gdf = gpd.read_file(geojson_path)
    # Convert to EPSG:4326 if needed
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    geojson = gdf.geometry.unary_union.__geo_interface__
    ee_geom = ee.Geometry(geojson)
    return ee_geom, gdf


def get_gedi_l2a(ee_geom, start_date="2019-04-01", end_date="2023-03-31"):
    """
    Retrieve GEDI L2A canopy height shots within the AOI.

    Uses rh98 (98th percentile relative height) as canopy top height.
    Filters for quality_flag == 1 and degrade_flag == 0.

    Parameters
    ----------
    ee_geom : ee.Geometry
        Area of interest.
    start_date, end_date : str
        Date range for GEDI data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: latitude, longitude, rh98, rh75, rh50,
        quality_flag, sensitivity, date.
    """
    gedi = (
        ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY")
        .filterBounds(ee_geom)
        .filterDate(start_date, end_date)
        .select(["rh98", "rh75", "rh50", "rh25", "quality_flag", "degrade_flag", "sensitivity"])
    )

    # Quality filter: good shots only
    gedi_filtered = gedi.map(
        lambda img: img.updateMask(
            img.select("quality_flag").eq(1).And(img.select("degrade_flag").eq(0))
        )
    )

    # Sample the collection as points within AOI
    # GEDI footprints are ~25m, so we sample at 25m scale
    def extract_points(img):
        date = img.date().format("YYYY-MM-dd")
        samples = img.sample(
            region=ee_geom,
            scale=25,
            geometries=True,
        )
        return samples.map(lambda f: f.set("date", date))

    points = gedi_filtered.map(extract_points).flatten()

    # Limit to avoid memory issues; fetch in batches if needed
    n_points = points.size().getInfo()
    print(f"Found {n_points} GEDI quality shots in AOI")

    if n_points == 0:
        return pd.DataFrame()

    # Cap at 5000 points to avoid GEE timeout
    if n_points > 5000:
        print(f"Capping at 5000 points (of {n_points})")
        points = points.limit(5000)

    features = points.getInfo()["features"]

    rows = []
    for f in features:
        props = f["properties"]
        coords = f["geometry"]["coordinates"]
        rows.append({
            "longitude": coords[0],
            "latitude": coords[1],
            "rh98": props.get("rh98"),
            "rh75": props.get("rh75"),
            "rh50": props.get("rh50"),
            "rh25": props.get("rh25"),
            "sensitivity": props.get("sensitivity"),
            "date": props.get("date"),
        })

    df = pd.DataFrame(rows)
    # GEDI rh values in GEE are already in meters
    return df


def get_meta_chm_asset(ee_geom):
    """
    Access the Meta Global Canopy Height Map from GEE Community Catalog.

    Returns an ee.Image clipped to the AOI.
    """
    # Meta Global Canopy Height from GEE Community Catalog (sat-io)
    meta_chm = ee.ImageCollection(
        "projects/sat-io/open-datasets/facebook/meta-canopy-height"
    ).filterBounds(ee_geom).mosaic().clip(ee_geom)

    return meta_chm


def export_meta_chm(ee_geom, output_path="data/meta_chm.tif", scale=1):
    """
    Export Meta CHM clipped to AOI as a GeoTIFF.

    Note: At 1m resolution, the AOI may produce a large file.
    Consider using scale=10 for initial exploration.

    Parameters
    ----------
    ee_geom : ee.Geometry
        Area of interest.
    output_path : str
        Local path for the exported GeoTIFF.
    scale : int
        Export resolution in meters. Default 1m (native).
        Use 10 for faster downloads.
    """
    import geemap

    meta_chm = get_meta_chm_asset(ee_geom)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    geemap.ee_export_image(
        meta_chm,
        filename=output_path,
        scale=scale,
        region=ee_geom,
        file_per_band=False,
    )
    print(f"Meta CHM exported to {output_path}")


if __name__ == "__main__":
    from src.config import AOI_PATH

    init_gee()
    ee_geom, gdf = load_aoi(str(AOI_PATH))

    # Test GEDI retrieval
    print("Fetching GEDI L2A data...")
    gedi_df = get_gedi_l2a(ee_geom)
    if not gedi_df.empty:
        print(f"GEDI shots: {len(gedi_df)}")
        print(gedi_df.describe())
        gedi_df.to_csv("data/gedi_l2a_aoi.csv", index=False)
        print("Saved to data/gedi_l2a_aoi.csv")

    # Test Meta CHM access
    print("\nAccessing Meta CHM...")
    meta_chm = get_meta_chm_asset(ee_geom)
    info = meta_chm.getInfo()
    print(f"Meta CHM bands: {[b['id'] for b in info['bands']]}")
