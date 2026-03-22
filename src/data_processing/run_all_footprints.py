"""Process all ALS footprints to build the combined 1km validation grid.

For each tile:
1. Get bounds in EPSG:4326
2. Download Meta CHM from GEE for that footprint (cached)
3. Resample both ALS and Meta to 1km grid
4. Filter cells with >=90% coverage
5. Spatial join with WWF ecoregions

All results are concatenated into a single DataFrame.

Usage:
    python -m src.data_processing.run_all_footprints
"""

import glob
import logging
import os
import sys
from pathlib import Path

import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds, reproject
from shapely.geometry import Point

from src.config import GEE_PROJECT, RES_DEG, MIN_COVERAGE, DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- Paths ----
META_CACHE_DIR = DATA_DIR / "meta_chm_cache"
ECOREGIONS_PATH = DATA_DIR / "vector" / "resolve_ecoregions_sa.gpkg"
ALS_DIR = DATA_DIR / "chm_reis_jackson"
OUTPUT_PATH = DATA_DIR / "validation_grid_1km_all.parquet"


def resample_to_grid(src_path, grid_tf, gw, gh, nodata_in=None):
    """Resample a raster to the 1km grid, returning (mean, coverage)."""
    with rasterio.open(src_path) as src:
        nd = nodata_in if nodata_in is not None else src.nodata
        data = np.full((gh, gw), np.nan, dtype=np.float64)
        reproject(
            rasterio.band(src, 1), data,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=grid_tf, dst_crs="EPSG:4326",
            resampling=Resampling.average, src_nodata=nd, dst_nodata=np.nan,
        )
        raw = src.read(1).astype(np.float64)
        if nd is not None and not np.isnan(nd):
            mask = (raw != nd).astype(np.float64)
        else:
            mask = (~np.isnan(raw)).astype(np.float64)
        cov = np.full((gh, gw), 0.0, dtype=np.float64)
        reproject(
            mask, cov,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=grid_tf, dst_crs="EPSG:4326",
            resampling=Resampling.average, src_nodata=0.0, dst_nodata=0.0,
        )
    return data, cov


def download_meta_chm(bounds_4326, cache_path):
    """Download Meta CHM from GEE for the given bounds. Returns local path."""
    if os.path.exists(cache_path):
        return cache_path

    w, s, e, n = bounds_4326
    # Small buffer to ensure full coverage
    ee_geom = ee.Geometry.Rectangle([w - 0.01, s - 0.01, e + 0.01, n + 0.01])
    meta_ic = ee.ImageCollection(
        "projects/sat-io/open-datasets/facebook/meta-canopy-height"
    ).filterBounds(ee_geom)
    meta_img = meta_ic.mosaic().clip(ee_geom)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    geemap.ee_export_image(
        meta_img, filename=cache_path, scale=10,
        region=ee_geom, file_per_band=False,
    )

    # Fix nodata to 255
    ds = rioxarray.open_rasterio(cache_path)
    ds = ds.rio.write_nodata(255)
    ds.rio.to_raster(cache_path)

    return cache_path


def process_single_tile(tile_path, eco_gdf):
    """Process one ALS tile. Returns a DataFrame or None."""
    acq_name = Path(tile_path).stem.replace("_th_chm", "")

    with rasterio.open(tile_path) as src:
        if src.crs is None:
            logger.warning("[%s] No CRS, skipping", acq_name)
            return None
        b4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

    # Build 1km grid
    x0 = np.floor(b4326[0] / RES_DEG) * RES_DEG
    y1 = np.ceil(b4326[3] / RES_DEG) * RES_DEG
    x1 = np.ceil(b4326[2] / RES_DEG) * RES_DEG
    y0 = np.floor(b4326[1] / RES_DEG) * RES_DEG
    gw = int(round((x1 - x0) / RES_DEG))
    gh = int(round((y1 - y0) / RES_DEG))
    grid_tf = from_origin(x0, y1, RES_DEG, RES_DEG)

    # Download Meta CHM (cached)
    meta_cache = str(META_CACHE_DIR / f"{acq_name}_meta_10m.tif")
    try:
        meta_path = download_meta_chm(b4326, meta_cache)
    except Exception as e:
        logger.warning("[%s] Meta CHM download failed: %s", acq_name, e)
        return None

    # Resample
    als_mean, als_cov = resample_to_grid(tile_path, grid_tf, gw, gh)
    meta_mean, meta_cov = resample_to_grid(meta_path, grid_tf, gw, gh, nodata_in=255.0)

    # Coordinates
    rows_idx, cols_idx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    lons = grid_tf.c + (cols_idx.ravel() + 0.5) * grid_tf.a
    lats = grid_tf.f + (rows_idx.ravel() + 0.5) * grid_tf.e
    als_v = als_mean.ravel()
    meta_v = meta_mean.ravel()
    als_c = als_cov.ravel()
    meta_c = meta_cov.ravel()

    # Filter
    valid = (
        (als_c >= MIN_COVERAGE) & (meta_c >= MIN_COVERAGE)
        & np.isfinite(als_v) & np.isfinite(meta_v)
    )
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return None

    # Grid IDs (global)
    global_cols = np.round((lons - RES_DEG / 2) / RES_DEG).astype(int)
    global_rows = np.round(-(lats + RES_DEG / 2) / RES_DEG).astype(int)

    # Ecoregion spatial join
    points = gpd.GeoDataFrame(
        {"idx": np.arange(len(idx))},
        geometry=[Point(lons[i], lats[i]) for i in idx],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, eco_gdf[["ECO_ID", "ECO_NAME", "geometry"]], how="left")
    joined = joined.drop_duplicates(subset="idx", keep="first").sort_values("idx")

    df = pd.DataFrame({
        "lon": np.round(lons[idx], 6),
        "lat": np.round(lats[idx], 6),
        "canopy_height_ref": np.round(als_v[idx], 2),
        "canopy_height_meta": np.round(meta_v[idx], 2),
        "ecoregion_id": joined["ECO_ID"].fillna(-1).astype(int).values,
        "ecoregion_name": joined["ECO_NAME"].fillna("Unknown").values,
        "grid_id": [f"g_{global_cols[i]}_{global_rows[i]}" for i in idx],
        "acquisition_name": acq_name,
        "year": 2017,
        "als_coverage": np.round(als_c[idx], 4),
        "meta_coverage": np.round(meta_c[idx], 4),
    })
    return df


def main():
    # Init GEE
    ee.Initialize(project=GEE_PROJECT)
    logger.info("GEE initialized")

    # Load ecoregions once
    eco_gdf = gpd.read_file(str(ECOREGIONS_PATH))
    logger.info("Loaded %d ecoregion polygons", len(eco_gdf))

    # Discover tiles
    tiles = sorted(glob.glob(str(ALS_DIR / "**/*_chm.tif"), recursive=True))
    logger.info("Found %d ALS tiles", len(tiles))

    # Process all tiles
    all_dfs = []
    failed = []
    for i, tile in enumerate(tiles):
        name = Path(tile).stem.replace("_th_chm", "")
        try:
            df = process_single_tile(tile, eco_gdf)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                logger.info(
                    "[%d/%d] %s: %d cells", i + 1, len(tiles), name, len(df)
                )
            else:
                logger.info("[%d/%d] %s: 0 cells (skipped)", i + 1, len(tiles), name)
        except Exception as e:
            logger.error("[%d/%d] %s FAILED: %s", i + 1, len(tiles), name, e)
            failed.append(name)

    if not all_dfs:
        logger.error("No data produced!")
        sys.exit(1)

    # Concatenate and assign sample_id
    result = pd.concat(all_dfs, ignore_index=True)
    result.insert(0, "sample_id", np.arange(len(result)))

    # Drop duplicate grid cells (if overlapping tiles produced the same cell,
    # keep the one with higher ALS coverage)
    before = len(result)
    result = result.sort_values("als_coverage", ascending=False).drop_duplicates(
        subset="grid_id", keep="first"
    ).sort_values("sample_id").reset_index(drop=True)
    result["sample_id"] = np.arange(len(result))
    if len(result) < before:
        logger.info("Deduplicated: %d → %d cells (removed %d overlapping)",
                     before, len(result), before - len(result))

    # Save
    result.to_parquet(str(OUTPUT_PATH), index=False)
    result.to_csv(str(OUTPUT_PATH).replace(".parquet", ".csv"), index=False)
    logger.info("Saved %d rows to %s", len(result), OUTPUT_PATH)

    # Summary
    logger.info("--- Summary ---")
    logger.info("  Tiles processed: %d / %d", len(tiles) - len(failed), len(tiles))
    logger.info("  Failed: %d", len(failed))
    logger.info("  Total 1km cells: %d", len(result))
    logger.info("  Unique acquisitions: %d", result["acquisition_name"].nunique())
    logger.info("  Unique ecoregions: %d", result["ecoregion_name"].nunique())
    logger.info("  ALS CHM: mean=%.1f, std=%.1f",
                result["canopy_height_ref"].mean(), result["canopy_height_ref"].std())
    logger.info("  Meta CHM: mean=%.1f, std=%.1f",
                result["canopy_height_meta"].mean(), result["canopy_height_meta"].std())
    if failed:
        logger.info("  Failed tiles: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
