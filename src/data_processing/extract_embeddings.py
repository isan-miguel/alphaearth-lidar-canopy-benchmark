"""
Extract Google AlphaEarth Satellite Embeddings paired with CHM labels at 30m.

For each CHM transect GeoTIFF:
1. Resample CHM from 1m to 30m (mean aggregation)
2. Get the transect footprint in WGS84
3. Sample Google embeddings (64 bands) at 30m on a regular grid
4. Pair each embedding pixel with the 30m CHM value
5. Save as training-ready parquet

Output: Parquet file with columns [longitude, latitude, A00..A63, chm, transect]
"""

import ee
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import pyproj
from pathlib import Path
import glob
import sys

from src.config import (
    GEE_PROJECT, EMBEDDING_COLLECTION, EMBEDDING_YEAR,
    SCALE, DATA_DIR, EMBEDDING_COLS,
)


def init_gee():
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)


def resample_chm_to_30m(tif_path):
    """
    Resample 1m CHM to 30m using mean aggregation.

    Returns the resampled xarray DataArray and its CRS.
    """
    ds = rioxarray.open_rasterio(tif_path)
    crs = ds.rio.crs
    res_x, res_y = ds.rio.resolution()  # (1.0, -1.0) typically

    # Coarsen by factor of 30 (1m -> 30m), take mean
    # Trim dimensions to be divisible by 30
    factor = SCALE
    ny, nx = ds.shape[1], ds.shape[2]
    ny_trim = (ny // factor) * factor
    nx_trim = (nx // factor) * factor
    ds_trimmed = ds[:, :ny_trim, :nx_trim]

    ds_30m = ds_trimmed.coarsen(x=factor, y=factor, boundary="trim").mean()
    ds_30m = ds_30m.rio.write_crs(crs)

    # Update transform for the coarsened grid
    orig_transform = ds.rio.transform()
    new_transform = orig_transform * orig_transform.scale(factor, factor)
    ds_30m = ds_30m.rio.write_transform(new_transform)

    ds.close()
    return ds_30m


def get_grid_points_wgs84(ds_30m):
    """
    Get center coordinates of all valid 30m pixels in WGS84.

    Returns list of (lon, lat, chm_value) tuples.
    """
    crs = ds_30m.rio.crs
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    data = ds_30m.values[0]  # (y, x)
    y_coords = ds_30m.y.values
    x_coords = ds_30m.x.values

    points = []
    for iy, y in enumerate(y_coords):
        for ix, x in enumerate(x_coords):
            val = data[iy, ix]
            if not np.isnan(val):
                lon, lat = transformer.transform(float(x), float(y))
                points.append((lon, lat, float(val)))

    return points


def extract_embeddings_for_transect(tif_path):
    """
    Extract embeddings and CHM at 30m for a single transect.

    1. Resample CHM to 30m
    2. Build grid of 30m pixel centers in WGS84
    3. Query GEE embeddings at those locations

    Returns DataFrame with [longitude, latitude, A00..A63, chm, transect]
    """
    tif_path = Path(tif_path)
    transect_id = tif_path.stem.replace("_th_chm", "")

    # Resample CHM
    ds_30m = resample_chm_to_30m(tif_path)
    points = get_grid_points_wgs84(ds_30m)
    ds_30m.close()

    if len(points) == 0:
        return pd.DataFrame()

    # Cap points per transect to avoid GEE limits
    # Transects are ~12.5km x 0.3km -> ~400x10 = ~4000 pixels at 30m
    # Most will be fewer since transects are narrow strips
    max_points = 5000
    if len(points) > max_points:
        rng = np.random.RandomState(hash(transect_id) % (2**31))
        idx = rng.choice(len(points), max_points, replace=False)
        points = [points[i] for i in idx]

    # Build ee.FeatureCollection from grid points
    ee_features = []
    chm_lookup = {}
    for lon, lat, chm_val in points:
        key = f"{lon:.7f}_{lat:.7f}"
        chm_lookup[key] = chm_val
        ee_features.append(
            ee.Feature(ee.Geometry.Point([lon, lat]), {"key": key})
        )

    # GEE has a limit on FeatureCollection size in sampleRegions
    # Process in batches of 2000
    batch_size = 2000
    all_rows = []

    for batch_start in range(0, len(ee_features), batch_size):
        batch = ee_features[batch_start:batch_start + batch_size]
        fc = ee.FeatureCollection(batch)

        # Get embedding image
        emb_img = (
            ee.ImageCollection(EMBEDDING_COLLECTION)
            .filterDate(f"{EMBEDDING_YEAR}-01-01", f"{int(EMBEDDING_YEAR)+1}-01-01")
            .filterBounds(fc.geometry())
            .mosaic()
        )

        samples = emb_img.sampleRegions(
            collection=fc,
            scale=SCALE,
            geometries=True,
        )

        n = samples.size().getInfo()
        if n == 0:
            continue

        features = samples.getInfo()["features"]
        for f in features:
            props = f["properties"]
            coords = f["geometry"]["coordinates"]
            key = props.get("key", "")
            row = {
                "longitude": coords[0],
                "latitude": coords[1],
                "chm": chm_lookup.get(key, np.nan),
                "transect": transect_id,
            }
            for i in range(64):
                row[f"A{i:02d}"] = props.get(f"A{i:02d}")
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Drop rows with missing data
    emb_cols = EMBEDDING_COLS
    df = df.dropna(subset=emb_cols + ["chm"])

    return df


def _process_one(tif):
    """Worker function for parallel extraction."""
    transect_id = Path(tif).stem.replace("_th_chm", "")
    try:
        df = extract_embeddings_for_transect(tif)
        if not df.empty:
            return transect_id, df
        return transect_id, None
    except Exception as e:
        return transect_id, f"ERROR: {e}"


def main():
    init_gee()

    chm_dir = DATA_DIR / "chm_reis_jackson"
    output_path = DATA_DIR / "embeddings_chm_30m.parquet"

    tifs = sorted(glob.glob(str(chm_dir / "**/*.tif"), recursive=True))
    print(f"Found {len(tifs)} CHM transect GeoTIFFs")
    print(f"Resolution: {SCALE}m, Embedding year: {EMBEDDING_YEAR}")

    # Load existing data to skip already-processed transects
    existing_df = None
    existing_ids = set()
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        existing_ids = set(existing_df["transect"].unique())
        print(f"Existing data: {len(existing_df)} samples from {len(existing_ids)} transects")

    # Allow specifying a subset
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        tifs = tifs[:n]
        print(f"Processing first {n} transects")

    # Filter to only unprocessed transects
    tifs_todo = [t for t in tifs
                 if Path(t).stem.replace("_th_chm", "") not in existing_ids]
    print(f"Remaining to process: {len(tifs_todo)} transects")

    if not tifs_todo:
        print("Nothing to do.")
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # GEE calls are I/O-bound, so threading helps significantly
    N_WORKERS = 8
    print(f"Using {N_WORKERS} parallel workers")

    all_dfs = [existing_df] if existing_df is not None else []
    lock = threading.Lock()
    n_done = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_process_one, tif): tif for tif in tifs_todo}

        for future in as_completed(futures):
            transect_id, result = future.result()
            n_done += 1

            if isinstance(result, pd.DataFrame):
                with lock:
                    all_dfs.append(result)
                print(f"[{n_done}/{len(tifs_todo)}] {transect_id}: {len(result)} samples")
            elif isinstance(result, str):
                print(f"[{n_done}/{len(tifs_todo)}] {transect_id}: {result}")
            else:
                print(f"[{n_done}/{len(tifs_todo)}] {transect_id}: skipped")

            # Save intermediate results every 20 transects
            if n_done % 20 == 0:
                with lock:
                    interim = pd.concat(all_dfs, ignore_index=True)
                    interim.to_parquet(output_path, index=False)
                    print(f"  -> Saved: {len(interim)} samples, "
                          f"{interim['transect'].nunique()} transects")

    if all_dfs:
        final = pd.concat(all_dfs, ignore_index=True)
        final.to_parquet(output_path, index=False)
        print(f"\nDone. {len(final)} samples from {final['transect'].nunique()} transects -> {output_path}")
        print(final[["chm"]].describe())
    else:
        print("No data extracted")


if __name__ == "__main__":
    main()
