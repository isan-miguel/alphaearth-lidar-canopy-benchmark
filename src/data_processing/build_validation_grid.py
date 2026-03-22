"""Build a 1km validation grid DataFrame over an ALS acquisition footprint.

Everything operates in EPSG:4326. The grid uses ~0.009° spacing (~1km at
tropical latitudes), snapped to clean multiples so the same grid aligns
across different acquisitions.

Resamples ALS CHM (reference) and Meta CHM to the common grid, filters
cells with >=90% valid pixel coverage in BOTH datasets, and enriches with
WWF ecoregion attributes.

Output schema (per Xu Liang's spec):
    sample_id, lon, lat, canopy_height_ref, ecoregion_id, ecoregion_name,
    grid_id, grid_coverage, year, month

Usage:
    python -m src.data_processing.build_validation_grid \
        --als-dir data/chm_reis_jackson \
        --meta-chm data/meta_chm_10m.tif \
        --aoi AOI.geojson \
        --output data/validation_grid_1km.parquet
"""

import argparse
import logging
import tempfile
from glob import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject

from src.config import RES_DEG as DEFAULT_GRID_DEG

logger = logging.getLogger(__name__)

CRS_4326 = "EPSG:4326"


# ---------------------------------------------------------------------------
# 1. Grid construction (EPSG:4326)
# ---------------------------------------------------------------------------

def build_1km_grid_transform(bounds_4326: tuple, res_deg: float = DEFAULT_GRID_DEG):
    """Create a degree-based grid snapped to clean multiples of res_deg.

    Snapping ensures the same grid aligns across different acquisitions.

    Args:
        bounds_4326: (west, south, east, north) in EPSG:4326.
        res_deg: Grid cell size in degrees (~0.009° ≈ 1km).

    Returns:
        (transform, width, height) for the grid.
    """
    west, south, east, north = bounds_4326

    # Snap to grid multiples
    x0 = np.floor(west / res_deg) * res_deg
    y1 = np.ceil(north / res_deg) * res_deg   # north edge (raster origin)
    x1 = np.ceil(east / res_deg) * res_deg
    y0 = np.floor(south / res_deg) * res_deg

    width = int(round((x1 - x0) / res_deg))
    height = int(round((y1 - y0) / res_deg))
    transform = from_origin(x0, y1, res_deg, res_deg)

    return transform, width, height


# ---------------------------------------------------------------------------
# 2. Resample a raster to the 1km grid (reproject to EPSG:4326 if needed)
# ---------------------------------------------------------------------------

def resample_to_grid(
    src_path: str,
    dst_transform,
    dst_width: int,
    dst_height: int,
    nodata_in: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a raster to the target EPSG:4326 grid using average aggregation.

    Handles reprojection from any source CRS to EPSG:4326.

    Returns:
        (mean_values, coverage) — both shape (dst_height, dst_width).
        coverage is fraction of valid source pixels per grid cell [0, 1].
    """
    with rasterio.open(src_path) as src:
        src_nodata = nodata_in if nodata_in is not None else src.nodata

        # Resample the actual data (average)
        data_out = np.full((dst_height, dst_width), np.nan, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1),
            destination=data_out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=CRS_4326,
            resampling=Resampling.average,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )

        # Build a valid-pixel mask (1=valid, 0=nodata) and resample with
        # average → fraction of valid source pixels per grid cell
        raw = src.read(1).astype(np.float64)
        if src_nodata is not None:
            if np.isnan(src_nodata):
                valid_mask = (~np.isnan(raw)).astype(np.float64)
            else:
                valid_mask = (raw != src_nodata).astype(np.float64)
        else:
            valid_mask = (~np.isnan(raw)).astype(np.float64)

        coverage_out = np.full((dst_height, dst_width), 0.0, dtype=np.float64)
        reproject(
            source=valid_mask,
            destination=coverage_out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=CRS_4326,
            resampling=Resampling.average,
            src_nodata=0.0,
            dst_nodata=0.0,
        )

    return data_out, coverage_out


# ---------------------------------------------------------------------------
# 3. Build VRT from ALS tile directory
# ---------------------------------------------------------------------------

def build_vrt(als_dir: str, vrt_path: str) -> str:
    """Build a GDAL VRT from all *_chm.tif files in the ALS directory tree.

    Uses rasterio's GDAL bindings (no osgeo dependency required).
    """
    tifs = sorted(glob(f"{als_dir}/**/*_chm.tif", recursive=True))
    if not tifs:
        raise FileNotFoundError(f"No *_chm.tif files found under {als_dir}")
    logger.info("Building VRT from %d ALS tiles", len(tifs))

    try:
        from osgeo import gdal
        vrt_ds = gdal.BuildVRT(vrt_path, tifs)
        if vrt_ds is None:
            raise RuntimeError("gdal.BuildVRT returned None")
        vrt_ds.FlushCache()
        vrt_ds = None
    except ImportError:
        # Fallback: call gdalbuildvrt via subprocess
        import subprocess
        filelist = Path(vrt_path).with_suffix(".txt")
        filelist.write_text("\n".join(tifs))
        subprocess.run(
            ["gdalbuildvrt", "-input_file_list", str(filelist), vrt_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        filelist.unlink()

    return vrt_path


# ---------------------------------------------------------------------------
# 4. Get ALS footprint bounds in EPSG:4326
# ---------------------------------------------------------------------------

def als_bounds_4326(vrt_path: str) -> tuple:
    """Get the bounding box of the ALS VRT reprojected to EPSG:4326."""
    with rasterio.open(vrt_path) as src:
        if src.crs.to_epsg() == 4326:
            return src.bounds

        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, CRS_4326, src.width, src.height, *src.bounds
        )
        bounds = rasterio.transform.array_bounds(dst_h, dst_w, dst_transform)
        # array_bounds returns (left, bottom, right, top)
        return bounds


# ---------------------------------------------------------------------------
# 5. WWF Ecoregion lookup via local spatial join
# ---------------------------------------------------------------------------

def lookup_ecoregions(
    lons: np.ndarray,
    lats: np.ndarray,
    ecoregions_path: str = "data/vector/resolve_ecoregions_sa.gpkg",
) -> pd.DataFrame:
    """Spatial-join points against local RESOLVE/WWF ecoregions file.

    Args:
        lons: Array of longitudes.
        lats: Array of latitudes.
        ecoregions_path: Path to the ecoregions GeoPackage.

    Returns:
        DataFrame with columns: ecoregion_id, ecoregion_name (same length as input).
    """
    from shapely.geometry import Point

    eco_gdf = gpd.read_file(ecoregions_path)

    points = gpd.GeoDataFrame(
        {"idx": np.arange(len(lons))},
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(points, eco_gdf[["ECO_ID", "ECO_NAME", "geometry"]], how="left")
    # sjoin can produce duplicates if a point lands on a polygon boundary;
    # keep first match per point
    joined = joined.drop_duplicates(subset="idx", keep="first").sort_values("idx")

    return pd.DataFrame({
        "ecoregion_id": joined["ECO_ID"].fillna(-1).astype(int).values,
        "ecoregion_name": joined["ECO_NAME"].fillna("Unknown").values,
    })


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------

def build_validation_dataframe(
    als_dir: str,
    meta_chm_path: str,
    aoi_path: str,
    output_path: str,
    grid_res_deg: float = DEFAULT_GRID_DEG,
    min_coverage: float = 0.90,
    acquisition_name: str = "",
    acquisition_year: int = 2017,
    acquisition_month: int | None = None,
    meta_nodata: float = 255.0,
    ecoregions_path: str = "data/vector/resolve_ecoregions_sa.gpkg",
) -> pd.DataFrame:
    """Build the validation sample DataFrame at ~1km resolution in EPSG:4326.

    Args:
        als_dir: Directory containing ALS CHM tiles (*_chm.tif).
        meta_chm_path: Path to Meta Global Canopy Height GeoTIFF.
        aoi_path: Path to AOI GeoJSON (used for logging only; grid extent
            comes from the ALS footprint).
        output_path: Where to save the output Parquet.
        grid_res_deg: Grid cell size in degrees (~0.009° ≈ 1km).
        min_coverage: Minimum fraction of valid pixels required (default 0.90).
        acquisition_name: Name of the ALS acquisition/transect (e.g. "NP_T-0891").
            If empty, derived from the first tile filename.
        acquisition_year: Year of ALS data acquisition.
        acquisition_month: Optional month of acquisition.
        meta_nodata: NoData value for the Meta CHM raster.
        ecoregions_path: Path to local RESOLVE/WWF ecoregions file.

    Returns:
        The validation DataFrame.
    """
    logger.info("Grid resolution: %.6f° (≈%.0f m at equator)", grid_res_deg, grid_res_deg * 111_000)

    # Derive acquisition name from first tile if not provided
    if not acquisition_name:
        first_tile = sorted(glob(f"{als_dir}/**/*_chm.tif", recursive=True))[0]
        # e.g. "NP_T-0891_th_chm.tif" → "NP_T-0891"
        acquisition_name = Path(first_tile).stem.replace("_th_chm", "")
        logger.info("Auto-derived acquisition_name: %s", acquisition_name)

    # Build VRT from ALS tiles
    with tempfile.TemporaryDirectory() as tmpdir:
        vrt_path = str(Path(tmpdir) / "als_mosaic.vrt")
        build_vrt(als_dir, vrt_path)

        # Get ALS extent in EPSG:4326
        bounds = als_bounds_4326(vrt_path)
        logger.info("ALS footprint (4326): W=%.4f S=%.4f E=%.4f N=%.4f", *bounds)

        # Build the common 1km grid
        grid_transform, grid_w, grid_h = build_1km_grid_transform(bounds, grid_res_deg)
        logger.info("Grid: %d x %d cells (%d total)", grid_w, grid_h, grid_w * grid_h)

        # Resample ALS CHM to the grid
        logger.info("Resampling ALS CHM to grid...")
        als_mean, als_coverage = resample_to_grid(
            vrt_path, grid_transform, grid_w, grid_h
        )

    # Resample Meta CHM to the same grid
    logger.info("Resampling Meta CHM to grid...")
    meta_mean, meta_coverage = resample_to_grid(
        meta_chm_path, grid_transform, grid_w, grid_h,
        nodata_in=meta_nodata,
    )

    # Cell centroids (already in EPSG:4326)
    rows_idx, cols_idx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    rows_flat = rows_idx.ravel()
    cols_flat = cols_idx.ravel()

    lons = grid_transform.c + (cols_flat + 0.5) * grid_transform.a
    lats = grid_transform.f + (rows_flat + 0.5) * grid_transform.e

    # Flatten arrays
    als_vals = als_mean.ravel()
    meta_vals = meta_mean.ravel()
    als_cov = als_coverage.ravel()
    meta_cov = meta_coverage.ravel()

    # Filter: both datasets need >= min_coverage and valid values
    valid = (
        (als_cov >= min_coverage)
        & (meta_cov >= min_coverage)
        & np.isfinite(als_vals)
        & np.isfinite(meta_vals)
    )
    logger.info(
        "Grid cells: %d total, %d pass %.0f%% coverage filter (%.1f%%)",
        len(valid), valid.sum(), min_coverage * 100,
        100 * valid.sum() / len(valid) if len(valid) > 0 else 0,
    )

    # Build the DataFrame
    idx = np.where(valid)[0]
    n = len(idx)

    # Grid ID uses absolute row/col indices from a global grid origin at (0°, 0°).
    # This makes IDs stable across acquisitions: the same 1km cell always gets
    # the same grid_id regardless of which tile covers it.
    global_cols = np.round((lons - grid_res_deg / 2) / grid_res_deg).astype(int)
    global_rows = np.round(-(lats + grid_res_deg / 2) / grid_res_deg).astype(int)
    grid_ids = [f"g_{global_cols[i]}_{global_rows[i]}" for i in idx]
    df = pd.DataFrame({
        "sample_id": np.arange(n),
        "lon": np.round(lons[idx], 6),
        "lat": np.round(lats[idx], 6),
        "canopy_height_ref": np.round(als_vals[idx], 2),
        "canopy_height_meta": np.round(meta_vals[idx], 2),
        "ecoregion_id": -1,
        "ecoregion_name": "",
        "grid_id": grid_ids,
        "acquisition_name": acquisition_name,
        "year": acquisition_year,
        "month": acquisition_month if acquisition_month is not None else pd.NA,
        "als_coverage": np.round(als_cov[idx], 4),
        "meta_coverage": np.round(meta_cov[idx], 4),
    })

    # Ecoregion enrichment via local spatial join
    logger.info("Looking up WWF ecoregions for %d grid cells...", n)
    try:
        eco_df = lookup_ecoregions(
            df["lon"].values, df["lat"].values, ecoregions_path
        )
        df["ecoregion_id"] = eco_df["ecoregion_id"].values
        df["ecoregion_name"] = eco_df["ecoregion_name"].values
    except Exception as e:
        logger.warning("Ecoregion lookup failed: %s. Columns left as defaults.", e)

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    logger.info("Saved %d rows to %s", n, output)

    # Summary
    logger.info("--- Summary ---")
    logger.info("  ALS ref CHM: mean=%.1f m, std=%.1f m",
                df["canopy_height_ref"].mean(), df["canopy_height_ref"].std())
    logger.info("  Meta CHM:    mean=%.1f m, std=%.1f m",
                df["canopy_height_meta"].mean(), df["canopy_height_meta"].std())
    logger.info("  ALS coverage:  mean=%.2f, min=%.2f",
                df["als_coverage"].mean(), df["als_coverage"].min())
    logger.info("  Meta coverage: mean=%.2f, min=%.2f",
                df["meta_coverage"].mean(), df["meta_coverage"].min())

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build 1km validation grid DataFrame for CHM comparison."
    )
    parser.add_argument("--als-dir", required=True, help="Directory with ALS CHM tiles.")
    parser.add_argument("--meta-chm", required=True, help="Path to Meta CHM GeoTIFF.")
    parser.add_argument("--aoi", required=True, help="Path to AOI GeoJSON.")
    parser.add_argument("--output", default="data/validation_grid_1km.parquet")
    parser.add_argument(
        "--grid-res-deg", type=float, default=DEFAULT_GRID_DEG,
        help="Grid cell size in degrees (default ~0.009° ≈ 1km).",
    )
    parser.add_argument("--min-coverage", type=float, default=0.90,
                        help="Min valid-pixel fraction (default 0.90).")
    parser.add_argument("--acquisition-name", default="",
                        help="ALS footprint/transect name (auto-derived from filename if empty).")
    parser.add_argument("--year", type=int, default=2017, help="ALS acquisition year.")
    parser.add_argument("--month", type=int, default=None, help="ALS acquisition month.")
    parser.add_argument("--meta-nodata", type=float, default=255.0,
                        help="Meta CHM nodata value.")
    parser.add_argument("--ecoregions", default="data/vector/resolve_ecoregions_sa.gpkg",
                        help="Path to RESOLVE/WWF ecoregions file.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    build_validation_dataframe(
        als_dir=args.als_dir,
        meta_chm_path=args.meta_chm,
        aoi_path=args.aoi,
        output_path=args.output,
        grid_res_deg=args.grid_res_deg,
        min_coverage=args.min_coverage,
        acquisition_name=args.acquisition_name,
        acquisition_year=args.year,
        acquisition_month=args.month,
        meta_nodata=args.meta_nodata,
        ecoregions_path=args.ecoregions,
    )


if __name__ == "__main__":
    main()
