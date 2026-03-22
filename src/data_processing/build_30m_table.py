"""Build 30m validation table and merge with 1km into a single multi-resolution table.

For each transect at 30m:
- canopy_height_ref: ALS CHM resampled to 30m (from embeddings parquet)
- canopy_height_meta: Meta CHM sampled at 30m points (from cached per-tile rasters)
- canopy_height_finetuned: K-fold XGBoost predictions (leave-fold-out)

Then merges 30m and 1km tables with a 'resolution' column (30 or 1000).

Usage:
    python -m src.data_processing.build_30m_table
"""

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from xgboost import XGBRegressor

from src.config import (
    DATA_DIR, EMBEDDING_COLS, TARGET_COL, MAX_CHM,
    RES_DEG, N_FOLDS, SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EMBEDDING_PATH = DATA_DIR / "embeddings_chm_30m.parquet"
VALIDATION_1KM_PATH = DATA_DIR / "validation_grid_1km_all.parquet"
META_CACHE_DIR = DATA_DIR / "meta_chm_cache"
ECOREGIONS_PATH = DATA_DIR / "vector" / "resolve_ecoregions_sa.gpkg"
OUTPUT_PATH = DATA_DIR / "validation_grid_multi_resolution.parquet"


def sample_meta_chm_for_transect(transect_name, lons, lats):
    """Sample Meta CHM at given lon/lat points from cached per-tile raster."""
    meta_path = META_CACHE_DIR / f"{transect_name}_meta_10m.tif"
    if not meta_path.exists():
        return np.full(len(lons), np.nan)

    with rasterio.open(meta_path) as src:
        # Transform lon/lat to raster CRS
        if src.crs.to_epsg() != 4326:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = t.transform(lons, lats)
        else:
            xs, ys = lons, lats

        # Sample at each point
        nodata = src.nodata
        data = src.read(1)
        values = np.full(len(lons), np.nan)

        for i, (x, y) in enumerate(zip(xs, ys)):
            row, col = src.index(x, y)
            if 0 <= row < src.height and 0 <= col < src.width:
                val = data[row, col]
                if nodata is not None and val == nodata:
                    continue
                values[i] = float(val)

    return values


def kfold_predictions(emb_df):
    """Run K-fold XGBoost and return predictions for every sample."""
    emb_df = emb_df.copy()
    emb_df["chm_pred"] = np.nan

    rng = np.random.RandomState(SEED)
    all_transects = np.array(sorted(emb_df["transect"].unique()))
    rng.shuffle(all_transects)
    folds = np.array_split(all_transects, N_FOLDS)

    for fold_idx, held_out in enumerate(folds):
        held_out_set = set(held_out)
        train_mask = ~emb_df["transect"].isin(held_out_set)
        test_mask = emb_df["transect"].isin(held_out_set)

        # Split training into train/val for early stopping
        train_transects = np.array(sorted(
            emb_df.loc[train_mask, "transect"].unique()
        ))
        rng2 = np.random.RandomState(SEED + fold_idx)
        rng2.shuffle(train_transects)
        n_val = max(1, int(len(train_transects) * 0.1))
        val_t = set(train_transects[:n_val])

        tr = emb_df[train_mask]
        val_inner = tr["transect"].isin(val_t)

        X_train = tr.loc[~val_inner, EMBEDDING_COLS].values
        y_train = tr.loc[~val_inner, TARGET_COL].values
        X_val = tr.loc[val_inner, EMBEDDING_COLS].values
        y_val = tr.loc[val_inner, TARGET_COL].values

        logger.info(
            "Fold %d/%d: train=%d, held-out=%d transects",
            fold_idx + 1, N_FOLDS, len(X_train), len(held_out),
        )

        model = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=SEED, early_stopping_rounds=20, eval_metric="rmse",
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        logger.info("  Best iteration: %d", model.best_iteration)

        # Predict held-out
        X_test = emb_df.loc[test_mask, EMBEDDING_COLS].values
        emb_df.loc[test_mask, "chm_pred"] = model.predict(X_test)

    return emb_df["chm_pred"].values


def main():
    # Load embeddings
    logger.info("Loading embeddings...")
    emb_df = pd.read_parquet(EMBEDDING_PATH)
    n_before = len(emb_df)
    emb_df = emb_df[emb_df[TARGET_COL] <= MAX_CHM].copy()
    logger.info("Loaded %d samples (%d after CHM filter), %d transects",
                n_before, len(emb_df), emb_df["transect"].nunique())

    # Step 1: K-fold fine-tuned predictions
    logger.info("=== Running K-fold XGBoost predictions ===")
    emb_df["chm_pred"] = kfold_predictions(emb_df)

    # Step 2: Sample Meta CHM at each 30m point
    logger.info("=== Sampling Meta CHM at 30m points ===")
    transects = emb_df["transect"].unique()
    meta_values = np.full(len(emb_df), np.nan)

    for i, transect in enumerate(transects):
        mask = emb_df["transect"] == transect
        idx = emb_df.index[mask]
        lons = emb_df.loc[idx, "longitude"].values
        lats = emb_df.loc[idx, "latitude"].values
        vals = sample_meta_chm_for_transect(transect, lons, lats)
        meta_values[emb_df.index.get_indexer(idx)] = vals

        if (i + 1) % 100 == 0:
            logger.info("  Sampled Meta CHM for %d/%d transects", i + 1, len(transects))

    emb_df["meta_chm"] = meta_values
    logger.info("Meta CHM sampled: %d valid out of %d",
                np.isfinite(meta_values).sum(), len(meta_values))

    # Step 3: Filter to points where all three products are valid
    valid_mask = (
        np.isfinite(emb_df[TARGET_COL].values)
        & np.isfinite(emb_df["meta_chm"].values)
        & np.isfinite(emb_df["chm_pred"].values)
        & (emb_df["meta_chm"].values != 255)  # Meta nodata
    )
    df_30m = emb_df[valid_mask].copy()
    logger.info("Valid 30m points (all 3 products): %d / %d", len(df_30m), len(emb_df))

    # Step 4: Add ecoregions via spatial join
    logger.info("Adding ecoregions...")
    import geopandas as gpd
    from shapely.geometry import Point

    eco_gdf = gpd.read_file(str(ECOREGIONS_PATH))

    # Sample in batches to avoid memory issues
    batch_size = 100000
    eco_ids = np.full(len(df_30m), -1, dtype=int)
    eco_names = np.full(len(df_30m), "Unknown", dtype=object)

    for start in range(0, len(df_30m), batch_size):
        end = min(start + batch_size, len(df_30m))
        batch_idx = df_30m.index[start:end]
        points_gdf = gpd.GeoDataFrame(
            {"idx": np.arange(end - start)},
            geometry=[
                Point(lon, lat) for lon, lat in zip(
                    df_30m.loc[batch_idx, "longitude"].values,
                    df_30m.loc[batch_idx, "latitude"].values,
                )
            ],
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(
            points_gdf, eco_gdf[["ECO_ID", "ECO_NAME", "geometry"]], how="left"
        )
        joined = joined.drop_duplicates(subset="idx", keep="first").sort_values("idx")
        eco_ids[start:end] = joined["ECO_ID"].fillna(-1).astype(int).values
        eco_names[start:end] = joined["ECO_NAME"].fillna("Unknown").values

        if start > 0:
            logger.info("  Ecoregion join: %d / %d", end, len(df_30m))

    # Step 5: Compute grid_id at 30m
    # Use same global grid system but at 30m (~0.00027° in EPSG:4326)
    res_30m_deg = 30.0 / 111000.0  # ~0.000270°
    global_cols_30 = np.round(df_30m["longitude"].values / res_30m_deg).astype(int)
    global_rows_30 = np.round(-df_30m["latitude"].values / res_30m_deg).astype(int)

    # Step 6: Build 30m DataFrame
    result_30m = pd.DataFrame({
        "sample_id": np.arange(len(df_30m)),
        "lon": np.round(df_30m["longitude"].values, 6),
        "lat": np.round(df_30m["latitude"].values, 6),
        "canopy_height_ref": np.round(df_30m[TARGET_COL].values, 2),
        "canopy_height_meta": np.round(df_30m["meta_chm"].values, 2),
        "canopy_height_finetuned": np.round(df_30m["chm_pred"].values, 2),
        "ecoregion_id": eco_ids,
        "ecoregion_name": eco_names,
        "grid_id": [f"g30_{c}_{r}" for c, r in zip(global_cols_30, global_rows_30)],
        "acquisition_name": df_30m["transect"].values,
        "year": 2017,
        "resolution": 30,
    })

    logger.info("30m table: %d rows, %d acquisitions",
                len(result_30m), result_30m["acquisition_name"].nunique())

    # Step 7: Load 1km table and add resolution column
    val_1km = pd.read_parquet(VALIDATION_1KM_PATH)
    val_1km["resolution"] = 1000

    # Ensure same columns in both
    shared_cols = [
        "sample_id", "lon", "lat",
        "canopy_height_ref", "canopy_height_meta", "canopy_height_finetuned",
        "ecoregion_id", "ecoregion_name", "grid_id",
        "acquisition_name", "year", "resolution",
    ]

    val_1km_sub = val_1km[shared_cols].copy()
    result_30m_sub = result_30m[shared_cols].copy()

    # Step 8: Concatenate and save
    combined = pd.concat([val_1km_sub, result_30m_sub], ignore_index=True)
    combined["sample_id"] = np.arange(len(combined))

    combined.to_parquet(str(OUTPUT_PATH), index=False)
    combined.to_csv(str(OUTPUT_PATH).replace(".parquet", ".csv"), index=False)

    logger.info("=== Combined table saved: %s ===", OUTPUT_PATH)
    logger.info("  Total rows: %d", len(combined))
    logger.info("  1km rows: %d", (combined["resolution"] == 1000).sum())
    logger.info("  30m rows: %d", (combined["resolution"] == 30).sum())

    # Metrics by resolution
    for res in [30, 1000]:
        sub = combined[
            (combined["resolution"] == res)
            & combined["canopy_height_finetuned"].notna()
        ]
        if len(sub) == 0:
            continue
        ref = sub["canopy_height_ref"].values
        logger.info("  --- %dm metrics (N=%d) ---", res, len(sub))
        for name, col in [("Meta", "canopy_height_meta"), ("Fine-tuned", "canopy_height_finetuned")]:
            y = sub[col].values
            valid = np.isfinite(y) & np.isfinite(ref)
            y, r = y[valid], ref[valid]
            mae = np.mean(np.abs(y - r))
            rmse = np.sqrt(np.mean((y - r) ** 2))
            bias = np.mean(y - r)
            r2 = 1 - np.sum((r - y) ** 2) / np.sum((r - r.mean()) ** 2)
            logger.info("    %s: MAE=%.2f  RMSE=%.2f  Bias=%.2f  R2=%.3f",
                        name, mae, rmse, bias, r2)


if __name__ == "__main__":
    main()
