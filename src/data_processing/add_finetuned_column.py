"""Add fine-tuned model predictions to the 1km validation grid for ALL footprints.

Uses K-fold spatial cross-validation (10 folds by transect) so each footprint
is predicted by a model that never saw it during training. Embeddings (64-dim
Google AlphaEarth) are the covariates.

For each fold:
1. Train XGBoost on the other 9 folds
2. Predict CHM at 30m for each held-out transect
3. Build 30m prediction raster per transect
4. Resample to 1km grid
5. Filter: only keep 1km cells with >=90% coverage of underlying 30m predictions

Usage:
    python -m src.data_processing.add_finetuned_column
"""

import glob
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from xgboost import XGBRegressor

from src.config import (
    DATA_DIR, EMBEDDING_COLS, TARGET_COL, MAX_CHM,
    RES_DEG, MIN_COVERAGE, N_FOLDS, SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- Paths ----
EMBEDDING_PATH = DATA_DIR / "embeddings_chm_30m.parquet"
VALIDATION_PATH = DATA_DIR / "validation_grid_1km_all.parquet"
ALS_DIR = DATA_DIR / "chm_reis_jackson"


def find_als_tile(target_name):
    """Find the ALS CHM tile path for a given acquisition name."""
    pattern = str(ALS_DIR / "**" / f"{target_name}*_chm.tif")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    return matches[0]


def build_30m_raster(target_df, als_tile_path, output_path):
    """Reconstruct a 30m GeoTIFF from embedding-level predictions."""
    with rasterio.open(als_tile_path) as src:
        native_crs = src.crs
        native_transform = src.transform
        native_h, native_w = src.height, src.width

    factor = 30
    w_30 = native_w // factor
    h_30 = native_h // factor
    tf_30 = native_transform * Affine.scale(factor, factor)

    t = Transformer.from_crs("EPSG:4326", native_crs, always_xy=True)
    px, py = t.transform(
        target_df["longitude"].values, target_df["latitude"].values
    )

    data = np.full((h_30, w_30), np.nan, dtype=np.float32)
    for x, y, val in zip(px, py, target_df["chm_pred"].values):
        col = int((x - tf_30.c) / tf_30.a)
        row = int((y - tf_30.f) / tf_30.e)
        if 0 <= col < w_30 and 0 <= row < h_30:
            data[row, col] = val

    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=h_30, width=w_30, count=1, dtype="float32",
        crs=native_crs, transform=tf_30, nodata=np.nan,
    ) as dst:
        dst.write(data, 1)

    return output_path


def resample_to_1km(raster_path, grid_tf, gw, gh):
    """Resample a raster to the 1km grid, returning (mean, coverage)."""
    with rasterio.open(raster_path) as src:
        nd = src.nodata
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


def process_transect(transect_name, emb_subset, als_tile_path, tmpdir):
    """Build 30m raster, resample to 1km, return {grid_id: (value, coverage)}."""
    pred_path = os.path.join(tmpdir, f"{transect_name}_pred_30m.tif")
    build_30m_raster(emb_subset, als_tile_path, pred_path)

    with rasterio.open(als_tile_path) as src:
        b4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

    x0 = np.floor(b4326[0] / RES_DEG) * RES_DEG
    y1 = np.ceil(b4326[3] / RES_DEG) * RES_DEG
    x1 = np.ceil(b4326[2] / RES_DEG) * RES_DEG
    y0 = np.floor(b4326[1] / RES_DEG) * RES_DEG
    gw = int(round((x1 - x0) / RES_DEG))
    gh = int(round((y1 - y0) / RES_DEG))
    grid_tf = from_origin(x0, y1, RES_DEG, RES_DEG)

    pred_mean, pred_cov = resample_to_1km(pred_path, grid_tf, gw, gh)

    rows_idx, cols_idx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    lons = grid_tf.c + (cols_idx.ravel() + 0.5) * grid_tf.a
    lats = grid_tf.f + (rows_idx.ravel() + 0.5) * grid_tf.e
    pred_v = pred_mean.ravel()
    pred_c = pred_cov.ravel()

    global_cols = np.round((lons - RES_DEG / 2) / RES_DEG).astype(int)
    global_rows = np.round(-(lats + RES_DEG / 2) / RES_DEG).astype(int)

    result = {}
    for i in range(len(lons)):
        if np.isfinite(pred_v[i]) and pred_c[i] >= MIN_COVERAGE:
            gid = f"g_{global_cols[i]}_{global_rows[i]}"
            result[gid] = round(float(pred_v[i]), 2)

    # Clean up temp raster
    os.remove(pred_path)
    return result


def main():
    # Load data
    logger.info("Loading embeddings...")
    emb_df = pd.read_parquet(EMBEDDING_PATH)
    emb_df = emb_df[emb_df[TARGET_COL] <= MAX_CHM].copy()
    logger.info("Loaded %d samples, %d transects", len(emb_df), emb_df["transect"].nunique())

    val_df = pd.read_parquet(VALIDATION_PATH)
    # Drop old finetuned column if exists
    if "canopy_height_finetuned" in val_df.columns:
        val_df = val_df.drop(columns=["canopy_height_finetuned"])
    logger.info("Loaded validation grid: %d rows", len(val_df))

    # Build K-fold split by transect
    rng = np.random.RandomState(SEED)
    all_transects = np.array(sorted(emb_df["transect"].unique()))
    rng.shuffle(all_transects)
    folds = np.array_split(all_transects, N_FOLDS)

    logger.info("Split %d transects into %d folds (avg %.0f per fold)",
                len(all_transects), N_FOLDS, len(all_transects) / N_FOLDS)

    # Global lookup: grid_id -> predicted CHM value
    all_predictions = {}  # grid_id -> value
    total_processed = 0
    total_failed = 0

    with tempfile.TemporaryDirectory(prefix="chm_pred_") as tmpdir:
        for fold_idx, held_out_transects in enumerate(folds):
            held_out_set = set(held_out_transects)
            train_mask = ~emb_df["transect"].isin(held_out_set)

            # Further split training into train/val for early stopping
            train_transects = np.array(sorted(
                emb_df.loc[train_mask, "transect"].unique()
            ))
            rng2 = np.random.RandomState(SEED + fold_idx)
            rng2.shuffle(train_transects)
            n_val = max(1, int(len(train_transects) * 0.1))
            val_t = set(train_transects[:n_val])

            tr = emb_df[train_mask].copy()
            val_mask_inner = tr["transect"].isin(val_t)

            X_train = tr.loc[~val_mask_inner, EMBEDDING_COLS].values
            y_train = tr.loc[~val_mask_inner, TARGET_COL].values
            X_val = tr.loc[val_mask_inner, EMBEDDING_COLS].values
            y_val = tr.loc[val_mask_inner, TARGET_COL].values

            logger.info(
                "=== Fold %d/%d: training on %d samples, held-out %d transects ===",
                fold_idx + 1, N_FOLDS, len(X_train), len(held_out_transects),
            )

            model = XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                random_state=SEED,
                early_stopping_rounds=20,
                eval_metric="rmse",
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            logger.info("  Best iteration: %d", model.best_iteration)

            # Predict for each held-out transect
            for t_idx, transect in enumerate(held_out_transects):
                als_tile = find_als_tile(transect)
                if als_tile is None:
                    total_failed += 1
                    continue

                t_emb = emb_df[emb_df["transect"] == transect].copy()
                if len(t_emb) == 0:
                    total_failed += 1
                    continue

                t_emb["chm_pred"] = model.predict(t_emb[EMBEDDING_COLS].values)

                try:
                    preds = process_transect(transect, t_emb, als_tile, tmpdir)
                    all_predictions.update(preds)
                    total_processed += 1

                    if (total_processed % 50) == 0:
                        logger.info(
                            "  Progress: %d transects processed, %d 1km cells so far",
                            total_processed, len(all_predictions),
                        )
                except Exception as e:
                    logger.warning("  %s failed: %s", transect, e)
                    total_failed += 1

            logger.info(
                "  Fold %d done: %d transects processed this fold",
                fold_idx + 1, len(held_out_transects) - sum(
                    1 for t in held_out_transects if find_als_tile(t) is None
                ),
            )

    logger.info("=== All folds complete ===")
    logger.info("  Processed: %d transects", total_processed)
    logger.info("  Failed: %d transects", total_failed)
    logger.info("  Total 1km predictions (>=%.0f%% coverage): %d",
                MIN_COVERAGE * 100, len(all_predictions))

    # Add to validation grid
    val_df["canopy_height_finetuned"] = val_df["grid_id"].map(all_predictions)
    matched = val_df["canopy_height_finetuned"].notna().sum()
    logger.info("Matched %d / %d validation cells", matched, len(val_df))

    # Save
    val_df.to_parquet(str(VALIDATION_PATH), index=False)
    val_df.to_csv(str(VALIDATION_PATH).replace(".parquet", ".csv"), index=False)
    logger.info("Saved updated validation grid to %s", VALIDATION_PATH)

    # Summary metrics
    valid = val_df.dropna(subset=["canopy_height_finetuned"])
    if len(valid) > 0:
        ref = valid["canopy_height_ref"].values
        meta = valid["canopy_height_meta"].values
        pred = valid["canopy_height_finetuned"].values

        logger.info("=== Overall 1km Metrics ===")
        for name, y_pred in [("Meta", meta), ("Fine-tuned", pred)]:
            mae = np.mean(np.abs(y_pred - ref))
            rmse = np.sqrt(np.mean((y_pred - ref) ** 2))
            bias = np.mean(y_pred - ref)
            r2 = 1 - np.sum((ref - y_pred) ** 2) / np.sum((ref - ref.mean()) ** 2)
            logger.info("  %s: MAE=%.2f  RMSE=%.2f  Bias=%.2f  R2=%.3f  N=%d",
                        name, mae, rmse, bias, r2, len(y_pred))


if __name__ == "__main__":
    main()
