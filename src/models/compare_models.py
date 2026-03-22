"""
Three-way comparison of CHM prediction models on the SAME held-out test set:
1. XGBoost baseline (Google AlphaEarth embeddings)
2. Deep learning fine-tuned MLP (Google AlphaEarth embeddings)
3. Meta Global Canopy Height Map

All evaluated against ALS CHM from Reis/Jackson on test transects.
Also generates side-by-side wall-to-wall raster predictions over the AOI
with Sentinel-2 RGB for visual context.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ee

from src.config import (
    GEE_PROJECT, DATA_DIR, MODEL_DIR, AOI_PATH,
    MAX_CHM, EMBED_DIM, EMBEDDING_COLS, SCALE, EMBEDDING_YEAR,
)


def init_gee():
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)


def evaluate(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "bias": np.nan, "n": 0}
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
        "n": len(y_true),
    }


def get_meta_chm_at_locations(lons, lats, batch_size=2000):
    """Sample Meta Global CHM at given locations via GEE."""
    meta_col = ee.ImageCollection(
        "projects/sat-io/open-datasets/facebook/meta-canopy-height"
    )
    all_vals = np.full(len(lons), np.nan)

    for start in range(0, len(lons), batch_size):
        end = min(start + batch_size, len(lons))
        features = [
            ee.Feature(ee.Geometry.Point([float(lons[i]), float(lats[i])]), {"idx": i})
            for i in range(start, end)
        ]
        fc = ee.FeatureCollection(features)
        meta_img = meta_col.filterBounds(fc.geometry()).mosaic()
        sampled = meta_img.sampleRegions(collection=fc, scale=10, geometries=False)
        n = sampled.size().getInfo()
        if n > 0:
            for f in sampled.getInfo()["features"]:
                props = f["properties"]
                idx = props.get("idx")
                val = None
                for key in ["b1", "canopy_height", "B1"]:
                    if key in props:
                        val = props[key]
                        break
                if val is None:
                    for key, v in props.items():
                        if key != "idx" and isinstance(v, (int, float)):
                            val = v
                            break
                if val is not None and val != 255:
                    all_vals[idx] = float(val)

        pct = min(end, len(lons)) / len(lons) * 100
        print(f"\r  Meta CHM sampling: {pct:.0f}%", end="", flush=True)
    print()
    return all_vals


def extract_aoi_embeddings(aoi_geojson_path, scale=30):
    """Extract Google embeddings for every pixel in the AOI."""
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.read_file(aoi_geojson_path)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    geojson = gdf.geometry.union_all().__geo_interface__
    ee_geom = ee.Geometry(geojson)
    emb_img = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{EMBEDDING_YEAR}-01-01", f"{int(EMBEDDING_YEAR)+1}-01-01")
        .filterBounds(ee_geom)
        .mosaic()
    )

    bounds = gdf.total_bounds
    deg_per_pixel = scale / 111320.0
    lons = np.arange(bounds[0], bounds[2], deg_per_pixel)
    lats = np.arange(bounds[3], bounds[1], -deg_per_pixel)
    grid_lons, grid_lats = np.meshgrid(lons, lats)

    poly = gdf.geometry.union_all()
    mask = np.zeros(grid_lons.shape, dtype=bool)
    for iy in range(len(lats)):
        for ix in range(len(lons)):
            if poly.contains(Point(grid_lons[iy, ix], grid_lats[iy, ix])):
                mask[iy, ix] = True

    valid_lons = grid_lons[mask]
    valid_lats = grid_lats[mask]
    print(f"  AOI grid: {len(lons)}x{len(lats)}, {mask.sum()} inside AOI")

    all_embeddings = np.full((len(valid_lons), EMBED_DIM), np.nan)
    batch_size = 2000
    for start in range(0, len(valid_lons), batch_size):
        end = min(start + batch_size, len(valid_lons))
        features = [
            ee.Feature(ee.Geometry.Point([float(valid_lons[i]), float(valid_lats[i])]), {"idx": i})
            for i in range(start, end)
        ]
        fc = ee.FeatureCollection(features)
        sampled = emb_img.sampleRegions(collection=fc, scale=scale, geometries=False)
        n = sampled.size().getInfo()
        if n > 0:
            for f in sampled.getInfo()["features"]:
                props = f["properties"]
                idx = props["idx"]
                for j in range(EMBED_DIM):
                    key = f"A{j:02d}"
                    if key in props and props[key] is not None:
                        all_embeddings[idx, j] = props[key]
        pct = min(end, len(valid_lons)) / len(valid_lons) * 100
        print(f"\r  AOI embeddings: {pct:.0f}%", end="", flush=True)
    print()

    return {
        "lons": lons, "lats": lats,
        "grid_lons": grid_lons, "grid_lats": grid_lats,
        "mask": mask, "valid_lons": valid_lons, "valid_lats": valid_lats,
        "embeddings": all_embeddings,
    }


def predict_xgboost(embeddings):
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.load_model(str(MODEL_DIR / "xgboost_baseline_best.json"))
    valid = ~np.any(np.isnan(embeddings), axis=1)
    preds = np.full(len(embeddings), np.nan)
    if valid.sum() > 0:
        preds[valid] = model.predict(embeddings[valid])
    return preds


def predict_mlp(embeddings):
    import torch
    from src.models.mlp_finetune import MLPRegressor

    cp = torch.load(MODEL_DIR / "mlp_best.pt", weights_only=False, map_location="cpu")
    model = MLPRegressor(input_dim=EMBED_DIM, hidden_dim=256, n_layers=4, dropout=0.1)
    model.load_state_dict(cp["state_dict"])
    model.eval()

    valid = ~np.any(np.isnan(embeddings), axis=1)
    preds = np.full(len(embeddings), np.nan)
    if valid.sum() > 0:
        X = (embeddings[valid] - cp["X_mean"]) / (cp["X_std"] + 1e-8)
        with torch.no_grad():
            batch_size = 4096
            parts = []
            for i in range(0, len(X), batch_size):
                parts.append(model(torch.from_numpy(X[i:i+batch_size].astype(np.float32))).numpy())
            preds[valid] = np.concatenate(parts)
    return preds


def main():
    init_gee()

    print("=" * 70)
    print("THREE-WAY CHM COMPARISON — TEST SET (all vs ALS CHM)")
    print("=" * 70)

    # Load data and split
    df = pd.read_parquet(DATA_DIR / "embeddings_chm_30m.parquet")
    df = df[df["chm"] <= MAX_CHM].copy()

    split_df = pd.read_csv(MODEL_DIR / "spatial_split.csv")
    test_t = set(split_df[split_df["split"] == "test"]["transect"])
    test_df = df[df["transect"].isin(test_t)].copy()
    print(f"Test set: {len(test_df)} samples, {len(test_t)} transects")

    models = {}

    # XGBoost test predictions
    xgb_path = MODEL_DIR / "xgboost_test_predictions.parquet"
    if xgb_path.exists():
        xgb = pd.read_parquet(xgb_path)
        m = evaluate(xgb["y_true"].values, xgb["y_pred"].values)
        models["XGBoost\n(Google Embeddings)"] = {
            "metrics": m, "y_true": xgb["y_true"].values, "y_pred": xgb["y_pred"].values,
        }
        print(f"\nXGBoost:     MAE={m['mae']:.2f}m  RMSE={m['rmse']:.2f}m  R²={m['r2']:.3f}  Bias={m['bias']:.2f}m")

    # MLP test predictions
    mlp_path = MODEL_DIR / "mlp_test_predictions.parquet"
    if mlp_path.exists():
        mlp = pd.read_parquet(mlp_path)
        m = evaluate(mlp["y_true"].values, mlp["y_pred"].values)
        models["Fine-tuned MLP\n(Google Embeddings)"] = {
            "metrics": m, "y_true": mlp["y_true"].values, "y_pred": mlp["y_pred"].values,
        }
        print(f"Fine-tuned MLP: MAE={m['mae']:.2f}m  RMSE={m['rmse']:.2f}m  R²={m['r2']:.3f}  Bias={m['bias']:.2f}m")

    # Meta CHM at test locations
    meta_cache = DATA_DIR / "meta_chm_at_test_transects.parquet"
    if meta_cache.exists():
        print("Using cached Meta CHM values at test locations")
        meta_df = pd.read_parquet(meta_cache)
    else:
        print("\nSampling Meta CHM at test transect locations...")
        meta_vals = get_meta_chm_at_locations(
            test_df["longitude"].values, test_df["latitude"].values
        )
        meta_df = pd.DataFrame({
            "chm": test_df["chm"].values,
            "meta_chm": meta_vals,
        })
        meta_df.to_parquet(meta_cache, index=False)

    valid = np.isfinite(meta_df["meta_chm"].values) & (meta_df["meta_chm"].values < MAX_CHM)
    if valid.sum() > 0:
        m = evaluate(meta_df["chm"].values[valid], meta_df["meta_chm"].values[valid])
        models["Meta Global CHM"] = {
            "metrics": m,
            "y_true": meta_df["chm"].values[valid],
            "y_pred": meta_df["meta_chm"].values[valid],
        }
        print(f"Meta CHM:    MAE={m['mae']:.2f}m  RMSE={m['rmse']:.2f}m  R²={m['r2']:.3f}  Bias={m['bias']:.2f}m")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'Model':<35} {'MAE(m)':<9} {'RMSE(m)':<9} {'R²':<8} {'Bias(m)':<9} {'N':<8}")
    print(f"{'-' * 70}")
    for name, data in models.items():
        m = data["metrics"]
        print(f"{name.replace(chr(10),' '):<35} {m['mae']:<9.2f} {m['rmse']:<9.2f} "
              f"{m['r2']:<8.3f} {m['bias']:<9.2f} {m['n']:<8}")
    print(f"{'=' * 70}")

    # Scatter plots
    colors_map = {
        "XGBoost\n(Google Embeddings)": "#4A90D9",
        "Fine-tuned MLP\n(Google Embeddings)": "#2ECC71",
        "Meta Global CHM": "#E67E22",
    }
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, models.items()):
        m = data["metrics"]
        ax.scatter(data["y_true"], data["y_pred"], s=1, alpha=0.08,
                   c=colors_map.get(name, "gray"), rasterized=True)
        ax.plot([0, MAX_CHM], [0, MAX_CHM], "r--", lw=1.5, label="1:1")
        ax.set_xlabel("ALS CHM (m)", fontsize=11)
        ax.set_ylabel("Predicted CHM (m)", fontsize=11)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.text(0.05, 0.95,
                f"MAE = {m['mae']:.2f} m\nRMSE = {m['rmse']:.2f} m\nR² = {m['r2']:.3f}\n"
                f"Bias = {m['bias']:.2f} m\nN = {m['n']:,}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
        ax.set_xlim(0, MAX_CHM)
        ax.set_ylim(0, MAX_CHM)
        ax.set_aspect("equal")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Test Set — All Models vs ALS CHM (Reis/Jackson)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "three_way_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nScatter plots saved to {MODEL_DIR / 'three_way_comparison.png'}")

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    model_names = [n.replace("\n", " ") for n in models.keys()]
    bar_colors = [colors_map.get(n, "gray") for n in models.keys()]
    for ax, key, label in zip(axes, ["mae", "rmse", "r2"], ["MAE (m)", "RMSE (m)", "R²"]):
        vals = [models[n]["metrics"][key] for n in models.keys()]
        bars = ax.bar(model_names, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10)
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("Test Set — Key Metrics (all vs ALS CHM)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "three_way_metrics_bars.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save CSV
    rows = [{"model": n.replace("\n", " "), "reference": "ALS CHM", **d["metrics"]}
            for n, d in models.items()]
    pd.DataFrame(rows).to_csv(MODEL_DIR / "three_way_comparison.csv", index=False)

    # ── Raster maps over AOI ──
    print(f"\n{'=' * 70}")
    print("WALL-TO-WALL RASTERS OVER AOI")
    print(f"{'=' * 70}")

    aoi_cache = DATA_DIR / "aoi_embeddings.npz"
    if aoi_cache.exists():
        print("Loading cached AOI embeddings...")
        cached = np.load(aoi_cache, allow_pickle=True)
        aoi_grid = {k: cached[k] for k in cached.files}
    else:
        print("Extracting Google embeddings over AOI...")
        aoi_grid = extract_aoi_embeddings(str(AOI_PATH), scale=SCALE)
        np.savez_compressed(aoi_cache, **aoi_grid)

    embeddings = aoi_grid["embeddings"]
    mask = aoi_grid["mask"]
    lons = aoi_grid["lons"]
    lats = aoi_grid["lats"]

    xgb_pred = predict_xgboost(embeddings)
    mlp_pred = predict_mlp(embeddings)

    xgb_raster = np.full(mask.shape, np.nan)
    mlp_raster = np.full(mask.shape, np.nan)
    xgb_raster[mask] = xgb_pred
    mlp_raster[mask] = mlp_pred

    # Meta CHM raster
    import rioxarray
    import rasterio
    ds = rioxarray.open_rasterio(DATA_DIR / "meta_chm_10m.tif")
    meta_data = ds.values[0].astype(np.float32)
    meta_data[meta_data == 255] = np.nan
    meta_lons, meta_lats = ds.x.values, ds.y.values
    ds.close()

    # S2 RGB
    s2_path = DATA_DIR / "s2_rgb_aoi.tif"
    if s2_path.exists():
        with rasterio.open(s2_path) as src:
            rgb = src.read()
            s2_bounds = src.bounds
        rgb_display = np.zeros((rgb.shape[1], rgb.shape[2], 3), dtype=np.float32)
        for i in range(3):
            band = rgb[i].astype(np.float32)
            p2, p98 = np.percentile(band[band > 0], [2, 98])
            rgb_display[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        s2_extent = [s2_bounds.left, s2_bounds.right, s2_bounds.bottom, s2_bounds.top]
        has_s2 = True
    else:
        has_s2 = False

    # Plot
    vmin, vmax = 0, 30
    cmap = plt.cm.YlGn
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    n_panels = 4 if has_s2 else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 8))

    extent_aoi = [lons[0], lons[-1], lats[-1], lats[0]]
    extent_meta = [meta_lons[0], meta_lons[-1], meta_lats[-1], meta_lats[0]]
    xlim = [min(lons[0], meta_lons[0]) - 0.002, max(lons[-1], meta_lons[-1]) + 0.002]
    ylim = [min(lats[-1], meta_lats[-1]) - 0.002, max(lats[0], meta_lats[0]) + 0.002]

    idx = 0
    if has_s2:
        axes[idx].imshow(rgb_display, extent=s2_extent, aspect="equal", interpolation="bilinear")
        axes[idx].set_title("Sentinel-2 RGB\n(2017, 10m)", fontsize=13, fontweight="bold")
        idx += 1

    axes[idx].imshow(xgb_raster, cmap=cmap, norm=norm, extent=extent_aoi, aspect="equal", interpolation="nearest")
    axes[idx].set_title("XGBoost\n(Google Embeddings, 30m)", fontsize=13, fontweight="bold")
    idx += 1

    axes[idx].imshow(mlp_raster, cmap=cmap, norm=norm, extent=extent_aoi, aspect="equal", interpolation="nearest")
    axes[idx].set_title("Fine-tuned MLP\n(Google Embeddings, 30m)", fontsize=13, fontweight="bold")
    idx += 1

    axes[idx].imshow(meta_data, cmap=cmap, norm=norm, extent=extent_meta, aspect="equal", interpolation="nearest")
    axes[idx].set_title("Meta Global CHM\n(10m)", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("Longitude", fontsize=10)
        ax.grid(True, alpha=0.15, color="gray")
    axes[0].set_ylabel("Latitude", fontsize=11)

    chm_axes = axes[1:] if has_s2 else axes
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=list(chm_axes), orientation="horizontal", fraction=0.03, pad=0.08, aspect=50,
        label="Canopy Height (m)",
    )

    plt.suptitle("Canopy Height Predictions vs Satellite Imagery — AOI Tocantins, Brazil",
                 fontsize=16, fontweight="bold", y=1.01)

    plt.savefig(MODEL_DIR / "three_way_rasters.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Raster comparison saved to {MODEL_DIR / 'three_way_rasters.png'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
