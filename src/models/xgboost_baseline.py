"""
XGBoost baseline: predict canopy height from Google AlphaEarth embeddings.

Uses a spatial train/val/test split by transect to avoid data leakage.
70% train, 15% val (early stopping), 15% test.
Reports metrics on the held-out test set.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

from src.config import (
    DATA_DIR, MODEL_DIR, EMBEDDING_COLS, TARGET_COL,
    SEED, MAX_CHM, TRAIN_FRAC, VAL_FRAC,
)

DATA_PATH = DATA_DIR / "embeddings_chm_30m.parquet"


def evaluate(y_true, y_pred):
    """Compute regression metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
        "n": len(y_true),
    }


def spatial_train_val_test_split(df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SEED):
    """
    Split data by transect into train/val/test.
    Entire transects stay together to avoid spatial leakage.
    """
    rng = np.random.RandomState(seed)
    transects = np.array(sorted(df["transect"].unique()))
    rng.shuffle(transects)

    n = len(transects)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_t = set(transects[:n_train])
    val_t = set(transects[n_train:n_train + n_val])
    test_t = set(transects[n_train + n_val:])

    train_mask = df["transect"].isin(train_t)
    val_mask = df["transect"].isin(val_t)
    test_mask = df["transect"].isin(test_t)

    return train_mask, val_mask, test_mask, train_t, val_t, test_t


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded: {len(df)} samples, {df['transect'].nunique()} transects")
    print(f"CHM range: {df[TARGET_COL].min():.1f} – {df[TARGET_COL].max():.1f} m")

    # Filter outliers
    n_before = len(df)
    df = df[df[TARGET_COL] <= MAX_CHM].copy()
    print(f"Filtered {n_before - len(df)} samples with CHM > {MAX_CHM}m")

    # Spatial split
    train_mask, val_mask, test_mask, train_t, val_t, test_t = spatial_train_val_test_split(df)

    X_train = df.loc[train_mask, EMBEDDING_COLS].values
    y_train = df.loc[train_mask, TARGET_COL].values
    X_val = df.loc[val_mask, EMBEDDING_COLS].values
    y_val = df.loc[val_mask, TARGET_COL].values
    X_test = df.loc[test_mask, EMBEDDING_COLS].values
    y_test = df.loc[test_mask, TARGET_COL].values

    print(f"\nSpatial split:")
    print(f"  Train: {len(X_train)} samples ({len(train_t)} transects)")
    print(f"  Val:   {len(X_val)} samples ({len(val_t)} transects)")
    print(f"  Test:  {len(X_test)} samples ({len(test_t)} transects)")

    # Save split info for reuse by other models
    pd.DataFrame({
        "split": (["train"] * len(train_t) + ["val"] * len(val_t) + ["test"] * len(test_t)),
        "transect": sorted(train_t) + sorted(val_t) + sorted(test_t),
    }).to_csv(MODEL_DIR / "spatial_split.csv", index=False)
    print(f"  Split saved to {MODEL_DIR / 'spatial_split.csv'}")

    # Train
    print(f"\n{'=' * 60}")
    print("Training XGBoost...")
    print(f"{'=' * 60}")

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

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=20,
    )

    # Evaluate on test
    y_pred_test = model.predict(X_test)
    test_metrics = evaluate(y_test, y_pred_test)

    print(f"\n{'=' * 60}")
    print(f"Test set results:")
    print(f"  MAE  = {test_metrics['mae']:.2f} m")
    print(f"  RMSE = {test_metrics['rmse']:.2f} m")
    print(f"  R²   = {test_metrics['r2']:.3f}")
    print(f"  Bias = {test_metrics['bias']:.2f} m")
    print(f"  N    = {test_metrics['n']}")

    # Also evaluate on val and train for reference
    y_pred_val = model.predict(X_val)
    val_metrics = evaluate(y_val, y_pred_val)
    y_pred_train = model.predict(X_train)
    train_metrics = evaluate(y_train, y_pred_train)

    print(f"\nTrain: MAE={train_metrics['mae']:.2f}m  RMSE={train_metrics['rmse']:.2f}m  R²={train_metrics['r2']:.3f}")
    print(f"Val:   MAE={val_metrics['mae']:.2f}m  RMSE={val_metrics['rmse']:.2f}m  R²={val_metrics['r2']:.3f}")
    print(f"Test:  MAE={test_metrics['mae']:.2f}m  RMSE={test_metrics['rmse']:.2f}m  R²={test_metrics['r2']:.3f}")

    # Save model
    model_path = MODEL_DIR / "xgboost_baseline_best.json"
    model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Save test predictions
    test_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred_test,
    })
    test_df.to_parquet(MODEL_DIR / "xgboost_test_predictions.parquet", index=False)

    # Save metrics
    all_metrics = pd.DataFrame([
        {"split": "train", **train_metrics},
        {"split": "val", **val_metrics},
        {"split": "test", **test_metrics},
    ])
    all_metrics.to_csv(MODEL_DIR / "xgboost_metrics.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (name, yt, yp, m) in zip(axes, [
        ("Train", y_train, y_pred_train, train_metrics),
        ("Val", y_val, y_pred_val, val_metrics),
        ("Test", y_test, y_pred_test, test_metrics),
    ]):
        ax.scatter(yt, yp, s=1, alpha=0.1, c="steelblue", rasterized=True)
        ax.plot([0, MAX_CHM], [0, MAX_CHM], "r--", lw=1.5, label="1:1")
        ax.set_xlabel("ALS CHM (m)")
        ax.set_ylabel("Predicted CHM (m)")
        ax.set_title(f"{name} (N={m['n']:,})")
        ax.text(
            0.05, 0.95,
            f"MAE={m['mae']:.2f}m\nRMSE={m['rmse']:.2f}m\nR²={m['r2']:.3f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.set_xlim(0, MAX_CHM)
        ax.set_ylim(0, MAX_CHM)
        ax.set_aspect("equal")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("XGBoost — Google Embeddings → CHM (Spatial Split)", fontsize=14)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "xgboost_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {MODEL_DIR / 'xgboost_results.png'}")


if __name__ == "__main__":
    main()
