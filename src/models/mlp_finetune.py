"""
Fine-tune a deep learning regression model for canopy height prediction
using Google AlphaEarth satellite embeddings (64-dim).

Uses the same spatial train/val/test split as XGBoost for a fair comparison.
Architecture: MLP with residual connections.

Outputs:
  - data/models/mlp_test_predictions.parquet
  - data/models/mlp_metrics.csv
  - data/models/mlp_best.pt
  - data/models/mlp_results.png
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    DATA_DIR, MODEL_DIR, EMBEDDING_COLS, TARGET_COL,
    SEED, MAX_CHM, EMBED_DIM,
)

DATA_PATH = DATA_DIR / "embeddings_chm_30m.parquet"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BATCH_SIZE = 512
EPOCHS = 100
LR = 1e-3


class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class MLPRegressor(nn.Module):
    """Deep MLP with residual connections."""

    def __init__(self, input_dim=64, hidden_dim=256, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
        "n": len(y_true),
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    criterion = nn.MSELoss()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(X)
        n += len(X)
    return total_loss / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for X, _ in loader:
        preds.append(model(X.to(device)).cpu().numpy())
    return np.concatenate(preds)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_parquet(DATA_PATH)
    df = df[df[TARGET_COL] <= MAX_CHM].copy()
    print(f"Loaded: {len(df)} samples, {df['transect'].nunique()} transects")

    # Load the same spatial split used by XGBoost
    split_path = MODEL_DIR / "spatial_split.csv"
    if not split_path.exists():
        print("ERROR: Run xgboost_baseline.py first to generate the spatial split")
        return

    split_df = pd.read_csv(split_path)
    train_t = set(split_df[split_df["split"] == "train"]["transect"])
    val_t = set(split_df[split_df["split"] == "val"]["transect"])
    test_t = set(split_df[split_df["split"] == "test"]["transect"])

    train_mask = df["transect"].isin(train_t)
    val_mask = df["transect"].isin(val_t)
    test_mask = df["transect"].isin(test_t)

    X_train = df.loc[train_mask, EMBEDDING_COLS].values.astype(np.float32)
    y_train = df.loc[train_mask, TARGET_COL].values.astype(np.float32)
    X_val = df.loc[val_mask, EMBEDDING_COLS].values.astype(np.float32)
    y_val = df.loc[val_mask, TARGET_COL].values.astype(np.float32)
    X_test = df.loc[test_mask, EMBEDDING_COLS].values.astype(np.float32)
    y_test = df.loc[test_mask, TARGET_COL].values.astype(np.float32)

    print(f"Train: {len(X_train)} ({len(train_t)} transects)")
    print(f"Val:   {len(X_val)} ({len(val_t)} transects)")
    print(f"Test:  {len(X_test)} ({len(test_t)} transects)")

    # Normalize embeddings (fit on train only)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(EmbeddingDataset(X_test, y_test), batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    model = MLPRegressor(input_dim=EMBED_DIM, hidden_dim=256, n_layers=4, dropout=0.1).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}, Max epochs: {EPOCHS}, LR: {LR}")
    print(f"{'=' * 60}")

    best_val_rmse = float("inf")
    patience = 15
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        scheduler.step()

        val_pred = predict(model, val_loader, DEVICE)
        val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save({
                "state_dict": model.state_dict(),
                "X_mean": X_mean,
                "X_std": X_std,
            }, MODEL_DIR / "mlp_best.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_rmse={val_rmse:.2f}m")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best val_rmse={best_val_rmse:.2f}m)")
            break

    # Load best model and evaluate
    checkpoint = torch.load(MODEL_DIR / "mlp_best.pt", weights_only=False, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    y_pred_train = predict(model, train_loader, DEVICE)
    y_pred_val = predict(model, val_loader, DEVICE)
    y_pred_test = predict(model, test_loader, DEVICE)

    train_metrics = evaluate(y_train, y_pred_train)
    val_metrics = evaluate(y_val, y_pred_val)
    test_metrics = evaluate(y_test, y_pred_test)

    print(f"\n{'=' * 60}")
    print(f"Train: MAE={train_metrics['mae']:.2f}m  RMSE={train_metrics['rmse']:.2f}m  R²={train_metrics['r2']:.3f}")
    print(f"Val:   MAE={val_metrics['mae']:.2f}m  RMSE={val_metrics['rmse']:.2f}m  R²={val_metrics['r2']:.3f}")
    print(f"Test:  MAE={test_metrics['mae']:.2f}m  RMSE={test_metrics['rmse']:.2f}m  R²={test_metrics['r2']:.3f}")

    # Save test predictions
    pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred_test,
    }).to_parquet(MODEL_DIR / "mlp_test_predictions.parquet", index=False)

    # Save metrics
    pd.DataFrame([
        {"split": "train", **train_metrics},
        {"split": "val", **val_metrics},
        {"split": "test", **test_metrics},
    ]).to_csv(MODEL_DIR / "mlp_metrics.csv", index=False)

    print(f"\nBest model saved to {MODEL_DIR / 'mlp_best.pt'}")
    print(f"Test predictions saved to {MODEL_DIR / 'mlp_test_predictions.parquet'}")

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (name, yt, yp, m) in zip(axes, [
        ("Train", y_train, y_pred_train, train_metrics),
        ("Val", y_val, y_pred_val, val_metrics),
        ("Test", y_test, y_pred_test, test_metrics),
    ]):
        ax.scatter(yt, yp, s=1, alpha=0.1, c="forestgreen", rasterized=True)
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

    plt.suptitle("Fine-tuned MLP — Google Embeddings → CHM (Spatial Split)", fontsize=14)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "mlp_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {MODEL_DIR / 'mlp_results.png'}")


if __name__ == "__main__":
    main()
