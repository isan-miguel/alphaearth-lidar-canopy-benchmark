"""Centralized configuration for the CHM comparison project.

All shared constants and paths are defined here. The GEE project ID
is read from the environment variable GEE_PROJECT to avoid hardcoding
personal credentials.
"""

import os
from pathlib import Path

# ── Project root (two levels up from this file) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Google Earth Engine ──
GEE_PROJECT = os.environ.get("GEE_PROJECT", "your-gee-project-id")

# ── Paths ──
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
AOI_PATH = PROJECT_ROOT / "AOI.geojson"

# ── Embedding constants ──
EMBED_DIM = 64
EMBEDDING_COLS = [f"A{i:02d}" for i in range(EMBED_DIM)]
EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBEDDING_YEAR = "2017"  # EBA campaign was 2016-2018

# ── Target / filtering ──
TARGET_COL = "chm"
MAX_CHM = 60  # meters — outlier cap

# ── Spatial grid ──
SCALE = 30  # target resolution in meters
RES_DEG = 1.0 / 111.0  # ~1 km in degrees at tropical latitudes
MIN_COVERAGE = 0.90  # minimum valid-pixel fraction per grid cell

# ── Training ──
SEED = 42
N_FOLDS = 10
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
