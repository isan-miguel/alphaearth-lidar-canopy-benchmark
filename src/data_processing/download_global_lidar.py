"""
Download airborne LiDAR point cloud tiles for canopy height analysis.

Downloads one tile per region from open national LiDAR programs:
  - USA:        USGS 3DEP via Microsoft Planetary Computer (COPC format, no auth)
  - Finland:    National Land Survey via open file service (LAZ)
  - Borneo:     Zenodo — Milodowski et al. 2021, NERC ALS over SAFE/Danum (LAS)
  - BC, Canada: LidarBC portal (manual)
  - Philippines: Phil-LiDAR / LiPAD portal (manual)

No authentication is required for USA or Finland sources.

Usage:
    python -m src.data_processing.download_global_lidar
    python -m src.data_processing.download_global_lidar usa_washington
    python -m src.data_processing.download_global_lidar finland
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import requests

from src.config import GLOBAL_SAMPLES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Regions with open airborne LiDAR ──────────────────────────────────────
# bbox: (west, south, east, north)

SAMPLE_REGIONS = {
    "usa_washington": {
        "bbox": (-122.25, 46.70, -122.15, 46.80),
        "description": "USGS 3DEP — temperate conifer, Gifford Pinchot NF, WA",
        "source": "usgs_3dep",
    },
    "finland": {
        "bbox": (25.00, 61.80, 25.10, 61.90),
        "description": "NLS Finland — boreal forest, central Finland",
        "source": "nls_finland",
    },
    "borneo_sabah": {
        "bbox": (116.95, 4.62, 117.85, 5.05),
        "description": "Zenodo — tropical rainforest, SAFE/Danum/Maliau, Sabah, Borneo",
        "source": "zenodo_borneo",
    },
    "canada_bc": {
        "bbox": (-123.25, 49.30, -123.15, 49.40),
        "description": "BC LiDAR — temperate rainforest, North Shore Mountains, BC",
        "source": "bc_lidar",
    },
    "philippines": {
        "bbox": (121.00, 14.05, 121.10, 14.15),
        "description": "Phil-LiDAR — tropical, Laguna de Bay area",
        "source": "phil_lidar",
    },
}

TIMEOUT = 30  # seconds for initial connection; streaming reads have no timeout


# ── Shared helpers ────────────────────────────────────────────────────────

def _download_file(url, out_path, label=""):
    """Stream-download a file with progress reporting."""
    log.info("  Downloading %s ...", label or url)
    resp = requests.get(url, stream=True, timeout=TIMEOUT)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(
                    f"\r  {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)",
                    end="", flush=True,
                )
    if total:
        print()
    log.info("  Saved %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)
    return out_path


def _check_cached(out_dir, extensions=(".laz", ".copc.laz", ".las")):
    """Return first existing point cloud file in out_dir, or None."""
    for ext in extensions:
        for f in out_dir.glob(f"*{ext}"):
            return f
    return None


# ── Source: USGS 3DEP via Planetary Computer STAC ─────────────────────────

STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
STAC_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"


def fetch_usgs_3dep(bbox, out_dir):
    """Search Planetary Computer STAC for a 3DEP COPC tile and download it."""
    log.info("  Searching Planetary Computer STAC for 3DEP COPC tiles ...")
    resp = requests.post(
        STAC_SEARCH_URL,
        json={
            "collections": ["3dep-lidar-copc"],
            "bbox": list(bbox),
            "limit": 1,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        log.warning("  No 3DEP COPC tiles found for bbox %s", bbox)
        return None

    item = features[0]
    asset_href = item["assets"]["data"]["href"]
    item_id = item.get("id", "unknown")
    log.info("  Found tile: %s", item_id)

    # Sign the URL (Planetary Computer requires SAS token for blob access)
    sign_resp = requests.get(
        STAC_SIGN_URL, params={"href": asset_href}, timeout=TIMEOUT,
    )
    sign_resp.raise_for_status()
    signed_href = sign_resp.json()["href"]

    filename = Path(asset_href).name
    out_path = out_dir / filename
    _download_file(signed_href, out_path, label=filename)
    return out_path


# ── Source: Finland NLS open laser data ───────────────────────────────────

KAPSI_BASE = "https://kartat.kapsi.fi/files/laser/etrs-tm35fin-n2000/mara_2m"


def fetch_nls_finland(bbox, out_dir):
    """Download one LAZ tile from the NLS Finland open data Kapsi.fi mirror.

    Directory structure: {KAPSI_BASE}/{year}/{sheet}/{batch}/{tile}.laz
    Tiles starting with 'K' are roughly 61-62°N latitude.
    """
    import re

    log.info("  Browsing Kapsi.fi mirror for Finnish laser tiles ...")

    # Scan recent years for tiles in the K-sheet area (61-62°N)
    for year in range(2022, 2007, -1):
        year_url = f"{KAPSI_BASE}/{year}/"
        try:
            resp = requests.get(year_url, timeout=TIMEOUT)
            if resp.status_code != 200:
                continue
        except requests.RequestException:
            continue

        # Find sheet directories starting with 'K' (61-62°N)
        sheets = re.findall(r'href="(K[0-9]+)/"', resp.text)
        if not sheets:
            continue

        # Pick the first K-sheet and drill into it
        sheet = sheets[0]
        sheet_url = f"{year_url}{sheet}/"
        try:
            resp2 = requests.get(sheet_url, timeout=TIMEOUT)
            batches = re.findall(r'href="([^"]+)/"', resp2.text)
            batches = [b for b in batches if b != ".."]
        except requests.RequestException:
            continue
        if not batches:
            continue

        batch_url = f"{sheet_url}{batches[0]}/"
        try:
            resp3 = requests.get(batch_url, timeout=TIMEOUT)
            laz_files = re.findall(r'href="([^"]+\.laz)"', resp3.text)
        except requests.RequestException:
            continue
        if not laz_files:
            continue

        # Download the first LAZ tile
        tile_name = laz_files[0]
        tile_url = f"{batch_url}{tile_name}"
        out_path = out_dir / tile_name
        _download_file(tile_url, out_path, label=tile_name)
        return out_path

    log.warning("  Could not find a Finland LAZ tile on Kapsi.fi")
    return None


# ── Source: Zenodo — Borneo ALS (Milodowski et al. 2021) ──────────────────

BORNEO_ZENODO_RECORD = "4572775"
BORNEO_FILENAME = "Carbon_plot_point_cloud_buffer.las"


def fetch_zenodo_borneo(bbox, out_dir):
    """Download ALS point cloud from Zenodo record 4572775.

    Milodowski et al. (2021) — NERC Airborne Research Facility survey
    over SAFE project, Maliau Basin, and Danum Valley in Sabah, Malaysian Borneo.
    ~1.35 GB LAS file covering tropical rainforest with varying degradation.
    """
    url = (
        f"https://zenodo.org/api/records/{BORNEO_ZENODO_RECORD}"
        f"/files/{BORNEO_FILENAME}/content"
    )
    out_path = out_dir / BORNEO_FILENAME
    _download_file(url, out_path, label=BORNEO_FILENAME)
    return out_path


# ── Source: BC LiDAR (Canada) ─────────────────────────────────────────────


def fetch_bc_lidar(bbox, out_dir):
    """Try to download a LAZ tile for British Columbia.

    BC LiDAR data is available at https://lidar.gov.bc.ca/ but requires
    manual download through their web portal. As a fallback, we try USGS 3DEP
    via Planetary Computer which sometimes has cross-border coverage.
    """
    # Try 3DEP first — it occasionally covers the US-Canada border area
    log.info("  Trying USGS 3DEP for cross-border coverage near BC ...")
    result = fetch_usgs_3dep(bbox, out_dir)
    if result:
        return result

    log.warning(
        "  BC LiDAR requires manual download:\n"
        "    1. Go to https://lidar.gov.bc.ca/\n"
        "    2. Navigate to bbox %s on the map\n"
        "    3. Select and download LAZ tiles to %s",
        bbox, out_dir,
    )
    return None


# ── Source: Philippines Phil-LiDAR ────────────────────────────────────────

def fetch_phil_lidar(bbox, out_dir):
    """Try to download Philippines LiDAR data.

    The Taal Open LiDAR dataset is publicly hosted. Other Phil-LiDAR data
    requires manual download from https://lipad.dream.upd.edu.ph/
    """
    log.info("  Checking Taal Open LiDAR (GitHub-hosted) ...")

    # Taal Open LiDAR has data for the Taal Volcano / Laguna area.
    # The dataset metadata is at https://phillidar-dad.github.io/taal-open-lidar.html
    # Actual LAZ files are hosted on a DREAM server.
    taal_base = "https://lipad.dream.upd.edu.ph"

    # Since the exact download URLs require session auth on LiPAD,
    # we note this as requiring manual download.
    log.warning(
        "  Philippines LiDAR requires manual download from LiPAD:\n"
        "    1. Create account at %s\n"
        "    2. Search for data near bbox %s\n"
        "    3. Download LAZ files to %s",
        taal_base, bbox, out_dir,
    )
    return None


# ── Source dispatch ───────────────────────────────────────────────────────

SOURCE_HANDLERS = {
    "usgs_3dep": fetch_usgs_3dep,
    "nls_finland": fetch_nls_finland,
    "zenodo_borneo": fetch_zenodo_borneo,
    "bc_lidar": fetch_bc_lidar,
    "phil_lidar": fetch_phil_lidar,
}


# ── Main ──────────────────────────────────────────────────────────────────

def process_region(region_name, region_info, base_dir):
    """Search and download one tile for a region. Returns manifest entry."""
    bbox = region_info["bbox"]
    source = region_info["source"]
    handler = SOURCE_HANDLERS.get(source)

    if handler is None:
        log.error("[%s] Unknown source: %s", region_name, source)
        return {"status": "error", "message": f"Unknown source: {source}"}

    out_dir = base_dir / region_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    cached = _check_cached(out_dir)
    if cached:
        log.info("[%s] Already have %s, skipping", region_name, cached.name)
        return {
            "status": "cached",
            "file": str(cached),
            "size_mb": round(cached.stat().st_size / 1e6, 1),
        }

    log.info("[%s] Source: %s — %s", region_name, source, region_info["description"])
    try:
        result_path = handler(bbox, out_dir)
    except Exception as exc:
        log.error("[%s] Download failed: %s", region_name, exc)
        return {"status": "error", "message": str(exc)}

    if result_path is None:
        return {"status": "not_available"}

    return {
        "status": "downloaded",
        "file": str(result_path),
        "size_mb": round(result_path.stat().st_size / 1e6, 1),
    }


def main():
    # Allow filtering to specific regions via CLI args
    regions = SAMPLE_REGIONS
    if len(sys.argv) > 1:
        names = sys.argv[1:]
        regions = {k: v for k, v in SAMPLE_REGIONS.items() if k in names}
        if not regions:
            log.error("Unknown region(s): %s. Available: %s",
                      names, list(SAMPLE_REGIONS.keys()))
            return

    manifest = {
        "download_date": datetime.now().isoformat(),
        "regions": {},
    }

    for region_name, region_info in regions.items():
        entry = {
            "bbox": list(region_info["bbox"]),
            "description": region_info["description"],
            "source": region_info["source"],
        }
        result = process_region(region_name, region_info, GLOBAL_SAMPLES_DIR)
        entry.update(result)
        manifest["regions"][region_name] = entry

    # Write manifest
    manifest_path = GLOBAL_SAMPLES_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest saved to %s", manifest_path)

    # Summary
    log.info("── Download summary ──")
    for name, entry in manifest["regions"].items():
        status = entry.get("status", "unknown")
        size = entry.get("size_mb", "")
        size_str = f" ({size} MB)" if size else ""
        log.info("  %-25s %s%s", name, status, size_str)


if __name__ == "__main__":
    main()
