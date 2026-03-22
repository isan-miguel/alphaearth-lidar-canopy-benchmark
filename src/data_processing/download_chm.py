"""
Download Reis/Jackson Canopy Height Models from Zenodo.
Source: https://zenodo.org/records/7104044
487 transects across the Brazilian Amazon, 1m resolution GeoTIFFs.
"""

import requests
import zipfile
import os
import sys
from pathlib import Path

from src.config import DATA_DIR


ZENODO_RECORD = "7104044"
OUT_DIR = DATA_DIR / "chm_reis_jackson"
PARTS = [
    "CanopyHeightModels_part1.zip",
    "CanopyHeightModels_part2.zip",
    "CanopyHeightModels_part3.zip",
    "CanopyHeightModels_part4.zip",
    "CanopyHeightModels_part5.zip",
    "CanopyHeightModels_part6.zip",
    "CanopyHeightModels_part7.zip",
]


def download_and_extract(filename, out_dir):
    """Download a zip from Zenodo and extract it."""
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{filename}/content"
    zip_path = out_dir / filename

    # Check if already extracted
    part_name = filename.replace(".zip", "")
    part_dir = out_dir / part_name
    if part_dir.exists() and any(part_dir.glob("*.tif")):
        n_tifs = len(list(part_dir.glob("*.tif")))
        print(f"  {filename}: already extracted ({n_tifs} TIFs), skipping")
        return

    print(f"  Downloading {filename}...")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))

    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r    {downloaded/1e9:.1f}/{total/1e9:.1f} GB ({pct:.0f}%)", end="")
    print()

    print(f"  Extracting {filename}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    os.remove(zip_path)
    n_tifs = len(list(part_dir.glob("*.tif")))
    print(f"  Done: {n_tifs} TIFs extracted")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Also download summary CSV
    csv_url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/Reis_Jackson_2022_summary_data.csv/content"
    csv_path = OUT_DIR / "summary_data.csv"
    if not csv_path.exists():
        print("Downloading summary CSV...")
        r = requests.get(csv_url, timeout=15)
        csv_path.write_bytes(r.content)

    # Download parts (skip already downloaded)
    parts_to_download = PARTS
    if len(sys.argv) > 1:
        # Allow specifying specific parts, e.g.: python download_chm.py 1 3 5
        indices = [int(x) - 1 for x in sys.argv[1:]]
        parts_to_download = [PARTS[i] for i in indices]

    print(f"Downloading {len(parts_to_download)} parts to {OUT_DIR}/")
    for part in parts_to_download:
        download_and_extract(part, OUT_DIR)

    # Final count
    all_tifs = list(OUT_DIR.glob("**/*.tif"))
    print(f"\nTotal CHM GeoTIFFs: {len(all_tifs)}")


if __name__ == "__main__":
    main()
