"""Build the play-set: real held-out OSV5M street-view images for the app's game.

The training images were deleted after embedding (by design), and OSV5M's original
Mapillary thumb URLs are expired signed CDN links. So the app's "play vs the model"
tab gets a small self-hosted set instead: one TEST zip shard, ~2 images per country,
resized to 640px, with ground truth and per-image creator attribution (OSV5M is
CC-BY-SA; ATTRIBUTION.json credits every author).

In-distribution matters: feeding the model random Commons photos (portraits, food,
aerials) made the game score meaningless -- it was trained on street view.

    python tools/make_playset.py --per-country 2 --max-total 320
"""
import argparse
import io
import json
import os
import sys
import zipfile
from collections import Counter

import pandas as pd
import requests
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "app", "playset")
ZIP_URL = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main/images/test/00.zip"
ZIP_LOCAL = os.path.join(DATA, "test_00.zip")


def main(per_country, max_total):
    meta = pd.read_csv(os.path.join(DATA, "test.csv"),
                       usecols=["id", "country", "latitude", "longitude",
                                "creator_username"]).dropna(subset=["country"])
    meta = meta.set_index("id")

    if not os.path.exists(ZIP_LOCAL):
        print("downloading test shard 0 (2.5 GB)...", flush=True)
        with requests.get(ZIP_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(ZIP_LOCAL, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)

    os.makedirs(OUT, exist_ok=True)
    kept = Counter()
    manifest, attribution = [], {}
    with zipfile.ZipFile(ZIP_LOCAL) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".jpg")]
        for name in names:
            if len(manifest) >= max_total:
                break
            try:
                img_id = int(os.path.splitext(os.path.basename(name))[0])
                m = meta.loc[img_id]
            except (ValueError, KeyError):
                continue
            country = m["country"]
            if kept[country] >= per_country:
                continue
            try:
                im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
            except Exception:
                continue
            im.thumbnail((640, 640))
            fn = f"{img_id}.jpg"
            im.save(os.path.join(OUT, fn), quality=85)
            kept[country] += 1
            manifest.append({"file": fn, "country": str(country),
                             "lat": float(m["latitude"]), "lon": float(m["longitude"])})
            attribution[fn] = str(m.get("creator_username") or "unknown")

    json.dump(manifest, open(os.path.join(OUT, "playset.json"), "w"), indent=0)
    json.dump({"license": "CC-BY-SA (OSV5M / Mapillary contributors)",
               "source": "https://huggingface.co/datasets/osv5m/osv5m",
               "creators": attribution},
              open(os.path.join(OUT, "ATTRIBUTION.json"), "w"), indent=0)
    os.remove(ZIP_LOCAL)
    print(f"playset: {len(manifest)} images, {len(kept)} countries -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-country", type=int, default=2)
    ap.add_argument("--max-total", type=int, default=320)
    a = ap.parse_args()
    main(a.per_country, a.max_total)
