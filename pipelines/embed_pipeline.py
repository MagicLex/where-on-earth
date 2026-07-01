"""Embed pipeline (F stage): OSV5M zip shards -> CLIP embeddings parquet.

Shard-parallel by design: each job instance takes a disjoint slice of the 98 train
zips (2.5 GB / ~52k images each), and for every image that survives the per-country
cap it computes the shared CLIP embedding. Images are deleted with the zip; only
vectors + labels leave the job. Several jobs run side by side on disjoint shards.

Per-country cap = greedy: rare countries keep everything they have in the processed
shards, over-represented ones (US alone is 24% of OSV5M) stop at the cap. Dropped
counts are logged -- no silent truncation.

    python pipelines/embed_pipeline.py --split train --shards 0-7 --cap 1500
    python pipelines/embed_pipeline.py --split test  --shards 0-1 --cap 150

Output: data/emb/<split>_shard_<i>.parquet (id, country, lat, lon, cell, emb[512]).
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import requests
from PIL import Image

# Job copies run from Resources/jobs/<name>/; anchor on the repo (see BLOCKERS).
import glob

def _find_root():
    cand = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [cand] + sorted(glob.glob("/hopsfs/Users/*/where-on-earth")):
        if os.path.exists(os.path.join(p, "embed_features.py")):
            return p
    raise RuntimeError("repo root with embed_features.py not found")

ROOT = _find_root()
sys.path.insert(0, ROOT)
from embed_features import embed_images, EMBED_DIM              # noqa: E402

DATA = os.path.join(ROOT, "data")
EMB = os.path.join(DATA, "emb")
BASE = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main/images"
SCRATCH = "/tmp"          # pod-local disk for the transient zip
BATCH = 64


def parse_shards(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def load_meta(split):
    meta = pd.read_parquet(os.path.join(DATA, f"meta_{split}.parquet"))
    meta = meta.dropna(subset=["country"])
    return meta.set_index("id")


def process_shard(split, shard, meta, kept_counter, cap, session):
    out_path = os.path.join(EMB, f"{split}_shard_{shard:02d}.parquet")
    if os.path.exists(out_path):
        done = pd.read_parquet(out_path)
        for c, n in done["country"].value_counts().items():
            kept_counter[c] += int(n)
        print(f"shard {shard}: already done ({len(done)} rows)", flush=True)
        return

    url = f"{BASE}/{split}/{shard:02d}.zip"
    zpath = os.path.join(SCRATCH, f"{split}_{shard:02d}.zip")
    t0 = time.time()
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"shard {shard}: downloaded {os.path.getsize(zpath)/1e9:.2f} GB "
          f"in {time.time()-t0:.0f}s", flush=True)

    rows, imgs, pend = [], [], []
    dropped_cap = dropped_meta = bad = 0
    t0 = time.time()
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".jpg")]
        for name in names:
            try:
                img_id = int(os.path.splitext(os.path.basename(name))[0])
            except ValueError:
                bad += 1
                continue
            try:
                m = meta.loc[img_id]
            except KeyError:
                dropped_meta += 1
                continue
            country = m["country"]
            if kept_counter[country] >= cap:
                dropped_cap += 1
                continue
            try:
                im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
            except Exception:
                bad += 1
                continue
            kept_counter[country] += 1
            imgs.append(im)
            pend.append((img_id, country, float(m["latitude"]), float(m["longitude"]),
                         str(m["quadtree_10_1000"])))
            if len(imgs) >= BATCH * 4:
                vecs = embed_images(imgs, BATCH)
                rows.extend([p + (v,) for p, v in zip(pend, list(vecs))])
                imgs, pend = [], []
                if len(rows) % 2560 == 0:
                    rate = len(rows) / (time.time() - t0)
                    print(f"  shard {shard}: {len(rows)} embedded ({rate:.1f}/s)", flush=True)
    if imgs:
        vecs = embed_images(imgs, BATCH)
        rows.extend([p + (v,) for p, v in zip(pend, list(vecs))])
    os.remove(zpath)

    df = pd.DataFrame(rows, columns=["id", "country", "latitude", "longitude", "cell", "emb"])
    os.makedirs(EMB, exist_ok=True)
    df.to_parquet(out_path)
    print(f"shard {shard}: kept {len(df)}, dropped {dropped_cap} (cap) "
          f"{dropped_meta} (no meta) {bad} (bad), {time.time()-t0:.0f}s embed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--shards", required=True)
    ap.add_argument("--cap", type=int, default=1500)
    a = ap.parse_args()
    shards = parse_shards(a.shards)
    meta = load_meta(a.split)
    print(f"split={a.split} shards={shards} cap={a.cap}/country "
          f"meta={len(meta):,} rows", flush=True)
    kept = Counter()
    s = requests.Session()
    for shard in shards:
        process_shard(a.split, shard, meta, kept, a.cap, s)
    print(json.dumps({"total_kept": sum(kept.values()),
                      "countries": len(kept)}), flush=True)


if __name__ == "__main__":
    main()
