"""Stream OSV5M's train.csv (2.9 GB) once and keep only what we need.

Output: data/meta_train.parquet with id, latitude, longitude, country, and the
10-level quadtree cell (the geo unit we split train/test by, so the same street
never sits on both sides). ~5.1M rows, ~150 MB. test.csv gets the same treatment.

    python collect/slim_metadata.py
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
BASE = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main"
KEEP = ["id", "latitude", "longitude", "country", "quadtree_10_1000"]


def slim(split):
    """Parse the locally-downloaded CSV in chunks (a URL read of 2.9 GB buffers in
    RAM and OOMs the terminal pod -- curl to disk first, then chunk-parse)."""
    src = os.path.join(DATA, f"{split}.csv")
    out = os.path.join(DATA, f"meta_{split}.parquet")
    if os.path.exists(out):
        df = pd.read_parquet(out)
        print(f"{split}: cached {len(df)} rows")
        return df
    if not os.path.exists(src):
        raise SystemExit(f"{src} missing; curl it from {BASE}/{split}.csv first")
    chunks = []
    for i, ch in enumerate(pd.read_csv(src, usecols=KEEP, chunksize=500_000,
                                       dtype={"id": "int64", "country": "string"})):
        chunks.append(ch)
        print(f"  {split} chunk {i}: {sum(len(c) for c in chunks):,} rows", flush=True)
    df = pd.concat(chunks, ignore_index=True)
    os.makedirs(DATA, exist_ok=True)
    df.to_parquet(out)
    print(f"{split}: {len(df):,} rows -> {out}")
    return df


if __name__ == "__main__":
    tr = slim("train")
    te = slim("test")
    print("\ncountries in train:", tr["country"].nunique())
    print(tr["country"].value_counts().head(10))
