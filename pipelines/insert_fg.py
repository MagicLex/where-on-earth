"""Insert the embedded sample into the feature store.

The images are gone -- this is the whole point. What the store holds is the
transform: 512-d CLIP vectors + country label + quadtree cell (the geo unit train/
test splits group by, so the same street never leaks across the split).

One FG (`geo_image_embeddings`), one FV (`geo_country_fv`) selecting emb + label +
cell. Reads every data/emb/<split>_shard_*.parquet written by the embed jobs.

    python pipelines/insert_fg.py
"""
import glob
import os
import sys

import pandas as pd

def _find_root():
    cand = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [cand] + sorted(glob.glob("/hopsfs/Users/*/where-on-earth")):
        if os.path.exists(os.path.join(p, "embed_features.py")):
            return p
    raise RuntimeError("repo root not found")

ROOT = _find_root()
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data")
EMB = os.path.join(DATA, "emb")
FG_NAME = "geo_image_embeddings"
FV_NAME = "geo_country_fv"


def load_parquets(split):
    paths = sorted(glob.glob(os.path.join(EMB, f"{split}_shard_*.parquet")))
    if not paths:
        return None
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    df["split"] = split
    return df


def main():
    import hopsworks
    frames = [f for f in (load_parquets("train"), load_parquets("test")) if f is not None]
    df = pd.concat(frames, ignore_index=True)
    df["ingested_at"] = pd.Timestamp.utcnow()
    print(f"{len(df):,} embeddings ({df['country'].nunique()} countries), "
          f"splits: {df['split'].value_counts().to_dict()}")

    proj = hopsworks.login()
    fs = proj.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FG_NAME, version=1,
        description="CLIP ViT-B/32 embeddings (512-d, L2-normalized) of OSV5M "
                    "street-view images + country label + quadtree cell for "
                    "leak-free geo splits. The images themselves are not stored.",
        primary_key=["id"], event_time="ingested_at", online_enabled=False)
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"inserted into {FG_NAME} v1 (materializing async)")

    query = fg.select(["id", "emb", "country", "cell", "latitude", "longitude", "split"])
    fs.get_or_create_feature_view(
        name=FV_NAME, version=1, query=query, labels=["country"],
        description="Image embedding -> country. Split by quadtree cell, never by row.")
    print(f"feature view {FV_NAME} v1 ready")


if __name__ == "__main__":
    main()
