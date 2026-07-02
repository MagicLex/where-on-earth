"""Feedback pipeline: played rounds -> labeled training rows. The flywheel's gear 2.

The app appends (image, true country, model guess) under data/feedback/ every time
a round is revealed or an upload is corrected. This job embeds those images with
the SAME shared module (no skew, as always), inserts them into the geo_feedback
feature group, and archives the processed files. Schedule it daily; the retrain
job picks the rows up from the FG.

    python pipelines/feedback_pipeline.py
"""
import glob
import json
import os
import shutil
import sys

import pandas as pd
from PIL import Image

def _find_root():
    cand = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [cand] + sorted(glob.glob("/hopsfs/Users/*/where-on-earth")):
        if os.path.exists(os.path.join(p, "embed_features.py")):
            return p
    raise RuntimeError("repo root not found")

ROOT = _find_root()
sys.path.insert(0, ROOT)
from embed_features import embed_images                       # noqa: E402

FB = os.path.join(ROOT, "data", "feedback")
ARCHIVE = os.path.join(FB, "processed")
FG_NAME = "geo_feedback"


def main():
    import hopsworks
    jl = os.path.join(FB, "feedback.jsonl")
    if not os.path.exists(jl):
        print("no feedback yet")
        return
    rows = [json.loads(l) for l in open(jl)]
    rows = [r for r in rows if r.get("true_country")]
    if not rows:
        print("no labeled feedback yet")
        return

    imgs, kept = [], []
    for r in rows:
        p = os.path.join(FB, f"{r['id']}.jpg")
        if not os.path.exists(p):
            continue
        imgs.append(Image.open(p).convert("RGB"))
        kept.append(r)
    if not imgs:
        print("no images to embed")
        return
    vecs = embed_images(imgs, batch_size=32)

    df = pd.DataFrame(kept)
    df["emb"] = list(vecs)
    df["event_time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.rename(columns={"true_country": "country"})[
        ["id", "country", "source", "pred_top1", "correct", "emb", "event_time"]]

    proj = hopsworks.login()
    fs = proj.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FG_NAME, version=1,
        description="Human-labeled rounds from the app (playset reveals + upload "
                    "corrections), embedded with the shared CLIP module. Extra "
                    "gold rows for the next head retrain.",
        primary_key=["id"], event_time="event_time", online_enabled=False)
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"ingested {len(df)} feedback rows into {FG_NAME}")

    # archive so the next run only sees new rounds
    os.makedirs(ARCHIVE, exist_ok=True)
    for r in kept:
        src = os.path.join(FB, f"{r['id']}.jpg")
        if os.path.exists(src):
            shutil.move(src, os.path.join(ARCHIVE, f"{r['id']}.jpg"))
    os.rename(jl, os.path.join(ARCHIVE, f"feedback_{int(df['event_time'].max().timestamp())}.jsonl"))


if __name__ == "__main__":
    main()
