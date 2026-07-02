"""Rescore pipeline: every played photo, re-scored by the CURRENTLY DEPLOYED model.

The flywheel's scoreboard. Each photo ever played (archived by the app under
data/feedback/) is sent back through the live endpoint after each retrain+deploy,
and the verdict lands in the geo_rescore FG keyed (photo id, model version). That
gives the per-photo trajectory the flywheel exists for: at which round did the
model start getting this one right, and per-version accuracy on the played set.

Scoring goes through the ENDPOINT, not a local model load: we measure what is
actually served, embedding path included. Run it after each `make serve`.

    python pipelines/rescore_pipeline.py
"""
import base64
import glob
import json
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
FB = os.path.join(ROOT, "data", "feedback")
FG_NAME = "geo_rescore"
DEPLOYMENT = "whereonearth"


def played_photos():
    """id -> (jpg path, true country) for every round ever played."""
    rows = {}
    for jl in glob.glob(os.path.join(FB, "feedback.jsonl")) + \
              glob.glob(os.path.join(FB, "processed", "feedback_*.jsonl")):
        for line in open(jl):
            r = json.loads(line)
            if not r.get("true_country"):
                continue
            for d in (FB, os.path.join(FB, "processed")):
                p = os.path.join(d, f"{r['id']}.jpg")
                if os.path.exists(p):
                    rows[r["id"]] = (p, r["true_country"])
                    break
    return rows


def main():
    import hopsworks
    photos = played_photos()
    if not photos:
        print("nothing played yet")
        return
    proj = hopsworks.login()
    dep = proj.get_model_serving().get_deployment(DEPLOYMENT)

    out = []
    version = None
    for pid, (path, truth) in photos.items():
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        res = dep.predict(inputs=[{"b64": b64}])["predictions"][0]
        if "guesses" not in res:
            continue
        version = int(res.get("model_version", 0))
        top = res["guesses"][0]
        out.append({"id": pid, "model_version": version, "true_country": truth,
                    "pred_top1": top["code"], "p_top1": top["p"],
                    "correct": top["code"] == truth,
                    "top5": bool(any(g["code"] == truth for g in res["guesses"]))})
    df = pd.DataFrame(out)
    df["event_time"] = pd.Timestamp.utcnow()
    acc = df["correct"].mean()
    print(f"model v{version}: {len(df)} played photos rescored, "
          f"top1 {acc:.3f}, top5 {df['top5'].mean():.3f}")

    fs = proj.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FG_NAME, version=1,
        description="Every played photo re-scored by each deployed model version "
                    "(via the live endpoint). The flywheel's scoreboard: per-photo "
                    "trajectory across retrains.",
        primary_key=["id", "model_version"], event_time="event_time",
        online_enabled=False)
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"-> {FG_NAME} (pk id+model_version, upsert-safe)")


if __name__ == "__main__":
    main()
