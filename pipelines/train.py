"""Training pipeline (T stage): embeddings -> country heads, judged by honest baselines.

The backbone stays frozen; only heads train, on the 512-d vectors in the store.
Split is BY QUADTREE CELL (group split), never by row: two frames of the same
street must land on the same side or accuracy is a lie.

Contenders (all CPU-cheap on vectors):
  centroid   nearest class-mean on the unit sphere
  logreg     multinomial logistic regression
  mlp        one-hidden-layer MLP

Baselines the winner must beat:
  majority   always predict the most common country
  zeroshot   CLIP text prompts 'a photo taken in X' (no training at all) --
             the embarrassing one to lose to.

Registers the champion with eval images; writes the head leaderboard FG.

    python pipelines/train.py
"""
import glob
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def _find_root():
    cand = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [cand] + sorted(glob.glob("/hopsfs/Users/*/where-on-earth")):
        if os.path.exists(os.path.join(p, "embed_features.py")):
            return p
    raise RuntimeError("repo root not found")

ROOT = _find_root()
sys.path.insert(0, ROOT)
from pipelines.insert_fg import load_parquets                     # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "models", "eval")
MODEL_NAME = "geo_country"
LEADERBOARD_FG = "geo_leaderboard"
SEED = 42
MIN_PER_COUNTRY = 80      # below this a class cannot be learned or fairly scored
TEST_CELL_FRAC = 0.15


def load_xy():
    df = load_parquets("train")
    counts = df["country"].value_counts()
    keep = counts[counts >= MIN_PER_COUNTRY].index
    dropped = counts[counts < MIN_PER_COUNTRY]
    print(f"{len(df):,} rows; keeping {len(keep)} countries, dropping "
          f"{len(dropped)} rare ones covering {dropped.sum():,} rows "
          f"({sorted(dropped.index[:12].tolist())}...)")
    df = df[df["country"].isin(keep)].reset_index(drop=True)

    # group split by quadtree cell: same street never on both sides
    rng = np.random.default_rng(SEED)
    cells = df["cell"].unique()
    test_cells = set(rng.choice(cells, size=int(len(cells) * TEST_CELL_FRAC),
                                replace=False))
    is_test = df["cell"].isin(test_cells).values
    X = np.stack(df["emb"].values).astype(np.float32)
    y = df["country"].values
    print(f"split: {(~is_test).sum():,} train / {is_test.sum():,} test "
          f"({len(test_cells)} test cells)")
    return X[~is_test], y[~is_test], X[is_test], y[is_test], df.loc[is_test]


def top_k(proba, classes, y, k):
    idx = np.argsort(-proba, axis=1)[:, :k]
    hits = (classes[idx] == y[:, None]).any(axis=1)
    return float(hits.mean())


def eval_head(name, clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    classes = np.asarray(clf.classes_)
    m = {"config": name,
         "top1": top_k(proba, classes, yte, 1),
         "top5": top_k(proba, classes, yte, 5)}
    print(f"  {name:9s} top1 {m['top1']:.4f}  top5 {m['top5']:.4f}", flush=True)
    return m, clf


class Centroid:
    """Nearest class-mean on the unit sphere; predict_proba = softmax of cosine."""
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        M = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        self.M_ = M / np.linalg.norm(M, axis=1, keepdims=True)
        return self
    def predict_proba(self, X):
        s = X @ self.M_.T
        e = np.exp((s - s.max(axis=1, keepdims=True)) * 20.0)
        return e / e.sum(axis=1, keepdims=True)


def baselines(yte, Xte, ytr):
    out = []
    # majority
    top = pd.Series(ytr).value_counts().index[:5].tolist()
    out.append({"config": "majority",
                "top1": float((yte == top[0]).mean()),
                "top5": float(np.isin(yte, top).mean())})
    # CLIP zero-shot from stored text embeddings
    txt = pd.read_parquet(os.path.join(DATA, "country_text_emb.parquet"))
    T = np.stack(txt["emb"].values)
    classes = txt["country"].values
    proba = Xte @ T.T
    out.append({"config": "clip_zeroshot",
                "top1": top_k(proba, classes, yte, 1),
                "top5": top_k(proba, classes, yte, 5)})
    for m in out:
        print(f"  {m['config']:13s} top1 {m['top1']:.4f}  top5 {m['top5']:.4f}", flush=True)
    return out


def plots(champ_name, clf, Xte, yte, df_test):
    os.makedirs(OUT, exist_ok=True)
    proba = clf.predict_proba(Xte)
    classes = np.asarray(clf.classes_)
    pred = classes[np.argmax(proba, axis=1)]

    # per-country accuracy on a lon/lat scatter (a world map with no map library)
    acc = pd.DataFrame({"country": yte, "ok": pred == yte}).groupby("country")["ok"].agg(["mean", "size"])
    pos = df_test.groupby("country")[["longitude", "latitude"]].mean()
    j = acc.join(pos).dropna()
    plt.figure(figsize=(10, 5))
    sc = plt.scatter(j["longitude"], j["latitude"], c=j["mean"], s=np.sqrt(j["size"]) * 3,
                     cmap="RdYlGn", vmin=0, vmax=1, alpha=0.85, edgecolors="none")
    plt.colorbar(sc, label="top-1 accuracy")
    plt.title(f"Where the model knows the world ({champ_name})")
    plt.xlabel("longitude"); plt.ylabel("latitude")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "accuracy_map.png"), dpi=120); plt.close()

    # best and worst countries (with enough test rows)
    big = j[j["size"] >= 30].sort_values("mean")
    sel = pd.concat([big.head(12), big.tail(12)])
    plt.figure(figsize=(7, 7))
    colors = ["#ef4444"] * 12 + ["#22c55e"] * 12
    plt.barh(sel.index, sel["mean"], color=colors)
    plt.xlabel("top-1 accuracy"); plt.title("Hardest and easiest countries")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "best_worst.png"), dpi=120); plt.close()

    json.dump({c: round(float(a), 4) for c, a in j["mean"].items()},
              open(os.path.join(OUT, "per_country_acc.json"), "w"), indent=0)


def main():
    import hopsworks
    import shutil
    proj = hopsworks.login()
    Xtr, ytr, Xte, yte, df_test = load_xy()

    print("\nbaselines:", flush=True)
    base = baselines(yte, Xte, ytr)

    print("heads:", flush=True)
    heads = {
        "centroid": Centroid(),
        "logreg": LogisticRegression(max_iter=2000, C=10.0, n_jobs=-1),
        # early_stopping=True breaks on string labels in this sklearn (isnan on str
        # y_pred); plateau-based n_iter_no_change stopping works fine without it.
        "mlp": MLPClassifier(hidden_layer_sizes=(512,), max_iter=30,
                             n_iter_no_change=3, random_state=SEED),
    }
    results, fitted = [], {}
    for name, clf in heads.items():
        m, f = eval_head(name, clf, Xtr, ytr, Xte, yte)
        results.append(m); fitted[name] = f

    champ = max(results, key=lambda m: m["top1"])["config"]
    zs = next(m for m in base if m["config"] == "clip_zeroshot")
    champ_m = next(m for m in results if m["config"] == champ)
    print(f"\nchampion: {champ} top1 {champ_m['top1']:.4f} "
          f"(zero-shot {zs['top1']:.4f}, lift {(champ_m['top1']-zs['top1'])*100:+.1f} pts)")

    plots(champ, fitted[champ], Xte, yte, df_test)

    # leaderboard FG
    lb = pd.DataFrame(base + results)
    lb["run_time"] = pd.Timestamp.utcnow()
    fs = proj.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=LEADERBOARD_FG, version=1,
        description="Head-vs-baseline leaderboard for country-from-image "
                    "(top-1/top-5 on a cell-grouped held-out split).",
        primary_key=["config"], event_time="run_time", online_enabled=False)
    fg.insert(lb, write_options={"wait_for_job": False})

    # register champion (refit on ALL rows for serving)
    Xall = np.concatenate([Xtr, Xte]); yall = np.concatenate([ytr, yte])
    final = fitted[champ]
    final.fit(Xall, yall)
    os.makedirs(OUT, exist_ok=True)
    joblib.dump(final, os.path.join(OUT, "model.joblib"))
    shutil.copy(os.path.join(DATA, "iso2name.json"), os.path.join(OUT, "iso2name.json"))
    shutil.copy(os.path.join(DATA, "country_text_emb.parquet"),
                os.path.join(OUT, "country_text_emb.parquet"))
    assets = os.path.join(ROOT, "assets"); os.makedirs(assets, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.png")) + glob.glob(os.path.join(OUT, "*.json")):
        shutil.copy(f, assets)

    metrics = {**{f"head_{k}": v for k, v in champ_m.items() if k != "config"},
               "zeroshot_top1": zs["top1"], "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
               "n_countries": int(len(np.unique(yall)))}
    mr = proj.get_model_registry()
    m = mr.python.create_model(
        name=MODEL_NAME, metrics=metrics,
        description=f"Country-from-photo head ({champ}) on frozen CLIP ViT-B/32 "
                    f"embeddings; cell-grouped split; beats CLIP zero-shot by "
                    f"{(champ_m['top1']-zs['top1'])*100:+.1f} pts top-1.")
    m.save(OUT)
    print(f"registered {MODEL_NAME} v{m.version}")


if __name__ == "__main__":
    main()
