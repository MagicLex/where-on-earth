"""One-shot: CLIP text embeddings of 'a photo taken in <country>' for all 222
countries -> data/country_text_emb.parquet.

This powers the zero-shot baseline: cosine(image_emb, text_emb) with no training.
Runs in the torch job env; the training job (sklearn env, no torch) then only needs
a numpy matmul against the stored vectors.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

def _find_root():
    cand = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [cand] + sorted(glob.glob("/hopsfs/Users/*/where-on-earth")):
        if os.path.exists(os.path.join(p, "embed_features.py")):
            return p
    raise RuntimeError("repo root not found")

ROOT = _find_root()
sys.path.insert(0, ROOT)
from embed_features import MODEL_ID                             # noqa: E402

DATA = os.path.join(ROOT, "data")


def main():
    from transformers import CLIPModel, CLIPTokenizerFast
    iso2name = json.load(open(os.path.join(DATA, "iso2name.json")))
    codes = sorted(iso2name)
    prompts = [f"a photo taken in {iso2name[c]}" for c in codes]
    model = CLIPModel.from_pretrained(MODEL_ID)
    model.eval()
    tok = CLIPTokenizerFast.from_pretrained(MODEL_ID)
    with torch.no_grad():
        t = tok(prompts, padding=True, return_tensors="pt")
        f = model.get_text_features(**t)
        if not isinstance(f, torch.Tensor):
            f = f.pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
    df = pd.DataFrame({"country": codes, "emb": list(f.numpy().astype(np.float32))})
    df.to_parquet(os.path.join(DATA, "country_text_emb.parquet"))
    print(f"embedded {len(df)} country prompts", flush=True)


if __name__ == "__main__":
    main()
