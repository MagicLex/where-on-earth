"""KServe predictor: photo in, country out.

The artifact bundles the trained head + iso2name + the SAME embed_features module
the pipeline used, so the online vector is computed by the identical code path
(frozen CLIP ViT-B/32, L2-normalized). The backbone weights download from the HF
hub once at pod start and live in the pod's cache after that.

    predict(inputs=[{"url": "https://..."}])          -> top-5 countries
    predict(inputs=[{"b64": "<base64 jpeg bytes>"}])  -> top-5 countries
"""
import base64
import io
import os
import sys

import joblib
import numpy as np

_PATH = os.environ.get("MODEL_FILES_PATH", "/mnt/models")
sys.path.insert(0, _PATH)
from embed_features import embed_images                       # noqa: E402

TOP_K = 5


class Predict:
    def __init__(self):
        import json
        self.head = joblib.load(os.path.join(_PATH, "model.joblib"))
        self.iso2name = json.load(open(os.path.join(_PATH, "iso2name.json")))
        self._session = None
        # warm the backbone at boot, not on the first user request
        from PIL import Image
        embed_images([Image.new("RGB", (64, 64))], batch_size=1)

    def _sess(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "where-on-earth/1.0"})
        return self._session

    def _to_image(self, item):
        from PIL import Image
        if item.get("b64"):
            raw = base64.b64decode(item["b64"])
        elif item.get("url"):
            raw = self._sess().get(item["url"], timeout=20).content
        else:
            return None
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((512, 512))
        return img

    def predict(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs.get("instances") or inputs.get("inputs") or []
        images, ok = [], []
        for item in inputs:
            img = self._to_image(item if isinstance(item, dict) else {})
            ok.append(img is not None)
            if img is not None:
                images.append(img)
        if not images:
            return [{"error": "no decodable image"} for _ in inputs]
        vecs = embed_images(images, batch_size=8)
        proba = self.head.predict_proba(vecs)
        classes = np.asarray(self.head.classes_)
        results, vi = [], 0
        for valid in ok:
            if not valid:
                results.append({"error": "no decodable image"})
                continue
            p = proba[vi]; vi += 1
            idx = np.argsort(-p)[:TOP_K]
            results.append({"guesses": [
                {"code": str(classes[i]),
                 "country": self.iso2name.get(str(classes[i]), str(classes[i])),
                 "p": round(float(p[i]), 4)} for i in idx]})
        return results
