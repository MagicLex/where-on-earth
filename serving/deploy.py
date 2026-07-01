"""Deploy the country-guesser as an always-on KServe endpoint.

The artifact bundles the trained head (model.joblib) + iso2name.json + the shared
embed_features.py, so the endpoint computes the identical CLIP vector at request
time. The backbone downloads from the HF hub at pod start (~600 MB, once).

Env: torch-inference-pipeline needs transformers for CLIP; if the base image lacks
it the pod will fail on import and we clone+pin (tools/build_envs.py pattern).
"""
import os
import shutil
import sys

def _find_root():
    import glob
    cand = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [cand] + sorted(glob.glob("/hopsfs/Users/*/where-on-earth")):
        if os.path.exists(os.path.join(p, "embed_features.py")):
            return p
    raise RuntimeError("repo root not found")

ROOT = _find_root()
sys.path.insert(0, ROOT)

CHAMPION = "geo_country"
SERVE_MODEL = "geo_country_serving"
DEPLOYMENT = "whereonearth"
SERVE_ENV = os.environ.get("SERVE_ENV", "torch-inference-pipeline")
_rel = __file__.split("/hopsfs/", 1)[1].rsplit("/", 1)[0] if "/hopsfs/" in __file__ else ""
PREDICTOR = f"/Projects/createnew/{_rel}/predictor.py" if _rel \
    else os.path.join(ROOT, "serving", "predictor.py")


def _schema():
    import pandas as pd
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    inp = Schema(pd.DataFrame({"url": ["https://example.com/photo.jpg"]}))
    out = Schema(pd.DataFrame({"code": ["FR"], "country": ["France"], "p": [0.42]}))
    return ModelSchema(input_schema=inp, output_schema=out)


def _bundle(proj, version=None):
    mr = proj.get_model_registry()
    if version is None:
        version = max(m.version for m in mr.get_models(CHAMPION))
    champ = mr.get_model(CHAMPION, version=version)
    src = champ.download()
    d = os.path.join(ROOT, "serving", "_artifact")
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in ("model.joblib", "iso2name.json"):
        shutil.copy(os.path.join(src, f), os.path.join(d, f))
    shutil.copy(os.path.join(ROOT, "embed_features.py"), os.path.join(d, "embed_features.py"))
    return d, version


def deploy_champion(proj):
    mr = proj.get_model_registry()
    d, version = _bundle(proj)
    sm = mr.python.create_model(SERVE_MODEL, model_schema=_schema(),
                                description=f"serving bundle of {CHAMPION} v{version}")
    sm.save(d)
    dep = sm.deploy(name=DEPLOYMENT, script_file=PREDICTOR, environment=SERVE_ENV)
    dep.start(await_running=900)
    return dep


if __name__ == "__main__":
    import hopsworks
    p = hopsworks.login()
    dep = deploy_champion(p)
    print("deployment:", dep.name, "| running:", dep.is_running())
    try:
        print("smoke:", dep.predict(inputs=[
            {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/"
                    "Paris_Night.jpg/640px-Paris_Night.jpg"}]))
    except Exception as e:
        print("smoke predict pending:", e)
