"""where-on-earth -- Streamlit front for the geoguessing endpoint.

Thin client: upload a photo (or pull a random geotagged one from Wikimedia Commons
and play against the model), the KServe endpoint embeds it with the same frozen
CLIP the training used and the head answers with a country distribution. The app
renders guesses, the honest leaderboard (including the zero-shot baseline the head
must beat), and the accuracy world map from training.
"""
import base64
import json
import random
from pathlib import Path

import hopsworks
import pandas as pd
import streamlit as st

DEPLOYMENT = "whereonearth"
LEADERBOARD_FG = "geo_leaderboard"
ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="where on earth", page_icon="globe_with_meridians",
                   layout="wide")


@st.cache_resource
def _project():
    return hopsworks.login()


@st.cache_resource
def _deployment():
    return _project().get_model_serving().get_deployment(DEPLOYMENT)


@st.cache_data(ttl=3600)
def get_leaderboard():
    fs = _project().get_feature_store()
    df = fs.get_feature_group(LEADERBOARD_FG, version=1).read()
    return df.sort_values("top1", ascending=False).reset_index(drop=True)


def guess(image_bytes):
    b64 = base64.b64encode(image_bytes).decode()
    out = _deployment().predict(inputs=[{"b64": b64}])
    return out["predictions"][0]


@st.cache_data
def load_playset():
    """Held-out OSV5M street-view images with ground truth (see tools/make_playset.py).

    In-distribution on purpose: the model was trained on street view, so the game
    is fair. Random Commons photos (portraits, food, aerials) made it look broken.
    """
    p = ROOT / "app" / "playset" / "playset.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def random_play_photo():
    ps = load_playset()
    if not ps:
        return None
    item = random.choice(ps)
    f = ROOT / "app" / "playset" / item["file"]
    return {"bytes": f.read_bytes(), "lat": item["lat"], "lon": item["lon"],
            "country": item["country"], "title": item["file"]}


def live_mapillary_photo():
    """Fresh worldwide street view, live from Mapillary (CC-BY-SA).

    Anchored on random playset points so the true country is known without
    reverse geocoding. Dense areas 500 on Mapillary's side ("reduce data"), so the
    bbox is tiny and a failed anchor just falls through to the next one.
    """
    import requests
    tok = st.secrets.get("MAPILLARY_TOKEN")
    ps = load_playset()
    if not tok or not ps:
        return None
    s = requests.Session()
    for item in random.sample(ps, min(5, len(ps))):
        lon, lat = item["lon"], item["lat"]
        bbox = f"{lon-0.004},{lat-0.004},{lon+0.004},{lat+0.004}"
        try:
            r = s.get("https://graph.mapillary.com/images", timeout=25,
                      params={"access_token": tok, "fields": "id,thumb_1024_url",
                              "bbox": bbox, "limit": 10})
            data = r.json().get("data", []) if r.status_code == 200 else []
            if not data:
                continue
            pick = random.choice(data)
            img = s.get(pick["thumb_1024_url"], timeout=25)
            if img.status_code != 200:
                continue
            return {"bytes": img.content, "lat": lat, "lon": lon,
                    "country": item["country"], "title": f"mapillary:{pick['id']}"}
        except Exception:
            continue
    return None


def log_feedback(image_bytes, true_country, guesses, source):
    """One played round = one labeled example. The flywheel's first gear.

    Appends the image + a jsonl row under data/feedback/ (HopsFS, survives the
    pod). A scheduled job embeds these with the SAME shared module and feeds them
    into the geo_feedback FG; the next retrain learns from every round played.
    See docs/FEEDBACK-LOOP.md.
    """
    import time
    import uuid
    fb = ROOT / "data" / "feedback"
    fb.mkdir(parents=True, exist_ok=True)
    fid = uuid.uuid4().hex
    (fb / f"{fid}.jpg").write_bytes(image_bytes)
    row = {"id": fid, "ts": int(time.time()), "source": source,
           "true_country": true_country,
           "pred_top1": guesses[0]["code"] if guesses else None,
           "correct": bool(guesses and guesses[0]["code"] == true_country)}
    with open(fb / "feedback.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


st.title("Where on earth was this taken?")
st.caption("Frozen CLIP eyes + a country head trained on OSV5M street view. "
           "It has never seen your photo, only 512 numbers describing it.")

tab_upload, tab_play, tab_honest = st.tabs(["Upload a photo", "Play vs the model",
                                            "Is it actually good?"])

with tab_upload:
    up = st.file_uploader("JPEG/PNG", type=["jpg", "jpeg", "png"])
    if up is not None:
        raw = up.read()
        col1, col2 = st.columns([1, 1])
        col1.image(raw, use_container_width=True)
        with col2, st.spinner("Looking at it..."):
            try:
                res = guess(raw)
                if "error" in res:
                    st.error(res["error"])
                else:
                    top = res["guesses"][0]
                    st.markdown(f"## {top['country']}  ({top['p']*100:.0f}%)")
                    st.bar_chart(pd.DataFrame(res["guesses"]).set_index("country")["p"])
                    st.caption("Nothing is uploaded anywhere but this project's own "
                               "endpoint; the image is embedded and discarded.")
                    st.session_state["last_upload"] = (raw, res["guesses"])
            except Exception as e:
                st.warning(f"endpoint unreachable: {e}")
    if "last_upload" in st.session_state:
        iso = json.loads((ROOT / "assets" / "iso2name.json").read_text())
        wrong = st.selectbox("Was it wrong? Tell it the real country (this trains "
                             "the next version)", ["-"] + sorted(iso.values()))
        if wrong != "-" and st.button("Teach it"):
            code = next(c for c, n in iso.items() if n == wrong)
            raw_b, gs = st.session_state.pop("last_upload")
            log_feedback(raw_b, code, gs, "upload")
            st.success("Logged. It goes into the next retrain.")

with tab_play:
    st.markdown("Real street view the model never trained on. Guess the country "
                "first, then see whether you beat it.")
    live = st.toggle("Live Mapillary (fresh worldwide imagery)",
                     value=False, help="Falls back to the held-out playset when "
                     "Mapillary is grumpy. Both are fair: ground truth is known.")
    if st.button("New photo"):
        st.session_state.pop("photo", None)
    if "photo" not in st.session_state or st.session_state.get("photo_live") != live:
        st.session_state["photo"] = (live and live_mapillary_photo()) or random_play_photo()
        st.session_state["photo_live"] = live
    photo = st.session_state.get("photo")
    if photo is None:
        st.info("playset missing; run tools/make_playset.py and redeploy.")
    else:
        c1, c2 = st.columns([1, 1])
        c1.image(photo["bytes"], use_container_width=True)
        with c2:
            if st.button("Reveal model guess + truth"):
                try:
                    res = guess(photo["bytes"])
                    if "guesses" in res:
                        hit = res["guesses"][0]["code"] == photo["country"]
                        for g in res["guesses"][:3]:
                            marker = " ✓" if g["code"] == photo["country"] else ""
                            st.markdown(f"- **{g['country']}** {g['p']*100:.0f}%{marker}")
                        st.markdown("**Model got it.**" if hit else
                                    f"**Model missed.** Truth: `{photo['country']}`")
                        src = "mapillary" if str(photo["title"]).startswith("mapillary:") \
                            else "playset"
                        log_feedback(photo["bytes"], photo["country"],
                                     res["guesses"], src)
                except Exception as e:
                    st.warning(f"endpoint unreachable: {e}")
                st.map(pd.DataFrame({"lat": [photo["lat"]], "lon": [photo["lon"]]}))
                st.caption("Street-view photo from the OSV5M test split "
                           "(CC-BY-SA, credits in playset/ATTRIBUTION.json). "
                           "Every revealed round is logged and feeds the next "
                           "retrain -- see the flywheel in docs/FEEDBACK-LOOP.md.")
                if st.button("Try another"):
                    st.session_state.pop("photo", None)
                    st.rerun()

with tab_honest:
    st.markdown("**The head has to beat CLIP zero-shot** (asking CLIP 'a photo "
                "taken in X' with no training at all) or it has no reason to exist.")
    try:
        st.dataframe(get_leaderboard()[["config", "top1", "top5"]],
                     use_container_width=True, hide_index=True)
    except Exception as e:
        st.info(f"leaderboard not materialized yet: {e}")
    for img, cap in [("accuracy_map.png", "Where the model knows the world"),
                     ("best_worst.png", "Hardest and easiest countries")]:
        p = ROOT / "assets" / img
        if p.exists():
            st.image(str(p), caption=cap)
