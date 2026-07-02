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
            except Exception as e:
                st.warning(f"endpoint unreachable: {e}")

with tab_play:
    st.markdown("A real held-out OSV5M street-view photo the model never trained "
                "on. Guess the country first, then see whether you beat it.")
    if st.button("New photo"):
        st.session_state.pop("photo", None)
    photo = st.session_state.get("photo") or random_play_photo()
    st.session_state["photo"] = photo
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
                except Exception as e:
                    st.warning(f"endpoint unreachable: {e}")
                st.map(pd.DataFrame({"lat": [photo["lat"]], "lon": [photo["lon"]]}))
                st.caption("Street-view photo from the OSV5M test split "
                           "(CC-BY-SA, credits in playset/ATTRIBUTION.json).")

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
