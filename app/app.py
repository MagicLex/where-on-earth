"""where-on-earth -- Streamlit front for the geoguessing endpoint.

Thin client: upload a photo (or pull a random geotagged one from Wikimedia Commons
and play against the model), the KServe endpoint embeds it with the same frozen
CLIP the training used and the head answers with a country distribution. The app
renders guesses, the honest leaderboard (including the zero-shot baseline the head
must beat), and the accuracy world map from training.
"""
import base64
import io
import json
import random
from pathlib import Path

import hopsworks
import pandas as pd
import requests
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


@st.cache_data(ttl=300)
def random_commons_photo():
    """A random geotagged Commons photo + its true coordinates (the reveal)."""
    s = requests.Session()
    s.headers.update({"User-Agent": "where-on-earth demo (adminaccounts@hopsworks.ai)"})
    lat = random.uniform(-45, 60)
    lon = random.uniform(-120, 150)
    r = s.get("https://commons.wikimedia.org/w/api.php", timeout=20, params={
        "action": "query", "list": "geosearch", "gscoord": f"{lat}|{lon}",
        "gsradius": 10000, "gsnamespace": 6, "gslimit": 20, "format": "json"})
    hits = r.json().get("query", {}).get("geosearch", [])
    if not hits:
        return None
    h = random.choice(hits)
    r = s.get("https://commons.wikimedia.org/w/api.php", timeout=20, params={
        "action": "query", "titles": h["title"], "prop": "imageinfo",
        "iiprop": "url", "iiurlwidth": 640, "format": "json"})
    info = list(r.json()["query"]["pages"].values())[0].get("imageinfo")
    if not info:
        return None
    img = s.get(info[0]["thumburl"], timeout=30).content
    return {"bytes": img, "lat": h["lat"], "lon": h["lon"], "title": h["title"]}


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
    st.markdown("A random geotagged photo from Wikimedia Commons. Guess first, "
                "then see whether you beat the model.")
    if st.button("New photo"):
        st.session_state.pop("photo", None)
        random_commons_photo.clear()
    photo = st.session_state.get("photo") or random_commons_photo()
    st.session_state["photo"] = photo
    if photo is None:
        st.info("Commons gave nothing for that random spot; hit New photo again.")
    else:
        c1, c2 = st.columns([1, 1])
        c1.image(photo["bytes"], use_container_width=True)
        with c2:
            if st.button("Reveal model guess + truth"):
                try:
                    res = guess(photo["bytes"])
                    if "guesses" in res:
                        for g in res["guesses"][:3]:
                            st.markdown(f"- **{g['country']}** {g['p']*100:.0f}%")
                except Exception as e:
                    st.warning(f"endpoint unreachable: {e}")
                st.map(pd.DataFrame({"lat": [photo["lat"]], "lon": [photo["lon"]]}))
                st.caption(f"True location: {photo['lat']:.3f}, {photo['lon']:.3f} "
                           f"({photo['title']})")

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
