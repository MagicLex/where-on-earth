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
ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="where on earth", page_icon="globe_with_meridians",
                   layout="wide")


@st.cache_resource
def _project():
    return hopsworks.login()


@st.cache_resource
def _deployment():
    return _project().get_model_serving().get_deployment(DEPLOYMENT)


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


def flag(code):
    """ISO2 -> flag emoji, pure python."""
    try:
        return "".join(chr(0x1F1E6 + ord(c) - 65) for c in code.upper()[:2])
    except Exception:
        return ""


def render_guesses(res, truth=None):
    """Top guesses as flag + name + confidence bar. One renderer, both tabs."""
    for g in res["guesses"]:
        hit = truth is not None and g["code"] == truth
        label = f"{flag(g['code'])} **{g['country']}**" + (" ✓" if hit else "")
        a, b = st.columns([2, 3])
        a.markdown(label)
        b.progress(min(1.0, float(g["p"])), text=f"{g['p']*100:.0f}%")


def log_feedback(image_bytes, true_country, guesses, source, model_version=None):
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
           "true_country": true_country, "model_version": model_version,
           "pred_top1": guesses[0]["code"] if guesses else None,
           "correct": bool(guesses and guesses[0]["code"] == true_country)}
    with open(fb / "feedback.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


st.title("Where on earth was this taken?")
st.caption("Frozen CLIP eyes + a country head trained on OSV5M street view. "
           "It has never seen your photo, only 512 numbers describing it.")

tab_upload, tab_play = st.tabs(["Your photo", "Play vs the model"])

with tab_upload:
    # two ways in: paste a screenshot (chat_input accepts clipboard files on
    # recent streamlit) or classic upload. Either lands in session state so the
    # verdict survives reruns and the endpoint is hit once per image.
    new_raw = None
    try:
        msg = st.chat_input("Click here, then paste your screenshot (Cmd+V / "
                            "Ctrl+V) or attach a photo",
                            accept_file=True, file_type=["jpg", "jpeg", "png"])
        if msg and msg.files:
            new_raw = msg.files[0].read()
        st.caption("paste zone active: click the box above first, then Cmd+V "
                   "(mac) / Ctrl+V")
    except TypeError:
        st.caption("this streamlit build has no paste support; use the uploader")
    up = st.file_uploader("...or upload", type=["jpg", "jpeg", "png"],
                          label_visibility="collapsed")
    if up is not None and st.session_state.get("upload_name") != up.name:
        new_raw = up.read()
        st.session_state["upload_name"] = up.name
    if new_raw:
        st.session_state["upload_raw"] = new_raw
        st.session_state.pop("upload_res", None)

    raw = st.session_state.get("upload_raw")
    if raw:
        col1, col2 = st.columns([1, 1], gap="large")
        col1.image(raw, use_container_width=True)
        with col2:
            if "upload_res" not in st.session_state:
                with st.spinner("Looking at it..."):
                    try:
                        st.session_state["upload_res"] = guess(raw)
                    except Exception as e:
                        st.warning(f"endpoint unreachable: {e}")
            res = st.session_state.get("upload_res")
            if res and "error" in res:
                st.error(res["error"])
            elif res:
                top = res["guesses"][0]
                st.markdown(f"## {flag(top['code'])} {top['country']} "
                            f"<small>(model v{res.get('model_version', '?')})</small>",
                            unsafe_allow_html=True)
                render_guesses(res)
                if st.button("Guess again", help="Same photo, fresh call. Useful "
                             "after a retrain: the model version may have moved."):
                    st.session_state.pop("upload_res", None)
                    st.rerun()
                st.caption("Nothing is uploaded anywhere but this project's own "
                           "endpoint; the image is embedded and discarded.")
                iso = json.loads((ROOT / "assets" / "iso2name.json").read_text())
                wrong = st.selectbox("Was it wrong? Teach it the real country",
                                     ["-"] + sorted(iso.values()))
                if wrong != "-" and st.button("Teach it"):
                    code = next(c for c, n in iso.items() if n == wrong)
                    log_feedback(raw, code, res["guesses"], "upload",
                                 res.get("model_version"))
                    st.success("Logged. It goes into the next retrain.")

with tab_play:
    st.markdown("Real street view the model never trained on. Guess the country "
                "first, then see whether you beat it.")
    live = st.toggle("Live Mapillary (fresh worldwide imagery)",
                     value=False, help="Falls back to the held-out playset when "
                     "Mapillary is grumpy. Both are fair: ground truth is known.")
    # buttons must be SIBLINGS: a button nested in another button's if-block never
    # fires (the outer state is False on the inner click's rerun).
    ba, bb = st.columns([1, 1])
    if ba.button("New photo") or bb.button("Try another"):
        st.session_state.pop("photo", None)
        st.session_state.pop("revealed", None)
    if "photo" not in st.session_state or st.session_state.get("photo_live") != live:
        mly = live_mapillary_photo() if live else None
        if live:
            k = "mly_ok" if mly else "mly_fail"
            st.session_state[k] = st.session_state.get(k, 0) + 1
        st.session_state["photo"] = mly or random_play_photo()
        st.session_state["photo_live"] = live
        st.session_state.pop("revealed", None)
    photo = st.session_state.get("photo")
    if photo is None:
        st.info("playset missing; run tools/make_playset.py and redeploy.")
    else:
        is_live = str(photo["title"]).startswith("mapillary:")
        if is_live:
            st.success(f"🟢 LIVE Mapillary ({photo['title'].split(':')[1]})", icon="🛰️")
        elif live:
            st.warning("Mapillary gave nothing for those anchors. This one is from "
                       "the held-out playset.", icon="📦")
        else:
            st.caption("📦 held-out playset")
        ok, fail = st.session_state.get("mly_ok", 0), st.session_state.get("mly_fail", 0)
        if live and (ok + fail):
            st.caption(f"live hit rate this session: {ok}/{ok+fail}")
        c1, c2 = st.columns([1, 1], gap="large")
        c1.image(photo["bytes"], use_container_width=True)
        with c2:
            if st.button("Reveal model guess + truth", type="primary",
                         use_container_width=True):
                try:
                    res = guess(photo["bytes"])
                    st.session_state["revealed"] = res
                    if "guesses" in res:
                        src = "mapillary" if str(photo["title"]).startswith("mapillary:") \
                            else "playset"
                        log_feedback(photo["bytes"], photo["country"],
                                     res["guesses"], src,
                                     res.get("model_version"))   # once per reveal
                except Exception as e:
                    st.warning(f"endpoint unreachable: {e}")
            res = st.session_state.get("revealed")
            if res and "guesses" in res:
                hit = res["guesses"][0]["code"] == photo["country"]
                if hit:
                    st.success(f"Model got it: {flag(photo['country'])} "
                               f"(v{res.get('model_version', '?')})")
                else:
                    st.error(f"Model missed. Truth: {flag(photo['country'])} "
                             f"`{photo['country']}` (v{res.get('model_version', '?')})")
                render_guesses(res, truth=photo["country"])
                st.map(pd.DataFrame({"lat": [photo["lat"]], "lon": [photo["lon"]]}),
                       height=220)
                st.caption("Street view with known ground truth (CC-BY-SA; playset "
                           "credits in playset/ATTRIBUTION.json). Every revealed "
                           "round feeds the next retrain -- docs/FEEDBACK-LOOP.md.")

st.caption("top-1 52.3% / top-5 79.8% over 173 countries, held-out places. "
           "Numbers, caveats and the eval maps: "
           "[github.com/MagicLex/where-on-earth](https://github.com/MagicLex/where-on-earth)")
