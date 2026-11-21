"""
ui/app.py
---------
Streamlit web app for the bean leaf disease pipeline ("Bean Leaf Doctor").

Run locally (the API must be running first):
    streamlit run ui/app.py
"""
import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
CLASS_LABELS = {"angular_leaf_spot": "Angular leaf spot", "bean_rust": "Bean rust", "healthy": "Healthy"}
COLORS = {"angular_leaf_spot": "#B8862B", "bean_rust": "#A23E26", "healthy": "#3A7D44"}
LEAF_GREEN = "#1F4D2B"

st.set_page_config(page_title="Bean Leaf Doctor", page_icon="🌱", layout="wide")

# ---------------------------------------------------------------- styling
# Leaf-vein pattern for the header band (inline SVG, no external image needed)
VEINS = (
    "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='260' height='140' "
    "viewBox='0 0 260 140'><g fill='none' stroke='%23ffffff' stroke-opacity='0.09' stroke-width='1.4'>"
    "<path d='M0 120 C70 95 150 80 260 20'/><path d='M60 101 C70 70 80 50 75 20'/>"
    "<path d='M110 88 C125 60 140 45 140 10'/><path d='M165 70 C185 50 200 35 205 5'/>"
    "<path d='M60 101 C80 115 100 125 120 140'/><path d='M110 88 C135 100 160 110 185 140'/>"
    "<path d='M165 70 C190 80 215 95 240 140'/></g></svg>\")"
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&display=swap');
h1, h2, h3, .hero-title, .dx-name {{ font-family: 'Bricolage Grotesque', sans-serif !important; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 2rem; max-width: 1150px; }}

.hero {{ background-color: {LEAF_GREEN}; background-image: {VEINS}; background-size: 260px 140px;
        border-radius: 14px; padding: 1.8rem 2rem 1.6rem; margin-bottom: 1.6rem; color: #F1F6EA; }}
.hero .hero-title {{ font-size: 2.4rem; font-weight: 700; line-height: 1.1; margin: 0; color: #FFFFFF; }}
.hero .hero-sub {{ font-size: 1.05rem; margin: .5rem 0 0; max-width: 62ch; opacity: .88; }}

.dx {{ border-left: 6px solid var(--dx-color); background: #FFFFFF; border-radius: 4px 12px 12px 4px;
      padding: 1.2rem 1.4rem; margin-bottom: 1rem; }}
.dx-kicker {{ font-size: .9rem; color: #5B6B5E; margin: 0; }}
.dx-name {{ font-size: 2.1rem; font-weight: 700; color: var(--dx-color); margin: .1rem 0 .2rem; line-height: 1.1; }}
.dx-conf {{ font-size: 1rem; color: #1C2A1F; margin: 0; }}
.dx-warn {{ margin-top: .7rem; padding: .5rem .75rem; background: #FFF4D6; border-radius: 8px; font-size: .95rem; }}

.prob-row {{ display: grid; grid-template-columns: 9.5rem 1fr 3.5rem; align-items: center; gap: .6rem; margin: .35rem 0; }}
.prob-track {{ background: #E7EEDC; border-radius: 99px; height: .7rem; overflow: hidden; }}
.prob-fill {{ height: 100%; border-radius: 99px; }}
.prob-pct {{ text-align: right; font-variant-numeric: tabular-nums; }}

.advice {{ background: #EEF4E6; border-radius: 10px; padding: .9rem 1.1rem; margin-top: 1rem; line-height: 1.55; }}
.story {{ background: #FFFFFF; border-left: 3px solid {LEAF_GREEN}; padding: .7rem 1rem; border-radius: 0 8px 8px 0;
         line-height: 1.6; max-width: 80ch; margin-bottom: 2rem; }}
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f"<div class='hero'><div class='hero-title'>{title}</div>"
                f"<div class='hero-sub'>{subtitle}</div></div>", unsafe_allow_html=True)


def style_fig(fig, height=340):
    fig.update_layout(height=height, showlegend=False, plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Source Sans Pro, sans-serif", color="#1C2A1F", size=14),
                      margin=dict(l=10, r=10, t=30, b=10), xaxis_title=None)
    fig.update_yaxes(gridcolor="#E7EEDC")
    return fig


# ---------------------------------------------------------------- API helpers
def api_get(path, timeout=30):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def api_post(path, timeout=60, **kwargs):
    try:
        r = requests.post(f"{API_URL}{path}", timeout=timeout, **kwargs)
        return r.status_code, r.json()
    except requests.RequestException as e:
        return None, {"detail": f"Cannot reach the API ({e.__class__.__name__}). Check that it is running."}


def metrics_table(metrics):
    names = {"accuracy": "Accuracy", "loss": "Loss", "precision": "Precision",
             "recall": "Recall", "f1_score": "F1-score"}
    return pd.DataFrame([(names[k], v) for k, v in metrics.items() if k in names], columns=["Metric", "Value"])


def api_offline_message():
    st.error(f"The model API at {API_URL} is not responding. Start it with `uvicorn api.main:app`, then refresh.")


# ---------------------------------------------------------------- pages
def page_dashboard():
    hero("Bean Leaf Doctor",
         "Diagnoses angular leaf spot and bean rust from a smartphone photo of a bean leaf, "
         "so farmers can act before the disease spreads.")
    health = api_get("/health", timeout=10)
    if not health:
        api_offline_message()
        return

    loaded = datetime.fromisoformat(health["model_loaded_at"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("API status", "Online")
    c2.metric("Model up-time", health["uptime"])
    c3.metric(f"Model loaded on {loaded.strftime('%d %b')}", loaded.strftime("%H:%M"))
    c4.metric("Retraining", health["retraining_status"].capitalize())
    st.button("Refresh", icon=":material/refresh:")

    stats = api_get("/stats")
    if not stats:
        return
    m = stats.get("current_metrics")
    if m:
        st.subheader("How well the current model performs")
        st.caption(f"Measured on 128 test images the model never saw · from {m['source']}")
        cols = st.columns(5)
        for col, key, label in zip(cols, ["accuracy", "precision", "recall", "f1_score", "loss"],
                                   ["Accuracy", "Precision", "Recall", "F1-score", "Loss"]):
            col.metric(label, f"{m[key]:.1%}" if key != "loss" else f"{m[key]:.3f}")

    st.subheader("Retraining history")
    runs = stats.get("training_runs", [])
    if runs:
        df = pd.DataFrame(runs)
        df = pd.DataFrame({
            "Version": df["model_version"],
            "Finished": df["finished_at"].str.replace("T", " "),
            "New images": df["num_new_images"],
            "Accuracy before": (df["old_accuracy"] * 100).round(2).astype(str) + "%",
            "Accuracy after": (df["new_accuracy"] * 100).round(2).astype(str) + "%",
            "Deployed": df["deployed"].map({1: "Yes", 0: "No (kept old model)"}),
        })
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("The model has not been retrained yet. Upload new images in **Upload & retrain**.")


def page_predict():
    hero("Diagnose a leaf", "Take a clear photo of one bean leaf, upload it, and get a diagnosis in seconds.")
    file = st.file_uploader("Leaf photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if not file:
        st.caption("Tip: fill the frame with one leaf, in daylight, with the spots visible.")
        return

    col_img, col_res = st.columns([1, 1.25], gap="large")
    col_img.image(file, caption=file.name, width="stretch")

    with st.spinner("Examining the leaf..."):
        code, res = api_post("/predict", files={"file": (file.name, file.getvalue(), file.type)})
    if code != 200:
        col_res.error(f"Could not diagnose this image: {res.get('detail')}")
        return

    cls = res["predicted_class"]
    warn = ("<div class='dx-warn'>The model is unsure about this leaf. Please confirm with an agronomist.</div>"
            if res["low_confidence"] else "")
    col_res.markdown(
        f"<div class='dx' style='--dx-color:{COLORS[cls]}'>"
        f"<div class='dx-kicker'>Diagnosis</div>"
        f"<div class='dx-name'>{CLASS_LABELS[cls]}</div>"
        f"<div class='dx-conf'>{res['confidence']:.1%} confident</div>{warn}</div>",
        unsafe_allow_html=True)

    rows = "".join(
        f"<div class='prob-row'><span>{CLASS_LABELS[c]}</span>"
        f"<div class='prob-track'><div class='prob-fill' style='width:{p*100:.1f}%;background:{COLORS[c]}'></div></div>"
        f"<span class='prob-pct'>{p:.1%}</span></div>"
        for c, p in sorted(res["probabilities"].items(), key=lambda kv: -kv[1]))
    advice = res["advice"].replace("Uncertain prediction - please confirm with an agronomist. ", "")
    col_res.markdown(f"{rows}<div class='advice'><strong>What to do:</strong> {advice}</div>",
                     unsafe_allow_html=True)
    col_res.caption(f"Answered in {res['latency_ms']:.0f} ms")


def page_insights():
    hero("What the data tells us",
         "1,295 field photos from Uganda, labelled by experts at NaCRRI. Here is what the images reveal "
         "about each disease, and why a deep learning model is needed.")
    stats = api_get("/stats")
    if not stats:
        api_offline_message()
        return

    st.subheader("How many photos of each class")
    counts = pd.DataFrame([
        {"Class": CLASS_LABELS[c], "key": c, "Split": "Training set" if s == "train" else "Test set",
         "Images": stats[f"{s}_counts"][c]} for s in ["train", "test"] for c in CLASS_NAMES])
    fig = px.bar(counts, x="Class", y="Images", color="key", facet_col="Split",
                 color_discrete_map=COLORS, text_auto=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(style_fig(fig), width="stretch")
    st.markdown("<div class='story'>The classes are <b>balanced</b> (about 385–393 training photos each), "
                "so the model is not biased toward one class and accuracy is a fair measure.</div>",
                unsafe_allow_html=True)

    with st.spinner("Measuring every training photo (only slow the first time)..."):
        data = api_get("/insights", timeout=300)
    if not data:
        st.error("The image measurements could not be loaded. Refresh the page to try again.")
        return
    df = pd.DataFrame(data["features"])
    df["Class"] = df["class"].map(CLASS_LABELS)

    stories = {
        "green_ratio": ("Disease drains the green from the leaf", "Green ratio",
                        "Healthy leaves are the <b>greenest</b> and bean rust the <b>least green</b>, because "
                        "rust leaves reddish-brown pustules. But the boxes overlap: field photos also show soil, "
                        "grass and hands, so colour alone cannot diagnose a leaf."),
        "brightness": ("Bean rust photos are darker", "Brightness (0 = black, 255 = white)",
                       "Bean rust photos are <b>darker</b> on average. That may be the disease, or shade and time "
                       "of day. To stop the model from learning “dark photo means rust”, brightness is varied "
                       "randomly during training."),
        "edge_density": ("Spots are too small to measure simply", "Edge density",
                         "Edge density is <b>almost the same</b> for every class: veins and the background create "
                         "most edges, while lesions are small. Simple measurements are not enough, which is why "
                         "a pretrained CNN (MobileNetV2) learns the lesion patterns instead."),
    }
    for feat, (title, axis, story) in stories.items():
        st.subheader(title)
        fig = px.box(df, x="Class", y=feat, color="class", color_discrete_map=COLORS, points="outliers",
                     labels={feat: axis})
        st.plotly_chart(style_fig(fig), width="stretch")
        st.markdown(f"<div class='story'>{story}</div>", unsafe_allow_html=True)


def page_retrain():
    hero("Teach the model new leaves",
         "Upload new labelled photos, then retrain. The current model is fine-tuned on them and only "
         "replaces the old one if it performs at least as well.")
    stats = api_get("/stats")
    if not stats:
        api_offline_message()
        return

    st.subheader("Step 1 · Upload photos of one class")
    label = st.selectbox("What do these photos show?", CLASS_NAMES, format_func=lambda c: CLASS_LABELS[c])
    files = st.file_uploader("Photos (select several at once)", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)
    if st.button("Upload photos", icon=":material/upload:", disabled=not files):
        payload = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        with st.spinner(f"Uploading {len(files)} photos..."):
            code, res = api_post("/upload", data={"label": label}, files=payload, timeout=300)
        if code == 200:
            st.success(f"Uploaded {res['saved']} photos as {CLASS_LABELS[label]}. "
                       "They were resized to 224×224 and saved in the database.")
            if res["failed"]:
                st.warning(f"These files are not images and were skipped: {', '.join(res['failed'])}")
            stats = api_get("/stats")
        else:
            st.error(res.get("detail"))

    st.subheader("Step 2 · Check the retraining trigger")
    n_new, threshold = stats["new_images_waiting"], stats["retrain_threshold"]
    st.progress(min(n_new / threshold, 1.0),
                text=f"{n_new} of {threshold} new photos needed before retraining is recommended")
    uploads = stats.get("uploads_by_label", {})
    if uploads:
        st.caption("Uploaded so far: " + ", ".join(
            f"{CLASS_LABELS[k]} {v['total']} ({v['new']} not yet used)" for k, v in uploads.items()))
    if n_new >= threshold:
        st.success("Enough new photos: retraining is recommended.")

    st.subheader("Step 3 · Retrain")
    if st.button("Retrain model", icon=":material/model_training:", type="primary", disabled=n_new == 0):
        code, res = api_post("/retrain")
        if code != 200:
            st.error(res.get("detail"))
            return
        box = st.empty()
        start = time.time()
        while True:
            state = api_get("/retrain/status") or {}
            if state.get("status") != "running":
                break
            box.info(f"Retraining… {int(time.time() - start)} s. This usually takes 1–3 minutes.")
            time.sleep(3)
        box.empty()
        st.session_state["last_retrain"] = state
    if n_new == 0:
        st.caption("Upload at least one photo to enable retraining.")

    state = st.session_state.get("last_retrain")
    if state and state.get("status") == "completed":
        r = state["result"]
        msg = ("Retraining finished and the new model is now live."
               if r["deployed"] else "Retraining finished, but the new model was less accurate, so the old one was kept.")
        st.success(f"{msg} Trained for {r['epochs_run']} epochs on "
                   f"{r['num_train_images'] + r['num_val_images']} photos ({r['num_new_images']} new).")
        comp = metrics_table(r["old_metrics"]).rename(columns={"Value": "Before"})
        comp["After"] = metrics_table(r["new_metrics"])["Value"]
        st.dataframe(comp, width="stretch", hide_index=True)
    elif state and state.get("status") == "failed":
        st.error(f"Retraining failed: {state.get('error')}")


# ---------------------------------------------------------------- navigation
nav = st.navigation([
    st.Page(page_dashboard, title="Dashboard", icon=":material/monitoring:", default=True),
    st.Page(page_predict, title="Diagnose a leaf", url_path="diagnose", icon=":material/eco:"),
    st.Page(page_insights, title="Data insights", url_path="insights", icon=":material/bar_chart:"),
    st.Page(page_retrain, title="Upload & retrain", url_path="retrain", icon=":material/model_training:"),
])

with st.sidebar:
    status = api_get("/health", timeout=10)
    if status:
        st.success(f"Model online for {status['uptime']}", icon=":material/check_circle:")
    else:
        st.error("Model API offline", icon=":material/error:")

nav.run()