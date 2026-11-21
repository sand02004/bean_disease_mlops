"""
api/main.py
-----------
FastAPI service for the bean leaf disease model.

Run locally (from the project folder):
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs to test every endpoint (Swagger UI).
"""
import json
import os
import threading
import time
import traceback
from datetime import datetime
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src import database as db
from src.model import LOG_PATH, MODEL_PATH, RETRAIN_THRESHOLD, retrain_model, should_retrain
from src.prediction import BeanDiseasePredictor
from src.preprocessing import (CLASS_NAMES, TEST_DIR, TRAIN_DIR, compute_dataset_features,
                               save_uploaded_image)

AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "false").lower() == "true"
MODEL_INFO_PATH = "models/model_info.json"

app = FastAPI(
    title="Bean Leaf Disease API",
    description="Predict bean leaf diseases from photos, upload new labelled images and retrain the model.",
    version="1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

START_TIME = time.time()
db.init_db()
predictor = BeanDiseasePredictor(MODEL_PATH)

# Shared state of the retraining job (only one retraining at a time)
retrain_state = {"status": "idle", "started_at": None, "finished_at": None,
                 "result": None, "error": None}
retrain_lock = threading.Lock()
_features_cache = None   # image features for the dashboard, computed once


# ---------------------------------------------------------------- helpers
def _format_uptime(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


def _count_images(directory):
    counts = {}
    for cls in CLASS_NAMES:
        folder = os.path.join(directory, cls)
        counts[cls] = len(os.listdir(folder)) if os.path.isdir(folder) else 0
    return counts


def _current_metrics():
    """Metrics of the model currently in use: last deployed retraining, else the notebook results."""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            runs = [r for r in json.load(f) if r.get("deployed")]
        if runs:
            return {"source": f"retraining {runs[-1]['model_version']}", **runs[-1]["new_metrics"]}
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH) as f:
            info = json.load(f)
        return {"source": "initial training (notebook)", **info["test_metrics"]}
    return None


def _run_retraining():
    """Runs in a background thread so the API keeps answering while the model trains."""
    try:
        n_new = db.count_new_uploads()
        result = retrain_model(num_new_images=n_new)
        db.log_training_run(result)
        db.mark_uploads_used()
        if result["deployed"]:
            predictor.reload()          # start using the new model immediately
        retrain_state.update(status="completed", result=result)
    except Exception as e:
        traceback.print_exc()
        retrain_state.update(status="failed", error=str(e))
    finally:
        retrain_state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        retrain_lock.release()


def _start_retraining():
    if not retrain_lock.acquire(blocking=False):
        return False
    retrain_state.update(status="running", result=None, error=None, finished_at=None,
                         started_at=datetime.now().isoformat(timespec="seconds"))
    threading.Thread(target=_run_retraining, daemon=True).start()
    return True


# ---------------------------------------------------------------- endpoints
@app.get("/")
def root():
    return {"message": "Bean Leaf Disease API is running. See /docs for all endpoints."}


@app.get("/health")
def health():
    """Model up-time and status (used by the UI)."""
    uptime = time.time() - START_TIME
    return {
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "uptime": _format_uptime(uptime),
        "started_at": datetime.fromtimestamp(START_TIME).isoformat(timespec="seconds"),
        "model_path": MODEL_PATH,
        "model_loaded_at": predictor.loaded_at,
        "retraining_status": retrain_state["status"],
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Upload ONE leaf image and get the predicted disease."""
    start = time.perf_counter()
    image_bytes = file.file.read()
    try:
        result = predictor.predict(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result["filename"] = file.filename
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return result


@app.post("/upload")
def upload(label: str = Form(..., description=f"One of {CLASS_NAMES}"),
           files: List[UploadFile] = File(...)):
    """Bulk-upload labelled images for retraining. Each image is preprocessed
    (RGB, 224x224), saved in data/uploads/<label>/ and recorded in the database."""
    if label not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"label must be one of {CLASS_NAMES}")
    saved, failed = [], []
    for f in files:
        try:
            path = save_uploaded_image(f.file.read(), label)
            db.add_upload(f.filename, path, label)
            saved.append(f.filename)
        except ValueError:
            failed.append(f.filename)

    n_new = db.count_new_uploads()
    triggered = False
    if AUTO_RETRAIN and should_retrain(n_new):
        triggered = _start_retraining()
    return {
        "label": label,
        "saved": len(saved),
        "failed": failed,
        "new_images_waiting": n_new,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "retrain_recommended": should_retrain(n_new),
        "auto_retrain_started": triggered,
    }


@app.post("/retrain")
def retrain():
    """Trigger retraining (the 'Retrain' button). Runs in the background."""
    if db.count_new_uploads() == 0:
        raise HTTPException(status_code=400, detail="No new uploaded images to retrain on. Upload images first.")
    if not _start_retraining():
        raise HTTPException(status_code=409, detail="Retraining is already running.")
    return {"message": "Retraining started. Check /retrain/status for progress.",
            "started_at": retrain_state["started_at"]}


@app.get("/retrain/status")
def retrain_status():
    return retrain_state


@app.get("/stats")
def stats():
    """Numbers for the dashboard: dataset sizes, uploads, model metrics, training history."""
    return {
        "classes": CLASS_NAMES,
        "train_counts": _count_images(TRAIN_DIR),
        "test_counts": _count_images(TEST_DIR),
        "uploads_by_label": db.upload_counts_by_label(),
        "new_images_waiting": db.count_new_uploads(),
        "retrain_threshold": RETRAIN_THRESHOLD,
        "current_metrics": _current_metrics(),
        "training_runs": db.get_training_runs(),
    }


@app.get("/insights")
def insights():
    """Per-image features (green ratio, brightness, edge density) of the training set,
    used by the dashboard visualizations. Computed once, then cached."""
    global _features_cache
    if _features_cache is None:
        _features_cache = compute_dataset_features(TRAIN_DIR)
    return {"num_images": len(_features_cache), "features": _features_cache}