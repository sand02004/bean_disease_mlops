"""
prediction.py
-------------
Load the trained model once and predict the disease of one leaf image.
"""
import os
import sys
import threading
from datetime import datetime

import numpy as np

from src.model import MODEL_PATH, load_model
from src.preprocessing import CLASS_NAMES, preprocess_for_prediction

CONFIDENCE_THRESHOLD = 0.60   # below this, the app asks the farmer to confirm with an expert

ADVICE = {
    "angular_leaf_spot": "Angular leaf spot detected. Remove infected leaves, avoid working in "
                         "the field when plants are wet, and use certified disease-free seed next season.",
    "bean_rust": "Bean rust detected. Remove and destroy infected leaves, improve air flow between "
                 "plants, and consult an agronomist about a suitable fungicide.",
    "healthy": "The leaf looks healthy. Keep monitoring your plants regularly.",
}


class BeanDiseasePredictor:
    """Keeps the model in memory so each prediction is fast.
    Call reload() after retraining to switch to the new model."""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self._lock = threading.Lock()
        self.reload()

    def reload(self):
        model = load_model(self.model_path)
        with self._lock:
            self.model = model
            self.loaded_at = datetime.now().isoformat(timespec="seconds")

    def predict(self, image):
        """image: file path, raw bytes or PIL image."""
        batch = preprocess_for_prediction(image)
        with self._lock:
            probs = self.model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        predicted = CLASS_NAMES[idx]
        low_confidence = confidence < CONFIDENCE_THRESHOLD
        return {
            "predicted_class": predicted,
            "confidence": round(confidence, 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)},
            "low_confidence": low_confidence,
            "advice": ("Uncertain prediction - please confirm with an agronomist. "
                       if low_confidence else "") + ADVICE[predicted],
        }


if __name__ == "__main__":
    # Run from the project folder:  python -m src.prediction path/to/leaf.jpg
    if len(sys.argv) != 2 or not os.path.exists(sys.argv[1]):
        print("Usage: python -m src.prediction path/to/leaf.jpg")
        sys.exit(1)
    predictor = BeanDiseasePredictor()
    print(predictor.predict(sys.argv[1]))