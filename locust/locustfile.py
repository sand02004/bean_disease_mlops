"""
locust/locustfile.py
--------------------
Flood test for the Bean Leaf API: simulated farmers sending leaf photos to /predict.

Run from the project folder, e.g.:
    locust -f locust/locustfile.py --host http://SERVER_IP:8000 --headless -u 100 -r 10 -t 2m --csv locust/results/containers_1
"""
import glob
import os
import random

from locust import HttpUser, between, task

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATHS = glob.glob(os.path.join(ROOT, "data", "test", "*", "*.jpg"))
if not IMAGE_PATHS:
    raise SystemExit("No test images found in data/test/. Run Locust from the project folder.")

# Load a sample of test images into memory once, so disk reading doesn't slow the test
random.seed(42)
IMAGES = [(os.path.basename(p), open(p, "rb").read())
          for p in random.sample(IMAGE_PATHS, min(30, len(IMAGE_PATHS)))]


class FarmerUser(HttpUser):
    """One simulated user: mostly asks for diagnoses, sometimes checks the service."""
    wait_time = between(1, 3)   # seconds a user waits between two requests

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGES)
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")},
                              name="/predict", catch_response=True) as response:
            if response.status_code == 200 and "predicted_class" in response.text:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")