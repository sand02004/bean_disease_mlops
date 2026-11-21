"""
preprocessing.py
----------------
Everything that prepares images before they reach the model.

Used by:
  - prediction (one uploaded image  -> model input)
  - retraining (folders of images   -> tf.data datasets)
  - the API    (saving uploaded images for retraining)

The model itself normalizes pixels to [-1, 1] (Rescaling layer inside the model),
so here we only need to: open the image, convert it to RGB and resize it to 224x224.
"""
import glob
import io
import os
import random
import uuid

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------- constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
UPLOAD_DIR = "data/uploads"


# ---------------------------------------------------------------- single images
def load_image(image):
    """Open an image from a file path, raw bytes or a PIL image.
    Returns an RGB PIL image resized to 224x224."""
    try:
        if isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            img = image
        else:  # file path
            img = Image.open(image)
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("The file is not a valid image.") from e
    return img.resize(IMG_SIZE, Image.BILINEAR)


def preprocess_for_prediction(image):
    """Turn one image into a model-ready batch of shape (1, 224, 224, 3).
    Pixel values stay in 0-255: the model normalizes them itself."""
    img = load_image(image)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def save_uploaded_image(image_bytes, label, upload_dir=UPLOAD_DIR):
    """Preprocess an uploaded image (RGB + resize) and save it in
    data/uploads/<label>/ so it can be used for retraining.
    Returns the saved file path."""
    if label not in CLASS_NAMES:
        raise ValueError(f"Unknown label '{label}'. Use one of {CLASS_NAMES}.")
    img = load_image(image_bytes)
    folder = os.path.join(upload_dir, label)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{uuid.uuid4().hex}.jpg")
    img.save(path, format="JPEG", quality=95)
    return path


# ---------------------------------------------------------------- datasets
def collect_image_paths(directories):
    """Find all images in <dir>/<class_name>/ for every directory given.
    Returns two lists: file paths and integer labels."""
    paths, labels = [], []
    for directory in directories:
        for idx, cls in enumerate(CLASS_NAMES):
            for ext in IMAGE_EXTENSIONS:
                found = glob.glob(os.path.join(directory, cls, f"*{ext}"))
                paths.extend(found)
                labels.extend([idx] * len(found))
    return paths, labels


def _load_and_resize(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img.set_shape(IMG_SIZE + (3,))
    return img, label


def _to_dataset(paths, labels, batch_size, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_train_val_datasets(directories=(TRAIN_DIR,), validation_split=0.15,
                             batch_size=BATCH_SIZE):
    """Build training and validation datasets from one or more folders
    (e.g. the original data/train plus the newly uploaded data/uploads)."""
    paths, labels = collect_image_paths(directories)
    if len(paths) == 0:
        raise ValueError(f"No images found in {directories}")

    combined = list(zip(paths, labels))
    random.Random(SEED).shuffle(combined)
    n_val = max(1, int(len(combined) * validation_split))
    val, train = combined[:n_val], combined[n_val:]

    train_ds = _to_dataset([p for p, _ in train], [l for _, l in train], batch_size, shuffle=True)
    val_ds = _to_dataset([p for p, _ in val], [l for _, l in val], batch_size, shuffle=False)
    return train_ds, val_ds, len(train), len(val)


def build_test_dataset(test_dir=TEST_DIR, batch_size=BATCH_SIZE):
    """Test dataset in a fixed order (no shuffling) so labels line up with predictions."""
    paths, labels = collect_image_paths([test_dir])
    return _to_dataset(paths, labels, batch_size, shuffle=False), np.array(labels)


def get_data_augmentation():
    """Same augmentation as in the notebook. Used during (re)training only."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(0.2),
    ], name="data_augmentation")