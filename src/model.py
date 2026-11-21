"""
model.py
--------
Model creation, evaluation and RETRAINING.

Retraining uses the EXISTING trained model (models/bean_disease_model.h5) as the
pre-trained starting point - it does not start from scratch. It fine-tunes it on
the original training images + the newly uploaded images, evaluates the new
model on the test set and only deploys it if it is not worse than the old one.
"""
import json
import os
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.preprocessing import (CLASS_NAMES, IMG_SIZE, TEST_DIR, TRAIN_DIR, UPLOAD_DIR,
                               build_test_dataset, build_train_val_datasets,
                               get_data_augmentation)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/bean_disease_model.h5")
ARCHIVE_DIR = "models/archive"
LOG_PATH = "models/training_log.json"

RETRAIN_THRESHOLD = 10      # trigger: retrain when at least this many new images are uploaded
ACCURACY_TOLERANCE = 0.02   # deploy the new model unless it is >2% less accurate than the old one


# ---------------------------------------------------------------- creation
def create_model(num_classes=len(CLASS_NAMES), weights="imagenet"):
    """Same architecture as the notebook (inference version, without augmentation):
    Input -> Rescaling [-1,1] -> MobileNetV2 (frozen) -> GAP -> Dropout -> Dense softmax."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights=weights)
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    return tf.keras.Model(inputs, outputs, name="bean_disease_classifier")


def load_model(path=MODEL_PATH):
    return tf.keras.models.load_model(path, compile=False)


# ---------------------------------------------------------------- evaluation
def evaluate_model(model, test_dir=TEST_DIR):
    """Accuracy, loss, precision, recall and F1 on the test set."""
    test_ds, y_true = build_test_dataset(test_dir)
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    loss = float(tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_prob)).numpy())
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "loss": round(loss, 4),
        "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


# ---------------------------------------------------------------- retraining
def should_retrain(num_new_images, threshold=RETRAIN_THRESHOLD):
    """Retraining trigger: True when enough new labelled images have been uploaded."""
    return num_new_images >= threshold


def _prepare_for_fine_tuning(model, unfreeze_last=30):
    """Unfreeze only the last layers of MobileNetV2. BatchNorm layers stay frozen
    so the small new dataset cannot disturb their statistics."""
    base = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if base is not None:
        base.trainable = True
        for layer in base.layers[:-unfreeze_last]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False


def _append_log(entry):
    log = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            log = json.load(f)
    log.append(entry)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def retrain_model(model_path=MODEL_PATH, data_dirs=(TRAIN_DIR, UPLOAD_DIR),
                  test_dir=TEST_DIR, epochs=5, learning_rate=1e-5, num_new_images=0):
    """Fine-tune the saved model on original + uploaded images.
    Returns a summary dict with old/new metrics and whether the new model was deployed."""
    started_at = datetime.now().isoformat(timespec="seconds")

    # 1. Old model = baseline
    old_model = load_model(model_path)
    old_metrics = evaluate_model(old_model, test_dir)

    # 2. Data: original training images + uploaded images
    existing_dirs = [d for d in data_dirs if os.path.isdir(d)]
    train_ds, val_ds, n_train, n_val = build_train_val_datasets(existing_dirs)

    # 3. Start from the saved model (pre-trained), add augmentation on top for training
    model = load_model(model_path)
    _prepare_for_fine_tuning(model)
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = get_data_augmentation()(inputs)
    outputs = model(x)
    training_model = tf.keras.Model(inputs, outputs)
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = training_model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True)],
    )

    # 4. Evaluate the new model (the weights of `model` were updated by training)
    new_metrics = evaluate_model(model, test_dir)

    # 5. Save a versioned copy, deploy only if not worse than before
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    versioned_path = os.path.join(ARCHIVE_DIR, f"bean_disease_model_{version}.h5")
    model.save(versioned_path)

    deployed = new_metrics["accuracy"] >= old_metrics["accuracy"] - ACCURACY_TOLERANCE
    if deployed:
        backup_path = os.path.join(ARCHIVE_DIR, f"bean_disease_model_before_{version}.h5")
        shutil.copy(model_path, backup_path)
        shutil.copy(versioned_path, model_path)

    summary = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "model_version": version,
        "num_new_images": num_new_images,
        "num_train_images": n_train,
        "num_val_images": n_val,
        "epochs_run": len(history.history["loss"]),
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "deployed": bool(deployed),
    }
    _append_log(summary)
    return summary


if __name__ == "__main__":
    # Run from the project folder:  python -m src.model
    result = retrain_model(epochs=3)
    print(json.dumps(result, indent=2))