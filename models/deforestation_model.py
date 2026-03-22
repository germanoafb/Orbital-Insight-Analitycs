"""
Deforestation detection model.

Uses a Random Forest classifier trained on NDVI-derived features to classify
areas as deforested (1) or non-deforested (0).

Features used:
    ndvi_mean, ndvi_std, ndvi_p25, ndvi_p50, ndvi_p75,
    fraction_vegetation, fraction_barren, fraction_water,
    ndvi_change (temporal delta).
"""

import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# Path for persisting trained model artefacts
MODEL_DIR = Path(__file__).parent
CLASSIFIER_PATH = MODEL_DIR / "deforestation_classifier.joblib"
SCALER_PATH = MODEL_DIR / "deforestation_scaler.joblib"

FEATURE_NAMES = [
    "ndvi_mean",
    "ndvi_std",
    "ndvi_p25",
    "ndvi_p50",
    "ndvi_p75",
    "fraction_vegetation",
    "fraction_barren",
    "fraction_water",
    "ndvi_change",
]


def generate_training_data(n_samples: int = 2000, seed: int = 42) -> tuple:
    """
    Generate simulated training data for the deforestation classifier.

    Positive class (1 = deforested): low NDVI, high ndvi_change (loss)
    Negative class (0 = intact):     high NDVI, low ndvi_change

    Args:
        n_samples: Total number of samples (split 50/50 by class).
        seed:      Random seed.

    Returns:
        (X, y) numpy arrays.
    """
    rng = np.random.default_rng(seed)
    half = n_samples // 2

    # --- Intact forest (label = 0) ---
    intact = np.column_stack(
        [
            rng.uniform(0.5, 0.9, half),   # ndvi_mean
            rng.uniform(0.05, 0.15, half),  # ndvi_std
            rng.uniform(0.4, 0.7, half),    # ndvi_p25
            rng.uniform(0.5, 0.85, half),   # ndvi_p50
            rng.uniform(0.6, 0.9, half),    # ndvi_p75
            rng.uniform(0.6, 0.95, half),   # fraction_vegetation
            rng.uniform(0.01, 0.1, half),   # fraction_barren
            rng.uniform(0.0, 0.05, half),   # fraction_water
            rng.uniform(-0.05, 0.05, half), # ndvi_change (stable)
        ]
    )

    # --- Deforested area (label = 1) ---
    deforested = np.column_stack(
        [
            rng.uniform(0.05, 0.35, half),  # ndvi_mean
            rng.uniform(0.05, 0.2, half),   # ndvi_std
            rng.uniform(0.0, 0.2, half),    # ndvi_p25
            rng.uniform(0.05, 0.3, half),   # ndvi_p50
            rng.uniform(0.1, 0.4, half),    # ndvi_p75
            rng.uniform(0.0, 0.25, half),   # fraction_vegetation
            rng.uniform(0.4, 0.8, half),    # fraction_barren
            rng.uniform(0.0, 0.1, half),    # fraction_water
            rng.uniform(-0.8, -0.15, half), # ndvi_change (large loss)
        ]
    )

    X = np.vstack([intact, deforested])
    y = np.array([0] * half + [1] * half)

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def train(n_samples: int = 2000, seed: int = 42) -> dict:
    """
    Train the Random Forest deforestation classifier and persist artefacts.

    Args:
        n_samples: Training set size.
        seed:      Random seed.

    Returns:
        Dictionary with accuracy, classification_report, and confusion_matrix.
    """
    X, y = generate_training_data(n_samples, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Persist artefacts
    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return {
        "accuracy": float(acc),
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_names": FEATURE_NAMES,
        "feature_importances": clf.feature_importances_.tolist(),
    }


def load_model():
    """Load persisted classifier and scaler (train if not found)."""
    if not CLASSIFIER_PATH.exists() or not SCALER_PATH.exists():
        train()
    clf = joblib.load(CLASSIFIER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, scaler


def predict(features: dict) -> dict:
    """
    Predict whether an area is deforested given a feature dictionary.

    Args:
        features: Dictionary with keys matching FEATURE_NAMES.

    Returns:
        Dictionary with 'label' (0/1), 'probability', and 'interpretation'.
    """
    clf, scaler = load_model()
    x = np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]])
    x_scaled = scaler.transform(x)
    label = int(clf.predict(x_scaled)[0])
    proba = float(clf.predict_proba(x_scaled)[0][1])
    interpretation = "Deforested area detected" if label == 1 else "Intact vegetation"
    return {
        "label": label,
        "deforestation_probability": proba,
        "interpretation": interpretation,
    }
