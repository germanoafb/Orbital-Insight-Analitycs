"""
Urban growth monitoring model.

Uses a Random Forest classifier to distinguish urban from non-urban land-use
pixels, then compares two temporal snapshots to quantify urban expansion.

Features used:
    ndvi_mean, ndvi_std, red_mean, nir_mean,
    built_up_index, brightness, wetness.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).parent
CLASSIFIER_PATH = MODEL_DIR / "urban_classifier.joblib"
SCALER_PATH = MODEL_DIR / "urban_scaler.joblib"

FEATURE_NAMES = [
    "ndvi_mean",
    "ndvi_std",
    "red_mean",
    "nir_mean",
    "built_up_index",   # (Red - NIR) / (Red + NIR)  ≈ negative NDVI
    "brightness",       # (Red + NIR) / 2
    "wetness",          # proxy: low for urban, high for vegetation
]


def generate_training_data(n_samples: int = 4000, seed: int = 42) -> tuple:
    """
    Generate simulated training data for the urban land-use classifier.

    Class 0 = non-urban (vegetation, water, bare soil)
    Class 1 = urban (impervious surfaces, buildings, roads)

    Args:
        n_samples: Total samples (split 60 % non-urban / 40 % urban).
        seed:      Random seed.

    Returns:
        (X, y) numpy arrays.
    """
    rng = np.random.default_rng(seed)
    n_urban = int(n_samples * 0.4)
    n_non = n_samples - n_urban

    # --- Non-urban ---
    ndvi_non = rng.uniform(0.2, 0.85, n_non)
    ndvi_std_non = rng.uniform(0.02, 0.15, n_non)
    red_non = rng.uniform(0.05, 0.3, n_non)
    nir_non = rng.uniform(0.3, 0.8, n_non)

    # --- Urban ---
    ndvi_urb = rng.uniform(-0.05, 0.25, n_urban)
    ndvi_std_urb = rng.uniform(0.05, 0.2, n_urban)
    red_urb = rng.uniform(0.2, 0.55, n_urban)
    nir_urb = rng.uniform(0.15, 0.4, n_urban)

    ndvi_mean = np.concatenate([ndvi_non, ndvi_urb])
    ndvi_std = np.concatenate([ndvi_std_non, ndvi_std_urb])
    red_mean = np.concatenate([red_non, red_urb])
    nir_mean = np.concatenate([nir_non, nir_urb])

    denom = red_mean + nir_mean
    built_up = np.where(denom > 0, (red_mean - nir_mean) / denom, 0)
    brightness = (red_mean + nir_mean) / 2
    wetness = ndvi_mean - brightness  # Simple proxy

    X = np.column_stack(
        [ndvi_mean, ndvi_std, red_mean, nir_mean, built_up, brightness, wetness]
    )
    y = np.array([0] * n_non + [1] * n_urban)

    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def train(n_samples: int = 4000, seed: int = 42) -> dict:
    """
    Train the Random Forest urban classifier and persist artefacts.

    Args:
        n_samples: Training set size.
        seed:      Random seed.

    Returns:
        Dictionary with accuracy and classification_report.
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
        max_depth=10,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return {
        "accuracy": float(acc),
        "classification_report": report,
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


def predict_map(
    ndvi_map: np.ndarray,
    red_map: np.ndarray,
    nir_map: np.ndarray,
) -> np.ndarray:
    """
    Classify every pixel of a satellite image as urban (1) or non-urban (0).

    Args:
        ndvi_map: 2-D NDVI array.
        red_map:  2-D Red band array.
        nir_map:  2-D NIR band array.

    Returns:
        2-D binary classification map.
    """
    clf, scaler = load_model()
    h, w = ndvi_map.shape
    flat_ndvi = ndvi_map.flatten()
    flat_red = red_map.flatten()
    flat_nir = nir_map.flatten()

    ndvi_std = np.full_like(flat_ndvi, np.std(flat_ndvi))
    denom = flat_red + flat_nir
    built_up = np.where(denom > 0, (flat_red - flat_nir) / denom, 0)
    brightness = (flat_red + flat_nir) / 2
    wetness = flat_ndvi - brightness

    X = np.column_stack(
        [flat_ndvi, ndvi_std, flat_red, flat_nir, built_up, brightness, wetness]
    )
    X_scaled = scaler.transform(X)
    labels = clf.predict(X_scaled)
    return labels.reshape(h, w)


def compute_urban_growth_stats(before_map: np.ndarray, after_map: np.ndarray) -> dict:
    """
    Compute urban growth statistics from two binary classification maps.

    Args:
        before_map: Binary classification map (earlier period).
        after_map:  Binary classification map (later period).

    Returns:
        Dictionary with area statistics and growth percentage.
    """
    total_pixels = before_map.size
    urban_before = int(np.sum(before_map == 1))
    urban_after = int(np.sum(after_map == 1))
    new_urban = int(np.sum((after_map == 1) & (before_map == 0)))
    lost_urban = int(np.sum((before_map == 1) & (after_map == 0)))
    growth_pct = ((urban_after - urban_before) / max(urban_before, 1)) * 100

    return {
        "total_pixels": total_pixels,
        "urban_before_pixels": urban_before,
        "urban_after_pixels": urban_after,
        "new_urban_pixels": new_urban,
        "lost_urban_pixels": lost_urban,
        "urban_before_pct": urban_before / total_pixels * 100,
        "urban_after_pct": urban_after / total_pixels * 100,
        "growth_pct": growth_pct,
    }
