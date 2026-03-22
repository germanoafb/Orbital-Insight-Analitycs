"""
Agricultural productivity prediction model.

Uses a Gradient Boosting regressor trained on agronomic and remote-sensing
features to forecast crop yield (tons/ha).

Features used:
    ndvi_mean, ndvi_p50, soil_moisture, temperature_avg,
    precipitation_mm, crop_type_encoded, days_to_harvest.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Path for persisting trained model artefacts
MODEL_DIR = Path(__file__).parent
REGRESSOR_PATH = MODEL_DIR / "agriculture_regressor.joblib"
SCALER_PATH = MODEL_DIR / "agriculture_scaler.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "agriculture_label_encoder.joblib"

CROP_TYPES = ["soy", "corn", "wheat", "cotton", "sugarcane"]

FEATURE_NAMES = [
    "ndvi_mean",
    "ndvi_p50",
    "soil_moisture",
    "temperature_avg",
    "precipitation_mm",
    "crop_type_encoded",
    "days_to_harvest",
]


def generate_training_data(n_samples: int = 3000, seed: int = 42) -> tuple:
    """
    Generate simulated training data for the agriculture productivity regressor.

    The target (yield_tons_ha) is a realistic function of the features
    with some added Gaussian noise.

    Args:
        n_samples: Number of samples.
        seed:      Random seed.

    Returns:
        (X, y) numpy arrays and a fitted LabelEncoder for crop types.
    """
    rng = np.random.default_rng(seed)
    le = LabelEncoder()
    le.fit(CROP_TYPES)

    ndvi_mean = rng.uniform(0.2, 0.9, n_samples)
    ndvi_p50 = ndvi_mean + rng.normal(0, 0.05, n_samples)
    ndvi_p50 = np.clip(ndvi_p50, 0.1, 1.0)
    soil_moisture = rng.uniform(0.1, 0.9, n_samples)
    temperature_avg = rng.uniform(15, 38, n_samples)  # °C
    precipitation_mm = rng.uniform(50, 900, n_samples)
    crop_type_raw = rng.choice(CROP_TYPES, n_samples)
    crop_type_encoded = le.transform(crop_type_raw).astype(float)
    days_to_harvest = rng.uniform(60, 180, n_samples)

    # Target: simplified agronomic model
    base_yield = (
        3.0
        + 8.0 * ndvi_mean
        + 3.0 * soil_moisture
        - 0.08 * (temperature_avg - 25) ** 2  # Optimal at ~25°C
        + 0.005 * precipitation_mm
        - 0.01 * days_to_harvest
    )
    # Per-crop multipliers
    crop_multipliers = {"soy": 1.0, "corn": 1.3, "wheat": 0.9, "cotton": 0.7, "sugarcane": 2.5}
    multiplier = np.array([crop_multipliers[c] for c in crop_type_raw])
    yield_tons_ha = np.clip(base_yield * multiplier + rng.normal(0, 0.5, n_samples), 0.5, 30.0)

    X = np.column_stack(
        [
            ndvi_mean,
            ndvi_p50,
            soil_moisture,
            temperature_avg,
            precipitation_mm,
            crop_type_encoded,
            days_to_harvest,
        ]
    )
    return X, yield_tons_ha, le


def train(n_samples: int = 3000, seed: int = 42) -> dict:
    """
    Train the Gradient Boosting regressor and persist artefacts.

    Args:
        n_samples: Training set size.
        seed:      Random seed.

    Returns:
        Dictionary with RMSE, MAE, R², and feature importances.
    """
    X, y, le = generate_training_data(n_samples, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )
    reg.fit(X_train_scaled, y_train)

    y_pred = reg.predict(X_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Persist artefacts
    joblib.dump(reg, REGRESSOR_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "feature_names": FEATURE_NAMES,
        "feature_importances": reg.feature_importances_.tolist(),
    }


def load_model():
    """Load persisted regressor, scaler, and label encoder (train if not found)."""
    if (
        not REGRESSOR_PATH.exists()
        or not SCALER_PATH.exists()
        or not LABEL_ENCODER_PATH.exists()
    ):
        train()
    reg = joblib.load(REGRESSOR_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return reg, scaler, le


def predict(features: dict) -> dict:
    """
    Predict agricultural productivity for a given set of features.

    Args:
        features: Dictionary with keys matching FEATURE_NAMES.
                  'crop_type' can be a string (e.g. 'soy') or numeric.

    Returns:
        Dictionary with 'predicted_yield_tons_ha' and 'interpretation'.
    """
    reg, scaler, le = load_model()

    # Encode crop type if provided as string
    crop_type = features.get("crop_type", "soy")
    if isinstance(crop_type, str):
        try:
            crop_encoded = float(le.transform([crop_type])[0])
        except ValueError:
            crop_encoded = 0.0
    else:
        crop_encoded = float(crop_type)

    x = np.array(
        [
            [
                features.get("ndvi_mean", 0.5),
                features.get("ndvi_p50", 0.5),
                features.get("soil_moisture", 0.5),
                features.get("temperature_avg", 25.0),
                features.get("precipitation_mm", 400.0),
                crop_encoded,
                features.get("days_to_harvest", 120.0),
            ]
        ]
    )
    x_scaled = scaler.transform(x)
    yield_pred = float(reg.predict(x_scaled)[0])
    yield_pred = max(0.5, yield_pred)

    if yield_pred >= 8:
        interp = "High productivity expected"
    elif yield_pred >= 4:
        interp = "Moderate productivity expected"
    else:
        interp = "Low productivity — consider agronomic interventions"

    return {
        "predicted_yield_tons_ha": round(yield_pred, 2),
        "interpretation": interp,
        "crop_type": crop_type,
    }
