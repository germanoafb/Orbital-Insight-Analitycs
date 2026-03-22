"""
Data processing utilities for satellite imagery and geospatial data.
Provides normalization, feature extraction, and helper functions used
across the backend and ML pipeline.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def normalize_array(arr: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """
    Min-max normalize a numpy array to [vmin, vmax].

    Args:
        arr:  Input array.
        vmin: Target minimum value.
        vmax: Target maximum value.

    Returns:
        Normalized array.
    """
    a_min, a_max = arr.min(), arr.max()
    if a_max == a_min:
        return np.full_like(arr, vmin, dtype=float)
    normalized = (arr - a_min) / (a_max - a_min)
    return normalized * (vmax - vmin) + vmin


def extract_ndvi_features(ndvi: np.ndarray) -> dict:
    """
    Extract statistical features from an NDVI array for use in ML models.

    Args:
        ndvi: 2-D NDVI array.

    Returns:
        Dictionary of features (mean, std, p25, p50, p75, fraction_veg,
        fraction_barren, fraction_water).
    """
    flat = ndvi.flatten()
    return {
        "ndvi_mean": float(np.mean(flat)),
        "ndvi_std": float(np.std(flat)),
        "ndvi_p25": float(np.percentile(flat, 25)),
        "ndvi_p50": float(np.percentile(flat, 50)),
        "ndvi_p75": float(np.percentile(flat, 75)),
        "fraction_vegetation": float(np.mean(flat >= 0.3)),
        "fraction_barren": float(np.mean((flat >= 0) & (flat < 0.3))),
        "fraction_water": float(np.mean(flat < 0)),
    }


def compute_change_metrics(before: np.ndarray, after: np.ndarray) -> dict:
    """
    Compute temporal-change metrics between two NDVI arrays.

    Args:
        before: NDVI array from the earlier period.
        after:  NDVI array from the later period.

    Returns:
        Dictionary with mean_change, loss_fraction, gain_fraction,
        and stable_fraction.
    """
    diff = after - before  # Positive = gain, negative = loss
    return {
        "mean_change": float(np.mean(diff)),
        "loss_fraction": float(np.mean(diff < -0.1)),
        "gain_fraction": float(np.mean(diff > 0.1)),
        "stable_fraction": float(np.mean(np.abs(diff) <= 0.1)),
        "max_loss": float(np.min(diff)),
        "max_gain": float(np.max(diff)),
    }


def dataframe_from_ndvi_timeseries(
    ndvi_series: list,
    timestamps: list,
) -> pd.DataFrame:
    """
    Build a DataFrame of NDVI statistics from a time-series of NDVI arrays.

    Args:
        ndvi_series: List of 2-D NDVI arrays ordered chronologically.
        timestamps:  Corresponding timestamp strings or datetime objects.

    Returns:
        DataFrame with one row per timestamp and statistical columns.
    """
    rows = []
    for ts, ndvi in zip(timestamps, ndvi_series):
        features = extract_ndvi_features(ndvi)
        features["timestamp"] = ts
        rows.append(features)
    df = pd.DataFrame(rows)
    df = df.set_index("timestamp")
    return df


def train_test_split_temporal(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into train and test sets.
    Keeps temporal order (no shuffling).

    Args:
        df:            Temporally ordered DataFrame.
        test_fraction: Fraction of the data to use for testing.

    Returns:
        (train_df, test_df) tuple.
    """
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def simulate_land_use_map(
    height: int = 64,
    width: int = 64,
    urban_fraction: float = 0.25,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate a binary land-use classification map.
    Values: 0 = non-urban, 1 = urban.

    Args:
        height:         Map height in pixels.
        width:          Map width in pixels.
        urban_fraction: Fraction of pixels classified as urban.
        seed:           Random seed.

    Returns:
        2-D binary array.
    """
    rng = np.random.default_rng(seed)
    flat = np.zeros(height * width, dtype=int)
    n_urban = int(flat.size * urban_fraction)
    idx = rng.choice(flat.size, size=n_urban, replace=False)
    flat[idx] = 1
    return flat.reshape(height, width)
