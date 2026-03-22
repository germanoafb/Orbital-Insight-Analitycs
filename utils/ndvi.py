"""
Utility functions for NDVI (Normalized Difference Vegetation Index) calculations.
NDVI = (NIR - Red) / (NIR + Red)
Values range from -1 to 1, where higher values indicate denser vegetation.
"""

import numpy as np


def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Calculate NDVI from Near-Infrared (NIR) and Red bands.

    Args:
        nir: Near-Infrared band array.
        red: Red band array.

    Returns:
        NDVI array with values between -1 and 1.
    """
    nir = nir.astype(float)
    red = red.astype(float)
    denominator = nir + red
    # Avoid division by zero
    ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)
    return ndvi


def classify_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """
    Classify NDVI values into vegetation categories.

    Categories:
        0 - Water / No data     (NDVI < 0)
        1 - Bare soil / Urban   (0 <= NDVI < 0.2)
        2 - Sparse vegetation   (0.2 <= NDVI < 0.4)
        3 - Moderate vegetation (0.4 <= NDVI < 0.6)
        4 - Dense vegetation    (NDVI >= 0.6)

    Args:
        ndvi: NDVI array.

    Returns:
        Classification array with integer category labels.
    """
    conditions = [
        ndvi < 0,
        (ndvi >= 0) & (ndvi < 0.2),
        (ndvi >= 0.2) & (ndvi < 0.4),
        (ndvi >= 0.4) & (ndvi < 0.6),
        ndvi >= 0.6,
    ]
    choices = [0, 1, 2, 3, 4]
    return np.select(conditions, choices, default=0).astype(int)


def ndvi_statistics(ndvi: np.ndarray) -> dict:
    """
    Compute descriptive statistics for an NDVI array.

    Args:
        ndvi: NDVI array.

    Returns:
        Dictionary with mean, std, min, max, and vegetation coverage percentage.
    """
    valid = ndvi[~np.isnan(ndvi)]
    vegetation_pct = float(np.sum(valid >= 0.3) / valid.size * 100) if valid.size > 0 else 0.0
    return {
        "mean": float(np.mean(valid)) if valid.size > 0 else 0.0,
        "std": float(np.std(valid)) if valid.size > 0 else 0.0,
        "min": float(np.min(valid)) if valid.size > 0 else 0.0,
        "max": float(np.max(valid)) if valid.size > 0 else 0.0,
        "vegetation_coverage_pct": vegetation_pct,
    }


def simulate_satellite_bands(
    height: int = 64,
    width: int = 64,
    vegetation_fraction: float = 0.6,
    seed: int = 42,
) -> dict:
    """
    Simulate NIR and Red satellite bands as numpy arrays.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        vegetation_fraction: Proportion of pixels with high vegetation.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'nir', 'red', and 'ndvi' arrays.
    """
    rng = np.random.default_rng(seed)
    n_pixels = height * width
    n_veg = int(n_pixels * vegetation_fraction)

    nir_flat = np.zeros(n_pixels)
    red_flat = np.zeros(n_pixels)

    # Vegetation pixels: high NIR, low Red → high NDVI
    veg_idx = rng.choice(n_pixels, size=n_veg, replace=False)
    nir_flat[veg_idx] = rng.uniform(0.5, 0.9, size=n_veg)
    red_flat[veg_idx] = rng.uniform(0.05, 0.2, size=n_veg)

    # Non-vegetation pixels: low NIR, moderate Red → low/negative NDVI
    non_veg_idx = np.setdiff1d(np.arange(n_pixels), veg_idx)
    nir_flat[non_veg_idx] = rng.uniform(0.1, 0.3, size=len(non_veg_idx))
    red_flat[non_veg_idx] = rng.uniform(0.2, 0.5, size=len(non_veg_idx))

    nir = nir_flat.reshape(height, width)
    red = red_flat.reshape(height, width)
    ndvi = calculate_ndvi(nir, red)

    return {"nir": nir, "red": red, "ndvi": ndvi}
