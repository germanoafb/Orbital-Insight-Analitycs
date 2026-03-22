"""
Data simulation scripts for Orbital Insight Analytics.

Generates realistic synthetic satellite datasets for:
    1. Deforestation time series (NDVI before/after)
    2. Agricultural productivity records
    3. Urban growth maps

All outputs are saved to the /data directory as CSV and NumPy .npy files.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Allow imports from the project root when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ndvi import simulate_satellite_bands, calculate_ndvi
from utils.data_processing import simulate_land_use_map

DATA_DIR = Path(__file__).parent
DATA_DIR.mkdir(exist_ok=True)


# ── 1. Deforestation dataset ──────────────────────────────────────────────────

def simulate_deforestation_series(
    n_regions: int = 50,
    n_years: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulate a multi-year NDVI time series for several Amazon-like regions.

    Each region starts with a high vegetation fraction and progressively loses
    vegetation over time (mimicking deforestation pressure).

    Args:
        n_regions: Number of geographic regions.
        n_years:   Observation years (starting from 2015).
        seed:      Random seed.

    Returns:
        DataFrame with columns:
            region_id, year, ndvi_mean, ndvi_std, vegetation_fraction,
            deforested (binary label).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for region_id in range(n_regions):
        # Starting vegetation fraction (0.5 – 0.95)
        initial_veg = rng.uniform(0.5, 0.95)
        # Annual deforestation rate (0 – 8 %)
        annual_loss = rng.uniform(0.0, 0.08)

        for year_idx in range(n_years):
            year = 2015 + year_idx
            veg_frac = max(0.05, initial_veg - annual_loss * year_idx)
            bands = simulate_satellite_bands(
                height=32, width=32,
                vegetation_fraction=veg_frac,
                seed=seed + region_id * 100 + year_idx,
            )
            ndvi = bands["ndvi"]
            ndvi_mean = float(np.mean(ndvi))
            ndvi_std = float(np.std(ndvi))
            deforested = 1 if veg_frac < 0.35 else 0

            rows.append(
                {
                    "region_id": region_id,
                    "year": year,
                    "ndvi_mean": round(ndvi_mean, 4),
                    "ndvi_std": round(ndvi_std, 4),
                    "vegetation_fraction": round(veg_frac, 4),
                    "annual_loss_rate": round(annual_loss, 4),
                    "deforested": deforested,
                }
            )

    return pd.DataFrame(rows)


# ── 2. Agricultural productivity dataset ─────────────────────────────────────

def simulate_agriculture_dataset(n_samples: int = 1000, seed: int = 1) -> pd.DataFrame:
    """
    Simulate a crop yield dataset with realistic feature correlations.

    Args:
        n_samples: Number of field observations.
        seed:      Random seed.

    Returns:
        DataFrame with agronomic features and target yield_tons_ha.
    """
    rng = np.random.default_rng(seed)
    crop_types = ["soy", "corn", "wheat", "cotton", "sugarcane"]
    crop_multipliers = {"soy": 3.5, "corn": 5.0, "wheat": 3.0, "cotton": 1.8, "sugarcane": 12.0}

    ndvi_mean = rng.uniform(0.2, 0.9, n_samples)
    soil_moisture = rng.uniform(0.1, 0.9, n_samples)
    temperature_avg = rng.uniform(15, 38, n_samples)
    precipitation_mm = rng.uniform(50, 900, n_samples)
    crop_type = rng.choice(crop_types, n_samples)
    days_to_harvest = rng.uniform(60, 180, n_samples)

    base_yield = (
        2.0
        + 6.0 * ndvi_mean
        + 2.5 * soil_moisture
        - 0.07 * (temperature_avg - 25) ** 2
        + 0.004 * precipitation_mm
        - 0.008 * days_to_harvest
    )
    multiplier = np.array([crop_multipliers[c] for c in crop_type]) / 3.5
    yield_tons_ha = np.clip(base_yield * multiplier + rng.normal(0, 0.4, n_samples), 0.3, 30.0)

    return pd.DataFrame(
        {
            "ndvi_mean": ndvi_mean.round(4),
            "soil_moisture": soil_moisture.round(4),
            "temperature_avg_c": temperature_avg.round(2),
            "precipitation_mm": precipitation_mm.round(1),
            "crop_type": crop_type,
            "days_to_harvest": days_to_harvest.round(0).astype(int),
            "yield_tons_ha": yield_tons_ha.round(3),
        }
    )


# ── 3. Urban growth dataset ───────────────────────────────────────────────────

def simulate_urban_growth_dataset(n_cities: int = 30, seed: int = 2) -> pd.DataFrame:
    """
    Simulate urban expansion statistics for a set of cities over a decade.

    Args:
        n_cities: Number of cities.
        seed:     Random seed.

    Returns:
        DataFrame with per-city annual urban area statistics.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for city_id in range(n_cities):
        initial_urban_km2 = rng.uniform(5, 500)
        annual_growth_pct = rng.uniform(0.5, 8.0)
        population_k = rng.uniform(10, 5000)

        for year_idx in range(10):
            year = 2015 + year_idx
            urban_area_km2 = initial_urban_km2 * ((1 + annual_growth_pct / 100) ** year_idx)
            rows.append(
                {
                    "city_id": city_id,
                    "year": year,
                    "urban_area_km2": round(urban_area_km2, 2),
                    "annual_growth_pct": round(annual_growth_pct, 2),
                    "population_thousands": round(population_k, 1),
                }
            )

    return pd.DataFrame(rows)


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_satellite_images(
    n_images: int = 10,
    height: int = 64,
    width: int = 64,
    seed: int = 99,
) -> None:
    """
    Save simulated satellite NDVI arrays as .npy files.

    Args:
        n_images: Number of image pairs (before / after) to generate.
        height:   Image height in pixels.
        width:    Image width in pixels.
        seed:     Base random seed.
    """
    img_dir = DATA_DIR / "satellite_images"
    img_dir.mkdir(exist_ok=True)

    for i in range(n_images):
        veg_before = np.random.default_rng(seed + i).uniform(0.5, 0.9)
        veg_after = max(0.1, veg_before - np.random.default_rng(seed + i + 1000).uniform(0, 0.4))

        before = simulate_satellite_bands(height, width, veg_before, seed=seed + i)
        after = simulate_satellite_bands(height, width, veg_after, seed=seed + i + 500)

        np.save(img_dir / f"ndvi_before_{i:02d}.npy", before["ndvi"])
        np.save(img_dir / f"ndvi_after_{i:02d}.npy", after["ndvi"])

    print(f"Saved {n_images} NDVI image pairs to {img_dir}")


def run_all(seed: int = 42) -> None:
    """Generate and save all synthetic datasets."""
    print("Generating deforestation time series …")
    df_def = simulate_deforestation_series(seed=seed)
    df_def.to_csv(DATA_DIR / "deforestation_dataset.csv", index=False)
    print(f"  → {len(df_def)} rows saved to deforestation_dataset.csv")

    print("Generating agricultural productivity dataset …")
    df_agr = simulate_agriculture_dataset(seed=seed)
    df_agr.to_csv(DATA_DIR / "agriculture_dataset.csv", index=False)
    print(f"  → {len(df_agr)} rows saved to agriculture_dataset.csv")

    print("Generating urban growth dataset …")
    df_urb = simulate_urban_growth_dataset(seed=seed)
    df_urb.to_csv(DATA_DIR / "urban_growth_dataset.csv", index=False)
    print(f"  → {len(df_urb)} rows saved to urban_growth_dataset.csv")

    print("Generating satellite NDVI image files …")
    save_satellite_images(seed=seed)

    print("\nAll datasets generated successfully.")


if __name__ == "__main__":
    run_all()
