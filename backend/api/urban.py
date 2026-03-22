"""
Urban growth monitoring API endpoints.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np

from models import urban_model
from utils.ndvi import simulate_satellite_bands
from utils.data_processing import simulate_land_use_map
from utils.visualization import urban_growth_map_to_base64

router = APIRouter()


class SimulateUrbanRequest(BaseModel):
    """Parameters for simulating urban growth between two time periods."""

    urban_fraction_before: float = Field(0.2, ge=0.01, le=0.99)
    urban_fraction_after: float = Field(0.4, ge=0.01, le=0.99)
    image_size: int = Field(64, ge=16, le=256)
    seed: int = Field(42)


class UrbanStatsRequest(BaseModel):
    """Direct input of urban pixel counts for growth statistics."""

    urban_pixels_before: int = Field(..., ge=0)
    urban_pixels_after: int = Field(..., ge=0)
    total_pixels: int = Field(..., ge=1)


@router.get("/train", summary="Train urban classification model")
def train_model() -> dict:
    """
    Retrain the Random Forest urban land-use classifier.
    Returns accuracy and evaluation metrics.
    """
    metrics = urban_model.train()
    return {"message": "Model trained successfully", "metrics": metrics}


@router.post("/simulate", summary="Simulate urban growth scenario")
def simulate_urban(request: SimulateUrbanRequest) -> dict:
    """
    Simulate two satellite images (before/after) and classify urban areas.
    Returns growth statistics and a base64-encoded comparison map.
    """
    size = request.image_size

    # Simulate satellite bands for before period
    before_bands = simulate_satellite_bands(
        size, size,
        vegetation_fraction=1.0 - request.urban_fraction_before,
        seed=request.seed,
    )
    # Simulate satellite bands for after period (more urban = less vegetation)
    after_bands = simulate_satellite_bands(
        size, size,
        vegetation_fraction=1.0 - request.urban_fraction_after,
        seed=request.seed + 1000,
    )

    # Classify urban pixels using the ML model
    before_map = urban_model.predict_map(
        before_bands["ndvi"], before_bands["red"], before_bands["nir"]
    )
    after_map = urban_model.predict_map(
        after_bands["ndvi"], after_bands["red"], after_bands["nir"]
    )

    # Compute growth statistics
    stats = urban_model.compute_urban_growth_stats(before_map, after_map)

    # Generate comparison image
    image_b64 = urban_growth_map_to_base64(before_map, after_map)

    return {
        "growth_stats": stats,
        "comparison_image_base64": image_b64,
    }


@router.post("/stats", summary="Compute urban growth statistics")
def compute_stats(request: UrbanStatsRequest) -> dict:
    """
    Compute urban growth statistics directly from pixel counts.
    """
    if request.urban_pixels_before > request.total_pixels:
        raise HTTPException(
            status_code=422,
            detail="urban_pixels_before cannot exceed total_pixels",
        )
    if request.urban_pixels_after > request.total_pixels:
        raise HTTPException(
            status_code=422,
            detail="urban_pixels_after cannot exceed total_pixels",
        )

    growth_pct = (
        (request.urban_pixels_after - request.urban_pixels_before)
        / max(request.urban_pixels_before, 1)
        * 100
    )
    return {
        "urban_before_pct": request.urban_pixels_before / request.total_pixels * 100,
        "urban_after_pct": request.urban_pixels_after / request.total_pixels * 100,
        "growth_pct": growth_pct,
        "new_urban_pixels": max(0, request.urban_pixels_after - request.urban_pixels_before),
    }


@router.get("/demo", summary="Run a demo urban growth analysis")
def demo() -> dict:
    """
    Run a demonstration urban growth simulation.
    """
    req = SimulateUrbanRequest(
        urban_fraction_before=0.15,
        urban_fraction_after=0.38,
        image_size=64,
        seed=42,
    )
    return simulate_urban(req)
