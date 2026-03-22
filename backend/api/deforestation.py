"""
Deforestation detection API endpoints.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np

from models import deforestation_model
from utils.ndvi import simulate_satellite_bands, ndvi_statistics, calculate_ndvi
from utils.data_processing import extract_ndvi_features, compute_change_metrics
from utils.visualization import ndvi_to_base64, deforestation_map_to_base64

router = APIRouter()


class DeforestationRequest(BaseModel):
    """Input features for deforestation prediction."""

    ndvi_mean: float = Field(0.5, ge=-1, le=1, description="Mean NDVI of the area")
    ndvi_std: float = Field(0.1, ge=0, description="Std dev of NDVI")
    ndvi_p25: float = Field(0.4, ge=-1, le=1, description="25th percentile NDVI")
    ndvi_p50: float = Field(0.5, ge=-1, le=1, description="Median NDVI")
    ndvi_p75: float = Field(0.6, ge=-1, le=1, description="75th percentile NDVI")
    fraction_vegetation: float = Field(0.6, ge=0, le=1)
    fraction_barren: float = Field(0.2, ge=0, le=1)
    fraction_water: float = Field(0.1, ge=0, le=1)
    ndvi_change: float = Field(0.0, ge=-2, le=2, description="NDVI change vs previous period")


class SimulateDeforestationRequest(BaseModel):
    """Parameters for simulating a deforestation scenario."""

    vegetation_before: float = Field(0.8, ge=0.05, le=1.0)
    vegetation_after: float = Field(0.3, ge=0.05, le=1.0)
    image_size: int = Field(64, ge=16, le=256, description="Image height/width in pixels")
    seed: int = Field(42)


@router.get("/train", summary="Train deforestation classifier")
def train_model() -> dict:
    """
    Retrain the deforestation Random Forest classifier on simulated data.
    Returns accuracy and evaluation metrics.
    """
    metrics = deforestation_model.train()
    return {"message": "Model trained successfully", "metrics": metrics}


@router.post("/predict", summary="Predict deforestation from features")
def predict_deforestation(request: DeforestationRequest) -> dict:
    """
    Predict whether an area is deforested based on NDVI-derived features.
    """
    try:
        result = deforestation_model.predict(request.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@router.post("/simulate", summary="Simulate a deforestation scenario")
def simulate_deforestation(request: SimulateDeforestationRequest) -> dict:
    """
    Simulate satellite imagery before and after deforestation.
    Returns NDVI statistics, change metrics, and base64-encoded maps.
    """
    size = request.image_size

    before_bands = simulate_satellite_bands(size, size, request.vegetation_before, seed=request.seed)
    after_bands = simulate_satellite_bands(
        size, size, request.vegetation_after, seed=request.seed + 1000
    )

    ndvi_before = before_bands["ndvi"]
    ndvi_after = after_bands["ndvi"]

    stats_before = ndvi_statistics(ndvi_before)
    stats_after = ndvi_statistics(ndvi_after)
    change = compute_change_metrics(ndvi_before, ndvi_after)

    # Run prediction on aggregate features
    features = extract_ndvi_features(ndvi_after)
    features["ndvi_change"] = change["mean_change"]
    prediction = deforestation_model.predict(features)

    # Generate comparison image (base64)
    image_b64 = deforestation_map_to_base64(ndvi_before, ndvi_after)

    return {
        "ndvi_stats_before": stats_before,
        "ndvi_stats_after": stats_after,
        "change_metrics": change,
        "prediction": prediction,
        "comparison_image_base64": image_b64,
    }


@router.get("/demo", summary="Run a demo deforestation analysis")
def demo() -> dict:
    """
    Run a demo analysis with pre-defined parameters.
    Returns a full deforestation report.
    """
    req = SimulateDeforestationRequest(
        vegetation_before=0.82,
        vegetation_after=0.28,
        image_size=64,
        seed=42,
    )
    return simulate_deforestation(req)
