"""
Agricultural productivity prediction API endpoints.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

from models import agriculture_model

router = APIRouter()

CROP_TYPES = agriculture_model.CROP_TYPES


class AgricultureRequest(BaseModel):
    """Input features for crop yield prediction."""

    ndvi_mean: float = Field(0.6, ge=-1, le=1, description="Mean NDVI of the field")
    ndvi_p50: float = Field(0.6, ge=-1, le=1, description="Median NDVI")
    soil_moisture: float = Field(0.5, ge=0, le=1, description="Volumetric soil moisture [0-1]")
    temperature_avg: float = Field(25.0, description="Average temperature in °C")
    precipitation_mm: float = Field(400.0, ge=0, description="Seasonal precipitation (mm)")
    crop_type: str = Field("soy", description=f"Crop type. One of: {CROP_TYPES}")
    days_to_harvest: float = Field(120.0, ge=30, le=365, description="Days until harvest")


class TrendRequest(BaseModel):
    """Request body for generating a multi-year productivity trend."""

    ndvi_values: List[float] = Field(
        ..., min_items=2, description="List of annual NDVI mean values"
    )
    soil_moisture: float = Field(0.5, ge=0, le=1)
    temperature_avg: float = Field(25.0)
    precipitation_mm: float = Field(400.0)
    crop_type: str = Field("soy")
    days_to_harvest: float = Field(120.0)
    start_year: int = Field(2015)


@router.get("/train", summary="Train agriculture regression model")
def train_model() -> dict:
    """
    Retrain the Gradient Boosting agriculture regressor on simulated data.
    Returns RMSE, MAE, R², and feature importances.
    """
    metrics = agriculture_model.train()
    return {"message": "Model trained successfully", "metrics": metrics}


@router.post("/predict", summary="Predict crop yield")
def predict_yield(request: AgricultureRequest) -> dict:
    """
    Predict crop yield (tons/ha) from agronomic and NDVI features.
    """
    if request.crop_type not in CROP_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid crop_type '{request.crop_type}'. Must be one of {CROP_TYPES}",
        )
    try:
        result = agriculture_model.predict(request.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@router.post("/trend", summary="Predict yield trend over multiple years")
def predict_trend(request: TrendRequest) -> dict:
    """
    Predict crop yield for each year given a sequence of NDVI values.
    Returns per-year predictions suitable for charting.
    """
    if request.crop_type not in CROP_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid crop_type '{request.crop_type}'. Must be one of {CROP_TYPES}",
        )

    years = [request.start_year + i for i in range(len(request.ndvi_values))]
    predictions = []
    for ndvi_val in request.ndvi_values:
        feat = {
            "ndvi_mean": ndvi_val,
            "ndvi_p50": ndvi_val,
            "soil_moisture": request.soil_moisture,
            "temperature_avg": request.temperature_avg,
            "precipitation_mm": request.precipitation_mm,
            "crop_type": request.crop_type,
            "days_to_harvest": request.days_to_harvest,
        }
        result = agriculture_model.predict(feat)
        predictions.append(result["predicted_yield_tons_ha"])

    return {
        "years": years,
        "ndvi_values": request.ndvi_values,
        "predicted_yield_tons_ha": predictions,
        "crop_type": request.crop_type,
    }


@router.get("/demo", summary="Run a demo agriculture prediction")
def demo() -> dict:
    """
    Run a demonstration prediction for a high-NDVI soy field.
    """
    req = AgricultureRequest(
        ndvi_mean=0.72,
        ndvi_p50=0.71,
        soil_moisture=0.65,
        temperature_avg=26.0,
        precipitation_mm=550.0,
        crop_type="soy",
        days_to_harvest=110.0,
    )
    return predict_yield(req)
