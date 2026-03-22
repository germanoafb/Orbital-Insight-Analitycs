"""
FastAPI backend for Orbital Insight Analytics.

Provides REST API endpoints for:
  - Deforestation detection  (/api/deforestation)
  - Agricultural productivity prediction  (/api/agriculture)
  - Urban growth monitoring  (/api/urban)
  - Data simulation  (/api/simulate)
"""

import sys
import os
from pathlib import Path

# Ensure project root is on the Python path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import io

from backend.api.deforestation import router as deforestation_router
from backend.api.agriculture import router as agriculture_router
from backend.api.urban import router as urban_router

app = FastAPI(
    title="Orbital Insight Analytics API",
    description=(
        "REST API for satellite data analysis: "
        "deforestation detection, agricultural productivity, and urban growth monitoring."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount feature routers
app.include_router(deforestation_router, prefix="/api/deforestation", tags=["Deforestation"])
app.include_router(agriculture_router, prefix="/api/agriculture", tags=["Agriculture"])
app.include_router(urban_router, prefix="/api/urban", tags=["Urban Growth"])


@app.get("/", summary="Health check")
def root() -> dict:
    """Return a simple health-check response."""
    return {
        "status": "ok",
        "service": "Orbital Insight Analytics API",
        "version": "1.0.0",
    }


@app.get("/api/health", summary="Detailed health check")
def health() -> dict:
    """Return detailed service status."""
    return {
        "status": "healthy",
        "endpoints": [
            "/api/deforestation",
            "/api/agriculture",
            "/api/urban",
        ],
    }
