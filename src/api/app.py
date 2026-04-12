"""
app.py — FastAPI REST API for the NatWest AI Predictive Forecasting engine.

Exposes three endpoints that wrap the core forecasting, anomaly detection,
and scenario modelling logic. Designed for easy integration with any frontend
(Streamlit, React, etc.) or external service.

Run with:
    uvicorn src.api.app:app --reload --port 8000
"""

import sys
import os
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so core modules are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.forecaster import Forecaster
from src.core.anomaly_detector import AnomalyDetector
from src.core.scenario_runner import ScenarioForecaster
from src.core.explainer import AnomalyExplainer

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NatWest AI Predictive Forecasting API",
    description=(
        "Lightweight REST API wrapping Holt-Winters forecasting, "
        "rolling Z-score anomaly detection, and what-if scenario modelling."
    ),
    version="1.0.0",
)

# Allow CORS for local Streamlit / React frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Request / Response Schemas
# =====================================================================

class DataPoint(BaseModel):
    """A single time-series observation."""
    date: str = Field(..., example="2023-10-15", description="Date string (YYYY-MM-DD)")
    value: float = Field(..., example=160.0, description="Metric value for this date")


class ForecastRequest(BaseModel):
    """Payload for the /forecast endpoint."""
    data: List[DataPoint] = Field(..., description="Historical time-series data points")
    target_column: str = Field(default="value", description="Name of the metric column")
    weeks_ahead: int = Field(default=4, ge=1, le=6, description="Forecast horizon (1–6 weeks)")


class ForecastRow(BaseModel):
    date: str
    baseline_avg: float
    likely_estimate: float
    low_bound: float
    high_bound: float


class ForecastResponse(BaseModel):
    forecast: List[ForecastRow]
    weeks_ahead: int
    method: str = "Holt-Winters Exponential Smoothing (additive trend)"


class AnomalyRequest(BaseModel):
    """Payload for the /detect-anomalies endpoint."""
    data: List[DataPoint]
    target_column: str = Field(default="value")
    window_size: int = Field(default=4, ge=2, description="Rolling window for baseline")
    z_score_threshold: float = Field(default=2.5, ge=1.0, description="Z-score cutoff")


class AnomalyRow(BaseModel):
    date: str
    actual_value: float
    expected_value: float
    z_score: float
    explanation: str


class AnomalyResponse(BaseModel):
    anomalies: List[AnomalyRow]
    total_found: int
    threshold_used: float


class ScenarioRequest(BaseModel):
    """Payload for the /scenario endpoint."""
    data: List[DataPoint]
    target_column: str = Field(default="value")
    weeks_ahead: int = Field(default=4, ge=1, le=6)
    percentage_change: float = Field(..., example=10.0, description="What-if modifier (e.g. 10.0 for +10%)")


class ScenarioRow(BaseModel):
    date: str
    likely_estimate: float
    scenario_value: float
    numerical_impact: float


class ScenarioResponse(BaseModel):
    scenario: List[ScenarioRow]
    percentage_change: float
    weeks_ahead: int


# =====================================================================
# Helper
# =====================================================================

def _build_dataframe(data: List[DataPoint], target_column: str) -> pd.DataFrame:
    """Convert a list of DataPoint objects into a pandas DataFrame."""
    records = [{"date": dp.date, target_column: dp.value} for dp in data]
    df = pd.DataFrame(records)
    if len(df) < 8:
        raise HTTPException(
            status_code=400,
            detail="At least 8 data points are required for reliable forecasting.",
        )
    return df


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/", tags=["Health"])
def health_check():
    """Simple health-check endpoint."""
    return {"status": "healthy", "service": "NatWest AI Forecasting API"}


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def forecast(request: ForecastRequest):
    """
    Generate a short-term forecast with uncertainty bounds.

    Accepts historical time-series data and returns a 1–6 week projection
    comparing a simple moving-average baseline with a Holt-Winters model.
    """
    df = _build_dataframe(request.data, request.target_column)

    try:
        fc = Forecaster(df, target_col=request.target_column, date_col="date")
        result = fc.generate_forecast(weeks_ahead=request.weeks_ahead)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Forecasting failed: {str(exc)}")

    rows = [
        ForecastRow(
            date=str(row["Date"].date()),
            baseline_avg=row["Baseline_Avg"],
            likely_estimate=row["Likely_Estimate"],
            low_bound=row["Low_Bound"],
            high_bound=row["High_Bound"],
        )
        for _, row in result.iterrows()
    ]

    return ForecastResponse(
        forecast=rows,
        weeks_ahead=request.weeks_ahead,
    )


@app.post("/detect-anomalies", response_model=AnomalyResponse, tags=["Anomaly Detection"])
def detect_anomalies(request: AnomalyRequest):
    """
    Scan historical data for anomalous spikes or dips.

    Returns flagged data points with Z-scores and AI-generated explanations.
    """
    df = _build_dataframe(request.data, request.target_column)

    try:
        detector = AnomalyDetector(df, target_col=request.target_column, date_col="date")
        anomalies = detector.detect_anomalies(
            window_size=request.window_size,
            dynamic_z_score_threshold=request.z_score_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Detection failed: {str(exc)}")

    explainer = AnomalyExplainer()
    rows = []
    for date_idx, row in anomalies.iterrows():
        date_str = str(date_idx.date()) if hasattr(date_idx, "date") else str(date_idx)
        actual = row[request.target_column]
        expected = row["Rolling_Mean"]
        z_score = row["Z_Score"]

        explanation = explainer.generate_explanation(
            date=date_str,
            actual_value=actual,
            expected_value=expected,
            z_score=z_score,
        )
        rows.append(AnomalyRow(
            date=date_str,
            actual_value=round(actual, 2),
            expected_value=round(expected, 2),
            z_score=round(z_score, 2),
            explanation=explanation,
        ))

    return AnomalyResponse(
        anomalies=rows,
        total_found=len(rows),
        threshold_used=request.z_score_threshold,
    )


@app.post("/scenario", response_model=ScenarioResponse, tags=["Scenario Modelling"])
def scenario(request: ScenarioRequest):
    """
    Apply a what-if percentage modifier to the baseline forecast.

    Returns side-by-side comparison of baseline vs. adjusted projections.
    """
    df = _build_dataframe(request.data, request.target_column)

    try:
        fc = Forecaster(df, target_col=request.target_column, date_col="date")
        forecast_df = fc.generate_forecast(weeks_ahead=request.weeks_ahead)
        scenario_df = ScenarioForecaster.apply_scenario(
            forecast_df, percentage_change=request.percentage_change
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Scenario modelling failed: {str(exc)}")

    scenario_col = f"Scenario_({request.percentage_change:+.1f}%)"
    rows = [
        ScenarioRow(
            date=str(row["Date"].date()),
            likely_estimate=row["Likely_Estimate"],
            scenario_value=row[scenario_col],
            numerical_impact=row["Numerical_Impact"],
        )
        for _, row in scenario_df.iterrows()
    ]

    return ScenarioResponse(
        scenario=rows,
        percentage_change=request.percentage_change,
        weeks_ahead=request.weeks_ahead,
    )
