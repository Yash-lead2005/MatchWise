
import json
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "xgboost.joblib"
SCHEMA_PATH = BASE_DIR / "models" / "xgboost_schema.json"


# Load model and schema
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not SCHEMA_PATH.exists():
    raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r") as file:
    schema = json.load(file)

FEATURE_COLUMNS = schema["feature_columns"]
THRESHOLD = float(schema["threshold"])

REVIEW_MARGIN = 0.15
LOWER_THRESHOLD = max(0.0, THRESHOLD - REVIEW_MARGIN)
UPPER_THRESHOLD = min(1.0, THRESHOLD + REVIEW_MARGIN)


# Create FastAPI application
app = FastAPI(
    title="MatchWise API",
    description="Duplicate product detection and human review API",
    version="1.0.0"
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Request schema
class PredictionRequest(BaseModel):
    feature_values: dict[str, float | None] = Field(
        ...,
        description="Numerical features for a product pair"
    )


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to MatchWise API",
        "status": "running",
        "version": "1.0.0"
    }


# Health endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": "XGBoost",
        "number_of_features": len(FEATURE_COLUMNS)
    }


# Model information endpoint
@app.get("/model-info")
def model_info():
    return {
        "model_name": "XGBoost Classifier",
        "number_of_features": len(FEATURE_COLUMNS),
        "threshold": THRESHOLD,
        "review_margin": REVIEW_MARGIN,
        "required_features": FEATURE_COLUMNS
    }


# Prediction endpoint
@app.post("/predict")
def predict_match(request: PredictionRequest):

    received_features = request.feature_values

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in received_features
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features",
                "missing_features": missing_features
            }
        )

    try:
        feature_data = {
            feature: received_features[feature]
            for feature in FEATURE_COLUMNS
        }

        input_df = pd.DataFrame(
            [feature_data],
            columns=FEATURE_COLUMNS
        )

        for column in FEATURE_COLUMNS:
            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="raise"
            )

        if input_df.isin([float("inf"), float("-inf")]).any().any():
            raise ValueError("Infinite values are not allowed")

        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        if probability >= UPPER_THRESHOLD:
            decision = "AUTO_MATCH"

        elif probability <= LOWER_THRESHOLD:
            decision = "AUTO_NON_MATCH"

        else:
            decision = "HUMAN_REVIEW"

        return {
            "success": True,
            "model": "XGBoost",
            "match_probability": round(probability, 4),
            "threshold": round(THRESHOLD, 4),
            "decision": decision,
            "message": "Prediction generated successfully"
        }

    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feature values: {str(error)}"
        )


# Startup event
@app.on_event("startup")
def startup_event():
    print("\n========== MATCHWISE API ==========")
    print("Model: XGBoost")
    print("Features:", len(FEATURE_COLUMNS))
    print("Threshold:", THRESHOLD)
    print("API is ready!")
    print("===================================\n")
