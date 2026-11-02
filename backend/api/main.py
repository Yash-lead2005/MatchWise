
import json
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "xgboost.joblib"
SCHEMA_PATH = BASE_DIR / "models" / "xgboost_schema.json"


# --------------------------------------------------
# 2. LOAD MODEL AND SCHEMA
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not SCHEMA_PATH.exists():
    raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r") as file:
    schema = json.load(file)

FEATURE_COLUMNS = schema["feature_columns"]
THRESHOLD = float(schema["threshold"])


# --------------------------------------------------
# 3. CALIBRATED DECISION THRESHOLDS
# Obtained from validation dataset
# --------------------------------------------------

AUTO_MATCH_THRESHOLD = 0.9975168704986572

AUTO_NON_MATCH_THRESHOLD = 0.91334404493331909


# --------------------------------------------------
# 4. SHARED DECISION FUNCTION
# --------------------------------------------------

def get_match_decision(probability):

    if probability >= AUTO_MATCH_THRESHOLD:

        return "AUTO_MATCH"

    elif probability <= AUTO_NON_MATCH_THRESHOLD:

        return "AUTO_NON_MATCH"

    else:

        return "HUMAN_REVIEW"


# --------------------------------------------------
# 5. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="MatchWise API",
    description="Duplicate product detection and human review API",
    version="1.0.0"
)


# --------------------------------------------------
# 6. CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# 7. REQUEST SCHEMAS
# --------------------------------------------------

class PredictionRequest(BaseModel):

    feature_values: dict[str, float | None] = Field(
        ...,
        description="Numerical features for a product pair"
    )


class ProductRecord(BaseModel):

    name: str = Field(
        ...,
        min_length=1,
        description="Product name"
    )

    description: str | None = None

    manufacturer: str | None = None

    price: str | float | None = None


class MatchRequest(BaseModel):

    product_a: ProductRecord

    product_b: ProductRecord


# --------------------------------------------------
# 8. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Welcome to MatchWise API",
        "status": "running",
        "version": "1.0.0"
    }


# --------------------------------------------------
# 9. HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": "XGBoost",
        "number_of_features": len(FEATURE_COLUMNS)
    }


# --------------------------------------------------
# 10. MODEL INFORMATION ENDPOINT
# --------------------------------------------------

@app.get("/model-info")
def model_info():

    return {

        "model_name": "XGBoost Classifier",

        "number_of_features": len(FEATURE_COLUMNS),

        "original_model_threshold": THRESHOLD,

        "auto_match_threshold": AUTO_MATCH_THRESHOLD,

        "auto_non_match_threshold": AUTO_NON_MATCH_THRESHOLD,

        "required_features": FEATURE_COLUMNS

    }


# --------------------------------------------------
# 11. PREDICTION ENDPOINT
# Accepts already-generated numerical features
# --------------------------------------------------

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

        if input_df.isin([
            float("inf"),
            float("-inf")
        ]).any().any():

            raise ValueError(
                "Infinite values are not allowed"
            )

        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        decision = get_match_decision(probability)

        return {

            "success": True,

            "model": "XGBoost",

            "match_probability": round(
                probability, 4
            ),

            "auto_match_threshold": round(
                AUTO_MATCH_THRESHOLD, 4
            ),

            "auto_non_match_threshold": round(
                AUTO_NON_MATCH_THRESHOLD, 4
            ),

            "decision": decision,

            "message": "Prediction generated successfully"

        }

    except (ValueError, TypeError) as error:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid feature values: {str(error)}"
        )


# --------------------------------------------------
# 12. STARTUP EVENT
# --------------------------------------------------

@app.on_event("startup")
def startup_event():

    print("\n========== MATCHWISE API ==========")

    print("Model: XGBoost")

    print("Features:", len(FEATURE_COLUMNS))

    print(
        "AUTO_MATCH threshold:",
        AUTO_MATCH_THRESHOLD
    )

    print(
        "AUTO_NON_MATCH threshold:",
        AUTO_NON_MATCH_THRESHOLD
    )

    print("API is ready!")

    print("===================================\n")


# --------------------------------------------------
# 13. RAW PRODUCT MATCHING ENDPOINT
# Day 16 + Day 17 improvements
# --------------------------------------------------

@app.post("/match")
def match_products(request: MatchRequest):

    from api.feature_builder import build_features

    try:

        # Convert Pydantic objects into dictionaries.
        product_a = request.product_a.model_dump()

        product_b = request.product_b.model_dump()

        # Generate the exact 19 model features.
        feature_df = build_features(
            product_a,
            product_b
        )

        # Generate XGBoost probability.
        probability = float(
            model.predict_proba(feature_df)[0][1]
        )

        # Apply calibrated three-level decision policy.
        decision = get_match_decision(probability)

        return {

            "success": True,

            "model": "XGBoost",

            "match_probability": round(
                probability, 4
            ),

            "auto_match_threshold": round(
                AUTO_MATCH_THRESHOLD, 4
            ),

            "auto_non_match_threshold": round(
                AUTO_NON_MATCH_THRESHOLD, 4
            ),

            "decision": decision,

            "features_generated": len(
                feature_df.columns
            ),

            "message": "Raw product matching completed successfully."

        }

    except (ValueError, TypeError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Matching failed: {str(error)}"
        )
