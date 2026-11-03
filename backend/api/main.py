
import json
from pathlib import Path

import joblib
import pandas as pd

from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from api.database import (
    save_matching_record,
    get_connection
)


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
# Calibrated thresholds for the new 15-feature XGBoost model

AUTO_MATCH_THRESHOLD = 0.9971
AUTO_NON_MATCH_THRESHOLD = 0.9005

UPPER_THRESHOLD = AUTO_MATCH_THRESHOLD
LOWER_THRESHOLD = AUTO_NON_MATCH_THRESHOLD

REVIEW_MARGIN = (
    AUTO_MATCH_THRESHOLD - AUTO_NON_MATCH_THRESHOLD
) / 2


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
    
class ReviewDecisionRequest(BaseModel):

    decision: Literal["MATCH", "NON_MATCH"] = Field(
        ...,
        description="Human review decision"
    )

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
        if probability >= UPPER_THRESHOLD:

            decision = "AUTO_MATCH"

        elif probability <= LOWER_THRESHOLD:

            decision = "AUTO_NON_MATCH"

        else:

            decision = "HUMAN_REVIEW"

        # Save comparison and prediction in SQLite.
        record_id = save_matching_record(
            product_a=product_a,
            product_b=product_b,
            probability=probability,
            model_decision=decision,
            auto_match_threshold=UPPER_THRESHOLD,
            auto_non_match_threshold=LOWER_THRESHOLD
        )

        # Return prediction and database record ID.
        return {

            "success": True,

            "record_id": record_id,

            "model": "XGBoost",

            "match_probability": round(
                probability, 4
            ),

            "auto_match_threshold": round(
                UPPER_THRESHOLD, 4
            ),

            "auto_non_match_threshold": round(
                LOWER_THRESHOLD, 4
            ),

            "decision": decision,

            "review_status": (
                "PENDING"
                if decision == "HUMAN_REVIEW"
                else "NOT_REQUIRED"
            ),

            "features_generated": len(
                feature_df.columns
            ),

            "message": "Product matching completed and saved successfully."

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



# --------------------------------------------------
# DAY 20 - STEP 3: PENDING HUMAN REVIEW API
# --------------------------------------------------

@app.get("/reviews/pending")
def get_pending_reviews():

    try:

        with get_connection() as connection:

            rows = connection.execute("""

                SELECT
                    id,
                    product_a,
                    product_b,
                    match_probability,
                    model_decision,
                    review_status,
                    auto_match_threshold,
                    auto_non_match_threshold,
                    created_at

                FROM matching_records

                WHERE review_status = 'PENDING'

                ORDER BY created_at DESC

            """).fetchall()

        reviews = []

        for row in rows:

            reviews.append({

                "id": row["id"],

                "product_a": json.loads(
                    row["product_a"]
                ),

                "product_b": json.loads(
                    row["product_b"]
                ),

                "match_probability": round(
                    row["match_probability"], 4
                ),

                "model_decision": row["model_decision"],

                "review_status": row["review_status"],

                "auto_match_threshold": row[
                    "auto_match_threshold"
                ],

                "auto_non_match_threshold": row[
                    "auto_non_match_threshold"
                ],

                "created_at": row["created_at"]

            })

        return {

            "success": True,

            "total_pending": len(reviews),

            "reviews": reviews

        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve pending reviews: {str(error)}"
        )
        
        

# --------------------------------------------------
# DAY 20 - STEP 4: HUMAN APPROVE / REJECT API
# --------------------------------------------------

@app.post("/reviews/{record_id}/decision")
def submit_review_decision(
    record_id: int,
    request: ReviewDecisionRequest
):

    try:

        with get_connection() as connection:

            # Check whether the record exists and is pending.
            row = connection.execute("""

                SELECT id, review_status, model_decision
                FROM matching_records
                WHERE id = ?

            """, (record_id,)).fetchone()

            if row is None:

                raise HTTPException(
                    status_code=404,
                    detail="Matching record not found."
                )

            if row["review_status"] != "PENDING":

                raise HTTPException(
                    status_code=409,
                    detail="This record has already been reviewed or does not require review."
                )

            # Save the human decision separately.
            connection.execute("""

                UPDATE matching_records

                SET
                    human_decision = ?,
                    review_status = 'REVIEWED',
                    reviewed_at = STRFTIME(
                        '%Y-%m-%dT%H:%M:%fZ',
                        'NOW'
                    )

                WHERE id = ?
                AND review_status = 'PENDING'

            """, (
                request.decision,
                record_id
            ))

        return {

            "success": True,

            "record_id": record_id,

            "model_decision": row["model_decision"],

            "human_decision": request.decision,

            "review_status": "REVIEWED",

            "message": "Human review decision saved successfully."

        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to save review decision: {str(error)}"
        )



# --------------------------------------------------
# DAY 20 - STEP 5: MATCHING HISTORY API
# --------------------------------------------------

@app.get("/history")
def get_matching_history(
    limit: int = 50,
    offset: int = 0
):

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 100."
        )

    if offset < 0:
        raise HTTPException(
            status_code=400,
            detail="Offset cannot be negative."
        )

    try:

        with get_connection() as connection:

            # Total number of saved comparisons.
            total_records = connection.execute("""

                SELECT COUNT(*)
                FROM matching_records

            """).fetchone()[0]

            # Retrieve matching history.
            rows = connection.execute("""

                SELECT
                    id,
                    product_a,
                    product_b,
                    match_probability,
                    model_decision,
                    human_decision,
                    review_status,
                    auto_match_threshold,
                    auto_non_match_threshold,
                    created_at,
                    reviewed_at

                FROM matching_records

                ORDER BY id DESC

                LIMIT ? OFFSET ?

            """, (limit, offset)).fetchall()

        history = []

        for row in rows:

            # Human decision takes priority when available.
            final_decision = (
                row["human_decision"]
                if row["human_decision"] is not None
                else row["model_decision"]
            )

            history.append({

                "id": row["id"],

                "product_a": json.loads(
                    row["product_a"]
                ),

                "product_b": json.loads(
                    row["product_b"]
                ),

                "match_probability": round(
                    row["match_probability"], 4
                ),

                "model_decision": row["model_decision"],

                "human_decision": row["human_decision"],

                "final_decision": final_decision,

                "review_status": row["review_status"],

                "auto_match_threshold": row[
                    "auto_match_threshold"
                ],

                "auto_non_match_threshold": row[
                    "auto_non_match_threshold"
                ],

                "created_at": row["created_at"],

                "reviewed_at": row["reviewed_at"]

            })

        return {

            "success": True,

            "total_records": total_records,

            "returned_records": len(history),

            "limit": limit,

            "offset": offset,

            "history": history

        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve matching history: {str(error)}"
        )


# --------------------------------------------------
# DAY 20 - STEP 6: ANALYTICS API
# --------------------------------------------------

@app.get("/analytics")
def get_matching_analytics():

    try:

        with get_connection() as connection:

            stats = connection.execute("""

                SELECT

                    COUNT(*) AS total_comparisons,

                    SUM(
                        CASE
                            WHEN model_decision = 'AUTO_MATCH'
                            THEN 1 ELSE 0
                        END
                    ) AS auto_matches,

                    SUM(
                        CASE
                            WHEN model_decision = 'AUTO_NON_MATCH'
                            THEN 1 ELSE 0
                        END
                    ) AS auto_non_matches,

                    SUM(
                        CASE
                            WHEN review_status = 'PENDING'
                            THEN 1 ELSE 0
                        END
                    ) AS pending_reviews,

                    SUM(
                        CASE
                            WHEN review_status = 'REVIEWED'
                            THEN 1 ELSE 0
                        END
                    ) AS completed_reviews,

                    SUM(
                        CASE
                            WHEN human_decision = 'MATCH'
                            THEN 1 ELSE 0
                        END
                    ) AS human_confirmed_matches,

                    SUM(
                        CASE
                            WHEN human_decision = 'NON_MATCH'
                            THEN 1 ELSE 0
                        END
                    ) AS human_confirmed_non_matches,

                    AVG(match_probability)
                    AS average_match_probability

                FROM matching_records

            """).fetchone()

        return {

            "success": True,

            "total_comparisons":
                stats["total_comparisons"] or 0,

            "auto_matches":
                stats["auto_matches"] or 0,

            "auto_non_matches":
                stats["auto_non_matches"] or 0,

            "pending_reviews":
                stats["pending_reviews"] or 0,

            "completed_reviews":
                stats["completed_reviews"] or 0,

            "human_confirmed_matches":
                stats["human_confirmed_matches"] or 0,

            "human_confirmed_non_matches":
                stats["human_confirmed_non_matches"] or 0,

            "average_match_probability": round(
                stats["average_match_probability"] or 0,
                4
            )

        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve analytics: {str(error)}"
        )
