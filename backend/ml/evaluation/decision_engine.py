
import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "data" / "reports"

PREDICTION_FILE = DATA_DIR / "xgboost_test_predictions.csv"
SCHEMA_FILE = MODEL_DIR / "xgboost_schema.json"

OUTPUT_DIR = DATA_DIR / "review"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

# Review cases close to the model's selected threshold.
REVIEW_MARGIN = 0.15


# --------------------------------------------------
# 3. LOAD MODEL SCHEMA
# --------------------------------------------------

if not SCHEMA_FILE.exists():
    raise FileNotFoundError(
        f"Model schema not found: {SCHEMA_FILE}"
    )

with open(SCHEMA_FILE, "r") as file:
    schema = json.load(file)

threshold = float(schema["threshold"])

if not 0 <= threshold <= 1:
    raise ValueError("Model threshold must be between 0 and 1.")

lower_threshold = max(0.0, threshold - REVIEW_MARGIN)
upper_threshold = min(1.0, threshold + REVIEW_MARGIN)

print("Model threshold:", round(threshold, 4))
print("Lower boundary:", round(lower_threshold, 4))
print("Upper boundary:", round(upper_threshold, 4))


# --------------------------------------------------
# 4. LOAD PREDICTIONS
# --------------------------------------------------

if not PREDICTION_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {PREDICTION_FILE}"
    )

df = pd.read_csv(PREDICTION_FILE)

required_columns = [
    "match_probability",
    "predicted_label"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df.empty:
    raise ValueError("Prediction file is empty.")

df["match_probability"] = pd.to_numeric(
    df["match_probability"],
    errors="coerce"
)

if df["match_probability"].isna().any():
    raise ValueError("Invalid match probabilities found.")

if not df["match_probability"].between(0, 1).all():
    raise ValueError("Probabilities must be between 0 and 1.")


# --------------------------------------------------
# 5. CREATE REVIEW DECISIONS
# --------------------------------------------------

def assign_decision(probability):

    if probability >= upper_threshold:
        return "AUTO_MATCH"

    elif probability <= lower_threshold:
        return "AUTO_NON_MATCH"

    else:
        return "HUMAN_REVIEW"


df["decision_status"] = df["match_probability"].apply(
    assign_decision
)


# --------------------------------------------------
# 6. PREPARE HUMAN REVIEW QUEUE
# --------------------------------------------------

review_queue = df[
    df["decision_status"] == "HUMAN_REVIEW"
].copy()

review_queue = review_queue.sort_values(
    by="match_probability",
    ascending=False
)

review_queue = review_queue.reset_index(drop=True)

review_queue.insert(
    0,
    "review_id",
    [f"REV-{i + 1:05d}" for i in range(len(review_queue))]
)

review_queue["review_status"] = "PENDING"
review_queue["human_decision"] = ""
review_queue["reviewer_note"] = ""


# --------------------------------------------------
# 7. CREATE SEPARATE DECISION BUCKETS
# --------------------------------------------------

auto_matches = df[
    df["decision_status"] == "AUTO_MATCH"
].copy()

auto_non_matches = df[
    df["decision_status"] == "AUTO_NON_MATCH"
].copy()


# --------------------------------------------------
# 8. REMOVE GROUND-TRUTH LABELS FROM REVIEW DATA
# --------------------------------------------------

# The reviewer must not see the answer key.
# Actual labels are retained only in the original
# evaluation predictions file.

for output_df in [
    review_queue,
    auto_matches,
    auto_non_matches
]:
    if "actual_label" in output_df.columns:
        output_df.drop(
            columns=["actual_label"],
            inplace=True
        )


# --------------------------------------------------
# 9. SAVE OUTPUT FILES
# --------------------------------------------------

review_queue.to_csv(
    OUTPUT_DIR / "human_review_queue.csv",
    index=False
)

auto_matches.to_csv(
    OUTPUT_DIR / "auto_match_candidates.csv",
    index=False
)

auto_non_matches.to_csv(
    OUTPUT_DIR / "auto_non_match_candidates.csv",
    index=False
)


# --------------------------------------------------
# 10. GENERATE SUMMARY
# --------------------------------------------------

total_records = len(df)

summary = {
    "model": "XGBoost",
    "model_threshold": threshold,
    "review_margin": REVIEW_MARGIN,
    "lower_boundary": lower_threshold,
    "upper_boundary": upper_threshold,
    "total_records": total_records,
    "auto_match_count": len(auto_matches),
    "human_review_count": len(review_queue),
    "auto_non_match_count": len(auto_non_matches),
    "auto_match_percentage": (
        round(len(auto_matches) / total_records * 100, 2)
    ),
    "human_review_percentage": (
        round(len(review_queue) / total_records * 100, 2)
    ),
    "auto_non_match_percentage": (
        round(len(auto_non_matches) / total_records * 100, 2)
    )
}

with open(
    REPORT_DIR / "decision_engine_summary.json",
    "w"
) as file:
    json.dump(summary, file, indent=4)


# --------------------------------------------------
# 11. DISPLAY RESULTS
# --------------------------------------------------

print("\n========== DAY 14: DECISION ENGINE ==========")

print("Total records:", total_records)

print("\nDecision distribution:")
print("Auto-match:", len(auto_matches))
print("Human review:", len(review_queue))
print("Auto-non-match:", len(auto_non_matches))

print("\nDecision percentages:")
print("Auto-match:", summary["auto_match_percentage"], "%")
print("Human review:", summary["human_review_percentage"], "%")
print("Auto-non-match:", summary["auto_non_match_percentage"], "%")

print("\nFiles generated:")
print("1. data/processed/review/human_review_queue.csv")
print("2. data/processed/review/auto_match_candidates.csv")
print("3. data/processed/review/auto_non_match_candidates.csv")
print("4. data/reports/decision_engine_summary.json")

print("\nDay 14 completed successfully!")
