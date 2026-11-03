
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"

SCHEMA_PATH = PROJECT_ROOT / "models" / "xgboost_schema.json"

VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "validation_features.csv"
)

REPORT_DIR = PROJECT_ROOT / "data" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD MODEL AND VALIDATION DATA
# --------------------------------------------------

print("\nLoading model and validation dataset...")

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r") as file:
    schema = json.load(file)

df = pd.read_csv(VALIDATION_PATH)

FEATURE_COLUMNS = schema["feature_columns"]

X_val = df[FEATURE_COLUMNS]
y_val = df["label"].astype(int).to_numpy()

probabilities = model.predict_proba(X_val)[:, 1]

print("Validation records:", len(df))
print("Validation matches:", int(y_val.sum()))
print("Validation non-matches:", int((y_val == 0).sum()))


# --------------------------------------------------
# 3. FIND AUTO-MATCH THRESHOLD
# Target: at least 95% precision
# --------------------------------------------------

TARGET_MATCH_PRECISION = 0.95

candidate_thresholds = np.unique(
    np.concatenate(([0.0], probabilities, [1.0]))
)

match_results = []

for threshold in candidate_thresholds:

    predicted_matches = probabilities >= threshold

    count = int(predicted_matches.sum())

    if count == 0:
        continue

    precision = float(
        (y_val[predicted_matches] == 1).mean()
    )

    recall = float(
        (predicted_matches & (y_val == 1)).sum()
        / max((y_val == 1).sum(), 1)
    )

    match_results.append({
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "predicted_matches": count
    })


eligible_matches = [
    result for result in match_results
    if result["precision"] >= TARGET_MATCH_PRECISION
]

if eligible_matches:

    # Lowest qualifying threshold gives the highest recall.
    best_match = min(
        eligible_matches,
        key=lambda result: result["threshold"]
    )

else:

    best_match = None


# --------------------------------------------------
# 4. FIND AUTO-NON-MATCH THRESHOLD
# Target: at least 99% non-match accuracy
# --------------------------------------------------

TARGET_NON_MATCH_ACCURACY = 0.99

non_match_results = []

for threshold in candidate_thresholds:

    predicted_non_matches = probabilities <= threshold

    count = int(predicted_non_matches.sum())

    if count == 0:
        continue

    accuracy = float(
        (y_val[predicted_non_matches] == 0).mean()
    )

    non_match_results.append({
        "threshold": float(threshold),
        "accuracy": accuracy,
        "predicted_non_matches": count
    })


eligible_non_matches = [
    result for result in non_match_results
    if result["accuracy"] >= TARGET_NON_MATCH_ACCURACY
]

if eligible_non_matches:

    # Highest qualifying threshold gives the greatest coverage.
    best_non_match = max(
        eligible_non_matches,
        key=lambda result: result["threshold"]
    )

else:

    best_non_match = None


# --------------------------------------------------
# 5. DISPLAY CALIBRATION RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("MATCHWISE THRESHOLD CALIBRATION")
print("=" * 60)

if best_match:

    print("\nRecommended AUTO_MATCH threshold:")
    print("Threshold:", round(best_match["threshold"], 4))
    print("Precision:", round(best_match["precision"], 4))
    print("Recall:", round(best_match["recall"], 4))
    print("Predicted matches:", best_match["predicted_matches"])

else:

    print("\nNo threshold achieved the target match precision.")
    print("Target:", TARGET_MATCH_PRECISION)


if best_non_match:

    print("\nRecommended AUTO_NON_MATCH threshold:")
    print("Threshold:", round(best_non_match["threshold"], 4))
    print("Accuracy:", round(best_non_match["accuracy"], 4))
    print(
        "Predicted non-matches:",
        best_non_match["predicted_non_matches"]
    )

else:

    print("\nNo threshold achieved the target non-match accuracy.")
    print("Target:", TARGET_NON_MATCH_ACCURACY)


# --------------------------------------------------
# 6. EVALUATE THE PROPOSED THREE-LEVEL POLICY
# --------------------------------------------------

policy_result = None

if best_match and best_non_match:

    upper = best_match["threshold"]
    lower = best_non_match["threshold"]

    if lower < upper:

        decisions = np.full(
            len(probabilities),
            "HUMAN_REVIEW",
            dtype=object
        )

        decisions[probabilities >= upper] = "AUTO_MATCH"
        decisions[probabilities <= lower] = "AUTO_NON_MATCH"

        auto_match_count = int(
            (decisions == "AUTO_MATCH").sum()
        )

        auto_non_match_count = int(
            (decisions == "AUTO_NON_MATCH").sum()
        )

        review_count = int(
            (decisions == "HUMAN_REVIEW").sum()
        )

        policy_result = {
            "auto_match_threshold": float(upper),
            "auto_non_match_threshold": float(lower),
            "auto_match_count": auto_match_count,
            "auto_non_match_count": auto_non_match_count,
            "human_review_count": review_count,
            "human_review_rate": float(
                review_count / len(y_val)
            ),
            "automatic_coverage": float(
                (auto_match_count + auto_non_match_count)
                / len(y_val)
            )
        }

        print("\n" + "=" * 60)
        print("PROPOSED HUMAN REVIEW POLICY")
        print("=" * 60)

        print("AUTO_MATCH:", auto_match_count)
        print("AUTO_NON_MATCH:", auto_non_match_count)
        print("HUMAN_REVIEW:", review_count)

        print(
            "Human review rate:",
            f"{policy_result['human_review_rate']:.2%}"
        )

        print(
            "Automatic coverage:",
            f"{policy_result['automatic_coverage']:.2%}"
        )

    else:

        print("\nWARNING: The two recommended thresholds overlap.")
        print("A valid three-level policy cannot be formed.")

# --------------------------------------------------
# 7. SAVE REPORT
# --------------------------------------------------

report = {
    "dataset": "validation",
    "validation_records": int(len(y_val)),
    "target_auto_match_precision": TARGET_MATCH_PRECISION,
    "target_auto_non_match_accuracy": TARGET_NON_MATCH_ACCURACY,
    "recommended_auto_match": best_match,
    "recommended_auto_non_match": best_non_match,
    "proposed_policy": policy_result
}

report_path = REPORT_DIR / "threshold_calibration.json"

with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

print("\nCalibration report saved:")
print(report_path)

print("\nTHRESHOLD CALIBRATION COMPLETED!")
