
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"

SCHEMA_PATH = PROJECT_ROOT / "models" / "xgboost_schema.json"

TEST_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test_features.csv"
)

REPORT_DIR = PROJECT_ROOT / "data" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD MODEL, SCHEMA AND TEST DATA
# --------------------------------------------------

print("\nLoading model and test dataset...")

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r") as file:
    schema = json.load(file)

test_df = pd.read_csv(TEST_DATA_PATH)

FEATURE_COLUMNS = schema["feature_columns"]

THRESHOLD = float(schema["threshold"])

REVIEW_MARGIN = 0.15

LOWER_THRESHOLD = max(
    0.0,
    THRESHOLD - REVIEW_MARGIN
)

print("Model loaded successfully.")
print("Test records:", len(test_df))
print("Threshold:", THRESHOLD)
print("Lower threshold:", LOWER_THRESHOLD)


# --------------------------------------------------
# 3. PREPARE TEST FEATURES
# --------------------------------------------------

if "label" not in test_df.columns:
    raise ValueError("Label column missing from test dataset.")

missing_columns = [
    column
    for column in FEATURE_COLUMNS
    if column not in test_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing feature columns: {missing_columns}"
    )

X_test = test_df[FEATURE_COLUMNS]

y_test = test_df["label"].astype(int)


# --------------------------------------------------
# 4. GENERATE MODEL PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# --------------------------------------------------
# 5. STANDARD CLASSIFICATION METRICS
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()


# --------------------------------------------------
# 6. APPLY THREE-LEVEL DECISION POLICY
# --------------------------------------------------

decisions = []

for probability in probabilities:

    if probability >= THRESHOLD:

        decisions.append("AUTO_MATCH")

    elif probability <= LOWER_THRESHOLD:

        decisions.append("AUTO_NON_MATCH")

    else:

        decisions.append("HUMAN_REVIEW")


decision_series = pd.Series(decisions)

auto_match_mask = (
    decision_series == "AUTO_MATCH"
)

auto_non_match_mask = (
    decision_series == "AUTO_NON_MATCH"
)

review_mask = (
    decision_series == "HUMAN_REVIEW"
)

automatic_mask = (
    auto_match_mask | auto_non_match_mask
)


# --------------------------------------------------
# 7. AUTOMATIC DECISION PERFORMANCE
# --------------------------------------------------

auto_match_count = int(auto_match_mask.sum())

auto_non_match_count = int(auto_non_match_mask.sum())

review_count = int(review_mask.sum())

automatic_count = int(automatic_mask.sum())

total_count = len(y_test)


if auto_match_count > 0:

    auto_match_precision = float(
        (y_test[auto_match_mask] == 1).mean()
    )

else:

    auto_match_precision = None


if auto_non_match_count > 0:

    auto_non_match_accuracy = float(
        (y_test[auto_non_match_mask] == 0).mean()
    )

else:

    auto_non_match_accuracy = None


if automatic_count > 0:

    automatic_accuracy = float(
        (
            predictions[automatic_mask.to_numpy()]
            == y_test[automatic_mask]
        ).mean()
    )

else:

    automatic_accuracy = None


review_rate = review_count / total_count

automatic_coverage = automatic_count / total_count


# --------------------------------------------------
# 8. PRINT RESULTS
# --------------------------------------------------

print("\n" + "=" * 55)
print("MATCHWISE CLASSIFICATION RESULTS")
print("=" * 55)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\n" + "=" * 55)
print("HUMAN REVIEW DECISION POLICY RESULTS")
print("=" * 55)

print("Total test pairs:", total_count)

print(
    f"AUTO_MATCH: {auto_match_count} "
    f"({auto_match_count / total_count:.2%})"
)

print(
    f"AUTO_NON_MATCH: {auto_non_match_count} "
    f"({auto_non_match_count / total_count:.2%})"
)

print(
    f"HUMAN_REVIEW: {review_count} "
    f"({review_rate:.2%})"
)

print(
    f"Automatic coverage: {automatic_coverage:.2%}"
)

print("Automatic decision accuracy:", automatic_accuracy)

print("Auto-match precision:", auto_match_precision)

print("Auto-non-match accuracy:", auto_non_match_accuracy)


# --------------------------------------------------
# 9. SAVE EVALUATION REPORT
# --------------------------------------------------

report = {

    "model": "XGBoost",

    "test_records": int(total_count),

    "threshold": THRESHOLD,

    "lower_threshold": LOWER_THRESHOLD,

    "classification_metrics": {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1)

    },

    "confusion_matrix": {

        "true_negatives": int(tn),

        "false_positives": int(fp),

        "false_negatives": int(fn),

        "true_positives": int(tp)

    },

    "decision_policy": {

        "auto_match_count": auto_match_count,

        "auto_non_match_count": auto_non_match_count,

        "human_review_count": review_count,

        "automatic_coverage": float(automatic_coverage),

        "human_review_rate": float(review_rate),

        "automatic_accuracy": automatic_accuracy,

        "auto_match_precision": auto_match_precision,

        "auto_non_match_accuracy": auto_non_match_accuracy

    }

}


report_path = (
    REPORT_DIR / "decision_policy_evaluation.json"
)

with open(report_path, "w") as file:

    json.dump(
        report,
        file,
        indent=4
    )


print("\n" + "=" * 55)

print("DAY 18 EVALUATION COMPLETED!")

print("Report saved at:")

print(report_path)

print("=" * 55)
