
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "models" / "xgboost.joblib"

SCHEMA_PATH = ROOT / "models" / "xgboost_schema.json"

TEST_PATH = ROOT / "data" / "processed" / "test_features.csv"

REPORT_DIR = ROOT / "data" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CALIBRATED THRESHOLDS
# --------------------------------------------------

AUTO_MATCH_THRESHOLD = 0.9975168704986572

AUTO_NON_MATCH_THRESHOLD = 0.91334404493331909


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r") as file:
    schema = json.load(file)

FEATURE_COLUMNS = schema["feature_columns"]

df = pd.read_csv(TEST_PATH)

X_test = df[FEATURE_COLUMNS]

y_test = df["label"].astype(int)


# --------------------------------------------------
# 4. MODEL PREDICTIONS
# --------------------------------------------------

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= float(schema["threshold"])
).astype(int)


# --------------------------------------------------
# 5. CLASSIFICATION METRICS
# --------------------------------------------------

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
# 6. THREE-LEVEL DECISION POLICY
# --------------------------------------------------

decisions = []

for probability in probabilities:

    if probability >= AUTO_MATCH_THRESHOLD:

        decisions.append("AUTO_MATCH")

    elif probability <= AUTO_NON_MATCH_THRESHOLD:

        decisions.append("AUTO_NON_MATCH")

    else:

        decisions.append("HUMAN_REVIEW")


decisions = pd.Series(decisions)

match_mask = decisions == "AUTO_MATCH"

non_match_mask = decisions == "AUTO_NON_MATCH"

review_mask = decisions == "HUMAN_REVIEW"

automatic_mask = match_mask | non_match_mask


# --------------------------------------------------
# 7. DECISION PERFORMANCE
# --------------------------------------------------

total = len(df)

match_count = int(match_mask.sum())

non_match_count = int(non_match_mask.sum())

review_count = int(review_mask.sum())

automatic_count = match_count + non_match_count


match_precision = (
    float((y_test[match_mask] == 1).mean())
    if match_count else None
)

non_match_accuracy = (
    float((y_test[non_match_mask] == 0).mean())
    if non_match_count else None
)

automatic_accuracy = (
    float(
        (
            decisions[automatic_mask].to_numpy()
            == y_test[automatic_mask].map(
                {1: "AUTO_MATCH", 0: "AUTO_NON_MATCH"}
            ).to_numpy()
        ).mean()
    )
    if automatic_count else None
)

# The previous calculation compares decision labels
# against mapped true labels for automatic decisions.


# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 55)
print("MATCHWISE FINAL TEST EVALUATION")
print("=" * 55)

print("Total test pairs:", total)

print("\nStandard classification metrics:")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 score:", round(f1, 4))

print("\nConfusion matrix:")
print("True negatives:", int(tn))
print("False positives:", int(fp))
print("False negatives:", int(fn))
print("True positives:", int(tp))

print("\nCalibrated decision policy:")

print(
    f"AUTO_MATCH: {match_count} "
    f"({match_count / total:.2%})"
)

print(
    f"AUTO_NON_MATCH: {non_match_count} "
    f"({non_match_count / total:.2%})"
)

print(
    f"HUMAN_REVIEW: {review_count} "
    f"({review_count / total:.2%})"
)

print(
    "Automatic coverage:",
    f"{automatic_count / total:.2%}"
)

print("Auto-match precision:", match_precision)

print("Auto-non-match accuracy:", non_match_accuracy)

print("Automatic decision accuracy:", automatic_accuracy)


# --------------------------------------------------
# 9. SAVE REPORT
# --------------------------------------------------

report = {

    "dataset": "test",

    "total_pairs": int(total),

    "thresholds": {

        "auto_match": AUTO_MATCH_THRESHOLD,

        "auto_non_match": AUTO_NON_MATCH_THRESHOLD

    },

    "classification_metrics": {

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

        "auto_match_count": match_count,

        "auto_non_match_count": non_match_count,

        "human_review_count": review_count,

        "automatic_coverage": float(
            automatic_count / total
        ),

        "auto_match_precision": match_precision,

        "auto_non_match_accuracy": non_match_accuracy,

        "automatic_accuracy": automatic_accuracy

    }

}

report_path = REPORT_DIR / "final_policy_evaluation.json"

with open(report_path, "w") as file:

    json.dump(report, file, indent=4)

print("\nReport saved:", report_path)

print("\nFINAL TEST EVALUATION COMPLETED!")
