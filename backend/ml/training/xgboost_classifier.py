
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "data" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train_features.csv")
val_df = pd.read_csv(DATA_DIR / "validation_features.csv")
test_df = pd.read_csv(DATA_DIR / "test_features.csv")

print("\nDatasets loaded successfully!")
print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Testing:", test_df.shape)


# --------------------------------------------------
# 3. IDENTIFY LABEL AND ID COLUMNS
# --------------------------------------------------

possible_labels = ["label", "is_match", "target"]

LABEL_COL = next(
    (col for col in possible_labels if col in train_df.columns),
    None
)

if LABEL_COL is None:
    raise ValueError(
        "Could not find the label column. "
        "Check the column names in train_features.csv."
    )

ID_COLS = [
    col for col in ["abt_id", "buy_id"]
    if col in train_df.columns
]

FEATURE_COLS = [
    col for col in train_df.columns
    if col not in ID_COLS + [LABEL_COL]
]

if not FEATURE_COLS:
    raise ValueError("No feature columns found.")

# Ensure the same features are used in every split.
for split_name, df in [
    ("Validation", val_df),
    ("Testing", test_df)
]:
    missing = set(FEATURE_COLS) - set(df.columns)

    if missing:
        raise ValueError(
            f"{split_name} dataset is missing columns: {missing}"
        )

print("\nLabel column:", LABEL_COL)
print("ID columns:", ID_COLS)
print("Number of features:", len(FEATURE_COLS))


# --------------------------------------------------
# 4. PREPARE FEATURES AND LABELS
# --------------------------------------------------

X_train = train_df[FEATURE_COLS]
y_train = train_df[LABEL_COL].astype(int)

X_val = val_df[FEATURE_COLS]
y_val = val_df[LABEL_COL].astype(int)

X_test = test_df[FEATURE_COLS]
y_test = test_df[LABEL_COL].astype(int)

# Handle class imbalance.
positive_count = int((y_train == 1).sum())
negative_count = int((y_train == 0).sum())

if positive_count == 0 or negative_count == 0:
    raise ValueError("Training data must contain both classes.")

scale_pos_weight = negative_count / positive_count

print("\nClass distribution:")
print("Positive:", positive_count)
print("Negative:", negative_count)
print("Scale pos weight:", round(scale_pos_weight, 2))


# --------------------------------------------------
# 5. BUILD XGBOOST PIPELINE
# --------------------------------------------------

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "classifier",
        XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )
    )
])


# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

print("\nTraining XGBoost model...")

pipeline.fit(X_train, y_train)

print("Training completed successfully!")


# --------------------------------------------------
# 7. FIND BEST THRESHOLD USING VALIDATION DATA
# --------------------------------------------------

val_probabilities = pipeline.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_probabilities
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = int(np.argmax(f1_scores))
best_threshold = float(thresholds[best_index])

print("\nBest validation threshold:", round(best_threshold, 4))
print("Best validation F1:", round(float(f1_scores[best_index]), 4))


# --------------------------------------------------
# 8. FINAL TEST EVALUATION
# --------------------------------------------------

test_probabilities = pipeline.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(y_test, test_predictions)
precision_value = precision_score(
    y_test, test_predictions, zero_division=0
)
recall_value = recall_score(
    y_test, test_predictions, zero_division=0
)
f1_value = f1_score(
    y_test, test_predictions, zero_division=0
)
average_precision = average_precision_score(
    y_test, test_probabilities
)

conf_matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

print("\n========== XGBOOST TEST RESULTS ==========")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision_value, 4))
print("Recall:", round(recall_value, 4))
print("F1 Score:", round(f1_value, 4))
print("Average Precision:", round(average_precision, 4))
print("\nConfusion Matrix:")
print(conf_matrix)


# --------------------------------------------------
# 9. SAVE TEST PREDICTIONS
# --------------------------------------------------

prediction_df = test_df[ID_COLS].copy()

prediction_df["actual_label"] = y_test.values
prediction_df["match_probability"] = test_probabilities
prediction_df["predicted_label"] = test_predictions

prediction_df.to_csv(
    DATA_DIR / "xgboost_test_predictions.csv",
    index=False
)


# --------------------------------------------------
# 10. SAVE MODEL
# --------------------------------------------------

joblib.dump(
    pipeline,
    MODEL_DIR / "xgboost.joblib"
)

schema = {
    "model_name": "XGBoost Classifier",
    "label_column": LABEL_COL,
    "id_columns": ID_COLS,
    "feature_columns": FEATURE_COLS,
    "threshold": best_threshold,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42
}

with open(
    MODEL_DIR / "xgboost_schema.json",
    "w"
) as file:
    json.dump(schema, file, indent=4)


# --------------------------------------------------
# 11. SAVE EVALUATION REPORT
# --------------------------------------------------

report = {
    "model": "XGBoost Classifier",
    "threshold": best_threshold,
    "test_metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision_value),
        "recall": float(recall_value),
        "f1_score": float(f1_value),
        "average_precision": float(average_precision)
    },
    "confusion_matrix": conf_matrix.tolist(),
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "number_of_features": int(len(FEATURE_COLS)),
    "class_distribution": {
        "positive": positive_count,
        "negative": negative_count
    }
}

with open(
    REPORT_DIR / "xgboost_evaluation.json",
    "w"
) as file:
    json.dump(report, file, indent=4)


# --------------------------------------------------
# 12. SAVE FEATURE IMPORTANCE
# --------------------------------------------------

classifier = pipeline.named_steps["classifier"]

importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": classifier.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

importance_df.to_csv(
    REPORT_DIR / "xgboost_feature_importance.csv",
    index=False
)

print("\nAll files saved successfully!")
print("Model: models/xgboost.joblib")
print("Schema: models/xgboost_schema.json")
print("Predictions: data/processed/xgboost_test_predictions.csv")
print("Report: data/reports/xgboost_evaluation.json")
print("Feature importance: data/reports/xgboost_feature_importance.csv")
