
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve
)


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_features(split_name):

    path = PROCESSED_DIR / f"{split_name}_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}. Complete Day 6 first."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{split_name} dataset is empty.")

    return df


# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

def prepare_features(df, feature_columns):

    X = df[feature_columns].copy()

    for column in feature_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    return X.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------
# BUILD LINEAR SVM PIPELINE
# --------------------------------------------------

def build_model():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=10000,
                random_state=RANDOM_STATE,
                dual="auto"
            )
        )
    ])


# --------------------------------------------------
# SELECT THRESHOLD USING VALIDATION DATA
# --------------------------------------------------

def find_best_threshold(y_true, decision_scores):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        decision_scores
    )

    precision = precision[:-1]
    recall = recall[:-1]

    if len(thresholds) == 0:
        raise ValueError(
            "Unable to calculate validation thresholds."
        )

    f1_scores = (
        2 * precision * recall /
        (precision + recall + 1e-10)
    )

    best_index = int(np.argmax(f1_scores))

    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1_score": float(f1_scores[best_index])
    }


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_model(y_true, decision_scores, threshold):

    predictions = (
        decision_scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "total_pairs": int(len(y_true)),
        "actual_duplicates": int(np.sum(y_true == 1)),
        "predicted_duplicates": int(np.sum(predictions == 1)),

        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "average_precision": float(
            average_precision_score(
                y_true,
                decision_scores
            )
        ),

        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("MATCHWISE - LINEAR SVM TRAINING")
    print("=" * 65)

    train_df = load_features("train")
    validation_df = load_features("validation")
    test_df = load_features("test")

    excluded = {"abt_id", "buy_id", "label"}

    feature_columns = [
        column
        for column in train_df.columns
        if column not in excluded
    ]

    # Check consistent feature schemas.
    for split_name, df in [
        ("validation", validation_df),
        ("test", test_df)
    ]:

        missing = [
            column for column in feature_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{split_name} missing features: {missing}"
            )

    X_train = prepare_features(train_df, feature_columns)
    X_validation = prepare_features(validation_df, feature_columns)
    X_test = prepare_features(test_df, feature_columns)

    y_train = train_df["label"].astype(int)
    y_validation = validation_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    print("\nDataset sizes:")
    print("Training:", len(X_train))
    print("Validation:", len(X_validation))
    print("Testing:", len(X_test))
    print("Features:", len(feature_columns))

    # Train SVM.
    model = build_model()

    print("\nTraining Linear SVM...")

    model.fit(X_train, y_train)

    print("Training completed successfully.")

    # Get decision scores.
    train_scores = model.decision_function(X_train)

    validation_scores = model.decision_function(
        X_validation
    )

    test_scores = model.decision_function(X_test)

    # Select threshold using validation only.
    best_threshold = find_best_threshold(
        y_validation,
        validation_scores
    )

    threshold = best_threshold["threshold"]

    print("\nBEST VALIDATION THRESHOLD")
    print("Threshold:", round(threshold, 4))
    print("Precision:", round(
        best_threshold["precision"], 4
    ))
    print("Recall:", round(
        best_threshold["recall"], 4
    ))
    print("F1:", round(
        best_threshold["f1_score"], 4
    ))

    # Evaluate all splits.
    train_metrics = evaluate_model(
        y_train,
        train_scores,
        threshold
    )

    validation_metrics = evaluate_model(
        y_validation,
        validation_scores,
        threshold
    )

    test_metrics = evaluate_model(
        y_test,
        test_scores,
        threshold
    )

    print("\nTRAIN RESULTS")
    print(json.dumps(train_metrics, indent=4))

    print("\nVALIDATION RESULTS")
    print(json.dumps(validation_metrics, indent=4))

    print("\nTEST RESULTS")
    print(json.dumps(test_metrics, indent=4))

    # Save model.
    model_path = MODEL_DIR / "linear_svm.joblib"

    joblib.dump(model, model_path)

    # Save model schema.
    schema = {
        "model_name": "Linear Support Vector Machine",
        "feature_columns": feature_columns,
        "threshold": threshold,
        "score_type": "decision_function score",
        "configuration": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 10000,
            "random_state": RANDOM_STATE
        }
    }

    with open(
        MODEL_DIR / "linear_svm_schema.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(schema, file, indent=4)

    # Save test predictions.
    predictions = test_df[
        ["abt_id", "buy_id", "label"]
    ].copy()

    predictions["decision_score"] = test_scores

    predictions["predicted_label"] = (
        test_scores >= threshold
    ).astype(int)

    predictions.to_csv(
        PROCESSED_DIR / "svm_test_predictions.csv",
        index=False
    )

    # Save evaluation report.
    report = {
        "model": "Linear Support Vector Machine",
        "algorithm": "LinearSVC",
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "threshold_selection": {
            "method": "Maximum validation F1",
            "result": best_threshold
        },
        "results": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics
        },
        "interpretation": (
            "Decision scores are not calibrated probabilities."
        )
    }

    with open(
        REPORT_DIR / "svm_evaluation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("LINEAR SVM TRAINING COMPLETED")
    print("Saved model:", model_path)
    print("Saved svm_evaluation.json")
    print("Saved svm_test_predictions.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
