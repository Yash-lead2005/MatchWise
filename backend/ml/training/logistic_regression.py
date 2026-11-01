
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

MODEL_NAME = "Logistic Regression"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_features(split_name):

    path = PROCESSED_DIR / f"{split_name}_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. Complete Day 6 first."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{split_name} dataset is empty.")

    required_columns = ["abt_id", "buy_id", "label"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    return df


# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

def prepare_features(df, feature_columns):

    X = df[feature_columns].copy()

    # Convert all feature values to numeric.
    for column in feature_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Replace infinite values with missing values.
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


# --------------------------------------------------
# BUILD MODEL PIPELINE
# --------------------------------------------------

def build_model():

    pipeline = Pipeline([
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
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                solver="lbfgs"
            )
        )
    ])

    return pipeline


# --------------------------------------------------
# FIND BEST THRESHOLD ON VALIDATION DATA
# --------------------------------------------------

def find_best_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    # precision and recall contain one extra final value.
    precision = precision[:-1]
    recall = recall[:-1]

    if len(thresholds) == 0:
        raise ValueError(
            "Unable to calculate classification thresholds."
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
# EVALUATION METRICS
# --------------------------------------------------

def evaluate_model(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
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
                probabilities
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
# MAIN TRAINING PIPELINE
# --------------------------------------------------

def main():

    print("=" * 65)
    print("MATCHWISE - LOGISTIC REGRESSION TRAINING")
    print("=" * 65)

    # Load all three splits.
    train_df = load_features("train")
    validation_df = load_features("validation")
    test_df = load_features("test")

    # Exclude IDs and labels from model input.
    excluded_columns = {
        "abt_id",
        "buy_id",
        "label"
    }

    feature_columns = [
        column
        for column in train_df.columns
        if column not in excluded_columns
    ]

    # Verify identical feature schemas.
    for split_name, df in [
        ("validation", validation_df),
        ("test", test_df)
    ]:

        missing = [
            col for col in feature_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{split_name} is missing features: {missing}"
            )

    X_train = prepare_features(
        train_df, feature_columns
    )

    X_validation = prepare_features(
        validation_df, feature_columns
    )

    X_test = prepare_features(
        test_df, feature_columns
    )

    y_train = train_df["label"].astype(int)
    y_validation = validation_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    print("\nDataset sizes:")
    print("Training:", len(X_train))
    print("Validation:", len(X_validation))
    print("Testing:", len(X_test))

    print("\nNumber of input features:", len(feature_columns))

    print("\nTraining Logistic Regression...")

    # Train only on training data.
    model = build_model()

    model.fit(X_train, y_train)

    print("Training completed successfully.")

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    best_threshold = find_best_threshold(
        y_validation,
        validation_probabilities
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

    # --------------------------------------------------
    # EVALUATE ALL SPLITS
    # --------------------------------------------------

    train_probabilities = model.predict_proba(
        X_train
    )[:, 1]

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    train_metrics = evaluate_model(
        y_train,
        train_probabilities,
        threshold
    )

    validation_metrics = evaluate_model(
        y_validation,
        validation_probabilities,
        threshold
    )

    test_metrics = evaluate_model(
        y_test,
        test_probabilities,
        threshold
    )

    print("\nTRAIN RESULTS")
    print(json.dumps(train_metrics, indent=4))

    print("\nVALIDATION RESULTS")
    print(json.dumps(validation_metrics, indent=4))

    print("\nTEST RESULTS")
    print(json.dumps(test_metrics, indent=4))

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    model_path = MODEL_DIR / "logistic_regression.joblib"

    joblib.dump(model, model_path)

    # Save feature schema separately.
    schema = {
        "model_name": MODEL_NAME,
        "feature_columns": feature_columns,
        "threshold": threshold
    }

    with open(
        MODEL_DIR / "logistic_regression_schema.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(schema, file, indent=4)

    # --------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------

    predictions = test_df[
        ["abt_id", "buy_id", "label"]
    ].copy()

    predictions["match_probability"] = test_probabilities

    predictions["predicted_label"] = (
        test_probabilities >= threshold
    ).astype(int)

    predictions.to_csv(
        PROCESSED_DIR / "logistic_regression_test_predictions.csv",
        index=False
    )

    # --------------------------------------------------
    # SAVE EVALUATION REPORT
    # --------------------------------------------------

    report = {
        "model": MODEL_NAME,
        "algorithm": "Supervised binary classification",
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "configuration": {
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "solver": "lbfgs"
        },
        "threshold_selection": {
            "method": "Maximum validation F1",
            "result": best_threshold
        },
        "results": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics
        }
    }

    with open(
        REPORT_DIR / "logistic_regression_evaluation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("LOGISTIC REGRESSION COMPLETED")
    print("Saved model:", model_path)
    print("Saved evaluation report.")
    print("Saved test predictions.")
    print("=" * 65)


if __name__ == "__main__":
    main()
