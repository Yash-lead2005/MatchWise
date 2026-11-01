
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

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

MODEL_CONFIGS = {
    "Gaussian Naive Bayes": {
        "filename": "gaussian_nb.joblib"
    },
    "K-Nearest Neighbours": {
        "filename": "knn.joblib"
    }
}


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

    X = X.replace([np.inf, -np.inf], np.nan)

    return X


# --------------------------------------------------
# BUILD MODELS
# --------------------------------------------------

def build_models():

    # Balanced class priors prevent the majority class
    # from dominating the Naive Bayes prior probabilities.
    naive_bayes = Pipeline([
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
            GaussianNB(priors=[0.5, 0.5])
        )
    ])

    knn = Pipeline([
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
            KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                metric="minkowski",
                p=2,
                n_jobs=-1
            )
        )
    ])

    return {
        "Gaussian Naive Bayes": naive_bayes,
        "K-Nearest Neighbours": knn
    }


# --------------------------------------------------
# FIND BEST THRESHOLD USING VALIDATION
# --------------------------------------------------

def find_best_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
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
# TRAIN AND EVALUATE ONE MODEL
# --------------------------------------------------

def train_and_evaluate(
    model_name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    feature_columns
):

    print("\n" + "-" * 60)
    print(f"TRAINING: {model_name}")
    print("-" * 60)

    model.fit(X_train, y_train)

    print("Training completed.")

    # Validation probabilities.
    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    best_threshold = find_best_threshold(
        y_validation,
        validation_probabilities
    )

    threshold = best_threshold["threshold"]

    print("\nSelected validation threshold:", round(threshold, 4))

    # Predict all splits.
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

    print("\nTEST RESULTS")
    print(json.dumps(test_metrics, indent=4))

    # Save model.
    filename = MODEL_CONFIGS[model_name]["filename"]
    model_path = MODEL_DIR / filename

    joblib.dump(model, model_path)

    # Save model schema.
    schema = {
        "model_name": model_name,
        "feature_columns": feature_columns,
        "threshold": threshold
    }

    schema_path = MODEL_DIR / f"{filename.replace('.joblib', '_schema.json')}"

    with open(schema_path, "w", encoding="utf-8") as file:
        json.dump(schema, file, indent=4)

    # Save test predictions.
    predictions = pd.DataFrame({
        "abt_id": test_ids_abt,
        "buy_id": test_ids_buy,
        "actual_label": y_test.to_numpy(),
        "match_probability": test_probabilities,
        "predicted_label": (
            test_probabilities >= threshold
        ).astype(int)
    })

    prediction_filename = (
        f"{filename.replace('.joblib', '')}_test_predictions.csv"
    )

    predictions.to_csv(
        PROCESSED_DIR / prediction_filename,
        index=False
    )

    return {
        "model": model_name,
        "threshold_selection": best_threshold,
        "results": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics
        },
        "model_file": filename,
        "prediction_file": prediction_filename
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    global test_ids_abt, test_ids_buy

    print("=" * 65)
    print("MATCHWISE - NAIVE BAYES AND KNN TRAINING")
    print("=" * 65)

    train_df = load_features("train")
    validation_df = load_features("validation")
    test_df = load_features("test")

    excluded = {"abt_id", "buy_id", "label"}

    feature_columns = [
        column for column in train_df.columns
        if column not in excluded
    ]

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

    test_ids_abt = test_df["abt_id"].astype(str)
    test_ids_buy = test_df["buy_id"].astype(str)

    print("\nDataset sizes:")
    print("Training:", len(X_train))
    print("Validation:", len(X_validation))
    print("Testing:", len(X_test))
    print("Features:", len(feature_columns))

    models = build_models()

    results = {}

    for model_name, model in models.items():

        result = train_and_evaluate(
            model_name,
            model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
            feature_columns
        )

        results[model_name] = result

    # Create a compact model comparison.
    comparison = []

    for model_name, result in results.items():

        test_metrics = result["results"]["test"]

        comparison.append({
            "model": model_name,
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "average_precision": test_metrics["average_precision"]
        })

    comparison_df = pd.DataFrame(comparison)

    comparison_df.to_csv(
        REPORT_DIR / "day9_model_comparison.csv",
        index=False
    )

    report = {
        "day": 9,
        "models": results,
        "test_model_comparison": comparison
    }

    with open(
        REPORT_DIR / "day9_model_evaluation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("DAY 9 MODEL TRAINING COMPLETED")
    print("=" * 65)

    print("\nTEST MODEL COMPARISON")
    print(comparison_df.to_string(index=False))

    print("\nSaved:")
    print("models/gaussian_nb.joblib")
    print("models/knn.joblib")
    print("data/reports/day9_model_evaluation.json")
    print("data/reports/day9_model_comparison.csv")


if __name__ == "__main__":
    main()
