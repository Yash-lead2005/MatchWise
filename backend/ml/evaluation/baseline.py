
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD FEATURES
# --------------------------------------------------

def load_features(split_name):

    path = PROCESSED_DIR / f"{split_name}_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Complete Day 6 first."
        )

    df = pd.read_csv(path)

    required_columns = [
        "name_char_similarity",
        "name_token_similarity",
        "name_jaccard_similarity",
        "name_tfidf_similarity",
        "description_tfidf_similarity",
        "manufacturer_exact_match",
        "manufacturer_conflict",
        "blocking_similarity",
        "label"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}"
        )

    if df.empty:
        raise ValueError(f"{split_name} feature file is empty.")

    return df


# --------------------------------------------------
# RULE-BASED MATCHING SCORE
# --------------------------------------------------

def calculate_score(df):

    score = (
        0.25 * df["name_char_similarity"].fillna(0)
        + 0.15 * df["name_tfidf_similarity"].fillna(0)
        + 0.20 * df["description_tfidf_similarity"].fillna(0)
        + 0.15 * df["manufacturer_exact_match"].fillna(0)
        + 0.10 * df["blocking_similarity"].fillna(0)
        + 0.15 * df["name_jaccard_similarity"].fillna(0)
    )

    # Penalise a known manufacturer disagreement.
    score = score - (
        0.10 * df["manufacturer_conflict"].fillna(0)
    )

    return score.clip(0, 1)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def evaluate_predictions(y_true, y_pred, scores):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "total_pairs": int(len(y_true)),
        "actual_duplicates": int(np.sum(y_true == 1)),
        "predicted_duplicates": int(np.sum(y_pred == 1)),

        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),

        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
    }


# --------------------------------------------------
# FIND BEST THRESHOLD USING VALIDATION ONLY
# --------------------------------------------------

def find_best_threshold(validation_df):

    y_true = validation_df["label"].astype(int).to_numpy()
    scores = calculate_score(validation_df).to_numpy()

    threshold_results = []

    # Test a range of possible decision thresholds.
    thresholds = np.arange(0.20, 0.901, 0.01)

    for threshold in thresholds:

        predictions = (scores >= threshold).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        threshold_results.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        })

    # Select the highest validation F1.
    # In a tie, prefer the higher threshold.
    best = max(
        threshold_results,
        key=lambda item: (
            item["f1_score"],
            item["threshold"]
        )
    )

    return best, threshold_results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("MATCHWISE - RULE-BASED BASELINE EVALUATION")
    print("=" * 65)

    train_df = load_features("train")
    validation_df = load_features("validation")
    test_df = load_features("test")

    print("\nDataset sizes:")
    print("Training:", len(train_df))
    print("Validation:", len(validation_df))
    print("Testing:", len(test_df))

    # Step 1: Find threshold using validation only.
    best_threshold, threshold_results = find_best_threshold(
        validation_df
    )

    threshold = best_threshold["threshold"]

    print("\nBEST VALIDATION THRESHOLD")
    print("Threshold:", round(threshold, 3))
    print("Validation precision:", round(
        best_threshold["precision"], 4
    ))
    print("Validation recall:", round(
        best_threshold["recall"], 4
    ))
    print("Validation F1:", round(
        best_threshold["f1_score"], 4
    ))

    # Step 2: Evaluate training data for reference.
    train_scores = calculate_score(train_df).to_numpy()
    train_labels = train_df["label"].astype(int).to_numpy()

    train_predictions = (
        train_scores >= threshold
    ).astype(int)

    train_metrics = evaluate_predictions(
        train_labels,
        train_predictions,
        train_scores
    )

    # Step 3: Evaluate validation.
    validation_scores = calculate_score(
        validation_df
    ).to_numpy()

    validation_labels = (
        validation_df["label"].astype(int).to_numpy()
    )

    validation_predictions = (
        validation_scores >= threshold
    ).astype(int)

    validation_metrics = evaluate_predictions(
        validation_labels,
        validation_predictions,
        validation_scores
    )

    # Step 4: Evaluate test ONCE using selected threshold.
    test_scores = calculate_score(test_df).to_numpy()
    test_labels = test_df["label"].astype(int).to_numpy()

    test_predictions = (
        test_scores >= threshold
    ).astype(int)

    test_metrics = evaluate_predictions(
        test_labels,
        test_predictions,
        test_scores
    )

    print("\nTRAIN METRICS")
    print(json.dumps(train_metrics, indent=4))

    print("\nVALIDATION METRICS")
    print(json.dumps(validation_metrics, indent=4))

    print("\nTEST METRICS")
    print(json.dumps(test_metrics, indent=4))

    # Save predictions for manual inspection.
    test_output = test_df[
        ["abt_id", "buy_id", "label"]
    ].copy()

    test_output["matching_score"] = test_scores
    test_output["predicted_label"] = test_predictions

    test_output.to_csv(
        PROCESSED_DIR / "baseline_test_predictions.csv",
        index=False
    )

    # Save all results.
    report = {
        "model": "Rule-Based Matching Baseline",
        "score_formula": {
            "name_char_similarity": 0.25,
            "name_tfidf_similarity": 0.15,
            "description_tfidf_similarity": 0.20,
            "manufacturer_exact_match": 0.15,
            "blocking_similarity": 0.10,
            "name_jaccard_similarity": 0.15,
            "manufacturer_conflict_penalty": -0.10
        },
        "threshold_selection": {
            "method": "Highest validation F1",
            "best_threshold": threshold,
            "validation_result": best_threshold
        },
        "results": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics
        },
        "threshold_search": threshold_results
    }

    with open(
        REPORT_DIR / "baseline_evaluation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("BASELINE EVALUATION COMPLETED")
    print("Selected threshold:", threshold)
    print("Saved baseline_evaluation.json")
    print("Saved baseline_test_predictions.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
