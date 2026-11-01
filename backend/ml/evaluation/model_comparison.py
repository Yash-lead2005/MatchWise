
import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]

REPORT_DIR = BASE_DIR / "data" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. REPORT FILES
# --------------------------------------------------

REPORT_FILES = [
    "baseline_evaluation.json",
    "logistic_regression_evaluation.json",
    "day9_model_evaluation.json",
    "svm_evaluation.json",
    "day11_model_evaluation.json",
    "xgboost_evaluation.json"
]


# --------------------------------------------------
# 3. METRIC ALIASES
# --------------------------------------------------

METRIC_ALIASES = {
    "accuracy": ["accuracy", "accuracy_score"],
    "precision": ["precision", "precision_score"],
    "recall": ["recall", "recall_score"],
    "f1_score": ["f1_score", "f1", "f1-score"],
    "average_precision": [
        "average_precision",
        "average_precision_score",
        "ap"
    ]
}


def extract_metrics(data):
    """
    Extract metrics from a dictionary, supporting
    different report structures.
    """

    result = {}

    if not isinstance(data, dict):
        return result

    for metric_name, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if alias in data:
                try:
                    result[metric_name] = float(data[alias])
                except (TypeError, ValueError):
                    pass
                break

    return result


# --------------------------------------------------
# 4. RECURSIVELY FIND METRIC SECTIONS
# --------------------------------------------------

def find_metric_sections(data, path="root"):

    sections = []

    if isinstance(data, dict):

        metrics = extract_metrics(data)

        if len(metrics) >= 2:
            sections.append({
                "path": path,
                "metrics": metrics
            })

        for key, value in data.items():
            sections.extend(
                find_metric_sections(value, f"{path}.{key}")
            )

    elif isinstance(data, list):

        for index, value in enumerate(data):
            sections.extend(
                find_metric_sections(value, f"{path}[{index}]")
            )

    return sections


# --------------------------------------------------
# 5. INFER MODEL NAME AND SPLIT
# --------------------------------------------------

def infer_model_name(filename, path):

    combined = (filename + " " + path).lower()

    model_names = [
        ("logistic regression", "Logistic Regression"),
        ("gaussian_nb", "Gaussian Naive Bayes"),
        ("naive bayes", "Gaussian Naive Bayes"),
        ("knn", "KNN"),
        ("linear_svm", "Linear SVM"),
        ("svm", "Linear SVM"),
        ("decision_tree", "Decision Tree"),
        ("random_forest", "Random Forest"),
        ("xgboost", "XGBoost"),
        ("baseline", "Rule-Based Baseline")
    ]

    for keyword, model_name in model_names:
        if keyword in combined:
            return model_name

    return Path(filename).stem.replace("_", " ").title()


def infer_split(path):

    path = path.lower()

    if "validation" in path or "val_metrics" in path:
        return "validation"

    if "training" in path or "train_metrics" in path:
        return "training"

    if "test" in path:
        return "test"

    return "unspecified"


# --------------------------------------------------
# 6. LOAD AND PROCESS REPORTS
# --------------------------------------------------

comparison_rows = []
missing_files = []

for filename in REPORT_FILES:

    file_path = REPORT_DIR / filename

    if not file_path.exists():
        print(f"WARNING: Report not found: {filename}")
        missing_files.append(filename)
        continue

    try:
        with open(file_path, "r") as file:
            report_data = json.load(file)

        sections = find_metric_sections(report_data)

        if not sections:
            print(f"WARNING: No metrics found in {filename}")
            continue

        for section in sections:

            model_name = infer_model_name(
                filename,
                section["path"]
            )

            split_name = infer_split(section["path"])

            row = {
                "model": model_name,
                "split": split_name,
                "source_report": filename,
                "report_section": section["path"]
            }

            row.update(section["metrics"])

            comparison_rows.append(row)

        print(f"Processed: {filename}")

    except (json.JSONDecodeError, OSError) as error:
        print(f"ERROR reading {filename}: {error}")


# --------------------------------------------------
# 7. CHECK RESULTS
# --------------------------------------------------

if not comparison_rows:
    raise ValueError(
        "No model metrics could be extracted. "
        "Check the evaluation JSON files."
    )

comparison_df = pd.DataFrame(comparison_rows)

metric_columns = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "average_precision"
]

for column in metric_columns:
    if column not in comparison_df.columns:
        comparison_df[column] = None


# --------------------------------------------------
# 8. REMOVE DUPLICATE ENTRIES
# --------------------------------------------------

comparison_df = comparison_df.drop_duplicates(
    subset=["model", "split", "source_report", "report_section"]
)


# --------------------------------------------------
# 9. SAVE COMPLETE COMPARISON
# --------------------------------------------------

comparison_df.to_csv(
    REPORT_DIR / "model_comparison_all.csv",
    index=False
)


# --------------------------------------------------
# 10. CREATE VALIDATION COMPARISON
# --------------------------------------------------

validation_df = comparison_df[
    comparison_df["split"] == "validation"
].copy()

if not validation_df.empty:

    validation_df = validation_df.sort_values(
        by="f1_score",
        ascending=False,
        na_position="last"
    )

    validation_df.to_csv(
        REPORT_DIR / "model_comparison_validation.csv",
        index=False
    )

    print("\n========== VALIDATION MODEL COMPARISON ==========")

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "average_precision"
    ]

    print(
        validation_df[display_columns].to_string(index=False)
    )

    best_model = validation_df.dropna(
        subset=["f1_score"]
    )

    if not best_model.empty:
        print(
            "\nBest validation F1 candidate:",
            best_model.iloc[0]["model"]
        )

else:
    print(
        "\nWARNING: No validation metrics were found "
        "in the existing reports."
    )

    print(
        "We will need to generate validation metrics "
        "before selecting the best model."
    )


# --------------------------------------------------
# 11. SAVE TEST COMPARISON SEPARATELY
# --------------------------------------------------

test_df = comparison_df[
    comparison_df["split"] == "test"
].copy()

if not test_df.empty:

    test_df.to_csv(
        REPORT_DIR / "model_comparison_test.csv",
        index=False
    )


# --------------------------------------------------
# 12. SAVE SUMMARY JSON
# --------------------------------------------------

summary = {
    "models_found": sorted(
        comparison_df["model"].dropna().unique().tolist()
    ),
    "validation_comparison_available": not validation_df.empty,
    "missing_report_files": missing_files,
    "total_metric_sections": int(len(comparison_df)),
    "selection_rule": (
        "Use validation F1 and Average Precision for "
        "model selection. Do not select models using "
        "test performance."
    )
}

with open(
    REPORT_DIR / "model_comparison_summary.json",
    "w"
) as file:
    json.dump(summary, file, indent=4)


# --------------------------------------------------
# 13. FINAL STATUS
# --------------------------------------------------

print("\n========== DAY 13 COMPLETED ==========")

print("\nFiles generated:")
print("1. model_comparison_all.csv")
print("2. model_comparison_validation.csv (if available)")
print("3. model_comparison_test.csv (if available)")
print("4. model_comparison_summary.json")

print("\nReports directory:")
print(REPORT_DIR)
