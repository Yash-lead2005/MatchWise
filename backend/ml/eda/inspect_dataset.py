
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Resolve paths relative to the MatchWise project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def find_csv(keyword):
    """Find the expected CSV file by its filename."""

    files = list(RAW_DIR.glob("*.csv"))

    if keyword.lower() == "abt":
        matches = [
            f for f in files
            if f.stem.lower() == "abt"
        ]

    elif keyword.lower() == "buy":
        matches = [
            f for f in files
            if f.stem.lower() == "buy"
        ]

    elif keyword.lower() == "mapping":
        matches = [
            f for f in files
            if "mapping" in f.stem.lower()
        ]

    else:
        matches = [
            f for f in files
            if keyword.lower() in f.stem.lower()
        ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file for {keyword}. "
            f"Found: {[f.name for f in matches]}"
        )

    return matches[0]



def load_csv(path):
    """Load CSV while handling common encoding variations."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def inspect_dataframe(name, df):
    print("\n" + "=" * 65)
    print(f"DATASET: {name}")
    print("=" * 65)

    print(f"\nRows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nFirst five records:")
    print(df.head().to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nDuplicate complete rows:")
    print(df.duplicated().sum())

    print("\nUnique values per column:")
    print(df.nunique(dropna=True).to_string())


def main():
    print("MATCHWISE - ABT-BUY DATASET INSPECTION")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data folder: {RAW_DIR}")

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            "data/raw folder not found. Create it in the project root."
        )

    csv_files = list(RAW_DIR.glob("*.csv"))
    print("\nCSV files found:", [f.name for f in csv_files])

    abt_file = find_csv("abt")
    buy_file = find_csv("buy")
    mapping_file = find_csv("mapping")

    abt = load_csv(abt_file)
    buy = load_csv(buy_file)
    mapping = load_csv(mapping_file)

    inspect_dataframe("ABT CATALOGUE", abt)
    inspect_dataframe("BUY CATALOGUE", buy)
    inspect_dataframe("GROUND-TRUTH MAPPING", mapping)

    # Missing-value report
    missing_report = pd.concat(
        {
            "Abt": abt.isnull().sum(),
            "Buy": buy.isnull().sum()
        },
        axis=1
    ).fillna(0)

    missing_report.to_csv(
        REPORT_DIR / "missing_values.csv"
    )

    # Dataset summary
    summary = {
        "abt": {
            "file": abt_file.name,
            "rows": int(abt.shape[0]),
            "columns": abt.columns.tolist(),
            "missing_values": {
                str(k): int(v)
                for k, v in abt.isnull().sum().items()
            },
            "duplicate_rows": int(abt.duplicated().sum())
        },
        "buy": {
            "file": buy_file.name,
            "rows": int(buy.shape[0]),
            "columns": buy.columns.tolist(),
            "missing_values": {
                str(k): int(v)
                for k, v in buy.isnull().sum().items()
            },
            "duplicate_rows": int(buy.duplicated().sum())
        },
        "mapping": {
            "file": mapping_file.name,
            "rows": int(mapping.shape[0]),
            "columns": mapping.columns.tolist()
        }
    }

    with open(
        REPORT_DIR / "dataset_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    # Record-count chart
    plt.figure(figsize=(8, 5))
    counts = {
        "Abt": len(abt),
        "Buy": len(buy),
        "Known Matches": len(mapping)
    }

    sns.barplot(
        x=list(counts.keys()),
        y=list(counts.values())
    )

    plt.title("Abt-Buy Dataset Record Counts")
    plt.ylabel("Number of records")
    plt.xlabel("Dataset")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "record_counts.png",
        dpi=150
    )
    plt.close()

    # Missing-value chart for product catalogues
    missing_plot = pd.DataFrame({
        "Abt": abt.isnull().sum(),
        "Buy": buy.isnull().sum()
    }).fillna(0)

    plt.figure(figsize=(9, 5))
    sns.heatmap(
        missing_plot,
        annot=True,
        fmt=".0f",
        cmap="Blues"
    )
    plt.title("Missing Values by Catalogue and Column")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "missing_values.png",
        dpi=150
    )
    plt.close()

    print("\n" + "=" * 65)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"Reports saved in: {REPORT_DIR}")
    print("- dataset_summary.json")
    print("- missing_values.csv")
    print("- record_counts.png")
    print("- missing_values.png")


if __name__ == "__main__":
    main()
