
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_file(filename):
    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run cleaning.py first."
        )

    return pd.read_csv(path, dtype={"id": "string"})


def main():
    abt = inspect_file("abt_clean.csv")
    buy = inspect_file("buy_clean.csv")
    mapping = inspect_file("ground_truth_clean.csv")

    report = {
        "abt": {
            "rows": int(len(abt)),
            "columns": abt.columns.tolist(),
            "missing_values": {
                str(k): int(v)
                for k, v in abt.isnull().sum().items()
            },
            "duplicate_ids": int(abt["id"].duplicated().sum())
        },
        "buy": {
            "rows": int(len(buy)),
            "columns": buy.columns.tolist(),
            "missing_values": {
                str(k): int(v)
                for k, v in buy.isnull().sum().items()
            },
            "duplicate_ids": int(buy["id"].duplicated().sum())
        },
        "ground_truth": {
            "rows": int(len(mapping)),
            "duplicate_pairs": int(
                mapping.duplicated(["abt_id", "buy_id"]).sum()
            )
        }
    }

    with open(
        REPORT_DIR / "preprocessing_validation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("=" * 60)
    print("MATCHWISE - PREPROCESSING VALIDATION")
    print("=" * 60)

    print("\nAbt records:", len(abt))
    print("Buy records:", len(buy))
    print("Known matching pairs:", len(mapping))

    print("\nAbt cleaned columns:")
    print(abt.columns.tolist())

    print("\nBuy cleaned columns:")
    print(buy.columns.tolist())

    print("\nMapping columns:")
    print(mapping.columns.tolist())

    print("\nValidation report:")
    print(json.dumps(report, indent=4))

    print("\nValidation completed successfully.")


if __name__ == "__main__":
    main()
