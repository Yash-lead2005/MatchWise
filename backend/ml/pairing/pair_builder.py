
from pathlib import Path
from collections import defaultdict
import json

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# --------------------------------------------------
# UNION-FIND: KEEP MATCHED RECORDS TOGETHER
# --------------------------------------------------

class UnionFind:

    def __init__(self):
        self.parent = {}

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item

    def find(self, item):
        if item not in self.parent:
            self.add(item)

        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])

        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)

        if root1 != root2:
            self.parent[root2] = root1


# --------------------------------------------------
# LOAD PROCESSED DATA
# --------------------------------------------------

def load_data():

    abt_path = PROCESSED_DIR / "abt_clean.csv"
    buy_path = PROCESSED_DIR / "buy_clean.csv"
    mapping_path = PROCESSED_DIR / "ground_truth_clean.csv"

    for path in [abt_path, buy_path, mapping_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}. "
                "Complete Day 3 preprocessing first."
            )

    abt = pd.read_csv(abt_path, dtype={"id": "string"})
    buy = pd.read_csv(buy_path, dtype={"id": "string"})

    mapping = pd.read_csv(
        mapping_path,
        dtype={"abt_id": "string", "buy_id": "string"}
    )

    required = {"id"}
    if not required.issubset(abt.columns):
        raise ValueError("Abt catalogue must contain an id column.")

    if not required.issubset(buy.columns):
        raise ValueError("Buy catalogue must contain an id column.")

    if not {"abt_id", "buy_id"}.issubset(mapping.columns):
        raise ValueError(
            "Ground-truth file must contain abt_id and buy_id."
        )

    return abt, buy, mapping


# --------------------------------------------------
# BUILD CONNECTED COMPONENTS
# --------------------------------------------------

def build_components(abt, buy, mapping):

    uf = UnionFind()

    # Add every record, including unmatched records.
    for record_id in abt["id"].dropna():
        uf.add(f"A:{record_id}")

    for record_id in buy["id"].dropna():
        uf.add(f"B:{record_id}")

    # Connect known matching records.
    for row in mapping.itertuples(index=False):
        uf.union(
            f"A:{row.abt_id}",
            f"B:{row.buy_id}"
        )

    components = defaultdict(lambda: {
        "abt_ids": [],
        "buy_ids": [],
        "positive_count": 0
    })

    # Collect catalogue records into their components.
    for record_id in abt["id"].dropna():
        node = f"A:{record_id}"
        root = uf.find(node)
        components[root]["abt_ids"].append(str(record_id))

    for record_id in buy["id"].dropna():
        node = f"B:{record_id}"
        root = uf.find(node)
        components[root]["buy_ids"].append(str(record_id))

    # Count known matches in each component.
    for row in mapping.itertuples(index=False):
        root = uf.find(f"A:{row.abt_id}")
        components[root]["positive_count"] += 1

    component_rows = []

    for index, details in enumerate(components.values()):

        component_rows.append({
            "component_id": index,
            "abt_ids": details["abt_ids"],
            "buy_ids": details["buy_ids"],
            "positive_count": details["positive_count"],
            "has_match": int(details["positive_count"] > 0)
        })

    return pd.DataFrame(component_rows)


# --------------------------------------------------
# SPLIT COMPONENTS WITHOUT RECORD LEAKAGE
# --------------------------------------------------

def split_components(components):

    component_ids = components["component_id"]
    labels = components["has_match"]

    # First hold out 15% for testing.
    remaining_ids, test_ids = train_test_split(
        component_ids,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    remaining = components[
        components["component_id"].isin(remaining_ids)
    ]

    # Validation is 15% of the total dataset.
    validation_fraction = VALIDATION_RATIO / (
        TRAIN_RATIO + VALIDATION_RATIO
    )

    train_ids, validation_ids = train_test_split(
        remaining["component_id"],
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=remaining["has_match"]
    )

    return {
        "train": set(train_ids),
        "validation": set(validation_ids),
        "test": set(test_ids)
    }


# --------------------------------------------------
# GET RECORD IDS FOR EACH SPLIT
# --------------------------------------------------

def get_split_records(components, split_ids):

    selected = components[
        components["component_id"].isin(split_ids)
    ]

    abt_ids = set()
    buy_ids = set()

    for row in selected.itertuples(index=False):
        abt_ids.update(row.abt_ids)
        buy_ids.update(row.buy_ids)

    return abt_ids, buy_ids


# --------------------------------------------------
# CREATE POSITIVE AND NEGATIVE PAIRS
# --------------------------------------------------

def create_pairs(abt_ids, buy_ids, mapping, split_name):

    # Positive pairs are taken only from official ground truth.
    positive = mapping[
        mapping["abt_id"].isin(abt_ids)
        & mapping["buy_id"].isin(buy_ids)
    ][["abt_id", "buy_id"]].copy()

    positive["label"] = 1

    positive_keys = set(
        zip(positive["abt_id"], positive["buy_id"])
    )

    # Generate all cross-catalogue non-match pairs within this split.
    negative_records = []

    for a_id in sorted(abt_ids):
        for b_id in sorted(buy_ids):

            if (a_id, b_id) not in positive_keys:
                negative_records.append((a_id, b_id))

    # Keep the full negative set for test evaluation.
    # Sample train/validation negatives at 3:1.
    if split_name in {"train", "validation"}:

        desired_count = min(
            len(negative_records),
            len(positive) * 3
        )

        negative_df = pd.DataFrame(
            negative_records,
            columns=["abt_id", "buy_id"]
        )

        negative_df = negative_df.sample(
            n=desired_count,
            random_state=RANDOM_STATE
        )

    else:
        negative_df = pd.DataFrame(
            negative_records,
            columns=["abt_id", "buy_id"]
        )

    negative_df["label"] = 0

    pairs = pd.concat(
        [positive, negative_df],
        ignore_index=True
    )

    pairs["split"] = split_name

    return pairs


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    print("=" * 65)
    print("MATCHWISE - PAIR CONSTRUCTION AND DATA SPLITTING")
    print("=" * 65)

    abt, buy, mapping = load_data()

    print("\nLoaded records:")
    print("Abt:", len(abt))
    print("Buy:", len(buy))
    print("Known matches:", len(mapping))

    components = build_components(abt, buy, mapping)

    print("\nConnected components:", len(components))
    print(
        "Components containing known matches:",
        int(components["has_match"].sum())
    )

    split_ids = split_components(components)

    all_pairs = []
    split_summary = {}

    for split_name in ["train", "validation", "test"]:

        abt_ids, buy_ids = get_split_records(
            components,
            split_ids[split_name]
        )

        pairs = create_pairs(
            abt_ids,
            buy_ids,
            mapping,
            split_name
        )

        all_pairs.append(pairs)

        positive_count = int((pairs["label"] == 1).sum())
        negative_count = int((pairs["label"] == 0).sum())

        split_summary[split_name] = {
            "components": len(split_ids[split_name]),
            "abt_records": len(abt_ids),
            "buy_records": len(buy_ids),
            "positive_pairs": positive_count,
            "negative_pairs": negative_count,
            "total_pairs": len(pairs)
        }

        print(f"\n{split_name.upper()} SPLIT")
        print("Abt records:", len(abt_ids))
        print("Buy records:", len(buy_ids))
        print("Positive pairs:", positive_count)
        print("Negative pairs:", negative_count)
        print("Total pairs:", len(pairs))

    # Verify record-level separation.
    record_sets = {}

    for split_name in ["train", "validation", "test"]:
        a_ids, b_ids = get_split_records(
            components,
            split_ids[split_name]
        )

        record_sets[split_name] = {
            "A:" + x for x in a_ids
        } | {
            "B:" + x for x in b_ids
        }

    split_names = ["train", "validation", "test"]

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):

            first = split_names[i]
            second = split_names[j]

            overlap = (
                record_sets[first] & record_sets[second]
            )

            if overlap:
                raise RuntimeError(
                    f"Data leakage detected between {first} "
                    f"and {second}: {len(overlap)} records."
                )

    # Save each split separately.
    for split_name, pairs in zip(
        ["train", "validation", "test"],
        all_pairs
    ):
        pairs.to_csv(
            PROCESSED_DIR / f"{split_name}_pairs.csv",
            index=False
        )

    # Save split metadata.
    metadata = {
        "random_state": RANDOM_STATE,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO
        },
        "split_summary": split_summary,
        "record_level_leakage_check": "PASSED",
        "negative_sampling": {
            "train": "Up to 3 negatives per positive",
            "validation": "Up to 3 negatives per positive",
            "test": "All within-split non-match pairs"
        }
    }

    with open(
        REPORT_DIR / "pair_split_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 65)
    print("PAIR CONSTRUCTION COMPLETED")
    print("Record-level leakage check: PASSED")
    print("Saved train_pairs.csv")
    print("Saved validation_pairs.csv")
    print("Saved test_pairs.csv")
    print("Saved pair_split_summary.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
