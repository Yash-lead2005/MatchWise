
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_csv(filename, id_columns=None):

    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Complete previous days first."
        )

    dtype = {
        column: "string"
        for column in (id_columns or [])
    }

    return pd.read_csv(path, dtype=dtype)


def load_catalogues():

    abt = load_csv("abt_clean.csv", ["id"])
    buy = load_csv("buy_clean.csv", ["id"])

    mapping = load_csv(
        "ground_truth_clean.csv",
        ["abt_id", "buy_id"]
    )

    return abt, buy, mapping


# --------------------------------------------------
# SAFE TEXT UTILITIES
# --------------------------------------------------

def get_text(row, column):

    value = row.get(column, "")

    if pd.isna(value):
        return ""

    return str(value).strip()


def token_set(text):

    return set(re.findall(r"\b\w+\b", text.lower()))


def jaccard_similarity(text1, text2):

    tokens1 = token_set(text1)
    tokens2 = token_set(text2)

    if not tokens1 and not tokens2:
        return 0.0

    union = tokens1 | tokens2

    if not union:
        return 0.0

    return len(tokens1 & tokens2) / len(union)


def get_price(row):

    value = row.get("price_numeric", np.nan)

    if pd.isna(value):
        return np.nan

    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


# --------------------------------------------------
# TF-IDF VECTORIZERS
# --------------------------------------------------

def fit_text_vectorizer(train_abt, train_buy, column):

    texts = pd.concat(
        [
            train_abt[column].fillna("").astype(str),
            train_buy[column].fillna("").astype(str)
        ],
        ignore_index=True
    )

    # Avoid an empty vocabulary if a field has no text.
    if not texts.str.strip().any():
        vectorizer = None
        return vectorizer

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
        max_features=100000
    )

    vectorizer.fit(texts)

    return vectorizer


def transform_catalogue(df, column, vectorizer):

    if vectorizer is None:
        return None

    texts = df[column].fillna("").astype(str)

    return vectorizer.transform(texts)


# --------------------------------------------------
# BUILD RECORD LOOKUPS
# --------------------------------------------------

def build_lookup(df):

    return {
        str(row["id"]): row
        for _, row in df.iterrows()
    }


def build_position_lookup(df):

    return {
        str(record_id): position
        for position, record_id in enumerate(df["id"])
    }


# --------------------------------------------------
# PAIRWISE FEATURE CALCULATION
# --------------------------------------------------

def calculate_pair_features(
    pair,
    abt_lookup,
    buy_lookup,
    abt_positions,
    buy_positions,
    abt_name_vectors,
    buy_name_vectors,
    abt_desc_vectors,
    buy_desc_vectors
):

    abt_id = str(pair["abt_id"])
    buy_id = str(pair["buy_id"])

    if abt_id not in abt_lookup or buy_id not in buy_lookup:
        raise ValueError(
            f"Candidate references unknown records: {abt_id}, {buy_id}"
        )

    a = abt_lookup[abt_id]
    b = buy_lookup[buy_id]

    # Extract cleaned text.
    name_a = get_text(a, "name_clean")
    name_b = get_text(b, "name_clean")

    desc_a = get_text(a, "description_clean")
    desc_b = get_text(b, "description_clean")

    manufacturer_a = get_text(a, "manufacturer_clean")
    manufacturer_b = get_text(b, "manufacturer_clean")

    # Name similarity.
    name_char_similarity = fuzz.ratio(
        name_a, name_b
    ) / 100.0

    name_token_similarity = fuzz.token_sort_ratio(
        name_a, name_b
    ) / 100.0

    name_jaccard = jaccard_similarity(
        name_a, name_b
    )

    # Description similarity.
    description_similarity = fuzz.ratio(
        desc_a, desc_b
    ) / 100.0

    description_jaccard = jaccard_similarity(
        desc_a, desc_b
    )

    # Manufacturer comparison.
    manufacturer_a_present = bool(manufacturer_a)
    manufacturer_b_present = bool(manufacturer_b)

    manufacturer_both_present = (
        manufacturer_a_present and manufacturer_b_present
    )

    manufacturer_exact_match = int(
        manufacturer_both_present
        and manufacturer_a == manufacturer_b
    )

    manufacturer_conflict = int(
        manufacturer_both_present
        and manufacturer_a != manufacturer_b
    )

    # Price comparison.
    price_a = get_price(a)
    price_b = get_price(b)

    price_both_present = (
        not pd.isna(price_a) and not pd.isna(price_b)
    )

    if price_both_present:

        price_difference = abs(price_a - price_b)

        price_relative_difference = (
            price_difference /
            max(abs(price_a), abs(price_b), 1e-8)
        )

    else:
        price_difference = np.nan
        price_relative_difference = np.nan

    # TF-IDF cosine similarity using already-fitted vectorizers.
    a_position = abt_positions[abt_id]
    b_position = buy_positions[buy_id]

    if abt_name_vectors is not None:
        name_tfidf_similarity = float(
            abt_name_vectors[a_position]
            .multiply(buy_name_vectors[b_position])
            .sum()
        )
    else:
        name_tfidf_similarity = 0.0

    if abt_desc_vectors is not None:
        description_tfidf_similarity = float(
            abt_desc_vectors[a_position]
            .multiply(buy_desc_vectors[b_position])
            .sum()
        )
    else:
        description_tfidf_similarity = 0.0

    return {
        "name_char_similarity": name_char_similarity,
        "name_token_similarity": name_token_similarity,
        "name_jaccard_similarity": name_jaccard,
        "name_tfidf_similarity": name_tfidf_similarity,

        "description_similarity": description_similarity,
        "description_jaccard_similarity": description_jaccard,
        "description_tfidf_similarity": description_tfidf_similarity,

        "manufacturer_exact_match": manufacturer_exact_match,
        "manufacturer_conflict": manufacturer_conflict,
        "manufacturer_both_present": int(
            manufacturer_both_present
        ),

        "price_difference": price_difference,
        "price_relative_difference": price_relative_difference,
        "price_both_present": int(price_both_present),

        "name_length_difference": abs(
            len(name_a) - len(name_b)
        ),

        "name_token_count_difference": abs(
            len(token_set(name_a)) - len(token_set(name_b))
        ),

        "name_missing_either": int(
            not name_a or not name_b
        ),

        "description_missing_either": int(
            not desc_a or not desc_b
        ),

        "manufacturer_missing_either": int(
            not manufacturer_a or not manufacturer_b
        ),

        "blocking_similarity": float(
            pair.get("blocking_similarity", 0.0)
        )
    }


# --------------------------------------------------
# BUILD FEATURE DATASET FOR ONE SPLIT
# --------------------------------------------------

def create_feature_dataset(
    split_name,
    abt,
    buy,
    mapping,
    vectorizers
):

    candidate_path = (
        PROCESSED_DIR / f"{split_name}_candidates.csv"
    )

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"{candidate_path} not found. Run Day 5 first."
        )

    candidates = pd.read_csv(
        candidate_path,
        dtype={"abt_id": "string", "buy_id": "string"}
    )

    if candidates.empty:
        raise ValueError(
            f"No candidate pairs found for {split_name}."
        )

    # Candidate IDs define the records in this split.
    abt_ids = set(candidates["abt_id"].astype(str))
    buy_ids = set(candidates["buy_id"].astype(str))

    split_abt = abt[
        abt["id"].astype(str).isin(abt_ids)
    ].copy().reset_index(drop=True)

    split_buy = buy[
        buy["id"].astype(str).isin(buy_ids)
    ].copy().reset_index(drop=True)

    abt_lookup = build_lookup(split_abt)
    buy_lookup = build_lookup(split_buy)

    abt_positions = build_position_lookup(split_abt)
    buy_positions = build_position_lookup(split_buy)

    # Apply training-fitted vectorizers.
    abt_name_vectors = transform_catalogue(
        split_abt,
        "name_clean",
        vectorizers["name"]
    )

    buy_name_vectors = transform_catalogue(
        split_buy,
        "name_clean",
        vectorizers["name"]
    )

    abt_desc_vectors = transform_catalogue(
        split_abt,
        "description_clean",
        vectorizers["description"]
    )

    buy_desc_vectors = transform_catalogue(
        split_buy,
        "description_clean",
        vectorizers["description"]
    )

    # Ground-truth labels.
    true_matches = set(
        zip(
            mapping["abt_id"].astype(str),
            mapping["buy_id"].astype(str)
        )
    )

    rows = []

    for pair in candidates.to_dict("records"):

        features = calculate_pair_features(
            pair,
            abt_lookup,
            buy_lookup,
            abt_positions,
            buy_positions,
            abt_name_vectors,
            buy_name_vectors,
            abt_desc_vectors,
            buy_desc_vectors
        )

        abt_id = str(pair["abt_id"])
        buy_id = str(pair["buy_id"])

        features["abt_id"] = abt_id
        features["buy_id"] = buy_id

        # Benchmark closed-world labelling:
        # known ground-truth pair = 1; other candidate = 0.
        features["label"] = int(
            (abt_id, buy_id) in true_matches
        )

        rows.append(features)

    result = pd.DataFrame(rows)

    return result


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("MATCHWISE - PAIRWISE FEATURE ENGINEERING")
    print("=" * 65)

    abt, buy, mapping = load_catalogues()

    # Load training candidates to identify training records.
    train_candidates = load_csv(
        "train_candidates.csv",
        ["abt_id", "buy_id"]
    )

    train_abt_ids = set(
        train_candidates["abt_id"].astype(str)
    )

    train_buy_ids = set(
        train_candidates["buy_id"].astype(str)
    )

    train_abt = abt[
        abt["id"].astype(str).isin(train_abt_ids)
    ]

    train_buy = buy[
        buy["id"].astype(str).isin(train_buy_ids)
    ]

    # Fit text vectorizers ONLY on training records.
    name_vectorizer = fit_text_vectorizer(
        train_abt,
        train_buy,
        "name_clean"
    )

    description_vectorizer = fit_text_vectorizer(
        train_abt,
        train_buy,
        "description_clean"
    )

    vectorizers = {
        "name": name_vectorizer,
        "description": description_vectorizer
    }

    joblib.dump(
        vectorizers,
        MODEL_DIR / "text_vectorizers.joblib"
    )

    print("\nFitted vectorizers on training catalogue records.")

    summary = {}

    for split_name in ["train", "validation", "test"]:

        print(f"\nProcessing {split_name} candidates...")

        result = create_feature_dataset(
            split_name,
            abt,
            buy,
            mapping,
            vectorizers
        )

        output_path = (
            PROCESSED_DIR / f"{split_name}_features.csv"
        )

        result.to_csv(output_path, index=False)

        feature_columns = [
            col for col in result.columns
            if col not in ["abt_id", "buy_id", "label"]
        ]

        summary[split_name] = {
            "rows": int(len(result)),
            "feature_count": len(feature_columns),
            "positive_pairs": int((result["label"] == 1).sum()),
            "negative_pairs": int((result["label"] == 0).sum()),
            "feature_columns": feature_columns
        }

        print("Candidate pairs:", len(result))
        print("Feature count:", len(feature_columns))
        print("Positive labels:", int((result["label"] == 1).sum()))
        print("Negative labels:", int((result["label"] == 0).sum()))
        print("Saved:", output_path)

    with open(
        REPORT_DIR / "feature_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING COMPLETED")
    print("Saved text_vectorizers.joblib")
    print("Saved train_features.csv")
    print("Saved validation_features.csv")
    print("Saved test_features.csv")
    print("Saved feature_summary.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
