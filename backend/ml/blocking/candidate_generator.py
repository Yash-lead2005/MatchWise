
from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 20
RANDOM_STATE = 42


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    abt = pd.read_csv(
        PROCESSED_DIR / "abt_clean.csv",
        dtype={"id": "string"}
    )

    buy = pd.read_csv(
        PROCESSED_DIR / "buy_clean.csv",
        dtype={"id": "string"}
    )

    mapping = pd.read_csv(
        PROCESSED_DIR / "ground_truth_clean.csv",
        dtype={"abt_id": "string", "buy_id": "string"}
    )

    return abt, buy, mapping


# --------------------------------------------------
# PREPARE PRODUCT TEXT
# --------------------------------------------------

def build_product_text(df):

    columns = [
        col for col in [
            "name_clean",
            "description_clean",
            "manufacturer_clean"
        ]
        if col in df.columns
    ]

    if not columns:
        raise ValueError(
            "No cleaned text columns found. "
            "Check the output of Day 3 preprocessing."
        )

    text = df[columns].fillna("").astype(str).agg(" ".join, axis=1)

    # Avoid empty documents.
    text = text.replace("", "unknownproduct")

    return text


# --------------------------------------------------
# GENERATE CANDIDATE PAIRS
# --------------------------------------------------

def generate_candidates(
    abt,
    buy,
    vectorizer,
    top_k=TOP_K
):

    if abt.empty or buy.empty:
        return pd.DataFrame(
            columns=["abt_id", "buy_id", "blocking_similarity"]
        )

    abt_text = build_product_text(abt)
    buy_text = build_product_text(buy)

    abt_vectors = vectorizer.transform(abt_text)
    buy_vectors = vectorizer.transform(buy_text)

    # Fit nearest-neighbour search on Buy products.
    neighbours = NearestNeighbors(
        n_neighbors=min(top_k, len(buy)),
        metric="cosine",
        algorithm="brute"
    )

    neighbours.fit(buy_vectors)

    distances, indices = neighbours.kneighbors(abt_vectors)

    candidates = []

    for abt_position, (row_distances, row_indices) in enumerate(
        zip(distances, indices)
    ):

        abt_id = str(abt.iloc[abt_position]["id"])

        for distance, buy_position in zip(
            row_distances,
            row_indices
        ):

            buy_id = str(buy.iloc[buy_position]["id"])

            similarity = 1.0 - float(distance)

            candidates.append({
                "abt_id": abt_id,
                "buy_id": buy_id,
                "blocking_similarity": similarity
            })

    result = pd.DataFrame(candidates)

    return result.drop_duplicates(
        subset=["abt_id", "buy_id"]
    ).reset_index(drop=True)


# --------------------------------------------------
# GET RECORDS BELONGING TO A SPLIT
# --------------------------------------------------

def load_split_records(split_name, abt, buy):

    path = PROCESSED_DIR / f"{split_name}_pairs.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Complete Day 4 first."
        )

    pairs = pd.read_csv(
        path,
        dtype={"abt_id": "string", "buy_id": "string"}
    )

    abt_ids = set(pairs["abt_id"].dropna().astype(str))
    buy_ids = set(pairs["buy_id"].dropna().astype(str))

    split_abt = abt[
        abt["id"].astype(str).isin(abt_ids)
    ].copy()

    split_buy = buy[
        buy["id"].astype(str).isin(buy_ids)
    ].copy()

    return split_abt, split_buy, pairs


# --------------------------------------------------
# BLOCKING EVALUATION
# --------------------------------------------------

def evaluate_blocking(candidates, pairs, abt, buy):

    # Known positive pairs in the selected split.
    true_matches = set(
        zip(
            pairs.loc[pairs["label"] == 1, "abt_id"].astype(str),
            pairs.loc[pairs["label"] == 1, "buy_id"].astype(str)
        )
    )

    candidate_pairs = set(
        zip(
            candidates["abt_id"].astype(str),
            candidates["buy_id"].astype(str)
        )
    )

    if not true_matches:
        raise ValueError(
            "No positive matching pairs found in this split."
        )

    found_matches = true_matches.intersection(candidate_pairs)

    blocking_recall = len(found_matches) / len(true_matches)

    total_possible_pairs = len(abt) * len(buy)

    candidate_count = len(candidate_pairs)

    reduction_ratio = (
        1 - candidate_count / total_possible_pairs
        if total_possible_pairs > 0 else 0
    )

    return {
        "abt_records": len(abt),
        "buy_records": len(buy),
        "total_possible_pairs": total_possible_pairs,
        "candidate_pairs": candidate_count,
        "known_matches": len(true_matches),
        "matches_found_by_blocking": len(found_matches),
        "matches_missed_by_blocking": (
            len(true_matches) - len(found_matches)
        ),
        "blocking_recall": blocking_recall,
        "candidate_reduction_ratio": reduction_ratio
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("MATCHWISE - CANDIDATE GENERATION AND BLOCKING")
    print("=" * 65)

    abt, buy, mapping = load_data()

    print("\nLoaded catalogues:")
    print("Abt:", len(abt))
    print("Buy:", len(buy))
    print("Ground-truth matches:", len(mapping))

    # Use training records to learn the text vocabulary.
    train_abt, train_buy, _ = load_split_records(
        "train", abt, buy
    )

    training_text = pd.concat(
        [
            build_product_text(train_abt),
            build_product_text(train_buy)
        ],
        ignore_index=True
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
        max_features=100000
    )

    vectorizer.fit(training_text)

    print("\nTF-IDF vocabulary size:", len(vectorizer.vocabulary_))
    print("Candidate neighbours per Abt record:", TOP_K)

    summary = {
        "method": "Character TF-IDF + Cosine Nearest Neighbours",
        "top_k": TOP_K,
        "vectorizer": {
            "analyzer": "char_wb",
            "ngram_range": [2, 5],
            "sublinear_tf": True
        },
        "splits": {}
    }

    for split_name in ["train", "validation", "test"]:

        split_abt, split_buy, split_pairs = load_split_records(
            split_name, abt, buy
        )

        candidates = generate_candidates(
            split_abt,
            split_buy,
            vectorizer,
            TOP_K
        )

        # Evaluate how many known matches blocking retained.
        metrics = evaluate_blocking(
            candidates,
            split_pairs,
            split_abt,
            split_buy
        )

        summary["splits"][split_name] = metrics

        candidates.to_csv(
            PROCESSED_DIR / f"{split_name}_candidates.csv",
            index=False
        )

        print(f"\n{split_name.upper()} RESULTS")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    # Save evaluation report.
    with open(
        REPORT_DIR / "blocking_evaluation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 65)
    print("BLOCKING COMPLETED")
    print("Saved train_candidates.csv")
    print("Saved validation_candidates.csv")
    print("Saved test_candidates.csv")
    print("Saved blocking_evaluation.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
