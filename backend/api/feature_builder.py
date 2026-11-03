
import html
import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rapidfuzz import fuzz


# --------------------------------------------------
# 1. PATHS AND SAVED VECTORIZERS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

VECTORIZER_PATH = BASE_DIR / "models" / "text_vectorizers.joblib"
BLOCKING_VECTORIZER_PATH = BASE_DIR / "models" / "blocking_vectorizer.joblib"
SCHEMA_PATH = BASE_DIR / "models" / "xgboost_schema.json"

for path in [
    VECTORIZER_PATH,
    BLOCKING_VECTORIZER_PATH,
    SCHEMA_PATH
]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

VECTORIZERS = joblib.load(VECTORIZER_PATH)
BLOCKING_VECTORIZER = joblib.load(BLOCKING_VECTORIZER_PATH)

with open(SCHEMA_PATH, "r") as file:
    SCHEMA = json.load(file)

FEATURE_COLUMNS = SCHEMA["feature_columns"]


# --------------------------------------------------
# 2. TEXT CLEANING
# Same cleaning logic as Day 3
# --------------------------------------------------

def clean_text(value):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    text = str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# 3. TOKEN AND JACCARD SIMILARITY
# --------------------------------------------------

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


# --------------------------------------------------
# 4. PRICE EXTRACTION
# --------------------------------------------------

def get_price(value):

    if value is None or pd.isna(value):
        return np.nan

    price_text = re.sub(r"[^\d.-]", "", str(value))

    try:
        return float(price_text)
    except (ValueError, TypeError):
        return np.nan


# --------------------------------------------------
# 5. TF-IDF COSINE SIMILARITY
# Reuse saved training vectorizers
# --------------------------------------------------

def calculate_tfidf_similarity(text1, text2, vectorizer):

    if not text1 or not text2:
        return 0.0

    vectors = vectorizer.transform([text1, text2])

    similarity = vectors[0].multiply(vectors[1]).sum()

    return float(similarity)


# --------------------------------------------------
# 6. BLOCKING SIMILARITY
# Use the exact saved blocking vectorizer
# --------------------------------------------------

def calculate_blocking_similarity(
    name_a,
    desc_a,
    manufacturer_a,
    name_b,
    desc_b,
    manufacturer_b
):

    # Match the text composition used during candidate generation.
    product_text_a = " ".join([
        name_a,
        desc_a,
        manufacturer_a
    ]).strip()

    product_text_b = " ".join([
        name_b,
        desc_b,
        manufacturer_b
    ]).strip()

    if not product_text_a:
        product_text_a = "unknownproduct"

    if not product_text_b:
        product_text_b = "unknownproduct"

    vectors = BLOCKING_VECTORIZER.transform([
        product_text_a,
        product_text_b
    ])

    similarity = vectors[0].multiply(vectors[1]).sum()

    return float(similarity)


# --------------------------------------------------
# 7. BUILD FEATURES FOR TWO PRODUCTS
# --------------------------------------------------

def build_features(product_a, product_b):

    # Clean product names and descriptions.
    name_a = clean_text(product_a.get("name"))
    name_b = clean_text(product_b.get("name"))

    desc_a = clean_text(product_a.get("description"))
    desc_b = clean_text(product_b.get("description"))

    # Optional manufacturer values are retained only
    # for the blocking similarity calculation.
    manufacturer_a = clean_text(
        product_a.get("manufacturer")
    )

    manufacturer_b = clean_text(
        product_b.get("manufacturer")
    )

    # ----------------------------------
    # 1. NAME SIMILARITY
    # ----------------------------------

    name_char_similarity = (
        fuzz.ratio(name_a, name_b) / 100.0
    )

    name_token_similarity = (
        fuzz.token_sort_ratio(name_a, name_b) / 100.0
    )

    name_jaccard = jaccard_similarity(
        name_a, name_b
    )

    # ----------------------------------
    # 2. DESCRIPTION SIMILARITY
    # ----------------------------------

    description_similarity = (
        fuzz.ratio(desc_a, desc_b) / 100.0
    )

    description_jaccard = jaccard_similarity(
        desc_a, desc_b
    )

    # ----------------------------------
    # 3. TF-IDF SIMILARITY
    # ----------------------------------

    name_tfidf_similarity = calculate_tfidf_similarity(
        name_a,
        name_b,
        VECTORIZERS["name"]
    )

    description_tfidf_similarity = calculate_tfidf_similarity(
        desc_a,
        desc_b,
        VECTORIZERS["description"]
    )

    # ----------------------------------
    # 4. PRICE COMPARISON
    # ----------------------------------

    price_a = get_price(product_a.get("price"))
    price_b = get_price(product_b.get("price"))

    price_both_present = (
        not pd.isna(price_a)
        and not pd.isna(price_b)
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

    # ----------------------------------
    # 5. BLOCKING SIMILARITY
    # ----------------------------------

    blocking_similarity = calculate_blocking_similarity(
        name_a,
        desc_a,
        manufacturer_a,
        name_b,
        desc_b,
        manufacturer_b
    )

    # ----------------------------------
    # 6. FINAL 15 FEATURES
    # ----------------------------------

    features = {

        "name_char_similarity": name_char_similarity,

        "name_token_similarity": name_token_similarity,

        "name_jaccard_similarity": name_jaccard,

        "name_tfidf_similarity": name_tfidf_similarity,

        "description_similarity": description_similarity,

        "description_jaccard_similarity": description_jaccard,

        "description_tfidf_similarity": description_tfidf_similarity,

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

        "blocking_similarity": blocking_similarity
    }

    # Enforce exact training feature order.
    feature_df = pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    return feature_df
