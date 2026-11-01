
import re
import html
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def find_file(keyword):
    """Find one input CSV without confusing catalogue and mapping files."""
    files = list(RAW_DIR.glob("*.csv"))

    if keyword == "abt":
        matches = [f for f in files if f.stem.lower() == "abt"]
    elif keyword == "buy":
        matches = [f for f in files if f.stem.lower() == "buy"]
    elif keyword == "mapping":
        matches = [f for f in files if "mapping" in f.stem.lower()]
    else:
        matches = [f for f in files if keyword.lower() in f.stem.lower()]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {keyword} file. Found: {[f.name for f in matches]}"
        )

    return matches[0]


def load_csv(path):
    """Load a CSV with common encoding support."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def clean_text(value):
    """
    Normalise text for comparison while retaining the original columns.

    This does not remove meaningful product information such as
    storage sizes, colours, model numbers or package quantities.
    """
    if pd.isna(value):
        return ""

    text = str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalise_column_names(df):
    """Standardise column headers for reliable downstream processing."""
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")
        for col in df.columns
    ]
    return df


def clean_catalogue(df, source_name):
    """
    Clean a product catalogue.

    Original values are preserved. Additional *_clean columns are
    created for text fields, and price_numeric is created if price exists.
    """
    df = normalise_column_names(df)

    if "id" not in df.columns:
        raise ValueError(
            f"{source_name}: expected an 'id' column. "
            f"Found columns: {df.columns.tolist()}"
        )

    # IDs are identifiers, not numeric measurements.
    df["id"] = df["id"].astype("string").str.strip()

    if df["id"].isna().any() or (df["id"] == "").any():
        raise ValueError(f"{source_name}: missing or blank product IDs found.")

    text_columns = [
        col for col in ["name", "description", "manufacturer"]
        if col in df.columns
    ]

    if "name" not in text_columns:
        raise ValueError(
            f"{source_name}: expected a product 'name' column. "
            f"Found columns: {df.columns.tolist()}"
        )

    for col in text_columns:
        df[f"{col}_clean"] = df[col].apply(clean_text)

    if "price" in df.columns:
        # Remove common currency symbols and thousands separators.
        price_text = (
            df["price"]
            .astype("string")
            .str.replace(r"[^\d.\-]", "", regex=True)
        )
        df["price_numeric"] = pd.to_numeric(
            price_text, errors="coerce"
        )

    # Explicit missingness indicators. Missing does not mean agreement.
    for col in text_columns:
        df[f"{col}_missing"] = df[col].isna() | (
            df[col].astype("string").str.strip() == ""
        )

    if "price" in df.columns:
        df["price_missing"] = df["price_numeric"].isna()

    df["source_catalogue"] = source_name

    return df


def clean_mapping(df):
    """Standardise the official ground-truth mapping columns."""
    df = normalise_column_names(df)

    # Expected benchmark mapping columns are idAbt and idBuy.
    required = ["idabt", "idbuy"]

    # Normalisation above turns these into idabt and idbuy.
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Could not identify the expected mapping columns "
            f"{required}. Found: {df.columns.tolist()}"
        )

    mapping = df[required].copy()
    mapping.columns = ["abt_id", "buy_id"]

    mapping["abt_id"] = mapping["abt_id"].astype("string").str.strip()
    mapping["buy_id"] = mapping["buy_id"].astype("string").str.strip()

    if mapping.isna().any().any():
        raise ValueError("Ground-truth mapping contains missing IDs.")

    if mapping.duplicated().any():
        print("WARNING: duplicate mapping pairs found.")

    return mapping.drop_duplicates().reset_index(drop=True)


def main():
    print("=" * 65)
    print("MATCHWISE - DATA PREPROCESSING")
    print("=" * 65)

    abt_path = find_file("abt")
    buy_path = find_file("buy")
    mapping_path = find_file("mapping")

    abt_raw = load_csv(abt_path)
    buy_raw = load_csv(buy_path)
    mapping_raw = load_csv(mapping_path)

    print("\nOriginal catalogue shapes:")
    print("Abt:", abt_raw.shape)
    print("Buy:", buy_raw.shape)
    print("Mapping:", mapping_raw.shape)

    abt_clean = clean_catalogue(abt_raw, "Abt")
    buy_clean = clean_catalogue(buy_raw, "Buy")
    mapping_clean = clean_mapping(mapping_raw)

    # Check for duplicate IDs, but do not silently remove records.
    if abt_clean["id"].duplicated().any():
        raise ValueError("Abt catalogue contains duplicate product IDs.")

    if buy_clean["id"].duplicated().any():
        raise ValueError("Buy catalogue contains duplicate product IDs.")

    # Validate that mapping IDs refer to records in the catalogues.
    abt_ids = set(abt_clean["id"].dropna())
    buy_ids = set(buy_clean["id"].dropna())

    invalid_abt = ~mapping_clean["abt_id"].isin(abt_ids)
    invalid_buy = ~mapping_clean["buy_id"].isin(buy_ids)

    if invalid_abt.any() or invalid_buy.any():
        raise ValueError(
            "Ground-truth mapping contains IDs missing from a catalogue. "
            f"Invalid Abt IDs: {int(invalid_abt.sum())}; "
            f"Invalid Buy IDs: {int(invalid_buy.sum())}."
        )

    # Save processed data separately from raw source files.
    abt_clean.to_csv(PROCESSED_DIR / "abt_clean.csv", index=False)
    buy_clean.to_csv(PROCESSED_DIR / "buy_clean.csv", index=False)
    mapping_clean.to_csv(
        PROCESSED_DIR / "ground_truth_clean.csv", index=False
    )

    print("\nPreprocessing completed.")
    print("Abt cleaned:", abt_clean.shape)
    print("Buy cleaned:", buy_clean.shape)
    print("Ground-truth pairs:", len(mapping_clean))
    print("\nSaved files:")
    print(PROCESSED_DIR / "abt_clean.csv")
    print(PROCESSED_DIR / "buy_clean.csv")
    print(PROCESSED_DIR / "ground_truth_clean.csv")


if __name__ == "__main__":
    main()
