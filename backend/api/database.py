
import sqlite3
from pathlib import Path
from contextlib import contextmanager


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

DATABASE_DIR = BASE_DIR / "data" / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATABASE_DIR / "matchwise.db"


@contextmanager
def get_connection():

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def initialize_database():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS matching_records (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                product_a TEXT NOT NULL,

                product_b TEXT NOT NULL,

                match_probability REAL NOT NULL,

                model_decision TEXT NOT NULL,

                human_decision TEXT DEFAULT NULL,

                review_status TEXT NOT NULL DEFAULT 'NOT_REQUIRED',

                auto_match_threshold REAL NOT NULL,

                auto_non_match_threshold REAL NOT NULL,

                created_at TEXT NOT NULL DEFAULT (
                    STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'NOW')
                ),

                reviewed_at TEXT DEFAULT NULL

            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_matching_review_status
            ON matching_records(review_status)
        """)

    print("MatchWise SQLite database initialized.")
    print(f"Database location: {DATABASE_PATH}")


def save_matching_record(
    product_a,
    product_b,
    probability,
    model_decision,
    auto_match_threshold,
    auto_non_match_threshold
):

    review_status = (
        "PENDING"
        if model_decision == "HUMAN_REVIEW"
        else "NOT_REQUIRED"
    )

    with get_connection() as connection:

        cursor = connection.execute("""

            INSERT INTO matching_records (

                product_a,
                product_b,
                match_probability,
                model_decision,
                review_status,
                auto_match_threshold,
                auto_non_match_threshold

            )

            VALUES (?, ?, ?, ?, ?, ?, ?)

        """, (

            __import__("json").dumps(product_a),
            __import__("json").dumps(product_b),
            probability,
            model_decision,
            review_status,
            auto_match_threshold,
            auto_non_match_threshold

        ))

        return cursor.lastrowid
