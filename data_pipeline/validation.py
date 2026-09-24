"""
Validation module for running automated checks on dataset shape, types,
database schema, foreign key enforcement, and SQL JOIN vs pd.merge equivalence.
"""

import sqlite3
import pandas as pd
from database import DEFAULT_DB_PATH, get_connection, verify_foreign_keys_enabled
from queries import execute_all_queries

GBP_TO_INR_RATE = 105.50

def validate_dataframe(df: pd.DataFrame) -> dict:
    """Validate raw/cleaned DataFrame properties."""
    results = {}

    # Check 1: Books count >= 60
    results["books_count_gte_60"] = {
        "status": len(df) >= 60,
        "detail": f"Book count = {len(df)}"
    }

    # Check 2: Categories count >= 3
    num_cats = df["category"].nunique() if "category" in df.columns else 0
    results["categories_count_gte_3"] = {
        "status": num_cats >= 3,
        "detail": f"Category count = {num_cats}"
    }

    # Check 3: price_gbp numeric
    results["price_gbp_numeric"] = {
        "status": pd.api.types.is_numeric_dtype(df["price_gbp"]),
        "detail": f"Dtype = {df['price_gbp'].dtype}"
    }

    # Check 4: rating integer
    results["rating_integer"] = {
        "status": pd.api.types.is_integer_dtype(df["rating"]),
        "detail": f"Dtype = {df['rating'].dtype}"
    }

    # Check 5: rating between 1 and 5
    valid_ratings = df["rating"].between(1, 5).all()
    results["rating_between_1_and_5"] = {
        "status": bool(valid_ratings),
        "detail": f"Min rating = {df['rating'].min()}, Max rating = {df['rating'].max()}"
    }

    # Check 6: in_stock boolean
    results["in_stock_boolean"] = {
        "status": pd.api.types.is_bool_dtype(df["in_stock"]),
        "detail": f"Dtype = {df['in_stock'].dtype}"
    }

    # Check 7: price_inr == price_gbp * 105.50
    expected_inr = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)
    inr_diff = (df["price_inr"] - expected_inr).abs().max()
    results["price_inr_conversion_correct"] = {
        "status": inr_diff < 0.02,
        "detail": f"Max difference = {inr_diff:.4f}"
    }

    return results

def validate_database_schema(db_path: str = DEFAULT_DB_PATH) -> dict:
    """Validate database tables, primary keys, foreign keys, and enforcement."""
    results = {}
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()

        # Check 8 & 9: Tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        results["categories_table_exists"] = {
            "status": "categories" in tables,
            "detail": f"Tables found: {tables}"
        }
        results["books_table_exists"] = {
            "status": "books" in tables,
            "detail": f"Tables found: {tables}"
        }

        # Check 10: Primary Keys exist
        cursor.execute("PRAGMA table_info(categories);")
        cat_pk = any(col[5] > 0 for col in cursor.fetchall())

        cursor.execute("PRAGMA table_info(books);")
        books_pk = any(col[5] > 0 for col in cursor.fetchall())

        results["primary_keys_exist"] = {
            "status": cat_pk and books_pk,
            "detail": f"categories PK={cat_pk}, books PK={books_pk}"
        }

        # Check 11: Foreign Keys relationship
        cursor.execute("PRAGMA foreign_key_list(books);")
        fk_list = cursor.fetchall()
        results["foreign_key_relationship_exists"] = {
            "status": len(fk_list) > 0,
            "detail": f"FK list: {fk_list}"
        }

        # Check 12: Foreign Keys active
        fk_enforced = verify_foreign_keys_enabled(db_path)
        results["foreign_keys_enforced"] = {
            "status": fk_enforced,
            "detail": f"PRAGMA foreign_keys = {1 if fk_enforced else 0}"
        }
    finally:
        conn.close()

    return results

def validate_sql_and_pandas_equivalence(db_path: str = DEFAULT_DB_PATH) -> dict:
    """Validate SQL execution and compare SQL JOIN with in-memory pd.merge()."""
    results = {}

    # Check 13: 5 SQL queries execution
    query_results = execute_all_queries(db_path)
    results["sql_5_queries_executed"] = {
        "status": len(query_results) >= 5,
        "detail": f"Executed {len(query_results)} queries successfully"
    }

    # Check 14: Required SQL clauses represented
    required_clauses = ["SELECT", "WHERE", "ORDER BY", "LIMIT", "DISTINCT", "BETWEEN", "JOIN"]
    all_sql_text = " ".join([q["sql"] for q in query_results.values()]).upper()
    missing_clauses = [c for c in required_clauses if c not in all_sql_text]
    results["required_sql_clauses_represented"] = {
        "status": len(missing_clauses) == 0,
        "detail": f"Represented: {required_clauses} (Missing: {missing_clauses})"
    }

    # Check 15: JOIN query execution
    join_df = query_results["query_5_join"]["data"]
    results["join_executed_successfully"] = {
        "status": len(join_df) > 0,
        "detail": f"JOIN returned {len(join_df)} rows"
    }

    # Check 16: pd.read_sql loading multiple tables
    conn = get_connection(db_path)
    try:
        df_cat = pd.read_sql("SELECT * FROM categories;", conn)
        df_book = pd.read_sql("SELECT * FROM books;", conn)
        results["pd_read_sql_loaded_2_queries"] = {
            "status": len(df_cat) > 0 and len(df_book) > 0,
            "detail": f"Loaded categories ({len(df_cat)} rows) and books ({len(df_book)} rows)"
        }

        # Check 17: SQL JOIN vs pd.merge Equivalence Test
        sql_join_query = """
        SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.book_id ASC;
        """
        sql_join_df = pd.read_sql(sql_join_query, conn)

        # In-memory pd.merge using raw books and categories DataFrames
        raw_books_df = pd.read_sql("SELECT * FROM books;", conn)
        raw_cats_df = pd.read_sql("SELECT * FROM categories;", conn)

        pandas_merged_df = pd.merge(
            raw_books_df,
            raw_cats_df,
            on="category_id",
            how="inner"
        )
        pandas_merged_df = pandas_merged_df[[
            "book_id", "title", "category_name", "price_gbp", "price_inr", "rating", "in_stock"
        ]].sort_values("book_id").reset_index(drop=True)

        sql_join_df = sql_join_df.sort_values("book_id").reset_index(drop=True)

        is_equivalent = sql_join_df.equals(pandas_merged_df)

        results["sql_join_vs_pd_merge_equivalent"] = {
            "status": is_equivalent,
            "detail": f"SQL JOIN rows={len(sql_join_df)}, pd.merge rows={len(pandas_merged_df)}, match={is_equivalent}"
        }
    finally:
        conn.close()

    return results

def run_full_validation(df_scraped: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> dict:
    """Run complete validation suite and return results."""
    print("\n==================================================")
    print("RUNNING AUTOMATED VALIDATION SUITE")
    print("==================================================")

    df_val = validate_dataframe(df_scraped)
    db_val = validate_database_schema(db_path)
    sql_val = validate_sql_and_pandas_equivalence(db_path)

    all_checks = {**df_val, **db_val, **sql_val}
    all_passed = True

    for check_name, info in all_checks.items():
        status_str = "PASS" if info["status"] else "FAIL"
        if not info["status"]:
            all_passed = False
        print(f"[{status_str}] {check_name:<38} | {info['detail']}")

    print("--------------------------------------------------")
    print(f"Overall Validation Result: {'PASS' if all_passed else 'FAIL'}")
    print("==================================================\n")

    return {
        "all_passed": all_passed,
        "checks": all_checks
    }

if __name__ == "__main__":
    from scraper import scrape_books
    from database import save_to_database
    df = scrape_books()
    db = save_to_database(df)
    run_full_validation(df, db)
