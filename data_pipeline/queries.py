"""
Module 1: SQL Queries & Pandas Integration Component
Defines, executes, and formats 5 mandatory SQL queries demonstrating key SQL clauses
and pandas pd.read_sql() integration.
"""

import sqlite3
import pandas as pd
from database import DEFAULT_DB_PATH, get_connection

QUERIES = {
    "query_1_where": {
        "description": "High Rated Books (Rating >= 4)",
        "sql": """
        SELECT book_id, title, price_gbp, price_inr, rating
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC, title ASC;
        """
    },
    "query_2_order_by_limit": {
        "description": "Top 5 Most Expensive Books in INR",
        "sql": """
        SELECT book_id, title, price_gbp, price_inr
        FROM books
        ORDER BY price_inr DESC
        LIMIT 5;
        """
    },
    "query_3_distinct": {
        "description": "Distinct Categories in Database",
        "sql": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name ASC;
        """
    },
    "query_4_between": {
        "description": "Books in Price Range £20 to £40",
        "sql": """
        SELECT book_id, title, price_gbp, price_inr, rating
        FROM books
        WHERE price_gbp BETWEEN 20.0 AND 40.0
        ORDER BY price_gbp ASC;
        """
    },
    "query_5_join": {
        "description": "SQL JOIN: Books with Category Names",
        "sql": """
        SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name ASC, b.title ASC;
        """
    }
}

def execute_sql_query(query_key: str, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Executes a named SQL query against SQLite database using pd.read_sql().
    """
    if query_key not in QUERIES:
        raise KeyError(f"Query key '{query_key}' not defined.")

    conn = get_connection(db_path)
    try:
        sql = QUERIES[query_key]["sql"]
        df = pd.read_sql(sql, conn)
        return df
    finally:
        conn.close()

def execute_all_queries(db_path: str = DEFAULT_DB_PATH) -> dict:
    """
    Executes all 5 defined SQL queries and returns a dictionary of resulting DataFrames.
    """
    results = {}
    conn = get_connection(db_path)
    try:
        for key, info in QUERIES.items():
            print(f"[INFO] Executing Query: {info['description']}")
            df = pd.read_sql(info["sql"], conn)
            results[key] = {
                "description": info["description"],
                "sql": info["sql"],
                "data": df
            }
    finally:
        conn.close()

    return results

if __name__ == "__main__":
    res = execute_all_queries()
    for qname, qdata in res.items():
        print(f"\n--- {qdata['description']} ---")
        print("SQL:\n", qdata["sql"].strip())
        print("Result Sample (first 3 rows):\n", qdata["data"].head(3))
