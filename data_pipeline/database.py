"""
Module 1: Database Component
Manages SQLite database schema creation, category normalization, foreign key constraints,
and population from cleaned pandas DataFrame.
"""

import os
import sqlite3
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "zepto_books.db")

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Connect to SQLite database and enable foreign key enforcement.
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def create_schema(conn: sqlite3.Connection) -> None:
    """
    Create normalized schema with categories and books tables.
    """
    cursor = conn.cursor()

    # Drop existing tables if re-initialization is needed
    cursor.execute("DROP TABLE IF EXISTS books;")
    cursor.execute("DROP TABLE IF EXISTS categories;")

    # Table 1: categories
    cursor.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)

    # Table 2: books
    cursor.execute("""
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    print("[INFO] SQLite schema created successfully with foreign key enforcement.")

def save_to_database(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Populates categories and books tables from cleaned DataFrame.
    Returns path to database file.
    """
    conn = get_connection(db_path)
    try:
        create_schema(conn)
        cursor = conn.cursor()

        # 1. Populate categories table uniquely
        unique_categories = sorted(df["category"].unique())
        cat_map = {}

        for cat_name in unique_categories:
            cursor.execute("INSERT INTO categories (category_name) VALUES (?);", (cat_name,))
            cat_id = cursor.lastrowid
            cat_map[cat_name] = cat_id

        # 2. Populate books table using category_id foreign keys
        books_data = []
        for _, row in df.iterrows():
            cat_id = cat_map[row["category"]]
            books_data.append((
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                1 if bool(row["in_stock"]) else 0,
                cat_id
            ))

        cursor.executemany("""
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?);
        """, books_data)

        conn.commit()
        print(f"[INFO] Inserted {len(cat_map)} categories and {len(books_data)} books into '{db_path}'.")
    finally:
        conn.close()

    return db_path

def verify_foreign_keys_enabled(db_path: str = DEFAULT_DB_PATH) -> bool:
    """Check if foreign key enforcement is active in SQLite connection."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys;")
        status = cursor.fetchone()[0]
        return status == 1
    finally:
        conn.close()

if __name__ == "__main__":
    from scraper import scrape_books
    df_books = scrape_books()
    db_file = save_to_database(df_books)
    print("Foreign Keys Enabled:", verify_foreign_keys_enabled(db_file))
