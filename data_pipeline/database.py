"""
Database module for managing SQLite database creation, schema setup,
foreign key enforcement, and data insertion.
"""

import os
import sqlite3
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "zepto_books.db")

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Connect to SQLite database and enable foreign key constraints."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def create_schema(conn: sqlite3.Connection) -> None:
    """Create normalized categories and books relational tables."""
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS books;")
    cursor.execute("DROP TABLE IF EXISTS categories;")

    # Categories table
    cursor.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)

    # Books table with Foreign Key reference to categories
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
    print("[Info] SQLite schema created with foreign keys enabled.")

def save_to_database(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> str:
    """Populate categories and books tables from cleaned pandas DataFrame."""
    conn = get_connection(db_path)
    try:
        create_schema(conn)
        cursor = conn.cursor()

        # 1. Insert unique categories
        unique_categories = sorted(df["category"].unique())
        cat_map = {}

        for cat_name in unique_categories:
            cursor.execute("INSERT INTO categories (category_name) VALUES (?);", (cat_name,))
            cat_map[cat_name] = cursor.lastrowid

        # 2. Insert books referencing category_id
        books_rows = []
        for _, row in df.iterrows():
            cat_id = cat_map[row["category"]]
            books_rows.append((
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
        """, books_rows)

        conn.commit()
        print(f"[Info] Inserted {len(cat_map)} categories and {len(books_rows)} books into '{db_path}'.")
    finally:
        conn.close()

    return db_path

def verify_foreign_keys_enabled(db_path: str = DEFAULT_DB_PATH) -> bool:
    """Check if foreign key enforcement PRAGMA is active in SQLite."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys;")
        result = cursor.fetchone()
        return result is not None and result[0] == 1
    finally:
        conn.close()

if __name__ == "__main__":
    from scraper import scrape_books
    df_books = scrape_books()
    db_file = save_to_database(df_books)
    print("Foreign Keys Active:", verify_foreign_keys_enabled(db_file))
