# Module 1: Data Pipeline (Zepto Data & AI Platform)

This module implements an end-to-end automated data pipeline that extracts book data from `books.toscrape.com`, performs data cleaning and currency conversion, loads normalized tables into an SQLite database with foreign key enforcement, executes SQL queries with Pandas integration, validates SQL JOIN vs `pd.merge()` equivalence, and runs an automated 17-point validation suite.

---

## 1. Module Purpose
The purpose of Module 1 is to establish a reliable, reproducible, and fully automated ETL (Extract, Transform, Load) pipeline for e-commerce book catalog data. It demonstrates best practices in web scraping, data normalization, database schema design, SQL querying, and pandas analysis.

---

## 2. Architecture & Data Flow

```
[ books.toscrape.com ]
         │
         ▼ (requests & BeautifulSoup)
[ Raw Scraped Records ] ──► (scraper.py)
         │
         ▼ (Data Cleaning & Imputation)
[ Cleaned DataFrame ] (price_inr = price_gbp * 105.50)
         │
         ▼ (database.py)
[ SQLite Database: zepto_books.db ]
  ├── categories (category_id PK, category_name UNIQUE)
  └── books (book_id PK, category_id FK -> categories)
         │
         ├──► (queries.py) ──► 5 SQL Queries + pd.read_sql()
         │                          │
         │                          ▼
         └──► (validation.py) ──► SQL JOIN vs pd.merge() Equivalence Check
                                   + 17 Automated Validation Checks
```

---

## 3. Installation & Requirements

Ensure Python 3.8+ is installed. Install required dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies used:
- `requests` (HTTP extraction)
- `beautifulsoup4` (HTML parsing)
- `pandas` (Data cleaning, transformation, and SQL integration)
- `sqlite3` (Built-in relational database engine)

---

## 4. How to Run the Module

Execute the pipeline using either of the following commands from the repository root:

```bash
python data_pipeline/run_pipeline.py
```

Or using python module syntax:

```bash
python -m data_pipeline.run_pipeline
```

---

## 5. Data Source
- **Website**: [Books to Scrape](https://books.toscrape.com/)
- **Scope**: Multi-category pagination scraping until at least 60 books across at least 3 categories are retrieved.

---

## 6. Scraping Approach
- Uses `requests` with HTTP timeout handling (`timeout=10`).
- Iterates over side categories (`Travel`, `Mystery`, `Historical Fiction`, etc.) and handles pagination (`<li class="next">`) within category links.
- Extracts `title`, `price_gbp`, `rating`, `in_stock`, and `category`.

---

## 7. Data Cleaning Decisions
- **Title**: Stripped whitespace and replaced malformed Unicode quotes (`â€™` -> `'`).
- **Category**: Trimmed whitespace.
- **Price (GBP)**: Extracted numeric floating-point values. Missing values are imputed using the column median of valid values.
- **Rating**: Converted CSS star-rating classes (`"One"`, `"Two"`, `"Three"`, `"Four"`, `"Five"`) to integers (`1`..`5`).
- **In Stock**: Normalized availability string into boolean (`True`/`False`).

---

## 8. Currency Conversion (GBP → INR)
- **Constant**: `GBP_TO_INR_RATE = 105.50`
- **Formula**: `price_inr = round(price_gbp * 105.50, 2)`

---

## 9. SQLite Database Schema

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

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
```

---

## 10. Implemented SQL Queries

1. **WHERE Filtering**: Filter books with rating >= 4.
2. **ORDER BY + LIMIT**: Retrieve top 5 most expensive books in INR.
3. **DISTINCT**: Retrieve list of unique category names.
4. **IN / BETWEEN**: Filter books with `price_gbp` between 20.0 and 40.0.
5. **JOIN**: Join `books` and `categories` tables on `category_id`.

---

## 11. Pandas Analysis & SQL Integration
- SQL queries are executed directly into Pandas DataFrames using `pd.read_sql_query()`.
- Demonstrates loading raw relational data for further in-memory analysis.

---

## 12. SQL JOIN vs `pd.merge()` Validation
To validate relational consistency, an in-memory `pd.merge()` is performed on raw `books` and `categories` DataFrames:

```python
pandas_merged_df = pd.merge(raw_books_df, raw_cats_df, on="category_id", how="inner")
```

The output structure and row order are compared against `sql_join_df`. Equivalence is empirically asserted using `sql_join_df.equals(pandas_merged_df)`.

---

## 13. Automated Validation Suite Summary
The pipeline executes 17 automated checks:
1. `books_count_gte_60`: Books count >= 60 (PASS)
2. `categories_count_gte_3`: Categories count >= 3 (PASS)
3. `price_gbp_numeric`: Float dtype (PASS)
4. `rating_integer`: Int dtype (PASS)
5. `rating_between_1_and_5`: Rating between 1 and 5 (PASS)
6. `in_stock_boolean`: Boolean dtype (PASS)
7. `price_inr_conversion_correct`: Price INR calculation correct (PASS)
8. `categories_table_exists`: `categories` table created (PASS)
9. `books_table_exists`: `books` table created (PASS)
10. `primary_keys_exist`: Primary keys present (PASS)
11. `foreign_key_relationship_exists`: FK relation established (PASS)
12. `foreign_keys_enforced`: `PRAGMA foreign_keys = 1` active (PASS)
13. `sql_5_queries_executed`: 5 SQL queries executed (PASS)
14. `required_sql_clauses_represented`: SELECT, WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, JOIN present (PASS)
15. `join_executed_successfully`: SQL JOIN returned data (PASS)
16. `pd_read_sql_loaded_2_queries`: `pd.read_sql` loaded multiple tables (PASS)
17. `sql_join_vs_pd_merge_equivalent`: `pd.merge()` matches SQL JOIN (PASS)

---

## 14. Important Design Decisions
- **Isolation**: Work is confined exclusively to `data_pipeline/`.
- **Reproducibility**: Database schema and data population are dynamically re-creatable via `run_pipeline.py`.
- **Strict Typing & Constraints**: SQLite foreign keys are explicitly enforced via connection pragmas.

---

## 15. Limitations & Assumptions
- Fixed conversion rate of 1 GBP = 105.50 INR as requested.
- Scraping target relies on `books.toscrape.com` HTML layout structure remaining constant.
