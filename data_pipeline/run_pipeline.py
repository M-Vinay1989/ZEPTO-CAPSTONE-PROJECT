"""
Module 1: Main Pipeline Orchestrator
Single entry point for data scraping, SQLite creation, SQL execution, Pandas integration,
equivalence testing, output generation, and automated validation.
"""

import sys
import os
import json
import pandas as pd

# Add module parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper import scrape_books
from database import save_to_database, DEFAULT_DB_PATH
from queries import execute_all_queries
from validation import run_full_validation

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

def export_outputs(query_results: dict, validation_results: dict, df_scraped: pd.DataFrame) -> None:
    """
    Saves pipeline outputs, SQL query results, and validation reports to output directory.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Save Scraped Dataset CSV
    csv_path = os.path.join(OUTPUT_DIR, "scraped_books_dataset.csv")
    df_scraped.to_csv(csv_path, index=False)
    print(f"[OUTPUT] Scraped dataset saved to '{csv_path}'")

    # 2. Save SQL Query Results Report
    report_path = os.path.join(OUTPUT_DIR, "sql_query_results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Module 1: SQL Query Results & Execution Output\n\n")
        for qkey, info in query_results.items():
            f.write(f"## {info['description']}\n\n")
            f.write("```sql\n" + info["sql"].strip() + "\n```\n\n")
            f.write("### Query Output:\n\n")
            f.write(info["data"].to_markdown(index=False))
            f.write("\n\n---\n\n")
    print(f"[OUTPUT] SQL query report saved to '{report_path}'")

    # 3. Save Validation Summary
    val_path = os.path.join(OUTPUT_DIR, "validation_summary.json")
    val_export = {
        "all_passed": validation_results["all_passed"],
        "checks": {k: v for k, v in validation_results["checks"].items()}
    }
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_export, f, indent=2)
    print(f"[OUTPUT] Validation summary saved to '{val_path}'")

def run_pipeline() -> bool:
    """
    Executes the complete Module 1 Data Pipeline.
    """
    print("==================================================")
    print("STARTING MODULE 1: ZEPTO DATA PIPELINE EXECUTION")
    print("==================================================")

    try:
        # Step 1: Data Extraction & Cleaning
        print("\n--- STEP 1: Web Scraping & Data Cleaning ---")
        df_scraped = scrape_books(min_books=60, min_categories=3)
        print(f"Scraped & Cleaned: {len(df_scraped)} books across {df_scraped['category'].nunique()} categories.")

        # Step 2: Database Setup & Population
        print("\n--- STEP 2: SQLite Database Population ---")
        db_path = save_to_database(df_scraped, DEFAULT_DB_PATH)

        # Step 3: SQL Query Execution & Pandas Integration
        print("\n--- STEP 3: Executing SQL Queries & Pandas Analysis ---")
        query_results = execute_all_queries(db_path)

        # Step 4: Validation Suite Execution
        print("\n--- STEP 4: Running Automated Validation Suite ---")
        val_res = run_full_validation(df_scraped, db_path)

        # Step 5: Export Generated Outputs
        print("\n--- STEP 5: Exporting Generated Outputs ---")
        export_outputs(query_results, val_res, df_scraped)

        print("\n==================================================")
        print("MODULE 1 PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("==================================================")
        return val_res["all_passed"]

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
