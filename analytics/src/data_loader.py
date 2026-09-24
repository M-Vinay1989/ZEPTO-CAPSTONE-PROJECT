"""
Data Loader and Synthetic Generator Module for Module 2: Customer Analytics.
Generates reproducible customer dataset with synthetic data quality issues and provides profiling utilities.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "customer_spending.csv"

def generate_synthetic_data(n_samples: int = 1000, seed: int = 42, output_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Generates a synthetic customer dataset with realistic attributes, continuous target `annual_spending`,
    and data quality issues (missing values, duplicates, malformed entries) for cleaning demonstration.
    """
    output_path = Path(output_path).resolve()
    np.random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Base Attributes
    customer_ids = [f"CUST_{1001 + i}" for i in range(n_samples)]
    age = np.random.randint(18, 70, size=n_samples)
    income = np.random.uniform(25000, 130000, size=n_samples).round(2)
    membership_years = np.random.uniform(0.5, 10.0, size=n_samples).round(1)
    number_of_purchases = np.random.randint(2, 60, size=n_samples)
    average_order_value = np.random.uniform(20.0, 300.0, size=n_samples).round(2)
    discount_usage = np.random.uniform(0.0, 0.6, size=n_samples).round(2)
    website_visits = np.random.randint(5, 120, size=n_samples)

    categories = ["Electronics", "Fashion", "Groceries", "Home & Kitchen", "Books"]
    preferred_category = np.random.choice(categories, size=n_samples, p=[0.25, 0.25, 0.20, 0.15, 0.15])

    # 2. Target Variable Generation (realistic domain relationship + noise)
    noise = np.random.normal(0, 350, size=n_samples)
    annual_spending = (
        (number_of_purchases * average_order_value * 0.85)
        + (income * 0.015)
        + (membership_years * 120.0)
        - (discount_usage * 150.0)
        + noise
    ).round(2)

    # Ensure non-negative annual spending
    annual_spending = np.clip(annual_spending, 50.0, None)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "income": income,
        "membership_years": membership_years,
        "number_of_purchases": number_of_purchases,
        "average_order_value": average_order_value,
        "discount_usage": discount_usage,
        "website_visits": website_visits,
        "preferred_category": preferred_category,
        "annual_spending": annual_spending
    })

    # 3. Inject Data Quality Issues
    # Missing values (~4% missing in income and website_visits)
    missing_income_idx = np.random.choice(n_samples, size=int(n_samples * 0.04), replace=False)
    missing_visits_idx = np.random.choice(n_samples, size=int(n_samples * 0.04), replace=False)
    df.loc[missing_income_idx, "income"] = np.nan
    df.loc[missing_visits_idx, "website_visits"] = np.nan

    # Invalid numerical value (e.g. negative age / income in 3 rows)
    invalid_idx = np.random.choice(n_samples, size=3, replace=False)
    df.loc[invalid_idx[0], "age"] = -5
    df.loc[invalid_idx[1], "income"] = -1000.0

    # Duplicate rows (inject 20 duplicate rows)
    dup_rows = df.iloc[:20].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Save CSV
    df.to_csv(output_path, index=False)
    print(f"[Data Loader] Synthetic dataset created at '{output_path}' with {len(df)} rows.")
    return df

def load_dataset(filepath: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Loads dataset from CSV filepath."""
    filepath = Path(filepath).resolve()
    if not filepath.exists():
        print(f"[Data Loader] File not found at '{filepath}'. Generating dataset...")
        return generate_synthetic_data(output_path=filepath)
    df = pd.read_csv(filepath)
    return df

def profile_dataset(df: pd.DataFrame) -> dict:
    """Profiles dataset structure, dimensions, missing values, duplicates, and summary stats."""
    null_counts = df.isnull().sum().to_dict()
    num_duplicates = int(df.duplicated().sum())

    profile = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": null_counts,
        "num_duplicates": num_duplicates,
        "summary_stats": df.describe().to_dict()
    }
    return profile

if __name__ == "__main__":
    df_gen = generate_synthetic_data()
    prof = profile_dataset(df_gen)
    print("Dataset Profile Summary:")
    print(f"Shape: {prof['n_rows']} rows x {prof['n_cols']} cols")
    print(f"Nulls: {prof['null_counts']}")
    print(f"Duplicates: {prof['num_duplicates']}")
