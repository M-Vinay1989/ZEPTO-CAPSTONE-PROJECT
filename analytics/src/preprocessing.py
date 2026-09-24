"""
Preprocessing and Cleaning Module for Module 2: Customer Analytics.
Handles data cleaning, missing value median imputation, outlier fix, feature engineering,
train/test splitting, and ColumnTransformer/Pipeline scaling and encoding.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans dataset by handling duplicates, invalid values (negative age/income),
    and median imputing missing numerical entries.
    """
    df_clean = df.copy()

    # 1. Drop exact duplicate rows
    n_before = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    n_dups_removed = n_before - len(df_clean)
    print(f"[Cleaning] Removed {n_dups_removed} duplicate rows.")

    # 2. Fix invalid numerical entries (e.g. negative age or negative income)
    df_clean["age"] = df_clean["age"].astype(float)
    invalid_age = df_clean["age"] < 0
    if invalid_age.any():
        median_age = df_clean.loc[~invalid_age, "age"].median()
        df_clean.loc[invalid_age, "age"] = median_age
        print(f"[Cleaning] Fixed {invalid_age.sum()} invalid age entries with median: {median_age}")

    invalid_income = df_clean["income"] < 0
    if invalid_income.any():
        df_clean.loc[invalid_income, "income"] = np.nan
        print(f"[Cleaning] Replaced {invalid_income.sum()} negative income entries with NaN for imputation.")

    # 3. Median imputation for missing values
    for num_col in ["income", "website_visits"]:
        if df_clean[num_col].isnull().any():
            median_val = df_clean[num_col].median()
            df_clean[num_col] = df_clean[num_col].fillna(median_val)
            print(f"[Cleaning] Imputed missing values in '{num_col}' with median: {median_val:.2f}")

    # Ensure target variable is valid numeric float and contains no NaNs
    df_clean["annual_spending"] = pd.to_numeric(df_clean["annual_spending"], errors="coerce")
    df_clean = df_clean.dropna(subset=["annual_spending"]).reset_index(drop=True)

    return df_clean

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain-relevant features without target leakage:
    1. purchase_frequency: Purchases per membership year.
    2. visit_conversion_rate: Number of purchases per website visit.
    """
    df_fe = df.copy()

    # Feature 1: Purchase frequency (purchases per year of membership)
    df_fe["purchase_frequency"] = (
        df_fe["number_of_purchases"] / (df_fe["membership_years"] + 0.1)
    ).round(2)

    # Feature 2: Visit conversion rate (purchases per visit)
    df_fe["visit_conversion_rate"] = (
        df_fe["number_of_purchases"] / (df_fe["website_visits"] + 1.0)
    ).round(4)

    print("[Feature Engineering] Created 'purchase_frequency' and 'visit_conversion_rate'.")
    return df_fe

def prepare_data(df: pd.DataFrame, target_col: str = "annual_spending", test_size: float = 0.2, seed: int = 42):
    """
    Splits feature matrix X and target y into train/test sets,
    and returns fit ColumnTransformer preprocessor with X_train_prep and X_test_prep.
    """
    X = df.drop(columns=[target_col, "customer_id"], errors="ignore")
    y = df[target_col]

    # Categorical and numerical feature identification
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Scikit-learn Preprocessing Pipeline
    num_pipeline = Pipeline([("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # Fit on training data only to prevent data leakage
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # Extract feature names after one-hot encoding
    encoded_cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(encoded_cat_names)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_prep": X_train_prep,
        "X_test_prep": X_test_prep,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }

if __name__ == "__main__":
    from data_loader import load_dataset
    df_raw = load_dataset()
    df_clean = clean_dataset(df_raw)
    df_fe = engineer_features(df_clean)
    data_dict = prepare_data(df_fe)
    print(f"X_train shape: {data_dict['X_train_prep'].shape}, X_test shape: {data_dict['X_test_prep'].shape}")
