"""
Automated Test Suite for Module 2: Customer Analytics & Predictive Modeling.
Tests dataset generation, loading, data cleaning, feature engineering, train/test split,
ColumnTransformer preprocessing, regression model training, and output report generation.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analytics.src.data_loader import generate_synthetic_data, load_dataset, profile_dataset
from analytics.src.preprocessing import clean_dataset, engineer_features, prepare_data
from analytics.src.modeling import train_linear_regression, train_random_forest, train_all_models
from analytics.src.evaluation import calculate_metrics, evaluate_and_export

def test_dataset_generation_and_loading(tmp_path):
    """Test synthetic data generation and loading."""
    csv_file = os.path.join(tmp_path, "test_spending.csv")
    df = generate_synthetic_data(n_samples=200, seed=123, output_path=csv_file)

    assert os.path.exists(csv_file)
    assert len(df) > 200  # includes injected duplicate rows
    assert "annual_spending" in df.columns
    assert "customer_id" in df.columns
    assert "income" in df.columns

    # Test loading
    df_loaded = load_dataset(csv_file)
    assert len(df_loaded) == len(df)

def test_data_profiling():
    """Test dataset profiling helper."""
    df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C2"],
        "age": [25, 30, 30],
        "annual_spending": [100.0, 200.0, 200.0]
    })
    prof = profile_dataset(df)
    assert prof["n_rows"] == 3
    assert prof["n_cols"] == 3
    assert prof["num_duplicates"] == 1

def test_data_cleaning():
    """Test duplicate removal, invalid numerical value fix, and missing value imputation."""
    df_raw = pd.DataFrame({
        "customer_id": ["C1", "C2", "C2", "C3"],
        "age": [25, -5, -5, 40],
        "income": [50000.0, np.nan, np.nan, 80000.0],
        "website_visits": [10, 20, 20, np.nan],
        "annual_spending": [500.0, 1000.0, 1000.0, 1500.0]
    })

    df_clean = clean_dataset(df_raw)

    assert len(df_clean) == 3  # 1 duplicate row removed
    assert (df_clean["age"] >= 0).all()  # Invalid age corrected
    assert df_clean["income"].isnull().sum() == 0  # Missing income imputed
    assert df_clean["website_visits"].isnull().sum() == 0  # Missing visits imputed

def test_feature_engineering():
    """Test feature engineering creates new non-leaking columns."""
    df = pd.DataFrame({
        "number_of_purchases": [10, 20],
        "membership_years": [2.0, 5.0],
        "website_visits": [50, 100],
        "annual_spending": [1000.0, 2000.0]
    })
    df_fe = engineer_features(df)

    assert "purchase_frequency" in df_fe.columns
    assert "visit_conversion_rate" in df_fe.columns
    assert df_fe["purchase_frequency"].iloc[0] > 0

def test_train_test_split_and_preprocessing():
    """Test prepare_data returns correctly split and scaled matrices."""
    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(100)],
        "age": np.random.randint(20, 60, 100),
        "income": np.random.uniform(30000, 90000, 100),
        "preferred_category": np.random.choice(["Electronics", "Fashion"], 100),
        "annual_spending": np.random.uniform(500, 5000, 100)
    })

    d_dict = prepare_data(df, test_size=0.2, seed=42)

    assert d_dict["X_train_prep"].shape[0] == 80
    assert d_dict["X_test_prep"].shape[0] == 20
    assert d_dict["X_train_prep"].shape[1] == d_dict["X_test_prep"].shape[1]
    assert len(d_dict["feature_names"]) > 0

def test_model_training_and_evaluation(tmp_path):
    """Test Linear Regression and Random Forest training and metrics calculation."""
    df = generate_synthetic_data(n_samples=300, seed=42, output_path=os.path.join(tmp_path, "data.csv"))
    df_clean = clean_dataset(df)
    df_fe = engineer_features(df_clean)
    data_dict = prepare_data(df_fe, test_size=0.2, seed=42)

    models_dict = train_all_models(data_dict)

    assert "linear_regression" in models_dict
    assert "random_forest" in models_dict

    # Check metrics
    out_dir = os.path.join(tmp_path, "output")
    results = evaluate_and_export(models_dict, data_dict, output_dir=out_dir)

    assert os.path.exists(os.path.join(out_dir, "model_results.json"))
    assert os.path.exists(os.path.join(out_dir, "analysis_summary.md"))
    assert "MAE" in results["linear_regression"]
    assert "R2" in results["random_forest"]
    assert results["random_forest"]["R2"] <= 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
