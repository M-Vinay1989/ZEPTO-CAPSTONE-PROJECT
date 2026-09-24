"""
Main Orchestrator Entry Point for Module 2: Customer Analytics & Predictive Modeling.
Runs dataset loading/generation, profiling, cleaning, EDA, feature engineering,
preprocessing, model training, evaluation, plot export, and summary report generation.
"""

import sys
import os

# Add repository root folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analytics.src.data_loader import generate_synthetic_data, load_dataset, profile_dataset
from analytics.src.preprocessing import clean_dataset, engineer_features, prepare_data
from analytics.src.analysis import run_full_eda
from analytics.src.modeling import train_all_models
from analytics.src.evaluation import evaluate_and_export

def run_pipeline() -> bool:
    """Executes full Module 2 Analytics & Predictive Modeling workflow."""
    print("==================================================")
    print("STARTING MODULE 2: CUSTOMER ANALYTICS PIPELINE")
    print("==================================================")

    try:
        # Step 1: Load/Generate Dataset
        print("\n--- STEP 1: Data Loading & Synthetic Generation ---")
        df_raw = load_dataset()
        profile = profile_dataset(df_raw)
        print(f"Loaded Dataset: {profile['n_rows']} rows x {profile['n_cols']} columns.")

        # Step 2: Data Cleaning
        print("\n--- STEP 2: Data Cleaning & Imputation ---")
        df_clean = clean_dataset(df_raw)
        print(f"Cleaned Dataset: {len(df_clean)} rows remaining.")

        # Step 3: Exploratory Data Analysis
        print("\n--- STEP 3: Exploratory Data Analysis (EDA) ---")
        eda_figs = run_full_eda(df_clean)
        print(f"Generated {len(eda_figs)} EDA plot figures in 'analytics/output/figures/'.")

        # Step 4: Feature Engineering & Preprocessing
        print("\n--- STEP 4: Feature Engineering & Preprocessing Pipeline ---")
        df_fe = engineer_features(df_clean)
        data_dict = prepare_data(df_fe)
        print(f"Features Prepared: X_train shape = {data_dict['X_train_prep'].shape}, X_test shape = {data_dict['X_test_prep'].shape}.")

        # Step 5: Regression Modeling
        print("\n--- STEP 5: Training Regression Models ---")
        models_dict = train_all_models(data_dict)

        # Step 6: Model Evaluation & Report Generation
        print("\n--- STEP 6: Model Evaluation & Output Export ---")
        results = evaluate_and_export(models_dict, data_dict)

        print("\n==================================================")
        print("MODULE 2 PIPELINE COMPLETED SUCCESSFULLY")
        print("==================================================")

        print("\nModel Evaluation Summary:")
        for m_name, metrics in results.items():
            print(f"  [{m_name.upper()}] MAE: ${metrics['MAE']} | RMSE: ${metrics['RMSE']} | R²: {metrics['R2']}")

        return True

    except Exception as err:
        print(f"\n[CRITICAL ERROR] Module 2 pipeline failed: {err}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
