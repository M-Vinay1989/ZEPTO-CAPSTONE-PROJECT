"""
Regression Modeling Module for Module 2: Customer Analytics.
Trains Linear Regression (baseline model) and Random Forest Regressor (non-linear model).
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def train_linear_regression(X_train_prep, y_train):
    """
    Trains baseline Linear Regression model.
    """
    lr_model = LinearRegression()
    lr_model.fit(X_train_prep, y_train)
    print("[Modeling] Linear Regression baseline model trained.")
    return lr_model

def train_random_forest(X_train_prep, y_train, n_estimators: int = 100, seed: int = 42):
    """
    Trains Random Forest Regressor model with reproducible seed.
    """
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
    rf_model.fit(X_train_prep, y_train)
    print(f"[Modeling] Random Forest Regressor trained (trees={n_estimators}).")
    return rf_model

def train_all_models(data_dict: dict) -> dict:
    """
    Trains both Linear Regression and Random Forest models and predicts on test set.
    """
    X_train_prep = data_dict["X_train_prep"]
    y_train = data_dict["y_train"]
    X_test_prep = data_dict["X_test_prep"]
    y_test = data_dict["y_test"]

    # 1. Linear Regression
    lr_model = train_linear_regression(X_train_prep, y_train)
    y_pred_lr = lr_model.predict(X_test_prep)

    # 2. Random Forest Regressor
    rf_model = train_random_forest(X_train_prep, y_train)
    y_pred_rf = rf_model.predict(X_test_prep)

    return {
        "linear_regression": {
            "model": lr_model,
            "y_pred": y_pred_lr,
            "y_test": y_test
        },
        "random_forest": {
            "model": rf_model,
            "y_pred": y_pred_rf,
            "y_test": y_test
        }
    }

if __name__ == "__main__":
    from data_loader import load_dataset
    from preprocessing import clean_dataset, engineer_features, prepare_data
    df = engineer_features(clean_dataset(load_dataset()))
    d_dict = prepare_data(df)
    models_out = train_all_models(d_dict)
    print("Models trained successfully!")
