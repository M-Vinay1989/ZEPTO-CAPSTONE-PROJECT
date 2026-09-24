# Module 2: Customer Analytics & Predictive Modeling

This module implements a complete customer analytics and regression modeling pipeline for the Zepto Data & AI Platform capstone project.

---

## 1. Module Objective & Central Analytical Question

- **Objective**: The analysis focuses on understanding which customer characteristics are associated with annual spending and evaluating how accurately spending can be predicted.
- **Central Analytical Question**:
  > *"Can we predict a customer's annual spending from their demographic and purchasing characteristics?"*

---

## 2. Architecture & Data Flow

```
[ Synthetic Customer Generator ] ──► analytics/data/customer_spending.csv
                 │
                 ▼ (src/data_loader.py)
[ Data Loading & Profiling ] (1,020 raw rows -> 1,000 unique profiles)
                 │
                 ▼ (src/preprocessing.py)
[ Data Cleaning & Imputation ] (Duplicate removal, invalid entry fix, median imputation)
                 │
                 ├──► (src/analysis.py) ──► EDA Visualizations in analytics/output/figures/
                 │
                 ▼ (src/preprocessing.py)
[ Feature Engineering & Preprocessing ] (StandardScaler, OneHotEncoder, ColumnTransformer)
                 │
                 ▼ (src/modeling.py)
[ Model Training ] (Linear Regression & Random Forest Regressor)
                 │
                 ▼ (src/evaluation.py)
[ Evaluation & Summary ] ──► model_results.json & analysis_summary.md
```

---

## 3. Dataset Overview & Features

- **Source**: Synthetic customer dataset generated reproducibly (`seed=42`).
- **Rows**: The raw dataset contains 1,020 rows representing 1,000 unique customer profiles, with 20 duplicate rows included to demonstrate data cleaning.
- **Predictor Variables**:
  - `age`: Customer age in years (18–70)
  - `income`: Annual income in USD ($25,000–$130,000)
  - `membership_years`: Duration of membership in years (0.5–10.0)
  - `number_of_purchases`: Total purchases made (2–60)
  - `average_order_value`: Mean order dollar amount ($20.00–$300.00)
  - `discount_usage`: Percentage of orders using discounts (0.0–0.6)
  - `website_visits`: Annual website visit frequency (5–120)
  - `preferred_category`: Primary product category (`Electronics`, `Fashion`, `Groceries`, `Home & Kitchen`, `Books`)
- **Target Variable**:
  - `annual_spending`: Continuous float representing annual spending in USD ($50.00+).

---

## 4. Data Cleaning & Preprocessing

1. **Duplicate Removal**: Identified and dropped 20 exact duplicate rows.
2. **Invalid Entries**: Corrected negative age entries to column median. Replaced negative income values with `NaN`.
3. **Missing Value Imputation**: Imputed missing values in `income` and `website_visits` using column medians.
4. **Feature Engineering**:
   - `purchase_frequency`: `number_of_purchases / (membership_years + 0.1)`
   - `visit_conversion_rate`: `number_of_purchases / (website_visits + 1.0)`
   *(Note: The small constants `+ 0.1` and `+ 1.0` help avoid division-by-zero errors when the denominator is zero).*
5. **ColumnTransformer Preprocessing**:
   - Numerical features: `StandardScaler()`
   - Categorical features: `OneHotEncoder(handle_unknown='ignore')`
   - Split: 80% Train / 20% Test (`seed=42`). Preprocessor fitted **only** on training data to prevent data leakage.

---

## 5. Regression Models & Evaluation Metrics

Two regression models were implemented and evaluated on the test dataset:

1. **Linear Regression**: Baseline interpretable reference model.
2. **Random Forest Regressor**: Non-linear ensemble model (`n_estimators=100`, `random_state=42`).

### Evaluation Metrics & Performance Summary:
- **MAE** (Mean Absolute Error): Average absolute magnitude of prediction errors in USD.
- **MSE** (Mean Squared Error): Variance of prediction errors in USD².
- **RMSE** (Root Mean Squared Error): Standard deviation of prediction errors in USD.
- **R² Score** (Coefficient of Determination): Proportion of variance explained by model predictors.

> **Key Finding**: The Random Forest Regressor achieved an R² of 0.9677 on the held-out test set. In this synthetic dataset, average order value, number of purchases, and income were among the most important predictive features. Results on real customer data may differ.

---

## 6. How to Run the Module

### 1. Run Complete Pipeline via Main CLI
From the repository root:

```bash
python analytics/src/main.py
```
*or*
```bash
python -m analytics.src.main
```

### 2. Run Automated Test Suite
```bash
pytest analytics/tests/test_analytics.py
```

---

## 7. Limitations & Assumptions

- Data is generated synthetically using Gaussian noise assumptions.
- Real-world e-commerce datasets may contain long-tailed spending distributions, seasonal trends, and non-stationary customer behaviors.
