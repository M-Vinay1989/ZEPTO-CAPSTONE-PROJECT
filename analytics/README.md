# Module 2 — Titanic Analytics Pipeline & Predictive Modeling

This directory contains the complete, reproducible implementation of **Module 2 — Titanic Analytics Pipeline & Predictive Modeling** for the Zepto Data & AI Platform capstone project.

---

## 1. Overview & Dataset Source
The analytics pipeline utilizes the Seaborn **Titanic** dataset to perform end-to-end exploratory data analysis (EDA), data cleaning, feature engineering, classification modeling, class imbalance experimentation, hyperparameter tuning, regression side-task modeling, and complete model pipeline serialization.

---

## 2. Dataset Loading Architecture & Offline Fallback
- **Single Load Rule**: The raw dataset is loaded **exactly once** across the entire analytics module using Seaborn's dataset loader in `analytics/01_eda.ipynb`:
  ```python
  df = sns.load_dataset("titanic")
  df.to_csv("titanic.csv", index=False)
  ```
- **Offline Fallback**: `analytics/titanic.csv` is committed as the mandatory offline dataset snapshot.
- **Modeling Isolation**: `analytics/02_modeling.ipynb` starts strictly from `pd.read_csv("titanic.csv")` and **never** calls `sns.load_dataset("titanic")`.

---

## 3. Missing-Value Breakdown & Cleaning Decisions

| Column | Measured Missing Count | Missing % | Category Threshold | Chosen Treatment | Reason for Treatment |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **deck** | 688 | 77.22% | >30% missing | Drop Column | Over 77% missing data; imputing would introduce severe synthetic noise without adding reliable predictive signal. |
| **age** | 177 | 19.87% | 5%–30% missing | Median Imputation | 19.87% missing; median (28.0 years) is robust against right-skewness and preserves sample size. |
| **embarked** / **embark_town** | 2 | 0.22% | <5% missing | Drop Rows | Only 2 rows missing (0.22%); dropping retains 99.78% of data without imputation bias. |

---

## 4. Distinction Between EDA Cleaning and Modeling Preprocessing

- **EDA Cleaning**: Applied directly on a copy of the raw DataFrame (`df_cleaned`). Missing rows for `embarked` are dropped, `deck` is dropped, and `age` missing values are median-imputed directly into `df_cleaned["age"]`. All subsequent EDA summaries, distributions, and correlation matrices operate on `df_cleaned`.
  ```python
  df_cleaned = df.dropna(subset=["embarked"]).copy()
  df_cleaned["age"] = df_cleaned["age"].fillna(df_cleaned["age"].median())
  df_cleaned = df_cleaned.drop(columns=["deck"])
  ```
- **Modeling Train-Only Preprocessing**: To eliminate data leakage, feature preprocessing for modeling is **fitted ONLY on the training fold (`X_train`)** after performing an 80/20 stratified split. Preprocessing relies on Scikit-Learn `ColumnTransformer` inside a `Pipeline`.

---

## 5. Age and Fare Univariate EDA

### Age Feature EDA & Outlier Analysis
- **Summary**: Min = 0.42 years, Max = 80.0 years, Median = 28.0 years.
- **Interquartile Range (IQR)**:
  - $Q1 = 20.00$ years
  - $Q3 = 38.00$ years
  - $\text{IQR} = 18.00$ years
  - $\text{Lower Bound} = Q1 - 1.5 \times \text{IQR} = -7.00$ years
  - $\text{Upper Bound} = Q3 + 1.5 \times \text{IQR} = 65.00$ years
- **IQR Outlier Count**: Exactly **8 passengers** exceed 65.0 years (e.g., 70, 71, 74, 80 years).

### Fare Feature EDA & Skewness Determination
- **Central Tendency Metrics**:
  - $\text{Mean} = \$32.10$
  - $\text{Median} = \$14.45$
  - $\text{Mode} = \$8.05$
  - $Q1 = \$7.91$, $Q3 = \$31.00$, $\text{IQR} = \$23.09$
  - $\text{Lower Bound} = -\$26.72$, $\text{Upper Bound} = \$65.63$
- **IQR Outlier Count**: Exactly **114 passengers** exceed $\$65.63$.
- **Skewness Conclusion**: Comparing central tendency metrics yields $\text{Mean (\$32.10)} > \text{Median (\$14.45)} > \text{Mode (\$8.05)}$. Because the mean is substantially higher than the median and mode, ticket fare is strongly **right-skewed**.

---

## 6. Bivariate Survival Analysis
Calculated via explicit pandas boolean masking expressions:
- **Survival by Sex**:
  - Female Survival Rate: **74.04%**
  - Male Survival Rate: **18.89%**
- **Survival by Passenger Class**:
  - 1st Class Survival Rate: **62.62%**
  - 2nd Class Survival Rate: **47.28%**
  - 3rd Class Survival Rate: **24.24%**
- **Survival by Sex + Pclass (Boolean Masking)**:
  - 1st Class Females (`(df["sex"] == "female") & (df["pclass"] == 1)`): **96.74%**
  - 3rd Class Males (`(df["sex"] == "male") & (df["pclass"] == 3)`): **13.54%**
  - High Fare OR 1st Class (`(df["fare"] > 100) | (df["pclass"] == 1)`): **64.71%**

---

## 7. Exact Six-Column Correlation Matrix
Restricted strictly to the required 6 numerical columns: `["survived", "pclass", "age", "sibsp", "parch", "fare"]`.

| Feature 1 | Feature 2 | Correlation ($r$) | Absolute $|r|$ | Direction | Programmatic Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `pclass` | `fare` | **-0.5482** | 0.5482 | Negative | Higher numerical class (3rd class) correlates strongly with lower fares. |
| `sibsp` | `parch` | **+0.4145** | 0.4145 | Positive | Number of siblings/spouses positively correlates with parents/children (family travel). |

---

## 8. Multivariate Data Story & Visualizations
Saved under `analytics/outputs/charts/`:
1. `age_distribution.png`: Unimodal distribution centered at 28 years with 8 upper outliers (>65 yrs).
2. `fare_distribution.png`: Extreme right-skewness with a long tail extending to $512.33 and 114 upper outliers (> $65.63).
3. `bivariate_survival.png`: Survival advantage of females (74.04% vs 18.89%) and 1st class passengers (62.62% vs 24.24%).
4. `correlation_heatmap.png`: Heatmap confirming pclass-fare ($r=-0.5482$) as the dominant relationship.
5. `decision_tree.png`: Tree structure (depth=4) with split rules, sample counts, and class labels (`["Not Survived", "Survived"]`).
6. `roc_curves.png`: Compares ROC curves across models; Logistic Regression (AUC = 0.8437), Baseline RF (AUC = 0.8271), Decision Tree (AUC = 0.8152).
7. `residual_plot.png`: Predicted fare vs residuals scatter plot demonstrating heteroscedasticity.

---

## 9. EDA-Only Standardization Sanity Check
Z-score normalization check on `df_cleaned`:
- `age`: Before (Mean=29.65, Std=13.00) $\rightarrow$ After (Mean=0.0000, Std=1.0000)
- `fare`: Before (Mean=32.10, Std=49.70) $\rightarrow$ After (Mean=0.0000, Std=1.0000)
*Note*: This standardized data is used purely for EDA verification and **MUST NOT** feed the modeling pipeline.

---

## 10. Predictive Modeling Pipeline & Preprocessing

### Stratified Train/Test Split
- 80% Training Set (712 rows) / 20% Test Set (179 rows) stratified on target $y = \text{survived}$ (`random_state=42`).
- Train Class Balance: 38.34% Survived / 61.66% Not Survived.
- Test Class Balance: 38.55% Survived / 61.45% Not Survived.

### Train-Only Preprocessing
- Preprocessing uses `ColumnTransformer` inside a `Pipeline`.
- Numeric features (`age`, `fare`, `sibsp`, `parch`, `pclass`): `SimpleImputer(strategy='median')` + `StandardScaler()`.
- Categorical features (`sex`, `embarked`): `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`.
- Fitted **strictly on training fold (`X_train`)** to eliminate data leakage.

---

## 11. Classifier Performance Evaluation

| Model | Model Type | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | Baseline | 0.8045 | 0.7931 | 0.6667 | 0.7244 | **0.8437** | Verified |
| **Decision Tree** | Baseline | 0.7877 | **0.8605** | 0.5362 | 0.6607 | 0.8152 | Verified |
| **Baseline Random Forest** | Baseline | **0.8156** | 0.8000 | **0.6957** | **0.7442** | 0.8271 | **Best Baseline** |
| **Tuned Random Forest** | Tuned | **0.8156** | 0.8163 | 0.5797 | 0.6780 | 0.8231 | Verified |

---

## 12. Class Imbalance Handling Experiment

| Strategy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **Variant A (Baseline)** | **0.8000** | 0.6957 | **0.7442** |
| **Variant B (class_weight="balanced")** | 0.7742 | 0.6957 | 0.7328 |
| **Variant C (SMOTE Train-Only)** | 0.7538 | **0.7101** | 0.7313 |

- **SMOTE Training-Only Behavior**: SMOTE oversampling is applied **strictly on `X_train_prep`**, ensuring synthetic samples never enter the test fold.

---

## 13. Hyperparameter Tuning & Random Forest OOB Score
- **GridSearchCV**: 5-fold CV over `n_estimators: [50, 100, 200]`, `max_depth: [None, 5, 10]`, `max_features: ['sqrt', 'log2']`.
- **Best Parameters**: `{'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 100}`
- **Best Cross-Validation F1 Score**: `0.7458`
- **Tuned Random Forest OOB Score**: `0.8202` (configured with `oob_score=True`).

---

## 14. Fare Regression Side-Task & Residual Analysis
Multivariate linear regression predicting ticket `fare` from explicit predictors (`age`, `sibsp`, `parch`, `pclass`, `survived`, `sex`, `embarked`):
- **MAE**: $20.8977$
- **MSE**: $932.2520$
- **RMSE**: $30.5328$
- **$R^2$**: $0.3975$
- **Adjusted $R^2$**: $0.3617$ (calculated using $n=179$ test observations and $p=10$ preprocessed predictor features).
- **Residual Heteroscedasticity Analysis**: **YES**. Residual plot shows a funnel-shaped error distribution where residual variance expands significantly for higher predicted fares (> $100).

---

## 15. Final Classifier Recommendation
We recommend the **Baseline Random Forest Classifier** for deployment based on held-out test performance. On the held-out test set, Baseline Random Forest achieved an F1 score of 0.7442, accuracy of 0.8156, precision of 0.8000, recall of 0.6957, and ROC-AUC of 0.8271. In comparison, Logistic Regression achieved an F1 score of 0.7244 (AUC 0.8437), Decision Tree achieved an F1 score of 0.6607, and Tuned Random Forest achieved an F1 score of 0.6780 (accuracy 0.8156, precision 0.8163, recall 0.5797, AUC 0.8231). Although 5-fold GridSearchCV produced a best cross-validation F1 score of 0.7458 and an OOB score of 0.8202 for the Tuned Random Forest, the Baseline Random Forest demonstrated superior test set F1 generalization (0.7442 vs 0.6780).

---

## 16. Complete Pipeline Export & Reload Verification
- The complete fitted pipeline (preprocessing + selected classifier) is exported to `analytics/best_model_pipeline.joblib`:
  ```python
  full_pipeline = Pipeline([
      ('preprocessor', preprocessor),
      ('classifier', classifiers["Baseline Random Forest"])
  ])
  joblib.dump(full_pipeline, "best_model_pipeline.joblib")
  ```
- **Reload Verification**:
  ```python
  loaded_pipeline = joblib.load("best_model_pipeline.joblib")
  raw_input = pd.DataFrame([{
      "pclass": 1, "sex": "female", "age": 29.0,
      "sibsp": 0, "parch": 0, "fare": 211.3375, "embarked": "S"
  }])
  prediction = loaded_pipeline.predict(raw_input) # Outputs: [1] (Survived)
  probabilities = loaded_pipeline.predict_proba(raw_input) # Outputs: [[0.02, 0.98]]
  ```

---

## 17. Execution Instructions

To execute the complete analytics module and regenerate all notebook outputs:

```bash
# 1. Run EDA notebook to generate charts and titanic.csv
python -m jupyter nbconvert --to notebook --execute analytics/01_eda.ipynb --inplace

# 2. Run Modeling notebook to generate metrics CSVs and best_model_pipeline.joblib
python -m jupyter nbconvert --to notebook --execute analytics/02_modeling.ipynb --inplace
```

---

## 18. Dependency Requirements

Module 2 requires the following dependencies (specified in root `requirements.txt`):
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `imbalanced-learn>=0.10.0`
- `joblib>=1.2.0`
- `jupyter>=1.0.0`
