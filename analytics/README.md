# Module 2 — Customer Analytics & Predictive Modeling

This directory contains the complete implementation of **Module 2 — Customer Analytics and Predictive Modeling** for the Zepto Data & AI Platform capstone project.

---

## 1. Overview & Dataset Source
The analytics pipeline utilizes the Seaborn **Titanic** dataset to perform end-to-end exploratory data analysis (EDA), data cleaning, feature engineering, classification modeling, class imbalance experimentation, hyperparameter tuning, regression side-task modeling, and complete model pipeline serialization.

---

## 2. Dataset Loading Architecture & Offline Fallback
- **Single Load Rule**: The raw dataset is loaded **exactly once** across the entire module using Seaborn's dataset loader in `analytics/01_eda.ipynb`:
  ```python
  df = sns.load_dataset("titanic")
  df.to_csv("titanic.csv", index=False)
  ```
- **Offline Fallback**: `analytics/titanic.csv` is committed as the mandatory offline dataset source.
- **Modeling Isolation**: `analytics/02_modeling.ipynb` starts strictly from `pd.read_csv("titanic.csv")` and **never** calls `sns.load_dataset("titanic")`.

---

## 3. Missing-Value Breakdown & Cleaning Strategy

| Column | Measured Missing Count | Missing % | Category Rule | Chosen Strategy | Justification |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **deck** | 688 | 77.22% | >30% missing | Drop Column | Over 77% of data is missing; imputing would introduce severe synthetic noise without predictive benefit. |
| **age** | 177 | 19.87% | 5%–30% missing | Median Imputation | 19.87% missing; median (28.0 years) is robust against right-skewness and preserves sample size. |
| **embarked** / **embark_town** | 2 | 0.22% | <5% missing | Drop Rows | Only 2 rows missing (0.22%); dropping retains 99.78% of data without imputation bias. |

---

## 4. Feature Profiling & Exploratory Findings

### Age Feature EDA & Outlier Bounds
- **Summary**: Min = 0.42 years, Max = 80.0 years, Median = 28.0 years.
- **Interquartile Range (IQR)**:
  - $Q1 = 20.00$ years
  - $Q3 = 38.00$ years
  - $\text{IQR} = 18.00$ years
  - $\text{Lower Bound} = Q1 - 1.5 \times \text{IQR} = -7.00$ years
  - $\text{Upper Bound} = Q3 + 1.5 \times \text{IQR} = 65.00$ years
- **Outlier Count**: Exactly **8 passengers** exceed 65.0 years (e.g. 70, 71, 74, 80).

### Fare Feature EDA & Skewness Analysis
- **Summary Metrics**:
  - $\text{Mean} = \$32.10$
  - $\text{Median} = \$14.45$
  - $\text{Mode} = \$8.05$
  - $Q1 = \$7.91$, $Q3 = \$31.00$, $\text{IQR} = \$23.09$
  - $\text{Lower Bound} = -\$26.72$, $\text{Upper Bound} = \$65.63$
- **Outlier Count**: Exactly **114 passengers** exceed $\$65.63$.
- **Skewness Determination**: Ticket fare is strongly **right-skewed** ($\text{Mean (\$32.10)} > \text{Median (\$14.45)} > \text{Mode (\$8.05)}$). Most passengers paid steerage rates ($\$8.05$), while a small number of luxury passengers paid up to $\$512.33$, pulling the mean upward.

### Bivariate Survival Analysis
- **By Sex**:
  - Female Survival Rate: **74.04%**
  - Male Survival Rate: **18.89%**
- **By Pclass**:
  - 1st Class Survival Rate: **62.62%**
  - 2nd Class Survival Rate: **47.28%**
  - 3rd Class Survival Rate: **24.24%**
- **By Sex + Pclass (Boolean Masking)**:
  - 1st Class Females (`(sex == 'female') & (pclass == 1)`): **96.74%**
  - 3rd Class Males (`(sex == 'male') & (pclass == 3)`): **13.54%**

---

## 5. Correlation Heatmap & Top 2 Pairwise Correlations
Restricted strictly to the required 6 numerical columns: `["survived", "pclass", "age", "sibsp", "parch", "fare"]`.

| Feature 1 | Feature 2 | Correlation ($r$) | Absolute $|r|$ | Direction | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `pclass` | `fare` | **-0.5482** | 0.5482 | Negative | Higher numerical class (3rd class) correlates strongly with lower fares. |
| `sibsp` | `parch` | **+0.4145** | 0.4145 | Positive | Number of siblings/spouses positively correlates with parents/children (family travel). |

---

## 6. Visualization & Chart Interpretations
All generated charts are saved under `analytics/outputs/charts/`:
1. `age_distribution.png`: Age follows a unimodal, slightly right-skewed distribution centered at 28 years with 8 upper outliers (>65 yrs).
2. `fare_distribution.png`: Fare exhibits extreme right-skewness with a long tail extending to $512.33 and 114 upper outliers (> $65.63).
3. `bivariate_survival.png`: Highlights the strong survival advantage of females (74.04% vs 18.89%) and 1st class passengers (62.62% vs 24.24%).
4. `correlation_heatmap.png`: Renders the 6-column correlation matrix, confirming pclass-fare ($r=-0.5482$) as the dominant relationship.
5. `decision_tree.png`: Displays tree topology up to max_depth=4 with node split criteria, sample counts, and class values.
6. `roc_curves.png`: Compares ROC curves across all three models; Logistic Regression leads in AUC (0.8437), followed by Random Forest (0.8271) and Decision Tree (0.8152).
7. `residual_plot.png`: Displays predicted fare vs residuals, proving clear heteroscedasticity (funnel expansion for predicted fares > $100).

---

## 7. Standardization Sanity Check
Z-score normalization check performed on `age` and `fare`:
- `age`: Before (Mean=29.65, Std=13.00) $\rightarrow$ After (Mean=0.0000, Std=1.0000)
- `fare`: Before (Mean=32.10, Std=49.70) $\rightarrow$ After (Mean=0.0000, Std=1.0000)
*Note*: This standardized data is used purely for EDA verification and does not leak into the modeling pipeline.

---

## 8. Predictive Modeling & Preprocessing Architecture

### Stratified Train/Test Split
- Split: **80% Training Set (712 rows)** / **20% Test Set (179 rows)** stratified by $y = \text{survived}$.
- Train target balance: 38.34% Survived / 61.66% Not Survived.
- Test target balance: 38.55% Survived / 61.45% Not Survived.

### Train-Only Preprocessing
- Preprocessing uses `ColumnTransformer` inside a `Pipeline`.
- Numeric features (`age`, `fare`, `sibsp`, `parch`, `pclass`): `SimpleImputer(strategy='median')` + `StandardScaler()`.
- Categorical features (`sex`, `embarked`): `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`.
- Preprocessing is fitted **strictly on training data** (`X_train`) to prevent data leakage.

---

## 9. Classifier Performance Evaluation

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8045 | 0.7931 | 0.6667 | 0.7244 | **0.8437** |
| **Decision Tree** | 0.7877 | **0.8605** | 0.5362 | 0.6607 | 0.8152 |
| **Random Forest (Tuned)** | **0.8156** | 0.8000 | **0.6957** | **0.7442** | 0.8271 |

---

## 10. Class Imbalance Experiment Results

| Strategy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **Variant A (Baseline)** | **0.8000** | 0.6957 | **0.7442** |
| **Variant B (class_weight="balanced")** | 0.7742 | 0.6957 | 0.7328 |
| **Variant C (SMOTE Train-Only)** | 0.7538 | **0.7101** | 0.7313 |

- **SMOTE Rule**: SMOTE oversampling is applied **strictly on the training fold** (`X_train_prep`), ensuring zero synthetic samples enter the test set.

---

## 11. Hyperparameter Tuning & Random Forest OOB Score
- **GridSearchCV**: 5-fold cross-validation over `n_estimators: [50, 100, 200]`, `max_depth: [None, 5, 10]`, `max_features: ['sqrt', 'log2']`.
- **Best Parameters**: `{'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 100}`
- **Best CV F1 Score**: `0.7458`
- **Random Forest OOB Score**: `0.8202` (configured with `oob_score=True`).

---

## 12. Fare Regression Side-Task & Residual Analysis
Multivariate linear regression predicting ticket `fare`:
- **MAE**: $20.8977$
- **MSE**: $932.2520$
- **RMSE**: $30.5328$
- **$R^2$**: $0.3975$
- **Adjusted $R^2$**: $0.3617$
- **Evidence of Heteroscedasticity**: **YES**. Residual plot shows a funnel-shaped error distribution where residual variance expands significantly for higher predicted fares (> $100).

---

## 13. Final Classifier Recommendation
We recommend the **Random Forest Classifier (Tuned)** for deployment. It achieves the highest overall accuracy (**81.56%**) and F1 score (**0.7442**), combined with strong precision (**80.00%**) and high AUC (**0.8271**). In emergency survival prediction, balancing precision and recall while maintaining high ROC-AUC provides the most reliable performance.

---

## 14. Pipeline Export & Reload Verification
- The complete fitted pipeline (preprocessing + tuned classifier) is exported to `analytics/best_model_pipeline.joblib`:
  ```python
  full_pipeline = Pipeline([
      ('preprocessor', preprocessor),
      ('classifier', best_rf)
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

## 15. Execution Guide
To execute the complete analytics module and regenerate all artifacts from scratch:

```bash
# 1. Run EDA notebook to generate charts and titanic.csv
python -m jupyter nbconvert --to notebook --execute analytics/01_eda.ipynb --inplace

# 2. Run Modeling notebook to generate metrics CSVs and best_model_pipeline.joblib
python -m jupyter nbconvert --to notebook --execute analytics/02_modeling.ipynb --inplace
```
