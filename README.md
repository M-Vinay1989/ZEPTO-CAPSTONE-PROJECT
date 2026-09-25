# Zepto Data & AI Platform

A unified end-to-end data engineering, analytics, and grounded GenAI platform built for the B.Tech CSE capstone project.

---

## 1. Overview

The **Zepto Data & AI Platform** is a multi-module engineering and data science project designed to demonstrate full-lifecycle data processing, predictive customer modeling, and grounded artificial intelligence workflows.

The project is structured into three primary modules:
1. **Data Pipeline**: Automated web scraping, data cleaning, relational SQLite database storage, SQL analytical querying, and pandas-based validation.
2. **Titanic Analytics Pipeline & Predictive Modeling**: Dataset profiling via Seaborn Titanic dataset, missing-value strategy, univariate EDA (Age & Fare skewness), bivariate survival rates with boolean masking, exact 6-column correlation heatmap, stratified train/test split, train-only ColumnTransformer preprocessing, classification suite (Logistic Regression, Decision Tree, Random Forest), class imbalance experiment (Baseline, class_weight, SMOTE train-fold ONLY), Random Forest GridSearch with OOB score evaluation, Fare multivariate linear regression side-task with residual heteroscedasticity analysis, and full joblib pipeline export.
3. **Grounded Support Assistant**: A Retrieval-Augmented Generation (RAG) customer support assistant featuring text chunking, vector search, domain typo normalization, and zero-hallucination refusal guardrails.

> **Disclaimer**: The policy documents in `support_assistant/documents/` are **synthetic project data** created solely for demonstration purposes. They do **NOT** represent real Zepto live infrastructure or official Zepto policies.

---

## 2. Project Architecture

The overall system architecture follows a clean, modular pipeline design:

```
========================================================================================
                                1. DATA PIPELINE
========================================================================================
Web Scraping         Cleaning & Transformation         Relational Database        Validation
(Books to Scrape) ──► (Price INR, Rating, Stock) ──► (SQLite: zepto_books.db) ──► (SQL JOIN vs
                                                                                 pandas merge)

========================================================================================
                          2. TITANIC ANALYTICS & MODELING
========================================================================================
Seaborn Titanic       Single Load & Fallback            EDA & Preprocessing        Modeling Suite
(sns.load_dataset) ──► (titanic.csv Snapshot)       ──► (Stratified 80/20 Split, ──► (Logistic Reg, Decision
                                                             ColumnTransformer)        Tree, Random Forest, Fare
                                                                                       Regression, Pipeline Export)

========================================================================================
                           3. GROUNDED SUPPORT ASSISTANT
========================================================================================
Policy Docs          Chunking & Indexing               Vector Retrieval           Grounded Response
(Synthetic .txt) ──► (FAISS / TF-IDF Fallback)  ──► (Top-k Context Search)  ──► (Source Attribution /
                                                                                 Refusal Guardrail)
========================================================================================
```

---

## 3. Repository Structure

```
zepto-data-ai-platform/
├── README.md                            # Root project documentation
├── requirements.txt                      # Project dependency specification
├── data_pipeline/                        # Module 1: Web scraping & SQL database pipeline
│   ├── README.md                        # Module 1 documentation
│   ├── run_pipeline.py                  # End-to-end pipeline execution entry point
│   ├── scraper.py                       # Web scraping module (requests & BeautifulSoup)
│   ├── database.py                      # SQLite schema setup and data insertion
│   ├── queries.py                       # Analytical SQL queries execution
│   ├── validation.py                    # Automated 17-check validation suite
│   ├── data/
│   │   └── zepto_books.db               # SQLite database file
│   └── output/                          # Pipeline export artifacts
│       ├── scraped_books_dataset.csv    # Cleaned dataset export
│       ├── sql_query_results.md         # SQL query output report
│       └── validation_summary.json      # Automated validation results JSON
├── analytics/                           # Module 2: Titanic Analytics Pipeline & Predictive Modeling
│   ├── 01_eda.ipynb                     # Exploratory Data Analysis & Titanic dataset single load
│   ├── 02_modeling.ipynb                # Classification, GridSearch, SMOTE, Fare regression & serialization
│   ├── titanic.csv                      # Raw offline dataset snapshot fallback
│   ├── best_model_pipeline.joblib       # Serialized complete fitted Scikit-Learn Pipeline
│   ├── README.md                        # Module 2 technical documentation
│   └── outputs/                         # Exported charts and metrics CSVs
│       ├── charts/                      # EDA and diagnostic plots (.png)
│       ├── model_metrics.csv            # Classifier baseline evaluation metrics
│       ├── imbalance_comparison.csv     # Class imbalance experiment metrics
│       └── regression_metrics.csv       # Fare regression metrics
└── support_assistant/                   # Module 3: Grounded GenAI Support Assistant
    ├── README.md                        # Module 3 documentation
    ├── src/
    │   ├── main.py                      # Interactive CLI & build entry point
    │   ├── document_loader.py           # Policy document loader
    │   ├── chunker.py                   # Document text chunker
    │   ├── embeddings.py                # Embedding generator (with TF-IDF fallback)
    │   ├── vector_store.py              # FAISS vector store manager
    │   ├── retriever.py                 # Inner-product similarity search retriever
    │   ├── assistant.py                 # Grounded response synthesis & refusal rules
    │   └── evaluation.py                # Benchmark evaluator (24 test cases)
    ├── tests/
    │   └── test_assistant.py            # Pytest suite for support assistant
    ├── documents/                       # Synthetic customer policy files (.txt)
    │   ├── account_policy.txt
    │   ├── cancellation_policy.txt
    │   ├── delivery_policy.txt
    │   ├── payment_policy.txt
    │   └── refund_policy.txt
    ├── data/
    │   └── vector_store/                # FAISS index and metadata cache
    └── output/                          # Benchmark evaluation reports
        ├── evaluation_results.json      # Evaluation benchmark JSON metrics
        └── evaluation_summary.md        # Evaluation markdown report
```

---

## 4. Module 1: Data Pipeline

### Implementation Details
- **Data Source**: Scrapes practice book data from [Books to Scrape](http://books.toscrape.com/).
- **Extraction Tools**: Implemented using `requests` and `BeautifulSoup` to parse HTML structures.
- **Extracted Fields**: `title`, `price_gbp`, `rating` (1-5 integer scale), `in_stock` (boolean), and `category`.
- **Currency Conversion**: Converts British Pounds (GBP) to Indian Rupees (INR) using the fixed conversion rate:
  $$\text{price\_inr} = \text{price\_gbp} \times 105.50$$
- **Relational Storage**: SQLite database (`zepto_books.db`) with normalized schema:
  - `categories` table (`category_id` PRIMARY KEY, `category_name`)
  - `books` table (`book_id` PRIMARY KEY, `title`, `price_gbp`, `price_inr`, `rating`, `in_stock`, `category_id` FOREIGN KEY)
  - Foreign key constraints enabled via `PRAGMA foreign_keys = ON;`.
- **SQL Analysis**: Executes 5 analytical SQL queries covering `WHERE` filtering, `ORDER BY` sorting, `LIMIT`, `DISTINCT`, `BETWEEN` ranges, and `JOIN` operations.
- **Data Validation**: Compares the result of SQLite `INNER JOIN` directly with pandas `pd.merge()` to verify relational consistency.

### Verified Pipeline Results
- **Dataset Size**: 69 books collected across 3 categories (*Travel*, *Mystery*, *Sequential Art*).
- **Validation Suite**: Passed **17/17 automated validation checks** covering data types, value bounds, foreign key enforcement, query correctness, and join equivalence.

### Execution Command
```bash
python -m data_pipeline.run_pipeline
```

---

## 5. Module 2: Titanic Analytics Pipeline & Predictive Modeling

### Implementation Details & Architecture
- **Single Load Rule**: Raw Titanic dataset is loaded **exactly once** via `sns.load_dataset("titanic")` in `analytics/01_eda.ipynb` and immediately exported to `analytics/titanic.csv`.
- **Offline Fallback Isolation**: `analytics/02_modeling.ipynb` reads the committed CSV using `pd.read_csv("titanic.csv")` and **never** calls `sns.load_dataset("titanic")`.
- **Cleaning vs Preprocessing Distinction**:
  - **Raw Offline Dataset**: `analytics/titanic.csv` is the raw dataset snapshot.
  - **EDA Cleaning**: Performed on `df_cleaned` for exploratory visualization (drops `deck`, median-imputes `age` directly into `df_cleaned["age"]`, drops 2 `embarked` missing rows).
  - **Modeling Preprocessing**: Performed on `X_train` **strictly after** an 80/20 stratified split using Scikit-Learn `ColumnTransformer` inside a `Pipeline` to prevent data leakage.
- **Exploratory Data Analysis (EDA)**:
  - Missing-value breakdown: `deck` (77.22% missingness $\rightarrow$ drop column), `age` (19.87% missingness $\rightarrow$ median impute 28.0 yrs), `embarked` (0.22% missingness $\rightarrow$ drop 2 rows).
  - Outlier Analysis: Age IQR = 18.00 yrs (8 upper outliers > 65.0 yrs); Fare IQR = $23.09 (114 upper outliers > $65.63).
  - Skewness Analysis: Ticket fare is strongly right-skewed ($\text{Mean (\$32.10)} > \text{Median (\$14.45)} > \text{Mode (\$8.05)}$).
  - Bivariate Survival Rates: Females 74.04% vs Males 18.89%; 1st Class 62.62% vs 3rd Class 24.24% (1st Class Females: 96.74%, 3rd Class Males: 13.54%).
  - Exact 6-Column Correlation Matrix: `["survived", "pclass", "age", "sibsp", "parch", "fare"]`. Strongest pair: `pclass` & `fare` ($r = -0.5482$).
- **Predictive Classification Modeling**:
  - Stratified 80/20 train/test split (712 train rows / 179 test rows).
  - Classifiers: Logistic Regression, Decision Tree (depth=4 visual tree plot), Baseline Random Forest, and Tuned Random Forest.
  - Class Imbalance Experiment: Variant A (Baseline F1=0.7442), Variant B (`class_weight="balanced"` F1=0.7328), Variant C (SMOTE applied **strictly on `X_train_prep`** F1=0.7313).
  - Hyperparameter Tuning: 5-fold GridSearchCV on Random Forest with `oob_score=True` (Best Params: `max_depth: 10`, `max_features: 'sqrt'`, `n_estimators: 100`; Best CV F1 = 0.7458; OOB score = 0.8202).
- **Fare Multivariate Regression Side-Task**:
  - Linear regression predicting `fare` (MAE = 20.8977, RMSE = 30.5328, $R^2 = 0.3975$, Adjusted $R^2 = 0.3617$ using $n=179, p=10$).
  - Residual plot shows a funnel-shaped error distribution confirming heteroscedasticity.
- **Pipeline Export & Reload Verification**: Complete fitted pipeline (preprocessing + classifier) exported to `analytics/best_model_pipeline.joblib` and verified on un-preprocessed raw input.

### Verified Classification Performance Results (Held-Out Test Set, 179 rows)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 0.8045 | 0.7931 | 0.6667 | 0.7244 | **0.8437** | Verified Baseline |
| **Decision Tree (depth=4)** | 0.7877 | **0.8605** | 0.5362 | 0.6607 | 0.8152 | Verified Baseline |
| **Baseline Random Forest** | **0.8156** | 0.8000 | **0.6957** | **0.7442** | 0.8271 | **Recommended Model** |
| **Tuned Random Forest** | **0.8156** | 0.8163 | 0.5797 | 0.6780 | 0.8231 | Verified Tuned |

### Classifier Recommendation Rationale
We recommend the **Baseline Random Forest Classifier** for primary deployment. While the Decision Tree achieves higher precision (86.05%), its recall drops to 53.62%. Baseline Random Forest delivers the highest overall F1 score (0.7442), superior recall (69.57%), strong precision (80.00%), top accuracy (81.56%), and a robust OOB score of 0.8202. Although GridSearchCV identified optimal cross-validation parameters (`max_depth=10`, best CV F1=0.7458), Baseline Random Forest achieved superior generalization on the held-out test set (F1=0.7442 vs 0.6780).

### Execution Commands
```bash
# Execute EDA notebook
python -m jupyter nbconvert --to notebook --execute analytics/01_eda.ipynb --inplace

# Execute Modeling notebook
python -m jupyter nbconvert --to notebook --execute analytics/02_modeling.ipynb --inplace
```

---

## 6. Module 3: Grounded Support Assistant

### Implementation Details
- **Policy Ingestion**: Loads 5 synthetic Zepto-style customer support policy documents (`refund_policy.txt`, `cancellation_policy.txt`, `payment_policy.txt`, `delivery_policy.txt`, `account_policy.txt`).
- **Document Chunking**: Splits documents into logical section chunks (19 total chunks) preserving document titles, categories, and section headings while omitting header disclaimers.
- **Embedding Backend & Fallback Mechanism**:
  - Designed for 384-dimensional dense embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`).
  - Automatically switches to a built-in **TF-IDF Vectorizer fallback** if PyTorch encounters environment initialization issues (`[WinError 1114]`).
  - Logs a clean single-line notification (`[Embeddings] Embedding backend: TF-IDF fallback`) without stack traces.
- **FAISS Vector Store & Retrieval**: Indexes 19 normalized L2 vectors into a local FAISS index (`index.faiss`) for fast inner-product similarity retrieval (`top_k=4`).
- **Domain Typo Normalization**: Normalizes spelling variations (e.g. `paymant` $\rightarrow$ `payment`, `refnd` $\rightarrow$ `refund`, `cancelltion` $\rightarrow$ `cancellation`) via conservative fuzzy matching (`cutoff=0.82`).
- **Grounded Answer Synthesis & Refusal Guardrail**: Synthesizes answers strictly from retrieved context with explicit source document attribution. Out-of-scope queries trigger a zero-hallucination refusal:
  > *"The available policy documents do not provide enough information to answer this question."*

### Verified Results
- **Pytest Suite**: Passed **11/11 unit tests**.
- **Benchmark Evaluation**: Passed **24/24 evaluation test cases (100.0% accuracy)** across grounded policy questions, typo robustness queries, and out-of-scope refusal questions.

### Execution Commands
```bash
# 1. Build / Rebuild FAISS Vector Index
python -m support_assistant.src.main --build

# 2. Query Single Question via CLI
python -m support_assistant.src.main --query "How long does a refund take?"

# 3. Run Pytest Suite
pytest support_assistant/tests/test_assistant.py -v

# 4. Run Automated Benchmark Evaluation
python -m support_assistant.src.evaluation
```

---

## 7. Technology Stack

### System & Core Utilities
- **Python 3.12**
- **Pytest** (Automated unit testing)
- **Pathlib** & **Pickle** (Object serialization and cross-platform path handling)

### Module 1: Data Pipeline
- `requests` (HTTP web page retrieval)
- `beautifulsoup4` (HTML parsing & element extraction)
- `sqlite3` (Relational database management)
- `pandas` (Tabular data manipulation & join validation)

### Module 2: Titanic Analytics & Modeling
- `pandas` & `numpy` (Numerical processing & dataframes)
- `scikit-learn` (Column transformers, standard scaler, one-hot encoder, logistic regression, decision tree, random forest, linear regression)
- `imbalanced-learn` (SMOTE train-fold oversampling)
- `joblib` (Model pipeline serialization)
- `matplotlib` & `seaborn` (Data visualization & heatmap generation)
- `jupyter` (Notebook analysis execution)

### Module 3: Grounded Support Assistant
- `faiss-cpu` (Vector index storage and similarity search)
- `scikit-learn` (`TfidfVectorizer` fallback embedding backend)
- `difflib` (Fuzzy query typo normalization)
- `sentence-transformers` (Supported primary embedding model interface with automatic fallback)

---

## 8. Installation and Setup

Clone the repository and install all required dependencies using `pip`:

```bash
# 1. Clone the repository
git clone https://github.com/M-Vinay1989/zepto-data-ai-platform.git
cd zepto-data-ai-platform

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 9. Running the Project

Each module can be executed independently using standard commands:

```bash
# Execute Module 1: Data Pipeline
python -m data_pipeline.run_pipeline

# Execute Module 2: Titanic Analytics Pipeline
python -m jupyter nbconvert --to notebook --execute analytics/01_eda.ipynb --inplace
python -m jupyter nbconvert --to notebook --execute analytics/02_modeling.ipynb --inplace

# Execute Module 3: Build Vector Index & Query Assistant
python -m support_assistant.src.main --build
python -m support_assistant.src.main --query "What payment options are accepted by Zepto?"
```

---

## 10. Verification and Verification Commands

Full system verification can be executed across the platform:

```bash
# 1. Module 1: Run Data Pipeline & Automated Validation Suite
python -m data_pipeline.run_pipeline
# Result: 17/17 validation checks PASSED

# 2. Module 2: Run Notebooks & Pipeline Verification
python -m jupyter nbconvert --to notebook --execute analytics/01_eda.ipynb --inplace
python -m jupyter nbconvert --to notebook --execute analytics/02_modeling.ipynb --inplace
python -c "import joblib, pandas as pd; p=joblib.load('analytics/best_model_pipeline.joblib'); raw=pd.DataFrame([{'pclass':1, 'sex':'female', 'age':29.0, 'sibsp':0, 'parch':0, 'fare':211.3375, 'embarked':'S'}]); print('Prediction:', p.predict(raw))"
# Result: Notebooks executed with 0 errors; Pipeline reloads and predicts on raw input

# 3. Module 3: Run Support Assistant Unit Tests
pytest support_assistant/tests/test_assistant.py -v
# Result: 11/11 unit tests PASSED

# 4. Module 3: Run Benchmark Evaluation Suite
python -m support_assistant.src.evaluation
# Result: 24/24 evaluation cases PASSED (100.0% accuracy)
```

---

## 11. Key Design Decisions

- **SQLite Relational Storage (Module 1)**: Serverless relational database with foreign key constraints, normalized into `categories` and `books` tables.
- **Single Dataset Load Rule (Module 2)**: Raw Titanic dataset is loaded **once** in `01_eda.ipynb` and saved to `titanic.csv`. Modeling starts strictly from `titanic.csv`.
- **Train-Only Preprocessing (Module 2)**: Train/test split occurs before preprocessing; `ColumnTransformer` is fitted strictly on `X_train` to eliminate data leakage.
- **Train-Fold SMOTE (Module 2)**: Oversampling occurs exclusively on the training fold (`X_train_prep`), preventing synthetic sample leakage into test data.
- **Random Forest OOB Evaluation (Module 2)**: Out-of-Bag scoring (`oob_score=True`) provides an internal cross-validation benchmark alongside GridSearchCV.
- **Full Pipeline Serialization (Module 2)**: Exporting the entire preprocessing + classifier pipeline ensures raw un-preprocessed inputs can be passed directly to `joblib.load()`.
- **Document Section Chunking (Module 3)**: Splitting text by logical policy headers ensures retrieved chunks maintain contextual coherence.
- **Grounding & Refusal Guardrails (Module 3)**: Similarity thresholds and corpus checks prevent hallucinated responses to out-of-scope queries.

---

## 12. Limitations

- **Synthetic Policy Documents**: Support policies in Module 3 are demonstration documents and do not constitute legal or official Zepto SLAs.
- **External Web Dependency**: Module 1 scraping relies on the availability and structure of the external *Books to Scrape* practice website.
- **TF-IDF Fallback in Certain Environments**: On Windows environments experiencing PyTorch DLL initialization issues, the assistant gracefully uses TF-IDF embeddings.
- **Bounded Support Knowledge**: The assistant's knowledge is strictly limited to the text contained within the 5 provided policy files; out-of-scope queries are explicitly refused.

---

## 13. Git Workflow

The project was developed using a clean Git branching and integration workflow:
- **Feature Branch**: `feature/module-1-data-pipeline`
- **Analytics Revamp Commits**:
  - `49a0ae0`: `feat(analytics): rebuild Titanic EDA and modeling notebooks with train-only preprocessing`
  - `1ef3323`: `docs(analytics): update analytics README and root requirements for Titanic pipeline compliance`
- `master` is synchronized with `origin/master`.
- Working tree verified clean.

---

## 14. Module Documentation

For detailed technical documentation specific to each module, refer to:
- [Module 1: Data Pipeline](data_pipeline/)
- [Module 2: Titanic Analytics Pipeline & Predictive Modeling](analytics/README.md)
- [Module 3: Grounded GenAI Support Assistant](support_assistant/README.md)

---

## 15. Final Project Status

| Component / Module | Verification Metric | Status |
|---|---|:---:|
| **Data Pipeline (Module 1)** | 17 / 17 Validation Checks Passed | **PASS** |
| **Titanic Analytics (Module 2)** | Both notebooks executed with 0 errors; pipeline reloaded & verified | **PASS** |
| **Support Assistant (Module 3)** | 11 / 11 Pytest Unit Tests Passed | **PASS** |
| **Support Evaluation (Module 3)** | 24 / 24 Benchmark Cases Passed (100.0%) | **PASS** |
| **Git Repository Status** | Synchronized with `origin/master`, working tree clean | **PASS** |

---

*Zepto Data & AI Platform Capstone Project — Final Verified Release*
