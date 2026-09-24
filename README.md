# Zepto Data & AI Platform

A unified end-to-end data engineering, analytics, and grounded GenAI platform built for the B.Tech CSE capstone project.

---

## 1. Overview

The **Zepto Data & AI Platform** is a multi-module engineering and data science project designed to demonstrate full-lifecycle data processing, predictive customer modeling, and grounded artificial intelligence workflows.

The project is structured into three primary modules:
1. **Data Pipeline**: Automated web scraping, data cleaning, relational SQLite database storage, SQL analytical querying, and pandas-based validation.
2. **Customer Analytics and Predictive Modeling**: Data profiling, duplicate handling, exploratory data analysis (EDA), feature engineering, and customer spending regression models.
3. **Grounded Support Assistant**: A Retrieval-Augmented Generation (RAG) customer support assistant featuring text chunking, vector search, domain typo normalization, and zero-hallucination refusal guardrails.

> **Disclaimer**: The customer spending dataset (`customer_spending.csv`) and policy documents in `support_assistant/documents/` are **synthetic project data** created solely for demonstration purposes. They do **NOT** represent real Zepto customer transaction records, live infrastructure, or official Zepto policies.

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
                          2. CUSTOMER ANALYTICS & MODELING
========================================================================================
Synthetic CSV        Preprocessing & Cleaning         EDA & Features             Regression Models
(1,020 rows)     ──► (Deduplication, Scaling,   ──► (Correlation, Encoding) ──► (Linear Regression,
                      Train/Test Split)                                          Random Forest)

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
├── analytics/                           # Module 2: Customer analytics & predictive modeling
│   ├── README.md                        # Module 2 documentation
│   ├── notebooks/
│   │   └── analytics_pipeline.ipynb     # Jupyter notebook demonstration
│   ├── src/
│   │   ├── main.py                      # Analytics pipeline entry point
│   │   ├── data_loader.py               # Raw CSV data loader
│   │   ├── preprocessing.py            # Data cleaning & scikit-learn transformers
│   │   ├── analysis.py                 # Exploratory data analysis (EDA)
│   │   ├── modeling.py                 # Linear Regression & Random Forest models
│   │   └── evaluation.py               # Regression metrics evaluation (MAE, RMSE, R²)
│   ├── tests/
│   │   └── test_analytics.py            # Pytest suite for analytics module
│   ├── data/
│   │   └── customer_spending.csv        # Synthetic customer dataset (1,020 rows)
│   └── output/                          # Analytics outputs and visual figures
│       ├── analysis_summary.md          # Data profiling & modeling report
│       ├── model_results.json           # Model evaluation metrics JSON
│       └── figures/                     # EDA and model diagnostic plots
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

## 5. Module 2: Customer Analytics

### Implementation Details
- **Dataset**: Synthetic customer dataset (`customer_spending.csv`) containing 1,020 raw records representing 1,000 unique customer profiles, with 20 duplicate rows included to demonstrate data cleaning.
- **Preprocessing Pipeline**:
  - Deduplication: Removes duplicate customer records.
  - Cleaning: Imputes/filters invalid values.
  - Train/Test Split: 80% training set (800 rows), 20% held-out test set (200 rows).
  - Column Transformations: Numerical standard scaling (`StandardScaler`) and categorical one-hot encoding (`OneHotEncoder`).
- **Exploratory Data Analysis (EDA)**: Computes statistical summaries, feature correlations, and distribution figures saved under `analytics/output/figures/`.
- **Predictive Regression Modeling**: Trains two predictive models to forecast customer annual spending:
  1. **Linear Regression** (Interpretable baseline)
  2. **Random Forest Regressor** (Ensemble non-linear model, `n_estimators=100`, `random_state=42`)
- **Evaluation Metrics**: Evaluated on the held-out test set using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).

### Verified Model Performance Results (Held-Out Test Set)

| Model | MAE (₹) | RMSE (₹) | $R^2$ Score | Status |
|---|:---:|:---:|:---:|:---:|
| **Linear Regression** (Baseline) | 898.39 | 1,199.83 | 0.8596 | Verified |
| **Random Forest Regressor** | **456.92** | **575.60** | **0.9677** | **Best Model** |

> *Note*: These model evaluation results are computed on the synthetic dataset and held-out test split for demonstration purposes and do not represent real Zepto customer behavior.

### Verification Command & Test Results
```bash
pytest analytics/tests/test_analytics.py -v
```
- **Test Result**: Passed **6/6 unit tests**.

---

## 6. Module 3: Grounded Support Assistant

### Implementation Details
- **Policy Ingestion**: Loads 5 synthetic Zepto-style customer support policy documents (`refund_policy.txt`, `cancellation_policy.txt`, `payment_policy.txt`, `delivery_policy.txt`, `account_policy.txt`).
- **Document Chunking**: Splits documents into logical section chunks (19 total chunks) preserving document titles, categories, and section headings while omitting header disclaimers.
- **Embedding Backend & Fallback Mechanism**:
  - The architecture is designed to generate 384-dimensional dense embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`).
  - In the verified Windows Anaconda environment, the PyTorch C++ library encountered a DLL initialization issue (`[WinError 1114]`).
  - The application automatically switches to a built-in **TF-IDF Vectorizer fallback** (configured with unigrams, bi-grams, sublinear TF scaling, unit L2 normalization, and domain term stemming).
  - The TF-IDF fallback logs a clean single-line notification (`[Embeddings] Embedding backend: TF-IDF fallback (Reason: ...)`) without emitting stack traces.
- **FAISS Vector Store & Retrieval**: Indexes 19 normalized L2 vectors into a local FAISS index (`index.faiss`) for fast inner-product similarity retrieval (`top_k=4`).
- **Domain Typo Normalization**: Normalizes spelling variations (e.g. `paymant` $\rightarrow$ `payment`, `refnd` $\rightarrow$ `refund`, `cancelltion` $\rightarrow$ `cancellation`) via conservative fuzzy matching (`cutoff=0.82`, max length delta $\le 2$).
- **Grounded Answer Synthesis & Refusal Guardrail**: Synthesizes answers strictly from retrieved context with explicit source document attribution. Queries that fall below vector similarity thresholds or request out-of-scope policy topics trigger a zero-hallucination refusal:
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

The project relies strictly on established Python data engineering, science, and testing libraries:

### System & Core Utilities
- **Python 3.12**
- **Pytest** (Automated unit testing)
- **Pathlib** & **Pickle** (Object serialization and cross-platform path handling)

### Module 1: Data Pipeline
- `requests` (HTTP web page retrieval)
- `beautifulsoup4` (HTML parsing & element extraction)
- `sqlite3` (Relational database management)
- `pandas` (Tabular data manipulation & join validation)

### Module 2: Customer Analytics
- `pandas` & `numpy` (Numerical processing & dataframes)
- `scikit-learn` (Column transformers, standard scaler, one-hot encoder, linear regression, random forest)
- `matplotlib` & `seaborn` (Data visualization & figure generation)
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

Each module can be executed independently using standard Python module commands:

```bash
# Execute Module 1: Data Pipeline
python -m data_pipeline.run_pipeline

# Execute Module 2: Customer Analytics Pipeline
python -m analytics.src.main

# Execute Module 3: Build Vector Index & Query Assistant
python -m support_assistant.src.main --build
python -m support_assistant.src.main --query "What payment options are accepted by Zepto?"
```

---

## 10. Testing and Verification

Full system verification can be executed across all three modules:

```bash
# 1. Module 1: Run Data Pipeline & Automated Validation Suite
python -m data_pipeline.run_pipeline
# Result: 17/17 validation checks PASSED

# 2. Module 2: Run Customer Analytics Unit Tests
pytest analytics/tests/test_analytics.py -v
# Result: 6/6 unit tests PASSED

# 3. Module 3: Run Support Assistant Unit Tests
pytest support_assistant/tests/test_assistant.py -v
# Result: 11/11 unit tests PASSED

# 4. Module 3: Run Benchmark Evaluation Suite
python -m support_assistant.src.evaluation
# Result: 24/24 evaluation cases PASSED (100.0% accuracy)
```

> **Environment Note**: During testing on Windows Anaconda environments, PyTorch/SentenceTransformers encountered a DLL initialization error (`[WinError 1114]`). The Support Assistant gracefully engaged its TF-IDF fallback, successfully passing all 11 unit tests and 24 benchmark evaluation cases without interruption.

---

## 11. Design Decisions

- **SQLite Relational Storage**: Selected for Module 1 to provide lightweight, serverless relational database functionality with foreign key constraints, ideal for embedded data pipelines.
- **Normalized Schema**: Divided scraped data into `categories` and `books` tables to prevent data redundancy and demonstrate relational SQL modeling.
- **SQL JOIN vs. Pandas Merge Validation**: Validates database integrity by comparing SQL engine JOIN results directly against pandas `pd.merge()` in memory.
- **Linear Regression Baseline**: Used in Module 2 to provide a clear, interpretable linear baseline before evaluating complex ensemble methods.
- **Random Forest Regressor**: Chosen as the primary predictive model for its capacity to capture non-linear feature interactions and customer spending habits without overfitting.
- **Scikit-Learn ColumnTransformer**: Standardized preprocessing pipeline ensuring numerical scaling and categorical one-hot encoding are applied consistently without data leakage.
- **Document Section Chunking**: Splitting text by logical policy section headers ensures retrieved chunks maintain contextual coherence for similarity search.
- **Vector Search Retrieval**: Inner-product similarity search over normalized vectors provides fast context matching for user queries.
- **Grounding & Refusal Guardrails**: Explicit similarity thresholds and corpus domain checks prevent the assistant from hallucinating answers to unsupported topics.
- **Robust TF-IDF Fallback**: Designed an automatic fallback mechanism so the RAG assistant remains functional on systems where deep learning C++ DLL binaries are unavailable.

---

## 12. Limitations

- **Synthetic Customer Data**: The customer analytics dataset is synthetic and does not reflect actual commercial transaction volumes or real customer behavior at Zepto.
- **Synthetic Policy Documents**: Support policies are demonstration documents and do not constitute legal or official Zepto SLAs.
- **External Web Dependency**: Module 1 scraping relies on the availability and structure of the external *Books to Scrape* practice website.
- **TF-IDF Fallback in Certain Environments**: On environments experiencing PyTorch C++ DLL loading errors, the assistant uses TF-IDF vector embeddings rather than dense transformer embeddings.
- **Bounded Support Knowledge**: The assistant's knowledge is strictly limited to the text contained within the 5 provided policy files; out-of-scope queries are explicitly refused.

---

## 13. Git Workflow

The project was developed using a clean Git branching and integration workflow:
- **Feature Branch**: `feature/module-1-data-pipeline`
- Development commits were authored and verified on the feature branch.
- The feature branch was merged into `master`.
- **Final Merge Commit**: `95acd50` (`95acd50601b9ee79c90ff04d95efacc9645643ff`).
- `master` is fully synchronized with `origin/master`.
- Working tree verified clean.

---

## 14. Module Documentation

For detailed technical documentation specific to each module, refer to:
- [Module 1: Data Pipeline](data_pipeline/)
- [Module 2: Customer Analytics & Predictive Modeling](analytics/README.md)
- [Module 3: Grounded GenAI Support Assistant](support_assistant/README.md)

---

## 15. Final Project Status

| Component / Module | Verification Metric | Status |
|---|---|:---:|
| **Data Pipeline (Module 1)** | 17 / 17 Validation Checks Passed | **PASS** |
| **Customer Analytics (Module 2)** | 6 / 6 Pytest Unit Tests Passed | **PASS** |
| **Support Assistant (Module 3)** | 11 / 11 Pytest Unit Tests Passed | **PASS** |
| **Support Evaluation (Module 3)** | 24 / 24 Benchmark Cases Passed (100.0%) | **PASS** |
| **Git Repository Status** | Merged into `master` (`95acd50`), synchronized with `origin/master` | **PASS** |

---
*Zepto Data & AI Platform Capstone Project — Final Verified Release*
