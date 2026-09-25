# Module 3: Grounded GenAI Support Assistant

> **Disclaimer & Synthetic Document Notice**: All 9 policy documents located in `documents/` (`refund_policy.txt`, `cancellation_policy.txt`, `payment_policy.txt`, `delivery_policy.txt`, `account_policy.txt`, `returns_policy.txt`, `zepto_pass_policy.txt`, `order_tracking_policy.txt`, `privacy_terms_policy.txt`) are **synthetic demonstration documents** created specifically for the Zepto Data & AI Platform capstone project. They do **NOT** represent official Zepto commercial policies or SLA terms.

## 📌 Objective

Module 3 implements a **Retrieval-Augmented Generation (RAG)** support assistant powered by **ChromaDB**, **LangGraph**, **MOCK_LLM**, and **FastAPI**. It answers customer support policy questions strictly grounded in Zepto's policy documents and enforces a **zero-hallucination refusal rule** when queried about out-of-scope or ungrounded topics, returning source document attribution for verified policy inquiries.

---

## 🏗️ Architecture & LangGraph Workflow Overview

```
User Question (POST /ask or CLI)
       │
       ▼
 [LangGraph StateGraph Workflow]
       │
       ├──► Node 1: normalize_query   (Spelling typo correction against domain vocabulary)
       │
       ├──► Node 2: retrieve_context  (ChromaDB vector search against persistent HNSW index)
       │
       └──► Node 3: generate_answer   (MOCK_LLM deterministic answer synthesis & refusal check)
       │
       ▼
Grounded Answer Response + Source Document Attribution Metadata
```

---

## 📂 Module Structure

```
support_assistant/
├── README.md                          # Module documentation & architecture guide
├── documents/                         # Knowledge Base: 9 realistic synthetic policy documents
│   ├── account_policy.txt             # Account management & deletion policy
│   ├── cancellation_policy.txt        # Order cancellation policy & fees
│   ├── delivery_policy.txt            # Delivery SLA, rider dispatch & restocking fees
│   ├── order_tracking_policy.txt      # Live GPS tracking & delivery partner contact
│   ├── payment_policy.txt             # Payment methods, COD & billing policy
│   ├── privacy_terms_policy.txt       # Privacy terms & personal data protection
│   ├── refund_policy.txt              # Returns, Wallet & bank refund timelines
│   ├── returns_policy.txt             # Doorstep inspection & non-perishable return window
│   └── zepto_pass_policy.txt          # Zepto Pass membership & VIP support policy
├── data/
│   └── chroma_db/                     # Persistent ChromaDB Vector Store
│       ├── chroma.sqlite3             # ChromaDB persistent collection SQLite index
│       └── metadata.json              # Document chunk metadata JSON
├── output/                            # Benchmark evaluation outputs
│   ├── evaluation_results.json        # Detailed benchmark JSON metrics
│   └── evaluation_summary.md          # Formatted Markdown report
├── src/
│   ├── api.py                         # FastAPI web server (POST /ask, GET /health)
│   ├── assistant.py                   # Grounded response generator & refusal guardrails
│   ├── chunker.py                     # Document chunking logic
│   ├── document_loader.py             # File loading utility (9 documents)
│   ├── embeddings.py                  # Embedding generator (SentenceTransformers / TF-IDF)
│   ├── evaluation.py                  # Automated benchmark evaluator (24 test cases)
│   ├── main.py                        # CLI entry point & server launcher (--serve)
│   ├── mock_llm.py                    # Deterministic zero-cost MOCK_LLM generator
│   ├── retriever.py                   # ChromaDB vector store retriever
│   ├── vector_store.py                # ChromaDB persistent collection manager
│   └── workflow.py                    # LangGraph StateGraph executable graph workflow
└── tests/
    └── test_assistant.py              # Pytest test suite (ChromaDB, LangGraph, MOCK_LLM, FastAPI)
```

---

## 🚀 Execution & Usage Workflows

### 1. Build / Rebuild ChromaDB Vector Index Workflow
Rebuild the persistent ChromaDB collection from all 9 support documents:

```bash
python -m support_assistant.src.main --build
```
*Process*: Loads 9 policy text files, chunks text into logical section chunks, computes vector embeddings, and persists the ChromaDB vector store under `support_assistant/data/chroma_db/`.

### 2. Launch FastAPI Web Server
Start the local FastAPI application using Uvicorn:

```bash
python -m support_assistant.src.main --serve
```
Access interactive API docs (Swagger UI) at `http://127.0.0.1:8000/docs`.

#### API Endpoints:
- `GET /health` / `GET /status`: System health, ingested document count (9), and active backends.
- `POST /ask`: Asks a support question via the LangGraph StateGraph workflow.

**Example Request (`POST /ask`)**:
```json
{
  "question": "How long does a refund take for UPI payments?",
  "top_k": 4
}
```

**Example Response**:
```json
{
  "query": "How long does a refund take for UPI payments?",
  "answer": "3. Refund Processing Timelines\n- Zepto Wallet Refunds: Processed instantly upon approval (within 15 minutes).\n- UPI / Credit Card / Debit Card Refunds: Processed within 3 to 5 business days depending on the customer's bank network.\n- Net Banking Refunds: Processed within 5 to 7 business days.",
  "is_grounded": true,
  "sources": [
    "refund_policy.txt"
  ],
  "top_score": 0.3245,
  "vector_db": "ChromaDB",
  "workflow": "LangGraph StateGraph",
  "llm_backend": "MOCK_LLM (Deterministic Zero-Cost)",
  "langgraph_executed": true
}
```

### 3. Query Question via CLI
Query a question directly from the command line through LangGraph:

```bash
python -m support_assistant.src.main --query "What payment options are accepted by Zepto?"
```

### 4. Interactive CLI Mode
Launch an interactive CLI Q&A session:

```bash
python -m support_assistant.src.main
```

---

## 📊 Evaluation & Verification

### Run Automated Benchmark Evaluator
Executes 24 benchmark test cases (grounded questions + ungrounded refusals) and outputs JSON & Markdown summary reports:

```bash
python -m support_assistant.src.evaluation
```

### Run Pytest Unit Test Suite
Runs pytest unit tests verifying document ingestion (8+ docs), ChromaDB persistence, LangGraph workflow execution, MOCK_LLM operation without paid API keys, and FastAPI endpoints:

```bash
pytest support_assistant/tests/test_assistant.py -v
```

---

## 🛠️ Original Rubric Compliance Summary

| Requirement | Rubric Criteria | Module 3 Implementation | Compliance Status |
|---|---|---|:---:|
| **1. Knowledge Base** | At least 8 realistic support documents | 9 realistic support policy text files in `documents/` ingested into vector store | **COMPLIANT** |
| **2. Vector Database** | ChromaDB vector store persistence | Persistent ChromaDB collection (`zepto_support_policies`) stored under `data/chroma_db/` | **COMPLIANT** |
| **3. RAG Retrieval** | Genuine retrieval flow with source attribution | Query embedding search against ChromaDB returning relevant chunks & source attribution | **COMPLIANT** |
| **4. LangGraph** | StateGraph workflow for assistant pipeline | Executable `StateGraph` state machine (`normalize_query -> retrieve_context -> generate_answer`) | **COMPLIANT** |
| **5. MOCK_LLM** | Deterministic mock LLM mode without paid keys | `MockLLM` generator synthesizing answers without requiring external API credentials | **COMPLIANT** |
| **6. FastAPI** | Web API endpoints for support assistant | `POST /ask` and `GET /health` endpoints exposed via FastAPI app (`api.py`) | **COMPLIANT** |
| **7. Tests** | Pytest unit test suite | 9 automated pytest cases testing ChromaDB, LangGraph, MOCK_LLM, and FastAPI | **COMPLIANT** |
| **8. Documentation** | Architecture, run instructions, API examples | Comprehensive README documentation with architecture diagram & API request/response examples | **COMPLIANT** |
