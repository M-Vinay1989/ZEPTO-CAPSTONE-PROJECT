# Module 3: Grounded GenAI Support Assistant

> **Disclaimer & Synthetic Document Notice**: All policy documents located in `documents/` (`refund_policy.txt`, `cancellation_policy.txt`, `payment_policy.txt`, `delivery_policy.txt`, `account_policy.txt`) are **synthetic demonstration documents** created specifically for the Zepto Data & AI Platform capstone project. They do **NOT** represent official Zepto commercial policies or SLA terms.

## 📌 Objective

Module 3 implements a **Retrieval-Augmented Generation (RAG)** support assistant that answers customer support policy questions strictly grounded in Zepto's policy documents. It enforces a **zero-hallucination refusal rule** when queried about out-of-scope or ungrounded topics, ensuring that customer inquiries receive precise, verifiable policy answers with source document attribution.

---

## 🏗️ Architecture & Pipeline Overview

```
Policy Documents (.txt) 
       │
       ▼
[Document Loader]  (Loads raw policy text files from documents/)
       │
       ▼
[Text Chunker]     (Splits documents into section chunks with metadata)
       │
       ▼
[Embeddings]       (Generates L2-normalized vector embeddings via SentenceTransformers / TF-IDF)
       │
       ▼
[FAISS Vector Store] (Stores vectors in index.faiss and chunk metadata in metadata.json)
       │
       ▼
[Retriever]        (Executes inner-product similarity search for top-k context chunks)
       │
       ▼
[Grounded Generator] (Synthesizes answers with strict similarity & keyword grounding guardrails)
       │
       ├──► Grounded Question  ──► Concise Answer + Source Attribution
       └──► Ungrounded Question ──► "The available policy documents do not provide..."
```

---

## 📂 Module Structure

```
support_assistant/
├── README.md                          # Module documentation & architecture guide
├── documents/                         # Synthetic policy documents
│   ├── account_policy.txt             # Account management policy
│   ├── cancellation_policy.txt        # Order cancellation policy
│   ├── delivery_policy.txt            # Delivery SLA and fees policy
│   ├── payment_policy.txt             # Payment methods and billing policy
│   └── refund_policy.txt              # Returns and refund conditions policy
├── data/
│   └── vector_store/                  # Persistent FAISS vector store
│       ├── index.faiss                # FAISS vector index file
│       ├── metadata.json              # Chunk metadata JSON
      └── tfidf_vectorizer.pkl       # Serialized TF-IDF vectorizer fallback
├── output/                            # Benchmark evaluation outputs
│   ├── evaluation_results.json        # Detailed benchmark JSON metrics
│   └── evaluation_summary.md          # Formatted Markdown report
├── src/
│   ├── assistant.py                   # Grounded response generator, typo normalization & refusal guardrails
│   ├── chunker.py                     # Document chunking logic
│   ├── document_loader.py             # File loading utility
│   ├── embeddings.py                  # Embedding generator (SentenceTransformers / TF-IDF fallback)
│   ├── evaluation.py                  # Automated benchmark evaluator (24 test cases)
│   ├── main.py                        # Interactive CLI entry point
│   ├── retriever.py                   # FAISS similarity search retriever
│   └── vector_store.py                # FAISS index builder and manager
└── tests/
    └── test_assistant.py              # Pytest test suite covering unit & retrieval tests
```

---

## 🚀 Execution & Usage Workflows

### 1. Build / Rebuild Vector Index Workflow
Whenever policy documents in `documents/` are modified, added, or updated, rebuild the local FAISS index and vocabulary cache:

```bash
python -m support_assistant.src.main --build
```
*Process*: Loads policy files, chunks text into logical sections, computes L2-normalized L2/Cosine embeddings (or TF-IDF fallback matrix), serializes `index.faiss` and `metadata.json`, and updates vocabulary caches.

### 2. Query Single Question via CLI
To ask a specific question and view answer & source attribution:

```bash
python -m support_assistant.src.main --query "What payment options are accepted by Zepto?"
```

### 3. Launch Interactive CLI Loop
To start an interactive Q&A session:

```bash
python -m support_assistant.src.main
```

---

## 📊 Evaluation & Verification

### Run Automated Benchmark
Executes 24 benchmark test cases (grounded + ungrounded refusals) and outputs metrics to `output/`:

```bash
python -m support_assistant.src.evaluation
```

### Run Unit Test Suite
Runs pytest unit tests covering document ingestion, chunking, vector store indexing, retrieval quality, response generation, refusal guardrails, typo normalization, and output validation:

```bash
pytest support_assistant/tests/test_assistant.py -v
```

---

## 🌐 Real-World Enterprise Deployment Considerations

While this module provides a self-contained, offline-compatible RAG implementation for local demonstration, migrating to enterprise production deployment involves several infrastructure and architectural enhancements:

1. **Embedding Backend**:
   - *Local Fallback*: Uses TF-IDF with bi-grams and domain stemming to ensure execution on environments without PyTorch/GPU native binaries.
   - *Production*: Deploy dense transformer embeddings (e.g. `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, or OpenAI `text-embedding-3-small`) running on GPU-accelerated inference microservices (e.g. Triton Inference Server or TEI).

2. **Vector Database & Search Scaling**:
   - *Local*: Local FAISS flat index stored on disk (`index.faiss`).
   - *Production*: Managed vector databases (e.g. Pinecone, Qdrant, Milvus, or PGVector) with HNSW indexing, multi-tenant isolation, metadata filtering, and real-time document upserts.

3. **LLM Synthesis & Safety Guardrails**:
   - *Local*: Deterministic context summary extraction with strict keyword/similarity threshold guardrails.
   - *Production*: Instruction-tuned GenAI model (e.g. Gemini 1.5 Flash / GPT-4o) with system prompts instructing the model to ground answers strictly in retrieved chunks, combined with automated hallucination guardrails (e.g. NeMo Guardrails or Llama Guard).
