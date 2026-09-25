"""
Unit Tests for Module 3: Grounded GenAI Support Assistant (Rubric Compliant).
Verifies:
1. Knowledge Base: at least 8 support documents available
2. Vector DB: ChromaDB persistent collection creation, loading, and querying
3. RAG Retrieval: relevant chunks retrieved from ChromaDB
4. LangGraph: executable StateGraph workflow execution
5. MOCK_LLM: deterministic zero-cost response generation without external API credentials
6. FastAPI: POST /ask and GET /health endpoints
7. End-to-End: question -> retrieve -> generate flow
"""

import sys
import pytest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT / "src"))

from document_loader import load_support_documents, DOCUMENTS_DIR
from chunker import chunk_all_documents
from vector_store import build_vector_store, load_vector_store, CHROMA_DB_DIR
from retriever import retrieve_relevant_chunks
from mock_llm import get_mock_llm
from workflow import run_langgraph_workflow, support_assistant_graph
from assistant import ask_assistant, normalize_query_typos, UNSUPPORTED_RESPONSE
from api import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_support_documents_count():
    """Verify at least 8 realistic support policy documents are available and non-empty."""
    docs = load_support_documents(DOCUMENTS_DIR)
    assert len(docs) >= 8, f"Rubric requires at least 8 documents, found {len(docs)}"
    
    doc_names = [d["document_name"] for d in docs]
    required_docs = [
        "refund_policy.txt", "delivery_policy.txt", "cancellation_policy.txt",
        "payment_policy.txt", "account_policy.txt", "returns_policy.txt",
        "zepto_pass_policy.txt", "order_tracking_policy.txt", "privacy_terms_policy.txt"
    ]
    for r_doc in required_docs:
        assert r_doc in doc_names, f"Expected support document {r_doc} missing!"
    
    for doc in docs:
        assert doc["text"].strip(), f"Document {doc['document_name']} is empty!"
        assert "[DISCLAIMER:" in doc["text"], f"Synthetic disclaimer missing in {doc['document_name']}"

def test_chromadb_vector_store_persistence():
    """Verify persistent ChromaDB collection can be created, persisted, and loaded."""
    chroma_dir = CHROMA_DB_DIR
    build_meta = build_vector_store(store_dir=chroma_dir)
    
    assert build_meta["document_count"] >= 8
    assert build_meta["vector_db"] == "ChromaDB"
    assert build_meta["chunk_count"] > 0
    
    load_client, load_coll, chunks = load_vector_store(store_dir=chroma_dir)
    assert load_coll.count() == build_meta["chunk_count"]
    assert len(chunks) == build_meta["chunk_count"]

def test_chromadb_rag_retriever():
    """Verify retrieval queries ChromaDB vector store and returns relevant context chunks."""
    query = "How long does a refund take for UPI?"
    chunks = retrieve_relevant_chunks(query, top_k=3)
    
    assert len(chunks) == 3
    top_chunk = chunks[0]
    assert "score" in top_chunk
    assert top_chunk["score"] > 0.0
    assert "refund_policy.txt" in [c["document_name"] for c in chunks]

def test_langgraph_workflow_execution():
    """Verify LangGraph StateGraph workflow executes correctly with nodes: normalize -> retrieve -> generate."""
    query = "What payment methods are supported on Zepto?"
    response = run_langgraph_workflow(query, top_k=4)
    
    assert response.get("langgraph_executed") is True
    assert response.get("is_grounded") is True
    assert "payment_policy.txt" in response.get("sources", [])
    assert response.get("answer") != UNSUPPORTED_RESPONSE

def test_mock_llm_zero_credentials():
    """Verify MOCK_LLM operates deterministically without requiring paid API keys."""
    mock_llm = get_mock_llm()
    chunks = retrieve_relevant_chunks("What is Zepto Pass membership?", top_k=3)
    resp = mock_llm.generate("What is Zepto Pass membership?", chunks)
    
    assert resp["llm_backend"] == "MOCK_LLM (Deterministic Zero-Cost)"
    assert resp["mock_llm_enabled"] is True
    assert resp["is_grounded"] is True
    assert "zepto_pass_policy.txt" in resp["sources"]

def test_fastapi_health_endpoint():
    """Verify FastAPI GET /health endpoint returns HTTP 200 and system metadata."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["documents_ingested"] >= 8
    assert data["vector_db"] == "ChromaDB"
    assert data["workflow_engine"] == "LangGraph"
    assert data["llm_mode"] == "MOCK_LLM"

def test_fastapi_ask_endpoint():
    """Verify FastAPI POST /ask endpoint returns grounded answer and source metadata."""
    payload = {"question": "How long does a refund take?", "top_k": 4}
    res = client.post("/ask", json=payload)
    assert res.status_code == 200
    data = res.json()
    
    assert data["query"] == "How long does a refund take?"
    assert data["is_grounded"] is True
    assert "refund_policy.txt" in data["sources"]
    assert data["vector_db"] == "ChromaDB"
    assert data["workflow"] == "LangGraph StateGraph"
    assert data["llm_backend"] == "MOCK_LLM (Deterministic Zero-Cost)"

def test_typo_query_normalization():
    """Verify query typo normalization works correctly."""
    norm = normalize_query_typos("What happens if my paymant fails?")
    assert "payment" in norm.lower()

def test_ungrounded_query_refusal():
    """Verify zero-hallucination refusal for out-of-scope queries."""
    res = client.post("/ask", json={"question": "Does Zepto deliver packages to London?"})
    assert res.status_code == 200
    data = res.json()
    assert data["is_grounded"] is False
    assert data["answer"] == UNSUPPORTED_RESPONSE
    assert data["sources"] == []
