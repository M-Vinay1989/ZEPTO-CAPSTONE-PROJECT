"""
Unit Tests for Module 3: Grounded GenAI Support Assistant.
Verifies document loading, chunking, vector store indexing, retrieval accuracy,
grounded response generation, and zero-hallucination refusal behavior for ungrounded queries.
"""

import sys
import pytest
from pathlib import Path

# Add support_assistant/src to path
MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT / "src"))

from document_loader import load_documents, DOCUMENTS_DIR
from chunker import chunk_all_documents, chunk_document
from vector_store import build_vector_store, load_vector_store, VECTOR_STORE_DIR
from retriever import retrieve_relevant_chunks
from embeddings import generate_embeddings, generate_query_embedding
from assistant import ask_assistant, generate_grounded_answer, normalize_query_typos, UNSUPPORTED_RESPONSE
from evaluation import run_evaluation, BENCHMARK_QUESTIONS, OUTPUT_DIR

def test_document_loader():
    """Verify document loader correctly reads all synthetic policy files."""
    docs = load_documents(DOCUMENTS_DIR)
    assert len(docs) == 5, f"Expected 5 policy documents, got {len(docs)}"
    
    doc_names = [d["document_name"] for d in docs]
    assert "refund_policy.txt" in doc_names
    assert "delivery_policy.txt" in doc_names
    assert "cancellation_policy.txt" in doc_names
    assert "payment_policy.txt" in doc_names
    assert "account_policy.txt" in doc_names
    
    for doc in docs:
        assert doc["text"].strip(), f"Document {doc['document_name']} is empty!"
        assert "[DISCLAIMER:" in doc["text"], f"Synthetic disclaimer missing in {doc['document_name']}"

def test_chunker():
    """Verify document chunking retains metadata and skips disclaimer chunks."""
    docs = load_documents(DOCUMENTS_DIR)
    chunks = chunk_all_documents(docs)
    
    assert len(chunks) >= 15, f"Expected at least 15 chunks, got {len(chunks)}"
    
    first_chunk = chunks[0]
    required_keys = {"chunk_id", "document_name", "category", "chunk_index", "text", "source_path"}
    assert required_keys.issubset(first_chunk.keys()), f"Missing keys in chunk: {required_keys - set(first_chunk.keys())}"
    
    # Ensure no chunk consists solely of the disclaimer block
    for chunk in chunks:
        assert not chunk["text"].startswith("[DISCLAIMER"), f"Disclaimer chunk found in index: {chunk['chunk_id']}"

def test_embedding_backend_and_consistency():
    """Verify embedding generation dimensions, float32 precision, and L2 normalization."""
    texts = ["How long does a refund take?", "Delivery is completed in 10 to 15 minutes."]
    vecs = generate_embeddings(texts)
    
    assert vecs.shape[0] == 2
    assert vecs.shape[1] == 384, f"Expected 384 embedding dimensions, got {vecs.shape[1]}"
    assert vecs.dtype == "float32"
    
    # Check L2 normalization (unit length vectors)
    import numpy as np
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"Embeddings not L2 normalized: {norms}"
    
    q_vec = generate_query_embedding("What payment options exist?")
    assert q_vec.shape == (384,)
    assert q_vec.dtype == "float32"

def test_vector_store_build_and_load():
    """Verify FAISS vector store index build and load operations."""
    store_dir = VECTOR_STORE_DIR
    build_info = build_vector_store(store_dir=store_dir)
    
    assert Path(build_info["index_path"]).exists(), "FAISS index file missing!"
    assert Path(build_info["metadata_path"]).exists(), "Metadata JSON file missing!"
    assert build_info["count"] > 0, "Vector store index is empty!"

    index, chunks = load_vector_store(store_dir)
    assert index.ntotal == build_info["count"]
    assert len(chunks) == build_info["count"]

def test_retriever():
    """Verify top-k similarity retrieval returns relevant chunks."""
    query = "What is the refund policy for damaged items?"
    chunks = retrieve_relevant_chunks(query, top_k=3)
    
    assert len(chunks) == 3, f"Expected 3 retrieved chunks, got {len(chunks)}"
    top_chunk = chunks[0]
    assert "score" in top_chunk
    assert top_chunk["score"] > 0.0, "Similarity score should be positive"
    assert top_chunk["document_name"] == "refund_policy.txt", f"Top match should be refund_policy.txt, got {top_chunk['document_name']}"

def test_retrieval_quality_specific_chunks():
    """Verify specific queries retrieve exact policy section chunks."""
    q1 = "How long does a UPI refund take?"
    chunks1 = retrieve_relevant_chunks(q1, top_k=3)
    top_sources = [c["document_name"] for c in chunks1]
    assert "refund_policy.txt" in top_sources
    
    q2 = "What happens if my order is delayed?"
    chunks2 = retrieve_relevant_chunks(q2, top_k=3)
    top_sources2 = [c["document_name"] for c in chunks2]
    assert "delivery_policy.txt" in top_sources2

def test_typo_query_normalization():
    """Verify query normalization corrects typos and handles grounded vs ungrounded queries."""
    norm1 = normalize_query_typos("What happens if my paymant fails?")
    assert "payment" in norm1.lower()
    
    norm2 = normalize_query_typos("How long does a refnd take?")
    assert "refund" in norm2.lower()

    # Verify typo query gets successfully grounded
    resp = ask_assistant("What happens if my paymant fails?")
    assert resp["is_grounded"] is True
    assert "payment_policy.txt" in resp["sources"]

def test_refund_query_grounded():
    """Verify specific refund duration question 'How long does a refund take?' is grounded."""
    query = "How long does a refund take?"
    response = ask_assistant(query, top_k=4)
    
    assert response["is_grounded"] is True, "Refund query should be marked grounded"
    assert "refund_policy.txt" in response["sources"], f"Expected refund_policy.txt in sources, got: {response['sources']}"
    assert response["answer"] != UNSUPPORTED_RESPONSE

def test_grounded_response():
    """Verify assistant produces grounded response with source attribution for policy questions."""
    query = "What payment methods are supported?"
    response = ask_assistant(query, top_k=4)
    
    assert response["is_grounded"] is True, "Query should be marked grounded"
    assert len(response["sources"]) > 0, "Sources list should not be empty"
    assert "payment_policy.txt" in response["sources"]
    assert response["answer"] != UNSUPPORTED_RESPONSE
    assert "[DISCLAIMER" not in response["answer"]

def test_ungrounded_refusal():
    """Verify assistant strictly refuses out-of-scope questions without hallucinating."""
    query = "Does Zepto provide international delivery?"
    response = ask_assistant(query, top_k=4)
    
    assert response["is_grounded"] is False, "Ungrounded query should not be marked grounded"
    assert response["answer"] == UNSUPPORTED_RESPONSE, f"Expected refusal message, got: {response['answer']}"
    assert response["sources"] == [], "Sources should be empty for ungrounded queries"

def test_evaluation_and_outputs():
    """Verify full evaluation pipeline runs cleanly and creates report artifacts."""
    summary = run_evaluation(OUTPUT_DIR)
    
    assert summary["total_questions"] == len(BENCHMARK_QUESTIONS)
    assert summary["accuracy_percentage"] == 100.0, f"Accuracy dropped below 100%: {summary['accuracy_percentage']}%"
    
    json_path = OUTPUT_DIR / "evaluation_results.json"
    md_path = OUTPUT_DIR / "evaluation_summary.md"
    
    assert json_path.exists(), "evaluation_results.json was not created!"
    assert md_path.exists(), "evaluation_summary.md was not created!"
