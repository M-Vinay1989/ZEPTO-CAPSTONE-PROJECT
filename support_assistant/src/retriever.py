"""
Retriever Module for Module 3: Grounded GenAI Support Assistant.
Performs vector similarity search against local FAISS vector store to retrieve top-k policy context chunks.
"""

import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embeddings import generate_query_embedding
from vector_store import load_vector_store, VECTOR_STORE_DIR

def retrieve_relevant_chunks(query: str, top_k: int = 3, store_dir: Path = VECTOR_STORE_DIR) -> List[Dict]:
    """
    Retrieves top-k relevant document chunks for a user question from FAISS vector store.
    Returns list of chunk dicts containing text, document_name, category, score, and chunk_id.
    """
    if not query or not query.strip():
        return []

    index, chunks = load_vector_store(store_dir)
    query_vec = generate_query_embedding(query.strip())
    query_matrix = np.array([query_vec], dtype=np.float32)

    # Search top_k in FAISS
    scores, indices = index.search(query_matrix, min(top_k, len(chunks)))

    retrieved = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk_data = dict(chunks[idx])
        chunk_data["score"] = float(score)
        chunk_data["rank"] = rank + 1
        retrieved.append(chunk_data)

    print(f"[Retriever] Retrieved {len(retrieved)} relevant chunks for query: '{query}'")
    return retrieved

if __name__ == "__main__":
    results = retrieve_relevant_chunks("What is the refund policy?", top_k=3)
    for r in results:
        print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Source: {r['document_name']} | Text: {r['text'][:80]}...")
