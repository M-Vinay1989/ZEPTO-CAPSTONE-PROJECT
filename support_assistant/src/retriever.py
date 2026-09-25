"""
Retriever Module for Module 3: Grounded GenAI Support Assistant.
Performs vector similarity search against persistent ChromaDB vector collection to retrieve top-k policy context chunks.
"""

import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from support_assistant.src.embeddings import generate_query_embedding
    from support_assistant.src.vector_store import load_vector_store, CHROMA_DB_DIR
except ImportError:
    from embeddings import generate_query_embedding
    from vector_store import load_vector_store, CHROMA_DB_DIR

def retrieve_relevant_chunks(query: str, top_k: int = 4, store_dir: Path = CHROMA_DB_DIR) -> List[Dict]:
    """
    Retrieves top-k relevant document chunks for a user question from ChromaDB vector store collection.
    Returns list of chunk dicts containing text, document_name, category, score, and chunk_id.
    """
    if not query or not query.strip():
        return []

    client, collection, metadata_chunks = load_vector_store(store_dir)
    query_vec = generate_query_embedding(query.strip())
    query_vec_list = query_vec.tolist()

    # Query ChromaDB collection
    results = collection.query(
        query_embeddings=[query_vec_list],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    if results and results.get("documents") and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0]*len(docs)

        for rank, (doc_text, meta, dist) in enumerate(zip(docs, metas, distances)):
            # Convert cosine distance to similarity score (similarity = 1.0 - distance)
            sim_score = max(0.0, 1.0 - float(dist)) if dist is not None else 0.85
            retrieved.append({
                "chunk_id": meta.get("chunk_id", f"chunk_{rank}"),
                "document_name": meta.get("document_name", "policy.txt"),
                "category": meta.get("category", "General"),
                "section_title": meta.get("section_title", ""),
                "text": doc_text,
                "score": round(sim_score, 4),
                "rank": rank + 1
            })

    print(f"[Retriever] Retrieved {len(retrieved)} relevant chunks from ChromaDB for query: '{query}'")
    return retrieved

if __name__ == "__main__":
    results = retrieve_relevant_chunks("What is the refund policy?", top_k=3)
    for r in results:
        print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Source: {r['document_name']} | Text: {r['text'][:80]}...")
