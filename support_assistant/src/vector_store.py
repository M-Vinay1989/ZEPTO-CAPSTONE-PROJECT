"""
Vector Store Module for Module 3: Grounded GenAI Support Assistant.
Builds, saves, and loads local FAISS vector index and chunk metadata.
"""

import sys
from pathlib import Path
import json
import numpy as np
import faiss

sys.path.insert(0, str(Path(__file__).resolve().parent))

from document_loader import load_documents
from chunker import chunk_all_documents
from embeddings import generate_embeddings, generate_query_embedding

VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_store"

def build_vector_store(chunks: list = None, store_dir: Path = VECTOR_STORE_DIR) -> dict:
    """
    Builds local FAISS vector store from document chunks and saves index and metadata to disk.
    """
    store_dir = Path(store_dir).resolve()
    store_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load documents and chunk if not provided
    if chunks is None:
        documents = load_documents()
        chunks = chunk_all_documents(documents)

    if not chunks:
        raise ValueError("No chunks generated. Unable to build vector store.")

    # 2. Extract chunk texts and generate normalized embeddings
    chunk_texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)

    # 3. Create FAISS Index (IndexFlatIP for cosine similarity on normalized vectors)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # 4. Save FAISS index to file
    index_path = store_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    # 5. Save chunks metadata to JSON
    meta_path = store_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"[Vector Store] Built and saved FAISS index ({index.ntotal} vectors) to '{store_dir}'.")
    return {
        "index_path": str(index_path),
        "metadata_path": str(meta_path),
        "count": index.ntotal
    }

def load_vector_store(store_dir: Path = VECTOR_STORE_DIR):
    """
    Loads FAISS index and chunk metadata from disk. If not found, builds them dynamically.
    """
    store_dir = Path(store_dir).resolve()
    index_path = store_dir / "index.faiss"
    meta_path = store_dir / "metadata.json"

    if not index_path.exists() or not meta_path.exists():
        print(f"[Vector Store] Index or metadata missing in '{store_dir}'. Building new vector store...")
        build_vector_store(store_dir)

    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks

if __name__ == "__main__":
    build_info = build_vector_store()
    print("Vector Store Build Info:", build_info)
