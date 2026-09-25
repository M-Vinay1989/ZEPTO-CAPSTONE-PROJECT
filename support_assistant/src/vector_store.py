"""
Vector Store Module for Module 3: Grounded GenAI Support Assistant.
Builds, saves, and loads persistent ChromaDB vector store collection and document chunk metadata.
"""

import sys
import os
import types
import shutil
from pathlib import Path
import json

# Block broken onnxruntime C++ DLL loading on Windows if onnxruntime is absent or incompatible
if "onnxruntime" not in sys.modules:
    dummy_ort = types.ModuleType("onnxruntime")
    class DummySession:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, *args, **kwargs):
            pass
    dummy_ort.InferenceSession = DummySession
    dummy_ort.SessionOptions = lambda: None
    sys.modules["onnxruntime"] = dummy_ort

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from support_assistant.src.document_loader import load_documents
    from support_assistant.src.chunker import chunk_all_documents
    from support_assistant.src.embeddings import generate_embeddings
except ImportError:
    from document_loader import load_documents
    from chunker import chunk_all_documents
    from embeddings import generate_embeddings

CHROMA_DB_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "zepto_support_policies"

class PolicyEmbeddingFunction(EmbeddingFunction):
    """
    Custom ChromaDB EmbeddingFunction wrapping Module 3 vector embedding pipeline.
    """
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vecs = generate_embeddings(list(input))
        return vecs.tolist()

def build_vector_store(chunks: list = None, store_dir: Path = CHROMA_DB_DIR) -> dict:
    """
    Builds local persistent ChromaDB vector store collection from document chunks.
    Saves collection and metadata to disk under support_assistant/data/chroma_db/.
    """
    store_dir = Path(store_dir).resolve()
    
    # 1. Load documents and chunk if not provided
    if chunks is None:
        documents = load_documents()
        chunks = chunk_all_documents(documents)

    if not chunks:
        raise ValueError("No chunks generated. Unable to build vector store.")

    # Remove stale sqlite database if present
    if store_dir.exists():
        try:
            shutil.rmtree(store_dir, ignore_errors=True)
        except Exception:
            pass
            
    store_dir.mkdir(parents=True, exist_ok=True)

    # 2. Create persistent ChromaDB client
    client = chromadb.PersistentClient(path=str(store_dir))

    emb_fn = PolicyEmbeddingFunction()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=emb_fn
    )

    ids = [c["chunk_id"] for c in chunks]
    chunk_texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "document_name": c["document_name"],
            "category": c["category"],
            "chunk_id": c["chunk_id"],
            "section_title": c.get("section_title", "")
        }
        for c in chunks
    ]

    # Add to ChromaDB collection
    collection.add(
        ids=ids,
        documents=chunk_texts,
        metadatas=metadatas
    )

    # Save metadata JSON for fast local offline lookup
    meta_path = store_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    count = collection.count()
    print(f"[Vector Store] Built and saved ChromaDB collection '{COLLECTION_NAME}' ({count} vectors) in '{store_dir}'.")
    return {
        "store_dir": str(store_dir),
        "collection_name": COLLECTION_NAME,
        "count": count,
        "metadata_path": str(meta_path),
        "document_count": len(set(c["document_name"] for c in chunks)),
        "vector_db": "ChromaDB",
        "chunk_count": count
    }

def load_vector_store(store_dir: Path = CHROMA_DB_DIR):
    """
    Loads persistent ChromaDB collection and chunk metadata from disk.
    If collection does not exist, builds it dynamically.
    """
    store_dir = Path(store_dir).resolve()
    client = chromadb.PersistentClient(path=str(store_dir))
    emb_fn = PolicyEmbeddingFunction()

    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
        if collection.count() == 0:
            raise ValueError("Collection empty")
    except Exception:
        print(f"[Vector Store] ChromaDB collection missing or empty in '{store_dir}'. Building new vector store...")
        build_info = build_vector_store(store_dir=store_dir)
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

    meta_path = store_dir / "metadata.json"
    chunks = []
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

    return client, collection, chunks

if __name__ == "__main__":
    build_info = build_vector_store()
    print("Vector Store Build Info:", build_info)
