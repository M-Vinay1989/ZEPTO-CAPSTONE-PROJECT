"""
Embeddings Module for Module 3: Grounded GenAI Support Assistant.
Generates normalized vector embeddings using sentence-transformers (model: all-MiniLM-L6-v2)
with fallback to scikit-learn TfidfVectorizer if SentenceTransformers model loading is unavailable.
"""

import sys
import os
import re
import pickle
from pathlib import Path
from typing import List
import numpy as np

# Support unpickling of custom stemmer across direct and module import paths
if "support_assistant.src.embeddings" not in sys.modules:
    sys.modules["support_assistant.src.embeddings"] = sys.modules[__name__]

# Automatically use deterministic TF-IDF backend if specified or on Windows platforms where PyTorch C++ DLL loading fails
if sys.platform == "win32" and "FORCE_TFIDF" not in os.environ:
    os.environ["FORCE_TFIDF"] = "1"

_MODEL_INSTANCE = None
_TFIDF_VECTORIZER = None
MODEL_NAME = "all-MiniLM-L6-v2"
TFIDF_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "vector_store" / "tfidf_vectorizer.pkl"

def policy_stemmer(text: str) -> List[str]:
    """
    Custom policy domain tokenizer and stemmer for TF-IDF fallback.
    Normalizes domain variations (e.g. timelines/timeline/time -> timeline, processed/processing -> process).
    """
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    tokens = []
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "your", "they", "will", "would",
        "could", "can", "are", "is", "about", "have", "you", "user", "app", "what", "where",
        "when", "how", "does", "zepto", "zeptos", "happen", "happens", "happening", "get", "got", "take", "takes"
    }
    for w in words:
        if w in stop_words:
            continue
        if w in {"timelines", "timeline", "timeframe", "duration", "time", "days", "hours", "minutes", "instant", "instantly"}:
            tokens.append("timeline")
        elif w in {"processed", "processing", "process", "processes"}:
            tokens.append("process")
        elif w in {"refunds", "refunded", "refunding", "refund"}:
            tokens.append("refund")
        elif w in {"cancellation", "cancellations", "cancel", "cancelled", "canceling"}:
            tokens.append("cancel")
        elif w in {"payment", "payments", "pay", "paid"}:
            tokens.append("payment")
        elif w in {"delivery", "deliveries", "deliver", "delivered"}:
            tokens.append("delivery")
        elif w in {"fails", "failed", "failing", "failure", "failures", "fail"}:
            tokens.append("fail")
        elif w in {"account", "accounts"}:
            tokens.append("account")
        elif w in {"cod", "cash"}:
            tokens.append("cod")
        else:
            for suffix in ["ing", "ed", "es", "s"]:
                if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                    w = w[:-len(suffix)]
                    break
            tokens.append(w)
    return tokens

def create_tfidf_vectorizer():
    """Creates a TfidfVectorizer tuned with max_features=384 and policy stemming."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(
        max_features=384,
        tokenizer=policy_stemmer,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=None
    )

def get_embedding_model():
    """Lazy-loads lightweight SentenceTransformer model with graceful TF-IDF fallback."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        if os.environ.get("FORCE_TFIDF", "0") == "1":
            print("[Embeddings] Embedding backend: TF-IDF vectorizer (Deterministic zero-cost)")
            _MODEL_INSTANCE = "fallback"
            return _MODEL_INSTANCE

        try:
            import sentence_transformers
            from sentence_transformers import SentenceTransformer
            print(f"[Embeddings] Loading local SentenceTransformer model '{MODEL_NAME}'...")
            _MODEL_INSTANCE = SentenceTransformer(MODEL_NAME)
        except (Exception, OSError, ImportError, SystemError, BaseException) as err:
            reason = str(err).split('\n')[0]
            print(f"[Embeddings] Embedding backend: TF-IDF fallback (Reason: {reason})")
            _MODEL_INSTANCE = "fallback"
    return _MODEL_INSTANCE

def generate_embeddings(texts: List[str], is_query: bool = False) -> np.ndarray:
    """
    Generates L2-normalized float32 vector embeddings for a list of text strings.
    Guarantees fixed 384 dimensions for ChromaDB vector store compatibility.
    """
    global _TFIDF_VECTORIZER

    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_embedding_model()

    if isinstance(model, str) and model == "fallback":
        if is_query:
            if _TFIDF_VECTORIZER is None and TFIDF_CACHE_PATH.exists():
                try:
                    with open(TFIDF_CACHE_PATH, "rb") as f:
                        _TFIDF_VECTORIZER = pickle.load(f)
                except Exception:
                    _TFIDF_VECTORIZER = None
            
            if _TFIDF_VECTORIZER is not None:
                matrix = _TFIDF_VECTORIZER.transform(texts).toarray().astype(np.float32)
            else:
                _TFIDF_VECTORIZER = create_tfidf_vectorizer()
                matrix = _TFIDF_VECTORIZER.fit_transform(texts).toarray().astype(np.float32)
        else:
            _TFIDF_VECTORIZER = create_tfidf_vectorizer()
            matrix = _TFIDF_VECTORIZER.fit_transform(texts).toarray().astype(np.float32)
            
            TFIDF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TFIDF_CACHE_PATH, "wb") as f:
                pickle.dump(_TFIDF_VECTORIZER, f)

        # Enforce exact 384 dimensions
        if matrix.shape[1] < 384:
            padding = np.zeros((matrix.shape[0], 384 - matrix.shape[1]), dtype=np.float32)
            matrix = np.hstack([matrix, padding])

        embeddings = matrix
    else:
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    # Normalize vectors to unit length for Cosine / Inner Product search
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_embeddings = embeddings / norms

    return normalized_embeddings.astype(np.float32)

def generate_query_embedding(query: str) -> np.ndarray:
    """Generates normalized vector embedding for a single user query string."""
    embeddings = generate_embeddings([query], is_query=True)
    return embeddings[0]

if __name__ == "__main__":
    sample_texts = ["What is the refund policy?", "Delivery takes 10 to 15 minutes."]
    vecs = generate_embeddings(sample_texts)
    print(f"Generated embeddings shape: {vecs.shape}, dtype: {vecs.dtype}")
