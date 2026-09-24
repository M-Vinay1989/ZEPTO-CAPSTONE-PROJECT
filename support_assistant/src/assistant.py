"""
Grounded GenAI Support Assistant Module for Module 3.
Synthesizes concise, grounded customer support answers strictly from retrieved policy context
and enforces zero-hallucination refusal when policy documents lack sufficient information.
"""

import sys
import re
import string
import difflib
from pathlib import Path
from typing import List, Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retriever import retrieve_relevant_chunks, VECTOR_STORE_DIR
from vector_store import load_vector_store

SIMILARITY_THRESHOLD = 0.08  # Minimum vector similarity threshold for policy context relevance

UNSUPPORTED_RESPONSE = "The available policy documents do not provide enough information to answer this question."

STOP_WORDS = {
    "what", "where", "when", "which", "how", "does", "zepto", "zeptos", "policy", "for", "with",
    "have", "about", "that", "this", "from", "your", "they", "will", "would", "could",
    "can", "are", "is", "the", "and", "or", "in", "on", "at", "to", "a", "an", "do", "i", "my", "by",
    "me", "it", "its", "has", "been", "get", "gets", "got", "getting", "take", "takes", "taking", "long", "happens",
    "happen", "happening", "please", "tell", "give", "provides", "provide", "providing", "much", "many",
    "want", "possible", "should", "now", "need", "like", "way", "already", "just", "still", "doing",
    "arrive", "arriving", "arrives", "after", "fast", "quick", "quickly", "speed"
}

_CORPUS_WORDS_CACHE: Set[str] = None

def get_corpus_words(store_dir: Path = VECTOR_STORE_DIR, force_reload: bool = False) -> Set[str]:
    """Retrieves set of all normalized words present across all indexed policy document chunks."""
    global _CORPUS_WORDS_CACHE
    if _CORPUS_WORDS_CACHE is None or force_reload:
        try:
            _, chunks = load_vector_store(store_dir)
            all_text = " ".join(c["text"] for c in chunks).lower()
            _CORPUS_WORDS_CACHE = set(re.findall(r'\b[a-z]{3,}\b', all_text))
        except Exception:
            _CORPUS_WORDS_CACHE = set()
    return _CORPUS_WORDS_CACHE

def normalize_word(word: str) -> str:
    """Strips common English suffixes and maps policy term variations to normalized stems."""
    w = word.lower().strip(string.punctuation)
    if w in {"late", "delayed", "delays", "delay"}:
        return "delay"
    if w in {"quickly", "instant", "instantly", "quick"}:
        return "instant"
    if w in {"failed", "fails", "failing", "failure", "fail"}:
        return "fail"
    if w in {"processed", "processing", "process", "processes"}:
        return "process"
    for suffix in ["'s", "ing", "ed", "es", "s", "ment", "tion", "al"]:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w

def term_matches_words(term: str, word_set: Set[str]) -> bool:
    """Checks if a query term or its stem exists in a set of words."""
    term_lower = term.lower()
    if term_lower in word_set:
        return True
    term_stem = normalize_word(term_lower)
    for w in word_set:
        if term_stem == w or term_stem == normalize_word(w):
            return True
    return False

def normalize_query_typos(query: str, store_dir: Path = VECTOR_STORE_DIR) -> str:
    """
    Normalizes spelling typos in query tokens by matching unknown words against known domain vocabulary.
    Only unknown tokens (not in stop words or domain vocabulary) are checked with conservative cutoff=0.82.
    """
    if not query or not query.strip():
        return query

    domain_vocab = get_corpus_words(store_dir)
    if not domain_vocab:
        return query

    words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
    corrected_query = query
    normalized_pairs = []

    for w in words:
        w_lower = w.lower()
        if w_lower in STOP_WORDS or w_lower in domain_vocab:
            continue

        matches = difflib.get_close_matches(w_lower, list(domain_vocab), n=1, cutoff=0.82)
        if matches:
            best_match = matches[0]
            if abs(len(w_lower) - len(best_match)) <= 2:
                corrected_query = re.sub(r'\b' + re.escape(w) + r'\b', best_match, corrected_query, flags=re.IGNORECASE)
                normalized_pairs.append((w, best_match))

    if normalized_pairs:
        for orig, corr in normalized_pairs:
            print(f"[Assistant] Query normalization: '{orig}' -> '{corr}'")

    return corrected_query

def generate_grounded_answer(query: str, retrieved_chunks: List[Dict], store_dir: Path = VECTOR_STORE_DIR) -> Dict:
    """
    Generates a grounded response based ONLY on retrieved policy context chunks.
    Enforces refusal for ungrounded queries where context similarity is low or policy is absent.
    """
    if not retrieved_chunks:
        return {
            "query": query,
            "answer": UNSUPPORTED_RESPONSE,
            "is_grounded": False,
            "sources": [],
            "top_score": 0.0
        }

    top_score = retrieved_chunks[0].get("score", 0.0)

    # 1. Similarity Threshold Guardrail
    if top_score < SIMILARITY_THRESHOLD:
        return {
            "query": query,
            "answer": UNSUPPORTED_RESPONSE,
            "is_grounded": False,
            "sources": [],
            "top_score": top_score
        }

    # Normalize query terms
    clean_query = re.sub(r"'s\b", "", query.lower())
    clean_query = clean_query.translate(str.maketrans("", "", string.punctuation))
    query_terms = [word for word in clean_query.split() if len(word) >= 3 and word not in STOP_WORDS]

    # 2. Global Corpus Topic Guardrail: Check if query contains ungrounded topics absent from entire policy base
    corpus_words = get_corpus_words(store_dir, force_reload=True)
    if corpus_words and query_terms:
        out_of_corpus_terms = [term for term in query_terms if not term_matches_words(term, corpus_words)]
        if out_of_corpus_terms:
            return {
                "query": query,
                "answer": UNSUPPORTED_RESPONSE,
                "is_grounded": False,
                "sources": [],
                "top_score": top_score
            }

    # 3. Context Grounding Check: Ensure retrieved chunks contain query content
    context_paragraphs = [chunk["text"] for chunk in retrieved_chunks]
    joined_context = "\n\n".join(context_paragraphs).lower()
    context_words = set(re.findall(r'\b[a-z]{3,}\b', joined_context))

    if query_terms:
        missing_context_terms = [term for term in query_terms if not term_matches_words(term, context_words)]
        if len(missing_context_terms) >= max(2, len(query_terms)):
            return {
                "query": query,
                "answer": UNSUPPORTED_RESPONSE,
                "is_grounded": False,
                "sources": [],
                "top_score": top_score
            }

    # Extract unique sources cited
    sources = sorted(list(set(chunk["document_name"] for chunk in retrieved_chunks)))

    # Strip disclaimer lines from summary presentation
    clean_lines = [line for line in "\n\n".join(context_paragraphs).split("\n") if not line.startswith("[DISCLAIMER")]
    grounded_summary = "\n".join(clean_lines).strip()

    return {
        "query": query,
        "answer": grounded_summary,
        "is_grounded": True,
        "sources": sources,
        "top_score": top_score,
        "chunks_used": [c["chunk_id"] for c in retrieved_chunks]
    }

def ask_assistant(query: str, top_k: int = 3, store_dir: Path = VECTOR_STORE_DIR) -> Dict:
    """
    Main assistant pipeline function: performs fuzzy query normalization, retrieves context, and generates grounded answer.
    """
    normalized_query = normalize_query_typos(query, store_dir=store_dir)
    chunks = retrieve_relevant_chunks(normalized_query, top_k=top_k, store_dir=store_dir)
    response = generate_grounded_answer(normalized_query, chunks, store_dir=store_dir)
    response["original_query"] = query
    return response

if __name__ == "__main__":
    q1 = "What happens if my paymant fails?"
    resp1 = ask_assistant(q1)
    print("--- Question 1 (Typo) ---")
    print("Q:", q1)
    print("Answer:\n", resp1["answer"])
    print("Sources:", resp1["sources"])

    q2 = "What is Zepto's policy for international delivery?"
    resp2 = ask_assistant(q2)
    print("\n--- Question 2 (Ungrounded) ---")
    print("Q:", q2)
    print("Answer:\n", resp2["answer"])
    print("Sources:", resp2["sources"])
