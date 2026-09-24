"""
Text Chunker Module for Module 3: Grounded GenAI Support Assistant.
Splits policy documents into semantically coherent policy section chunks with source metadata.
"""

from typing import List, Dict

def chunk_document(doc: Dict[str, str], min_chunk_len: int = 50) -> List[Dict[str, str]]:
    """
    Splits a single document dictionary into section chunks based on section headers and paragraphs.
    """
    doc_name = doc["document_name"]
    category = doc["category"]
    full_text = doc["text"]

    # Split by double newlines (paragraphs / policy numbered sections)
    raw_sections = full_text.split("\n\n")
    chunks = []
    chunk_idx = 1

    for section in raw_sections:
        cleaned_section = section.strip()
        if cleaned_section.startswith("[DISCLAIMER"):
            continue
        if len(cleaned_section) < min_chunk_len and not cleaned_section.startswith("1."):
            continue

        chunk_id = f"{doc_name}_chunk_{chunk_idx}"
        chunks.append({
            "chunk_id": chunk_id,
            "document_name": doc_name,
            "category": category,
            "chunk_index": chunk_idx,
            "text": cleaned_section,
            "source_path": doc.get("source_path", "")
        })
        chunk_idx += 1

    return chunks

def chunk_all_documents(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Chunks a list of policy documents.
    """
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)

    print(f"[Chunker] Created {len(all_chunks)} chunks across {len(documents)} documents.")
    return all_chunks

if __name__ == "__main__":
    from document_loader import load_documents
    docs = load_documents()
    chunks = chunk_all_documents(docs)
    for c in chunks[:3]:
        print(f"[{c['chunk_id']}] Document: {c['document_name']} | Text snippet: {c['text'][:80]}...")
