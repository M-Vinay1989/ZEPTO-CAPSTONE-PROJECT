"""
Document Loader Module for Module 3: Grounded GenAI Support Assistant.
Loads synthetic policy text documents from support_assistant/documents/ directory.
"""

from pathlib import Path
from typing import List, Dict

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"

def load_documents(docs_dir: Path = DOCUMENTS_DIR) -> List[Dict[str, str]]:
    """
    Loads text policy documents from specified directory.
    Returns list of document dictionaries containing document_name, category, text, and source_path.
    """
    docs_dir = Path(docs_dir).resolve()
    documents = []

    if not docs_dir.exists():
        print(f"[Document Loader Warning] Directory '{docs_dir}' does not exist.")
        return []

    txt_files = sorted(list(docs_dir.glob("*.txt")))
    if not txt_files:
        print(f"[Document Loader Warning] No .txt documents found in '{docs_dir}'.")
        return []

    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                print(f"[Document Loader Warning] Skipping empty document '{file_path.name}'.")
                continue

            doc_name = file_path.name
            category = file_path.stem.replace("_policy", "").replace("_", " ").title()

            documents.append({
                "document_name": doc_name,
                "category": category,
                "text": content,
                "source_path": str(file_path)
            })
        except Exception as err:
            print(f"[Document Loader Error] Failed to read '{file_path}': {err}")

    print(f"[Document Loader] Loaded {len(documents)} policy documents from '{docs_dir}'.")
    return documents

load_support_documents = load_documents

if __name__ == "__main__":
    docs = load_documents()
    for d in docs:
        print(f"Document: {d['document_name']} | Category: {d['category']} | Length: {len(d['text'])} chars")
