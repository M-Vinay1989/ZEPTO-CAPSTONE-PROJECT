"""
Main CLI Interface for Zepto Grounded GenAI Support Assistant (Module 3).
Provides commands to rebuild vector index and interactively query policy documents.
"""

import sys
import argparse
from pathlib import Path

# Add parent path for local module imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from document_loader import load_documents, DOCUMENTS_DIR
from chunker import chunk_all_documents
from vector_store import build_vector_store, VECTOR_STORE_DIR
from assistant import ask_assistant

def build_index():
    """
    Loads documents, splits into chunks, and builds/saves local vector store index.
    """
    print("=" * 60)
    print("Building Zepto Support Assistant Vector Store Index...")
    print("=" * 60)
    
    docs = load_documents(DOCUMENTS_DIR)
    print(f"[1/3] Loaded {len(docs)} policy documents.")
    
    chunks = chunk_all_documents(docs)
    print(f"[2/3] Chunked documents into {len(chunks)} total section chunks.")
    
    vector_store = build_vector_store(chunks, VECTOR_STORE_DIR)
    print(f"[3/3] Vector store index successfully saved to: {VECTOR_STORE_DIR}")
    print("=" * 60)
    print("Index build complete!")

def handle_query(query: str):
    """
    Executes query against RAG assistant and prints answer and source attribution.
    """
    response = ask_assistant(query)
    print("\n" + "=" * 60)
    print(f"QUESTION: {response['query']}")
    print("-" * 60)
    print(f"ANSWER:\n{response['answer']}")
    print("-" * 60)
    if response["is_grounded"]:
        print(f"GROUNDED: Yes (Sources: {', '.join(response['sources'])})")
    else:
        print("GROUNDED: No (Ungrounded Query - Refused)")
    print("=" * 60 + "\n")

def run_interactive():
    """
    Interactive CLI loop for querying policy documents.
    """
    print("=" * 60)
    print("Zepto Grounded GenAI Support Assistant - Interactive CLI")
    print("Type your policy question below (or 'exit' / 'quit' to stop).")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nEnter Question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting Support Assistant CLI. Goodbye!")
                break
            handle_query(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI.")
            break

def main():
    parser = argparse.ArgumentParser(description="Zepto Grounded GenAI Support Assistant")
    parser.add_argument("--build", action="store_true", help="Rebuild vector store index from policy documents")
    parser.add_argument("--query", type=str, help="Single query to ask the support assistant")
    
    args = parser.parse_args()
    
    if args.build:
        build_index()
    elif args.query:
        handle_query(args.query)
    else:
        # Default to building if store doesn't exist, else interactive mode
        index_file = VECTOR_STORE_DIR / "index.faiss"
        if not index_file.exists():
            print("Vector store index not found. Building index for first-time use...")
            build_index()
        run_interactive()

if __name__ == "__main__":
    main()
