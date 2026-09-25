"""
Main CLI & Server Launcher for Zepto Grounded GenAI Support Assistant (Module 3).
Provides commands to rebuild vector index, query policy documents via LangGraph, and launch FastAPI server.
"""

import sys
import argparse
from pathlib import Path

# Add parent path for local module imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from support_assistant.src.document_loader import load_support_documents, DOCUMENTS_DIR
    from support_assistant.src.chunker import chunk_all_documents
    from support_assistant.src.vector_store import build_vector_store, CHROMA_DB_DIR
    from support_assistant.src.workflow import run_langgraph_workflow
except ImportError:
    from document_loader import load_support_documents, DOCUMENTS_DIR
    from chunker import chunk_all_documents
    from vector_store import build_vector_store, CHROMA_DB_DIR
    from workflow import run_langgraph_workflow

def build_index():
    """
    Loads documents, splits into chunks, and builds persistent ChromaDB vector store index.
    """
    print("=" * 60)
    print("Building Zepto Support Assistant ChromaDB Vector Store Index...")
    print("=" * 60)
    
    docs = load_support_documents(DOCUMENTS_DIR)
    print(f"[1/3] Loaded {len(docs)} policy documents.")
    
    chunks = chunk_all_documents(docs)
    print(f"[2/3] Chunked documents into {len(chunks)} total section chunks.")
    
    build_info = build_vector_store(chunks, CHROMA_DB_DIR)
    print(f"[3/3] Persistent ChromaDB collection successfully stored at: {CHROMA_DB_DIR}")
    print("=" * 60)
    print(f"Index build complete! Persistent vectors: {build_info['count']}")

def handle_query(query: str):
    """
    Executes query through LangGraph StateGraph workflow and prints answer and source attribution.
    """
    response = run_langgraph_workflow(query)
    print("\n" + "=" * 60)
    print(f"QUESTION: {response.get('query', query)}")
    print("-" * 60)
    print(f"ANSWER:\n{response.get('answer', '')}")
    print("-" * 60)
    if response.get("is_grounded", False):
        print(f"GROUNDED: Yes (Sources: {', '.join(response.get('sources', []))})")
    else:
        print("GROUNDED: No (Ungrounded Query - Refused)")
    print("=" * 60 + "\n")

def run_server(host: str = "127.0.0.1", port: int = 8000):
    """
    Launches FastAPI web server via Uvicorn.
    """
    import uvicorn
    print(f"Launching Zepto Support Assistant FastAPI Server at http://{host}:{port}")
    uvicorn.run("support_assistant.src.api:app", host=host, port=port, reload=False)

def run_interactive():
    """
    Interactive CLI loop for querying policy documents.
    """
    print("=" * 60)
    print("Zepto Grounded GenAI Support Assistant - Interactive CLI")
    print("Powered by ChromaDB, LangGraph, and MOCK_LLM")
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
    parser.add_argument("--build", action="store_true", help="Rebuild persistent ChromaDB vector store index from policy documents")
    parser.add_argument("--query", type=str, help="Single query to ask the support assistant")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI web server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for FastAPI server")
    parser.add_argument("--port", type=int, default=8000, help="Port number for FastAPI server")
    
    args = parser.parse_args()
    
    if args.build:
        build_index()
    elif args.serve:
        run_server(args.host, args.port)
    elif args.query:
        handle_query(args.query)
    else:
        chroma_file = CHROMA_DB_DIR / "chroma.sqlite3"
        if not chroma_file.exists():
            print("ChromaDB vector store index not found. Building index for first-time use...")
            build_index()
        run_interactive()

if __name__ == "__main__":
    main()
