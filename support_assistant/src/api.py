"""
FastAPI Web Application Module for Module 3: Grounded GenAI Support Assistant.
Exposes POST /ask and GET /health endpoints using LangGraph workflow, ChromaDB vector retrieval, and MOCK_LLM.
"""

import sys
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from support_assistant.src.workflow import run_langgraph_workflow
    from support_assistant.src.document_loader import load_support_documents, DOCUMENTS_DIR
    from support_assistant.src.vector_store import load_vector_store, CHROMA_DB_DIR
except ImportError:
    from workflow import run_langgraph_workflow
    from document_loader import load_support_documents, DOCUMENTS_DIR
    from vector_store import load_vector_store, CHROMA_DB_DIR

app = FastAPI(
    title="Zepto Support Assistant API",
    description="Grounded GenAI Support Assistant API powered by ChromaDB, LangGraph, and MOCK_LLM",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str = Field(..., example="How long does a refund take?", description="Support question to ask the assistant")
    top_k: Optional[int] = Field(default=4, ge=1, le=10, description="Number of context chunks to retrieve")

class AnswerResponse(BaseModel):
    query: str
    answer: str
    is_grounded: bool
    sources: List[str]
    top_score: float
    vector_db: str = "ChromaDB"
    workflow: str = "LangGraph StateGraph"
    llm_backend: str = "MOCK_LLM (Deterministic Zero-Cost)"
    langgraph_executed: bool = True

class HealthResponse(BaseModel):
    status: str
    documents_ingested: int
    vector_db: str
    workflow_engine: str
    llm_mode: str
    version: str

@app.get("/health", response_model=HealthResponse)
@app.get("/status", response_model=HealthResponse)
def health_check():
    """Returns system operational status, document counts, and pipeline components."""
    docs = load_support_documents(DOCUMENTS_DIR)
    return HealthResponse(
        status="ok",
        documents_ingested=len(docs),
        vector_db="ChromaDB",
        workflow_engine="LangGraph",
        llm_mode="MOCK_LLM",
        version="1.0.0"
    )

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Executes RAG retrieval and answer generation workflow for a support query via LangGraph.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    result = run_langgraph_workflow(query=request.question, top_k=request.top_k)
    
    return AnswerResponse(
        query=result.get("query", request.question),
        answer=result.get("answer", ""),
        is_grounded=result.get("is_grounded", False),
        sources=result.get("sources", []),
        top_score=result.get("top_score", 0.0),
        vector_db="ChromaDB",
        workflow="LangGraph StateGraph",
        llm_backend="MOCK_LLM (Deterministic Zero-Cost)",
        langgraph_executed=result.get("langgraph_executed", True)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
