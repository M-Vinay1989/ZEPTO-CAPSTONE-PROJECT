"""
LangGraph Workflow Module for Module 3: Grounded GenAI Support Assistant.
Implements an executable StateGraph workflow representing the assistant state machine:
question -> normalize_query -> retrieve_context -> generate_answer -> return_response.
"""

import sys
from pathlib import Path
from typing import List, Dict, TypedDict, Optional
from langgraph.graph import StateGraph, START, END

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assistant import normalize_query_typos
from retriever import retrieve_relevant_chunks, CHROMA_DB_DIR
from mock_llm import get_mock_llm

class SupportAssistantState(TypedDict):
    query: str
    normalized_query: Optional[str]
    top_k: int
    chunks: Optional[List[Dict]]
    response: Optional[Dict]
    is_grounded: Optional[bool]
    sources: Optional[List[str]]
    top_score: Optional[float]

# 1. Node Functions
def normalize_query_node(state: SupportAssistantState) -> Dict:
    query = state.get("query", "")
    norm_query = normalize_query_typos(query)
    return {"normalized_query": norm_query}

def retrieve_context_node(state: SupportAssistantState) -> Dict:
    norm_query = state.get("normalized_query") or state.get("query", "")
    top_k = state.get("top_k", 4)
    chunks = retrieve_relevant_chunks(norm_query, top_k=top_k)
    return {"chunks": chunks}

def generate_answer_node(state: SupportAssistantState) -> Dict:
    norm_query = state.get("normalized_query") or state.get("query", "")
    chunks = state.get("chunks", [])
    mock_llm = get_mock_llm()
    resp = mock_llm.generate(norm_query, chunks)
    return {
        "response": resp,
        "is_grounded": resp.get("is_grounded", False),
        "sources": resp.get("sources", []),
        "top_score": resp.get("top_score", 0.0)
    }

# 2. Build LangGraph StateGraph
builder = StateGraph(SupportAssistantState)

builder.add_node("normalize_query", normalize_query_node)
builder.add_node("retrieve_context", retrieve_context_node)
builder.add_node("generate_answer", generate_answer_node)

builder.add_edge(START, "normalize_query")
builder.add_edge("normalize_query", "retrieve_context")
builder.add_edge("retrieve_context", "generate_answer")
builder.add_edge("generate_answer", END)

# Compile LangGraph StateGraph executable graph
support_assistant_graph = builder.compile()

def run_langgraph_workflow(query: str, top_k: int = 4) -> Dict:
    """
    Executes the LangGraph StateGraph workflow end-to-end for a user question.
    """
    initial_state = {
        "query": query,
        "normalized_query": query,
        "top_k": top_k,
        "chunks": [],
        "response": {},
        "is_grounded": False,
        "sources": [],
        "top_score": 0.0
    }
    final_state = support_assistant_graph.invoke(initial_state)
    resp = final_state.get("response", {})
    resp["langgraph_executed"] = True
    resp["workflow_nodes"] = ["normalize_query", "retrieve_context", "generate_answer"]
    return resp

if __name__ == "__main__":
    test_q = "How long does a refund take?"
    result = run_langgraph_workflow(test_q)
    print("--- LangGraph Execution Result ---")
    print("Query:", result.get("query"))
    print("Answer:\n", result.get("answer"))
    print("Sources:", result.get("sources"))
    print("LangGraph Executed:", result.get("langgraph_executed"))
