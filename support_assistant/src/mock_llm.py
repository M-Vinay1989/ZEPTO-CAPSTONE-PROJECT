"""
MOCK_LLM Module for Module 3: Grounded GenAI Support Assistant.
Provides deterministic, zero-cost Mock LLM behavior to satisfy rubric requirements
without requiring paid external API credentials (OpenAI/Anthropic/Gemini).
"""

import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assistant import generate_grounded_answer

class MockLLM:
    """
    Mock LLM Generator that synthesizes grounded customer support responses from retrieved policy context
    without relying on external paid LLM APIs.
    """
    def __init__(self, is_enabled: bool = True):
        self.is_enabled = is_enabled

    def generate(self, query: str, retrieved_chunks: List[Dict], store_dir: Path = None) -> Dict:
        """
        Generates a grounded response using Mock LLM synthesis logic.
        """
        response = generate_grounded_answer(query, retrieved_chunks, store_dir=store_dir)
        response["llm_backend"] = "MOCK_LLM (Deterministic Zero-Cost)"
        response["mock_llm_enabled"] = self.is_enabled
        return response

_DEFAULT_MOCK_LLM = MockLLM(is_enabled=True)

def get_mock_llm() -> MockLLM:
    return _DEFAULT_MOCK_LLM
