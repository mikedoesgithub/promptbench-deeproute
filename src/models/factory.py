"""Factory function for instantiating model adapters based on CLI flags."""

from __future__ import annotations
from typing import Optional, Dict, Any

from src.models.base import BaseModelAdapter
from src.models.mock_adapter import MockModelAdapter
from src.models.openai_adapter import OpenAIAdapter
from src.models.gemini_adapter import GeminiAdapter
from src.models.anthropic_adapter import AnthropicAdapter


def get_model_adapter(
    model_name: str,
    mock: bool = False,
    oracle: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
) -> BaseModelAdapter:
    """Returns the appropriate BaseModelAdapter based on model_name and mock flag."""
    if mock or oracle or model_name.startswith("mock"):
        return MockModelAdapter(model_name=model_name, oracle=oracle, **kwargs)

    lower = model_name.lower()
    if "gemini" in lower:
        return GeminiAdapter(model_name=model_name, api_key=api_key, temperature=temperature, **kwargs)
    elif "claude" in lower:
        return AnthropicAdapter(model_name=model_name, api_key=api_key, temperature=temperature, **kwargs)
    else:
        # Default to OpenAI compatible (GPT-4o-mini, vLLM, Ollama, Groq, Mistral, etc.)
        return OpenAIAdapter(model_name=model_name, api_key=api_key, base_url=base_url, temperature=temperature, **kwargs)
