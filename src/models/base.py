"""Abstract base adapter for LLM inference with JSON parsing and latency measurement."""

from __future__ import annotations
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from src.schemas import (
    AtomicPrediction,
    MultiLabelPrediction,
    CompoundMultiPrediction,
    AdversarialStressPrediction,
    SubTask,
    TopKCategory,
)


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    """Robustly extracts JSON dictionary from raw model text (stripping markdown fences if present)."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


class BaseModelAdapter(ABC):
    """Abstract base class for model adapters."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    async def predict_atomic(self, prompt: str) -> AtomicPrediction:
        """Classifies a prompt into primary category and domain."""
        pass

    @abstractmethod
    async def predict_multilabel(self, prompt: str) -> MultiLabelPrediction:
        """Extracts multi-faceted metadata tags from a prompt."""
        pass

    @abstractmethod
    async def predict_compound(self, prompt: str) -> CompoundMultiPrediction:
        """Decomposes compound prompt into sub-tasks and routing DAG."""
        pass

    async def predict_adversarial(self, prompt: str) -> AdversarialStressPrediction:
        """Predicts category and safety flag under adversarial stress."""
        # Default fallback delegates to atomic prediction
        atomic_pred = await self.predict_atomic(prompt)
        p_lower = prompt.lower()
        safety = "benign_standard"
        subagent = "coder"
        if any(w in p_lower for w in ["override", "bypass", "dan", "rm -rf", "shadow", "unauthorized"]):
            safety = "indirect_prompt_injection"
            subagent = "safety_sentinel"
        return AdversarialStressPrediction(
            predicted_category=atomic_pred.predicted_category,
            predicted_domain=atomic_pred.predicted_domain,
            predicted_safety_flag=safety,
            predicted_subagent=subagent,
            confidence=atomic_pred.confidence,
            latency_ms=atomic_pred.latency_ms,
            raw_output=atomic_pred.raw_output
        )
