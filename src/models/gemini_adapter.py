"""Gemini REST API adapter for fast Flash models."""

from __future__ import annotations
import os
import time
from typing import Optional, Dict, Any, List
import httpx

from src.models.base import BaseModelAdapter, extract_json_payload
from src.models.prompts import (
    SYSTEM_ROUTING_PROMPT,
    ATOMIC_PROMPT_TEMPLATE,
    MULTILABEL_PROMPT_TEMPLATE,
    COMPOUND_PROMPT_TEMPLATE,
)
from src.schemas import (
    AtomicPrediction,
    MultiLabelPrediction,
    CompoundMultiPrediction,
    SubTask,
    TopKCategory,
)


class GeminiAdapter(BaseModelAdapter):
    """Adapter for Google Gemini Flash models via Generative Language REST API."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 15.0,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.temperature = temperature
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def _call_gemini(self, system_msg: str, user_msg: str) -> tuple[Dict[str, Any], float]:
        """Calls Gemini generateContent endpoint with JSON response schema."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "system_instruction": {
                "parts": [{"text": system_msg}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_msg}]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": self.temperature,
            }
        }

        t0 = time.perf_counter()
        resp = await self.client.post(url, json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API call failed ({resp.status_code}): {resp.text}")

        res_json = resp.json()
        raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
        parsed = extract_json_payload(raw_text)
        return parsed, latency_ms

    async def predict_atomic(self, prompt: str) -> AtomicPrediction:
        user_prompt = ATOMIC_PROMPT_TEMPLATE.format(prompt=prompt)
        parsed, lat = await self._call_gemini(SYSTEM_ROUTING_PROMPT, user_prompt)

        top_k = []
        if "top_k" in parsed and isinstance(parsed["top_k"], list):
            for item in parsed["top_k"]:
                if isinstance(item, dict) and "category" in item:
                    top_k.append(TopKCategory(category=item["category"], confidence=float(item.get("confidence", 0.5))))

        return AtomicPrediction(
            predicted_category=parsed.get("predicted_category", "unknown"),
            predicted_domain=parsed.get("predicted_domain"),
            confidence=float(parsed.get("confidence", 1.0)),
            top_k=top_k,
            latency_ms=round(lat, 2),
            raw_output=str(parsed)
        )

    async def predict_multilabel(self, prompt: str) -> MultiLabelPrediction:
        user_prompt = MULTILABEL_PROMPT_TEMPLATE.format(prompt=prompt)
        parsed, lat = await self._call_gemini(SYSTEM_ROUTING_PROMPT, user_prompt)

        return MultiLabelPrediction(
            domains=parsed.get("domains", []),
            intents=parsed.get("intents", []),
            modalities=parsed.get("modalities", []),
            complexity=parsed.get("complexity"),
            safety_flags=parsed.get("safety_flags", ["benign_standard"]),
            tool_dependencies=parsed.get("tool_dependencies", []),
            confidence=float(parsed.get("confidence", 1.0)),
            latency_ms=round(lat, 2),
            raw_output=str(parsed)
        )

    async def predict_compound(self, prompt: str) -> CompoundMultiPrediction:
        user_prompt = COMPOUND_PROMPT_TEMPLATE.format(prompt=prompt)
        parsed, lat = await self._call_gemini(SYSTEM_ROUTING_PROMPT, user_prompt)

        sub_tasks = []
        for task in parsed.get("sub_tasks", []):
            sub_tasks.append(SubTask(
                id=str(task.get("id", f"task_{len(sub_tasks)+1}")),
                text_span=str(task.get("text_span", "")),
                assigned_subagent=str(task.get("assigned_subagent", "coordinator")),
                intent_category=str(task.get("intent_category", "unknown")),
                dependencies=[str(d) for d in task.get("dependencies", [])]
            ))

        return CompoundMultiPrediction(
            is_compound=bool(parsed.get("is_compound", False)),
            sub_tasks=sub_tasks,
            latency_ms=round(lat, 2),
            raw_output=str(parsed)
        )
