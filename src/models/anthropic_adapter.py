"""Anthropic Messages API adapter for Claude models with dynamic prompt injection and sanitization."""

from __future__ import annotations
import os
import time
from typing import Optional, Dict, Any, List
import httpx

from src.models.base import BaseModelAdapter, extract_json_payload
from src.models.prompts import (
    build_atomic_system_prompt,
    build_compound_system_prompt,
    build_multilabel_system_prompt,
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


class AnthropicAdapter(BaseModelAdapter):
    """Adapter for Anthropic Claude models via Messages API."""

    def __init__(
        self,
        model_name: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 15.0,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def _call_anthropic(self, system_msg: str, user_msg: str) -> tuple[Dict[str, Any], float]:
        """Calls Anthropic /v1/messages and extracts JSON response."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "system": system_msg,
            "messages": [{"role": "user", "content": user_msg}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        t0 = time.perf_counter()
        resp = await self.client.post(url, headers=headers, json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            raise RuntimeError(f"Anthropic API call failed ({resp.status_code}): {resp.text}")

        res_json = resp.json()
        raw_text = res_json["content"][0]["text"]
        parsed = extract_json_payload(raw_text)
        return parsed, latency_ms

    async def predict_atomic(self, prompt: str) -> AtomicPrediction:
        system_prompt = build_atomic_system_prompt(self.taxonomy) if self.taxonomy else SYSTEM_ROUTING_PROMPT
        user_prompt = ATOMIC_PROMPT_TEMPLATE.format(prompt=prompt)
        parsed, lat = await self._call_anthropic(system_prompt, user_prompt)

        raw_cat = parsed.get("predicted_category", "unknown")
        sanitized_cat = self.sanitize_category(raw_cat)
        raw_dom = parsed.get("predicted_domain")

        top_k = []
        if "top_k" in parsed and isinstance(parsed["top_k"], list):
            for item in parsed["top_k"]:
                if isinstance(item, dict) and "category" in item:
                    top_k.append(TopKCategory(
                        category=self.sanitize_category(item["category"]),
                        confidence=float(item.get("confidence", 0.5))
                    ))

        return AtomicPrediction(
            predicted_category=sanitized_cat,
            predicted_domain=raw_dom,
            confidence=float(parsed.get("confidence", 1.0)),
            top_k=top_k,
            latency_ms=round(lat, 2),
            raw_output=str(parsed)
        )

    async def predict_multilabel(self, prompt: str) -> MultiLabelPrediction:
        system_prompt = build_multilabel_system_prompt(self.taxonomy) if self.taxonomy else SYSTEM_ROUTING_PROMPT
        user_prompt = MULTILABEL_PROMPT_TEMPLATE.format(prompt=prompt)
        parsed, lat = await self._call_anthropic(system_prompt, user_prompt)

        sanitized_intents = [self.sanitize_category(cat) for cat in parsed.get("intents", [])]

        return MultiLabelPrediction(
            domains=parsed.get("domains", []),
            intents=sanitized_intents,
            modalities=parsed.get("modalities", []),
            complexity=parsed.get("complexity"),
            safety_flags=parsed.get("safety_flags", ["benign_standard"]),
            tool_dependencies=parsed.get("tool_dependencies", []),
            confidence=float(parsed.get("confidence", 1.0)),
            latency_ms=round(lat, 2),
            raw_output=str(parsed)
        )

    async def predict_compound(self, prompt: str) -> CompoundMultiPrediction:
        system_prompt = build_compound_system_prompt(self.subagents, self.taxonomy) if self.subagents else SYSTEM_ROUTING_PROMPT
        user_prompt = COMPOUND_PROMPT_TEMPLATE.format(prompt=prompt)
        parsed, lat = await self._call_anthropic(system_prompt, user_prompt)

        sub_tasks = []
        for task in parsed.get("sub_tasks", []):
            sub_tasks.append(SubTask(
                id=str(task.get("id", f"task_{len(sub_tasks)+1}")),
                text_span=str(task.get("text_span", "")),
                assigned_subagent=self.sanitize_subagent(str(task.get("assigned_subagent", "coordinator"))),
                intent_category=self.sanitize_category(str(task.get("intent_category", "unknown"))),
                dependencies=[str(d) for d in task.get("dependencies", [])]
            ))

        return CompoundMultiPrediction(
            is_compound=bool(parsed.get("is_compound", False)),
            sub_tasks=sub_tasks,
            latency_ms=round(lat, 2),
            raw_output=str(parsed)
        )
