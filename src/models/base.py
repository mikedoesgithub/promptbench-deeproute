"""Abstract base adapter for LLM inference with JSON parsing, category sanitization, and latency measurement."""

from __future__ import annotations
import difflib
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


def sanitize_category(raw_category: str, valid_categories: Optional[List[str]] = None) -> str:
    """Sanitizes and fuzzy-normalizes messy model output strings against valid taxonomy keys.
    
    Handles:
    - `"SWE - Code Generation"` -> `"swe_code_generation"`
    - `"swe_code_generation.py"` -> `"swe_code_generation"`
    - `"category: swe_code_generation"` -> `"swe_code_generation"`
    - `"swe-code-generation"` -> `"swe_code_generation"`
    - `"SWE_CODE_GENERATION"` -> `"swe_code_generation"`
    """
    if not raw_category:
        return "unknown"

    cleaned = raw_category.strip().strip("'\"`")
    
    # If already an exact match in valid_categories, return immediately
    if valid_categories and cleaned in valid_categories:
        return cleaned

    # Strip common prefixes e.g. "category:", "predicted_category:", "class:"
    cleaned = re.sub(r"^(?:predicted_)?(?:category|class)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()

    # Strip common file extensions e.g. .py, .sh, .sql, .js, .ts
    cleaned = re.sub(r"\.(?:py|sh|sql|js|ts|json|yaml|yml|md|rs|go|cpp|c|java)$", "", cleaned, flags=re.IGNORECASE).strip()

    # Direct match check after prefix/ext stripping
    if valid_categories and cleaned in valid_categories:
        return cleaned

    # Normalize into snake_case: replace non-alphanumeric (spaces, hyphens, slashes) with underscore
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", cleaned).strip("_").lower()

    if valid_categories and normalized in valid_categories:
        return normalized

    if not valid_categories:
        return normalized or raw_category

    # 1. Check if normalized is exact match with any valid category
    for vc in valid_categories:
        if normalized == vc.lower():
            return vc

    # 2. Difflib close matches (handles minor typos, missing characters)
    close_matches = difflib.get_close_matches(normalized, valid_categories, n=1, cutoff=0.6)
    if close_matches:
        return close_matches[0]

    # 3. Substring containment (e.g. "code_generation" in "swe_code_generation")
    normalized_tokens = [tok for tok in normalized.split("_") if len(tok) > 2]
    best_candidate = None
    best_overlap = 0

    for vc in valid_categories:
        vc_tokens = set(vc.lower().split("_"))
        overlap = sum(1 for tok in normalized_tokens if tok in vc_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_candidate = vc

    if best_candidate and best_overlap >= 2:
        return best_candidate

    return normalized or raw_category


def sanitize_subagent(raw_subagent: str, valid_subagents: Optional[List[str]] = None) -> str:
    """Sanitizes and normalizes subagent names against the valid 14-subagent registry."""
    if not raw_subagent:
        return "coordinator"

    cleaned = raw_subagent.strip().strip("'\"`").lower()
    cleaned = re.sub(r"^(?:subagent|agent|specialist)\s*:\s*", "", cleaned).strip()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned).strip("_")

    if valid_subagents and cleaned in valid_subagents:
        return cleaned

    # Common aliases
    alias_map = {
        "code": "coder",
        "software_engineer": "coder",
        "developer": "coder",
        "research": "researcher",
        "analyst": "researcher",
        "write": "writer",
        "prose": "writer",
        "copywriter": "writer",
        "math": "math_specialist",
        "mathematician": "math_specialist",
        "data": "data_engineer",
        "sql": "data_engineer",
        "system": "sysadmin",
        "devops": "sysadmin",
        "linux": "sysadmin",
        "finance": "finance_officer",
        "financial": "finance_officer",
        "hardware": "hardware_engineer",
        "pcb": "hardware_engineer",
        "embedded": "hardware_engineer",
        "browser": "browser_navigator",
        "computer_use": "browser_navigator",
        "vision": "vision_analyst",
        "multimodal": "vision_analyst",
        "image": "vision_analyst",
        "mcp": "mcp_orchestrator",
        "protocol": "mcp_orchestrator",
        "safety": "safety_sentinel",
        "sentinel": "safety_sentinel",
        "security_gatekeeper": "safety_sentinel",
        "interviewer": "triage_interviewer",
        "triage": "triage_interviewer",
        "clarification": "triage_interviewer",
        "admin": "coordinator",
    }

    if cleaned in alias_map:
        mapped = alias_map[cleaned]
        if not valid_subagents or mapped in valid_subagents:
            return mapped

    if valid_subagents:
        close = difflib.get_close_matches(cleaned, valid_subagents, n=1, cutoff=0.6)
        if close:
            return close[0]

    return cleaned or "coordinator"


class BaseModelAdapter(ABC):
    """Abstract base class for model adapters with dynamic taxonomy and sanitization support."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.taxonomy: Optional[Dict[str, Any]] = None
        self.subagents: Optional[Dict[str, Any]] = None
        self.valid_categories: List[str] = []
        self.valid_domains: List[str] = []
        self.valid_subagents: List[str] = []

        if "taxonomy" in kwargs and kwargs["taxonomy"]:
            self.set_taxonomy(kwargs["taxonomy"])
        if "subagents" in kwargs and kwargs["subagents"]:
            self.set_subagents(kwargs["subagents"])

    def set_configs(self, taxonomy: Dict[str, Any], subagents: Optional[Dict[str, Any]] = None):
        """Configures the adapter with authoritative taxonomy and subagent registries."""
        if taxonomy:
            self.set_taxonomy(taxonomy)
        if subagents:
            self.set_subagents(subagents)

    def set_taxonomy(self, taxonomy: Dict[str, Any]):
        """Sets the taxonomy dictionary and populates valid_categories and valid_domains lists."""
        self.taxonomy = taxonomy
        cats = []
        domains = []

        if "tier1_domains" in taxonomy:
            for d_key, d_val in taxonomy["tier1_domains"].items():
                domains.append(d_key)
                cats.extend(d_val.get("categories", []))
        elif "granular_categories" in taxonomy:
            for c_key, c_val in taxonomy["granular_categories"].items():
                cats.append(c_key)
                d = c_val.get("domain")
                if d and d not in domains:
                    domains.append(d)

        self.valid_categories = sorted(list(set(cats)))
        self.valid_domains = sorted(list(set(domains)))

    def set_subagents(self, subagents: Dict[str, Any]):
        """Sets the subagents dictionary and populates the valid_subagents list."""
        self.subagents = subagents
        agents_map = subagents.get("subagents", subagents) if isinstance(subagents, dict) else {}
        self.valid_subagents = sorted(list(agents_map.keys()))

    def sanitize_category(self, raw_category: Optional[str]) -> str:
        """Sanitizes raw model category prediction against loaded valid categories."""
        return sanitize_category(raw_category or "", self.valid_categories)

    def sanitize_subagent(self, raw_subagent: Optional[str]) -> str:
        """Sanitizes raw model subagent prediction against loaded valid subagents."""
        return sanitize_subagent(raw_subagent or "", self.valid_subagents)

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
