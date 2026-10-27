"""Pydantic schemas for ground-truth datasets, model predictions, and evaluation metrics."""

from __future__ import annotations
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# ==========================================
# Ground Truth Schemas
# ==========================================

class AtomicGroundTruth(BaseModel):
    id: str
    prompt: str
    domain: str
    primary_category: str
    operational_modality: str = "single_turn_direct"
    complexity: str = "atomic"
    safety_flag: str = "benign_standard"
    adversarial_type: Optional[str] = "none"
    explanation: Optional[str] = None


class MultiLabelGroundTruth(BaseModel):
    id: str
    prompt: str
    domains: List[str]
    intents: List[str]
    modalities: List[str]
    complexity: str
    safety_flags: List[str]
    tool_dependencies: List[str] = Field(default_factory=list)
    adversarial_type: Optional[str] = "none"
    explanation: Optional[str] = None


class SubTask(BaseModel):
    id: str
    text_span: str
    assigned_subagent: str
    intent_category: str
    dependencies: List[str] = Field(default_factory=list)


class CompoundMultiGroundTruth(BaseModel):
    id: str
    prompt: str
    is_compound: bool
    sub_tasks: List[SubTask] = Field(default_factory=list)
    dag_pattern: Optional[str] = "atomic"  # "linear_pipeline", "parallel_map_reduce", "diamond_dag", "dynamic_cyclic", "atomic_fakeout"
    adversarial_type: Optional[str] = "none"
    explanation: Optional[str] = None


class AdversarialStressGroundTruth(BaseModel):
    id: str
    prompt: str
    stress_type: str  # "prompt_injection", "multilingual_codeswitch", "misleading_distractor"
    expected_domain: str
    expected_category: str
    expected_safety_flag: str
    expected_subagent: str
    multilingual_languages: List[str] = Field(default_factory=list)
    distractor_keywords: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None


# ==========================================
# Model Prediction Schemas
# ==========================================

class TopKCategory(BaseModel):
    category: str
    confidence: float


class AtomicPrediction(BaseModel):
    predicted_category: str
    predicted_domain: Optional[str] = None
    confidence: float = 1.0
    top_k: List[TopKCategory] = Field(default_factory=list)
    latency_ms: float = 0.0
    raw_output: Optional[str] = None


class MultiLabelPrediction(BaseModel):
    domains: List[str] = Field(default_factory=list)
    intents: List[str] = Field(default_factory=list)
    modalities: List[str] = Field(default_factory=list)
    complexity: Optional[str] = None
    safety_flags: List[str] = Field(default_factory=list)
    tool_dependencies: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    latency_ms: float = 0.0
    raw_output: Optional[str] = None


class CompoundMultiPrediction(BaseModel):
    is_compound: bool
    sub_tasks: List[SubTask] = Field(default_factory=list)
    latency_ms: float = 0.0
    raw_output: Optional[str] = None


class AdversarialStressPrediction(BaseModel):
    predicted_category: str
    predicted_domain: Optional[str] = None
    predicted_safety_flag: str = "benign_standard"
    predicted_subagent: Optional[str] = None
    confidence: float = 1.0
    latency_ms: float = 0.0
    raw_output: Optional[str] = None


# ==========================================
# Evaluation Result Schemas
# ==========================================

class ConfusionPair(BaseModel):
    ground_truth: str
    predicted: str
    count: int


class LatencyStats(BaseModel):
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float = 0.0
    min_ms: float
    max_ms: float
    throughput_rps: float
    concurrency: int = 10


class ConcurrencySweepPoint(BaseModel):
    concurrency: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    laps_score: float


class AtomicEvalResult(BaseModel):
    total_samples: int
    top1_accuracy: float
    top3_accuracy: float
    macro_f1: float
    micro_f1: float
    ece: float
    domain_accuracy: float
    laps_score: float = 0.0
    confusion_pairs: List[ConfusionPair] = Field(default_factory=list)
    latency: LatencyStats
    per_category_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    domain_breakdown: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    concurrency_sweep: List[ConcurrencySweepPoint] = Field(default_factory=list)


class MultiLabelEvalResult(BaseModel):
    total_samples: int
    exact_match_ratio: float
    hamming_loss: float
    jaccard_index: float
    macro_f1: float
    micro_f1: float
    laps_score: float = 0.0
    facet_accuracies: Dict[str, float] = Field(default_factory=dict)
    latency: LatencyStats
    concurrency_sweep: List[ConcurrencySweepPoint] = Field(default_factory=list)


class CompoundMultiEvalResult(BaseModel):
    total_samples: int
    is_compound_accuracy: float
    task_boundary_f1: float
    task_boundary_precision: float
    task_boundary_recall: float
    agent_allocation_accuracy: float
    intent_category_accuracy: float
    dag_structural_validity: float
    dag_topological_match_pct: float
    dependency_edge_f1: float
    dependency_edge_jaccard: float
    mean_graph_edit_distance: float
    laps_score: float = 0.0
    agent_confusion_matrix: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    latency: LatencyStats
    dag_pattern_breakdown: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    concurrency_sweep: List[ConcurrencySweepPoint] = Field(default_factory=list)


class AdversarialStressEvalResult(BaseModel):
    total_samples: int
    overall_accuracy: float
    injection_detection_rate: float
    multilingual_accuracy: float
    distractor_robustness_rate: float
    safety_flag_accuracy: float
    laps_score: float = 0.0
    latency: LatencyStats
    confusion_pairs: List[ConfusionPair] = Field(default_factory=list)
    stress_breakdown: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class BenchmarkSummary(BaseModel):
    model_name: str
    timestamp: str
    total_runtime_seconds: float
    overall_laps_score: float = 0.0
    atomic_results: Optional[AtomicEvalResult] = None
    multilabel_results: Optional[MultiLabelEvalResult] = None
    compound_results: Optional[CompoundMultiEvalResult] = None
    adversarial_results: Optional[AdversarialStressEvalResult] = None
