"""Fuzzing and adversarial robustness unit tests for the benchmark harness."""

import pytest
import numpy as np
from src.models.base import extract_json_payload
from src.schemas import (
    AtomicGroundTruth,
    AtomicPrediction,
    MultiLabelGroundTruth,
    MultiLabelPrediction,
    CompoundMultiGroundTruth,
    CompoundMultiPrediction,
    SubTask,
    TopKCategory,
)
from src.evaluators.atomic_metrics import evaluate_atomic_predictions, compute_ece, compute_laps
from src.evaluators.multilabel_metrics import evaluate_multilabel_predictions
from src.evaluators.dag_decomposition_metrics import (
    check_dag_validity,
    evaluate_compound_multi_predictions,
    compute_token_span_iou,
)


def test_extract_json_payload_valid_and_markdown():
    text = "Here is the result:\n```json\n{\"predicted_category\": \"swe_code_generation\"}\n```\nHope that helps!"
    parsed = extract_json_payload(text)
    assert parsed["predicted_category"] == "swe_code_generation"


def test_extract_json_payload_unclosed_fence():
    text = "```json\n{\"predicted_category\": \"math_formal_proof\", \"confidence\": 0.95}"
    parsed = extract_json_payload(text)
    assert parsed["predicted_category"] == "math_formal_proof"


def test_extract_json_payload_no_fence_raw_text():
    text = "Some random text before {\"is_compound\": false, \"sub_tasks\": []} and text after."
    parsed = extract_json_payload(text)
    assert parsed["is_compound"] is False


def test_extract_json_payload_malformed_raises_json_decode_error():
    with pytest.raises(Exception):
        extract_json_payload("This is completely broken and has no json brackets at all.")


def test_atomic_evaluator_with_hallucinated_category():
    gt = [AtomicGroundTruth(
        id="1", prompt="test", domain="software_engineering",
        primary_category="swe_code_generation"
    )]
    pred = [AtomicPrediction(
        predicted_category="alien_hallucinated_category_404",
        predicted_domain="non_existent_domain",
        confidence=0.99,
        latency_ms=45.0
    )]
    result = evaluate_atomic_predictions(gt, pred)
    assert result.top1_accuracy == 0.0
    assert result.domain_accuracy == 0.0
    assert len(result.confusion_pairs) == 1
    assert result.confusion_pairs[0].ground_truth == "swe_code_generation"
    assert result.confusion_pairs[0].predicted == "alien_hallucinated_category_404"


def test_atomic_evaluator_with_nan_or_clamped_confidence():
    gt = [AtomicGroundTruth(id="1", prompt="test", domain="math_symbolic", primary_category="math_formal_proof")]
    pred = [AtomicPrediction(
        predicted_category="math_formal_proof",
        confidence=1.5,  # Out of range, should clamp to 1.0
        latency_ms=30.0
    )]
    result = evaluate_atomic_predictions(gt, pred)
    assert result.top1_accuracy == 1.0
    assert result.ece <= 0.05


def test_multilabel_evaluator_with_empty_and_unknown_facets():
    gt = [MultiLabelGroundTruth(
        id="1", prompt="test", domains=["software_engineering"],
        intents=["swe_code_generation"], modalities=["single_turn_direct"],
        complexity="atomic", safety_flags=["benign_standard"]
    )]
    pred = [MultiLabelPrediction(
        domains=["unknown_domain_foo"],
        intents=[],
        modalities=["unknown_modality_bar"],
        complexity="non_standard_complexity",
        safety_flags=[],
        latency_ms=25.0
    )]
    result = evaluate_multilabel_predictions(gt, pred)
    assert result.exact_match_ratio == 0.0
    assert result.jaccard_index < 0.2
    assert result.hamming_loss > 0.0


def test_compound_dag_validity_with_dangling_dependencies():
    tasks = [
        SubTask(id="task_1", text_span="task a", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=["task_9999_missing"])
    ]
    is_valid, err = check_dag_validity(tasks)
    assert is_valid is False
    assert "undefined dependency" in err


def test_compound_dag_validity_with_self_dependency():
    tasks = [
        SubTask(id="task_1", text_span="task a", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=["task_1"])
    ]
    is_valid, err = check_dag_validity(tasks)
    assert is_valid is False
    assert "self-dependency" in err


def test_compound_dag_validity_with_duplicate_ids():
    tasks = [
        SubTask(id="task_1", text_span="task a", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=[]),
        SubTask(id="task_1", text_span="task b", assigned_subagent="writer", intent_category="prose_longform_essay", dependencies=[]),
    ]
    is_valid, err = check_dag_validity(tasks)
    assert is_valid is False
    assert "Duplicate task id" in err


def test_compound_evaluator_with_empty_prediction_on_compound_gt():
    gt = [CompoundMultiGroundTruth(
        id="1", prompt="multi prompt", is_compound=True,
        sub_tasks=[
            SubTask(id="t1", text_span="part 1", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=[]),
            SubTask(id="t2", text_span="part 2", assigned_subagent="writer", intent_category="prose_email_correspondence", dependencies=["t1"]),
        ]
    )]
    pred = [CompoundMultiPrediction(
        is_compound=False,
        sub_tasks=[],
        latency_ms=10.0
    )]
    res = evaluate_compound_multi_predictions(gt, pred)
    assert res.is_compound_accuracy == 0.0
    assert res.task_boundary_f1 == 0.0
    assert res.agent_allocation_accuracy == 0.0


def test_laps_score_formula_edge_cases():
    # Perfect accuracy and sub-50ms latency -> LAPS = 100.0
    assert compute_laps(1.0, 40.0) == 100.0
    assert compute_laps(1.0, 50.0) == 100.0
    # 500ms latency: log10(500/50) = log10(10) = 1.0 -> denominator = 2.0 -> LAPS = 50.0
    assert compute_laps(1.0, 500.0) == 50.0
    # 0% accuracy -> LAPS = 0.0
    assert compute_laps(0.0, 30.0) == 0.0
