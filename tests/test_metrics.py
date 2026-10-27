"""Unit tests verifying metric calculations: ECE, IoU, DAG validity, GED, Multi-label F1."""

import pytest
import numpy as np
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
from src.evaluators.atomic_metrics import (
    compute_ece,
    evaluate_atomic_predictions,
)
from src.evaluators.multilabel_metrics import (
    evaluate_multilabel_predictions,
    _set_jaccard,
)
from src.evaluators.dag_decomposition_metrics import (
    compute_token_span_iou,
    check_dag_validity,
    align_sub_tasks,
    compute_dag_edge_metrics,
    evaluate_compound_multi_predictions,
)


def test_ece_perfect_calibration():
    # If confidence equals accuracy in every bin, ECE should be 0.0
    confidences = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1])
    # 9 correct at 0.9, 0 correct at 0.1 (so 0/1 = 0 acc at 0.1 conf)
    accuracies = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    ece = compute_ece(confidences, accuracies, num_bins=10)
    # Bin [0.8, 0.9): 9 samples, conf 0.9, acc 1.0 -> diff 0.1
    # Bin [0.1, 0.2): 1 sample, conf 0.1, acc 0.0 -> diff 0.1
    # Weighted ECE = 0.9 * 0.1 + 0.1 * 0.1 = 0.1
    assert abs(ece - 0.1) < 1e-4

    # Perfectly calibrated case:
    # 5 samples with conf 0.8: 4 correct (acc=0.8), 1 incorrect
    conf2 = np.array([0.85, 0.85, 0.85, 0.85, 0.85])
    acc2 = np.array([1.0, 1.0, 1.0, 1.0, 0.0])  # acc = 0.8
    # bin [0.8, 0.9): mean conf = 0.85, mean acc = 0.8 -> diff = 0.05
    ece2 = compute_ece(conf2, acc2, num_bins=10)
    assert abs(ece2 - 0.05) < 1e-4


def test_atomic_evaluation():
    gts = [
        AtomicGroundTruth(
            id="1", prompt="write quicksort in python",
            domain="software_engineering", primary_category="swe_code_generation",
            operational_modality="single_turn_direct", complexity="atomic", safety_flag="benign_standard"
        ),
        AtomicGroundTruth(
            id="2", prompt="find papers on attention mechanisms",
            domain="research_retrieval", primary_category="research_literature_synthesis",
            operational_modality="read_only_lookup", complexity="atomic", safety_flag="benign_standard"
        )
    ]
    preds = [
        AtomicPrediction(
            predicted_category="swe_code_generation",
            predicted_domain="software_engineering",
            confidence=0.95,
            latency_ms=45.0
        ),
        AtomicPrediction(
            predicted_category="research_factual_search",  # Wrong category, right domain
            predicted_domain="research_retrieval",
            confidence=0.70,
            top_k=[
                TopKCategory(category="research_factual_search", confidence=0.70),
                TopKCategory(category="research_literature_synthesis", confidence=0.25)
            ],
            latency_ms=60.0
        )
    ]
    res = evaluate_atomic_predictions(gts, preds)
    assert res.top1_accuracy == 0.5
    assert res.top3_accuracy == 1.0
    assert res.domain_accuracy == 1.0
    assert res.latency.p50_ms == 52.5


def test_multilabel_evaluation():
    gts = [
        MultiLabelGroundTruth(
            id="1", prompt="test",
            domains=["software_engineering"],
            intents=["swe_code_generation", "swe_test_writing"],
            modalities=["autonomous_agent_loop"],
            complexity="sequential_pipeline",
            safety_flags=["benign_standard"],
            tool_dependencies=["bash", "python_repl"]
        )
    ]
    preds = [
        MultiLabelPrediction(
            domains=["software_engineering"],
            intents=["swe_code_generation", "swe_test_writing"],
            modalities=["autonomous_agent_loop"],
            complexity="sequential_pipeline",
            safety_flags=["benign_standard"],
            tool_dependencies=["bash", "python_repl"],
            latency_ms=30.0
        )
    ]
    res = evaluate_multilabel_predictions(gts, preds)
    assert res.exact_match_ratio == 1.0
    assert res.hamming_loss == 0.0
    assert res.jaccard_index == 1.0
    assert res.macro_f1 == 1.0


def test_token_span_iou():
    s1 = "write a Python benchmark script to test it"
    s2 = "Write a python benchmark script"
    iou = compute_token_span_iou(s1, s2)
    # tokens s1: write, a, python, benchmark, script, to, test, it (8)
    # tokens s2: write, a, python, benchmark, script (5)
    # intersection: 5, union: 8 -> 5/8 = 0.625
    assert abs(iou - 0.625) < 1e-4


def test_dag_validity_checker():
    # Valid linear chain
    tasks_valid = [
        SubTask(id="t1", text_span="search papers", assigned_subagent="researcher", intent_category="literature_synthesis", dependencies=[]),
        SubTask(id="t2", text_span="write code", assigned_subagent="coder", intent_category="code_generation", dependencies=["t1"]),
        SubTask(id="t3", text_span="draft email", assigned_subagent="writer", intent_category="prose_email_drafting", dependencies=["t2"]),
    ]
    is_valid, err = check_dag_validity(tasks_valid)
    assert is_valid is True
    assert err is None

    # Invalid cycle: t1 -> t2 -> t3 -> t1
    tasks_cycle = [
        SubTask(id="t1", text_span="search papers", assigned_subagent="researcher", intent_category="literature_synthesis", dependencies=["t3"]),
        SubTask(id="t2", text_span="write code", assigned_subagent="coder", intent_category="code_generation", dependencies=["t1"]),
        SubTask(id="t3", text_span="draft email", assigned_subagent="writer", intent_category="prose_email_drafting", dependencies=["t2"]),
    ]
    is_valid, err = check_dag_validity(tasks_cycle)
    assert is_valid is False
    assert "Cycle" in err

    # Invalid dangling reference
    tasks_dangling = [
        SubTask(id="t1", text_span="search", assigned_subagent="researcher", intent_category="literature_synthesis", dependencies=["t999"]),
    ]
    is_valid, err = check_dag_validity(tasks_dangling)
    assert is_valid is False
    assert "undefined dependency" in err

    # Self-dependency
    tasks_self = [
        SubTask(id="t1", text_span="search", assigned_subagent="researcher", intent_category="literature_synthesis", dependencies=["t1"]),
    ]
    is_valid, err = check_dag_validity(tasks_self)
    assert is_valid is False
    assert "self-dependency" in err


def test_compound_evaluation_bipartite_and_dag():
    gt = CompoundMultiGroundTruth(
        id="c1",
        prompt="Search for papers on Speculative Decoding, write a benchmark in Python, and email results to CTO.",
        is_compound=True,
        dag_pattern="sequential_pipeline",
        sub_tasks=[
            SubTask(id="gt_1", text_span="Search for papers on Speculative Decoding", assigned_subagent="researcher", intent_category="literature_synthesis", dependencies=[]),
            SubTask(id="gt_2", text_span="write a benchmark in Python", assigned_subagent="coder", intent_category="code_generation", dependencies=["gt_1"]),
            SubTask(id="gt_3", text_span="email results to CTO", assigned_subagent="writer", intent_category="prose_email_drafting", dependencies=["gt_2"]),
        ]
    )

    pred = CompoundMultiPrediction(
        is_compound=True,
        sub_tasks=[
            SubTask(id="p1", text_span="Search for papers on Speculative Decoding", assigned_subagent="researcher", intent_category="literature_synthesis", dependencies=[]),
            SubTask(id="p2", text_span="write a benchmark in Python", assigned_subagent="coder", intent_category="code_generation", dependencies=["p1"]),
            SubTask(id="p3", text_span="email results to CTO", assigned_subagent="writer", intent_category="prose_email_drafting", dependencies=["p2"]),
        ],
        latency_ms=85.0
    )

    res = evaluate_compound_multi_predictions([gt], [pred])
    assert res.is_compound_accuracy == 1.0
    assert res.task_boundary_f1 == 1.0
    assert res.agent_allocation_accuracy == 1.0
    assert res.dag_structural_validity == 1.0
    assert res.dag_topological_match_pct == 1.0
    assert res.dependency_edge_f1 == 1.0
    assert res.mean_graph_edit_distance == 0.0
