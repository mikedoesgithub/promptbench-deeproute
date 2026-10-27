"""Unit tests verifying DAG topology parsing, Hungarian matching, and Graph Edit Distance math."""

import pytest
from src.schemas import (
    CompoundMultiGroundTruth,
    CompoundMultiPrediction,
    SubTask,
)
from src.evaluators.dag_decomposition_metrics import (
    check_dag_validity,
    align_sub_tasks,
    compute_dag_edge_metrics,
    evaluate_compound_multi_predictions,
)


def test_diamond_dag_exact_match():
    # 4-node diamond: T1 -> [T2, T3] -> T4
    gt_tasks = [
        SubTask(id="g1", text_span="analyze database schema", assigned_subagent="data_engineer", intent_category="data_schema_migration", dependencies=[]),
        SubTask(id="g2", text_span="write sql migration script", assigned_subagent="data_engineer", intent_category="data_sql_query_authoring", dependencies=["g1"]),
        SubTask(id="g3", text_span="write locust load test in python", assigned_subagent="coder", intent_category="swe_test_writing", dependencies=["g1"]),
        SubTask(id="g4", text_span="draft performance report", assigned_subagent="writer", intent_category="prose_email_correspondence", dependencies=["g2", "g3"]),
    ]

    pred_tasks = [
        SubTask(id="p1", text_span="analyze database schema", assigned_subagent="data_engineer", intent_category="data_schema_migration", dependencies=[]),
        SubTask(id="p2", text_span="write sql migration script", assigned_subagent="data_engineer", intent_category="data_sql_query_authoring", dependencies=["p1"]),
        SubTask(id="p3", text_span="write locust load test in python", assigned_subagent="coder", intent_category="swe_test_writing", dependencies=["p1"]),
        SubTask(id="p4", text_span="draft performance report", assigned_subagent="writer", intent_category="prose_email_correspondence", dependencies=["p2", "p3"]),
    ]

    matched, _, _ = align_sub_tasks(gt_tasks, pred_tasks, iou_threshold=0.5)
    assert len(matched) == 4

    gt_to_pred = {r: c for r, c, _ in matched}
    pred_to_gt = {c: r for r, c, _ in matched}

    metrics = compute_dag_edge_metrics(gt_tasks, pred_tasks, gt_to_pred, pred_to_gt)
    assert metrics["edge_f1"] == 1.0
    assert metrics["edge_jaccard"] == 1.0
    assert metrics["topological_match"] == 1.0
    assert metrics["graph_edit_distance"] == 0.0


def test_diamond_dag_missing_join_edge():
    # Predicted T4 only depends on T2, missing edge from T3 -> T4
    gt_tasks = [
        SubTask(id="g1", text_span="init root", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=[]),
        SubTask(id="g2", text_span="worker alpha", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=["g1"]),
        SubTask(id="g3", text_span="worker beta", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=["g1"]),
        SubTask(id="g4", text_span="join reducer", assigned_subagent="writer", intent_category="prose_email_correspondence", dependencies=["g2", "g3"]),
    ]
    pred_tasks = [
        SubTask(id="p1", text_span="init root", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=[]),
        SubTask(id="p2", text_span="worker alpha", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=["p1"]),
        SubTask(id="p3", text_span="worker beta", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=["p1"]),
        SubTask(id="p4", text_span="join reducer", assigned_subagent="writer", intent_category="prose_email_correspondence", dependencies=["p2"]), # Missing p3
    ]

    matched, _, _ = align_sub_tasks(gt_tasks, pred_tasks)
    gt_to_pred = {r: c for r, c, _ in matched}
    pred_to_gt = {c: r for r, c, _ in matched}

    metrics = compute_dag_edge_metrics(gt_tasks, pred_tasks, gt_to_pred, pred_to_gt)
    assert metrics["topological_match"] == 0.0
    assert metrics["graph_edit_distance"] == 1.0  # 1 missing edge
    assert metrics["edge_f1"] < 1.0


def test_fan_out_fan_in_6_node_graph():
    # 1 root -> 4 parallel workers -> 1 join aggregator (total 6 nodes, 8 edges)
    gt_tasks = [
        SubTask(id="g1", text_span="incident triage root", assigned_subagent="coordinator", intent_category="admin_workflow_orchestration", dependencies=[]),
        SubTask(id="g2", text_span="sysadmin log check", assigned_subagent="sysadmin", intent_category="sys_log_analysis_forensics", dependencies=["g1"]),
        SubTask(id="g3", text_span="data engineer sql deadlock", assigned_subagent="data_engineer", intent_category="data_sql_query_authoring", dependencies=["g1"]),
        SubTask(id="g4", text_span="browser navigator frontend error", assigned_subagent="browser_navigator", intent_category="browser_form_submission", dependencies=["g1"]),
        SubTask(id="g5", text_span="coder review github pr", assigned_subagent="coder", intent_category="swe_code_refactoring", dependencies=["g1"]),
        SubTask(id="g6", text_span="writer aggregate postmortem", assigned_subagent="writer", intent_category="prose_longform_essay", dependencies=["g2", "g3", "g4", "g5"]),
    ]
    pred_tasks = [
        SubTask(id="p1", text_span="incident triage root", assigned_subagent="coordinator", intent_category="admin_workflow_orchestration", dependencies=[]),
        SubTask(id="p2", text_span="sysadmin log check", assigned_subagent="sysadmin", intent_category="sys_log_analysis_forensics", dependencies=["p1"]),
        SubTask(id="p3", text_span="data engineer sql deadlock", assigned_subagent="data_engineer", intent_category="data_sql_query_authoring", dependencies=["p1"]),
        SubTask(id="p4", text_span="browser navigator frontend error", assigned_subagent="browser_navigator", intent_category="browser_form_submission", dependencies=["p1"]),
        SubTask(id="p5", text_span="coder review github pr", assigned_subagent="coder", intent_category="swe_code_refactoring", dependencies=["p1"]),
        SubTask(id="p6", text_span="writer aggregate postmortem", assigned_subagent="writer", intent_category="prose_longform_essay", dependencies=["p2", "p3", "p4", "p5"]),
    ]

    is_valid, _ = check_dag_validity(pred_tasks)
    assert is_valid is True

    matched, _, _ = align_sub_tasks(gt_tasks, pred_tasks)
    assert len(matched) == 6
    metrics = compute_dag_edge_metrics(gt_tasks, pred_tasks, {r: c for r, c, _ in matched}, {c: r for r, c, _ in matched})
    assert metrics["topological_match"] == 1.0
    assert metrics["edge_f1"] == 1.0


def test_spurious_cross_edge_between_parallel_workers():
    # Adding spurious dependency between parallel workers g2 -> g3
    gt_tasks = [
        SubTask(id="g1", text_span="root", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=[]),
        SubTask(id="g2", text_span="worker 1", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=["g1"]),
        SubTask(id="g3", text_span="worker 2", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=["g1"]),
    ]
    pred_tasks = [
        SubTask(id="p1", text_span="root", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=[]),
        SubTask(id="p2", text_span="worker 1", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=["p1"]),
        SubTask(id="p3", text_span="worker 2", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=["p1", "p2"]), # Spurious edge
    ]

    matched, _, _ = align_sub_tasks(gt_tasks, pred_tasks)
    metrics = compute_dag_edge_metrics(gt_tasks, pred_tasks, {r: c for r, c, _ in matched}, {c: r for r, c, _ in matched})
    assert metrics["topological_match"] == 0.0
    assert metrics["graph_edit_distance"] == 1.0


def test_cyclic_verification_loop_allow_cyclic_flag():
    # Verification loop: Coder -> Tester -> Retry Coder
    cyclic_tasks = [
        SubTask(id="t1", text_span="implement", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=["t2"]),
        SubTask(id="t2", text_span="test", assigned_subagent="coder", intent_category="swe_test_writing", dependencies=["t1"]),
    ]
    # Without allow_cyclic, it fails
    valid, err = check_dag_validity(cyclic_tasks, allow_cyclic=False)
    assert valid is False
    assert "Cycle detected" in err

    # With allow_cyclic, it passes
    valid_allowed, _ = check_dag_validity(cyclic_tasks, allow_cyclic=True)
    assert valid_allowed is True


def test_false_compound_trap_evaluation():
    gt = CompoundMultiGroundTruth(
        id="fc1", prompt="First define user model, then hash password, then save.",
        is_compound=False, dag_pattern="atomic_fakeout",
        sub_tasks=[SubTask(id="g1", text_span="First define user model, then hash password, then save.", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=[])]
    )
    # Correct prediction: recognizes it is not compound
    pred_correct = CompoundMultiPrediction(
        is_compound=False,
        sub_tasks=[SubTask(id="p1", text_span="First define user model, then hash password, then save.", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=[])],
        latency_ms=25.0
    )
    res = evaluate_compound_multi_predictions([gt], [pred_correct])
    assert res.is_compound_accuracy == 1.0
    assert res.task_boundary_f1 == 1.0
    assert res.agent_allocation_accuracy == 1.0
    assert res.dag_topological_match_pct == 1.0

    # Incorrect prediction: model fell into the trap and decomposed it into 3 tasks
    pred_trap = CompoundMultiPrediction(
        is_compound=True,
        sub_tasks=[
            SubTask(id="p1", text_span="define user model", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=[]),
            SubTask(id="p2", text_span="hash password", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=["p1"]),
            SubTask(id="p3", text_span="save", assigned_subagent="coder", intent_category="swe_code_generation", dependencies=["p2"]),
        ],
        latency_ms=55.0
    )
    res_trap = evaluate_compound_multi_predictions([gt], [pred_trap])
    assert res_trap.is_compound_accuracy == 0.0
    assert res_trap.task_boundary_f1 == 0.0
    assert res_trap.dag_topological_match_pct == 0.0
