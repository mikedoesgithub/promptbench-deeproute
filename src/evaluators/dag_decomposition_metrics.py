"""Evaluation engine for Mode 3: 'Multi' Mode (Compound Decomposition & Subagent Routing DAG)."""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional, Any
import re
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment

from src.schemas import (
    CompoundMultiGroundTruth,
    CompoundMultiPrediction,
    CompoundMultiEvalResult,
    SubTask,
    LatencyStats,
)
from src.evaluators.atomic_metrics import calculate_latency_stats, compute_laps


def _tokenize(text: str) -> Set[str]:
    """Basic alphanumeric tokenization for span IoU computation."""
    tokens = re.findall(r"\w+", text.lower())
    return set(tokens)


def compute_token_span_iou(span1: str, span2: str) -> float:
    """Computes token-level Intersection over Union (IoU) between two text spans."""
    t1 = _tokenize(span1)
    t2 = _tokenize(span2)
    if not t1 and not t2:
        return 1.0
    union = t1.union(t2)
    if not union:
        return 0.0
    return len(t1.intersection(t2)) / len(union)


def check_dag_validity(sub_tasks: List[SubTask], allow_cyclic: bool = False) -> Tuple[bool, Optional[str]]:
    """Validates whether sub-tasks form a valid Directed Acyclic Graph (DAG).
    
    Checks:
    1. Unique task IDs.
    2. All dependencies reference valid task IDs.
    3. No self-dependencies.
    4. Graph is acyclic (unless explicitly marked as cyclic/verification loop).
    """
    if not sub_tasks:
        return True, None

    task_ids = set()
    for task in sub_tasks:
        if task.id in task_ids:
            return False, f"Duplicate task id: {task.id}"
        task_ids.add(task.id)

    g = nx.DiGraph()
    for tid in task_ids:
        g.add_node(tid)

    for task in sub_tasks:
        for dep in task.dependencies:
            if dep not in task_ids:
                return False, f"Task {task.id} references undefined dependency {dep}"
            if dep == task.id:
                return False, f"Task {task.id} has self-dependency"
            g.add_edge(dep, task.id)

    if not allow_cyclic and not nx.is_directed_acyclic_graph(g):
        return False, "Cycle detected in dependency graph"

    return True, None


def align_sub_tasks(
    gt_tasks: List[SubTask],
    pred_tasks: List[SubTask],
    iou_threshold: float = 0.4
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Aligns predicted sub-tasks with ground truth sub-tasks using bipartite matching."""
    if not gt_tasks or not pred_tasks:
        return [], list(range(len(gt_tasks))), list(range(len(pred_tasks)))

    n_gt = len(gt_tasks)
    n_pred = len(pred_tasks)
    iou_matrix = np.zeros((n_gt, n_pred), dtype=float)

    for i, gt_t in enumerate(gt_tasks):
        for j, pred_t in enumerate(pred_tasks):
            iou_matrix[i, j] = compute_token_span_iou(gt_t.text_span, pred_t.text_span)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched_pairs = []
    matched_gt = set()
    matched_pred = set()

    for r, c in zip(row_ind, col_ind):
        iou = iou_matrix[r, c]
        if iou >= iou_threshold:
            matched_pairs.append((int(r), int(c), float(iou)))
            matched_gt.add(r)
            matched_pred.add(c)

    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_pred = [j for j in range(n_pred) if j not in matched_pred]

    return matched_pairs, unmatched_gt, unmatched_pred


def compute_dag_edge_metrics(
    gt_tasks: List[SubTask],
    pred_tasks: List[SubTask],
    gt_to_pred_map: Dict[int, int],
    pred_to_gt_map: Dict[int, int]
) -> Dict[str, float]:
    """Computes edge precision, recall, F1, Jaccard, and topological match on aligned tasks."""
    gt_id_to_idx = {t.id: idx for idx, t in enumerate(gt_tasks)}
    pred_id_to_idx = {t.id: idx for idx, t in enumerate(pred_tasks)}

    gt_edges: Set[Tuple[int, int]] = set()
    for dep_idx, t in enumerate(gt_tasks):
        for prereq_id in t.dependencies:
            if prereq_id in gt_id_to_idx:
                gt_edges.add((gt_id_to_idx[prereq_id], dep_idx))

    pred_mapped_edges: Set[Tuple[int, int]] = set()
    unmapped_pred_edges = 0

    for dep_pred_idx, t in enumerate(pred_tasks):
        for prereq_id in t.dependencies:
            prereq_pred_idx = pred_id_to_idx.get(prereq_id)
            if prereq_pred_idx is not None:
                mapped_prereq = pred_to_gt_map.get(prereq_pred_idx)
                mapped_dep = pred_to_gt_map.get(dep_pred_idx)
                if mapped_prereq is not None and mapped_dep is not None:
                    pred_mapped_edges.add((mapped_prereq, mapped_dep))
                else:
                    unmapped_pred_edges += 1
            else:
                unmapped_pred_edges += 1

    total_pred_edges = len(pred_mapped_edges) + unmapped_pred_edges

    if len(gt_edges) == 0 and total_pred_edges == 0:
        return {
            "edge_f1": 1.0,
            "edge_jaccard": 1.0,
            "topological_match": 1.0,
            "graph_edit_distance": 0.0
        }

    correct_edges = len(gt_edges.intersection(pred_mapped_edges))
    precision = correct_edges / total_pred_edges if total_pred_edges > 0 else 0.0
    recall = correct_edges / len(gt_edges) if len(gt_edges) > 0 else 0.0

    if precision + recall > 0:
        edge_f1 = (2 * precision * recall) / (precision + recall)
    else:
        edge_f1 = 0.0

    union_edges = len(gt_edges.union(pred_mapped_edges)) + unmapped_pred_edges
    edge_jaccard = correct_edges / union_edges if union_edges > 0 else 0.0

    topological_match = 1.0 if (gt_edges == pred_mapped_edges and unmapped_pred_edges == 0) else 0.0
    ged = float(len(gt_edges.symmetric_difference(pred_mapped_edges)) + unmapped_pred_edges)

    return {
        "edge_f1": edge_f1,
        "edge_jaccard": edge_jaccard,
        "topological_match": topological_match,
        "graph_edit_distance": ged
    }


def evaluate_compound_multi_predictions(
    ground_truths: List[CompoundMultiGroundTruth],
    predictions: List[CompoundMultiPrediction],
    iou_threshold: float = 0.4,
    total_duration_seconds: Optional[float] = None,
    concurrency: int = 10
) -> CompoundMultiEvalResult:
    """Evaluates Mode 3 Compound Decomposition and Subagent Routing DAG predictions."""
    assert len(ground_truths) == len(predictions), "Mismatched ground truth and prediction counts"
    total_samples = len(ground_truths)
    if total_samples == 0:
        raise ValueError("Cannot evaluate empty test set")

    latencies = [pred.latency_ms for pred in predictions]

    is_compound_correct = 0
    total_tp_boundaries = 0
    total_fp_boundaries = 0
    total_fn_boundaries = 0

    total_matched_tasks = 0
    correct_agent_allocations = 0
    correct_intent_categories = 0

    valid_dags_count = 0
    topological_matches_count = 0
    edge_f1_scores = []
    edge_jaccard_scores = []
    ged_scores = []

    agent_confusion: Dict[str, Dict[str, int]] = {}
    pattern_stats: Dict[str, Dict[str, List[float]]] = {}

    for gt, pred in zip(ground_truths, predictions):
        pattern = gt.dag_pattern or "atomic"
        if pattern not in pattern_stats:
            pattern_stats[pattern] = {
                "boundary_f1": [],
                "agent_acc": [],
                "dag_valid": [],
                "topo_match": []
            }

        # 1. is_compound check
        if gt.is_compound == pred.is_compound:
            is_compound_correct += 1

        # 2. DAG structural validity
        allow_cyclic = (pattern == "dynamic_cyclic")
        is_valid_dag, err = check_dag_validity(pred.sub_tasks, allow_cyclic=allow_cyclic)
        if is_valid_dag:
            valid_dags_count += 1
        pattern_stats[pattern]["dag_valid"].append(1.0 if is_valid_dag else 0.0)

        # 3. Ground truth is atomic (fake-out trap)
        if not gt.is_compound:
            if not pred.is_compound and (len(pred.sub_tasks) <= 1):
                total_tp_boundaries += 1
                edge_f1_scores.append(1.0)
                edge_jaccard_scores.append(1.0)
                topological_matches_count += 1
                ged_scores.append(0.0)
                pattern_stats[pattern]["boundary_f1"].append(1.0)
                pattern_stats[pattern]["topo_match"].append(1.0)
                if gt.sub_tasks and pred.sub_tasks:
                    gt_agent = gt.sub_tasks[0].assigned_subagent
                    pred_agent = pred.sub_tasks[0].assigned_subagent
                    agent_confusion.setdefault(gt_agent, {}).setdefault(pred_agent, 0)
                    agent_confusion[gt_agent][pred_agent] += 1
                    total_matched_tasks += 1
                    if gt_agent == pred_agent:
                        correct_agent_allocations += 1
                        pattern_stats[pattern]["agent_acc"].append(1.0)
                    else:
                        pattern_stats[pattern]["agent_acc"].append(0.0)
            else:
                total_fp_boundaries += len(pred.sub_tasks)
                total_fn_boundaries += 1
                edge_f1_scores.append(0.0)
                edge_jaccard_scores.append(0.0)
                ged_scores.append(float(max(1, len(pred.sub_tasks))))
                pattern_stats[pattern]["boundary_f1"].append(0.0)
                pattern_stats[pattern]["topo_match"].append(0.0)
                pattern_stats[pattern]["agent_acc"].append(0.0)
            continue

        # 4. Compound ground truth: Align sub-tasks
        matched_pairs, unmatched_gt, unmatched_pred = align_sub_tasks(
            gt.sub_tasks, pred.sub_tasks, iou_threshold=iou_threshold
        )

        tp = len(matched_pairs)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)

        total_tp_boundaries += tp
        total_fp_boundaries += fp
        total_fn_boundaries += fn

        sample_boundary_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        sample_boundary_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sample_boundary_f1 = (2 * sample_boundary_p * sample_boundary_r / (sample_boundary_p + sample_boundary_r)) if (sample_boundary_p + sample_boundary_r) > 0 else 0.0
        pattern_stats[pattern]["boundary_f1"].append(sample_boundary_f1)

        gt_to_pred_map = {r: c for r, c, _ in matched_pairs}
        pred_to_gt_map = {c: r for r, c, _ in matched_pairs}

        sample_agent_correct = 0
        for r, c, _ in matched_pairs:
            total_matched_tasks += 1
            gt_t = gt.sub_tasks[r]
            pred_t = pred.sub_tasks[c]

            gt_agent = gt_t.assigned_subagent
            pred_agent = pred_t.assigned_subagent
            agent_confusion.setdefault(gt_agent, {}).setdefault(pred_agent, 0)
            agent_confusion[gt_agent][pred_agent] += 1

            if gt_agent == pred_agent:
                correct_agent_allocations += 1
                sample_agent_correct += 1

            if gt_t.intent_category == pred_t.intent_category:
                correct_intent_categories += 1

        if tp > 0:
            pattern_stats[pattern]["agent_acc"].append(sample_agent_correct / tp)
        else:
            pattern_stats[pattern]["agent_acc"].append(0.0)

        # 5. Dependency Edge Metrics
        edge_metrics = compute_dag_edge_metrics(
            gt.sub_tasks, pred.sub_tasks, gt_to_pred_map, pred_to_gt_map
        )
        edge_f1_scores.append(edge_metrics["edge_f1"])
        edge_jaccard_scores.append(edge_metrics["edge_jaccard"])
        ged_scores.append(edge_metrics["graph_edit_distance"])
        if edge_metrics["topological_match"] == 1.0:
            topological_matches_count += 1
        pattern_stats[pattern]["topo_match"].append(edge_metrics["topological_match"])

    boundary_precision = total_tp_boundaries / (total_tp_boundaries + total_fp_boundaries) if (total_tp_boundaries + total_fp_boundaries) > 0 else 0.0
    boundary_recall = total_tp_boundaries / (total_tp_boundaries + total_fn_boundaries) if (total_tp_boundaries + total_fn_boundaries) > 0 else 0.0
    boundary_f1 = (2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)) if (boundary_precision + boundary_recall) > 0 else 0.0

    agent_accuracy = correct_agent_allocations / total_matched_tasks if total_matched_tasks > 0 else 0.0
    intent_accuracy = correct_intent_categories / total_matched_tasks if total_matched_tasks > 0 else 0.0

    latency_stats = calculate_latency_stats(latencies, total_duration_seconds, concurrency=concurrency)

    # Composite accuracy for LAPS in compound mode: average of boundary F1, agent allocation, and topological match
    composite_compound_acc = (boundary_f1 + agent_accuracy + (topological_matches_count / total_samples)) / 3.0
    laps = compute_laps(composite_compound_acc, latency_stats.p95_ms)

    pattern_breakdown = {}
    for p, stats in pattern_stats.items():
        pattern_breakdown[p] = {
            "count": len(stats["dag_valid"]),
            "boundary_f1": round(float(np.mean(stats["boundary_f1"])), 4) if stats["boundary_f1"] else 0.0,
            "agent_accuracy": round(float(np.mean(stats["agent_acc"])), 4) if stats["agent_acc"] else 0.0,
            "dag_validity": round(float(np.mean(stats["dag_valid"])), 4) if stats["dag_valid"] else 0.0,
            "topological_match": round(float(np.mean(stats["topo_match"])), 4) if stats["topo_match"] else 0.0,
        }

    return CompoundMultiEvalResult(
        total_samples=total_samples,
        is_compound_accuracy=round(is_compound_correct / total_samples, 4),
        task_boundary_f1=round(boundary_f1, 4),
        task_boundary_precision=round(boundary_precision, 4),
        task_boundary_recall=round(boundary_recall, 4),
        agent_allocation_accuracy=round(agent_accuracy, 4),
        intent_category_accuracy=round(intent_accuracy, 4),
        dag_structural_validity=round(valid_dags_count / total_samples, 4),
        dag_topological_match_pct=round(topological_matches_count / total_samples, 4),
        dependency_edge_f1=round(float(np.mean(edge_f1_scores)), 4) if edge_f1_scores else 0.0,
        dependency_edge_jaccard=round(float(np.mean(edge_jaccard_scores)), 4) if edge_jaccard_scores else 0.0,
        mean_graph_edit_distance=round(float(np.mean(ged_scores)), 4) if ged_scores else 0.0,
        laps_score=laps,
        agent_confusion_matrix=agent_confusion,
        latency=latency_stats,
        dag_pattern_breakdown=pattern_breakdown
    )
