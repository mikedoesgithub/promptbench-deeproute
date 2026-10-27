"""Evaluation engine for Mode 2: Multi-Label Multi-Aspect Tagging."""

from __future__ import annotations
from typing import List, Dict, Set, Optional
import numpy as np
from sklearn.metrics import f1_score

from src.schemas import (
    MultiLabelGroundTruth,
    MultiLabelPrediction,
    MultiLabelEvalResult,
    LatencyStats,
)
from src.evaluators.atomic_metrics import calculate_latency_stats, compute_laps


def _extract_all_tags(gt: MultiLabelGroundTruth) -> Set[str]:
    """Flattens all facets of ground truth into prefixed tag strings."""
    tags = set()
    for d in gt.domains:
        tags.add(f"domain:{d}")
    for i in gt.intents:
        tags.add(f"intent:{i}")
    for m in gt.modalities:
        tags.add(f"modality:{m}")
    if gt.complexity:
        tags.add(f"complexity:{gt.complexity}")
    for s in gt.safety_flags:
        tags.add(f"safety:{s}")
    for t in gt.tool_dependencies:
        tags.add(f"tool:{t}")
    return tags


def _extract_pred_tags(pred: MultiLabelPrediction) -> Set[str]:
    """Flattens all facets of prediction into prefixed tag strings."""
    tags = set()
    for d in pred.domains:
        tags.add(f"domain:{d}")
    for i in pred.intents:
        tags.add(f"intent:{i}")
    for m in pred.modalities:
        tags.add(f"modality:{m}")
    if pred.complexity:
        tags.add(f"complexity:{pred.complexity}")
    for s in pred.safety_flags:
        tags.add(f"safety:{s}")
    for t in pred.tool_dependencies:
        tags.add(f"tool:{t}")
    return tags


def _set_jaccard(s1: Set[str], s2: Set[str]) -> float:
    """Computes Jaccard index between two sets."""
    if not s1 and not s2:
        return 1.0
    union = s1.union(s2)
    if not union:
        return 1.0
    return len(s1.intersection(s2)) / len(union)


def evaluate_multilabel_predictions(
    ground_truths: List[MultiLabelGroundTruth],
    predictions: List[MultiLabelPrediction],
    total_duration_seconds: Optional[float] = None,
    concurrency: int = 10
) -> MultiLabelEvalResult:
    """Evaluates multi-label / multi-aspect predictions."""
    assert len(ground_truths) == len(predictions), "Mismatched ground truth and prediction counts"
    total_samples = len(ground_truths)
    if total_samples == 0:
        raise ValueError("Cannot evaluate empty test set")

    latencies = [pred.latency_ms for pred in predictions]

    # Collect universal vocabulary of tags
    all_vocab: Set[str] = set()
    gt_tag_sets: List[Set[str]] = []
    pred_tag_sets: List[Set[str]] = []

    for gt, pred in zip(ground_truths, predictions):
        gt_tags = _extract_all_tags(gt)
        pred_tags = _extract_pred_tags(pred)
        gt_tag_sets.append(gt_tags)
        pred_tag_sets.append(pred_tags)
        all_vocab.update(gt_tags)
        all_vocab.update(pred_tags)

    sorted_vocab = sorted(list(all_vocab))
    vocab_to_idx = {tag: i for i, tag in enumerate(sorted_vocab)}
    num_tags = len(sorted_vocab)

    # Build binary indicator matrices
    y_true_matrix = np.zeros((total_samples, num_tags), dtype=int)
    y_pred_matrix = np.zeros((total_samples, num_tags), dtype=int)

    exact_matches = 0
    jaccard_scores = []

    facet_jaccards: Dict[str, List[float]] = {
        "domains": [],
        "intents": [],
        "modalities": [],
        "complexity": [],
        "safety_flags": [],
        "tool_dependencies": [],
    }

    for i in range(total_samples):
        gt = ground_truths[i]
        pred = predictions[i]
        gt_s = gt_tag_sets[i]
        pred_s = pred_tag_sets[i]

        for tag in gt_s:
            y_true_matrix[i, vocab_to_idx[tag]] = 1
        for tag in pred_s:
            y_pred_matrix[i, vocab_to_idx[tag]] = 1

        if gt_s == pred_s:
            exact_matches += 1

        jaccard_scores.append(_set_jaccard(gt_s, pred_s))

        facet_jaccards["domains"].append(_set_jaccard(set(gt.domains), set(pred.domains)))
        facet_jaccards["intents"].append(_set_jaccard(set(gt.intents), set(pred.intents)))
        facet_jaccards["modalities"].append(_set_jaccard(set(gt.modalities), set(pred.modalities)))
        facet_jaccards["complexity"].append(1.0 if gt.complexity == pred.complexity else 0.0)
        facet_jaccards["safety_flags"].append(_set_jaccard(set(gt.safety_flags), set(pred.safety_flags)))
        facet_jaccards["tool_dependencies"].append(_set_jaccard(set(gt.tool_dependencies), set(pred.tool_dependencies)))

    exact_match_ratio = exact_matches / total_samples
    mean_jaccard = float(np.mean(jaccard_scores))
    hamming_loss = float(np.mean(y_true_matrix != y_pred_matrix))

    macro_f1 = float(f1_score(y_true_matrix, y_pred_matrix, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true_matrix, y_pred_matrix, average="micro", zero_division=0))

    facet_accuracies = {
        facet: round(float(np.mean(scores)), 4)
        for facet, scores in facet_jaccards.items()
    }

    latency_stats = calculate_latency_stats(latencies, total_duration_seconds, concurrency=concurrency)
    laps = compute_laps(mean_jaccard, latency_stats.p95_ms)

    return MultiLabelEvalResult(
        total_samples=total_samples,
        exact_match_ratio=round(exact_match_ratio, 4),
        hamming_loss=round(hamming_loss, 4),
        jaccard_index=round(mean_jaccard, 4),
        macro_f1=round(macro_f1, 4),
        micro_f1=round(micro_f1, 4),
        laps_score=laps,
        facet_accuracies=facet_accuracies,
        latency=latency_stats,
    )
