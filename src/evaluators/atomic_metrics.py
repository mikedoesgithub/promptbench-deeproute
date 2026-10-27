"""Evaluation engine for Mode 1: Atomic (Single-Label) Classification."""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from src.schemas import (
    AtomicGroundTruth,
    AtomicPrediction,
    AtomicEvalResult,
    ConfusionPair,
    LatencyStats,
)


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE) across equal-width confidence bins.
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if len(confidences) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_laps(accuracy: float, p95_latency_ms: float) -> float:
    """Computes Latency-Accuracy Pareto Score (LAPS).
    
    LAPS = (Accuracy * 100) / (1 + log10(max(1.0, P95_Latency_ms / 50.0)))
    Rewards sub-50ms models that maintain high routing precision.
    """
    ratio = max(1.0, p95_latency_ms / 50.0)
    denom = 1.0 + float(np.log10(ratio))
    score = (accuracy * 100.0) / denom
    return round(float(score), 2)


def calculate_latency_stats(
    latencies_ms: List[float],
    total_duration_seconds: Optional[float] = None,
    concurrency: int = 10
) -> LatencyStats:
    """Calculates percentile latencies, standard deviation, and throughput."""
    if not latencies_ms:
        return LatencyStats(
            p50_ms=0.0, p90_ms=0.0, p95_ms=0.0, p99_ms=0.0,
            mean_ms=0.0, std_ms=0.0, min_ms=0.0, max_ms=0.0,
            throughput_rps=0.0, concurrency=concurrency
        )

    arr = np.array(latencies_ms)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    if total_duration_seconds and total_duration_seconds > 0:
        rps = len(latencies_ms) / total_duration_seconds
    else:
        total_sec = max(np.sum(arr) / 1000.0, 0.001)
        rps = len(latencies_ms) / total_sec

    return LatencyStats(
        p50_ms=round(p50, 2),
        p90_ms=round(p90, 2),
        p95_ms=round(p95, 2),
        p99_ms=round(p99, 2),
        mean_ms=round(mean_val, 2),
        std_ms=round(std_val, 2),
        min_ms=round(min_val, 2),
        max_ms=round(max_val, 2),
        throughput_rps=round(rps, 2),
        concurrency=concurrency
    )


def evaluate_atomic_predictions(
    ground_truths: List[AtomicGroundTruth],
    predictions: List[AtomicPrediction],
    category_to_domain_map: Optional[Dict[str, str]] = None,
    total_duration_seconds: Optional[float] = None,
    concurrency: int = 10
) -> AtomicEvalResult:
    """Evaluates atomic predictions against ground truths."""
    assert len(ground_truths) == len(predictions), "Mismatched ground truth and prediction counts"
    total_samples = len(ground_truths)
    if total_samples == 0:
        raise ValueError("Cannot evaluate empty test set")

    y_true = [gt.primary_category for gt in ground_truths]
    y_pred = [pred.predicted_category for pred in predictions]
    confidences = np.array([float(np.clip(pred.confidence, 0.0, 1.0)) for pred in predictions])
    latencies = [pred.latency_ms for pred in predictions]

    # Top-1 accuracy
    correct_mask = np.array([yt == yp for yt, yp in zip(y_true, y_pred)], dtype=float)
    top1_accuracy = float(np.mean(correct_mask))

    # Top-3 accuracy
    top3_correct = 0
    for gt, pred in zip(ground_truths, predictions):
        candidates = [pred.predicted_category]
        if pred.top_k:
            candidates.extend([item.category for item in pred.top_k])
        unique_candidates = []
        for c in candidates:
            if c not in unique_candidates:
                unique_candidates.append(c)
        if gt.primary_category in unique_candidates[:3]:
            top3_correct += 1
    top3_accuracy = float(top3_correct / total_samples)

    # Domain accuracy & domain-level breakdown
    domain_correct = 0
    domain_stats: Dict[str, Dict[str, int]] = {}

    for gt, pred in zip(ground_truths, predictions):
        gt_dom = gt.domain
        pred_dom = pred.predicted_domain
        if not pred_dom and category_to_domain_map:
            pred_dom = category_to_domain_map.get(pred.predicted_category, "unknown")

        if gt_dom not in domain_stats:
            domain_stats[gt_dom] = {"total": 0, "correct": 0, "cat_correct": 0}
        domain_stats[gt_dom]["total"] += 1

        if pred_dom == gt_dom:
            domain_correct += 1
            domain_stats[gt_dom]["correct"] += 1

        if pred.predicted_category == gt.primary_category:
            domain_stats[gt_dom]["cat_correct"] += 1

    domain_accuracy = float(domain_correct / total_samples)

    domain_breakdown = {}
    for dom, s in domain_stats.items():
        tot = s["total"]
        domain_breakdown[dom] = {
            "total_samples": float(tot),
            "domain_accuracy": round(s["correct"] / tot, 4) if tot > 0 else 0.0,
            "category_accuracy": round(s["cat_correct"] / tot, 4) if tot > 0 else 0.0,
        }

    # All unique labels for scikit-learn
    all_labels = sorted(list(set(y_true + y_pred)))
    macro_f1 = float(f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, labels=all_labels, average="micro", zero_division=0))

    # Per-category metrics
    per_cat = {}
    for label in sorted(list(set(y_true))):
        y_true_binary = [1 if y == label else 0 for y in y_true]
        y_pred_binary = [1 if y == label else 0 for y in y_pred]
        p = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
        r = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
        f = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
        per_cat[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": sum(y_true_binary)
        }

    # ECE
    ece = compute_ece(confidences, correct_mask, num_bins=10)

    # Confusion pairs
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    confusion_pairs = []
    for i, true_lbl in enumerate(all_labels):
        for j, pred_lbl in enumerate(all_labels):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append(
                    ConfusionPair(ground_truth=true_lbl, predicted=pred_lbl, count=int(cm[i, j]))
                )
    confusion_pairs.sort(key=lambda x: x.count, reverse=True)

    latency_stats = calculate_latency_stats(latencies, total_duration_seconds, concurrency=concurrency)
    laps = compute_laps(top1_accuracy, latency_stats.p95_ms)

    return AtomicEvalResult(
        total_samples=total_samples,
        top1_accuracy=round(top1_accuracy, 4),
        top3_accuracy=round(top3_accuracy, 4),
        macro_f1=round(macro_f1, 4),
        micro_f1=round(micro_f1, 4),
        ece=round(ece, 4),
        domain_accuracy=round(domain_accuracy, 4),
        laps_score=laps,
        confusion_pairs=confusion_pairs[:20],
        latency=latency_stats,
        per_category_metrics=per_cat,
        domain_breakdown=domain_breakdown
    )
