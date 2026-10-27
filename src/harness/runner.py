"""Async benchmark evaluation runner with concurrency saturation profiling and LAPS tracking."""

from __future__ import annotations
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from src.schemas import (
    AtomicGroundTruth,
    AtomicPrediction,
    AtomicEvalResult,
    MultiLabelGroundTruth,
    MultiLabelPrediction,
    MultiLabelEvalResult,
    CompoundMultiGroundTruth,
    CompoundMultiPrediction,
    CompoundMultiEvalResult,
    AdversarialStressGroundTruth,
    AdversarialStressPrediction,
    AdversarialStressEvalResult,
    BenchmarkSummary,
    ConcurrencySweepPoint,
)
from src.models.base import BaseModelAdapter
from src.evaluators.atomic_metrics import evaluate_atomic_predictions, compute_laps, calculate_latency_stats
from src.evaluators.multilabel_metrics import evaluate_multilabel_predictions
from src.evaluators.dag_decomposition_metrics import evaluate_compound_multi_predictions
from src.evaluators.adversarial_metrics import evaluate_adversarial_predictions


class BenchmarkRunner:
    """Orchestrates async evaluation across modes with bounded concurrency and latency saturation sweeps."""

    def __init__(
        self,
        adapter: BaseModelAdapter,
        datasets_dir: Path,
        taxonomy_path: Path,
        concurrency: int = 10,
        concurrency_sweep: Optional[List[int]] = None,
    ):
        self.adapter = adapter
        self.datasets_dir = datasets_dir
        self.taxonomy_path = taxonomy_path
        self.concurrency = concurrency
        self.concurrency_sweep = concurrency_sweep or []
        self.semaphore = asyncio.Semaphore(concurrency)

        # Load taxonomy and subagents configurations
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            tax_data = yaml.safe_load(f)
        self.taxonomy_data = tax_data

        subagents_path = self.taxonomy_path.parent / "subagents.yaml"
        self.subagents_data = {}
        if subagents_path.exists():
            with open(subagents_path, "r", encoding="utf-8") as f:
                self.subagents_data = yaml.safe_load(f)

        # Pass authoritative taxonomy and subagents to adapter
        if hasattr(self.adapter, "set_configs"):
            self.adapter.set_configs(self.taxonomy_data, self.subagents_data)

        self.cat_to_domain = {}
        if "tier1_domains" in tax_data:
            for dom_key, dom_val in tax_data.get("tier1_domains", {}).items():
                for cat in dom_val.get("categories", []):
                    self.cat_to_domain[cat] = dom_key
        if "granular_categories" in tax_data:
            for cat, info in tax_data.get("granular_categories", {}).items():
                self.cat_to_domain[cat] = info.get("domain", "unknown")

    def _resolve_dataset_path(self, primary_filename: str, fallback_filename: str) -> Path:
        primary = self.datasets_dir / primary_filename
        if primary.exists():
            return primary
        return self.datasets_dir / fallback_filename

    async def _run_atomic_sample(self, gt: AtomicGroundTruth, sem: asyncio.Semaphore) -> AtomicPrediction:
        async with sem:
            try:
                return await self.adapter.predict_atomic(gt.prompt)
            except Exception as e:
                return AtomicPrediction(
                    predicted_category="error",
                    confidence=0.0,
                    latency_ms=0.0,
                    raw_output=f"Error: {str(e)}"
                )

    async def _run_multilabel_sample(self, gt: MultiLabelGroundTruth, sem: asyncio.Semaphore) -> MultiLabelPrediction:
        async with sem:
            try:
                return await self.adapter.predict_multilabel(gt.prompt)
            except Exception as e:
                return MultiLabelPrediction(
                    domains=[],
                    intents=[],
                    modalities=[],
                    confidence=0.0,
                    latency_ms=0.0,
                    raw_output=f"Error: {str(e)}"
                )

    async def _run_compound_sample(self, gt: CompoundMultiGroundTruth, sem: asyncio.Semaphore) -> CompoundMultiPrediction:
        async with sem:
            try:
                return await self.adapter.predict_compound(gt.prompt)
            except Exception as e:
                return CompoundMultiPrediction(
                    is_compound=False,
                    sub_tasks=[],
                    latency_ms=0.0,
                    raw_output=f"Error: {str(e)}"
                )

    async def _run_adversarial_sample(self, gt: AdversarialStressGroundTruth, sem: asyncio.Semaphore) -> AdversarialStressPrediction:
        async with sem:
            try:
                return await self.adapter.predict_adversarial(gt.prompt)
            except Exception as e:
                return AdversarialStressPrediction(
                    predicted_category="error",
                    predicted_domain="error",
                    predicted_safety_flag="error",
                    confidence=0.0,
                    latency_ms=0.0,
                    raw_output=f"Error: {str(e)}"
                )

    async def run_concurrency_sweep(self, gts: List[AtomicGroundTruth]) -> List[ConcurrencySweepPoint]:
        """Performs a concurrency saturation sweep across specified thread bounds."""
        sweep_points: List[ConcurrencySweepPoint] = []
        sample_subset = gts[:min(len(gts), 100)]  # Fast representative batch

        for c in self.concurrency_sweep:
            sem = asyncio.Semaphore(c)
            t0 = time.perf_counter()
            preds = await asyncio.gather(*(self._run_atomic_sample(gt, sem) for gt in sample_subset))
            duration = time.perf_counter() - t0

            latencies = [p.latency_ms for p in preds]
            stats = calculate_latency_stats(latencies, duration, concurrency=c)
            correct = sum(1 for gt, p in zip(sample_subset, preds) if gt.primary_category == p.predicted_category)
            acc = correct / len(sample_subset) if sample_subset else 0.0
            laps = compute_laps(acc, stats.p95_ms)

            sweep_points.append(ConcurrencySweepPoint(
                concurrency=c,
                p50_ms=stats.p50_ms,
                p95_ms=stats.p95_ms,
                p99_ms=stats.p99_ms,
                throughput_rps=stats.throughput_rps,
                laps_score=laps
            ))

        return sweep_points

    async def run_atomic_benchmark(self, progress: Optional[Progress] = None) -> AtomicEvalResult:
        """Runs Mode 1: Atomic Core Classification benchmark."""
        path = self._resolve_dataset_path("atomic_core.jsonl", "atomic_eval.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            gts = [AtomicGroundTruth(**json.loads(line)) for line in f if line.strip()]

        task_id = None
        if progress:
            task_id = progress.add_task(f"[cyan]Evaluating Mode 1 (Atomic, {len(gts)} samples)...", total=len(gts))

        t0 = time.perf_counter()
        async def wrapped(gt):
            res = await self._run_atomic_sample(gt, self.semaphore)
            if progress and task_id is not None:
                progress.advance(task_id, 1)
            return res

        preds = await asyncio.gather(*(wrapped(gt) for gt in gts))
        duration = time.perf_counter() - t0

        result = evaluate_atomic_predictions(
            ground_truths=gts,
            predictions=preds,
            category_to_domain_map=self.cat_to_domain,
            total_duration_seconds=duration,
            concurrency=self.concurrency
        )

        if self.concurrency_sweep:
            sweep_results = await self.run_concurrency_sweep(gts)
            result.concurrency_sweep = sweep_results

        return result

    async def run_multilabel_benchmark(self, progress: Optional[Progress] = None) -> MultiLabelEvalResult:
        """Runs Mode 2: Multi-Label Faceted Tagging benchmark."""
        path = self._resolve_dataset_path("multilabel_faceted.jsonl", "multilabel_eval.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            gts = [MultiLabelGroundTruth(**json.loads(line)) for line in f if line.strip()]

        task_id = None
        if progress:
            task_id = progress.add_task(f"[magenta]Evaluating Mode 2 (Multi-Label, {len(gts)} samples)...", total=len(gts))

        t0 = time.perf_counter()
        async def wrapped(gt):
            res = await self._run_multilabel_sample(gt, self.semaphore)
            if progress and task_id is not None:
                progress.advance(task_id, 1)
            return res

        preds = await asyncio.gather(*(wrapped(gt) for gt in gts))
        duration = time.perf_counter() - t0

        return evaluate_multilabel_predictions(
            ground_truths=gts,
            predictions=preds,
            total_duration_seconds=duration,
            concurrency=self.concurrency
        )

    async def run_compound_benchmark(self, progress: Optional[Progress] = None) -> CompoundMultiEvalResult:
        """Runs Mode 3: Compound DAG Advanced Decomposition benchmark."""
        path = self._resolve_dataset_path("compound_dag_advanced.jsonl", "compound_multi_eval.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            gts = [CompoundMultiGroundTruth(**json.loads(line)) for line in f if line.strip()]

        task_id = None
        if progress:
            task_id = progress.add_task(f"[green]Evaluating Mode 3 (Compound DAG, {len(gts)} samples)...", total=len(gts))

        t0 = time.perf_counter()
        async def wrapped(gt):
            res = await self._run_compound_sample(gt, self.semaphore)
            if progress and task_id is not None:
                progress.advance(task_id, 1)
            return res

        preds = await asyncio.gather(*(wrapped(gt) for gt in gts))
        duration = time.perf_counter() - t0

        return evaluate_compound_multi_predictions(
            ground_truths=gts,
            predictions=preds,
            total_duration_seconds=duration,
            concurrency=self.concurrency
        )

    async def run_adversarial_benchmark(self, progress: Optional[Progress] = None) -> AdversarialStressEvalResult:
        """Runs Split 4: Adversarial Stress & Robustness benchmark."""
        path = self.datasets_dir / "adversarial_stress.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            gts = [AdversarialStressGroundTruth(**json.loads(line)) for line in f if line.strip()]

        task_id = None
        if progress:
            task_id = progress.add_task(f"[red]Evaluating Split 4 (Adversarial Stress, {len(gts)} samples)...", total=len(gts))

        t0 = time.perf_counter()
        async def wrapped(gt):
            res = await self._run_adversarial_sample(gt, self.semaphore)
            if progress and task_id is not None:
                progress.advance(task_id, 1)
            return res

        preds = await asyncio.gather(*(wrapped(gt) for gt in gts))
        duration = time.perf_counter() - t0

        return evaluate_adversarial_predictions(
            ground_truths=gts,
            predictions=preds,
            total_duration_seconds=duration,
            concurrency=self.concurrency
        )

    async def run_suite(self, mode: str = "all") -> BenchmarkSummary:
        """Executes selected benchmark mode(s) and collects complete summary."""
        t_start = time.perf_counter()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        atomic_res = None
        multilabel_res = None
        compound_res = None
        adversarial_res = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        ) as progress:
            if mode in ["atomic", "all"]:
                atomic_res = await self.run_atomic_benchmark(progress)
            if mode in ["multilabel", "all"]:
                multilabel_res = await self.run_multilabel_benchmark(progress)
            if mode in ["multi", "all"]:
                compound_res = await self.run_compound_benchmark(progress)
            if mode in ["adversarial", "stress", "all"]:
                adversarial_res = await self.run_adversarial_benchmark(progress)

        total_runtime = time.perf_counter() - t_start

        # Composite overall LAPS score
        laps_scores = []
        if atomic_res:
            laps_scores.append(atomic_res.laps_score)
        if multilabel_res:
            laps_scores.append(multilabel_res.laps_score)
        if compound_res:
            laps_scores.append(compound_res.laps_score)
        if adversarial_res:
            laps_scores.append(adversarial_res.laps_score)

        overall_laps = round(float(sum(laps_scores) / len(laps_scores)), 2) if laps_scores else 0.0

        return BenchmarkSummary(
            model_name=self.adapter.model_name,
            timestamp=timestamp,
            total_runtime_seconds=round(total_runtime, 2),
            overall_laps_score=overall_laps,
            atomic_results=atomic_res,
            multilabel_results=multilabel_res,
            compound_results=compound_res,
            adversarial_results=adversarial_res
        )
