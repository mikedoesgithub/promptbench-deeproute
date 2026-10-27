"""End-to-end integration tests for PromptBench-DeepRoute v2.0 runner, profiler, and HTML reporter."""

import pytest
import asyncio
from pathlib import Path
from src.models.mock_adapter import MockModelAdapter
from src.harness.runner import BenchmarkRunner
from src.harness.reporter import BenchmarkReporter

WORKSPACE_DIR = Path(__file__).resolve().parent.parent


@pytest.mark.asyncio
async def test_runner_atomic_core_mode():
    adapter = MockModelAdapter(model_name="mock-test-oracle", oracle=True)
    runner = BenchmarkRunner(
        adapter=adapter,
        datasets_dir=WORKSPACE_DIR / "datasets",
        taxonomy_path=WORKSPACE_DIR / "configs" / "taxonomy.yaml",
        concurrency=10,
    )
    result = await runner.run_atomic_benchmark()
    assert result.total_samples >= 100
    assert result.top1_accuracy >= 0.99
    assert result.domain_accuracy >= 0.99
    assert result.laps_score > 0.0
    assert len(result.domain_breakdown) >= 9


@pytest.mark.asyncio
async def test_runner_multilabel_faceted_mode():
    adapter = MockModelAdapter(model_name="mock-test-oracle", oracle=True)
    runner = BenchmarkRunner(
        adapter=adapter,
        datasets_dir=WORKSPACE_DIR / "datasets",
        taxonomy_path=WORKSPACE_DIR / "configs" / "taxonomy.yaml",
        concurrency=10,
    )
    result = await runner.run_multilabel_benchmark()
    assert result.total_samples >= 100
    assert result.exact_match_ratio >= 0.99
    assert result.jaccard_index >= 0.99
    assert result.laps_score > 0.0


@pytest.mark.asyncio
async def test_runner_compound_dag_advanced_mode():
    adapter = MockModelAdapter(model_name="mock-test-oracle", oracle=True)
    runner = BenchmarkRunner(
        adapter=adapter,
        datasets_dir=WORKSPACE_DIR / "datasets",
        taxonomy_path=WORKSPACE_DIR / "configs" / "taxonomy.yaml",
        concurrency=10,
    )
    result = await runner.run_compound_benchmark()
    assert result.total_samples >= 100
    assert result.is_compound_accuracy >= 0.99
    assert result.task_boundary_f1 >= 0.99
    assert result.agent_allocation_accuracy >= 0.99
    assert result.dag_structural_validity == 1.0
    assert result.dag_topological_match_pct >= 0.99
    assert result.laps_score > 0.0


@pytest.mark.asyncio
async def test_runner_adversarial_stress_mode():
    adapter = MockModelAdapter(model_name="mock-test-oracle", oracle=True)
    runner = BenchmarkRunner(
        adapter=adapter,
        datasets_dir=WORKSPACE_DIR / "datasets",
        taxonomy_path=WORKSPACE_DIR / "configs" / "taxonomy.yaml",
        concurrency=10,
    )
    result = await runner.run_adversarial_benchmark()
    assert result.total_samples == 200
    assert result.overall_accuracy >= 0.95
    assert result.injection_detection_rate >= 0.95
    assert result.multilingual_accuracy >= 0.95
    assert result.distractor_robustness_rate >= 0.95
    assert result.laps_score > 0.0


@pytest.mark.asyncio
async def test_runner_concurrency_saturation_sweep():
    adapter = MockModelAdapter(model_name="mock-sweep", oracle=True)
    runner = BenchmarkRunner(
        adapter=adapter,
        datasets_dir=WORKSPACE_DIR / "datasets",
        taxonomy_path=WORKSPACE_DIR / "configs" / "taxonomy.yaml",
        concurrency=10,
        concurrency_sweep=[1, 5, 10],
    )
    result = await runner.run_atomic_benchmark()
    assert len(result.concurrency_sweep) == 3
    for pt in result.concurrency_sweep:
        assert pt.concurrency in [1, 5, 10]
        assert pt.throughput_rps > 0.0
        assert pt.p95_ms > 0.0
        assert pt.laps_score > 0.0


@pytest.mark.asyncio
async def test_reporter_html_dashboard_generation(tmp_path):
    adapter = MockModelAdapter(model_name="mock-html-test", oracle=True)
    runner = BenchmarkRunner(
        adapter=adapter,
        datasets_dir=WORKSPACE_DIR / "datasets",
        taxonomy_path=WORKSPACE_DIR / "configs" / "taxonomy.yaml",
        concurrency=5,
    )
    summary = await runner.run_suite(mode="all")
    reporter = BenchmarkReporter()

    md_file = tmp_path / "test_report.md"
    json_file = tmp_path / "test_report.json"
    html_file = tmp_path / "test_dashboard.html"

    reporter.export_markdown_report(summary, md_file)
    reporter.export_json(summary, json_file)
    reporter.export_html_dashboard(summary, html_file)

    assert md_file.exists()
    assert json_file.exists()
    assert html_file.exists()

    html_text = html_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "PromptBench-DeepRoute Dashboard" in html_text
    assert "radar-chart" in html_text
    assert "pareto-chart" in html_text
    assert "Diamond Pattern" in html_text
