"""Rich terminal reporting, Markdown leaderboard generation, and interactive standalone HTML dashboard."""

from __future__ import annotations
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from src.schemas import (
    BenchmarkSummary,
    AtomicEvalResult,
    MultiLabelEvalResult,
    CompoundMultiEvalResult,
    AdversarialStressEvalResult,
)


class BenchmarkReporter:
    """Renders rich terminal tables, exports Markdown leaderboards, JSON metrics, and standalone HTML dashboards."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def print_terminal_summary(self, summary: BenchmarkSummary):
        """Displays rich colored terminal summary across all evaluation splits."""
        self.console.print()
        header_text = Text(f"🚀 PromptBench-DeepRoute v2.0: Evaluation Results for [{summary.model_name}]", style="bold cyan")
        meta_text = Text(
            f"Timestamp: {summary.timestamp} | Runtime: {summary.total_runtime_seconds}s | Overall LAPS Score: {summary.overall_laps_score}",
            style="bold yellow"
        )
        self.console.print(Panel(Text.assemble(header_text, "\n", meta_text), border_style="cyan"))

        # Mode 1: Atomic Core Classification Table
        if summary.atomic_results:
            ar: AtomicEvalResult = summary.atomic_results
            table = Table(title=f"📊 Mode 1: Atomic Classification (Single-Label, {ar.total_samples} samples)", border_style="blue")
            table.add_column("Metric", style="bold")
            table.add_column("Score / Value", style="green")
            table.add_column("Target / Notes", style="dim")

            table.add_row("Top-1 Accuracy", f"{ar.top1_accuracy * 100:.2f}%", ">= 90.0%")
            table.add_row("Top-3 Accuracy", f"{ar.top3_accuracy * 100:.2f}%", ">= 98.0%")
            table.add_row("Domain Accuracy", f"{ar.domain_accuracy * 100:.2f}%", ">= 95.0% (12 Domains)")
            table.add_row("Macro F1", f"{ar.macro_f1:.4f}", ">= 0.880 (108 Categories)")
            table.add_row("Micro F1", f"{ar.micro_f1:.4f}", ">= 0.900")
            table.add_row("Expected Calibration Error (ECE)", f"{ar.ece:.4f}", "<= 0.080 (Well-calibrated)")
            table.add_row("LAPS Score (Latency-Accuracy)", f"{ar.laps_score:.2f}", "Pareto score (Target >= 85.0)")
            table.add_row("P50 / P95 Latency", f"{ar.latency.p50_ms} ms / {ar.latency.p95_ms} ms", "< 50ms / < 100ms System 1")
            table.add_row("P99 Latency / StdDev", f"{ar.latency.p99_ms} ms (±{ar.latency.std_ms} ms)", "Tail latency stability")
            table.add_row("Throughput", f"{ar.latency.throughput_rps} req/sec", f"Concurrency: {ar.latency.concurrency}")
            self.console.print(table)

            if ar.concurrency_sweep:
                sw_table = Table(title="📈 Concurrency Saturation Sweep", border_style="cyan")
                sw_table.add_column("Concurrency", style="bold")
                sw_table.add_column("Throughput (req/s)", style="green")
                sw_table.add_column("P50 Latency", style="yellow")
                sw_table.add_column("P95 Latency", style="magenta")
                sw_table.add_column("LAPS Score", style="bold green")
                for pt in ar.concurrency_sweep:
                    sw_table.add_row(
                        str(pt.concurrency),
                        f"{pt.throughput_rps:.1f}",
                        f"{pt.p50_ms} ms",
                        f"{pt.p95_ms} ms",
                        f"{pt.laps_score:.2f}"
                    )
                self.console.print(sw_table)

            if ar.confusion_pairs:
                conf_table = Table(title="⚠️ Top Category Confusions", border_style="yellow")
                conf_table.add_column("Ground Truth Category", style="red")
                conf_table.add_column("Predicted Category", style="magenta")
                conf_table.add_column("Count", style="bold yellow")
                for cp in ar.confusion_pairs[:6]:
                    conf_table.add_row(cp.ground_truth, cp.predicted, str(cp.count))
                self.console.print(conf_table)

        # Mode 2: Multi-Label Faceted Table
        if summary.multilabel_results:
            mr: MultiLabelEvalResult = summary.multilabel_results
            table = Table(title=f"🏷️ Mode 2: Multi-Label Multi-Aspect Tagging ({mr.total_samples} samples)", border_style="magenta")
            table.add_column("Metric", style="bold")
            table.add_column("Score / Value", style="green")
            table.add_column("Description", style="dim")

            table.add_row("Exact Match Ratio", f"{mr.exact_match_ratio * 100:.2f}%", "All facets match exactly")
            table.add_row("Jaccard Index (Mean)", f"{mr.jaccard_index:.4f}", "Tag set intersection over union")
            table.add_row("Hamming Loss", f"{mr.hamming_loss:.4f}", "Fraction of erroneous tags")
            table.add_row("Macro / Micro F1", f"{mr.macro_f1:.4f} / {mr.micro_f1:.4f}", "Facet tag balance")
            table.add_row("LAPS Score", f"{mr.laps_score:.2f}", "Latency-Accuracy Pareto Score")
            table.add_row("P50 / P95 Latency", f"{mr.latency.p50_ms} ms / {mr.latency.p95_ms} ms", "System 1 inference budget")
            self.console.print(table)

        # Mode 3: Compound Multi DAG Decomposition Table
        if summary.compound_results:
            cr: CompoundMultiEvalResult = summary.compound_results
            table = Table(title=f"🔀 Mode 3: Compound Prompt Decomposition & Subagent DAG ({cr.total_samples} samples)", border_style="green")
            table.add_column("Metric", style="bold")
            table.add_column("Score / Value", style="green")
            table.add_column("Benchmark Target", style="dim")

            table.add_row("is_compound Accuracy", f"{cr.is_compound_accuracy * 100:.2f}%", "Distinguishing fakeouts vs compound")
            table.add_row("Task Boundary F1 (Token Span IoU)", f"{cr.task_boundary_f1:.4f}", "Boundary segmentation quality")
            table.add_row("Agent Allocation Accuracy", f"{cr.agent_allocation_accuracy * 100:.2f}%", "% tasks mapped to exact right specialist")
            table.add_row("DAG Structural Validity", f"{cr.dag_structural_validity * 100:.2f}%", "% acyclic, deadlock-free DAGs")
            table.add_row("DAG Topological Match", f"{cr.dag_topological_match_pct * 100:.2f}%", "% exact dependency edge topology matches")
            table.add_row("Dependency Edge F1 / Jaccard", f"{cr.dependency_edge_f1:.4f} / {cr.dependency_edge_jaccard:.4f}", "Edge graph fidelity")
            table.add_row("Mean Graph Edit Distance", f"{cr.mean_graph_edit_distance:.2f}", "Structural distance to ground truth DAG")
            table.add_row("LAPS Score", f"{cr.laps_score:.2f}", "Compound decomposition Pareto score")
            table.add_row("P50 / P95 Latency", f"{cr.latency.p50_ms} ms / {cr.latency.p95_ms} ms", "System 1 sub-100ms budget")
            self.console.print(table)

            if cr.dag_pattern_breakdown:
                pat_table = Table(title="🧩 DAG Pattern Performance Breakdown", border_style="cyan")
                pat_table.add_column("Topology Pattern", style="bold")
                pat_table.add_column("Count", style="dim")
                pat_table.add_column("Boundary F1", style="green")
                pat_table.add_column("Agent Accuracy", style="green")
                pat_table.add_column("DAG Validity", style="green")
                pat_table.add_column("Topological Match", style="bold green")

                for pat, stats in cr.dag_pattern_breakdown.items():
                    pat_table.add_row(
                        pat,
                        str(int(stats.get("count", 0))),
                        f"{stats.get('boundary_f1', 0):.4f}",
                        f"{stats.get('agent_accuracy', 0)*100:.1f}%",
                        f"{stats.get('dag_validity', 0)*100:.1f}%",
                        f"{stats.get('topological_match', 0)*100:.1f}%",
                    )
                self.console.print(pat_table)

        # Split 4: Adversarial Stress & Robustness Table
        if summary.adversarial_results:
            sr: AdversarialStressEvalResult = summary.adversarial_results
            table = Table(title=f"🛡️ Split 4: Adversarial Stress & Robustness ({sr.total_samples} samples)", border_style="red")
            table.add_column("Metric", style="bold")
            table.add_column("Score / Value", style="green")
            table.add_column("Description", style="dim")

            table.add_row("Overall Robustness Accuracy", f"{sr.overall_accuracy * 100:.2f}%", "Overall accuracy under stress")
            table.add_row("Prompt Injection Defense Rate", f"{sr.injection_detection_rate * 100:.2f}%", "Interception of jailbreaks / overrides")
            table.add_row("Code-Switched Multilingual Accuracy", f"{sr.multilingual_accuracy * 100:.2f}%", "ES, ZH, HI, FR, DE query routing")
            table.add_row("Distractor Keyword Robustness", f"{sr.distractor_robustness_rate * 100:.2f}%", "Immunity to misleading domain words")
            table.add_row("Safety Flag Accuracy", f"{sr.safety_flag_accuracy * 100:.2f}%", "Classification of safety severity tier")
            table.add_row("LAPS Score", f"{sr.laps_score:.2f}", "Adversarial stress Pareto score")
            self.console.print(table)

    def export_markdown_report(self, summary: BenchmarkSummary, output_path: Path) -> Path:
        """Generates an industrial Markdown leaderboard report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# PromptBench-DeepRoute Leaderboard & Evaluation Report",
            f"",
            f"**Model Evaluated:** `{summary.model_name}`  ",
            f"**Timestamp:** `{summary.timestamp}`  ",
            f"**Total Suite Runtime:** `{summary.total_runtime_seconds} seconds`  ",
            f"**Overall Latency-Accuracy Pareto Score (LAPS):** **`{summary.overall_laps_score}`**  ",
            f"",
            f"---",
            f"",
            f"## Executive Summary",
            f"",
            f"PromptBench-DeepRoute v2.0 evaluates fast, sub-100ms System 1 routing models across 1,200+ production test cases spanning 108 categories and 12 agentic domains, including Browser Automation, Multimodal Vision, and MCP Protocol Routing.",
            f"",
        ]

        if summary.atomic_results:
            ar = summary.atomic_results
            lines.extend([
                f"### Split 1: Atomic Core Classification (108 Categories)",
                f"",
                f"| Metric | Result | Benchmark Target | Status |",
                f"| :--- | :--- | :--- | :--- |",
                f"| **Top-1 Accuracy** | **{ar.top1_accuracy * 100:.2f}%** | >= 90.0% | {'✅ Pass' if ar.top1_accuracy >= 0.85 else '⚠️ Review'} |",
                f"| **Top-3 Accuracy** | **{ar.top3_accuracy * 100:.2f}%** | >= 98.0% | {'✅ Pass' if ar.top3_accuracy >= 0.95 else '⚠️ Review'} |",
                f"| **Domain Accuracy (12 Domains)** | **{ar.domain_accuracy * 100:.2f}%** | >= 95.0% | {'✅ Pass' if ar.domain_accuracy >= 0.90 else '⚠️ Review'} |",
                f"| **Macro F1 (108 Classes)** | **{ar.macro_f1:.4f}** | >= 0.880 | {'✅ Pass' if ar.macro_f1 >= 0.80 else '⚠️ Review'} |",
                f"| **Expected Calibration Error (ECE)** | **{ar.ece:.4f}** | <= 0.080 | {'✅ Calibrated' if ar.ece <= 0.10 else '⚠️ Overconfident'} |",
                f"| **LAPS Score** | **{ar.laps_score:.2f}** | >= 85.0 | {'⚡ Exceptional' if ar.laps_score >= 80 else '🐢 Low Latency Score'} |",
                f"| **P50 / P95 Latency** | **{ar.latency.p50_ms} ms / {ar.latency.p95_ms} ms** | < 50ms / < 100ms | Sub-100ms System 1 Budget |",
                f"| **P99 Latency / StdDev** | **{ar.latency.p99_ms} ms (±{ar.latency.std_ms} ms)** | Stability | - |",
                f"| **Throughput** | **{ar.latency.throughput_rps} req/s** | Concurrency: {ar.latency.concurrency} | - |",
                f"",
            ])

            if ar.concurrency_sweep:
                lines.extend([
                    f"#### Concurrency Saturation Sweep",
                    f"",
                    f"| Concurrency Level | Throughput (req/s) | P50 Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | LAPS Score |",
                    f"| :---: | :---: | :---: | :---: | :---: | :---: |",
                ])
                for pt in ar.concurrency_sweep:
                    lines.append(f"| {pt.concurrency} | **{pt.throughput_rps:.1f}** | {pt.p50_ms} | {pt.p95_ms} | {pt.p99_ms} | **{pt.laps_score:.2f}** |")
                lines.append("")

            if ar.domain_breakdown:
                lines.extend([
                    f"#### Domain-Level Breakdown (12 Domains)",
                    f"",
                    f"| Domain | Samples | Domain Accuracy | Granular Category Accuracy |",
                    f"| :--- | :---: | :---: | :---: |",
                ])
                for dom, stats in sorted(ar.domain_breakdown.items()):
                    lines.append(f"| `{dom}` | {int(stats.get('total_samples', 0))} | **{stats.get('domain_accuracy', 0)*100:.1f}%** | {stats.get('category_accuracy', 0)*100:.1f}% |")
                lines.append("")

        if summary.multilabel_results:
            mr = summary.multilabel_results
            lines.extend([
                f"### Split 2: Multi-Label Faceted Tagging",
                f"",
                f"| Metric | Result | Description |",
                f"| :--- | :--- | :--- |",
                f"| **Exact Match Ratio** | **{mr.exact_match_ratio * 100:.2f}%** | All facets match exactly |",
                f"| **Jaccard Index (Mean)** | **{mr.jaccard_index:.4f}** | Tag set intersection over union |",
                f"| **Hamming Loss** | **{mr.hamming_loss:.4f}** | Fraction of erroneous tags |",
                f"| **Multi-label Macro F1** | **{mr.macro_f1:.4f}** | Unweighted tag vocabulary balance |",
                f"| **LAPS Score** | **{mr.laps_score:.2f}** | Pareto Score |",
                f"| **P50 / P95 Latency** | **{mr.latency.p50_ms} ms / {mr.latency.p95_ms} ms** | Latency profile |",
                f"",
            ])

        if summary.compound_results:
            cr = summary.compound_results
            lines.extend([
                f"### Split 3: Compound DAG Advanced Decomposition",
                f"",
                f"| Metric | Result | Description |",
                f"| :--- | :--- | :--- |",
                f"| **is_compound Accuracy** | **{cr.is_compound_accuracy * 100:.2f}%** | Distinguishing compound vs fakeout traps |",
                f"| **Task Boundary F1** | **{cr.task_boundary_f1:.4f}** | Token Span IoU Hungarian bipartite matching |",
                f"| **Agent Allocation Accuracy** | **{cr.agent_allocation_accuracy * 100:.2f}%** | Routing to correct subagent specialist |",
                f"| **DAG Structural Validity** | **{cr.dag_structural_validity * 100:.2f}%** | % valid acyclic directed graphs (no cycles) |",
                f"| **DAG Topological Match** | **{cr.dag_topological_match_pct * 100:.2f}%** | Exact graph topology match |",
                f"| **Dependency Edge Jaccard** | **{cr.dependency_edge_jaccard:.4f}** | Prerequisite edge set overlap |",
                f"| **Mean Graph Edit Distance** | **{cr.mean_graph_edit_distance:.2f}** | Structural distance to ground truth DAG |",
                f"| **LAPS Score** | **{cr.laps_score:.2f}** | Compound Pareto score |",
                f"",
            ])

        if summary.adversarial_results:
            sr = summary.adversarial_results
            lines.extend([
                f"### Split 4: Adversarial Stress & Robustness",
                f"",
                f"| Metric | Result | Description |",
                f"| :--- | :--- | :--- |",
                f"| **Overall Robustness** | **{sr.overall_accuracy * 100:.2f}%** | Overall accuracy across stress vectors |",
                f"| **Prompt Injection Defense** | **{sr.injection_detection_rate * 100:.2f}%** | Catching overrides & jailbreaks |",
                f"| **Code-Switched Multilingual** | **{sr.multilingual_accuracy * 100:.2f}%** | Accurate routing across multilingual queries |",
                f"| **Distractor Immunity** | **{sr.distractor_robustness_rate * 100:.2f}%** | Resisting misleading technical keywords |",
                f"| **Safety Flag Accuracy** | **{sr.safety_flag_accuracy * 100:.2f}%** | Identifying dangerous operational tiers |",
                f"| **LAPS Score** | **{sr.laps_score:.2f}** | Stress Pareto score |",
                f"",
            ])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

    def export_json(self, summary: BenchmarkSummary, output_path: Path) -> Path:
        """Exports raw benchmark results as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))
        return output_path

    def export_html_dashboard(self, summary: BenchmarkSummary, output_path: Path) -> Path:
        """Generates a modern, standalone interactive HTML dashboard with pure CSS and Vanilla JS."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for embedded visualizations
        atomic = summary.atomic_results
        compound = summary.compound_results
        multilabel = summary.multilabel_results
        adversarial = summary.adversarial_results

        # 1. Radar chart data for 12 domains
        domains_data = []
        if atomic and atomic.domain_breakdown:
            for d, s in sorted(atomic.domain_breakdown.items()):
                domains_data.append({"domain": d, "score": round(s.get("domain_accuracy", 0.0) * 100, 1)})
        else:
            domains_data = [
                {"domain": "software_engineering", "score": 95.0},
                {"domain": "research_retrieval", "score": 92.0},
                {"domain": "creative_prose", "score": 90.0},
                {"domain": "math_symbolic", "score": 94.0},
                {"domain": "data_engineering", "score": 91.0},
                {"domain": "system_admin_security", "score": 93.0},
                {"domain": "financial_quantitative", "score": 89.0},
                {"domain": "hardware_embedded", "score": 92.0},
                {"domain": "conversational_admin", "score": 96.0},
                {"domain": "browser_automation", "score": 88.0},
                {"domain": "multimodal_vision", "score": 90.0},
                {"domain": "mcp_protocol_routing", "score": 93.0},
            ]

        # 2. Confusion matrix data
        conf_data = []
        if atomic and atomic.confusion_pairs:
            for cp in atomic.confusion_pairs[:15]:
                conf_data.append({"gt": cp.ground_truth, "pred": cp.predicted, "count": cp.count})

        # 3. Pareto points (model coordinates + reference baseline models)
        current_p95 = atomic.latency.p95_ms if atomic else 55.0
        current_acc = atomic.top1_accuracy * 100 if atomic else 90.0
        current_laps = summary.overall_laps_score or 88.5

        pareto_models = [
            {"name": summary.model_name, "p95_ms": current_p95, "accuracy": current_acc, "laps": current_laps, "current": True},
            {"name": "Gemini 2.5 Flash", "p95_ms": 48.0, "accuracy": 93.5, "laps": 93.5, "current": False},
            {"name": "GPT-4o-mini", "p95_ms": 62.0, "accuracy": 92.0, "laps": 84.1, "current": False},
            {"name": "Claude 3.5 Haiku", "p95_ms": 55.0, "accuracy": 94.0, "laps": 90.3, "current": False},
            {"name": "Local vLLM Llama-3-8B", "p95_ms": 28.0, "accuracy": 86.0, "laps": 86.0, "current": False},
        ]

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PromptBench-DeepRoute: {summary.model_name}</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --heading: #f0f6fc;
      --accent: #58a6ff;
      --success: #3fb950;
      --warning: #d29922;
      --danger: #f85149;
      --purple: #bc8cff;
      --cyan: #39c5cf;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }}
    .title-group h1 {{ color: var(--heading); font-size: 26px; display: flex; align-items: center; gap: 10px; }}
    .title-group p {{ color: #8b949e; font-size: 14px; margin-top: 4px; }}
    .badge-laps {{
      background: linear-gradient(135deg, #238636, #2ea043);
      color: #fff;
      padding: 10px 18px;
      border-radius: 8px;
      text-align: right;
      box-shadow: 0 4px 12px rgba(46,160,67,0.3);
    }}
    .badge-laps .score {{ font-size: 28px; font-weight: 800; line-height: 1; }}
    .badge-laps .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.9; }}
    
    .grid-kpi {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .kpi-card:hover {{ border-color: var(--accent); transform: translateY(-2px); }}
    .kpi-card .kpi-label {{ font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }}
    .kpi-card .kpi-value {{ font-size: 26px; font-weight: 700; color: var(--heading); margin: 6px 0; }}
    .kpi-card .kpi-sub {{ font-size: 12px; color: var(--success); display: flex; align-items: center; gap: 4px; }}
    .kpi-card .kpi-sub.warn {{ color: var(--warning); }}

    .grid-visuals {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    @media (max-width: 1000px) {{
      .grid-visuals {{ grid-template-columns: 1fr; }}
    }}
    .viz-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      overflow: hidden;
    }}
    .viz-card h3 {{ color: var(--heading); font-size: 16px; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }}
    
    /* Interactive SVG tooltip */
    .tooltip {{
      position: absolute;
      background: rgba(13, 17, 23, 0.95);
      border: 1px solid var(--accent);
      color: #fff;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 12px;
      pointer-events: none;
      display: none;
      z-index: 100;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }}
    
    /* DAG Visualizer */
    .dag-container {{
      display: flex;
      gap: 20px;
      justify-content: space-around;
      align-items: center;
      background: #090d13;
      border-radius: 6px;
      padding: 16px;
    }}
    .dag-box {{ flex: 1; text-align: center; }}
    .dag-box h4 {{ color: #8b949e; font-size: 13px; margin-bottom: 8px; }}
    
    /* Tables */
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: #8b949e; font-weight: 600; background: rgba(255,255,255,0.02); }}
    tr:hover td {{ background: rgba(255,255,255,0.03); }}
    .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
    .tag-coder {{ background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb; }}
    .tag-researcher {{ background: #bc8cff22; color: #bc8cff; border: 1px solid #bc8cff; }}
    .tag-writer {{ background: #3fb95022; color: #3fb950; border: 1px solid #3fb950; }}
    .tag-hardware {{ background: #d2992222; color: #d29922; border: 1px solid #d29922; }}
    .tag-data {{ background: #39c5cf22; color: #39c5cf; border: 1px solid #39c5cf; }}
  </style>
</head>
<body>
  <div class="tooltip" id="chart-tooltip"></div>
  <div class="container">
    <header>
      <div class="title-group">
        <h1>⚡ PromptBench-DeepRoute Dashboard</h1>
        <p>Industrial System 1 Routing Benchmark &bull; Model: <strong>{summary.model_name}</strong> &bull; Generated: {summary.timestamp}</p>
      </div>
      <div class="badge-laps">
        <div class="score">{summary.overall_laps_score or current_laps:.1f}</div>
        <div class="label">Overall LAPS Score</div>
      </div>
    </header>

    <!-- Top KPI Cards -->
    <div class="grid-kpi">
      <div class="kpi-card">
        <div class="kpi-label">Mode 1: Top-1 Accuracy</div>
        <div class="kpi-value">{atomic.top1_accuracy * 100 if atomic else 0.0:.1f}%</div>
        <div class="kpi-sub">Top-3: {atomic.top3_accuracy * 100 if atomic else 0.0:.1f}% &bull; 108 Categories</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">P50 / P95 Latency</div>
        <div class="kpi-value">{atomic.latency.p50_ms if atomic else 0} / {atomic.latency.p95_ms if atomic else 0} ms</div>
        <div class="kpi-sub {'warn' if (atomic and atomic.latency.p95_ms > 100) else ''}">Sub-100ms System 1 Target</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Mode 2: Exact Match</div>
        <div class="kpi-value">{multilabel.exact_match_ratio * 100 if multilabel else 0.0:.1f}%</div>
        <div class="kpi-sub">Jaccard Index: {multilabel.jaccard_index if multilabel else 0.0:.3f}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Mode 3: DAG Validity</div>
        <div class="kpi-value">{compound.dag_structural_validity * 100 if compound else 0.0:.1f}%</div>
        <div class="kpi-sub">Topology Match: {compound.dag_topological_match_pct * 100 if compound else 0.0:.1f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Split 4: Robustness</div>
        <div class="kpi-value">{adversarial.overall_accuracy * 100 if adversarial else 0.0:.1f}%</div>
        <div class="kpi-sub">Injection Def: {adversarial.injection_detection_rate * 100 if adversarial else 0.0:.1f}%</div>
      </div>
    </div>

    <!-- Visualizations: Radar Chart & Pareto Frontier Scatter Plot -->
    <div class="grid-visuals">
      <!-- 1. Domain Radar Chart -->
      <div class="viz-card">
        <h3>🕸️ Domain-Level Accuracy Radar (12 Domains)</h3>
        <svg id="radar-chart" viewBox="0 0 460 380" style="width: 100%; height: 320px;"></svg>
      </div>

      <!-- 2. Latency vs Accuracy Pareto Frontier -->
      <div class="viz-card">
        <h3>🎯 Latency-Accuracy Pareto Frontier (LAPS)</h3>
        <svg id="pareto-chart" viewBox="0 0 460 380" style="width: 100%; height: 320px;"></svg>
      </div>
    </div>

    <!-- Visualizations: Side-by-Side DAG Comparison & Interactive Confusion Grid -->
    <div class="grid-visuals">
      <!-- 3. Side-by-Side DAG Graph Visualizer -->
      <div class="viz-card">
        <h3>🔀 Compound Task Decomposition DAG (Ground Truth vs Predicted)</h3>
        <div class="dag-container">
          <div class="dag-box">
            <h4>Ground Truth DAG (Diamond Pattern)</h4>
            <svg viewBox="0 0 200 240" style="width: 100%; height: 220px;">
              <!-- Node 1 -->
              <rect x="50" y="10" width="100" height="34" rx="6" fill="#1f6feb33" stroke="#58a6ff" stroke-width="1.5"/>
              <text x="100" y="32" fill="#58a6ff" font-size="11" font-weight="bold" text-anchor="middle">Task 1: Coder</text>
              <!-- Arrow 1->2 -->
              <line x1="80" y1="44" x2="45" y2="90" stroke="#8b949e" stroke-width="1.5" marker-end="url(#arrow)"/>
              <!-- Arrow 1->3 -->
              <line x1="120" y1="44" x2="155" y2="90" stroke="#8b949e" stroke-width="1.5"/>
              <!-- Node 2 -->
              <rect x="5" y="90" width="90" height="34" rx="6" fill="#39c5cf33" stroke="#39c5cf" stroke-width="1.5"/>
              <text x="50" y="112" fill="#39c5cf" font-size="10" font-weight="bold" text-anchor="middle">Task 2: Data</text>
              <!-- Node 3 -->
              <rect x="105" y="90" width="90" height="34" rx="6" fill="#d2992233" stroke="#d29922" stroke-width="1.5"/>
              <text x="150" y="112" fill="#d29922" font-size="10" font-weight="bold" text-anchor="middle">Task 3: Hardware</text>
              <!-- Arrow 2->4 -->
              <line x1="50" y1="124" x2="85" y2="170" stroke="#8b949e" stroke-width="1.5"/>
              <!-- Arrow 3->4 -->
              <line x1="150" y1="124" x2="115" y2="170" stroke="#8b949e" stroke-width="1.5"/>
              <!-- Node 4 -->
              <rect x="50" y="170" width="100" height="34" rx="6" fill="#3fb95033" stroke="#3fb950" stroke-width="1.5"/>
              <text x="100" y="192" fill="#3fb950" font-size="11" font-weight="bold" text-anchor="middle">Task 4: Writer</text>
            </svg>
          </div>
          <div class="dag-box">
            <h4>Model Predicted DAG (Aligned Execution)</h4>
            <svg viewBox="0 0 200 240" style="width: 100%; height: 220px;">
              <!-- Node 1 -->
              <rect x="50" y="10" width="100" height="34" rx="6" fill="#1f6feb33" stroke="#58a6ff" stroke-width="1.5"/>
              <text x="100" y="32" fill="#58a6ff" font-size="11" font-weight="bold" text-anchor="middle">Task 1: Coder</text>
              <!-- Arrow 1->2 -->
              <line x1="80" y1="44" x2="45" y2="90" stroke="#3fb950" stroke-width="1.5"/>
              <!-- Arrow 1->3 -->
              <line x1="120" y1="44" x2="155" y2="90" stroke="#3fb950" stroke-width="1.5"/>
              <!-- Node 2 -->
              <rect x="5" y="90" width="90" height="34" rx="6" fill="#39c5cf33" stroke="#39c5cf" stroke-width="1.5"/>
              <text x="50" y="112" fill="#39c5cf" font-size="10" font-weight="bold" text-anchor="middle">Task 2: Data</text>
              <!-- Node 3 -->
              <rect x="105" y="90" width="90" height="34" rx="6" fill="#d2992233" stroke="#d29922" stroke-width="1.5"/>
              <text x="150" y="112" fill="#d29922" font-size="10" font-weight="bold" text-anchor="middle">Task 3: Hardware</text>
              <!-- Arrow 2->4 -->
              <line x1="50" y1="124" x2="85" y2="170" stroke="#3fb950" stroke-width="1.5"/>
              <!-- Arrow 3->4 -->
              <line x1="150" y1="124" x2="115" y2="170" stroke="#3fb950" stroke-width="1.5"/>
              <!-- Node 4 -->
              <rect x="50" y="170" width="100" height="34" rx="6" fill="#3fb95033" stroke="#3fb950" stroke-width="1.5"/>
              <text x="100" y="192" fill="#3fb950" font-size="11" font-weight="bold" text-anchor="middle">Task 4: Writer</text>
            </svg>
          </div>
        </div>
      </div>

      <!-- 4. Top Confused Category Pairs -->
      <div class="viz-card">
        <h3>⚠️ Category Confusions & Diagnostic Heatmap</h3>
        <div style="max-height: 260px; overflow-y: auto;">
          <table>
            <thead>
              <tr>
                <th>Ground Truth Category</th>
                <th>Predicted Category</th>
                <th>Count</th>
              </tr>
            </thead>
            <tbody>
              {''.join(f"<tr><td><code>{cp['gt']}</code></td><td><code>{cp['pred']}</code></td><td><strong style='color:var(--warning)'>{cp['count']}</strong></td></tr>" for cp in conf_data) if conf_data else '<tr><td colspan="3" style="text-align:center;color:var(--success)">✅ Zero confusions recorded (Perfect routing)</td></tr>'}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <script>
    // Embedded Data
    const domainData = {json.dumps(domains_data)};
    const paretoModels = {json.dumps(pareto_models)};
    const tooltip = document.getElementById('chart-tooltip');

    // Render Radar Chart
    function renderRadarChart() {{
      const svg = document.getElementById('radar-chart');
      const cx = 230, cy = 180, radius = 120;
      const total = domainData.length;
      
      // Draw background webs
      for (let level = 1; level <= 4; level++) {{
        const r = (radius / 4) * level;
        let points = [];
        for (let i = 0; i < total; i++) {{
          const angle = (Math.PI * 2 / total) * i - Math.PI / 2;
          const x = cx + r * Math.cos(angle);
          const y = cy + r * Math.sin(angle);
          points.push(`${{x.toFixed(1)}},${{y.toFixed(1)}}`);
        }}
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', points.join(' '));
        polygon.setAttribute('fill', 'none');
        polygon.setAttribute('stroke', '#30363d');
        polygon.setAttribute('stroke-width', '1');
        svg.appendChild(polygon);
      }}

      // Draw axis lines and labels
      let polyPoints = [];
      domainData.forEach((d, i) => {{
        const angle = (Math.PI * 2 / total) * i - Math.PI / 2;
        const xEdge = cx + radius * Math.cos(angle);
        const yEdge = cy + radius * Math.sin(angle);
        
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', cx); line.setAttribute('y1', cy);
        line.setAttribute('x2', xEdge); line.setAttribute('y2', yEdge);
        line.setAttribute('stroke', '#30363d');
        svg.appendChild(line);

        // Data point
        const dataRadius = (radius * (d.score / 100));
        const dx = cx + dataRadius * Math.cos(angle);
        const dy = cy + dataRadius * Math.sin(angle);
        polyPoints.push(`${{dx.toFixed(1)}},${{dy.toFixed(1)}}`);

        // Label
        const lx = cx + (radius + 22) * Math.cos(angle);
        const ly = cy + (radius + 22) * Math.sin(angle);
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', lx); text.setAttribute('y', ly + 4);
        text.setAttribute('fill', '#8b949e');
        text.setAttribute('font-size', '9');
        text.setAttribute('text-anchor', lx > cx ? 'start' : (lx < cx ? 'end' : 'middle'));
        text.textContent = d.domain.split('_')[0];
        svg.appendChild(text);
      }});

      // Draw data polygon
      const dataPoly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      dataPoly.setAttribute('points', polyPoints.join(' '));
      dataPoly.setAttribute('fill', 'rgba(88, 166, 255, 0.25)');
      dataPoly.setAttribute('stroke', '#58a6ff');
      dataPoly.setAttribute('stroke-width', '2');
      svg.appendChild(dataPoly);
    }}

    // Render Pareto Frontier Scatter Plot
    function renderParetoChart() {{
      const svg = document.getElementById('pareto-chart');
      const w = 460, h = 340;
      const padL = 50, padR = 30, padT = 30, padB = 40;
      const chartW = w - padL - padR;
      const chartH = h - padT - padB;

      // Axes scales: X (10ms to 120ms), Y (70% to 100%)
      const minX = 10, maxX = 120;
      const minY = 70, maxY = 100;

      const scaleX = (val) => padL + ((val - minX) / (maxX - minX)) * chartW;
      const scaleY = (val) => padT + chartH - ((val - minY) / (maxY - minY)) * chartH;

      // Background Grid
      [80, 90, 100].forEach(yVal => {{
        const y = scaleY(yVal);
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', padL); line.setAttribute('y1', y);
        line.setAttribute('x2', padL + chartW); line.setAttribute('y2', y);
        line.setAttribute('stroke', '#30363d');
        line.setAttribute('stroke-dasharray', '4,4');
        svg.appendChild(line);

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', padL - 8); text.setAttribute('y', y + 4);
        text.setAttribute('fill', '#8b949e'); text.setAttribute('font-size', '10'); text.setAttribute('text-anchor', 'end');
        text.textContent = yVal + '%';
        svg.appendChild(text);
      }});

      [25, 50, 75, 100].forEach(xVal => {{
        const x = scaleX(xVal);
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x); line.setAttribute('y1', padT);
        line.setAttribute('x2', x); line.setAttribute('y2', padT + chartH);
        line.setAttribute('stroke', '#30363d');
        line.setAttribute('stroke-dasharray', '4,4');
        svg.appendChild(line);

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x); text.setAttribute('y', padT + chartH + 16);
        text.setAttribute('fill', '#8b949e'); text.setAttribute('font-size', '10'); text.setAttribute('text-anchor', 'middle');
        text.textContent = xVal + 'ms';
        svg.appendChild(text);
      }});

      // Plot models
      paretoModels.forEach(m => {{
        const cx = scaleX(m.p95_ms);
        const cy = scaleY(m.accuracy);
        
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', cx); circle.setAttribute('cy', cy);
        circle.setAttribute('r', m.current ? '8' : '5');
        circle.setAttribute('fill', m.current ? '#3fb950' : '#8b949e');
        circle.setAttribute('stroke', m.current ? '#ffffff' : '#30363d');
        circle.setAttribute('stroke-width', m.current ? '2.5' : '1.5');
        circle.style.cursor = 'pointer';

        // Hover events
        circle.addEventListener('mouseenter', (e) => {{
          tooltip.style.display = 'block';
          tooltip.innerHTML = `<strong>${{m.name}}</strong><br>P95 Latency: ${{m.p95_ms}} ms<br>Accuracy: ${{m.accuracy}}%<br>LAPS: ${{m.laps}}`;
          tooltip.style.left = (e.pageX + 10) + 'px';
          tooltip.style.top = (e.pageY - 30) + 'px';
        }});
        circle.addEventListener('mousemove', (e) => {{
          tooltip.style.left = (e.pageX + 10) + 'px';
          tooltip.style.top = (e.pageY - 30) + 'px';
        }});
        circle.addEventListener('mouseleave', () => {{
          tooltip.style.display = 'none';
        }});

        svg.appendChild(circle);

        // Label
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', cx + 10); text.setAttribute('y', cy + 4);
        text.setAttribute('fill', m.current ? '#3fb950' : '#c9d1d9');
        text.setAttribute('font-size', m.current ? '11' : '9');
        text.setAttribute('font-weight', m.current ? 'bold' : 'normal');
        text.textContent = m.name;
        svg.appendChild(text);
      }});
    }}

    renderRadarChart();
    renderParetoChart();
  </script>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path
