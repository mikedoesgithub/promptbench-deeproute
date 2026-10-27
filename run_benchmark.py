#!/usr/bin/env python3
"""PromptBench-DeepRoute v2.0: Benchmark Suite CLI Entrypoint."""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure root workspace directory is on sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from src.models.factory import get_model_adapter
from src.harness.runner import BenchmarkRunner
from src.harness.reporter import BenchmarkReporter


def parse_args():
    parser = argparse.ArgumentParser(
        description="PromptBench-DeepRoute v2.0: Benchmark Suite for Fast System 1 Agentic Routing Models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock-deeproute-flash",
        help="Name of the model to benchmark (e.g. gpt-4o-mini, gemini-2.5-flash, claude-3-5-haiku, or mock-*)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["atomic", "multilabel", "multi", "adversarial", "stress", "all"],
        default="all",
        help="Benchmark evaluation mode: 'atomic', 'multilabel', 'multi', 'adversarial', or 'all'",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--concurrency-sweep",
        type=str,
        default=None,
        help="Comma-separated concurrency levels for throughput saturation sweep (e.g. '1,10,25,50')",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force using the high-speed offline Mock adapter without API calls",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Run mock adapter in Oracle mode (100 percent ground truth matching) to verify theoretical ceiling",
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.06,
        help="Simulated error noise rate for mock adapter (default: 0.06)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key override (defaults to environment variables OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL override for OpenAI-compatible endpoints (vLLM, Ollama, Groq)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save benchmark reports and JSON exports (default: results/)",
    )
    return parser.parse_args()


async def main_async():
    args = parse_args()

    datasets_dir = WORKSPACE_DIR / "datasets"
    taxonomy_path = WORKSPACE_DIR / "configs" / "taxonomy.yaml"
    output_dir = WORKSPACE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    concurrency_sweep_list = None
    if args.concurrency_sweep:
        concurrency_sweep_list = [int(x.strip()) for x in args.concurrency_sweep.split(",") if x.strip()]

    # 1. Instantiate model adapter
    adapter = get_model_adapter(
        model_name=args.model if not args.oracle else "oracle-perfect-baseline",
        mock=args.mock or args.oracle,
        oracle=args.oracle,
        noise_rate=args.noise_rate,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    reporter = BenchmarkReporter()
    reporter.console.print(f"[bold cyan]PromptBench-DeepRoute v2.0[/bold cyan] initializing...")
    reporter.console.print(
        f"Model: [bold yellow]{adapter.model_name}[/bold yellow] | "
        f"Mode: [bold]{args.mode}[/bold] | "
        f"Concurrency: [bold]{args.concurrency}[/bold]"
    )

    # 2. Instantiate and execute benchmark runner
    runner = BenchmarkRunner(
        adapter=adapter,
        datasets_dir=datasets_dir,
        taxonomy_path=taxonomy_path,
        concurrency=args.concurrency,
        concurrency_sweep=concurrency_sweep_list,
    )

    summary = await runner.run_suite(mode=args.mode)

    # 3. Render terminal output
    reporter.print_terminal_summary(summary)

    # 4. Generate persistent reports
    clean_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = adapter.model_name.replace("/", "_").replace(":", "_")
    md_filename = f"leaderboard_{clean_model_name}_{clean_timestamp}.md"
    json_filename = f"eval_{clean_model_name}_{clean_timestamp}.json"
    html_filename = f"dashboard_{clean_model_name}_{clean_timestamp}.html"

    md_path = output_dir / md_filename
    json_path = output_dir / json_filename
    html_path = output_dir / html_filename

    reporter.export_markdown_report(summary, md_path)
    reporter.export_json(summary, json_path)
    reporter.export_html_dashboard(summary, html_path)

    reporter.console.print()
    reporter.console.print(f"💾 [bold green]Benchmark artifacts successfully generated:[/bold green]")
    reporter.console.print(f"   • Markdown Leaderboard:  [bold underline]{md_path}[/bold underline]")
    reporter.console.print(f"   • Raw JSON Results:      [bold underline]{json_path}[/bold underline]")
    reporter.console.print(f"   • Interactive HTML App:  [bold underline]{html_path}[/bold underline]")
    reporter.console.print()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
