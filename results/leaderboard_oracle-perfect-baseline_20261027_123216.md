# PromptBench-DeepRoute Leaderboard & Evaluation Report

**Model Evaluated:** `oracle-perfect-baseline`  
**Timestamp:** `2026-10-27 12:32:15`  
**Total Suite Runtime:** `1.28 seconds`  
**Overall Latency-Accuracy Pareto Score (LAPS):** **`86.32`**  

---

## Executive Summary

PromptBench-DeepRoute v2.0 evaluates fast, sub-100ms System 1 routing models across 1,200+ production test cases spanning 108 categories and 12 agentic domains, including Browser Automation, Multimodal Vision, and MCP Protocol Routing.

### Split 1: Atomic Core Classification (108 Categories)

| Metric | Result | Benchmark Target | Status |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | **100.00%** | >= 90.0% | ✅ Pass |
| **Top-3 Accuracy** | **100.00%** | >= 98.0% | ✅ Pass |
| **Domain Accuracy (12 Domains)** | **100.00%** | >= 95.0% | ✅ Pass |
| **Macro F1 (108 Classes)** | **1.0000** | >= 0.880 | ✅ Pass |
| **Expected Calibration Error (ECE)** | **0.0200** | <= 0.080 | ✅ Calibrated |
| **LAPS Score** | **85.90** | >= 85.0 | ⚡ Exceptional |
| **P50 / P95 Latency** | **49.62 ms / 72.97 ms** | < 50ms / < 100ms | Sub-100ms System 1 Budget |
| **P99 Latency / StdDev** | **74.62 ms (±16.51 ms)** | Stability | - |
| **Throughput** | **3155.95 req/s** | Concurrency: 10 | - |

#### Domain-Level Breakdown (12 Domains)

| Domain | Samples | Domain Accuracy | Granular Category Accuracy |
| :--- | :---: | :---: | :---: |
| `browser_automation` | 27 | **100.0%** | 100.0% |
| `conversational_admin` | 31 | **100.0%** | 100.0% |
| `creative_prose` | 36 | **100.0%** | 100.0% |
| `data_engineering` | 36 | **100.0%** | 100.0% |
| `financial_quantitative` | 36 | **100.0%** | 100.0% |
| `hardware_embedded` | 36 | **100.0%** | 100.0% |
| `math_symbolic` | 36 | **100.0%** | 100.0% |
| `mcp_protocol_routing` | 27 | **100.0%** | 100.0% |
| `multimodal_vision` | 27 | **100.0%** | 100.0% |
| `research_retrieval` | 36 | **100.0%** | 100.0% |
| `software_engineering` | 36 | **100.0%** | 100.0% |
| `system_admin_security` | 36 | **100.0%** | 100.0% |

### Split 2: Multi-Label Faceted Tagging

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Exact Match Ratio** | **100.00%** | All facets match exactly |
| **Jaccard Index (Mean)** | **1.0000** | Tag set intersection over union |
| **Hamming Loss** | **0.0000** | Fraction of erroneous tags |
| **Multi-label Macro F1** | **1.0000** | Unweighted tag vocabulary balance |
| **LAPS Score** | **86.58** | Pareto Score |
| **P50 / P95 Latency** | **46.48 ms / 71.44 ms** | Latency profile |

### Split 3: Compound DAG Advanced Decomposition

| Metric | Result | Description |
| :--- | :--- | :--- |
| **is_compound Accuracy** | **100.00%** | Distinguishing compound vs fakeout traps |
| **Task Boundary F1** | **1.0000** | Token Span IoU Hungarian bipartite matching |
| **Agent Allocation Accuracy** | **100.00%** | Routing to correct subagent specialist |
| **DAG Structural Validity** | **100.00%** | % valid acyclic directed graphs (no cycles) |
| **DAG Topological Match** | **100.00%** | Exact graph topology match |
| **Dependency Edge Jaccard** | **1.0000** | Prerequisite edge set overlap |
| **Mean Graph Edit Distance** | **0.00** | Structural distance to ground truth DAG |
| **LAPS Score** | **86.67** | Compound Pareto score |

### Split 4: Adversarial Stress & Robustness

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Overall Robustness** | **100.00%** | Overall accuracy across stress vectors |
| **Prompt Injection Defense** | **100.00%** | Catching overrides & jailbreaks |
| **Code-Switched Multilingual** | **100.00%** | Accurate routing across multilingual queries |
| **Distractor Immunity** | **100.00%** | Resisting misleading technical keywords |
| **Safety Flag Accuracy** | **100.00%** | Identifying dangerous operational tiers |
| **LAPS Score** | **86.13** | Stress Pareto score |
