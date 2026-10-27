# PromptBench-DeepRoute Leaderboard & Evaluation Report

**Model Evaluated:** `mock-deeproute-flash`  
**Timestamp:** `2026-10-27 12:14:56`  
**Total Suite Runtime:** `1.08 seconds`  
**Overall Latency-Accuracy Pareto Score (LAPS):** **`27.44`**  

---

## Executive Summary

PromptBench-DeepRoute v2.0 evaluates fast, sub-100ms System 1 routing models across 1,200+ production test cases spanning 108 categories and 12 agentic domains, including Browser Automation, Multimodal Vision, and MCP Protocol Routing.

### Split 1: Atomic Core Classification (108 Categories)

| Metric | Result | Benchmark Target | Status |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | **29.75%** | >= 90.0% | ⚠️ Review |
| **Top-3 Accuracy** | **30.75%** | >= 98.0% | ⚠️ Review |
| **Domain Accuracy (12 Domains)** | **41.25%** | >= 95.0% | ⚠️ Review |
| **Macro F1 (108 Classes)** | **0.3259** | >= 0.880 | ⚠️ Review |
| **Expected Calibration Error (ECE)** | **0.6083** | <= 0.080 | ⚠️ Overconfident |
| **LAPS Score** | **25.68** | >= 85.0 | 🐢 Low Latency Score |
| **P50 / P95 Latency** | **46.25 ms / 72.0 ms** | < 50ms / < 100ms | Sub-100ms System 1 Budget |
| **P99 Latency / StdDev** | **74.52 ms (±16.08 ms)** | Stability | - |
| **Throughput** | **3238.4 req/s** | Concurrency: 10 | - |

#### Domain-Level Breakdown (12 Domains)

| Domain | Samples | Domain Accuracy | Granular Category Accuracy |
| :--- | :---: | :---: | :---: |
| `browser_automation` | 36 | **44.4%** | 27.8% |
| `conversational_admin` | 36 | **55.6%** | 38.9% |
| `creative_prose` | 36 | **41.7%** | 33.3% |
| `data_engineering` | 36 | **33.3%** | 25.0% |
| `financial_quantitative` | 36 | **22.2%** | 22.2% |
| `hardware_embedded` | 36 | **41.7%** | 16.7% |
| `math_symbolic` | 36 | **36.1%** | 27.8% |
| `mcp_protocol_routing` | 4 | **100.0%** | 100.0% |
| `multimodal_vision` | 36 | **61.1%** | 41.7% |
| `research_retrieval` | 36 | **27.8%** | 16.7% |
| `software_engineering` | 36 | **50.0%** | 38.9% |
| `system_admin_security` | 36 | **33.3%** | 30.6% |

### Split 2: Multi-Label Faceted Tagging

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Exact Match Ratio** | **0.00%** | All facets match exactly |
| **Jaccard Index (Mean)** | **0.4362** | Tag set intersection over union |
| **Hamming Loss** | **0.1492** | Fraction of erroneous tags |
| **Multi-label Macro F1** | **0.2687** | Unweighted tag vocabulary balance |
| **LAPS Score** | **37.54** | Pareto Score |
| **P50 / P95 Latency** | **45.02 ms / 72.61 ms** | Latency profile |

### Split 3: Compound DAG Advanced Decomposition

| Metric | Result | Description |
| :--- | :--- | :--- |
| **is_compound Accuracy** | **66.00%** | Distinguishing compound vs fakeout traps |
| **Task Boundary F1** | **0.1256** | Token Span IoU Hungarian bipartite matching |
| **Agent Allocation Accuracy** | **10.00%** | Routing to correct subagent specialist |
| **DAG Structural Validity** | **100.00%** | % valid acyclic directed graphs (no cycles) |
| **DAG Topological Match** | **33.33%** | Exact graph topology match |
| **Dependency Edge Jaccard** | **0.3333** | Prerequisite edge set overlap |
| **Mean Graph Edit Distance** | **4.39** | Structural distance to ground truth DAG |
| **LAPS Score** | **16.19** | Compound Pareto score |

### Split 4: Adversarial Stress & Robustness

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Overall Robustness** | **35.00%** | Overall accuracy across stress vectors |
| **Prompt Injection Defense** | **85.71%** | Catching overrides & jailbreaks |
| **Code-Switched Multilingual** | **14.29%** | Accurate routing across multilingual queries |
| **Distractor Immunity** | **0.00%** | Resisting misleading technical keywords |
| **Safety Flag Accuracy** | **90.00%** | Identifying dangerous operational tiers |
| **LAPS Score** | **30.35** | Stress Pareto score |
