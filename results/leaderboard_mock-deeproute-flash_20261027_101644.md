# PromptBench-DeepRoute Benchmark Leaderboard & Report

**Model Tested:** `oracle-perfect-baseline`  
**Timestamp:** `2026-10-27 10:16:43`  
**Total Suite Runtime:** `0.59 seconds`  

---

## Executive Summary

PromptBench-DeepRoute evaluates fast, sub-100ms System 1 routing models on prompt classification, multi-label facet tagging, and compound prompt decomposition into specialist subagent execution DAGs.

### Mode 1: Atomic (Single-Label) Classification

| Metric | Result | Benchmark Target | Status |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | **100.00%** | >= 90.0% | ✅ Pass |
| **Top-3 Accuracy** | **100.00%** | >= 98.0% | ✅ Pass |
| **Domain Accuracy** | **100.00%** | >= 95.0% | ✅ Pass |
| **Macro F1** | **1.0000** | >= 0.880 | ✅ Pass |
| **Micro F1** | **1.0000** | >= 0.900 | ✅ Pass |
| **Expected Calibration Error (ECE)** | **0.0200** | <= 0.080 | ✅ Calibrated |
| **P50 Latency** | **48.31 ms** | < 60 ms | ⚡ Sub-100ms System 1 |
| **P95 Latency** | **71.63 ms** | < 100 ms | ⚡ Fast |
| **P99 Latency** | **74.56 ms** | < 150 ms | - |
| **Throughput** | **3022.17 req/s** | Concurrency | - |

### Mode 2: Multi-Label Multi-Aspect Tagging

| Metric | Result | Target |
| :--- | :--- | :--- |
| **Exact Match Ratio** | **100.00%** | Exact facet matching |
| **Jaccard Index (Mean)** | **1.0000** | Intersection over union across tags |
| **Hamming Loss** | **0.0000** | Error rate across binary facets |
| **Multi-label Macro F1** | **1.0000** | Tag balance |
| **Multi-label Micro F1** | **1.0000** | Global tag score |
| **P50 Latency** | **49.05 ms** | System 1 latency |
| **P95 Latency** | **73.88 ms** | Tail latency |

#### Facet-wise Accuracies

| Facet | Jaccard / Accuracy Score |
| :--- | :--- |
| `domains` | **1.0000** |
| `intents` | **1.0000** |
| `modalities` | **1.0000** |
| `complexity` | **1.0000** |
| `safety_flags` | **1.0000** |
| `tool_dependencies` | **1.0000** |

### Mode 3: 'Multi' Mode (Compound Decomposition & Subagent Routing DAG)

| Metric | Result | Description |
| :--- | :--- | :--- |
| **is_compound Accuracy** | **100.00%** | Distinguishing compound vs atomic fake-outs |
| **Task Boundary F1** | **1.0000** | Token Span IoU bipartite matching quality |
| **Task Boundary Precision** | **1.0000** | Fraction of proposed sub-tasks that are valid |
| **Task Boundary Recall** | **1.0000** | Fraction of ground-truth sub-tasks discovered |
| **Agent Allocation Accuracy** | **100.00%** | % tasks routed to exact correct subagent |
| **DAG Structural Validity** | **100.00%** | % valid acyclic directed graphs (no cycles/deadlocks) |
| **DAG Topological Match** | **100.00%** | Exact dependency edge graph isomorphism |
| **Dependency Edge F1** | **1.0000** | Edge precision & recall on task dependencies |
| **Dependency Edge Jaccard** | **1.0000** | Edge set intersection over union |
| **Mean Graph Edit Distance** | **0.00** | Lower indicates closer structural DAG fidelity |
| **P50 Latency** | **47.33 ms** | Decomposition latency |
| **P95 Latency** | **71.48 ms** | Tail latency |

#### DAG Pattern Breakdown

| Pattern | Sample Count | Boundary F1 | Agent Acc | Valid DAG % | Topological Match % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `linear_pipeline` | 25.0 | 1.0000 | 100.0% | 100.0% | **100.0%** |
| `parallel_map_reduce` | 25.0 | 1.0000 | 100.0% | 100.0% | **100.0%** |
| `diamond_dag` | 25.0 | 1.0000 | 100.0% | 100.0% | **100.0%** |
| `atomic_fakeout` | 25.0 | 1.0000 | 100.0% | 100.0% | **100.0%** |

#### Subagent Allocation Confusion Matrix

Cross-tabulation of ground truth subagents vs model predicted subagents:

| GT Agent \ Pred Agent | `coder` | `coordinator` | `data_engineer` | `finance_officer` | `hardware_engineer` | `math_specialist` | `researcher` | `sysadmin` | `writer` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `coder` | 67 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `coordinator` | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `data_engineer` | 0 | 0 | 39 | 0 | 0 | 0 | 0 | 0 | 0 |
| `finance_officer` | 0 | 0 | 0 | 18 | 0 | 0 | 0 | 0 | 0 |
| `hardware_engineer` | 0 | 0 | 0 | 0 | 42 | 0 | 0 | 0 | 0 |
| `math_specialist` | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| `researcher` | 0 | 0 | 0 | 0 | 0 | 0 | 31 | 0 | 0 |
| `sysadmin` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 34 | 0 |
| `writer` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 60 |
