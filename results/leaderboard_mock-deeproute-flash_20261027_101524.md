# PromptBench-DeepRoute Benchmark Leaderboard & Report

**Model Tested:** `mock-deeproute-flash`  
**Timestamp:** `2026-10-27 10:15:24`  
**Total Suite Runtime:** `0.66 seconds`  

---

## Executive Summary

PromptBench-DeepRoute evaluates fast, sub-100ms System 1 routing models on prompt classification, multi-label facet tagging, and compound prompt decomposition into specialist subagent execution DAGs.

### Mode 1: Atomic (Single-Label) Classification

| Metric | Result | Benchmark Target | Status |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | **83.00%** | >= 90.0% | ⚠️ Review |
| **Top-3 Accuracy** | **86.00%** | >= 98.0% | ⚠️ Review |
| **Domain Accuracy** | **87.00%** | >= 95.0% | ⚠️ Review |
| **Macro F1** | **0.8343** | >= 0.880 | ✅ Pass |
| **Micro F1** | **0.8300** | >= 0.900 | ⚠️ Review |
| **Expected Calibration Error (ECE)** | **0.0748** | <= 0.080 | ✅ Calibrated |
| **P50 Latency** | **46.5 ms** | < 60 ms | ⚡ Sub-100ms System 1 |
| **P95 Latency** | **73.87 ms** | < 100 ms | ⚡ Fast |
| **P99 Latency** | **74.57 ms** | < 150 ms | - |
| **Throughput** | **1751.72 req/s** | Concurrency | - |

#### Top Confused Category Pairs (Failure Analysis)

| Ground Truth Category | Model Predicted Category | Confusion Count |
| :--- | :--- | :--- |
| `hardware_pcb_layout_routing` | `sys_config_audit_hardening` | **2** |
| `swe_code_generation` | `swe_code_refactoring` | **2** |
| `sys_config_audit_hardening` | `admin_general_chitchat` | **2** |
| `admin_clarification_solicitation` | `swe_debugging_bug_hunting` | **1** |
| `finance_ledger_accounting` | `admin_general_chitchat` | **1** |
| `hardware_firmware_development` | `math_logic_puzzles` | **1** |
| `hardware_protocol_reverse_eng` | `hardware_firmware_development` | **1** |
| `prose_copywriting_marketing` | `swe_code_generation` | **1** |

### Mode 2: Multi-Label Multi-Aspect Tagging

| Metric | Result | Target |
| :--- | :--- | :--- |
| **Exact Match Ratio** | **1.00%** | Exact facet matching |
| **Jaccard Index (Mean)** | **0.2214** | Intersection over union across tags |
| **Hamming Loss** | **0.1798** | Error rate across binary facets |
| **Multi-label Macro F1** | **0.2239** | Tag balance |
| **Multi-label Micro F1** | **0.3377** | Global tag score |
| **P50 Latency** | **50.03 ms** | System 1 latency |
| **P95 Latency** | **70.72 ms** | Tail latency |

#### Facet-wise Accuracies

| Facet | Jaccard / Accuracy Score |
| :--- | :--- |
| `domains` | **0.2108** |
| `intents` | **0.0752** |
| `modalities` | **0.2933** |
| `complexity` | **0.3000** |
| `safety_flags` | **1.0000** |
| `tool_dependencies` | **0.2450** |

### Mode 3: 'Multi' Mode (Compound Decomposition & Subagent Routing DAG)

| Metric | Result | Description |
| :--- | :--- | :--- |
| **is_compound Accuracy** | **100.00%** | Distinguishing compound vs atomic fake-outs |
| **Task Boundary F1** | **0.2145** | Token Span IoU bipartite matching quality |
| **Task Boundary Precision** | **0.2167** | Fraction of proposed sub-tasks that are valid |
| **Task Boundary Recall** | **0.2124** | Fraction of ground-truth sub-tasks discovered |
| **Agent Allocation Accuracy** | **76.92%** | % tasks routed to exact correct subagent |
| **DAG Structural Validity** | **100.00%** | % valid acyclic directed graphs (no cycles/deadlocks) |
| **DAG Topological Match** | **36.00%** | Exact dependency edge graph isomorphism |
| **Dependency Edge F1** | **0.3600** | Edge precision & recall on task dependencies |
| **Dependency Edge Jaccard** | **0.3600** | Edge set intersection over union |
| **Mean Graph Edit Distance** | **4.19** | Lower indicates closer structural DAG fidelity |
| **P50 Latency** | **53.19 ms** | Decomposition latency |
| **P95 Latency** | **72.3 ms** | Tail latency |

#### DAG Pattern Breakdown

| Pattern | Sample Count | Boundary F1 | Agent Acc | Valid DAG % | Topological Match % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `linear_pipeline` | 25.0 | 0.1600 | 16.0% | 100.0% | **16.0%** |
| `parallel_map_reduce` | 25.0 | 0.2800 | 28.0% | 100.0% | **28.0%** |
| `diamond_dag` | 25.0 | 0.0000 | 0.0% | 100.0% | **0.0%** |
| `atomic_fakeout` | 25.0 | 1.0000 | 40.0% | 100.0% | **100.0%** |

#### Subagent Allocation Confusion Matrix

Cross-tabulation of ground truth subagents vs model predicted subagents:

| GT Agent \ Pred Agent | `coder` | `data_engineer` | `finance_officer` | `researcher` | `sysadmin` | `writer` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `coder` | 15 | 0 | 0 | 0 | 0 | 0 |
| `data_engineer` | 3 | 7 | 0 | 0 | 0 | 0 |
| `finance_officer` | 0 | 0 | 3 | 0 | 0 | 0 |
| `hardware_engineer` | 3 | 0 | 0 | 0 | 0 | 0 |
| `math_specialist` | 3 | 0 | 0 | 0 | 0 | 0 |
| `researcher` | 3 | 0 | 0 | 4 | 0 | 0 |
| `sysadmin` | 3 | 0 | 0 | 0 | 7 | 0 |
| `writer` | 0 | 0 | 0 | 0 | 0 | 14 |
