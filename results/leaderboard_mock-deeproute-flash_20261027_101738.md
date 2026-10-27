# PromptBench-DeepRoute Benchmark Leaderboard & Report

**Model Tested:** `mock-deeproute-flash`  
**Timestamp:** `2026-10-27 10:17:38`  
**Total Suite Runtime:** `0.05 seconds`  

---

## Executive Summary

PromptBench-DeepRoute evaluates fast, sub-100ms System 1 routing models on prompt classification, multi-label facet tagging, and compound prompt decomposition into specialist subagent execution DAGs.

### Mode 2: Multi-Label Multi-Aspect Tagging

| Metric | Result | Target |
| :--- | :--- | :--- |
| **Exact Match Ratio** | **1.00%** | Exact facet matching |
| **Jaccard Index (Mean)** | **0.2214** | Intersection over union across tags |
| **Hamming Loss** | **0.1798** | Error rate across binary facets |
| **Multi-label Macro F1** | **0.2239** | Tag balance |
| **Multi-label Micro F1** | **0.3377** | Global tag score |
| **P50 Latency** | **48.39 ms** | System 1 latency |
| **P95 Latency** | **69.41 ms** | Tail latency |

#### Facet-wise Accuracies

| Facet | Jaccard / Accuracy Score |
| :--- | :--- |
| `domains` | **0.2108** |
| `intents` | **0.0752** |
| `modalities` | **0.2933** |
| `complexity` | **0.3000** |
| `safety_flags` | **1.0000** |
| `tool_dependencies` | **0.2450** |
