# PromptBench-DeepRoute Benchmark Leaderboard & Report

**Model Tested:** `mock-deeproute-flash`  
**Timestamp:** `2026-10-27 10:17:45`  
**Total Suite Runtime:** `0.05 seconds`  

---

## Executive Summary

PromptBench-DeepRoute evaluates fast, sub-100ms System 1 routing models on prompt classification, multi-label facet tagging, and compound prompt decomposition into specialist subagent execution DAGs.

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
| **P50 Latency** | **45.84 ms** | Decomposition latency |
| **P95 Latency** | **72.67 ms** | Tail latency |

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
