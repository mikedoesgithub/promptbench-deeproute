# PromptBench-DeepRoute Benchmark Leaderboard & Report

**Model Tested:** `mock-deeproute-flash`  
**Timestamp:** `2026-10-27 10:17:31`  
**Total Suite Runtime:** `0.34 seconds`  

---

## Executive Summary

PromptBench-DeepRoute evaluates fast, sub-100ms System 1 routing models on prompt classification, multi-label facet tagging, and compound prompt decomposition into specialist subagent execution DAGs.

### Mode 1: Atomic (Single-Label) Classification

| Metric | Result | Benchmark Target | Status |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | **85.00%** | >= 90.0% | ✅ Pass |
| **Top-3 Accuracy** | **86.00%** | >= 98.0% | ⚠️ Review |
| **Domain Accuracy** | **87.00%** | >= 95.0% | ⚠️ Review |
| **Macro F1** | **0.8481** | >= 0.880 | ✅ Pass |
| **Micro F1** | **0.8500** | >= 0.900 | ✅ Pass |
| **Expected Calibration Error (ECE)** | **0.0844** | <= 0.080 | ✅ Calibrated |
| **P50 Latency** | **44.49 ms** | < 60 ms | ⚡ Sub-100ms System 1 |
| **P95 Latency** | **73.53 ms** | < 100 ms | ⚡ Fast |
| **P99 Latency** | **74.61 ms** | < 150 ms | - |
| **Throughput** | **3044.41 req/s** | Concurrency | - |

#### Top Confused Category Pairs (Failure Analysis)

| Ground Truth Category | Model Predicted Category | Confusion Count |
| :--- | :--- | :--- |
| `hardware_pcb_layout_routing` | `sys_config_audit_hardening` | **2** |
| `sys_config_audit_hardening` | `admin_general_chitchat` | **2** |
| `admin_clarification_solicitation` | `swe_debugging_bug_hunting` | **1** |
| `finance_ledger_accounting` | `admin_general_chitchat` | **1** |
| `hardware_firmware_development` | `math_logic_puzzles` | **1** |
| `hardware_protocol_reverse_eng` | `hardware_firmware_development` | **1** |
| `prose_copywriting_marketing` | `swe_code_generation` | **1** |
| `prose_editing_proofreading` | `math_combinatorics_discrete` | **1** |
