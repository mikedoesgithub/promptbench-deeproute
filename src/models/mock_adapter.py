"""Mock model adapter for high-speed offline benchmarking and CI/CD validation."""

from __future__ import annotations
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.models.base import BaseModelAdapter
from src.schemas import (
    AtomicPrediction,
    MultiLabelPrediction,
    CompoundMultiPrediction,
    AdversarialStressPrediction,
    SubTask,
    TopKCategory,
)


class MockModelAdapter(BaseModelAdapter):
    """High-speed mock adapter simulating sub-100ms System 1 inference with realistic routing behavior."""

    def __init__(
        self,
        model_name: str = "mock-system1-fast",
        noise_rate: float = 0.08,
        min_latency_ms: float = 20.0,
        max_latency_ms: float = 75.0,
        oracle: bool = False,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.noise_rate = noise_rate
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.oracle = oracle

        self.atomic_cache: Dict[str, Any] = {}
        self.multilabel_cache: Dict[str, Any] = {}
        self.compound_cache: Dict[str, Any] = {}
        self.adversarial_cache: Dict[str, Any] = {}

        if self.oracle:
            self._load_ground_truth_caches()

    def _load_ground_truth_caches(self):
        root = Path(__file__).resolve().parent.parent.parent / "datasets"
        # Atomic
        for fn in ["atomic_core.jsonl", "atomic_eval.jsonl"]:
            if (root / fn).exists():
                with open(root / fn, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            self.atomic_cache[d["prompt"].strip()] = d

        # Multilabel
        for fn in ["multilabel_faceted.jsonl", "multilabel_eval.jsonl"]:
            if (root / fn).exists():
                with open(root / fn, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            self.multilabel_cache[d["prompt"].strip()] = d

        # Compound
        for fn in ["compound_dag_advanced.jsonl", "compound_multi_eval.jsonl"]:
            if (root / fn).exists():
                with open(root / fn, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            self.compound_cache[d["prompt"].strip()] = d

        # Adversarial
        if (root / "adversarial_stress.jsonl").exists():
            with open(root / "adversarial_stress.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        self.adversarial_cache[d["prompt"].strip()] = d

    async def _simulate_latency(self) -> float:
        """Simulates asynchronous network and model computation latency."""
        latency = random.uniform(self.min_latency_ms, self.max_latency_ms)
        await asyncio.sleep(latency / 1000.0 * 0.05)
        return round(latency, 2)

    async def predict_atomic(self, prompt: str) -> AtomicPrediction:
        lat = await self._simulate_latency()

        if self.oracle and prompt.strip() in self.atomic_cache:
            gt = self.atomic_cache[prompt.strip()]
            cat = gt["primary_category"]
            dom = gt["domain"]
            conf = 0.98
            top_k = [
                TopKCategory(category=cat, confidence=conf),
                TopKCategory(category="swe_code_generation" if cat != "swe_code_generation" else "research_factual_search", confidence=0.015),
                TopKCategory(category="prose_editing_proofreading" if cat != "prose_editing_proofreading" else "sys_shell_scripting", confidence=0.005)
            ]
            return AtomicPrediction(
                predicted_category=cat,
                predicted_domain=dom,
                confidence=conf,
                top_k=top_k,
                latency_ms=lat,
                raw_output=f'{{"predicted_category": "{cat}", "confidence": {conf:.2f}}}'
            )

        p_lower = prompt.lower()

        # Heuristic routing across all 12 domains
        # 1. Browser automation
        if any(w in p_lower for w in ["playwright", "click", "selector", "dom", "scrape", "button#", "input[", "accessibility tree", "aria", "tab", "har"]):
            dom = "browser_automation"
            if "scrape" in p_lower:
                cat = "browser_web_scraping"
            elif "form" in p_lower or "login" in p_lower or "submit" in p_lower:
                cat = "browser_form_submission"
            elif "canvas" in p_lower or "webgl" in p_lower:
                cat = "browser_canvas_inspection"
            elif "cookie" in p_lower or "localstorage" in p_lower:
                cat = "browser_session_cookies"
            elif "intercept" in p_lower or "mock" in p_lower:
                cat = "browser_network_interception"
            elif "screenshot" in p_lower:
                cat = "browser_screenshot_capture"
            elif "tab" in p_lower or "window" in p_lower or "iframe" in p_lower:
                cat = "browser_tab_management"
            elif "aria" in p_lower or "tree" in p_lower:
                cat = "browser_accessibility_tree"
            else:
                cat = "browser_dom_interaction"

        # 2. Multimodal vision
        elif any(w in p_lower for w in ["screenshot", "wireframe", "image", "diagram", "photograph", "chart image", "ocr", "design token", "bounding box"]):
            dom = "multimodal_vision"
            if "wireframe" in p_lower or "figma" in p_lower or "sketch" in p_lower:
                cat = "vision_ui_wireframe_analysis"
            elif "architecture" in p_lower or "er diagram" in p_lower or "sequence diagram" in p_lower:
                cat = "vision_architecture_diagram"
            elif "chart" in p_lower or "line graph" in p_lower or "pie chart" in p_lower:
                cat = "vision_chart_data_decoding"
            elif "pcb" in p_lower or "solder" in p_lower or "optical" in p_lower:
                cat = "vision_optical_inspection"
            elif "ocr" in p_lower or "invoice" in p_lower or "receipt" in p_lower or "handwritten" in p_lower:
                cat = "vision_document_ocr_parsing"
            elif "token" in p_lower or "palette" in p_lower or "typography" in p_lower:
                cat = "vision_design_token_extraction"
            elif "bounding box" in p_lower or "spatial" in p_lower or "overlap" in p_lower:
                cat = "vision_spatial_layout_reasoning"
            elif "video" in p_lower or "frame" in p_lower:
                cat = "vision_video_temporal_reasoning"
            elif "adversarial" in p_lower or "steganography" in p_lower:
                cat = "vision_multimodal_safety_check"
            else:
                cat = "vision_ui_wireframe_analysis"

        # 3. MCP protocol routing
        elif "mcp" in p_lower:
            dom = "mcp_protocol_routing"
            if "github" in p_lower:
                cat = "mcp_github_server"
            elif "postgres" in p_lower or "sql" in p_lower:
                cat = "mcp_postgres_server"
            elif "slack" in p_lower:
                cat = "mcp_slack_server"
            elif "terminal" in p_lower or "shell" in p_lower or "pytest" in p_lower:
                cat = "mcp_terminal_server"
            elif "file" in p_lower or "workspace" in p_lower:
                cat = "mcp_filesystem_server"
            elif "memory" in p_lower or "graph" in p_lower:
                cat = "mcp_memory_knowledge_graph"
            elif "fetch" in p_lower or "http" in p_lower:
                cat = "mcp_fetch_http_server"
            elif "brave" in p_lower or "search" in p_lower:
                cat = "mcp_brave_search_server"
            else:
                cat = "mcp_tool_negotiation_discovery"

        # 4. Math & symbolic
        elif any(w in p_lower for w in ["prove", "integral", "derivative", "probability", "bayes", "permutation", "puzzle", "eigenvalue", "entropy"]):
            dom = "math_symbolic"
            if "prove" in p_lower or "induction" in p_lower:
                cat = "math_formal_proof"
            elif "integral" in p_lower or "differential equation" in p_lower:
                cat = "math_algebra_calculus"
            elif "probability" in p_lower or "bayes" in p_lower:
                cat = "math_probability_statistics"
            elif "permutation" in p_lower or "rearrange" in p_lower or "chromatic" in p_lower:
                cat = "math_combinatorics_discrete"
            elif "puzzle" in p_lower or "knave" in p_lower:
                cat = "math_logic_puzzles"
            elif "cancellation" in p_lower or "floating-point" in p_lower:
                cat = "math_numerical_computation"
            elif "eigenvalue" in p_lower or "svd" in p_lower:
                cat = "math_linear_algebra"
            elif "kkt" in p_lower or "dual" in p_lower or "linear program" in p_lower:
                cat = "math_optimization_theory"
            else:
                cat = "math_information_theory"

        # 5. Data engineering
        elif any(w in p_lower for w in ["sql", "query", "cte", "postgres", "polars", "pandas", "dataframe", "kafka", "etl", "airflow", "iceberg", "delta lake"]):
            dom = "data_engineering"
            if "sql" in p_lower or "cte" in p_lower or "window function" in p_lower:
                cat = "data_sql_query_authoring"
            elif "polars" in p_lower or "pandas" in p_lower or "dataframe" in p_lower:
                cat = "data_dataframe_transformation"
            elif "kafka" in p_lower or "airflow" in p_lower or "etl" in p_lower:
                cat = "data_etl_pipeline_design"
            elif "vega-lite" in p_lower or "chart" in p_lower or "seaborn" in p_lower or "dash" in p_lower:
                cat = "data_visualization_spec"
            elif "alembic" in p_lower or "migration" in p_lower or "normalized" in p_lower:
                cat = "data_schema_migration"
            elif "profiling" in p_lower or "kurtosis" in p_lower or "outlier" in p_lower:
                cat = "data_statistical_profiling"
            elif "iceberg" in p_lower or "delta lake" in p_lower or "parquet" in p_lower:
                cat = "data_lakehouse_storage"
            elif "great expectations" in p_lower or "contract" in p_lower:
                cat = "data_quality_validation"
            else:
                cat = "data_vector_index_tuning"

        # 6. System admin & security
        elif any(w in p_lower for w in ["bash", "posix", "kubernetes", "pod", "sshd", "cve", "log", "backup", "iam", "terraform"]):
            dom = "system_admin_security"
            if "bash" in p_lower or "shell" in p_lower:
                cat = "sys_shell_scripting"
            elif "kubernetes" in p_lower or "pod" in p_lower or "docker" in p_lower:
                cat = "sys_container_orchestration"
            elif "sshd" in p_lower or "harden" in p_lower:
                cat = "sys_config_audit_hardening"
            elif "curl" in p_lower or "dns" in p_lower or "tcpdump" in p_lower:
                cat = "sys_network_troubleshooting"
            elif "cve" in p_lower or "vulnerability" in p_lower:
                cat = "sys_pentest_vulnerability_scan"
            elif "auth.log" in p_lower or "forensics" in p_lower or "evtx" in p_lower:
                cat = "sys_log_analysis_forensics"
            elif "backup" in p_lower or "disaster recovery" in p_lower:
                cat = "sys_backup_disaster_recovery"
            elif "iam" in p_lower or "rbac" in p_lower or "okta" in p_lower:
                cat = "sys_identity_access_management"
            else:
                cat = "sys_infrastructure_as_code"

        # 7. Financial & quantitative
        elif any(w in p_lower for w in ["runway", "burn rate", "ledger", "double-entry", "npv", "irr", "amortization", "capex", "procurement", "tokenomics", "section 127", "tax", "sharpe", "cap table", "cac", "ltv"]):
            dom = "financial_quantitative"
            if "runway" in p_lower or "burn" in p_lower or "budget" in p_lower:
                cat = "finance_budgeting_forecasting"
            elif "ledger" in p_lower or "journal" in p_lower:
                cat = "finance_ledger_accounting"
            elif "npv" in p_lower or "irr" in p_lower or "amortization" in p_lower or "apr" in p_lower:
                cat = "finance_rate_calculations"
            elif "capex" in p_lower or "procurement" in p_lower or "memo" in p_lower:
                cat = "finance_memo_purchase_proposal"
            elif "token" in p_lower or "vesting" in p_lower or "bonding curve" in p_lower:
                cat = "finance_crypto_tokenomics"
            elif "section 127" in p_lower or "tax" in p_lower or "179" in p_lower:
                cat = "finance_tax_deduction_strategy"
            elif "sharpe" in p_lower or "var" in p_lower:
                cat = "finance_portfolio_risk_modeling"
            elif "cap table" in p_lower or "dilution" in p_lower or "safe" in p_lower:
                cat = "finance_equity_cap_table"
            else:
                cat = "finance_unit_economics"

        # 8. Hardware & embedded
        elif any(w in p_lower for w in ["firmware", "stm32", "can controller", "i2c", "spi", "schematic", "bom", "jlcpcb", "kicad", "pcb", "decoupling", "rf", "thermal", "dfm"]):
            dom = "hardware_embedded"
            if "firmware" in p_lower or "c driver" in p_lower or "interrupt" in p_lower:
                cat = "hardware_firmware_development"
            elif "schematic" in p_lower or "relay" in p_lower:
                cat = "hardware_schematic_analysis"
            elif "bom" in p_lower or "jlcpcb" in p_lower or "lcsc" in p_lower:
                cat = "hardware_bom_optimization"
            elif "kicad" in p_lower or "routing" in p_lower or "trace" in p_lower:
                cat = "hardware_pcb_layout_routing"
            elif "decoupling" in p_lower or "signal integrity" in p_lower or "pdn" in p_lower:
                cat = "hardware_signal_integrity_power"
            elif "logic analyzer" in p_lower or "reverse engineer" in p_lower or "sniff" in p_lower:
                cat = "hardware_protocol_reverse_eng"
            elif "rf" in p_lower or "smith chart" in p_lower or "antenna" in p_lower:
                cat = "hardware_rf_wireless_design"
            elif "thermal" in p_lower or "heatsink" in p_lower:
                cat = "hardware_thermal_management"
            else:
                cat = "hardware_mfg_dfm_validation"

        # 9. Creative prose
        elif any(w in p_lower for w in ["essay", "proofread", "tone", "copywriting", "headline", "dialogue", "story", "email", "memo", "tutorial", "summarize", "brainstorm"]):
            dom = "creative_prose"
            if "essay" in p_lower:
                cat = "prose_longform_essay"
            elif "proofread" in p_lower or "spelling" in p_lower or "grammar" in p_lower:
                cat = "prose_editing_proofreading"
            elif "tone" in p_lower or "rewrite" in p_lower:
                cat = "prose_tone_transfer"
            elif "headline" in p_lower or "ad copy" in p_lower or "copywriting" in p_lower:
                cat = "prose_copywriting_marketing"
            elif "story" in p_lower or "dialogue" in p_lower:
                cat = "prose_storytelling_narrative"
            elif "email" in p_lower or "apologizing" in p_lower:
                cat = "prose_email_correspondence"
            elif "tutorial" in p_lower or "guide" in p_lower:
                cat = "prose_technical_writing"
            elif "summarize" in p_lower or "distill" in p_lower:
                cat = "prose_summarization_executive"
            else:
                cat = "prose_creative_ideation"

        # 10. Research retrieval
        elif any(w in p_lower for w in ["arxiv", "synthesis", "speculative decoding", "release date", "competitive", "matrix", "extract", "survey", "citation", "patent", "gdpr", "hipaa", "credibility"]):
            dom = "research_retrieval"
            if "arxiv" in p_lower or "synthesis" in p_lower:
                cat = "research_literature_synthesis"
            elif "release date" in p_lower or "specs" in p_lower or "version" in p_lower:
                cat = "research_factual_search"
            elif "competitive" in p_lower or "compare" in p_lower:
                cat = "research_competitive_intelligence"
            elif "extract" in p_lower or "table" in p_lower:
                cat = "research_data_extraction"
            elif "survey" in p_lower:
                cat = "research_deep_survey"
            elif "citation" in p_lower or "quote" in p_lower:
                cat = "research_citation_verification"
            elif "patent" in p_lower:
                cat = "research_patent_analysis"
            elif "gdpr" in p_lower or "hipaa" in p_lower or "compliance" in p_lower:
                cat = "research_regulatory_compliance"
            else:
                cat = "research_source_credibility"

        # 11. Software engineering default
        elif any(w in p_lower for w in ["rust", "python", "typescript", "go", "refactor", "bug", "docker", "test", "package", "api", "profile"]):
            dom = "software_engineering"
            if "refactor" in p_lower:
                cat = "swe_code_refactoring"
            elif "bug" in p_lower or "error" in p_lower or "exception" in p_lower:
                cat = "swe_debugging_bug_hunting"
            elif "docker" in p_lower or "pipeline" in p_lower:
                cat = "swe_devops_ci_cd"
            elif "architecture" in p_lower:
                cat = "swe_system_architecture"
            elif "test" in p_lower:
                cat = "swe_test_writing"
            elif "package" in p_lower or "conflict" in p_lower:
                cat = "swe_dependency_management"
            elif "api" in p_lower or "webhook" in p_lower:
                cat = "swe_api_integration"
            elif "profile" in p_lower:
                cat = "swe_performance_profiling"
            else:
                cat = "swe_code_generation"

        else:
            dom = "conversational_admin"
            if "remind" in p_lower or "timer" in p_lower or "cron" in p_lower:
                cat = "admin_scheduling_reminder"
            elif "convert" in p_lower or "format" in p_lower:
                cat = "admin_formatting_conversion"
            elif "prompt" in p_lower:
                cat = "admin_meta_prompt_optimization"
            elif "build me" in p_lower or "make my code" in p_lower:
                cat = "admin_clarification_solicitation"
            elif "subagents" in p_lower or "capabilities" in p_lower:
                cat = "admin_help_capabilities"
            else:
                cat = "admin_general_chitchat"

        # Apply noise
        conf = random.uniform(0.85, 0.99)
        if random.random() < self.noise_rate:
            if dom == "software_engineering":
                cat = "swe_code_refactoring" if cat != "swe_code_refactoring" else "swe_code_generation"
            elif dom == "research_retrieval":
                cat = "research_factual_search" if cat != "research_factual_search" else "research_literature_synthesis"
            conf = random.uniform(0.60, 0.78)

        top_k = [
            TopKCategory(category=cat, confidence=round(conf, 3)),
            TopKCategory(category="swe_code_generation" if cat != "swe_code_generation" else "research_factual_search", confidence=round((1 - conf) * 0.7, 3)),
            TopKCategory(category="prose_editing_proofreading" if cat != "prose_editing_proofreading" else "sys_shell_scripting", confidence=round((1 - conf) * 0.3, 3))
        ]

        return AtomicPrediction(
            predicted_category=cat,
            predicted_domain=dom,
            confidence=round(conf, 3),
            top_k=top_k,
            latency_ms=lat,
            raw_output=f'{{"predicted_category": "{cat}", "confidence": {conf:.2f}}}'
        )

    async def predict_multilabel(self, prompt: str) -> MultiLabelPrediction:
        lat = await self._simulate_latency()

        if self.oracle and prompt.strip() in self.multilabel_cache:
            gt = self.multilabel_cache[prompt.strip()]
            return MultiLabelPrediction(
                domains=gt["domains"],
                intents=gt["intents"],
                modalities=gt["modalities"],
                complexity=gt["complexity"],
                safety_flags=gt["safety_flags"],
                tool_dependencies=gt.get("tool_dependencies", []),
                latency_ms=lat,
                confidence=0.98
            )

        # Fallback heuristic
        atomic = await self.predict_atomic(prompt)
        domains = [atomic.predicted_domain or "software_engineering"]
        intents = [atomic.predicted_category]
        modalities = ["autonomous_agent_loop"]
        tools = ["python_repl", "file_io"]

        return MultiLabelPrediction(
            domains=domains,
            intents=intents,
            modalities=modalities,
            complexity="sequential_pipeline",
            safety_flags=["benign_standard"],
            tool_dependencies=tools,
            latency_ms=lat,
            confidence=0.92
        )

    async def predict_compound(self, prompt: str) -> CompoundMultiPrediction:
        lat = await self._simulate_latency()

        if self.oracle and prompt.strip() in self.compound_cache:
            gt = self.compound_cache[prompt.strip()]
            sub_tasks = [
                SubTask(
                    id=st["id"],
                    text_span=st["text_span"],
                    assigned_subagent=st["assigned_subagent"],
                    intent_category=st["intent_category"],
                    dependencies=st["dependencies"]
                )
                for st in gt.get("sub_tasks", [])
            ]
            return CompoundMultiPrediction(
                is_compound=gt["is_compound"],
                sub_tasks=sub_tasks,
                latency_ms=lat
            )

        p_lower = prompt.lower()
        if "adversarial false-compound" in p_lower or "first define the user model" in p_lower or "review our quarterly ledger" in p_lower:
            return CompoundMultiPrediction(
                is_compound=False,
                sub_tasks=[SubTask(id="task_1", text_span=prompt, assigned_subagent="coder", intent_category="swe_code_generation", dependencies=[])],
                latency_ms=lat
            )

        if "diamond dag" in p_lower or "first," in p_lower and "concurrently" in p_lower:
            tasks = [
                SubTask(id="task_1", text_span="Initial root task analysis", assigned_subagent="data_engineer", intent_category="data_schema_migration", dependencies=[]),
                SubTask(id="task_2", text_span="Parallel implementation task A", assigned_subagent="data_engineer", intent_category="data_sql_query_authoring", dependencies=["task_1"]),
                SubTask(id="task_3", text_span="Parallel implementation task B", assigned_subagent="coder", intent_category="swe_test_writing", dependencies=["task_1"]),
                SubTask(id="task_4", text_span="Final aggregation memo", assigned_subagent="writer", intent_category="prose_email_correspondence", dependencies=["task_2", "task_3"]),
            ]
            return CompoundMultiPrediction(is_compound=True, sub_tasks=tasks, latency_ms=lat)

        return CompoundMultiPrediction(
            is_compound=False,
            sub_tasks=[SubTask(id="task_1", text_span=prompt, assigned_subagent="coordinator", intent_category="admin_general_chitchat", dependencies=[])],
            latency_ms=lat
        )

    async def predict_adversarial(self, prompt: str) -> AdversarialStressPrediction:
        lat = await self._simulate_latency()

        if self.oracle and prompt.strip() in self.adversarial_cache:
            gt = self.adversarial_cache[prompt.strip()]
            return AdversarialStressPrediction(
                predicted_category=gt["expected_category"],
                predicted_domain=gt["expected_domain"],
                predicted_safety_flag=gt["expected_safety_flag"],
                predicted_subagent=gt["expected_subagent"],
                confidence=0.98,
                latency_ms=lat
            )

        p_lower = prompt.lower()
        if any(w in p_lower for w in ["override", "bypass", "dan", "rm -rf", "shadow", "drop the production", "phishing"]):
            safety = "indirect_prompt_injection" if "phishing" in p_lower or "override" in p_lower else "destructive_unauthorized"
            subagent = "safety_sentinel"
            cat = "sys_config_audit_hardening"
            dom = "system_admin_security"
        else:
            safety = "benign_standard"
            subagent = "coder"
            cat = "swe_code_generation"
            dom = "software_engineering"

        return AdversarialStressPrediction(
            predicted_category=cat,
            predicted_domain=dom,
            predicted_safety_flag=safety,
            predicted_subagent=subagent,
            confidence=0.92,
            latency_ms=lat
        )
