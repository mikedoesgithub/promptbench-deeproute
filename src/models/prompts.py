"""Prompt templates and dynamic system routing instructions for fast System 1 classification."""

from __future__ import annotations
from typing import Dict, Any, List, Optional


def build_atomic_system_prompt(taxonomy: Optional[Dict[str, Any]] = None) -> str:
    """Dynamically builds the atomic classification system prompt injecting all domains and categories."""
    domains_text_blocks = []
    total_categories = 0

    if taxonomy and "tier1_domains" in taxonomy:
        for dom_key, dom_val in taxonomy["tier1_domains"].items():
            display_name = dom_val.get("display_name", dom_key)
            description = dom_val.get("description", "")
            cats = dom_val.get("categories", [])
            total_categories += len(cats)
            cats_formatted = ", ".join(cats)
            domains_text_blocks.append(
                f"• Domain '{dom_key}' ({display_name}): {description}\n  Valid Categories: [{cats_formatted}]"
            )
    elif taxonomy and "granular_categories" in taxonomy:
        # Group by domain
        dom_map: Dict[str, List[str]] = {}
        for cat_key, cat_val in taxonomy["granular_categories"].items():
            dom = cat_val.get("domain", "general")
            dom_map.setdefault(dom, []).append(cat_key)
        for dom, cats in dom_map.items():
            total_categories += len(cats)
            domains_text_blocks.append(f"• Domain '{dom}': [{', '.join(cats)}]")

    domains_section = "\n".join(domains_text_blocks) if domains_text_blocks else "Use standard taxonomy categories."

    return f"""You are a high-speed, sub-100ms System 1 Agentic Router. Your job is to classify incoming user prompts into exactly ONE primary category and domain from the official taxonomy below.

CRITICAL INVARIANTS:
1. You MUST select 'predicted_category' strictly from the valid category keys listed below. Do NOT invent, translate, or abbreviate category names.
2. Select the matching 'predicted_domain' corresponding to the selected category.
3. Provide a calibrated confidence score between 0.0 and 1.0.
4. Output STRICT JSON only. Do not provide explanations, thoughts, or markdown text outside the JSON block.

OFFICIAL TAXONOMY ({total_categories} Granular Categories across {len(domains_text_blocks)} Domains):
{domains_section}

EXPECTED JSON SCHEMA:
{{
  "predicted_category": "<exact_category_key_from_list>",
  "predicted_domain": "<exact_domain_key>",
  "confidence": <float 0.0 to 1.0>,
  "top_k": [
    {{"category": "<category_key>", "confidence": <float>}},
    {{"category": "<category_key>", "confidence": <float>}},
    {{"category": "<category_key>", "confidence": <float>}}
  ]
}}"""


def build_compound_system_prompt(subagents: Optional[Dict[str, Any]] = None, taxonomy: Optional[Dict[str, Any]] = None) -> str:
    """Dynamically builds compound prompt decomposition instructions injecting all specialist subagents."""
    agent_blocks = []
    agents_dict = subagents.get("subagents", subagents) if isinstance(subagents, dict) else {}

    if agents_dict:
        for name, info in agents_dict.items():
            display_name = info.get("display_name", name)
            desc = info.get("description", "")
            allowed_doms = ", ".join(info.get("allowed_domains", []))
            agent_blocks.append(f"• '{name}' ({display_name}): {desc} [Domains: {allowed_doms}]")
    else:
        # Fallback default 14 subagents
        default_agents = [
            ("coder", "Software Engineer", "Code generation, refactoring, debugging, and testing."),
            ("researcher", "Research Analyst", "Literature search, factual retrieval, and data extraction."),
            ("writer", "Prose Author", "Drafting prose, marketing copy, summaries, and correspondence."),
            ("math_specialist", "Mathematician", "Mathematical proofs, calculus, algebra, and logic puzzles."),
            ("data_engineer", "Data Engineer", "SQL authoring, dataframe transforms, and ETL pipelines."),
            ("sysadmin", "DevOps & Security", "Shell scripts, Docker, Kubernetes, and log forensics."),
            ("finance_officer", "Financial Officer", "Budgeting, ledger accounting, interest, and purchase memos."),
            ("hardware_engineer", "Hardware Engineer", "Firmware, schematics, PCB layout, and BOM optimization."),
            ("coordinator", "Coordinator", "Administrative triage, scheduling, format conversion, and sequencing."),
            ("browser_navigator", "Browser Navigator", "DOM automation, web scraping, form submission, and canvas."),
            ("vision_analyst", "Vision Analyst", "UI wireframe analysis, architecture diagrams, and OCR parsing."),
            ("mcp_orchestrator", "MCP Orchestrator", "Routing requests to external MCP tool servers (GitHub, Postgres, Slack, Terminal)."),
            ("safety_sentinel", "Safety Sentinel", "Intercepting destructive actions, unauthorized commands, and injections."),
            ("triage_interviewer", "Triage Interviewer", "Detecting underspecified prompts and interviewing user for clarification.")
        ]
        for name, dname, desc in default_agents:
            agent_blocks.append(f"• '{name}' ({dname}): {desc}")

    agents_section = "\n".join(agent_blocks)

    return f"""You are a high-speed, sub-100ms System 1 Compound Task Decomposition & Subagent Routing Engine.

Your objective:
1. Determine whether the user prompt is COMPOUND (contains multiple distinct sub-tasks requiring different specialized subagents, or a multi-stage sequential/parallel pipeline) or ATOMIC (a single coherent task for one specialist agent).
   - Beware of FALSE-COMPOUND TRAPS: Multi-sentence prompts that describe a single coherent task for one agent (e.g. "First define User model, then hash passwords, then add save method") are ATOMIC (is_compound=false).
2. If COMPOUND (is_compound=true):
   - Decompose into distinct atomic sub-tasks.
   - Assign each sub-task to the exact optimal specialist from the available subagents roster below.
   - Assign the matching intent_category.
   - Define execution dependencies: list of preceding task IDs that must complete before this task can execute (allowing parallel branches and sequential pipelines).
3. If ATOMIC (is_compound=false):
   - Set is_compound=false.
   - sub_tasks must contain exactly one task with dependencies=[] representing the overall task.

AVAILABLE SPECIALIST SUBAGENTS:
{agents_section}

EXPECTED JSON SCHEMA:
{{
  "is_compound": true | false,
  "sub_tasks": [
    {{
      "id": "task_1",
      "text_span": "<exact substring or sentence of subtask>",
      "assigned_subagent": "<one_of_the_subagent_names>",
      "intent_category": "<category_name>",
      "dependencies": []
    }},
    {{
      "id": "task_2",
      "text_span": "<exact substring of dependent subtask>",
      "assigned_subagent": "<one_of_the_subagent_names>",
      "intent_category": "<category_name>",
      "dependencies": ["task_1"]
    }}
  ]
}}"""


def build_multilabel_system_prompt(taxonomy: Optional[Dict[str, Any]] = None) -> str:
    """Dynamically builds multi-label faceted tagging system prompt."""
    domains = []
    if taxonomy and "tier1_domains" in taxonomy:
        domains = list(taxonomy["tier1_domains"].keys())
    elif taxonomy and "granular_categories" in taxonomy:
        domains = sorted(list({v.get("domain", "") for v in taxonomy["granular_categories"].values() if v.get("domain")}))
    else:
        domains = [
            "software_engineering", "research_retrieval", "creative_prose", "math_symbolic",
            "data_engineering", "system_admin_security", "financial_quantitative", "hardware_embedded",
            "conversational_admin", "browser_automation", "multimodal_vision", "mcp_protocol_routing"
        ]

    domains_formatted = ", ".join(domains)

    return f"""You are a high-speed, sub-100ms System 1 Multi-Aspect Tagging Engine.
Your job is to analyze incoming user prompts and extract comprehensive faceted operational tags across domains, intents, execution modalities, task complexity tier, safety/security status, and required tool dependencies.

VALID VOCABULARY:
• Valid Domains: [{domains_formatted}]
• Valid Modalities: ["single_turn_direct", "autonomous_agent_loop", "human_in_the_loop", "destructive_write", "read_only_lookup"]
• Valid Complexity Tiers: ["atomic", "sequential_pipeline", "parallel_map_reduce", "diamond_dag", "dynamic_cyclic"]
• Valid Safety Flags: ["benign_standard", "indirect_prompt_injection", "exfiltration_attempt", "ambiguous_underspecified", "destructive_unconfirmed"]
• Valid Tool Dependencies: ["bash", "git", "python_repl", "file_io", "web_search", "browser", "mcp_client"]

EXPECTED JSON SCHEMA:
{{
  "domains": ["<domain1>", ...],
  "intents": ["<intent_category1>", ...],
  "modalities": ["<modality1>", ...],
  "complexity": "<complexity_tier>",
  "safety_flags": ["<safety_flag1>", ...],
  "tool_dependencies": ["<tool1>", ...],
  "confidence": <float 0.0 to 1.0>
}}"""


# Baseline static fallbacks for backwards compatibility
SYSTEM_ROUTING_PROMPT = """You are a high-speed, sub-100ms System 1 Agentic Router. Your job is to classify user queries, extract multifaceted metadata, or decompose compound tasks into specialist subagents. Respond strictly in valid JSON without explanations or markdown formatting outside JSON.
"""

ATOMIC_PROMPT_TEMPLATE = """Classify the following user prompt into exactly ONE primary category and domain from the provided taxonomy.
Also assign a calibrated confidence score between 0.0 and 1.0, and optionally provide top 3 candidates.

Output JSON format:
{{
  "predicted_category": "<exact category key from the 108 granular categories>",
  "predicted_domain": "<domain name>",
  "confidence": <float 0.0 to 1.0>,
  "top_k": [
    {{"category": "<cat1>", "confidence": <float>}},
    {{"category": "<cat2>", "confidence": <float>}},
    {{"category": "<cat3>", "confidence": <float>}}
  ]
}}

User Prompt:
"{prompt}"
"""

MULTILABEL_PROMPT_TEMPLATE = """Analyze the following user prompt and extract its multifaceted operational tags across domains, intents, execution modalities, task complexity tier, safety/security status, and required tool dependencies.

Output JSON format:
{{
  "domains": ["<domain1>", ...],
  "intents": ["<intent_category1>", ...],
  "modalities": ["single_turn_direct" | "autonomous_agent_loop" | "human_in_the_loop" | "destructive_write" | "read_only_lookup"],
  "complexity": "atomic" | "sequential_pipeline" | "parallel_map_reduce" | "diamond_dag" | "dynamic_cyclic",
  "safety_flags": ["benign_standard" | "indirect_prompt_injection" | "exfiltration_attempt" | "ambiguous_underspecified" | "destructive_unconfirmed"],
  "tool_dependencies": ["bash" | "git" | "python_repl" | "file_io" | "web_search" | "browser" | "mcp_client"],
  "confidence": <float 0.0 to 1.0>
}}

User Prompt:
"{prompt}"
"""

COMPOUND_PROMPT_TEMPLATE = """Determine whether the following user prompt is compound (contains multiple distinct sub-tasks requiring different specialized subagents) or atomic (a single coherent task for one agent).

Available 14 specialist subagents:
- coder: software engineering, implementation, refactoring, tests
- researcher: literature search, factual retrieval, competitive intelligence
- writer: prose writing, essays, emails, marketing copy
- math_specialist: mathematical proofs, algebra, calculus, statistics, logic puzzles
- data_engineer: SQL authoring, dataframes, ETL pipelines, visualizations
- sysadmin: shell scripts, Linux security, container orchestration, log analysis
- finance_officer: corporate budgeting, ledger math, rate calculations, purchase memos
- hardware_engineer: embedded firmware, schematics, BOM optimization, PCB layout
- coordinator: triage, formatting, scheduling, asking clarifying questions
- browser_navigator: DOM interaction, web scraping, form automation, session cookies
- vision_analyst: UI wireframe analysis, architecture diagrams, chart decoding, OCR
- mcp_orchestrator: routing to external tool servers (GitHub, Postgres, Slack, Terminal)
- safety_sentinel: intercepting destructive commands, prompt injections, and secret leakage
- triage_interviewer: clarifying underspecified requirements through user interview

If compound: split into atomic sub-tasks, assign the subagent and intent category for each, and specify prerequisite dependencies.
If atomic: set is_compound=false, with a single task in sub_tasks with empty dependencies.

Output JSON format:
{{
  "is_compound": true,
  "sub_tasks": [
    {{
      "id": "task_1",
      "text_span": "<exact substring or sentence of subtask>",
      "assigned_subagent": "<subagent_name>",
      "intent_category": "<category_name>",
      "dependencies": []
    }},
    {{
      "id": "task_2",
      "text_span": "<subtask text>",
      "assigned_subagent": "<subagent_name>",
      "intent_category": "<category_name>",
      "dependencies": ["task_1"]
    }}
  ]
}}

User Prompt:
"{prompt}"
"""
