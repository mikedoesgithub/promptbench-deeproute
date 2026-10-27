"""Prompt templates and system routing instructions for fast System 1 classification."""

SYSTEM_ROUTING_PROMPT = """You are a high-speed, sub-100ms System 1 Agentic Router. Your job is to classify user queries, extract multifaceted metadata, or decompose compound tasks into specialist subagents. Respond strictly in valid JSON without explanations or markdown formatting outside JSON.
"""

ATOMIC_PROMPT_TEMPLATE = """Classify the following user prompt into exactly ONE primary category and domain from the provided taxonomy.
Also assign a calibrated confidence score between 0.0 and 1.0, and optionally provide top 3 candidates.

Output JSON format:
{{
  "predicted_category": "<one of the 54 granular categories>",
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
  "complexity": "atomic" | "sequential_pipeline" | "parallel_map_reduce" | "dynamic_cyclic",
  "safety_flags": ["benign_standard" | "indirect_prompt_injection" | "exfiltration_attempt" | "ambiguous_underspecified"],
  "tool_dependencies": ["bash" | "git" | "python_repl" | "file_io" | "web_search"],
  "confidence": <float 0.0 to 1.0>
}}

User Prompt:
"{prompt}"
"""

COMPOUND_PROMPT_TEMPLATE = """Determine whether the following user prompt is compound (contains multiple distinct sub-tasks requiring different specialized subagents) or atomic (a single coherent task for one agent).

Available subagents:
- coder: software engineering, implementation, refactoring, tests
- researcher: literature search, factual retrieval, competitive intelligence
- writer: prose writing, essays, emails, marketing copy
- math_specialist: mathematical proofs, algebra, calculus, statistics, logic puzzles
- data_engineer: SQL authoring, dataframes, ETL pipelines, visualizations
- sysadmin: shell scripts, Linux security, container orchestration, log analysis
- finance_officer: corporate budgeting, ledger math, rate calculations, purchase memos
- hardware_engineer: embedded firmware, schematics, BOM optimization, PCB layout
- coordinator: triage, formatting, scheduling, asking clarifying questions

If compound: split into atomic sub-tasks, assign the subagent and intent category for each, and specify the prerequisite dependencies (IDs of tasks that must finish before this task starts).
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
