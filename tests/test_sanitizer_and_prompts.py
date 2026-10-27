"""Unit tests for category sanitization, subagent sanitization, and dynamic prompt builders."""

import pytest
import yaml
from pathlib import Path

from src.models.base import sanitize_category, sanitize_subagent
from src.models.prompts import (
    build_atomic_system_prompt,
    build_compound_system_prompt,
    build_multilabel_system_prompt,
)


@pytest.fixture
def taxonomy_and_subagents():
    root = Path(__file__).resolve().parent.parent
    with open(root / "configs" / "taxonomy.yaml", "r", encoding="utf-8") as f:
        tax = yaml.safe_load(f)
    with open(root / "configs" / "subagents.yaml", "r", encoding="utf-8") as f:
        subs = yaml.safe_load(f)
    return tax, subs


def test_sanitize_category_exact_match():
    valid = ["swe_code_generation", "swe_code_refactoring", "sys_shell_scripting"]
    assert sanitize_category("swe_code_generation", valid) == "swe_code_generation"
    assert sanitize_category("  swe_code_refactoring  ", valid) == "swe_code_refactoring"


def test_sanitize_category_messy_strings():
    valid = ["swe_code_generation", "swe_code_refactoring", "sys_shell_scripting"]
    # Messy capitalization and punctuation
    assert sanitize_category("SWE - Code Generation", valid) == "swe_code_generation"
    assert sanitize_category("swe_code_generation.py", valid) == "swe_code_generation"
    assert sanitize_category("category: swe_code_generation", valid) == "swe_code_generation"
    assert sanitize_category("swe-code-generation", valid) == "swe_code_generation"
    assert sanitize_category("SWE_CODE_GENERATION", valid) == "swe_code_generation"


def test_sanitize_category_fuzzy_matches():
    valid = ["swe_code_generation", "prose_longform_essay", "sys_container_orchestration"]
    # Minor typos
    assert sanitize_category("swe_code_gen", valid) == "swe_code_generation"
    assert sanitize_category("longform_essay", valid) == "prose_longform_essay"
    assert sanitize_category("container_orchestration", valid) == "sys_container_orchestration"


def test_sanitize_subagent_aliases_and_messy():
    valid = [
        "coder", "researcher", "writer", "math_specialist", "data_engineer",
        "sysadmin", "finance_officer", "hardware_engineer", "coordinator",
        "browser_navigator", "vision_analyst", "mcp_orchestrator",
        "safety_sentinel", "triage_interviewer"
    ]
    assert sanitize_subagent("coder", valid) == "coder"
    assert sanitize_subagent("Coder", valid) == "coder"
    assert sanitize_subagent("software_engineer", valid) == "coder"
    assert sanitize_subagent("subagent: browser_navigator", valid) == "browser_navigator"
    assert sanitize_subagent("mcp", valid) == "mcp_orchestrator"
    assert sanitize_subagent("safety", valid) == "safety_sentinel"
    assert sanitize_subagent("interviewer", valid) == "triage_interviewer"
    assert sanitize_subagent("vision", valid) == "vision_analyst"


def test_build_atomic_system_prompt_dynamic(taxonomy_and_subagents):
    tax, _ = taxonomy_and_subagents
    prompt = build_atomic_system_prompt(tax)
    assert "System 1 Agentic Router" in prompt
    assert "software_engineering" in prompt
    assert "swe_code_generation" in prompt
    assert "browser_automation" in prompt
    assert "mcp_protocol_routing" in prompt
    assert "108 Granular Categories" in prompt


def test_build_compound_system_prompt_dynamic(taxonomy_and_subagents):
    tax, subs = taxonomy_and_subagents
    prompt = build_compound_system_prompt(subs, tax)
    assert "Compound Task Decomposition" in prompt
    # All 14 subagents must be present
    for sub in [
        "coder", "researcher", "writer", "math_specialist", "data_engineer",
        "sysadmin", "finance_officer", "hardware_engineer", "coordinator",
        "browser_navigator", "vision_analyst", "mcp_orchestrator",
        "safety_sentinel", "triage_interviewer"
    ]:
        assert f"'{sub}'" in prompt


def test_build_multilabel_system_prompt_dynamic(taxonomy_and_subagents):
    tax, _ = taxonomy_and_subagents
    prompt = build_multilabel_system_prompt(tax)
    assert "Multi-Aspect Tagging Engine" in prompt
    assert "software_engineering" in prompt
    assert "mcp_protocol_routing" in prompt
    assert "single_turn_direct" in prompt


def test_promptbench_cli_help():
    import subprocess
    import sys
    res = subprocess.run([sys.executable, "-m", "run_benchmark", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "PromptBench-DeepRoute" in res.stdout
    assert "--model" in res.stdout
    assert "--oracle" in res.stdout

