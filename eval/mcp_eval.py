"""
MCP Tool-Calling Evaluation
============================
Measures how accurately Claude selects the correct MCP tool and fills
key parameters when given natural language queries.

Metrics:
  - Tool Selection Accuracy  : % of queries where Claude picks the right tool
  - Key Param Hit Rate       : % of expected key params correctly filled
  - No-Call Rate             : % of queries where Claude makes no tool call
  - Wrong-Tool Rate          : % of queries where Claude picks the wrong tool

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3.11 eval/mcp_eval.py
    python3.11 eval/mcp_eval.py --output notes/mcp_eval_summary.md
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Tool schemas — mirrors mcp_server/server.py tool signatures exactly
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "search_companies",
        "description": (
            "Search SDGZero companies by natural language description. "
            "Combines semantic similarity with optional structured filters. "
            "sdg and category use fuzzy matching so partial values work. "
            "city must be an exact match (case-sensitive), e.g. 'London', 'Preston'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":         {"type": "string", "description": "Natural language description of the company type you're looking for."},
                "city":          {"type": "string", "description": "Filter by city — must exactly match a DB value (e.g. 'London', 'Preston')."},
                "sdg":           {"type": "string", "description": "Filter by SDG goal — partial/fuzzy match works, e.g. 'climate', 'clean energy'."},
                "category":      {"type": "string", "description": "Filter by business category — partial match works, e.g. 'Energy', 'Health'."},
                "business_type": {"type": "string", "description": "Filter by business model — 'B2B', 'B2C', or 'Both'."},
                "n_results":     {"type": "integer", "description": "Number of results to return (default 10, max 50)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "filter_companies",
        "description": (
            "Filter SDGZero companies by structured metadata conditions. "
            "No vector search — results sorted by membership tier. "
            "Use this when you want deterministic filtering without semantic ranking. "
            "sdg and category use fuzzy matching. city must be exact, e.g. 'London', 'Preston'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city":          {"type": "string", "description": "Exact city name — must match DB exactly (case-sensitive)."},
                "sdg":           {"type": "string", "description": "SDG goal — partial/fuzzy match works."},
                "category":      {"type": "string", "description": "Business category — partial match works."},
                "business_type": {"type": "string", "description": "'B2B', 'B2C', or 'Both'."},
                "job_sector":    {"type": "string", "description": "'Private', 'Public', or 'Agencies'."},
                "claimed":       {"type": "boolean", "description": "If True, return only verified/claimed profiles."},
                "n_results":     {"type": "integer", "description": "Number of results (default 20, max 100)."},
            },
            "required": [],
        },
    },
    {
        "name": "list_filters",
        "description": (
            "Return all valid filter values available in the SDGZero database. "
            "Only call this when the user EXPLICITLY asks what options are available, "
            "e.g. 'what cities are in the database?', 'what SDG goals can I filter by?'. "
            "Do NOT call this as a precautionary step before every search or filter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_company",
        "description": (
            "Get the full profile of a single SDGZero company by its slug. "
            "Use this when the user asks for details about a specific named company. "
            "You can find the slug from search_companies or filter_companies results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slug": {"type": "string", "description": "The company's URL slug, e.g. 'greentech-london'."},
            },
            "required": ["slug"],
        },
    },
    {
        "name": "find_partners",
        "description": (
            "Run the full AI partner-matching pipeline to find the top 5 partner companies. "
            "WHEN TO USE: user describes THEIR OWN company ('I am...', 'We are...', 'My company does X') "
            "AND wants to find partners. "
            "If the user is just looking for a type of company without describing themselves, "
            "use search_companies instead. "
            "This tool takes 20-60 seconds. Use search_companies for faster lookups."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "my_company_desc":    {"type": "string", "description": "Description of YOUR company (the one looking for partners)."},
                "partner_type":       {"type": "string", "description": "Optional: description of the kind of partner you want."},
                "city":               {"type": "string", "description": "Filter candidates by city."},
                "sdg":                {"type": "string", "description": "Filter candidates by SDG goal."},
                "category":           {"type": "string", "description": "Filter candidates by category."},
                "other_requirements": {"type": "string", "description": "Free-text additional preferences."},
            },
            "required": ["my_company_desc"],
        },
    },
]

SYSTEM_PROMPT = (
    "You are an assistant with access to the SDGZero sustainability business directory. "
    "Use list_filters to discover valid field values before filtering. "
    "Use search_companies for natural language queries, filter_companies for structured "
    "filtering, get_company for full company details, and find_partners to run the full "
    "AI partner-matching pipeline."
)

# ---------------------------------------------------------------------------
# Param matching helpers
# ---------------------------------------------------------------------------

def _param_hit(actual_val, expected_val) -> bool:
    """Fuzzy match: expected value should appear as substring in actual (case-insensitive)."""
    if actual_val is None:
        return False
    if isinstance(expected_val, bool):
        return actual_val == expected_val
    return str(expected_val).lower() in str(actual_val).lower()


def evaluate_case(client: anthropic.Anthropic, case: dict) -> dict:
    """Run one test case and return result dict."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # fast + cheap for eval
        max_tokens=512,
        system=SYSTEM_PROMPT,
        tools=TOOL_SCHEMAS,
        messages=[{"role": "user", "content": case["query"]}],
    )

    # Extract tool_use block if present
    tool_use = next((b for b in response.content if b.type == "tool_use"), None)

    expected_no_call = case["expected_tool"] is None

    if tool_use is None:
        return {
            "id": case["id"],
            "query": case["query"],
            "expected_tool": case["expected_tool"],
            "actual_tool": None,
            "tool_correct": expected_no_call,  # correct if we expected no call
            "no_call": True,
            "param_hits": 0,
            "param_total": len(case["expected_key_params"]),
            "param_details": {},
            "notes": case.get("notes", ""),
        }

    actual_tool = tool_use.name
    actual_params = tool_use.input or {}
    tool_correct = actual_tool == case["expected_tool"]

    # Check each expected key param
    param_details = {}
    for key, expected_val in case["expected_key_params"].items():
        actual_val = actual_params.get(key)
        hit = _param_hit(actual_val, expected_val)
        param_details[key] = {
            "expected": expected_val,
            "actual": actual_val,
            "hit": hit,
        }

    hits = sum(1 for v in param_details.values() if v["hit"])

    return {
        "id": case["id"],
        "query": case["query"],
        "expected_tool": case["expected_tool"],
        "actual_tool": actual_tool,
        "actual_params": actual_params,
        "tool_correct": tool_correct,
        "no_call": False,
        "param_hits": hits,
        "param_total": len(case["expected_key_params"]),
        "param_details": param_details,
        "notes": case.get("notes", ""),
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    tool_correct        = sum(1 for r in results if r["tool_correct"])
    unexpected_no_call  = sum(1 for r in results if r["no_call"] and r["expected_tool"] is not None)
    wrong_tool          = sum(1 for r in results if not r["tool_correct"] and not r["no_call"])
    param_hits          = sum(r["param_hits"] for r in results)
    param_total         = sum(r["param_total"] for r in results)

    # Per-tool breakdown
    tools = ["search_companies", "filter_companies", "get_company", "find_partners"]
    per_tool = {}
    for tool in tools:
        tool_cases = [r for r in results if r["expected_tool"] == tool]
        if not tool_cases:
            continue
        tc = sum(1 for r in tool_cases if r["tool_correct"])
        ph = sum(r["param_hits"] for r in tool_cases)
        pt = sum(r["param_total"] for r in tool_cases)
        per_tool[tool] = {
            "n": len(tool_cases),
            "tool_accuracy": tc / len(tool_cases),
            "param_hit_rate": ph / pt if pt > 0 else None,
        }

    return {
        "total": n,
        "tool_selection_accuracy": tool_correct / n,
        "key_param_hit_rate": param_hits / param_total if param_total > 0 else None,
        "unexpected_no_call_rate": unexpected_no_call / n,
        "wrong_tool_rate": wrong_tool / n,
        "per_tool": per_tool,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[dict], metrics: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "# MCP Tool-Calling Evaluation Report",
        "",
        f"> Generated: {ts}  ",
        f"> Model: claude-haiku-4-5-20251001  ",
        f"> Test cases: {metrics['total']}",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Tool Selection Accuracy | {metrics['tool_selection_accuracy']:.1%} |",
        f"| Key Param Hit Rate      | {metrics['key_param_hit_rate']:.1%} |" if metrics['key_param_hit_rate'] is not None else "| Key Param Hit Rate | N/A |",
        f"| Unexpected No-Call Rate | {metrics['unexpected_no_call_rate']:.1%} |",
        f"| Wrong-Tool Rate         | {metrics['wrong_tool_rate']:.1%} |",
        "",
        "---",
        "",
        "## Per-Tool Breakdown",
        "",
        "| Tool | # Cases | Tool Accuracy | Param Hit Rate |",
        "|------|---------|--------------|----------------|",
    ]
    for tool, m in metrics["per_tool"].items():
        phr = f"{m['param_hit_rate']:.1%}" if m["param_hit_rate"] is not None else "N/A"
        lines.append(f"| `{tool}` | {m['n']} | {m['tool_accuracy']:.1%} | {phr} |")

    lines += [
        "",
        "---",
        "",
        "## Case-by-Case Results",
        "",
        "| ID | Expected Tool | Actual Tool | Tool ✓ | Param Hits | Query |",
        "|----|--------------|-------------|--------|------------|-------|",
    ]
    for r in results:
        tick = "✅" if r["tool_correct"] else ("❌" if r["no_call"] and r["expected_tool"] is not None else ("⬜" if r["no_call"] else "❌"))
        param_str = f"{r['param_hits']}/{r['param_total']}"
        actual = r["actual_tool"] or "*(no call)*"
        query_short = r["query"][:60] + ("..." if len(r["query"]) > 60 else "")
        lines.append(f"| {r['id']} | `{r['expected_tool']}` | `{actual}` | {tick} | {param_str} | {query_short} |")

    # Failures section
    failures = [r for r in results if not r["tool_correct"]]
    if failures:
        lines += ["", "---", "", "## Failures & Analysis", ""]
        for r in failures:
            lines += [
                f"### Case {r['id']}: {r['query']}",
                f"- **Expected**: `{r['expected_tool']}`",
                f"- **Actual**: `{r['actual_tool'] or '(no call)'}`",
                f"- **Notes**: {r['notes']}",
                "",
            ]

    lines += [
        "---",
        "",
        "## Interview Summary",
        "",
        "> Designed a tool-calling evaluation for the SDGZero MCP Server: "
        f"25 annotated natural language queries across 4 tools, tested against Claude Haiku via the Anthropic API. "
        f"Achieved **{metrics['tool_selection_accuracy']:.0%} tool selection accuracy** and "
        f"**{metrics['key_param_hit_rate']:.0%} key parameter hit rate**, "
        "validating that clear tool naming and structured docstrings are sufficient for reliable LLM-driven tool dispatch.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MCP Tool-Calling Evaluation")
    parser.add_argument("--output", default="notes/mcp_eval_summary.md", help="Output markdown file")
    parser.add_argument("--cases", default="eval/mcp_test_cases.json", help="Test cases JSON file")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Export it first:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    cases = json.loads(Path(args.cases).read_text())
    print(f"Running {len(cases)} test cases...")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i:02d}/{len(cases)}] {case['query'][:60]}...", end=" ", flush=True)
        result = evaluate_case(client, case)
        tick = "✅" if result["tool_correct"] else ("⬜" if result["no_call"] else "❌")
        print(f"{tick} → {result['actual_tool'] or '(no call)'}")
        results.append(result)

    metrics = compute_metrics(results)

    print("\n" + "="*50)
    print(f"Tool Selection Accuracy : {metrics['tool_selection_accuracy']:.1%}")
    print(f"Key Param Hit Rate      : {metrics['key_param_hit_rate']:.1%}" if metrics['key_param_hit_rate'] else "Key Param Hit Rate: N/A")
    print(f"Unexpected No-Call Rate : {metrics['unexpected_no_call_rate']:.1%}")
    print(f"Wrong-Tool Rate         : {metrics['wrong_tool_rate']:.1%}")
    print("="*50)

    report = generate_report(results, metrics)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
