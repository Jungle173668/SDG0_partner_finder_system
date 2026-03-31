"""
LangGraph pipeline definition — Multi-Agent Pipeline.

Graph structure:
    SearchAgent → ResearchAgent → ScoringAgent → ReportAgent → END

Step 1:   SearchAgent fully implemented, others are pass-through stubs.
Step 2:   SearchAgent upgraded with tool routing + three-level fallback.
Step 3:   ResearchAgent implemented (Tavily three-layer strategy).
Step 4: ✅ ScoringAgent implemented (Cross-encoder + LLM reasoning).
Step 5:   ReportAgent implemented (Markdown report + outreach drafts).

Usage:
    from agent.graph import build_graph
    pipeline = build_graph()
    result = pipeline.invoke({
        "user_company_desc": "We provide carbon audit services...",
        "filters": {"city": "London"},
        "other_requirements": "prefer B2B, under 50 employees",
        "session_id": "x7k2m9",
    })
    print(result["candidate_companies"])
"""

import logging
from typing import Optional
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.search_agent import search_agent_node
from agent.research_agent import research_agent_node
from agent.scoring_agent import scoring_agent_node
from agent.report_agent import report_agent_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stub nodes — pass state through unchanged
# Replaced one by one in Steps 3-5
# ---------------------------------------------------------------------------



def scoring_agent_stub(state: AgentState) -> dict:
    """
    ScoringAgent placeholder (Step 4).

    Will: Cross-encoder rerank Top-10 → Top-5,
          LLM generate reasoning for each,
          write scored_companies list.
    """
    logger.info("ScoringAgent: stub (Step 4 — not yet implemented)")
    # Pass through top-5 candidates with placeholder score
    candidates = state.get("candidate_companies", [])[:5]
    scored = [
        {**c, "cross_encoder_score": c.get("similarity", 0.0), "reasoning": "(Step 4 pending)"}
        for c in candidates
    ]
    return {"scored_companies": scored}


def report_agent_stub(state: AgentState) -> dict:
    """
    ReportAgent placeholder (Step 5).

    Will: format scored_companies into Markdown report,
          generate outreach message drafts,
          write report string.
    """
    logger.info("ReportAgent: stub (Step 5 — not yet implemented)")
    scored = state.get("scored_companies", [])

    lines = ["# SDGZero Partner Finder — Results\n"]
    lines.append(f"**Candidates analysed:** {len(state.get('candidate_companies', []))} companies\n")
    lines.append(f"**HyDE description:** {state.get('hypothetical_partner_desc', '')[:200]}...\n")
    lines.append("\n---\n")
    lines.append("## Top Matches\n")

    for i, company in enumerate(scored, 1):
        score = company.get("cross_encoder_score", 0.0)
        lines.append(f"### #{i} {company.get('name', 'Unknown')}")
        lines.append(f"- **Match score:** {score:.3f}")
        lines.append(f"- **Location:** {company.get('city', '')} {company.get('country', '')}")
        lines.append(f"- **SDGs:** {company.get('sdg_tags', '') or company.get('predicted_sdg_tags', '') or 'N/A'}")
        lines.append(f"- **Website:** {company.get('website', 'N/A')}")
        lines.append(f"- **LinkedIn:** {company.get('linkedin', 'N/A')}")
        lines.append("")

    return {"report": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> "CompiledGraph":
    """
    Build and compile the Multi-Agent LangGraph pipeline.

    Returns:
        Compiled LangGraph graph ready for invoke() / stream() calls.

    To extend:
        Replace *_stub nodes with real implementations.
        Add conditional edges for branching (e.g. zero-result fallback).
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("search", search_agent_node)
    graph.add_node("research", research_agent_node)
    graph.add_node("scoring", scoring_agent_node)
    graph.add_node("report", report_agent_node)

    # Linear pipeline: search → research → scoring → report → END
    graph.set_entry_point("search")
    graph.add_edge("search", "research")
    graph.add_edge("research", "scoring")
    graph.add_edge("scoring", "report")
    graph.add_edge("report", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience: single call that runs the whole pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    user_company_desc: str,
    partner_type_desc: str = "",
    filters: Optional[dict] = None,
    soft_filters: Optional[dict] = None,
    other_requirements: str = "",
    session_id: Optional[str] = None,
    allow_global_fallback: bool = False,
) -> AgentState:
    """
    Run the full Multi-Agent Pipeline and return the final state.

    Args:
        user_company_desc:   User's company description (required, most important input).
        partner_type_desc:   Explicit description of the target partner type (optional).
        filters:             SearchFilters dict (optional).
        other_requirements:  Free-text additional criteria (optional).
        session_id:          Session ID for URL sharing (optional, auto-generated if None).

    Returns:
        Final AgentState with all fields populated.

    Example:
        from agent.graph import run_pipeline
        state = run_pipeline(
            user_company_desc="We provide carbon audit and net-zero consulting...",
            filters={"city": "London", "business_type": "B2B"},
        )
        for c in state["candidate_companies"]:
            print(c["name"], c["similarity"])
    """
    import secrets

    pipeline = build_graph()

    initial_state: AgentState = {
        "session_id": session_id or secrets.token_urlsafe(4)[:6],
        "user_company_desc": user_company_desc,
        "partner_type_desc": partner_type_desc,
        "filters": filters or {},
        "soft_filters": soft_filters or {},
        "other_requirements": other_requirements,
        "allow_global_fallback": allow_global_fallback,
        "errors": [],
    }

    logger.info(f"Pipeline starting — session_id={initial_state['session_id']}")
    result = pipeline.invoke(initial_state)
    logger.info("Pipeline complete.")
    return result
