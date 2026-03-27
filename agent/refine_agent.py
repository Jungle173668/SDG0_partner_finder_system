"""
Refine Agent — interprets user feedback (liked/disliked companies + free text)
and outputs structured parameter changes for a new pipeline run.
"""

import json
import logging
import os
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


def _get_llm():
    """Create a refine-specific LLM instance with higher token limit."""
    provider = os.getenv("LLM_PROVIDER", "gemini")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            max_output_tokens=1024,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            max_tokens=1024,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            temperature=0,
            num_predict=1024,
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REFINE_SYSTEM = """\
You are a search parameter optimizer for a business partner matching system.

Given current search parameters, user feedback on result companies (liked/disliked), \
and optional free text from the user, output ONLY the fields that need to change.

## Changeable fields

A-class (any free text value):
- partner_type_desc — type of partner the user is looking for
- other_requirements — constraints or exclusions (append, do not delete existing)
- user_company_desc — only change if user explicitly asks

B-class (structured, only change if user EXPLICITLY mentions them):
- city — single city name
- sdg_tags — list of SDG tag strings
- categories — list of category strings
- business_type — single value (e.g. "B2B", "B2C", "Both")
- job_sector — single value (e.g. "Private", "Public", "Third Sector")
- company_size — single value (e.g. "Micro", "SME")
- claimed — boolean (true = verified profiles only)

## CRITICAL rules

1. ONLY output fields the user is asking to change. Never echo current values as changes.
2. B-class fields: include ONLY if the user's text or \
feedback clearly and explicitly requests a change to that specific field. \
Do NOT infer B-class changes from liked/disliked companies.
3. Liked companies → update partner_type_desc only. \
   If liked companies have no clear common industry, ignore them.
4. Disliked companies → append exclusion to other_requirements only.
5. If nothing meaningful can be changed, output action "unclear".
6. city mapping: phrases like "in London", "based in Manchester", "switch to Edinburgh", \
"only [city name]" → set `city` to the matching value from the valid cities list. \
Do NOT put city names in other_requirements or partner_type_desc.
7. sdg_tags mapping: any mention of SDG numbers or SDG names (e.g. "SDG 13", \
"Climate Action", "SDG7", "Good Health", "add SDG X as a filter") → set `sdg_tags` to \
the matching name(s) from the valid values list. Do NOT put SDG names in partner_type_desc.
8. categories mapping: phrases like "only tech companies", "focus on energy sector", \
"retail companies" where the meaning matches a category in the valid list → set `categories`. \
Do NOT put category names in partner_type_desc.
9. business_type mapping: "B2B only", "consumer-facing", "B2C companies", "works with both" \
→ set `business_type` to the matching value (e.g. "B2B", "B2C", "Both").
10. job_sector mapping: "private sector only", "public sector", "third sector", \
"charities only", "NGOs", "exclude public sector" → set `job_sector` to the matching value. \
Do NOT put this in other_requirements.
11. company_size mapping: "small companies", "SMEs", "micro businesses", "large corporations", \
"startups" → set `company_size` to the closest matching value from the valid list.
12. claimed mapping: "verified only", "certified profiles", "claimed businesses", \
"must be verified" → set `claimed` to true. Do NOT put this in other_requirements.
13. partner_type_desc MUST describe the PARTNER company's type — NEVER the user's own company type. \
The user's type comes from user_company_desc — do NOT copy it into partner_type_desc. \
e.g. user_company_desc = "we are a media company", user says "not skin-care" → \
do NOT set partner_type_desc to "media company".
14. Exclusion/negation about partner type ("not X", "don't need to be X", "exclude X", \
"they do not have to be X") → set partner_type_desc to "" (empty string, clears the constraint). \
Do NOT replace it with the user's own company type or any unrelated type. \
Only set partner_type_desc to a new value when the user explicitly says what they DO want \
(e.g. "find travel companies", "switch to retail brands").
15. Removing a B-class filter ("remove SDG tag", "no city filter", "clear the sector filter", \
"remove all filters") → set that field to null or [] in changes. \
e.g. "remove SDG tag" → {"sdg_tags": null}, "remove city filter" → {"city": null}.

## Output (JSON only, no markdown)

Refine case — include optional `modes` dict for B-class fields where mode matters:
{"action":"refine","changes":{...},"modes":{"city":"hard","sdg_tags":"soft"},"summary":"...≤10 words"}

Mode rules:
- "must", "only", "hard filter", "require", "strictly" → "hard"
- "prefer", "ideally", "if possible", "preferred", "boost", "better", "would be nice", "nice to have", "not must", "not required", "doesn't have to", "don't have to", "not necessary" → "soft"
- No qualifier mentioned → omit from modes (code will default to "hard")

Unclear case:
{"action":"unclear","summary":"Could not understand your request — please rephrase"}

## Few-shot examples (follow these exactly)

User: "must be in London"
→ {"action":"refine","changes":{"city":"London"},"modes":{"city":"hard"},"summary":"city → London (must)"}

User: "prefer London if possible"
→ {"action":"refine","changes":{"city":"London"},"modes":{"city":"soft"},"summary":"city → London (preferred)"}

User: "better in Blackburn, not must"
→ {"action":"refine","changes":{"city":"Blackburn"},"modes":{"city":"soft"},"summary":"city → Blackburn (preferred)"}

User: "would be nice if they're in Manchester"
→ {"action":"refine","changes":{"city":"Manchester"},"modes":{"city":"soft"},"summary":"city → Manchester (preferred)"}

User: "add SDG 13 Climate Action as a filter"
→ {"action":"refine","changes":{"sdg_tags":["Climate Action"]},"summary":"added SDG 13 filter"}

User: "prefer SDG 7 companies"
→ {"action":"refine","changes":{"sdg_tags":["Affordable And Clean Energy"]},"modes":{"sdg_tags":"soft"},"summary":"SDG 7 preferred"}

User: "only show B2B companies"
→ {"action":"refine","changes":{"business_type":"B2B"},"modes":{"business_type":"hard"},"summary":"business type → B2B only"}

User: "ideally private sector"
→ {"action":"refine","changes":{"job_sector":"Private"},"modes":{"job_sector":"soft"},"summary":"job sector → private (preferred)"}

User: "SMEs only"
→ {"action":"refine","changes":{"company_size":"SME"},"modes":{"company_size":"hard"},"summary":"company size → SME only"}

User: "verified profiles only"
→ {"action":"refine","changes":{"claimed":true},"modes":{"claimed":"hard"},"summary":"verified profiles only"}

User: "switch to Edinburgh, B2B, verified only"
→ {"action":"refine","changes":{"city":"Edinburgh","business_type":"B2B","claimed":true},"modes":{"city":"hard","business_type":"hard","claimed":"hard"},"summary":"city → Edinburgh, B2B, verified"}

User: "I want travel companies not marketing agencies"
→ {"action":"refine","changes":{"partner_type_desc":"travel companies"},"summary":"partner type → travel companies"}

User: "exclude beauty and cosmetics industry"
→ {"action":"refine","changes":{"other_requirements":"exclude beauty and cosmetics industry"},"summary":"excluded beauty/cosmetics"}

User: "they do not need to be skin-care companies, must be in London"
(current partner_type_desc = "skin-care companies", user_company_desc = "we are a media company")
→ {"action":"refine","changes":{"partner_type_desc":"","city":"London"},"modes":{"city":"hard"},"summary":"broadened partner type, city → London (must)"}

User: "remove the SDG tag filter"
→ {"action":"refine","changes":{"sdg_tags":null},"summary":"removed SDG tag filter"}

User: "remove city filter, I don't need London anymore"
→ {"action":"refine","changes":{"city":null},"summary":"removed city filter"}
"""

_REFINE_HUMAN = """\
Current params:
{current_params}

Valid values for structured fields:
{schema}

Liked companies (want more like these):
{liked}

Disliked companies (want fewer like these):
{disliked}

User text:
{user_text}

Output JSON now.\
"""


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_refine_agent(
    current_params: dict,
    schema: dict,
    liked: list[dict],
    disliked: list[dict],
    user_text: str,
) -> dict:
    """
    Run the refine LLM call.

    Returns a dict:
      { "action": "refine", "changes": {...}, "summary": "..." }
    or
      { "action": "unclear", "summary": "..." }
    """
    llm = _get_llm()

    def _fmt_companies(companies: list[dict]) -> str:
        if not companies:
            return "None"
        lines = []
        for c in companies:
            parts = [c.get("name", "Unknown")]
            if c.get("categories"):
                parts.append(f"category={c['categories']}")
            if c.get("sdg_tags"):
                parts.append(f"sdg={c['sdg_tags']}")
            if c.get("business_type"):
                parts.append(f"type={c['business_type']}")
            if c.get("city"):
                parts.append(f"city={c['city']}")
            lines.append("- " + ", ".join(parts))
        return "\n".join(lines)

    human_content = _REFINE_HUMAN.format(
        current_params=json.dumps(current_params, ensure_ascii=False),
        schema=json.dumps(schema, ensure_ascii=False),
        liked=_fmt_companies(liked),
        disliked=_fmt_companies(disliked),
        user_text=user_text.strip() if user_text else "None",
    )

    messages = [
        SystemMessage(content=_REFINE_SYSTEM),
        HumanMessage(content=human_content),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if model wraps anyway
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"refine_agent: JSON parse failed — raw={raw[:300]}")
        result = {"action": "unclear", "summary": "Could not parse response — please rephrase"}

    action = result.get("action", "unclear")
    changes = result.get("changes", {})
    logger.info(f"refine_agent: action={action} changes={list(changes.keys())}")
    return result
