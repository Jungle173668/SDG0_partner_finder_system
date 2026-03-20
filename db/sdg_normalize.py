"""
SDG tag normalisation — canonical names for all 17 UN Sustainable Development Goals.

Fixes known bad tags that originate from data-entry errors or ML prediction drift,
mapping them to the correct official SDG name(s).

Usage:
    from db.sdg_normalize import normalize_sdg_tags
    clean = normalize_sdg_tags(["Industry Innovation Cities And Communities", "Climate Action"])
    # → ["Industry, Innovation And Infrastructure", "Climate Action"]
"""

# Official SDG names (Title Case, no commas — DB stores tags as comma-separated strings
# so tag names must not contain commas themselves)
OFFICIAL_SDG_NAMES = [
    "No Poverty",                             # SDG 1
    "Zero Hunger",                            # SDG 2
    "Good Health And Well-Being",             # SDG 3
    "Quality Education",                      # SDG 4
    "Gender Equality",                        # SDG 5
    "Clean Water And Sanitation",             # SDG 6
    "Affordable And Clean Energy",            # SDG 7
    "Decent Work And Economic Growth",        # SDG 8
    "Industry Innovation And Infrastructure", # SDG 9 — no comma (comma is the field separator)
    "Reduced Inequalities",                   # SDG 10
    "Sustainable Cities And Communities",     # SDG 11
    "Responsible Consumption And Production", # SDG 12
    "Climate Action",                         # SDG 13
    "Life Below Water",                       # SDG 14
    "Life On Land",                           # SDG 15
    "Peace Justice And Strong Institutions",  # SDG 16
    "Partnerships For The Goals",             # SDG 17
]

# Lower-cased set for fast membership check
_OFFICIAL_LOWER = {s.lower() for s in OFFICIAL_SDG_NAMES}

# Display names — maps storage name → official UN name (for UI / reports)
# Only entries that differ from OFFICIAL_SDG_NAMES need to be listed here.
DISPLAY_NAMES: dict[str, str] = {
    "Industry Innovation And Infrastructure": "Industry, Innovation and Infrastructure",
}


def to_display_name(storage_name: str) -> str:
    """Return the official UN display name for a stored SDG tag."""
    return DISPLAY_NAMES.get(storage_name, storage_name)

# Explicit bad-tag → correct-tag(s) mapping.
# Each value is a list so one bad tag can expand into multiple correct ones.
_CORRECTIONS: dict[str, list[str]] = {
    # SDGZero platform bug: merged SDG 9 + SDG 11 into one string
    "industry innovation cities and communities": [
        "Industry Innovation And Infrastructure",
    ],
    # Alternate spellings / punctuation variants
    "industry innovation and infrastructure": [
        "Industry Innovation And Infrastructure",
    ],
    "peace, justice and strong institutions": [
        "Peace Justice And Strong Institutions",
    ],
    "reduced inequality": [
        "Reduced Inequalities",
    ],
}


def normalize_sdg_tags(tags: list[str]) -> list[str]:
    """
    Normalise a list of SDG tag strings:
      1. Apply explicit corrections for known bad tags.
      2. Drop anything that doesn't match an official SDG name.
      3. Deduplicate while preserving order.

    Args:
        tags: Raw SDG tag strings from DB or scraper.

    Returns:
        Cleaned list of official SDG name strings.
    """
    seen: set[str] = set()
    result: list[str] = []

    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue

        tag_lower = tag.lower()

        # Apply correction mapping first
        if tag_lower in _CORRECTIONS:
            replacements = _CORRECTIONS[tag_lower]
        elif tag_lower in _OFFICIAL_LOWER:
            # Already correct — find the canonical casing
            replacements = [t for t in OFFICIAL_SDG_NAMES if t.lower() == tag_lower]
        else:
            # Unknown tag — drop it with a warning (logged by caller if needed)
            replacements = []

        for replacement in replacements:
            key = replacement.lower()
            if key not in seen:
                seen.add(key)
                result.append(replacement)

    return result
