"""
ReportAgent — Step 5 implementation.

Responsibilities:
  1. Read scored_companies (Top-5) from ScoringAgent.
  2. Compute per-company radar dimensions (no LLM calls — pure data).
  3. Render a self-contained HTML report.
  4. Save the HTML file to reports/{session_id}.html for immediate browser preview.
  5. Write report path to AgentState.

HTML structure (two tabs):
  Results tab  — company cards with match bar, quality badge, SDG tags, reasoning
  Analysis tab — SDG coverage matrix, at-a-glance cards, radar chart (Chart.js CDN)

Design goals:
  - Zero external runtime dependencies (CSS inline, Chart.js loaded from CDN)
  - Open directly in browser during development — no server needed
  - Same HTML file can later be served by FastAPI / embedded in Next.js iframe
"""

import json
import logging
import os
from pathlib import Path

from agent.state import AgentState

logger = logging.getLogger(__name__)

_REPORTS_DIR = Path(__file__).parent.parent / "reports"

# ---------------------------------------------------------------------------
# Radar dimension helpers
# ---------------------------------------------------------------------------

_SOURCE_DEPTH = {
    "db":                30,   # Layer 1 only: database text
    "db+tavily_extract": 95,   # Layer 1 + 2: database + website crawl (best)
    "db+tavily_search":  65,   # Layer 1 + 3: database + search fallback
    "tavily_extract":    80,
    "tavily_search":     50,
}


def _radar_scores(company: dict, research_source: str) -> dict:
    """
    Compute 0-100 scores for each radar axis.

    Axes:
        match_pct      — cross_encoder_score × 100
        sdg_coverage   — number of unique SDG tags (tagged + predicted), capped at 100
        research_depth — quality of research source
        certified      — claimed == "Yes" → 100, else 0
        sector_fit     — proxy: same as match_pct (true sector fit needs more data)
    """
    sdg_tagged    = company.get("sdg_tags", "") or ""
    sdg_predicted = company.get("predicted_sdg_tags", "") or ""
    # Count unique SDG mentions across both fields
    all_sdg_text  = f"{sdg_tagged} {sdg_predicted}"
    sdg_count     = len({t.strip() for t in all_sdg_text.replace(",", " ").split() if t.strip()})
    sdg_score     = min(sdg_count * 12, 100)

    claimed = str(company.get("claimed", "")).lower()
    certified_score = 100 if claimed in ("yes", "true", "1") else 0

    match_pct = round(company.get("cross_encoder_score", 0) * 100, 1)

    return {
        "match_pct":      match_pct,
        "sdg_coverage":   sdg_score,
        "research_depth": _SOURCE_DEPTH.get(research_source, 30),
        "certified":      certified_score,
        "sector_fit":     match_pct,   # proxy until Phase 3
    }


# ---------------------------------------------------------------------------
# SDG tag parser
# ---------------------------------------------------------------------------

def _parse_sdg_tags(company: dict) -> tuple[list[str], list[str]]:
    """
    Return (tagged_list, predicted_list) — short labels like 'SDG 7'.
    """
    def _split(raw: str) -> list[str]:
        if not raw:
            return []
        return [t.strip() for t in raw.replace(",", "|").split("|") if t.strip()]

    tagged    = _split(company.get("sdg_tags", "") or "")
    predicted = _split(company.get("predicted_sdg_tags", "") or "")
    # Remove from predicted anything already in tagged
    predicted = [p for p in predicted if p not in tagged]
    return tagged, predicted


# ---------------------------------------------------------------------------
# HTML snippets
# ---------------------------------------------------------------------------

_QUALITY_COLOR = {
    "strong":   "#16a34a",   # green
    "partial":  "#d97706",   # amber
    "fallback": "#6b7280",   # gray
}
_QUALITY_LABEL = {
    "strong":   "Strong match",
    "partial":  "Partial match",
    "fallback":  "Fallback",
}


def _badge(quality: str) -> str:
    color = _QUALITY_COLOR.get(quality, "#6b7280")
    label = _QUALITY_LABEL.get(quality, quality.title())
    icon  = "✓" if quality == "strong" else ("△" if quality == "fallback" else "◐")
    return (
        f'<span class="badge" style="background:{color}20;color:{color};'
        f'border:1px solid {color}40">{icon} {label}</span>'
    )


def _sdg_chip(label: str, kind: str) -> str:
    """kind: 'tagged' | 'predicted' | 'missing'"""
    styles = {
        "tagged":    "background:#dcfce7;color:#166534;border:1px solid #86efac",
        "predicted": "background:#f0fdf4;color:#4ade80;border:1px dashed #86efac",
        "missing":   "background:#f3f4f6;color:#9ca3af;border:1px solid #e5e7eb",
    }
    markers = {"tagged": "✓", "predicted": "≈", "missing": "·"}
    style  = styles.get(kind, styles["missing"])
    marker = markers.get(kind, "")

    # Convert full name → "SDG N · Short name" displayed directly on chip
    sdg_key = _sdg_number(label)   # e.g. "SDG 7"
    if sdg_key:
        short_name = _SDG_NAMES.get(sdg_key, "")
        display = f"{sdg_key} · {short_name}" if short_name else sdg_key
        tooltip = label  # hover shows original full name from DB
    else:
        display = label[:22] + "…" if len(label) > 24 else label
        tooltip = label

    return f'<span class="chip" style="{style}" title="{tooltip}">{marker} {display}</span>'


def _progress_bar(pct: float, quality: str) -> str:
    color = _QUALITY_COLOR.get(quality, "#6b7280")
    return (
        f'<div class="bar-bg">'
        f'<div class="bar-fill" style="width:{pct}%;background:{color}"></div>'
        f'</div>'
        f'<span class="bar-pct">{int(pct)}%</span>'
    )


def _contact_links(company: dict) -> str:
    parts = []
    if company.get("url"):
        parts.append(f'<a href="{company["url"]}" target="_blank" class="link-btn sdgzero">SDGZero</a>')
    if company.get("website"):
        parts.append(f'<a href="{company["website"]}" target="_blank" class="link-btn">Website</a>')
    linkedin = company.get("linkedin", "") or ""
    if linkedin and "linkedin.com" in linkedin.lower():
        parts.append(f'<a href="{linkedin}" target="_blank" class="link-btn linkedin">LinkedIn</a>')
    if company.get("phone"):
        parts.append(f'<span class="link-btn phone">📞 {company["phone"]}</span>')
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Results tab: company cards
# ---------------------------------------------------------------------------

def _render_card(rank: int, company: dict, research_source: str) -> str:
    name    = company.get("name", "Unknown")
    city    = company.get("city", "")
    cats    = company.get("categories", "")
    biz     = company.get("business_type", "")
    claimed = str(company.get("claimed", "")).lower() in ("yes", "true", "1")
    quality = company.get("match_quality", "fallback")
    score   = company.get("cross_encoder_score", 0)
    pct     = round(score * 100, 1)
    reasoning = company.get("reasoning", "")

    tagged, predicted = _parse_sdg_tags(company)

    meta_parts = [p for p in [city, cats, biz, "Certified" if claimed else ""] if p]
    meta = " · ".join(meta_parts)

    sdg_chips = "".join(_sdg_chip(t, "tagged") for t in tagged[:5])
    sdg_chips += "".join(_sdg_chip(t, "predicted") for t in predicted[:3])

    fallback_note = ""
    if quality == "fallback":
        fallback_note = (
            '<div class="fallback-note">'
            'No closer match found within filters — shown for reference only.'
            '</div>'
        )

    return f"""
    <div class="card">
      <div class="card-header">
        <div>
          <span class="rank">#{rank}</span>
          <span class="company-name">{name}</span>
          <span class="meta">{meta}</span>
        </div>
        {_badge(quality)}
      </div>
      <div class="bar-row">
        {_progress_bar(pct, quality)}
      </div>
      <div class="sdg-row">{sdg_chips}</div>
      {fallback_note}
      <div class="reasoning">{reasoning}</div>
      <div class="contact-row">{_contact_links(company)}</div>
    </div>"""


# ---------------------------------------------------------------------------
# Analysis tab: SDG matrix
# ---------------------------------------------------------------------------

_ALL_SDGS = [
    "SDG 1", "SDG 2", "SDG 3", "SDG 4", "SDG 5", "SDG 6", "SDG 7",
    "SDG 8", "SDG 9", "SDG 10", "SDG 11", "SDG 12", "SDG 13",
    "SDG 14", "SDG 15", "SDG 16", "SDG 17",
]
_SDG_NAMES = {
    "SDG 1": "No Poverty", "SDG 2": "Zero Hunger", "SDG 3": "Good Health",
    "SDG 4": "Quality Education", "SDG 5": "Gender Equality",
    "SDG 6": "Clean Water", "SDG 7": "Affordable Energy",
    "SDG 8": "Decent Work", "SDG 9": "Innovation",
    "SDG 10": "Reduced Inequalities", "SDG 11": "Sustainable Cities",
    "SDG 12": "Responsible Consumption", "SDG 13": "Climate Action",
    "SDG 14": "Life Below Water", "SDG 15": "Life On Land",
    "SDG 16": "Peace & Justice", "SDG 17": "Partnerships",
}

# Full SDG name (as stored in DB) → canonical "SDG N" key
_SDG_FULLNAME_MAP = {
    "no poverty":                                    "SDG 1",
    "zero hunger":                                   "SDG 2",
    "good health and well-being":                    "SDG 3",
    "good health and well being":                    "SDG 3",
    "quality education":                             "SDG 4",
    "gender equality":                               "SDG 5",
    "clean water and sanitation":                    "SDG 6",
    "affordable and clean energy":                   "SDG 7",
    "decent work and economic growth":               "SDG 8",
    "industry innovation and infrastructure":        "SDG 9",
    "industry, innovation and infrastructure":       "SDG 9",
    "industry innovation cities and communities":    "SDG 9",   # DB variant
    "reduced inequalities":                          "SDG 10",
    "reduced inequality":                            "SDG 10",
    "sustainable cities and communities":            "SDG 11",
    "responsible consumption and production":        "SDG 12",
    "climate action":                                "SDG 13",
    "life below water":                              "SDG 14",
    "life on land":                                  "SDG 15",
    "peace justice and strong institutions":         "SDG 16",
    "peace, justice and strong institutions":        "SDG 16",
    "partnerships for the goals":                    "SDG 17",
    "partnerships":                                  "SDG 17",
}


def _sdg_number(label: str) -> str:
    """Map any SDG label (full name, 'SDG 7', slug, etc.) to canonical 'SDG N' key."""
    import re
    low = label.lower().strip()

    # 1. Exact full-name match
    if low in _SDG_FULLNAME_MAP:
        return _SDG_FULLNAME_MAP[low]

    # 2. Partial full-name match (covers slight variations)
    for name, key in _SDG_FULLNAME_MAP.items():
        if name in low or low in name:
            return key

    # 3. Already in "SDG N" format
    for key in _ALL_SDGS:
        if key.lower() in low:
            return key

    # 4. Bare number fallback  e.g. "sdg7", "goal 13"
    m = re.search(r'\b(\d{1,2})\b', low)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 17:
            return f"SDG {n}"

    return ""


def _render_sdg_matrix(scored: list[dict]) -> str:
    # Build per-company SDG sets
    company_sdgs = []
    for c in scored:
        tagged, predicted = _parse_sdg_tags(c)
        tagged_keys    = {_sdg_number(t) for t in tagged    if _sdg_number(t)}
        predicted_keys = {_sdg_number(p) for p in predicted if _sdg_number(p)}
        company_sdgs.append((c.get("name", "?")[:12], tagged_keys, predicted_keys))

    # Only show SDGs that appear in at least one company
    relevant = [s for s in _ALL_SDGS
                if any(s in t or s in p for _, t, p in company_sdgs)]
    if not relevant:
        return "<p>No SDG data available.</p>"

    # Header row
    headers = "".join(
        f'<th title="{_SDG_NAMES.get(s, s)}">{s}</th>' for s in relevant
    )
    company_headers = "".join(
        f'<th class="co-name">{name}</th>' for name, _, _ in company_sdgs
    )

    # Data rows — one row per SDG goal, one col per company
    rows = ""
    for sdg in relevant:
        cells = ""
        for _, tagged_keys, predicted_keys in company_sdgs:
            if sdg in tagged_keys:
                cells += '<td><span class="matrix-check tagged">✓</span></td>'
            elif sdg in predicted_keys:
                cells += '<td><span class="matrix-check predicted">≈</span></td>'
            else:
                cells += '<td><span class="matrix-check missing">·</span></td>'
        sdg_label = f"{sdg} {_SDG_NAMES.get(sdg, '')}"
        rows += f"<tr><td class='sdg-label'>{sdg_label}</td>{cells}</tr>"

    return f"""
    <table class="matrix-table">
      <thead>
        <tr>
          <th>SDG goal</th>
          {company_headers}
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="matrix-legend">
      <span class="tagged">✓ Tagged</span>
      <span class="predicted">≈ Predicted</span>
      <span class="missing">· Not present</span>
    </div>"""


# ---------------------------------------------------------------------------
# Analysis tab: at-a-glance cards
# ---------------------------------------------------------------------------

def _render_glance_cards(scored: list[dict]) -> str:
    cards = ""
    for i, c in enumerate(scored, 1):
        pct     = int(c.get("cross_encoder_score", 0) * 100)
        size    = c.get("company_size", "—") or "—"
        claimed = str(c.get("claimed", "")).lower() in ("yes", "true", "1")
        cert    = "Yes" if claimed else "No"
        cert_color = "#16a34a" if claimed else "#6b7280"
        quality = c.get("match_quality", "fallback")
        bar_color = _QUALITY_COLOR.get(quality, "#6b7280")
        cards += f"""
        <div class="glance-card">
          <div class="glance-rank">#{i}</div>
          <div class="glance-name">{c.get('name','?')[:18]}</div>
          <div class="glance-row"><span>Match</span><strong>{pct}%</strong></div>
          <div class="glance-row"><span>Size</span><strong>{size}</strong></div>
          <div class="glance-row"><span>Cert.</span>
            <strong style="color:{cert_color}">{cert}</strong></div>
          <div class="glance-bar" style="background:{bar_color}"></div>
        </div>"""
    return f'<div class="glance-row-wrap">{cards}</div>'


# ---------------------------------------------------------------------------
# Analysis tab: radar chart (Chart.js)
# ---------------------------------------------------------------------------

def _render_radar(scored: list[dict], research: dict) -> str:
    labels = ["Match %", "SDG coverage", "Research depth", "Certified", "Sector fit"]
    colors = ["#16a34a", "#2563eb", "#ea580c", "#dc2626", "#6b7280"]

    datasets = []
    for i, c in enumerate(scored):
        slug   = c.get("slug") or c.get("id", "")
        source = research.get(slug, {}).get("source", "db")
        dims   = _radar_scores(c, source)
        data   = [
            dims["match_pct"],
            dims["sdg_coverage"],
            dims["research_depth"],
            dims["certified"],
            dims["sector_fit"],
        ]
        color = colors[i % len(colors)]
        datasets.append({
            "label": f"#{i+1} {c.get('name','?')[:14]}",
            "data": data,
            "borderColor": color,
            "backgroundColor": color + "20",
            "pointBackgroundColor": color,
        })

    datasets_json = json.dumps(datasets)
    labels_json   = json.dumps(labels)

    axis_hints = {
        "Match %":        "How closely this company matches your search — higher is better",
        "SDG coverage":   "How many sustainability goals (SDGs) this company actively works on",
        "Research depth": "How detailed our information is: website data available (high) vs database only (low)",
        "Certified":      "Has this company verified and claimed their SDGZero profile?",
        "Sector fit":     "How well this company's industry aligns with what you're looking for",
    }
    hints_json = json.dumps(axis_hints)

    return f"""
    <div class="radar-wrap">
      <canvas id="radarChart"></canvas>
    </div>
    <div class="radar-legend">
      {"".join(f'<div class="radar-hint"><strong>{k}</strong> — {v}</div>' for k, v in axis_hints.items())}
    </div>
    <script>
    (function() {{
      const hints = {hints_json};
      const ctx = document.getElementById('radarChart');
      if (!ctx) return;
      new Chart(ctx, {{
        type: 'radar',
        data: {{
          labels: {labels_json},
          datasets: {datasets_json}
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: true,
          scales: {{
            r: {{
              min: 0, max: 100,
              ticks: {{ stepSize: 25, font: {{ size: 12 }}, backdropColor: 'transparent' }},
              pointLabels: {{ font: {{ size: 14, weight: '600' }}, padding: 12 }}
            }}
          }},
          plugins: {{
            legend: {{ position: 'bottom', labels: {{ font: {{ size: 13 }}, padding: 16 }} }},
            tooltip: {{
              callbacks: {{
                title: function(items) {{
                  const label = items[0]?.label || '';
                  return label + (hints[label] ? ': ' + hints[label] : '');
                }}
              }}
            }}
          }}
        }}
      }});
    }})();
    </script>"""


# ---------------------------------------------------------------------------
# Search criteria chips
# ---------------------------------------------------------------------------

def _render_criteria(filters: dict, fallback_lvl: int,
                     n_candidates: int, n_scored: int,
                     user_company_desc: str = "",
                     partner_type_desc: str = "") -> str:
    chips = ""
    for k, v in (filters or {}).items():
        if isinstance(v, list):
            for item in v:
                sdg_key = _sdg_number(item)
                if sdg_key:
                    short = _SDG_NAMES.get(sdg_key, "")
                    label = f"{sdg_key} · {short}" if short else sdg_key
                    chips += f'<span class="criteria-chip hard" title="{item}">{label} ✓</span>'
                else:
                    chips += f'<span class="criteria-chip hard">{item} ✓</span>'
        elif v is True:
            label = {"claimed": "Certified"}.get(k, k.replace("_", " ").title())
            chips += f'<span class="criteria-chip hard">{label} ✓</span>'
        else:
            chips += f'<span class="criteria-chip hard">{v} ✓</span>'

    warning = ""
    if fallback_lvl == 1:
        warning = '<div class="criteria-warn-row">⚠ Some filters were relaxed to find these results</div>'
    elif fallback_lvl == 2:
        warning = '<div class="criteria-warn-row">⚠ No matches within your filters — showing semantic matches only</div>'

    summary = f'<span class="criteria-summary">{n_candidates} candidates → {n_scored} recommended</span>'

    # Build the chips row — if no filters, show a "no filters" label
    if chips:
        chips_row = f'<div class="criteria-chips">{chips}</div>'
    else:
        chips_row = '<div class="criteria-chips"><span class="criteria-chip">No filters applied — semantic search only</span></div>'

    # Company description snippet (shown when provided)
    desc_section = ""
    if user_company_desc.strip():
        short_desc = user_company_desc.strip()[:200]
        if len(user_company_desc.strip()) > 200:
            short_desc += "…"
        desc_section = f"""
      <div class="criteria-desc-row">
        <span class="criteria-label">Your company:</span>
        <span class="criteria-desc-text">{short_desc}</span>
      </div>"""

    # Partner type — what kind of partner the user is looking for
    hyde_section = ""
    if partner_type_desc.strip():
        short_pt = partner_type_desc.strip()[:220]
        if len(partner_type_desc.strip()) > 220:
            short_pt += "…"
        hyde_section = f"""
      <div class="criteria-desc-row">
        <span class="criteria-label">Searched for:</span>
        <span class="criteria-desc-text criteria-desc-hyde">{short_pt}</span>
      </div>"""

    return f"""
    <div class="criteria-card">
      <div class="criteria-top">
        {chips_row}
        <div>{summary}</div>
      </div>
      {desc_section}
      {hyde_section}
      {warning}
    </div>"""


# ---------------------------------------------------------------------------
# Full HTML page
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f9fafb; color: #111827; font-size: 14px; }
.container { max-width: 860px; margin: 0 auto; padding: 24px 16px; }
h1 { font-size: 20px; font-weight: 700; color: #111827; margin-bottom: 4px; }
.subtitle { color: #6b7280; font-size: 13px; margin-bottom: 20px; }

/* Tabs */
.tabs { display: flex; gap: 0; border-bottom: 2px solid #e5e7eb; margin-bottom: 24px; }
.tab-btn { padding: 10px 20px; cursor: pointer; border: none; background: none;
           font-size: 14px; color: #6b7280; border-bottom: 2px solid transparent;
           margin-bottom: -2px; transition: all 0.15s; }
.tab-btn.active { color: #111827; border-bottom-color: #111827; font-weight: 600; }
.tab-btn:hover { color: #374151; }
.tab-pane { display: none; }
.tab-pane.active { display: block; }

/* Criteria card */
.criteria-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
                 padding: 14px 18px; margin-bottom: 20px; }
.criteria-top { display: flex; justify-content: space-between; align-items: center;
                flex-wrap: wrap; gap: 8px; }
.criteria-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.criteria-chip { padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 500;
                 background: #f3f4f6; color: #374151; border: 1px solid #d1d5db;
                 cursor: default; }
.criteria-chip.hard { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }
.criteria-warn-row { margin-top: 10px; padding: 6px 10px; border-radius: 8px; font-size: 12px;
                     background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
.criteria-summary { font-size: 13px; color: #6b7280; white-space: nowrap; }
.criteria-desc-row { margin-top: 10px; display: flex; gap: 8px; align-items: baseline;
                     flex-wrap: wrap; }
.criteria-label { font-size: 11px; font-weight: 700; color: #9ca3af; text-transform: uppercase;
                  letter-spacing: 0.06em; white-space: nowrap; }
.criteria-desc-text { font-size: 13px; color: #374151; line-height: 1.5; }
.criteria-desc-hyde { color: #6b7280; font-style: italic; }

/* Cards */
.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
        padding: 20px; margin-bottom: 16px; }
.card-header { display: flex; justify-content: space-between; align-items: flex-start;
               margin-bottom: 12px; }
.rank { font-size: 12px; color: #9ca3af; margin-right: 8px; }
.company-name { font-size: 17px; font-weight: 700; color: #111827; }
.meta { display: block; font-size: 12px; color: #6b7280; margin-top: 3px; }
.badge { padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
         white-space: nowrap; }
.bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.bar-bg { flex: 1; height: 6px; background: #f3f4f6; border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; transition: width 0.3s; }
.bar-pct { font-size: 13px; font-weight: 600; color: #374151; width: 36px; text-align: right; }
.sdg-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.chip { padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 500; }
.fallback-note { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px;
                 padding: 8px 12px; font-size: 12px; color: #6b7280;
                 margin-bottom: 12px; }
.reasoning { font-size: 14px; line-height: 1.6; color: #374151; margin-bottom: 14px; }
.contact-row { display: flex; gap: 8px; }
.link-btn { padding: 6px 14px; border-radius: 8px; font-size: 12px; font-weight: 500;
            text-decoration: none; background: #f3f4f6; color: #374151;
            border: 1px solid #e5e7eb; transition: background 0.15s; }
.link-btn:hover { background: #e5e7eb; }
.link-btn.linkedin { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }
.link-btn.sdgzero  { background: #f0fdf4; color: #15803d; border-color: #bbf7d0; }

/* Analysis tab */
.section-title { font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
                 text-transform: uppercase; color: #9ca3af; margin-bottom: 14px; }
.matrix-wrap { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
               padding: 20px; margin-bottom: 24px; overflow-x: auto; }
.matrix-table { border-collapse: collapse; width: 100%; font-size: 13px; }
.matrix-table th { padding: 8px 10px; text-align: center; font-weight: 600;
                   border-bottom: 1px solid #e5e7eb; color: #374151; }
.matrix-table th.co-name { font-size: 12px; max-width: 80px; word-break: break-word; }
.matrix-table td { padding: 7px 10px; text-align: center; border-bottom: 1px solid #f3f4f6; }
.sdg-label { text-align: left !important; font-size: 12px; color: #374151;
             white-space: nowrap; padding-right: 16px !important; }
.matrix-check { font-size: 15px; }
.matrix-check.tagged { color: #16a34a; }
.matrix-check.predicted { color: #86efac; }
.matrix-check.missing { color: #d1d5db; }
.matrix-legend { margin-top: 12px; display: flex; gap: 20px; font-size: 12px; }
.matrix-legend .tagged { color: #16a34a; font-weight: 600; }
.matrix-legend .predicted { color: #86efac; font-weight: 600; }
.matrix-legend .missing { color: #9ca3af; }

/* At a glance */
.glance-row-wrap { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }
.glance-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
               padding: 14px 16px; min-width: 130px; flex: 1; position: relative;
               overflow: hidden; }
.glance-rank { font-size: 11px; color: #9ca3af; margin-bottom: 4px; }
.glance-name { font-size: 13px; font-weight: 700; color: #111827; margin-bottom: 10px; }
.glance-row { display: flex; justify-content: space-between; font-size: 12px;
              color: #6b7280; margin-bottom: 4px; }
.glance-row strong { color: #111827; }
.glance-bar { position: absolute; bottom: 0; left: 0; right: 0; height: 3px; }

/* Radar */
.radar-section { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
                 padding: 24px; }
.radar-wrap { width: 100%; max-width: 640px; margin: 0 auto; }
.radar-legend { margin-top: 20px; border-top: 1px solid #f3f4f6; padding-top: 14px;
                display: flex; flex-direction: column; gap: 6px; }
.radar-hint { font-size: 12px; color: #6b7280; line-height: 1.5; }
.radar-hint strong { color: #374151; }
.link-btn.phone { cursor: default; background: #f3f4f6; color: #374151;
                  border-color: #e5e7eb; }
"""

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SDGZero Partner Report — {session_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>{css}</style>
</head>
<body>
<div class="container">
  <h1>SDGZero Partner Finder</h1>
  <p class="subtitle">Top {n_scored} partner recommendations &nbsp;·&nbsp; {search_method} search</p>

  {criteria_bar}

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('results', this)">Results</button>
    <button class="tab-btn" onclick="switchTab('analysis', this)">Analysis</button>
  </div>

  <!-- Results tab -->
  <div id="tab-results" class="tab-pane active">
    {cards}
  </div>

  <!-- Analysis tab -->
  <div id="tab-analysis" class="tab-pane">
    <p class="section-title">SDG Coverage Matrix</p>
    <div class="matrix-wrap">
      {sdg_matrix}
    </div>

    <p class="section-title">At a Glance</p>
    {glance_cards}

    <p class="section-title">Dimension Comparison</p>
    <div class="radar-section">
      {radar}
    </div>
  </div>
</div>

<script>
function switchTab(name, btn) {{
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# ReportAgent LangGraph node
# ---------------------------------------------------------------------------

def report_agent_node(state: AgentState) -> dict:
    """
    ReportAgent node — renders the HTML report and saves it to disk.

    Reads:
        scored_companies, candidate_companies, research_results,
        filters, search_fallback_level, search_method, session_id

    Writes:
        report  — path to the saved HTML file (str)
    """
    scored      = state.get("scored_companies", [])
    candidates  = state.get("candidate_companies", [])
    research    = state.get("research_results", {})
    filters     = state.get("filters", {})
    fallback_lvl = state.get("search_fallback_level", 0)
    search_method = state.get("search_method", "semantic")
    session_id  = state.get("session_id", "unknown")
    user_desc        = state.get("user_company_desc", "")
    partner_type_desc = state.get("partner_type_desc", "")

    if not scored:
        logger.warning("ReportAgent: no scored_companies — generating empty report")

    # Render pieces
    criteria_bar = _render_criteria(
        filters, fallback_lvl, len(candidates), len(scored),
        user_company_desc=user_desc,
        partner_type_desc=partner_type_desc,
    )

    cards = ""
    for i, company in enumerate(scored, 1):
        slug   = company.get("slug") or company.get("id", "")
        source = research.get(slug, {}).get("source", "db")
        cards += _render_card(i, company, source)

    sdg_matrix  = _render_sdg_matrix(scored)
    glance_cards = _render_glance_cards(scored)
    radar       = _render_radar(scored, research)

    html = _HTML_TEMPLATE.format(
        css=_CSS,
        session_id=session_id,
        search_method=search_method,
        fallback_lvl=fallback_lvl,
        n_scored=len(scored),
        criteria_bar=criteria_bar,
        cards=cards or "<p>No recommendations generated.</p>",
        sdg_matrix=sdg_matrix,
        glance_cards=glance_cards,
        radar=radar,
    )

    # Save to reports/
    _REPORTS_DIR.mkdir(exist_ok=True)
    report_path = _REPORTS_DIR / f"{session_id}.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"ReportAgent: saved → {report_path}")
    print(f"\n  Report saved → {report_path.resolve()}")
    print(f"  Open in browser: file://{report_path.resolve()}")

    return {"report": str(report_path.resolve())}
