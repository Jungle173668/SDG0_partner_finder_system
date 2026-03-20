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
import re
from pathlib import Path

from agent.state import AgentState

logger = logging.getLogger(__name__)

_REPORTS_DIR = Path(__file__).parent.parent / "reports"
_ICONS_DIR   = Path(__file__).parent.parent / "static" / "sdg-icons"


# ---------------------------------------------------------------------------
# Radar dimension helpers
# ---------------------------------------------------------------------------

_SOURCE_DEPTH = {
    "db":                30,
    "db+tavily_extract": 95,
    "db+tavily_search":  65,
    "tavily_extract":    80,
    "tavily_search":     50,
}


def _radar_scores(company: dict, research_source: str) -> dict:
    sdg_tagged    = company.get("sdg_tags", "") or ""
    sdg_predicted = company.get("predicted_sdg_tags", "") or ""
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
        "sector_fit":     match_pct,
    }


# ---------------------------------------------------------------------------
# SDG tag parser + helpers
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
    "industry innovation cities and communities":    "SDG 9",
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
    """Map any SDG label to canonical 'SDG N' key."""
    low = label.lower().strip()
    if low in _SDG_FULLNAME_MAP:
        return _SDG_FULLNAME_MAP[low]
    for name, key in _SDG_FULLNAME_MAP.items():
        if name in low or low in name:
            return key
    for key in _ALL_SDGS:
        if key.lower() in low:
            return key
    m = re.search(r'\b(\d{1,2})\b', low)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 17:
            return f"SDG {n}"
    return ""


def _parse_sdg_tags(company: dict) -> tuple[list[str], list[str]]:
    def _split(raw: str) -> list[str]:
        if not raw:
            return []
        return [t.strip() for t in raw.replace(",", "|").split("|") if t.strip()]

    tagged    = _split(company.get("sdg_tags", "") or "")
    predicted = _split(company.get("predicted_sdg_tags", "") or "")
    tagged_lower = {t.lower().strip() for t in tagged}
    predicted = [p for p in predicted if p.lower().strip() not in tagged_lower]
    return tagged, predicted


def _sdg_icon_src(sdg_key: str) -> str:
    """
    Return path to local SDG icon.

    Uses /static/... (server-absolute) when REPORT_STATIC_BASE env is set to 'server',
    otherwise falls back to ../static/... (relative, for file:// direct open).
    """
    m = re.search(r'(\d+)', sdg_key)
    if not m:
        return ""
    n = m.group(1)
    base = os.getenv("REPORT_STATIC_BASE", "relative")
    if base == "server":
        return f"/static/sdg-icons/{n}.png"
    return f"../static/sdg-icons/{n}.png"


# ---------------------------------------------------------------------------
# HTML snippet helpers
# ---------------------------------------------------------------------------

def _badge(quality: str) -> str:
    cls   = {"strong": "b-strong", "partial": "b-partial", "fallback": "b-fallback"}.get(quality, "b-fallback")
    label = {"strong": "Strong match", "partial": "Partial match", "fallback": "↓ Fallback"}.get(quality, quality.title())
    return f'<span class="badge {cls}">{label}</span>'


def _sdg_icon_pill(label: str, kind: str) -> str:
    """kind: 'tagged' | 'predicted'"""
    sdg_key = _sdg_number(label)
    if not sdg_key:
        return ""
    short = _SDG_NAMES.get(sdg_key, sdg_key)
    icon_src = _sdg_icon_src(sdg_key)
    suffix = " ★" if kind == "predicted" else ""
    cls = "sdg-icon-pill pred" if kind == "predicted" else "sdg-icon-pill hit"
    img = f'<img src="{icon_src}" alt="{sdg_key}">' if icon_src else ""
    tooltip = f"{sdg_key} · {short}{' (predicted)' if kind == 'predicted' else ''}"
    return f'<div class="{cls}" title="{tooltip}" data-tip="{tooltip}">{img}{sdg_key}{suffix}</div>'


def _bar_fill_class(quality: str) -> str:
    return {"strong": "", "partial": " amber", "fallback": " gray"}.get(quality, " gray")


def _contact_links(company: dict) -> str:
    parts = []
    if company.get("url"):
        parts.append(f'<a href="{company["url"]}" target="_blank" class="clink">SDGZero</a>')
    if company.get("website"):
        parts.append(f'<a href="{company["website"]}" target="_blank" class="clink">Website</a>')
    linkedin = company.get("linkedin", "") or ""
    if linkedin and "linkedin.com" in linkedin.lower():
        parts.append(f'<a href="{linkedin}" target="_blank" class="clink">LinkedIn</a>')
    if company.get("phone"):
        parts.append(f'<span class="clink" style="cursor:default;color:var(--sdg-muted)">📞 {company["phone"]}</span>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Results tab: company cards
# ---------------------------------------------------------------------------

def _render_card(rank: int, company: dict, research_source: str) -> str:
    name      = company.get("name", "Unknown")
    city      = company.get("city", "")
    cats      = company.get("categories", "")
    biz       = company.get("business_type", "")
    claimed   = str(company.get("claimed", "")).lower() in ("yes", "true", "1")
    quality   = company.get("match_quality", "fallback")
    score     = company.get("cross_encoder_score", 0)
    pct       = round(score * 100, 1)
    reasoning = company.get("reasoning", "")

    tagged, predicted = _parse_sdg_tags(company)

    meta_parts = [p for p in [city, cats, biz, "Verified ✓" if claimed else ""] if p]
    meta = " · ".join(meta_parts)

    sdg_pills  = "".join(_sdg_icon_pill(t, "tagged")    for t in tagged[:5])
    sdg_pills += "".join(_sdg_icon_pill(t, "predicted") for t in predicted[:3])

    rank_style = "" if quality != "fallback" else 'style="background:#b0bec5"'
    card_cls   = "co-card fallback" if quality == "fallback" else "co-card"
    bar_cls    = _bar_fill_class(quality)

    fallback_notice = ""
    if quality == "fallback":
        fallback_notice = """
      <div class="fallback-notice">
        <div class="fn-dot"></div>
        No closer match found within filters — shown for reference only.
      </div>"""

    return f"""
    <div class="{card_cls}">
      <div class="card-head">
        <div class="rank-bubble" {rank_style}>{rank}</div>
        <div style="flex:1">
          <div class="card-title">{name}</div>
          <div class="card-sub">{meta}</div>
        </div>
        {_badge(quality)}
      </div>
      <div class="bar-row">
        <div class="bar-bg"><div class="bar-fill{bar_cls}" style="width:{pct}%"></div></div>
        <span class="bar-pct">{int(pct)}%</span>
      </div>
      <div class="sdg-row">{sdg_pills}</div>
      {fallback_notice}
      <div class="reason">{reasoning}</div>
      <div class="card-foot">
        <div class="links">{_contact_links(company)}</div>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Analysis tab: SDG matrix
# ---------------------------------------------------------------------------

def _render_sdg_matrix(scored: list[dict]) -> str:
    company_sdgs = []
    for c in scored:
        tagged, predicted = _parse_sdg_tags(c)
        tagged_keys    = {_sdg_number(t) for t in tagged    if _sdg_number(t)}
        predicted_keys = {_sdg_number(p) for p in predicted if _sdg_number(p)}
        company_sdgs.append((c.get("name", "?")[:12], tagged_keys, predicted_keys))

    relevant = [s for s in _ALL_SDGS
                if any(s in t or s in p for _, t, p in company_sdgs)]
    if not relevant:
        return "<p>No SDG data available.</p>"

    company_headers = "".join(
        f'<th class="cc">{name}</th>' for name, _, _ in company_sdgs
    )

    rows = ""
    for sdg in relevant:
        sdg_label = f"{sdg} {_SDG_NAMES.get(sdg, '')}"
        cells = ""
        for _, tagged_keys, predicted_keys in company_sdgs:
            if sdg in tagged_keys:
                cells += '<td class="cell"><span class="dot dot-h">✓</span></td>'
            elif sdg in predicted_keys:
                cells += '<td class="cell"><span class="dot dot-p">★</span></td>'
            else:
                cells += '<td class="cell"><span class="dot dot-m">·</span></td>'
        rows += f'<tr><td class="sl">{sdg_label}</td>{cells}</tr>'

    return f"""
    <table class="mx">
      <thead>
        <tr>
          <th style="min-width:160px">SDG goal</th>
          {company_headers}
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="mx-leg">
      <span><span class="ld" style="background:#e6f7f5;border:1px solid #00a896"></span>Tagged</span>
      <span><span class="ld" style="background:#e6f7f5;opacity:.6"></span>Predicted ★</span>
      <span><span class="ld" style="background:#f7f9fc;border:1px solid #e0e0e0"></span>Not present</span>
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
        cert_color = "var(--sdg-teal)" if claimed else "#b0bec5"
        cert_label = "Yes" if claimed else "No"
        cards += f"""
        <div class="pc">
          <div class="pc-rank">#{i}</div>
          <div class="pc-name">{c.get('name','?')[:18]}</div>
          <div class="pc-row"><span class="pc-k">Match</span><span class="pc-v">{pct}%</span></div>
          <div class="pc-row"><span class="pc-k">Size</span><span class="pc-v" style="font-size:9px">{size}</span></div>
          <div class="pc-row"><span class="pc-k">Verified</span>
            <span class="pc-v" style="color:{cert_color}">{cert_label}</span></div>
          <div class="pb-bg"><div class="pb-fill" style="width:{pct}%"></div></div>
        </div>"""
    return f'<div class="pg">{cards}</div>'


# ---------------------------------------------------------------------------
# Analysis tab: radar chart (Chart.js)
# ---------------------------------------------------------------------------

def _render_radar(scored: list[dict], research: dict) -> str:
    labels = ["Match %", "SDG coverage", "Research depth", "Verified", "Sector fit"]
    colors = ["#00a896", "#0a1f3c", "#f4a11d", "#e8453c", "#5a6a80"]

    datasets = []
    for i, c in enumerate(scored):
        slug   = c.get("slug") or c.get("id", "")
        source = research.get(slug, {}).get("source", "db")
        dims   = _radar_scores(c, source)
        # Normalise to 0-10 scale for cleaner radar
        data = [
            round(dims["match_pct"] / 10, 1),
            round(dims["sdg_coverage"] / 10, 1),
            round(dims["research_depth"] / 10, 1),
            10 if dims["certified"] == 100 else 4,
            round(dims["sector_fit"] / 10, 1),
        ]
        color = colors[i % len(colors)]
        datasets.append({
            "label": f"#{i+1} {c.get('name','?')[:14]}",
            "data": data,
            "borderColor": color,
            "backgroundColor": color + "20",
            "borderWidth": 1.5,
            "pointRadius": 3,
            "pointBackgroundColor": color,
        })

    datasets_json = json.dumps(datasets)
    labels_json   = json.dumps(labels)

    legend_items = ""
    for i, c in enumerate(scored):
        color = colors[i % len(colors)]
        name  = c.get("name", "?")[:16]
        legend_items += f'<div class="rli"><span class="rld" style="background:{color}"></span>#{i+1} {name}</div>'

    return f"""
    <div class="radar-wrap">
      <div class="radar-cw"><canvas id="radarChart"></canvas></div>
      <div class="rleg">{legend_items}</div>
    </div>
    <script>
    (function() {{
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
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            r: {{
              min: 0, max: 10,
              ticks: {{ display: false }},
              grid: {{ color: 'rgba(10,31,60,0.08)' }},
              angleLines: {{ color: 'rgba(10,31,60,0.08)' }},
              pointLabels: {{ font: {{ size: 11 }}, color: '#5a6a80' }}
            }}
          }}
        }}
      }});
    }})();
    </script>"""


# ---------------------------------------------------------------------------
# Search summary bar
# ---------------------------------------------------------------------------

def _pill_for_filter(k: str, v, mode: str) -> str:
    """Render one filter condition as a pill (must/preferred)."""
    css = "pill hard" if mode == "hard" else "pill soft"
    label_suffix = " (must)" if mode == "hard" else " (preferred)"
    if isinstance(v, list):
        out = ""
        for item in v:
            sdg_key = _sdg_number(item)
            label = sdg_key if sdg_key else item
            out += f'<span class="{css}">{label}{label_suffix}</span>'
        return out
    if v is True or v == "Yes":
        label = {"claimed": "Verified"}.get(k, k.replace("_", " ").title())
        return f'<span class="{css}">{label}{label_suffix}</span>'
    return f'<span class="{css}">{v}{label_suffix}</span>'


def _render_criteria(filters: dict, fallback_lvl: int,
                     n_candidates: int, n_scored: int,
                     user_company_desc: str = "",
                     partner_type_desc: str = "",
                     soft_filters: dict = None) -> str:
    pills = ""
    for k, v in (filters or {}).items():
        pills += _pill_for_filter(k, v, "hard")
    for k, v in (soft_filters or {}).items():
        pills += _pill_for_filter(k, v, "soft")

    if not pills:
        pills = '<span class="pill">Semantic search only</span>'

    # Specific fallback message — level 1 relaxes sdg_tags + claimed
    fallback_tag = ""
    if fallback_lvl == 1:
        relaxed = []
        if (filters or {}).get("sdg_tags"):
            relaxed.append("SDG tags")
        if (filters or {}).get("claimed"):
            relaxed.append("Verified")
        if (filters or {}).get("categories"):
            relaxed.append("Categories")
        relaxed_str = " & ".join(relaxed) if relaxed else "some conditions"
        fallback_tag = f'<span class="fallback-tag">⚠ {relaxed_str} relaxed — not enough matches</span>'
    elif fallback_lvl == 2:
        fallback_tag = '<span class="fallback-tag">⚠ All filters dropped — semantic only</span>'

    desc_row = ""
    if user_company_desc.strip():
        short = user_company_desc.strip()[:200]
        if len(user_company_desc.strip()) > 200:
            short += "…"
        desc_row += f'<div class="summary-desc-row"><span class="summary-label">Your company</span><span class="summary-desc">{short}</span></div>'

    if partner_type_desc.strip():
        short = partner_type_desc.strip()[:200]
        if len(partner_type_desc.strip()) > 200:
            short += "…"
        desc_row += f'<div class="summary-desc-row"><span class="summary-label">Searched for</span><span class="summary-desc" style="font-style:italic;opacity:.8">{short}</span></div>'

    return f"""
    <div class="summary-bar">
      <div>
        <div class="summary-label">Your search</div>
        <div class="pills">{pills}</div>
        {desc_row}
      </div>
      <div class="summary-right">
        <span class="summary-count">{n_candidates} candidates · {n_scored} recommended</span>
        {fallback_tag}
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# CSS + HTML template
# ---------------------------------------------------------------------------

_CSS = """
*{box-sizing:border-box;margin:0;padding:0;}
:root {
  --sdg-navy: #0a1f3c;
  --sdg-teal: #00a896;
  --sdg-teal-light: #e6f7f5;
  --sdg-teal-mid: #b3e8e3;
  --sdg-amber: #f4a11d;
  --sdg-amber-light: #fef4e0;
  --sdg-red: #e8453c;
  --sdg-green: #3aaa35;
  --sdg-green-light: #eaf6e9;
  --sdg-surface: #f7f9fc;
  --sdg-border: rgba(10,31,60,0.12);
  --sdg-border-mid: rgba(10,31,60,0.22);
  --sdg-text: #0a1f3c;
  --sdg-muted: #5a6a80;
  --sdg-radius: 10px;
  --sdg-radius-sm: 6px;
}
body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--sdg-surface);
       color: var(--sdg-text); font-size: 14px; }
.container { max-width: 860px; margin: 0 auto; padding: 24px 16px 40px; }

/* Brand header */
.brand-header { display:flex; align-items:center; justify-content:space-between;
  padding:14px 0; border-bottom:2px solid var(--sdg-teal); margin-bottom:20px; }
.brand-logo { display:flex; align-items:center; gap:10px; }
.brand-tag { font-size:11px; color:var(--sdg-muted); margin-top:1px; }

/* Tabs */
.tabs { display:flex; border-bottom:1px solid var(--sdg-border); margin-bottom:20px; }
.tab-btn { padding:9px 20px; font-size:13px; font-weight:600; color:var(--sdg-muted);
  cursor:pointer; border:none; background:none;
  border-bottom:2.5px solid transparent; margin-bottom:-1px; }
.tab-btn.active { color:var(--sdg-teal); border-bottom-color:var(--sdg-teal); }
.tab-pane { display:none; }
.tab-pane.active { display:block; }

/* Summary bar */
.summary-bar { background:var(--sdg-navy); border-radius:var(--sdg-radius);
  padding:14px 18px; margin-bottom:18px;
  display:flex; align-items:flex-start; justify-content:space-between;
  gap:12px; flex-wrap:wrap; color:white; }
.summary-label { font-size:11px; color:rgba(255,255,255,0.55); font-weight:600;
  letter-spacing:.05em; text-transform:uppercase; margin-bottom:7px; }
.pills { display:flex; flex-wrap:wrap; gap:6px; }
.pill { font-size:11px; padding:3px 10px; border-radius:20px;
  border:1px solid rgba(255,255,255,0.25); color:rgba(255,255,255,0.85); }
.pill.hard { background:var(--sdg-teal); border-color:var(--sdg-teal); color:white; font-weight:600; }
.pill.soft { background:rgba(255,255,255,0.15); border-color:rgba(255,255,255,0.5); color:rgba(255,255,255,0.85); font-weight:600; }
.summary-right { display:flex; flex-direction:column; align-items:flex-end; gap:6px; }
.summary-count { font-size:12px; color:rgba(255,255,255,0.65); white-space:nowrap; }
.fallback-tag { font-size:11px; background:var(--sdg-amber); color:var(--sdg-navy);
  border-radius:20px; padding:3px 10px; font-weight:700; }
.summary-desc-row { margin-top:8px; display:flex; gap:8px; align-items:baseline; flex-wrap:wrap; }
.summary-desc { font-size:12px; color:rgba(255,255,255,0.75); line-height:1.5; }

/* Section label */
.sec-label { font-size:11px; font-weight:700; color:var(--sdg-muted);
  text-transform:uppercase; letter-spacing:.07em; margin:0 0 12px; }

/* Company card */
.co-card { background:white; border:1px solid var(--sdg-border); border-radius:var(--sdg-radius);
  padding:18px 20px; margin-bottom:10px; transition:border-color .15s, box-shadow .15s; }
.co-card:hover { border-color:var(--sdg-teal); box-shadow:0 2px 12px rgba(0,168,150,.1); }
.co-card.fallback { opacity:.85; }
.card-head { display:flex; align-items:flex-start; gap:10px; margin-bottom:12px; }
.rank-bubble { width:28px; height:28px; border-radius:50%; background:var(--sdg-navy);
  color:white; font-size:12px; font-weight:700; display:flex; align-items:center;
  justify-content:center; flex-shrink:0; margin-top:2px; }
.card-title { font-size:15px; font-weight:700; color:var(--sdg-navy); margin-bottom:2px; }
.card-sub { font-size:12px; color:var(--sdg-muted); }
.badge { font-size:11px; padding:3px 10px; border-radius:20px; font-weight:700; white-space:nowrap; flex-shrink:0; }
.b-strong  { background:var(--sdg-green-light); color:var(--sdg-green); }
.b-partial { background:var(--sdg-amber-light); color:#c07800; }
.b-fallback{ background:var(--sdg-surface); color:var(--sdg-muted); border:1px solid var(--sdg-border); }

/* Match bar */
.bar-row { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
.bar-bg { flex:1; height:5px; background:var(--sdg-border); border-radius:3px; overflow:hidden; }
.bar-fill { height:100%; border-radius:3px; background:var(--sdg-teal); }
.bar-fill.amber { background:var(--sdg-amber); }
.bar-fill.gray  { background:#b0bec5; }
.bar-pct { font-size:12px; font-weight:700; color:var(--sdg-navy); min-width:30px; text-align:right; }

/* SDG icon pills */
.sdg-row { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:12px; align-items:center; }
.sdg-icon-pill { display:flex; align-items:center; gap:4px; font-size:11px; font-weight:600;
  border-radius:5px; padding:3px 8px 3px 4px; }
.sdg-icon-pill img { width:18px; height:18px; border-radius:3px; }
.sdg-icon-pill.hit  { background:var(--sdg-teal-light); color:#00675e; }
.sdg-icon-pill.pred { background:var(--sdg-teal-light); color:#009688; opacity:.7; }

/* Reason */
.reason { font-size:13px; line-height:1.65; color:var(--sdg-text);
  background:var(--sdg-surface); border-radius:var(--sdg-radius-sm);
  padding:12px 14px; margin-bottom:12px; border-left:3px solid var(--sdg-teal); }
.co-card.fallback .reason { border-left-color:#b0bec5; }

/* Card footer links */
.card-foot { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px; }
.links { display:flex; gap:6px; flex-wrap:wrap; }
.clink { font-size:12px; font-weight:600; color:var(--sdg-teal); padding:4px 11px;
  border:1px solid var(--sdg-teal); border-radius:20px; cursor:pointer; text-decoration:none; }
.clink:hover { background:var(--sdg-teal-light); }

/* Fallback notice */
.fallback-notice { font-size:11px; color:#c07800; background:var(--sdg-amber-light);
  border-radius:var(--sdg-radius-sm); padding:6px 12px; margin-bottom:10px;
  display:flex; align-items:center; gap:6px; }
.fn-dot { width:6px; height:6px; border-radius:50%; background:var(--sdg-amber); flex-shrink:0; }

/* Analysis: matrix */
.card2 { background:white; border:1px solid var(--sdg-border); border-radius:var(--sdg-radius);
  padding:18px 20px; margin-bottom:16px; overflow-x:auto; }
.mx { border-collapse:collapse; width:100%; font-size:11px; }
.mx th { font-weight:700; color:var(--sdg-muted); padding:4px 7px; text-align:left;
  white-space:nowrap; font-size:10px; text-transform:uppercase; letter-spacing:.04em; }
.mx th.cc { text-align:center; min-width:72px; }
.mx td { padding:5px 7px; border-top:1px solid var(--sdg-border); }
.mx td.sl { font-size:11px; color:var(--sdg-text); font-weight:500; white-space:nowrap; }
.mx td.cell { text-align:center; }
.dot { display:inline-flex; align-items:center; justify-content:center;
  width:20px; height:20px; border-radius:4px; font-size:10px; font-weight:700; }
.dot-h { background:var(--sdg-teal-light); color:#00675e; }
.dot-p { background:var(--sdg-teal-light); color:#009688; opacity:.6; }
.dot-m { background:var(--sdg-surface); color:#ccc; }
.mx-leg { display:flex; gap:16px; margin-top:12px; font-size:11px; color:var(--sdg-muted); flex-wrap:wrap; }
.ld { display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:middle; margin-right:3px; }

/* Analysis: at-a-glance */
.pg { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; margin-bottom:16px; }
.pc { background:var(--sdg-surface); border-radius:var(--sdg-radius-sm); padding:12px 13px;
  border:1px solid var(--sdg-border); }
.pc-rank { font-size:10px; color:var(--sdg-muted); font-weight:600; margin-bottom:3px; }
.pc-name { font-size:12px; font-weight:700; color:var(--sdg-navy); margin-bottom:7px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.pc-row { display:flex; justify-content:space-between; font-size:10px; margin-bottom:2px; }
.pc-k { color:var(--sdg-muted); }
.pc-v { font-weight:700; color:var(--sdg-navy); }
.pb-bg { height:3px; background:var(--sdg-border); border-radius:2px; margin-top:8px; overflow:hidden; }
.pb-fill { height:100%; border-radius:2px; background:var(--sdg-teal); }

/* Analysis: radar */
.radar-wrap { display:flex; gap:20px; align-items:flex-start; flex-wrap:wrap; }
.radar-cw { flex:1; min-width:200px; position:relative; height:260px; }
.rleg { flex:0 0 160px; font-size:11px; padding-top:8px; }
.rli { display:flex; align-items:center; gap:7px; margin-bottom:8px;
  color:var(--sdg-muted); font-weight:500; }
.rld { width:10px; height:10px; border-radius:3px; flex-shrink:0; }
"""

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SDGZero Partner Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>{css}</style>
</head>
<body>
<div class="container">

  <!-- Brand header -->
  <div class="brand-header">
    <div class="brand-logo">
      <img src="{logo_src}" alt="SDGZero" style="height:40px;">
      <div style="margin-left:10px">
        <div class="brand-tag" style="font-size:12px;color:var(--sdg-muted);margin-top:2px">Partner Finder System &nbsp;·&nbsp; {search_method} search &nbsp;·&nbsp; Top {n_scored} recommendations</div>
      </div>
    </div>
  </div>

  {criteria_bar}

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('results', this)">Results</button>
    <button class="tab-btn" onclick="switchTab('analysis', this)">Analysis</button>
  </div>

  <!-- Results tab -->
  <div id="tab-results" class="tab-pane active">
    <div class="sec-label">Recommended partners</div>
    {cards}
  </div>

  <!-- Analysis tab -->
  <div id="tab-analysis" class="tab-pane">
    <div class="sec-label">SDG Coverage Matrix</div>
    <div class="card2">{sdg_matrix}</div>

    <div class="sec-label">At a Glance</div>
    {glance_cards}

    <div class="sec-label">Dimension Comparison</div>
    <div class="card2">{radar}</div>
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
        filters, search_fallback_level, search_method, session_id,
        user_company_desc, partner_type_desc

    Writes:
        report  — path to the saved HTML file (str)
    """
    scored        = state.get("scored_companies", [])
    candidates    = state.get("candidate_companies", [])
    research      = state.get("research_results", {})
    filters       = state.get("filters", {})
    soft_filters  = state.get("soft_filters") or {}
    fallback_lvl  = state.get("search_fallback_level", 0)
    search_method = state.get("search_method", "semantic")
    session_id    = state.get("session_id", "unknown")
    user_desc     = state.get("user_company_desc", "")
    partner_type  = state.get("partner_type_desc", "")

    if not scored:
        logger.warning("ReportAgent: no scored_companies — generating empty report")

    criteria_bar = _render_criteria(
        filters, fallback_lvl, len(candidates), len(scored),
        user_company_desc=user_desc,
        partner_type_desc=partner_type,
        soft_filters=soft_filters,
    )

    cards = ""
    for i, company in enumerate(scored, 1):
        slug   = company.get("slug") or company.get("id", "")
        source = research.get(slug, {}).get("source", "db")
        cards += _render_card(i, company, source)

    sdg_matrix   = _render_sdg_matrix(scored)
    glance_cards = _render_glance_cards(scored)
    radar        = _render_radar(scored, research)

    static_base = os.getenv("REPORT_STATIC_BASE", "relative")
    logo_src = "/static/SDG0logo.png" if static_base == "server" else "../static/SDG0logo.png"

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
        logo_src=logo_src,
    )

    _REPORTS_DIR.mkdir(exist_ok=True)
    report_path = _REPORTS_DIR / f"{session_id}.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"ReportAgent: saved → {report_path}")
    print(f"\n  Report saved → {report_path.resolve()}")
    print(f"  Open in browser: file://{report_path.resolve()}")

    return {"report": str(report_path.resolve())}
