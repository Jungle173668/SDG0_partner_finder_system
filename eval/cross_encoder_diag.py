"""
Cross-Encoder Anchor Comparison
================================
Tests three query anchors for cross-encoder reranking side-by-side:
  A — partner_type_desc  (user's explicit description of target company)
  B — hypothetical_partner_desc (HyDE-generated ideal partner profile)
  C — user_company_desc  (user's own company description)

For each of 8 test cases:
  1. Bi-encoder search → Top-10 candidates
  2. Cross-encoder score with all 3 anchors
  3. Print per-candidate score table + score spread (max-min)

Higher spread = anchor discriminates better between good and bad candidates.

Usage:
    python eval/cross_encoder_diag.py
    python eval/cross_encoder_diag.py --k 10        # top-K candidates
    python eval/cross_encoder_diag.py --case 1      # single case
    python eval/cross_encoder_diag.py --no-hyde     # skip HyDE (faster, uses cache if exists)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Test cases: PARTNER_CASES + partner_type_desc added
# user_desc   = what the user writes about their OWN company
# partner_type_desc = what the user writes about the company they WANT to find
# ---------------------------------------------------------------------------

# Each case comes in a pair: without (A) and with (B) partner_type_desc.
# Target company is a real company in the DB.
# A: only user_desc filled  → pipeline relies on HyDE to infer target
# B: both fields filled     → partner_type_desc used directly as anchor
TEST_CASES = [
    # ── Pair 1 ── target: Heat Engineer Software Ltd (Energy & Renewables / Tech)
    {
        "id": "1A",
        "label": "Solar installer — no target desc  →  Heat Engineer Software",
        "target": "Heat Engineer Software Ltd",
        "user_desc": (
            "We are a residential solar panel installation company helping UK "
            "homeowners switch to renewable energy and reduce their energy bills."
        ),
        "partner_type_desc": "",
    },
    {
        "id": "1B",
        "label": "Solar installer — WITH target desc →  Heat Engineer Software",
        "target": "Heat Engineer Software Ltd",
        "user_desc": (
            "We are a residential solar panel installation company helping UK "
            "homeowners switch to renewable energy and reduce their energy bills."
        ),
        "partner_type_desc": (
            "software for heating engineers to calculate heat pump sizing, "
            "energy efficiency and renewable heating system design"
        ),
    },

    # ── Pair 2 ── target: Stain Media (Purpose-driven marketing / B2B)
    {
        "id": "2A",
        "label": "Sustainable brand — no target desc  →  Stain Media",
        "target": "Stain Media",
        "user_desc": (
            "We produce certified organic food products and want to grow our "
            "brand presence among sustainability-conscious consumers."
        ),
        "partner_type_desc": "",
    },
    {
        "id": "2B",
        "label": "Sustainable brand — WITH target desc →  Stain Media",
        "target": "Stain Media",
        "user_desc": (
            "We produce certified organic food products and want to grow our "
            "brand presence among sustainability-conscious consumers."
        ),
        "partner_type_desc": (
            "a purpose-driven digital marketing or creative agency that "
            "specialises in sustainability storytelling and brand communications"
        ),
    },

    # ── Pair 3 ── target: Optima Prep Lab (HR & Recruitment / London)
    {
        "id": "3A",
        "label": "Fintech startup — no target desc  →  Optima Prep Lab",
        "target": "Optima Prep Lab",
        "user_desc": (
            "We are a fast-growing fintech startup and need help attracting "
            "top talent and building out our HR processes."
        ),
        "partner_type_desc": "",
    },
    {
        "id": "3B",
        "label": "Fintech startup — WITH target desc →  Optima Prep Lab",
        "target": "Optima Prep Lab",
        "user_desc": (
            "We are a fast-growing fintech startup and need help attracting "
            "top talent and building out our HR processes."
        ),
        "partner_type_desc": (
            "an HR consultancy or career coaching service that helps companies "
            "hire, assess and develop professional talent"
        ),
    },

    # ── Pair 4 ── target: Lancashire Women (Community / Women's charity)
    {
        "id": "4A",
        "label": "DEI training co — no target desc  →  Lancashire Women",
        "target": "Lancashire Women",
        "user_desc": (
            "We deliver diversity, equity and inclusion training programmes "
            "to UK corporates as part of their CSR commitments."
        ),
        "partner_type_desc": "",
    },
    {
        "id": "4B",
        "label": "DEI training co — WITH target desc →  Lancashire Women",
        "target": "Lancashire Women",
        "user_desc": (
            "We deliver diversity, equity and inclusion training programmes "
            "to UK corporates as part of their CSR commitments."
        ),
        "partner_type_desc": (
            "a women's charity or community organisation supporting women "
            "facing disadvantage, to refer our corporate clients to"
        ),
    },

    # ── Pair 5 ── target: MindFit (Performance psychology / London B2B)
    {
        "id": "5A",
        "label": "Sports academy — no target desc  →  MindFit",
        "target": "MindFit",
        "user_desc": (
            "We run a professional sports academy training elite junior "
            "athletes across multiple disciplines in the UK."
        ),
        "partner_type_desc": "",
    },
    {
        "id": "5B",
        "label": "Sports academy — WITH target desc →  MindFit",
        "target": "MindFit",
        "user_desc": (
            "We run a professional sports academy training elite junior "
            "athletes across multiple disciplines in the UK."
        ),
        "partner_type_desc": (
            "a performance psychology or mental fitness coaching company "
            "that works with athletes and high-performance teams"
        ),
    },

    # ── Pair 6 ── target: Zuri Adventures (Eco luxury travel)
    {
        "id": "6A",
        "label": "Corporate events co — no target desc  →  Zuri Adventures",
        "target": "Zuri Adventures",
        "user_desc": (
            "We organise executive retreats and corporate team-building events "
            "for large companies looking for unique offsite experiences."
        ),
        "partner_type_desc": "",
    },
    {
        "id": "6B",
        "label": "Corporate events co — WITH target desc →  Zuri Adventures",
        "target": "Zuri Adventures",
        "user_desc": (
            "We organise executive retreats and corporate team-building events "
            "for large companies looking for unique offsite experiences."
        ),
        "partner_type_desc": (
            "a luxury eco-travel or sustainable adventure tourism company "
            "offering bespoke experiences for corporate groups"
        ),
    },
]

_CACHE_PATH = Path(".eval_ce_diag_cache.json")


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def run_case(
    case: dict,
    companies: list[dict],
    all_embeddings: np.ndarray,
    company_ids: list[int],
    encoder,
    cross_encoder,
    cache: dict,
    k: int = 15,
    run_hyde: bool = True,
) -> dict:
    from agent.search_agent import _run_hyde, _averaged_embedding

    user_desc    = case["user_desc"]
    partner_type = case["partner_type_desc"]
    target_name  = case.get("target", "")
    label        = case["label"]
    id_to_co     = {c["id"]: c for c in companies}

    # ------------------------------------------------------------------
    # Step 1: HyDE (cached by user_desc)
    # ------------------------------------------------------------------
    hyde_desc = ""
    if run_hyde:
        cache_key = f"hyde2:{user_desc[:80]}|{partner_type[:40]}"
        if cache_key in cache:
            hyde_desc = cache[cache_key]
        else:
            try:
                t0 = time.time()
                partner_desc, expansions, inferred_type = _run_hyde(
                    user_company_desc=user_desc,
                    partner_type_desc=partner_type,
                )
                hyde_desc = partner_desc
                cache[cache_key] = hyde_desc
                _save_cache(cache)
                print(f"  HyDE ({time.time()-t0:.1f}s): {hyde_desc[:80]}...")
            except Exception as e:
                print(f"  HyDE failed: {e}")

    # ------------------------------------------------------------------
    # Step 2: Bi-encoder search
    # Use partner_type if filled, else user_desc
    # ------------------------------------------------------------------
    search_text = partner_type if partner_type.strip() else user_desc
    q_emb = encoder.encode([search_text], normalize_embeddings=True)[0]
    sims  = all_embeddings @ q_emb

    # Find target rank in full ranking (before top-k cut)
    sorted_idx   = np.argsort(-sims)
    target_bi_rank = None
    target_id      = None
    for rank, idx in enumerate(sorted_idx):
        cid = company_ids[idx]
        if id_to_co.get(cid, {}).get("name", "") == target_name:
            target_bi_rank = rank + 1
            target_id      = cid
            break

    top_idx  = sorted_idx[:k]
    top_ids  = [company_ids[i] for i in top_idx]
    candidates = [
        {**id_to_co[cid], "bi_sim": float(sims[top_idx[rank]])}
        for rank, cid in enumerate(top_ids)
        if cid in id_to_co
    ]

    # Inject target if it didn't make top-k (so we can still score it)
    target_in_topk = any(c.get("name") == target_name for c in candidates)
    if not target_in_topk and target_id and target_id in id_to_co:
        target_co = id_to_co[target_id]
        candidates.append({
            **target_co,
            "bi_sim": float(sims[sorted_idx[target_bi_rank - 1]]),
            "_injected": True,
        })

    # ------------------------------------------------------------------
    # Step 3: Cross-encoder — 3 anchors
    # A: partner_type_desc (only if filled), B: HyDE, C: user_desc
    # ------------------------------------------------------------------
    anchors = {}
    if partner_type.strip():
        anchors["A_partner_type"] = partner_type
    anchors["B_hyde"]      = hyde_desc if hyde_desc else None
    anchors["C_user_desc"] = user_desc

    results = []
    for c in candidates:
        doc = c.get("document") or ""
        is_target   = c.get("name", "") == target_name
        is_injected = c.get("_injected", False)
        row = {
            "name":       c.get("name", "?")[:38],
            "bi_sim":     round(c["bi_sim"], 4),
            "is_target":  is_target,
            "injected":   is_injected,
        }
        for anchor_key, anchor_text in anchors.items():
            if not anchor_text:
                row[anchor_key] = None
                continue
            raw = cross_encoder.predict([(anchor_text, doc)])
            row[anchor_key] = round(float(_sigmoid(raw[0])), 4)
        results.append(row)

    results.sort(key=lambda r: r.get("B_hyde") or r.get("A_partner_type") or 0, reverse=True)

    # Score spread per anchor
    spreads = {}
    for ak in anchors:
        vals = [r[ak] for r in results if r.get(ak) is not None]
        spreads[ak] = round(max(vals) - min(vals), 4) if len(vals) > 1 else 0.0

    # Target rank by each anchor
    target_ce_ranks = {}
    for ak in anchors:
        ranked = sorted(
            [r for r in results if r.get(ak) is not None],
            key=lambda r: r[ak], reverse=True
        )
        for i, r in enumerate(ranked):
            if r["is_target"]:
                target_ce_ranks[ak] = i + 1
                break
        else:
            target_ce_ranks[ak] = None

    return {
        "case_id":          case["id"],
        "label":            label,
        "target":           target_name,
        "hyde_desc":        hyde_desc[:120] if hyde_desc else "",
        "results":          results,
        "spreads":          spreads,
        "target_bi_rank":   target_bi_rank,
        "target_ce_ranks":  target_ce_ranks,
        "anchors":          list(anchors.keys()),
    }


def print_case(out: dict) -> None:
    w = 38
    anchors = out["anchors"]
    col_w   = 14

    print(f"\n{'='*95}")
    print(f"  [{out['label']}]")
    print(f"  Target : {out['target']}  (bi-encoder rank: {out['target_bi_rank']})")
    if out["hyde_desc"]:
        print(f"  HyDE   : {out['hyde_desc']}...")
    print(f"{'='*95}")

    header = f"  {'Company':<{w}}  {'bi_sim':>7}"
    for ak in anchors:
        header += f"  {ak[:col_w]:>{col_w}}"
    print(header)
    print(f"  {'-'*w}  {'-'*7}" + f"  {'-'*col_w}" * len(anchors))

    for r in out["results"]:
        flag = " ◀ TARGET" if r["is_target"] else ("  [injected]" if r["injected"] else "")
        line = f"  {r['name']:<{w}}  {r['bi_sim']:>7.4f}"
        for ak in anchors:
            v = r.get(ak)
            line += f"  {v:>{col_w}.4f}" if v is not None else f"  {'—':>{col_w}}"
        print(line + flag)

    print(f"\n  Target cross-encoder rank:")
    for ak, rank in out["target_ce_ranks"].items():
        print(f"    {ak:<22} rank = {rank}")

    print(f"\n  Score spread (max−min):")
    for ak, v in out["spreads"].items():
        bar = "█" * int(v * 40)
        print(f"    {ak:<22} {v:.4f}  {bar}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-encoder anchor comparison")
    ap.add_argument("--k",      type=int, default=10, help="Top-K candidates from bi-encoder")
    ap.add_argument("--case",   type=int, default=0,  help="Run single case by id (0 = all)")
    ap.add_argument("--no-hyde", action="store_true",  help="Skip HyDE calls (use cache only)")
    args = ap.parse_args()

    cases = [c for c in TEST_CASES if args.case == 0 or c["id"] == args.case]
    if not cases:
        print(f"No case with id={args.case}")
        return

    print("Loading DB companies + embeddings...", flush=True)
    from db.pg_store import PGStore
    store = PGStore()
    with store._cursor(dict_rows=True) as cur:
        cur.execute("SELECT id, name, categories, document, embedding FROM businesses")
        rows = cur.fetchall()

    companies      = [dict(r) for r in rows]
    all_embeddings = np.array([c["embedding"] for c in companies], dtype=np.float32)
    company_ids    = [c["id"] for c in companies]
    print(f"  {len(companies)} companies loaded.", flush=True)

    print("Loading models...", flush=True)
    from sentence_transformers import SentenceTransformer, CrossEncoder
    encoder       = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("  Models ready.", flush=True)

    cache   = _load_cache()
    outputs = []

    for case in cases:
        print(f"\n[{case['id']}/{len(TEST_CASES)}] {case['label']}...", flush=True)
        out = run_case(
            case, companies, all_embeddings, company_ids,
            encoder, cross_encoder, cache,
            k=args.k, run_hyde=not args.no_hyde,
        )
        outputs.append(out)
        print_case(out)

    # ------------------------------------------------------------------
    # Summary: paired comparison (A vs B cases)
    # ------------------------------------------------------------------
    print(f"\n{'='*95}")
    print("  SUMMARY — paired cases: does adding 'I am looking for...' improve target rank?")
    print(f"{'='*95}")
    print(f"  {'Pair':<6} {'Target':<35} {'bi-rank':>8}  {'CE rank (no desc)':>18}  {'CE rank (with desc)':>20}")
    print(f"  {'-'*6} {'-'*35} {'-'*8}  {'-'*18}  {'-'*20}")

    pairs = {}
    for o in outputs:
        pid = o["label"].split("—")[0].strip()  # "1A" → "1"
        pid = pid[:-1]  # strip A/B
        pairs.setdefault(pid, {})[o["label"][-1] if o["label"][-1] in "AB" else "?"] = o

    # Fallback: group by case id field
    pairs = {}
    for o in outputs:
        cid = str(o.get("case_id", "?"))
        pid = cid[:-1]
        variant = cid[-1]
        pairs.setdefault(pid, {})[variant] = o

    for pid, variants in sorted(pairs.items()):
        a = variants.get("A")
        b = variants.get("B")
        if not a or not b:
            continue
        target = a["target"]
        bi     = a["target_bi_rank"]
        ce_a   = a["target_ce_ranks"].get("C_user_desc") or a["target_ce_ranks"].get("B_hyde")
        ce_b   = b["target_ce_ranks"].get("A_partner_type") or b["target_ce_ranks"].get("B_hyde")
        arrow  = "↑ better" if (ce_b and ce_a and ce_b < ce_a) else ("↓ worse" if (ce_b and ce_a and ce_b > ce_a) else "= same")
        print(f"  {pid:<6} {target:<35} {str(bi):>8}  {str(ce_a):>18}  {str(ce_b):>18}  {arrow}")

    print(f"\n  Mean score spread per anchor:")
    all_anchor_keys = ["A_partner_type", "B_hyde", "C_user_desc"]
    for ak in all_anchor_keys:
        vals = [o["spreads"].get(ak) for o in outputs if o["spreads"].get(ak) is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        bar  = "█" * int(mean * 40)
        print(f"    {ak:<22} mean spread = {mean:.4f}  {bar}")


if __name__ == "__main__":
    main()
