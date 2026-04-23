# SDGZero Partner Finder

AI-powered business partner matching system for the [SDGZero](https://sdgzero.com) sustainability directory (~492 companies and growing).

Given a description of your company and what you're looking for, the system finds the best-matching partner companies using a multi-agent LangGraph pipeline, served through a FastAPI backend and Next.js frontend.

**Live demo:** [sdg-0-partner-finder-system.vercel.app](https://sdg-0-partner-finder-system.vercel.app)

---

## System Architecture

```
Next.js Frontend
      ↓  POST /api/search
FastAPI Backend  (async, session-based)  ←── RefineAgent (HITL, POST /api/refine)
      ↓                                              ↑ user feedback
LangGraph Multi-Agent Pipeline                       │
  SearchAgent ──→ ResearchAgent ──→ ScoringAgent ──→ ReportAgent
  │  HyDE                Tavily        Cross-encoder   │ HTML report
  │  Query Expansion     (3-tier)                      └──→ user
  │  Tool routing
  └─ LLM-as-Judge + CRAG Reflection (selective retry)
      ↓
PostgreSQL + pgvector  (492 companies, 384-dim embeddings)
```

---

## Quick Start

### Prerequisites

- Python 3.9+, Node.js 18+
- PostgreSQL 15+ with pgvector extension
- API keys: Groq (free) + Tavily (free)

### 1. Clone & install

```bash
git clone <repo-url>
cd SDG0-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in:
#   DATABASE_URL     — PostgreSQL connection string
#   GROQ_API_KEY     — https://console.groq.com  (free)
#   TAVILY_API_KEY   — https://tavily.com  (free, 1000 req/month)
#   LLM_PROVIDER     — groq | gemini | ollama  (default: groq)
```

### 3. Initialise database & ingest data

```bash
python -c "from db.pg_store import PGStore; PGStore().init_schema()"
python -m pipeline.update
```

### 4. (Optional) Train SDG classifier & backfill predictions

```bash
python -m ml.sdg_classifier train          # LogReg (~5 seconds)
python -m ml.sdg_classifier backfill_llm   # LLM predictions for untagged companies
```

### 5. Start backend + frontend

```bash
uvicorn api.main:app --reload --port 8000
cd frontend && npm run dev
```

Open [http://localhost:3000](http://localhost:3000) — API docs at [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

---

## Multi-Agent Pipeline

### Agent 1 — SearchAgent

Responsible for generating a high-quality candidate set.

**HyDE (Hypothetical Document Embedding)**
Prompts an LLM to write a 50–120 word profile of the ideal *partner* company (not the user's own company). Grounded in [Gao et al. 2022](https://arxiv.org/abs/2212.10496). The generated profile is embedded alongside 3–5 query expansion phrases; all vectors are averaged into one query vector for richer recall.

**Three search tools with smart routing**

| Condition | Tool used |
|---|---|
| Description + filters | `hybrid_search` — SQL WHERE + pgvector ORDER BY in one query |
| Description only | `semantic_search` — pure vector similarity |
| Filters only | `sql_filter` — pure SQL metadata match, no vector |

**Three-level fallback** — if filtered results are too few:
- Level 0: full conditions (SQL + vector)
- Level 1: relax SQL filters, keep vector
- Level 2: drop all filters, pure vector (always returns results)

**LLM-as-Judge + CRAG Reflection**
After retrieval, an LLM evaluates each candidate and flags poor matches. If fewer than 5 candidates pass, the agent reflects on *why* results are poor, updates the search direction, and retries — replacing only the bad candidates while keeping good ones. Based on [CRAG (Yan et al. 2024)](https://arxiv.org/abs/2401.15884) and [Reflexion (Shinn et al. 2023)](https://arxiv.org/abs/2303.11366).

### Agent 2 — ResearchAgent

Enriches each Top-10 candidate with up-to-date information via a three-tier strategy (runs in parallel across all candidates):

- **Layer 1** (always) — structured summary from DB fields (description, achievements, SDG involvement). Guaranteed coverage for all companies.
- **Layer 2** (if Tavily key configured + website URL exists) — Tavily Extract crawls the company's live website, layered on top of Layer 1 to capture recent activities and current focus areas
- **Layer 3** (fallback for Layer 2) — Tavily Search queries the company name when the website URL is missing or returns no content

If no Tavily key is configured, the pipeline silently falls back to Layer 1 only.

### Agent 3 — ScoringAgent

**Cross-encoder reranking** — scores each candidate against the HyDE-generated partner description using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Selects Top-5.

**LLM reasoning** — generates a human-readable explanation for each Top-5 match: why they fit, what they offer, and how they align with the user's goals.

### Agent 4 — ReportAgent

Formats the final Top-5 into a self-contained HTML report served via `/api/report/{id}`.

After viewing the report, users can refine their search using the Human-in-the-Loop interface (see Agent 5 below).

### Agent 5 — RefineAgent (Human-in-the-Loop)

Users can submit feedback after viewing results: rate each company (👍/👎) and describe what was wrong in free text. The RefineAgent interprets the feedback and returns updated search parameters:

- Infers what went wrong (wrong sector, wrong city, wrong SDG focus, etc.)
- Validates all structured filter values against the live DB schema — no hallucinated cities or SDG tags
- Carries forward unchanged filters from the original session
- Creates a new session with `parent_id` for full lineage tracking

The frontend then re-runs the full pipeline with the updated parameters.

Endpoint: `POST /api/refine/{session_id}`

---

## Hard vs Soft Filters

Every filter can be set to **Must** (hard) or **Prefer** (soft):

- **Must** — passed to PostgreSQL as a strict `WHERE` clause; non-matching companies are excluded before retrieval
- **Prefer** — passed to ScoringAgent as bonus scoring criteria; matching companies rank higher but nothing is excluded

---

## SDG Classifier (ML Module)

Multi-label SDG tag prediction for companies with missing labels. Three methods evaluated in CI, all tracked in MLflow + DagsHub:

| Method | Description | F1 |
|---|---|---|
| `logreg` | OneVsRest LogisticRegression on 384-dim pgvector embeddings. Retrained automatically on new confirmed labels. | 0.375 (5-fold CV) |
| `zero_shot` | Cosine similarity between pgvector embeddings and SDG keyword embeddings. No training required. | 0.510 |
| `llm` | Few-shot prompting via Groq (llama-3.1-8b-instant). **Current production champion.** | 0.574 |

> `zero_shot` and `logreg` serve as baselines for champion-challenger comparison. LLM outperforms both given the current small label set (~20 confirmed labels); as ground truth grows, the gap is expected to narrow.

**Backfill:** LLM predictions written to `predicted_sdg_tags` for all 492 companies.

---

## RAG Pipeline Evaluation

Evaluated against the original SDGZero website (WordPress MySQL LIKE search) and BM25 as baselines, across 3 query sets: SDG descriptions, motivational queries, and out-of-vocabulary (OOV) queries.

| Metric | Website (MySQL) | BM25 | Raw Semantic | Full Pipeline (HyDE) |
|---|---|---|---|---|
| Zero-result rate ↓ | 93–100% | 0% | 0% | **0%** |
| SDG Relevance@10 — motivational queries ↑ | 6.0% | 46.7% | 46.0% | **47.7%** |
| SDG Relevance@10 — OOV queries ↑ | 0.0% | 36.5% | 46.5% | **54.0%** |
| Category Consistency@5 ↑ | — | 92.7% | 96.0% | **96.7%** |
| HyDE Cross-Sector@5 ↑ | — | — | 57.5% | **77.5%** |

**Key findings:**
- The original website returns 0 results for 93–100% of natural language queries; the pipeline always returns 10
- HyDE improves OOV recall by +17.5pp over BM25 (54.0% vs 36.5%)
- HyDE improves cross-sector diversity by +20pp over plain semantic search (77.5% vs 57.5%)
- 96.7% category consistency across repeated queries

---

## MLOps

### ML Pipeline CI (GitHub Actions — weekly + manual)

Runs every Monday at 02:00 UTC or on manual trigger via `workflow_dispatch`:

1. Incremental scrape + embed new companies
2. LLM backfill for new companies (`skip_existing=True`)
3. Check confirmed label count → skip evaluation if fewer than 20
4. Retrain LogReg on all confirmed `sdg_tags` from DB (seconds)
5. Evaluate: `logreg` (trainable baseline) + `llm` (production champion)
6. Select champion → optionally promote to Production + commit `config/active_method.json` → triggers Railway redeploy

Manual trigger inputs: `method` (`all` / `logreg` / `llm`), `promote` (auto-promote champion), `skip_scrape` (eval only).

> SetFit and NLI are excluded from CI: SetFit requires GPU for practical training speed; NLI inference is O(n×17) cross-encoder calls, too slow on CPU runners. Both can still be run locally.

### CI (GitHub Actions — every push/PR to main)

- Spins up a real `pgvector/pgvector:pg16` PostgreSQL service container
- Validates all Python imports (`agent.graph`, `db.pg_store`, `api.main`)
- Initialises DB schema and runs API health check
- Builds Next.js frontend with production env

### Experiment Tracking

All evaluation runs logged to **MLflow + DagsHub**:
- Parameters: `method`, `threshold`
- Metrics: `f1`, `precision`, `recall`, `support`
- Champion selection: `select_champion.py` queries all `eval_*` runs, picks highest F1, registers to MLflow Model Registry, promotes to Production

---

## MCP Server

Exposes the system as an MCP (Model Context Protocol) server for Claude Desktop / Cursor / other MCP clients.

Five tools:
- `search_companies` — semantic search by natural language
- `filter_companies` — pure SQL metadata filter
- `get_company` — full company profile by slug
- `list_filters` — live filter schema (valid cities, SDGs, categories, etc.)
- `find_partners` — trigger the full multi-agent pipeline and return structured results

**Tool-calling evaluation** (39 test cases, Claude Haiku):

| Metric | Score |
|---|---|
| Tool Selection Accuracy | 97.4% |
| Key Param Hit Rate | 75.0% |
| Wrong-Tool Rate | 0.0% |

Tool descriptions and structured docstrings are sufficient for reliable LLM-driven tool dispatch without any fine-tuning.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/search` | Start a pipeline run; returns `session_id` immediately |
| `GET` | `/api/search/{id}` | Poll run status; returns `scored_companies` when done |
| `GET` | `/api/report/{id}` | Serve the generated HTML report (for iframe embed) |
| `POST` | `/api/refine/{session_id}` | Interpret user feedback; return updated search params |
| `GET` | `/api/schema` | Live filter values (cities, SDG tags, categories, …) |
| `GET` | `/api/health` | Health check |

Search runs asynchronously — frontend polls `GET /api/search/{id}` every 3 seconds until `status == "done"`.

---

## LLM Providers

| Provider | Model | Notes |
|---|---|---|
| `groq` (recommended) | llama-3.1-8b-instant | Free tier: 500K tokens/day |
| `gemini` | gemini-2.0-flash | Free tier available |
| `ollama` | llama3.1:8b | Unlimited (local) |

---

## Project Structure

```
agent/
  graph.py              # LangGraph pipeline definition
  state.py              # AgentState TypedDict (full field ownership docs)
  search_agent.py       # HyDE + query expansion + tool routing + 3-level fallback + LLM-Judge + CRAG
  research_agent.py     # Tavily 3-tier web research (parallel)
  scoring_agent.py      # Cross-encoder reranking + LLM reasoning
  report_agent.py       # HTML report generation
  refine_agent.py       # Human-in-the-Loop feedback interpretation
  tools.py              # semantic_search / sql_filter / hybrid_search (pgvector)
  schema_cache.py       # Disk-cached filter schema (7-day TTL)
  llm.py                # LLM factory (Groq / Gemini / Ollama)

api/
  main.py               # FastAPI app entry point
  session_store.py      # JSON-file session persistence (15-day TTL)
  routes/
    search.py           # POST /api/search, GET /api/search/{id}, GET /api/report/{id}
    refine.py           # POST /api/refine/{session_id}
    schema.py           # GET /api/schema

db/
  pg_store.py           # PostgreSQL + pgvector wrapper (connection pool, hybrid search)
  sdg_normalize.py      # SDG tag canonicalisation (fixes data-entry errors)

ml/
  sdg_classifier.py     # Multi-method SDG classifier + MLflow tracking + evaluate + backfill
  select_champion.py    # Champion-challenger selection via MLflow Model Registry

pipeline/
  ingest.py             # Full scrape → embed → store (one-time setup)
  update.py             # Incremental scrape → embed → store (weekly CI)

scraper/
  spider.py             # SDGZero directory scraper
  models.py             # Pydantic data models + SDG tag normalisation

mcp_server/
  server.py             # FastMCP server (5 tools)

eval/
  rag_eval.py           # RAG pipeline evaluation (OOV recall, cross-sector, consistency)
  mcp_eval.py           # MCP server tool evaluation
  mcp_test_cases.json   # MCP test case definitions

.github/workflows/
  ci.yml                # Push/PR CI: import checks, DB init, API health, frontend build
  ml-pipeline.yml       # Weekly ML pipeline: scrape → backfill → train → evaluate → promote

tests/
  test_hitl_flow.py     # HITL end-to-end: initial search → refine → compare
  test_refine_unit.py   # RefineAgent unit tests (no LLM/DB required)

frontend/
  app/
    page.tsx            # Search form
    results/[id]/       # Results page (polling + report iframe + refinement UI)
  components/
    SearchForm.tsx      # Search form with hard/soft filter toggles
    FilterRow.tsx       # Individual filter row (Must / Prefer toggle)

docker-compose.yml      # Local PostgreSQL + pgvector container
```

---

## Session Management

- Each search run gets a unique `session_id`
- Results stored in `sessions/{id}.json` and `reports/{id}.html`
- 15-day TTL, automatically enforced at read time
- Refined searches carry `parent_id` for full session lineage

---

## Deployment

| Service | Platform | Trigger |
|---|---|---|
| Backend (FastAPI) | Railway | Auto-deploys on push to `main` |
| Frontend (Next.js) | Vercel | Auto-deploys on push to `main` |
| ML experiments | DagsHub + MLflow | Logged on every CI evaluate run |

CI/CD flow:
```
push to main → GitHub Actions CI (tests + build) → Railway + Vercel auto-deploy
                    ↓ (every Monday)
             ML Pipeline CI → retrain → evaluate → promote champion → commit → redeploy
```
