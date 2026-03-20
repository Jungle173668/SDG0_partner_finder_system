# SDGZero Partner Finder

AI-powered business partner matching system for the [SDGZero](https://sdgzero.com) sustainability directory.

Given a company description, it finds the best-matching partner companies using a multi-agent LangGraph pipeline, served through a FastAPI backend and Next.js frontend.

---

## How It Works

A search goes through four agents in sequence:

```
SearchAgent → ResearchAgent → ScoringAgent → ReportAgent
```

1. **SearchAgent** — HyDE (hypothetical document embedding) + query expansion → semantic / hybrid / SQL search with three-level fallback → Top-10 candidates
2. **ResearchAgent** — Tavily web research per candidate to enrich company profiles
3. **ScoringAgent** — Cross-encoder reranking + LLM reasoning → Top-5 with match explanations
4. **ReportAgent** — Generates a self-contained HTML report with outreach drafts

### Hard vs Soft Filters

Every filter can be set to **Must** (hard) or **Prefer** (soft):

- **Must** — passed to the database as a strict WHERE clause; companies not matching are excluded
- **Prefer** — passed to ScoringAgent as bonus scoring criteria; companies matching score higher but no one is excluded

---

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- API keys: Google Gemini (free) + Tavily (free)

### 1. Clone & install Python dependencies

```bash
git clone <your-repo-url>
cd SDG0-system
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in:
#   GOOGLE_API_KEY   — https://aistudio.google.com/apikey  (free, 1500 req/day)
#   TAVILY_API_KEY   — https://tavily.com  (free, 1000 req/month)
#   LLM_PROVIDER     — gemini | groq | ollama  (default: gemini)
```

### 3. Set up the database

The ChromaDB vector database is not included in the repo. Populate it by running the scraper:

```bash
python -m pipeline.ingest
```

This scrapes the SDGZero directory, embeds all companies, and stores them in `./chroma_db/`.

### 4. (Optional) Train the SDG classifier

Predict SDG tags for companies that don't have them manually tagged:

```bash
# Train LogisticRegression classifier (~5 seconds)
python -m ml.sdg_classifier train

# Write predicted_sdg_tags into ChromaDB
python -m ml.sdg_classifier backfill
```

### 5. Start the backend

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## LLM Providers

Set `LLM_PROVIDER` in `.env`:

| Provider | Model | Free Tier |
|---|---|---|
| `gemini` (default) | gemini-2.0-flash | 1,500 req/day |
| `groq` | llama-3.1-8b-instant | 500K tokens/day |
| `ollama` | llama3.1:8b | Unlimited (local) |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/search` | Start a pipeline run, returns `session_id` immediately |
| `GET` | `/api/search/{id}` | Poll run status; returns `scored_companies` when done |
| `GET` | `/api/report/{id}` | Serve the generated HTML report (for iframe embed) |
| `GET` | `/api/schema` | Filter dropdown values (cities, SDG tags, categories, …) |
| `GET` | `/api/health` | Health check |

Search runs asynchronously — the frontend polls `GET /api/search/{id}` every 3 seconds until `status == "done"`.

---

## Project Structure

```
api/
  main.py               # FastAPI app entry point
  session_store.py      # JSON-file session persistence (15-day TTL)
  routes/
    search.py           # POST /api/search, GET /api/search/{id}, GET /api/report/{id}
    schema.py           # GET /api/schema

agent/
  graph.py              # LangGraph pipeline definition
  state.py              # AgentState TypedDict
  search_agent.py       # HyDE + semantic/hybrid/SQL search + 3-level fallback
  research_agent.py     # Tavily web research
  scoring_agent.py      # Cross-encoder reranking + LLM reasoning
  report_agent.py       # HTML report generation
  schema_cache.py       # Disk-cached filter schema (schema_cache.json, 7-day TTL)
  llm.py                # LLM factory (Gemini / Groq / Ollama)
  tools.py              # ChromaDB search tools

db/
  chroma_store.py       # ChromaDB wrapper
  sdg_normalize.py      # SDG tag canonicalization

ml/
  sdg_classifier.py     # SDG tag prediction
                        #   - LogisticRegression (default, fast, uses stored embeddings)
                        #   - SetFit (optional, higher accuracy, needs separate training)
                        #   - Zero-shot cosine similarity (fallback, no training required)

scraper/
  spider.py             # SDGZero directory scraper
  models.py             # Pydantic models + SDG tag normalisation on ingest

pipeline/
  ingest.py             # Scrape → embed → store pipeline

scripts/
  fix_sdg_tags.py       # One-time script: fix corrupted SDG tags in existing ChromaDB records

frontend/
  app/
    page.tsx            # Home page (search form)
    results/[id]/
      page.tsx          # Results page (polling + report iframe)
    globals.css         # Tailwind + global styles
    layout.tsx          # Shared HTML shell
  components/
    SearchForm.tsx      # Search form with advanced filters
    FilterRow.tsx       # Single filter row with Must / Prefer toggle
  lib/
    api.ts              # HTTP client (fetch wrappers for all backend endpoints)

reports/                # Generated HTML reports (auto-deleted after 15 days)
sessions/               # Session JSON files (auto-deleted after 15 days)
static/                 # Static assets (SDGZero logo, SDG icons)
chroma_db/              # ChromaDB vector database (not in repo)
```

---

## SDG Tag Prediction

Companies scraped from SDGZero may not have SDG tags filled in. The ML module predicts them:

```bash
# Train (uses existing ChromaDB embeddings — fast)
python -m ml.sdg_classifier train

# Backfill predictions into ChromaDB
python -m ml.sdg_classifier backfill

# Compare LogReg vs SetFit (optional, SetFit must be trained separately)
python -m ml.sdg_classifier train_setfit
python -m ml.sdg_classifier compare

# View current coverage stats
python -m ml.sdg_classifier stats
```

---

## Session Management

- Each search run gets a 6-character `session_id` (e.g. `K5N6kg`)
- Results are stored in `sessions/{id}.json` and `reports/{id}.html`
- Sessions expire after **15 days** and are deleted automatically
- Expired report pages show a user-friendly expiry message instead of a raw error

---

## Roadmap

- [x] Phase 1: Core pipeline (Search → Research → Score → Report)
- [x] Phase 2: HyDE + filter injection + cross-encoder reranking
- [x] Phase 3: FastAPI backend + Next.js frontend + hard/soft filter UI
- [ ] Phase 4: pgvector / PostgreSQL migration

---

## License

MIT
