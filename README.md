# SDGZero Partner Finder

AI-powered business partner matching system for the [SDGZero](https://sdgzero.com) sustainability directory.

Given a company description, it finds the best-matching partner companies using a multi-stage LangGraph pipeline:

1. **SearchAgent** — HyDE + query expansion → semantic/hybrid/SQL search → Top-10 candidates
2. **FilterAgent** — (Phase 3) soft filter scoring
3. **ResearchAgent** — Tavily web research per candidate
4. **ScoringAgent** — Cross-encoder reranking → Top-5 → LLM reasoning

## Quick Start

### 1. Clone & install

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
# Edit .env and fill in your keys:
#   GOOGLE_API_KEY  — from https://aistudio.google.com/apikey (free)
#   TAVILY_API_KEY  — from https://tavily.com (free, 1000 req/month)
```

### 3. Set up the database

The ChromaDB vector database is not included in the repo (too large). You need to populate it:

```bash
# Option A: Run the scraper (requires SDGZero access)
python scraper/scrape.py

# Option B: Ask the maintainer for a database dump
```

### 4. Run

```bash
# Interactive demo
python demo_agent.py

# Run test suite
python test_pipeline.py
```

## LLM Providers

Set `LLM_PROVIDER` in `.env`:

| Provider | Model | Free Tier |
|----------|-------|-----------|
| `gemini` (default) | gemini-2.0-flash | 1,500 req/day |
| `groq` | llama-3.1-8b-instant | 500K tokens/day |
| `ollama` | llama3.1:8b | Unlimited (local) |

## Project Structure

```
agent/
  search_agent.py     # HyDE + vector/hybrid/SQL search
  scoring_agent.py    # Cross-encoder rerank + LLM reasoning
  research_agent.py   # Tavily web research
  report_agent.py     # HTML report generation
  state.py            # LangGraph AgentState
  llm.py              # LLM factory (Gemini/Groq/Ollama)
  tools.py            # ChromaDB search tools
db/
  chroma_store.py     # ChromaDB wrapper
scraper/              # SDGZero web scraper
ml/                   # SDG tag prediction model (SetFit)
pipeline/             # LangGraph pipeline definition
```

## Roadmap

- [x] Phase 1: Core pipeline (Search → Research → Score → Report)
- [x] Phase 2: HyDE + filter injection + cross-encoder reranking
- [ ] Phase 3: FastAPI backend + Next.js frontend
- [ ] Phase 4: pgvector migration, hard/soft filter UI

## License

MIT
