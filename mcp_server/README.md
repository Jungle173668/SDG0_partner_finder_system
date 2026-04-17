# SDGZero MCP Server

Exposes the SDGZero Partner Finder as MCP tools for Claude Desktop / Cursor.

## Tools

| Tool | Description |
|------|-------------|
| `search_companies` | Semantic search by natural language + optional filters |
| `filter_companies` | Pure SQL metadata filtering (city, SDG, category, etc.) |
| `get_company` | Full profile of a single company by slug |
| `find_partners` | Full Multi-Agent Pipeline — returns top-5 partner report |

## Installation

```bash
pip install mcp[cli]
```

## Local Setup

### 1. Set environment variables

Ensure your `.env` file (or shell environment) has:

```
DATABASE_URL=postgresql://sdgzero:sdgzero@localhost:5432/sdgzero
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key   # required for find_partners
```

### 2. Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sdgzero": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/SDG0-system",
      "env": {
        "DATABASE_URL": "postgresql://sdgzero:sdgzero@localhost:5432/sdgzero",
        "GROQ_API_KEY": "your_groq_key",
        "TAVILY_API_KEY": "your_tavily_key"
      }
    }
  }
}
```

Replace `/absolute/path/to/SDG0-system` with the actual path to your project.

### 3. Restart Claude Desktop

After saving the config, restart Claude Desktop. You should see the SDGZero tools available.

## Example Queries in Claude Desktop

- *"Find me sustainability consultancies in London"*
  → calls `search_companies(query="sustainability consultancy", city="London")`

- *"List all companies focused on Climate Action"*
  → calls `filter_companies(sdg="Climate Action")`

- *"Tell me about heat-engineer-software-ltd"*
  → calls `get_company(slug="heat-engineer-software-ltd")`

- *"I run a carbon audit company in London. Find me partners."*
  → calls `find_partners(my_company_desc="...", city="London")`

## Notes

- `search_companies` and `filter_companies` are fast (< 1s).
- `find_partners` runs the full pipeline and takes 20–60 seconds.
- The server connects to your local PostgreSQL database by default.
  To use the Railway production DB, set `DATABASE_URL` to the public Railway URL.
