"""
Demo: semantic search over SDGZero businesses.

Usage:
    python demo_search.py
    python demo_search.py "renewable energy companies in UK"
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.chroma_store import BusinessStore


def print_results(results: list[dict]) -> None:
    if not results:
        print("  No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r['name']}")
        print(f"      Location   : {', '.join(filter(None, [r['city'], r['region'], r['country']]))}")
        print(f"      Categories : {r['categories'] or 'N/A'}")
        print(f"      Sector     : {r['job_sector'] or 'N/A'}")
        print(f"      SDGs       : {r['sdg_tags'] or 'N/A'}")
        print(f"      Phone      : {r['phone'] or 'N/A'}")
        print(f"      Website    : {r['website'] or 'N/A'}")
        if r['linkedin']:
            print(f"      LinkedIn   : {r['linkedin']}")
        print(f"      URL        : {r['url']}")
        print(f"      Similarity : {r['similarity']:.3f}")


def interactive_search(store: BusinessStore) -> None:
    print("\nSDGZero Semantic Search Demo")
    print("Type a query and press Enter. Type 'quit' to exit.\n")

    example_queries = [
        "companies working on clean energy and sustainability",
        "tech startups focused on reducing carbon emissions",
        "education and youth empowerment organizations",
        "health and wellness businesses",
        "financial services and accounting firms in London",
    ]
    print("Example queries:")
    for q in example_queries:
        print(f"  - {q}")

    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        results = store.search(query, n_results=5)
        print_results(results)


if __name__ == "__main__":
    store = BusinessStore(persist_dir="./chroma_db")

    if store.count() == 0:
        print("Database is empty. Run `python -m pipeline.ingest` first.")
        sys.exit(1)

    print(f"Database has {store.count()} businesses.")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nQuery: {query}")
        results = store.search(query, n_results=5)
        print_results(results)
    else:
        interactive_search(store)
