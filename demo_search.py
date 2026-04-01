"""
Demo: semantic search over SDG: Zero businesses (PostgreSQL + pgvector).

Usage:
    python demo_search.py
    python demo_search.py "renewable energy companies in UK"
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_results(results: list[dict]) -> None:
    if not results:
        print("  No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r['name']}")
        print(f"      Location   : {', '.join(filter(None, [r.get('city'), r.get('region'), r.get('country')]))}")
        print(f"      Categories : {r.get('categories') or 'N/A'}")
        print(f"      Sector     : {r.get('job_sector') or 'N/A'}")
        print(f"      SDGs       : {r.get('sdg_tags') or 'N/A'}")
        print(f"      Phone      : {r.get('phone') or 'N/A'}")
        print(f"      Website    : {r.get('website') or 'N/A'}")
        if r.get('linkedin'):
            print(f"      LinkedIn   : {r['linkedin']}")
        print(f"      URL        : {r.get('url')}")
        sim = r.get('similarity')
        if sim is not None:
            print(f"      Similarity : {sim:.3f}")


if __name__ == "__main__":
    from agent.tools import semantic_search

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Query: ").strip()
        if not query:
            sys.exit(0)

    print(f"\nQuery: {query}")
    results = semantic_search(query, n_results=5)
    print_results(results)
