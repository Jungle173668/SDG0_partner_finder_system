"""
ChromaDB storage layer for SDGZero businesses.
Handles embedding generation and vector search.
"""

import os
from typing import Optional

os.environ["ANONYMIZED_TELEMETRY"] = "False"  # suppress ChromaDB telemetry warnings

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from scraper.models import Business

COLLECTION_NAME = "sdgzero_businesses"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, free


class BusinessStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        print(f"Connected to ChromaDB at '{persist_dir}' — {self.collection.count()} docs")

    def upsert(self, businesses: list[Business], batch_size: int = 50) -> None:
        """Embed and upsert businesses into ChromaDB."""
        total = len(businesses)
        for i in range(0, total, batch_size):
            batch = businesses[i : i + batch_size]

            ids = [str(b.id) for b in batch]
            texts = [b.to_embedding_text() for b in batch]
            metadatas = [b.to_metadata() for b in batch]

            embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            print(f"  Upserted {min(i + batch_size, total)}/{total} businesses")

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic search over businesses.

        Args:
            query:     Natural language query
            n_results: Number of results to return
            where:     Optional ChromaDB metadata filter
                       e.g. {"city": "London"}
                       e.g. {"$and": [{"country": "United Kingdom"}, {"package_id": 9}]}

        Returns:
            List of rich business dicts ready for Agent consumption.
        """
        query_embedding = self.model.encode([query]).tolist()

        kwargs = {
            "query_embeddings": query_embedding,
            "n_results": n_results,
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        output = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            output.append({
                # identity
                "id": results["ids"][0][i],
                "slug": meta.get("slug", ""),
                "name": meta.get("name", ""),
                "url": meta.get("url", ""),
                # location
                "street": meta.get("street", ""),
                "city": meta.get("city", ""),
                "region": meta.get("region", ""),
                "country": meta.get("country", ""),
                "zip": meta.get("zip", ""),
                # contact
                "phone": meta.get("phone", ""),
                "website": meta.get("website", ""),
                "linkedin": meta.get("linkedin", ""),
                "facebook": meta.get("facebook", ""),
                "twitter": meta.get("twitter", ""),
                # business info
                "categories": meta.get("categories", ""),
                "job_sector": meta.get("job_sector", ""),
                "business_type": meta.get("business_type", ""),
                "company_size": meta.get("company_size", ""),
                "package_id": meta.get("package_id", 1),
                "claimed": meta.get("claimed", ""),
                "founder_name": meta.get("founder_name", ""),
                # sdg
                "sdg_tags": meta.get("sdg_tags", ""),
                "membership_tier": meta.get("membership_tier", ""),
                # search score
                "similarity": round(1 - results["distances"][0][i], 4),
                # full embedding text (for Agent RAG context)
                "document": results["documents"][0][i],
            })
        return output

    def get_by_slug(self, slug: str) -> Optional[dict]:
        """Fetch a single business by slug."""
        results = self.collection.get(
            where={"slug": slug},
            include=["metadatas", "documents"],
        )
        if not results["ids"]:
            return None
        meta = results["metadatas"][0]
        meta["document"] = results["documents"][0]
        return meta

    def count(self) -> int:
        return self.collection.count()
