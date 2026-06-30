"""
RAG retriever for Terraform AWS provider documentation.

Public interface used by the Planner Agent:
    rag_retriever(prompt: str, k: int = 3) -> list[str]

Extended interface for internal agent use:
    query_rag(prompt, k, resource_types, sections) -> list[dict]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    ST_EMBED_MODEL,
)
from tools.embeddings import make_embedding_function

# Singleton: reused across calls so the embedding model is loaded once
_collection: Optional[chromadb.Collection] = None


def _make_embedding_function():
    return make_embedding_function(EMBEDDING_BACKEND, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, ST_EMBED_MODEL)


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    _collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=_make_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _extract_resource_types(text: str) -> list[str]:
    """Pull explicit aws_* resource type mentions out of the prompt."""
    return list(set(re.findall(r"\baws_[a-z][a-z0-9_]+", text.lower())))


def _safe_query(
    collection: chromadb.Collection,
    prompt: str,
    k: int,
    where: dict,
) -> list[dict]:
    """
    Run a ChromaDB query, gracefully handling the case where the filtered
    document count is smaller than k (ChromaDB raises an error in that case).
    """
    for attempt_k in (k, max(1, k // 2), 1):
        try:
            results = collection.query(
                query_texts=[prompt],
                n_results=attempt_k,
                include=["documents", "metadatas", "distances"],
                **({"where": where} if where else {}),
            )
            output = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append(
                    {
                        "content": doc,
                        "resource_type": meta.get("resource_type", ""),
                        "section": meta.get("section", ""),
                        "doc_type": meta.get("doc_type", ""),
                        "score": round(1.0 - float(dist), 4),
                    }
                )
            return output
        except Exception:
            if attempt_k == 1:
                return []
    return []


def query_rag(
    prompt: str,
    k: int = DEFAULT_TOP_K,
    resource_types: list[str] | None = None,
    sections: list[str] | None = None,
) -> list[dict]:
    """
    Semantic search over Terraform docs.

    Args:
        prompt: Natural-language infrastructure description.
        k: Number of chunks to return.
        resource_types: Narrow results to these aws_* types (optional).
        sections: Narrow results to these doc sections, e.g. ["Argument Reference"].

    Returns:
        List of dicts — {content, resource_type, section, doc_type, score}.
        score is cosine similarity in [0, 1]; higher = more relevant.
    """
    collection = _get_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "Vector store is empty. Run `python scripts/ingest_tf_docs.py` first."
        )

    # Build optional metadata filter
    filter_parts = []
    if resource_types:
        if len(resource_types) == 1:
            filter_parts.append({"resource_type": resource_types[0]})
        else:
            filter_parts.append({"$or": [{"resource_type": rt} for rt in resource_types]})
    if sections:
        if len(sections) == 1:
            filter_parts.append({"section": sections[0]})
        else:
            filter_parts.append({"$or": [{"section": s} for s in sections]})

    if len(filter_parts) == 0:
        where: dict = {}
    elif len(filter_parts) == 1:
        where = filter_parts[0]
    else:
        where = {"$and": filter_parts}

    return _safe_query(collection, prompt, k, where)


def _build_priority_types(prompt: str, k: int) -> list[str]:
    """
    Build an ordered, deduplicated list of aws_* resource types for a prompt.

    Merge order — taxonomy first so explicitly-named services always get the
    leading slots, LLM supplements with types the taxonomy didn't catch:
      1. Taxonomy   — keyword patterns; precise for named services (VPC, S3, ECS…)
      2. Explicit   — any aws_* literals already in the prompt text
      3. LLM        — handles free-form phrasing; capped to avoid IAM flooding
    """
    from tools.resource_taxonomy import map_prompt_to_resources
    from tools.resource_extractor import llm_extract_resources

    taxonomy_types = map_prompt_to_resources(prompt)
    explicit_types = _extract_resource_types(prompt)
    llm_types = llm_extract_resources(prompt)

    # Only keep LLM types that the taxonomy/explicit pass didn't already cover,
    # and cap to avoid one service monopolising remaining slots.
    taxonomy_seen = set(taxonomy_types) | set(explicit_types)
    llm_new = [rt for rt in llm_types if rt not in taxonomy_seen]
    llm_capped = llm_new[: max(2, k // 2)]

    seen: set[str] = set()
    priority: list[str] = []
    for rt in taxonomy_types + explicit_types + llm_capped:
        if rt not in seen:
            seen.add(rt)
            priority.append(rt)
    return priority


def _run_pipeline(prompt: str, k: int) -> list[dict]:
    """
    Four-phase retrieval core shared by rag_retriever and rag_retriever_rich.

      1+2+3. Build priority list (taxonomy → explicit → LLM supplement)
      3b.    For each priority type fetch its Argument Reference chunk (one per type)
      4.     Semantic fill-up for remaining slots — skips types already covered

    Results are returned in two score-sorted groups:
      • Priority results first  (confirmed by taxonomy/LLM)
      • Semantic fill second    (kept separate so noise never outranks confirmed types)
    """
    priority_types = _build_priority_types(prompt, k)

    priority_results: list[dict] = []
    seen_content: set[str] = set()
    seen_resource_types: set[str] = set()   # one chunk per resource type

    for rt in priority_types[:k]:
        hits = query_rag(prompt, k=2, resource_types=[rt], sections=["Argument Reference"])
        for hit in hits:
            if hit["content"] not in seen_content:
                priority_results.append(hit)
                seen_content.add(hit["content"])
                seen_resource_types.add(hit["resource_type"])
                break

    # Semantic fill — only add types not already covered by the priority loop
    semantic_results: list[dict] = []
    remaining = k - len(priority_results)
    if remaining > 0:
        semantic = query_rag(prompt, k=k + len(priority_results))
        for r in semantic:
            if (
                r["content"] not in seen_content
                and r["resource_type"] not in seen_resource_types
            ):
                semantic_results.append(r)
                seen_content.add(r["content"])
                seen_resource_types.add(r["resource_type"])
                if len(semantic_results) >= remaining:
                    break

    # Sort each group by score independently so semantic noise never outranks
    # a resource that was explicitly identified by taxonomy or LLM.
    priority_results.sort(key=lambda r: r["score"], reverse=True)
    semantic_results.sort(key=lambda r: r["score"], reverse=True)
    return (priority_results + semantic_results)[:k]


def rag_retriever(prompt: str, k: int = 3) -> list[str]:
    """
    Main tool interface matching system_design.md.
    Returns top-k relevant Terraform doc snippets as plain strings.

    Four-phase retrieval strategy:
      1. Taxonomy     — keyword patterns for named services (VPC, S3, ECS, …)
      2. Explicit     — aws_* literals already in the prompt
      3. LLM          — free-form phrasing supplement (llama3.2:3b); capped so
                        generic IAM additions don't flood taxonomy results
      4. Semantic     — cosine-similarity fill for any remaining slots
    """
    return [r["content"] for r in _run_pipeline(prompt, k)]


def rag_retriever_rich(prompt: str, k: int = 5) -> list[dict]:
    """
    Same 4-phase pipeline as rag_retriever() but returns full dicts
    {content, resource_type, section, doc_type, score} instead of plain strings.
    Used by test scripts for debugging.
    """
    return _run_pipeline(prompt, k)


def collection_stats() -> dict:
    """Return basic stats about the loaded vector store."""
    col = _get_collection()
    count = col.count()
    if count == 0:
        return {"total_chunks": 0}

    # Fetch all metadata in pages to avoid memory spikes on very large collections
    resource_types: set[str] = set()
    doc_types: set[str] = set()
    sections: set[str] = set()
    page_size = 2000
    offset = 0
    while offset < count:
        page = col.get(limit=page_size, offset=offset, include=["metadatas"])
        for m in page["metadatas"]:
            resource_types.add(m.get("resource_type", ""))
            doc_types.add(m.get("doc_type", ""))
            sections.add(m.get("section", ""))
        offset += page_size

    return {
        "total_chunks": count,
        "unique_resource_types": len(resource_types),
        "doc_types": sorted(doc_types),
        "sections": sorted(sections),
    }
