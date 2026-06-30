"""
LLM-based resource type extractor.

Sends the user prompt to a local Ollama model and asks it to identify
which aws_* Terraform resource types are needed. Results are filtered
against the actual collection so hallucinated types are silently dropped.

This is the primary resource-identification step in rag_retriever.
The keyword taxonomy is kept as a fast supplement for well-known services.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL

# Model to use for extraction — llama3.2:3b is fast and accurate enough
EXTRACTOR_MODEL = "qwen2.5-coder:7b"

_SYSTEM_PROMPT = """\
You are a Terraform and AWS expert.
Given an infrastructure request, list every AWS Terraform resource type \
(aws_* format) that must be created to fulfil the request.
Include all supporting resources: IAM roles, security groups, log groups, \
subnets, etc.
Output ONLY a valid JSON array of resource type strings. No explanation, \
no markdown, no extra text."""

# Cached set of resource types that actually exist in the collection.
# Populated lazily on first call so importing this module has no side effects.
_valid_types_cache: Optional[set[str]] = None


def _get_valid_types() -> set[str]:
    """Return the set of resource types present in the ChromaDB collection."""
    global _valid_types_cache
    if _valid_types_cache is not None:
        return _valid_types_cache

    # Import here to avoid circular dependency
    from tools.rag_retriever import _get_collection
    col = _get_collection()

    valid: set[str] = set()
    page_size = 2000
    offset = 0
    total = col.count()
    while offset < total:
        page = col.get(limit=page_size, offset=offset, include=["metadatas"])
        for m in page["metadatas"]:
            rt = m.get("resource_type", "")
            if rt:
                valid.add(rt)
        offset += page_size

    _valid_types_cache = valid
    return valid


def llm_extract_resources(
    prompt: str,
    model: str = EXTRACTOR_MODEL,
    timeout: int = 45,
) -> list[str]:
    """
    Ask the local LLM which aws_* resource types the prompt requires.

    Returns an ordered, deduplicated list of resource types that are
    confirmed to exist in the ChromaDB collection.
    Hallucinated or unknown types are silently dropped.
    Returns [] on any error so callers can fall back gracefully.
    """
    full_prompt = f"{_SYSTEM_PROMPT}\n\nInfrastructure request: {prompt}\n\nOutput:"

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        raw = r.json().get("response", "")
    except Exception:
        return []

    # Extract every aws_* token the model produced
    candidates = re.findall(r"\baws_[a-z][a-z0-9_]+", raw)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for rt in candidates:
        if rt not in seen:
            seen.add(rt)
            deduped.append(rt)

    # Filter to types that actually exist in the collection
    valid = _get_valid_types()
    return [rt for rt in deduped if rt in valid]
