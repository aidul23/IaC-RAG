"""
RAG-grounded extraction — Approach 4 (two-pass pipeline).

How it works:
  Pass 1  LLM reads the prompt → initial aws_* resource list
  RAG-1   Retrieve Argument Reference docs for those initial types
  Pass 2  LLM reads the prompt PLUS the retrieved docs → corrected/expanded list
  RAG-2   Final retrieval using the corrected list, sorted by score

The key insight: Terraform docs say which arguments are "Required" and often
reference other resource types. Pass 2 exploits this to catch resources that
the initial LLM guess missed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEFAULT_TOP_K, OLLAMA_BASE_URL
from tools.resource_extractor import _get_valid_types, llm_extract_resources
from tools.rag_retriever import query_rag

# Model used for Pass 2 reasoning.
# A larger model (llama3.1:8b, mistral:7b) gives better results here because
# it needs to read doc snippets and reason about implicit dependencies.
# Change this to any model you have pulled in Ollama.
PASS2_MODEL = "llama3.2:3b"

_PASS2_SYSTEM = """\
You are a Terraform and AWS expert.

You are given:
  1. An infrastructure request (the original prompt)
  2. Terraform documentation excerpts for an initial set of resources

Your task:
  - Read the documentation carefully, paying attention to "Required" fields
  - Notice when a Required field references another resource type
    (e.g. "subnet_ids – (Required) List of VPC subnet IDs" means aws_subnet is needed)
  - Return a COMPLETE, corrected list of ALL aws_* resource types needed

Output ONLY a valid JSON array of aws_* resource type strings.
No explanation. No markdown. No extra text."""


def _extract_required_hints(chunk: str, max_chars: int = 500) -> str:
    """
    Pull lines that mention 'Required' from a doc chunk.
    These reveal which other resources must exist, keeping the Pass 2 prompt
    short enough to fit in a small context window.
    """
    lines = chunk.split("\n")
    required_lines = [
        ln.strip()
        for ln in lines
        if "Required" in ln and "Optional" not in ln and ln.strip()
    ]
    if required_lines:
        return "\n".join(required_lines)[:max_chars]
    return chunk[:max_chars]


def _pass2_correct(
    prompt: str,
    initial_types: list[str],
    pass1_docs: list[dict],
    model: str = PASS2_MODEL,
    timeout: int = 90,
) -> list[str]:
    """
    Pass 2: LLM reads the retrieved docs and returns a corrected resource list.

    Falls back to `initial_types` if the model is unavailable or returns nothing.
    """
    # Build compact doc context — one block per unique resource type
    seen_types: set[str] = set()
    doc_blocks: list[str] = []
    for doc in pass1_docs:
        rt = doc["resource_type"]
        if rt in seen_types:
            continue
        seen_types.add(rt)
        hint = _extract_required_hints(doc["content"])
        if hint:
            doc_blocks.append(f"### {rt}\n{hint}")

    doc_context = "\n\n".join(doc_blocks)

    full_prompt = (
        f"{_PASS2_SYSTEM}\n\n"
        f"Infrastructure request:\n{prompt}\n\n"
        f"Initial resource list: {initial_types}\n\n"
        f"Terraform documentation (Required fields only):\n{doc_context}\n\n"
        f"Complete corrected resource list (JSON array):"
    )

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 400},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        raw = r.json().get("response", "")
    except Exception:
        return initial_types

    candidates = re.findall(r"\baws_[a-z][a-z0-9_]+", raw)

    seen: set[str] = set()
    deduped: list[str] = []
    for rt in candidates:
        if rt not in seen:
            seen.add(rt)
            deduped.append(rt)

    valid = _get_valid_types()
    corrected = [rt for rt in deduped if rt in valid]

    return corrected if corrected else initial_types


def rag_retriever_v2(
    prompt: str,
    k: int = DEFAULT_TOP_K,
) -> tuple[list[dict], dict]:
    """
    Two-pass RAG retrieval (Approach 4).

    Returns:
        results   – list of dicts {content, resource_type, section, doc_type, score}
                    sorted by score descending, length <= k
        debug     – dict with pass1_types and pass2_types for comparison/logging
    """
    # ── Pass 1: initial LLM extraction ──────────────────────────────────────
    initial_types = llm_extract_resources(prompt)

    # ── RAG-1: retrieve docs for the initial types ───────────────────────────
    pass1_docs: list[dict] = []
    seen_content: set[str] = set()

    if initial_types:
        for rt in initial_types:
            hits = query_rag(prompt, k=2, resource_types=[rt], sections=["Argument Reference"])
            for hit in hits:
                if hit["content"] not in seen_content:
                    pass1_docs.append(hit)
                    seen_content.add(hit["content"])
                    break
    else:
        # No LLM output — seed with pure semantic search
        pass1_docs = query_rag(prompt, k=min(8, k))
        seen_content = {d["content"] for d in pass1_docs}

    # ── Pass 2: LLM reads docs and corrects the list ─────────────────────────
    corrected_types = _pass2_correct(prompt, initial_types, pass1_docs)

    # ── RAG-2: final retrieval using the corrected types ─────────────────────
    priority_results: list[dict] = []
    seen_content_final: set[str] = set()
    seen_rt_final: set[str] = set()

    for rt in corrected_types[:k]:
        hits = query_rag(prompt, k=2, resource_types=[rt], sections=["Argument Reference"])
        for hit in hits:
            if hit["content"] not in seen_content_final:
                priority_results.append(hit)
                seen_content_final.add(hit["content"])
                seen_rt_final.add(hit["resource_type"])
                break

    # Semantic fill — skip types already covered, keep in a separate list
    semantic_results: list[dict] = []
    remaining = k - len(priority_results)
    if remaining > 0:
        semantic = query_rag(prompt, k=k + len(priority_results))
        for r in semantic:
            if (
                r["content"] not in seen_content_final
                and r["resource_type"] not in seen_rt_final
            ):
                semantic_results.append(r)
                seen_content_final.add(r["content"])
                seen_rt_final.add(r["resource_type"])
                if len(semantic_results) >= remaining:
                    break

    # Priority results first (sorted by score), semantic fill second (sorted by score)
    priority_results.sort(key=lambda r: r["score"], reverse=True)
    semantic_results.sort(key=lambda r: r["score"], reverse=True)
    targeted = priority_results + semantic_results

    debug = {
        "pass1_types": initial_types,
        "pass2_types": corrected_types,
    }
    return targeted[:k], debug
