"""
Embedding function factories for ChromaDB.

Supports two backends:
  - "ollama"               : calls the local Ollama REST API (no extra pip dep)
  - "sentence_transformers": uses sentence-transformers (all-MiniLM-L6-v2)

The returned objects satisfy ChromaDB's EmbeddingFunction protocol:
they are callable with list[str] -> list[list[float]].
"""

from __future__ import annotations

import re
from typing import List

import requests as _requests

# nomic-embed-text has a 2048-token context; ~6000 chars is a safe upper bound
_MAX_TEXT_CHARS = 6000
_FALLBACK_BATCH = 10  # batch size when retrying after a 400 error


def _sanitize(text: str) -> str:
    """Ensure text is safe to send to Ollama: non-empty, ASCII-safe, bounded length."""
    if not text or not text.strip():
        return "empty"
    # Remove null bytes and lone surrogates that break JSON serialisation
    text = text.replace("\x00", " ")
    text = re.sub(r"[\ud800-\udfff]", "", text)
    return text[:_MAX_TEXT_CHARS]


class OllamaEmbedding:
    """
    Calls the Ollama /api/embed endpoint in batches.
    Requires the local Ollama daemon to be running with nomic-embed-text pulled.
    Falls back to single-item requests if a batch returns HTTP 400.
    """

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.url = f"{base_url.rstrip('/')}/api/embed"

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        r = _requests.post(
            self.url,
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        clean = [_sanitize(t) for t in texts]
        try:
            return self._embed_batch(clean)
        except _requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 400:
                # Bisect the batch to find and isolate the problematic text
                result: List[List[float]] = []
                for i in range(0, len(clean), _FALLBACK_BATCH):
                    sub = clean[i : i + _FALLBACK_BATCH]
                    try:
                        result.extend(self._embed_batch(sub))
                    except _requests.HTTPError:
                        # Final fallback: embed one at a time, replace failures with zeros
                        for t in sub:
                            try:
                                result.extend(self._embed_batch([t]))
                            except Exception:
                                result.append([0.0] * 768)
                return result
            raise

    # ChromaDB 1.x calls __call__ for document embedding, embed_query for query embedding
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embed(list(input))

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self._embed(list(input))

    def name(self) -> str:
        return f"ollama_{self.model}"

    def get_config(self) -> dict:
        return {"model": self.model, "url": self.url}

    @classmethod
    def build_from_config(cls, config: dict) -> "OllamaEmbedding":
        return cls(model=config["model"], base_url=config["url"])


def make_embedding_function(backend: str, ollama_url: str, ollama_model: str, st_model: str):
    """
    Return the right embedding function based on the configured backend.

    Args:
        backend      : "ollama" or "sentence_transformers"
        ollama_url   : Ollama base URL
        ollama_model : Ollama model name (e.g. "nomic-embed-text")
        st_model     : SentenceTransformer model name (e.g. "all-MiniLM-L6-v2")
    """
    if backend == "ollama":
        return OllamaEmbedding(model=ollama_model, base_url=ollama_url)

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=st_model)
