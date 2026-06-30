"""
One-time ingestion script: downloads Terraform AWS provider docs from GitHub,
parses each doc into section-level chunks, and stores them in ChromaDB.

Usage:
    python scripts/ingest_tf_docs.py                    # full ingest (~1700 files)
    python scripts/ingest_tf_docs.py --limit 50         # quick test run
    python scripts/ingest_tf_docs.py --doc-type r       # resources only
    python scripts/ingest_tf_docs.py --doc-type d       # data sources only
    python scripts/ingest_tf_docs.py --workers 8        # parallel downloads
    python scripts/ingest_tf_docs.py --clear            # wipe collection first
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chromadb
import requests
from tqdm import tqdm

# Project root on sys.path so config imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHROMADB_BATCH_SIZE,
    EMBEDDING_BACKEND,
    GITHUB_TOKEN,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    REQUEST_DELAY_SEC,
    ST_EMBED_MODEL,
    TF_PROVIDER_REPO,
    TF_PROVIDER_REF,
)
from tools.embeddings import make_embedding_function

RAW_BASE = f"https://raw.githubusercontent.com/{TF_PROVIDER_REPO}/{TF_PROVIDER_REF}"
API_BASE = f"https://api.github.com/repos/{TF_PROVIDER_REPO}"

SECTION_SKIP = {"Import", "Timeouts"}  # sections with low value for generation


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------

def _gh_headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


# Module-level cache so the tree is fetched only once even when called for both r/ and d/
_repo_tree_cache: list[dict] | None = None


def _get_repo_tree() -> list[dict]:
    """
    Fetch the complete file tree for TF_PROVIDER_REF using the Git Trees API.
    One API call, no 1000-item cap (Contents API limit).
    Result is cached for the lifetime of the process.
    """
    global _repo_tree_cache
    if _repo_tree_cache is not None:
        return _repo_tree_cache

    url = f"{API_BASE}/git/trees/{TF_PROVIDER_REF}?recursive=1"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    r.raise_for_status()
    data = r.json()

    if data.get("truncated"):
        print("WARNING: GitHub returned a truncated tree. Some files may be missing.")

    _repo_tree_cache = [item for item in data["tree"] if item["type"] == "blob"]
    return _repo_tree_cache


def fetch_file_list(doc_type: str) -> list[dict]:
    """
    Return list of {name, path} for ALL markdown docs under website/docs/{r|d}/.
    Uses the Git Trees API — no 1000-item cap unlike the Contents API.
    """
    tree = _get_repo_tree()
    prefix = f"website/docs/{doc_type}/"
    valid_ext = (".html.markdown", ".html.md", ".mdx", ".md")

    return [
        {"name": item["path"].split("/")[-1], "path": item["path"]}
        for item in tree
        if item["path"].startswith(prefix)
        # Direct children only — skip any subdirectory files
        and "/" not in item["path"][len(prefix):]
        and item["path"].endswith(valid_ext)
    ]


def download_raw(path: str) -> str:
    """Download a file from raw.githubusercontent.com."""
    url = f"{RAW_BASE}/{path}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_JSX_TAG_RE = re.compile(r"<[A-Z][A-Za-z]+[^>]*>.*?</[A-Z][A-Za-z]+>", re.DOTALL)
_INLINE_JSX_RE = re.compile(r"<[A-Z][A-Za-z]+[^/]*/?>")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")       # keep link text, drop URL
_INTERNAL_LINK_RE = re.compile(r"\[`([^`]+)`\]\([^)]+\)")


def _clean(text: str) -> str:
    text = _FRONT_MATTER_RE.sub("", text)
    text = _JSX_TAG_RE.sub("", text)
    text = _INLINE_JSX_RE.sub("", text)
    # Simplify internal doc links: [`aws_foo`](/...) → `aws_foo`
    text = _INTERNAL_LINK_RE.sub(r"`\1`", text)
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_resource_type(filename: str, doc_type: str) -> str:
    """
    'lb_listener.html.markdown' + 'r'  -> 'aws_lb_listener'
    'vpc.html.markdown'         + 'd'  -> 'data.aws_vpc'   (Terraform data-source notation)
    """
    stem = filename
    for ext in (".html.markdown", ".html.md", ".mdx", ".md"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    base = f"aws_{stem}"
    return f"data.{base}" if doc_type == "d" else base


def _split_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """
    Split text into chunks ≤ max_chars. Tries to break on paragraph boundaries.
    Adjacent chunks share `overlap` characters of context.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    paragraphs = re.split(r"\n\n+", text)
    current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
                # Carry overlap into next chunk
                tail = current[-overlap:] if overlap else ""
                current = (tail + "\n\n" + para).strip() if tail else para
            else:
                # Single paragraph exceeds limit — hard-split on lines
                lines = para.split("\n")
                sub = ""
                for line in lines:
                    trial = (sub + "\n" + line).strip() if sub else line
                    if len(trial) <= max_chars:
                        sub = trial
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = line
                current = sub

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


def parse_doc(content: str, resource_type: str, doc_type: str, source_file: str) -> list[dict]:
    """
    Parse a Terraform markdown doc into a list of ChromaDB-ready chunks.

    Each chunk:
        id       – unique string, stable across re-ingestion
        content  – text stored and embedded
        metadata – resource_type, doc_type, section, source_file
    """
    content = _clean(content)

    # Split on level-2 headings: intro + each ## Section
    raw_parts = re.split(r"\n## ", content)

    intro_text = raw_parts[0].strip()
    sections: list[tuple[str, str]] = []
    for part in raw_parts[1:]:
        nl = part.find("\n")
        if nl == -1:
            sections.append((part.strip(), ""))
        else:
            sections.append((part[:nl].strip(), part[nl:].strip()))

    chunks: list[dict] = []

    # Stable ID prefix: include doc_type so r/ and d/ never collide
    id_prefix = f"{doc_type}__{resource_type}"

    # Intro chunk (resource title + description)
    if intro_text:
        prefix = f"Resource: {resource_type}\nSection: Description\n\n"
        for i, text in enumerate(_split_text(prefix + intro_text)):
            # Every sub-chunk carries the resource context header
            if i > 0 and not text.startswith("Resource:"):
                text = prefix + text
            chunks.append(
                {
                    "id": f"{id_prefix}__desc__{i}",
                    "content": text,
                    "metadata": {
                        "resource_type": resource_type,
                        "doc_type": doc_type,
                        "section": "Description",
                        "source_file": source_file,
                    },
                }
            )

    # Per-section chunks
    for sec_idx, (sec_name, sec_body) in enumerate(sections):
        if sec_name in SECTION_SKIP or not sec_body:
            continue

        prefix = f"Resource: {resource_type}\nSection: {sec_name}\n\n"
        for chunk_idx, text in enumerate(_split_text(prefix + sec_body)):
            # Every sub-chunk carries the resource context header
            if chunk_idx > 0 and not text.startswith("Resource:"):
                text = prefix + text
            chunks.append(
                {
                    "id": f"{id_prefix}__s{sec_idx}__{chunk_idx}",
                    "content": text,
                    "metadata": {
                        "resource_type": resource_type,
                        "doc_type": doc_type,
                        "section": sec_name,
                        "source_file": source_file,
                    },
                }
            )

    return chunks


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def _make_ef():
    return make_embedding_function(EMBEDDING_BACKEND, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, ST_EMBED_MODEL)


def get_or_create_collection(clear: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    if clear:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
            print(f"Cleared collection '{CHROMA_COLLECTION_NAME}'")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=_make_ef(),
        metadata={"hnsw:space": "cosine"},
    )


def upsert_batch(collection: chromadb.Collection, chunks: list[dict]) -> None:
    """Upsert a list of chunks (idempotent — re-running is safe)."""
    for start in range(0, len(chunks), CHROMADB_BATCH_SIZE):
        batch = chunks[start : start + CHROMADB_BATCH_SIZE]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["content"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


# ---------------------------------------------------------------------------
# Per-file worker (used in thread pool)
# ---------------------------------------------------------------------------

def process_file(entry: dict, doc_type_label: str) -> tuple[list[dict], str | None]:
    """
    Download + parse one doc file.
    Returns (chunks, error_message). error_message is None on success.
    """
    try:
        content = download_raw(entry["path"])
        resource_type = _extract_resource_type(entry["name"], doc_type_label)
        chunks = parse_doc(content, resource_type, doc_type_label, entry["path"])
        if REQUEST_DELAY_SEC:
            time.sleep(REQUEST_DELAY_SEC)
        return chunks, None
    except Exception as exc:
        return [], str(exc)


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

def run_ingestion(
    doc_types: list[str],
    limit: int | None,
    workers: int,
    clear: bool,
) -> None:
    collection = get_or_create_collection(clear=clear)

    # Collect existing IDs to skip already-ingested docs (incremental update)
    existing_ids: set[str] = set()
    if not clear:
        total_existing = collection.count()
        if total_existing > 0:
            print(f"Collection already has {total_existing} chunks — skipping already-ingested resources.")
            existing = collection.get(include=["metadatas"])
            existing_ids = {
                m.get("resource_type", "") for m in existing["metadatas"]
            }

    all_files: list[tuple[dict, str]] = []  # (entry, doc_type_label)
    for dt in doc_types:
        print(f"Fetching file list for docs/{dt}/ ...")
        files = fetch_file_list(dt)
        print(f"  Found {len(files)} files")
        if limit:
            files = files[:limit]
        all_files.extend((f, dt) for f in files)

    # Filter out already-ingested resource types
    if existing_ids:
        before = len(all_files)
        all_files = [
            (f, dt)
            for f, dt in all_files
            if _extract_resource_type(f["name"], dt) not in existing_ids
        ]
        skipped = before - len(all_files)
        if skipped:
            print(f"Skipping {skipped} already-ingested resource types.")

    if not all_files:
        print("Nothing to ingest — collection is up to date.")
        return

    print(f"\nIngesting {len(all_files)} files with {workers} worker(s)...\n")

    total_chunks = 0
    errors: list[str] = []
    pending: list[dict] = []

    def flush(force: bool = False) -> None:
        nonlocal total_chunks
        if pending and (force or len(pending) >= CHROMADB_BATCH_SIZE):
            upsert_batch(collection, pending)
            total_chunks += len(pending)
            pending.clear()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_file, entry, dt): entry["name"]
            for entry, dt in all_files
        }
        with tqdm(total=len(futures), unit="file") as bar:
            for future in as_completed(futures):
                fname = futures[future]
                chunks, err = future.result()
                if err:
                    errors.append(f"{fname}: {err}")
                else:
                    pending.extend(chunks)
                    if len(pending) >= CHROMADB_BATCH_SIZE:
                        flush()
                bar.set_postfix(chunks=total_chunks + len(pending), errors=len(errors))
                bar.update(1)

    flush(force=True)

    print(f"\nDone. {total_chunks} chunks stored in '{CHROMA_DB_PATH}'.")
    print(f"Collection total: {collection.count()} chunks.")
    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Terraform AWS provider docs into ChromaDB")
    parser.add_argument(
        "--doc-type",
        choices=["r", "d", "both"],
        default="both",
        help="r = resources, d = data sources, both = all (default)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max files per doc-type (for quick tests)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel download workers (default 6)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete and recreate the collection before ingesting",
    )
    args = parser.parse_args()

    doc_types = ["r", "d"] if args.doc_type == "both" else [args.doc_type]
    run_ingestion(
        doc_types=doc_types,
        limit=args.limit,
        workers=args.workers,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()
