import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# --- ChromaDB ---
CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = "terraform_aws_docs"

# --- Embedding ---
# "ollama" uses nomic-embed-text locally (no internet needed after pull)
# "sentence_transformers" uses all-MiniLM-L6-v2 (downloaded on first use)
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "ollama")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = "nomic-embed-text"
ST_EMBED_MODEL = "all-MiniLM-L6-v2"

# --- GitHub source ---
TF_PROVIDER_REPO = "hashicorp/terraform-provider-aws"
TF_PROVIDER_REF = "main"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# --- Ingestion ---
# Each markdown section becomes a chunk; sections longer than this are split further
CHUNK_MAX_CHARS = 1500
CHUNK_OVERLAP_CHARS = 150
CHROMADB_BATCH_SIZE = 50    # upsert batch size
REQUEST_DELAY_SEC = 0.05    # between raw.githubusercontent.com downloads

# --- RAG query ---
DEFAULT_TOP_K = 15
