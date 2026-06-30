# IaC-RAG: Retrieval-Augmented Generation for Terraform AWS Infrastructure

A RAG system that retrieves relevant Terraform AWS provider documentation given a natural-language infrastructure prompt. Built as a component of a multi-agent Infrastructure-as-Code (IaC) generation system, evaluated against the [IaC-Eval dataset](https://huggingface.co/datasets/autoiac-project/iac-eval).

---

## Overview

Given a prompt like:

> *"Set up a VPC with two subnets, an internet gateway, and a security group for MySQL access"*

The system retrieves the exact Terraform resource documentation needed to generate the correct HCL code — `aws_vpc`, `aws_subnet`, `aws_internet_gateway`, `aws_security_group`, etc.

The project implements and compares **two retrieval strategies**:

| | Strategy | Description |
|---|---|---|
| **V1** | One-pass | Keyword taxonomy + LLM extraction → semantic fill |
| **V2** | Two-pass (RAG-grounded) | LLM extracts → RAG retrieves docs → LLM re-reads docs and corrects list |

---

## Architecture

```
Prompt
  │
  ├─► Keyword Taxonomy      (80+ AWS service patterns → aws_* types)
  ├─► LLM Extraction        (llama3.2:3b identifies resource types)
  └─► Explicit Mentions     (aws_* literals in the prompt)
        │
        ▼
  Priority Resource List
        │
        ├─► Argument Reference lookup per type  (ChromaDB)
        └─► Semantic fill for remaining slots   (cosine similarity)
              │
              ▼
        Ranked Results (sorted by score, priority types first)
```

**V2 adds a second pass:**
```
Pass 1 results → RAG retrieves their docs → LLM reads "Required" fields
              → corrected resource list → final retrieval
```

### Key components

| File | Purpose |
|------|---------|
| `config.py` | All configuration (models, paths, chunk sizes) |
| `scripts/ingest_tf_docs.py` | Downloads Terraform AWS docs from GitHub → ChromaDB |
| `tools/embeddings.py` | Ollama / SentenceTransformers embedding wrapper |
| `tools/resource_taxonomy.py` | Keyword → resource type mapping (80+ patterns) |
| `tools/resource_extractor.py` | LLM-based resource type extraction (Pass 1) |
| `tools/rag_retriever.py` | V1 retriever: `rag_retriever(prompt, k)` |
| `tools/rag_retriever_v2.py` | V2 retriever: two-pass RAG-grounded extraction |
| `scripts/test_rag.py` | Test V1 against built-in / IaC-Eval prompts |
| `scripts/test_rag_compare.py` | Side-by-side comparison of V1 vs V2 |

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally
- The following models pulled in Ollama:

```bash
ollama pull nomic-embed-text   # embedding model
ollama pull llama3.2:3b        # LLM for resource extraction
```

---

## Installation

**1. Create and activate a conda environment:**

```bash
conda create -n iac-eval python=3.10 -y
conda activate iac-eval
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. (Optional) Set environment variables:**

```bash
# GitHub token — increases API rate limit during ingestion
export GITHUB_TOKEN=ghp_your_token_here

# HuggingFace token — needed to load IaC-Eval dataset
export HF_TOKEN=hf_your_token_here
```

---

## Step 1 — Ingest Terraform Documentation

This downloads all Terraform AWS provider docs from GitHub and stores them in ChromaDB. Only needs to be run once.

```bash
# Full ingest (~2,300 resource files, takes ~20 minutes)
python scripts/ingest_tf_docs.py

# Quick test (first 50 files only)
python scripts/ingest_tf_docs.py --limit 50

# Resources only (skip data sources)
python scripts/ingest_tf_docs.py --doc-type r

# Wipe and re-ingest from scratch
python scripts/ingest_tf_docs.py --clear

# Faster with more parallel workers
python scripts/ingest_tf_docs.py --workers 8
```

Check the collection was built correctly:

```bash
python scripts/test_rag.py --stats
```

---

## Step 2 — Run the RAG System

### Option A — V1 only (one-pass retrieval)

```bash
# Interactive mode — type any prompt
python scripts/test_rag.py --interactive

# Built-in test prompts
python scripts/test_rag.py --k 10

# Real prompts from IaC-Eval dataset
python scripts/test_rag.py --iac-eval 10 --k 10
```

**Inside interactive mode:**
```
Prompt> Create an S3 bucket with versioning and server-side encryption
Prompt> :k 15       # change number of results
Prompt> :quit       # exit
```

---

### Option B — V2 only (two-pass RAG-grounded extraction)

Returns only the resources the LLM identified as necessary — no fixed count, no noise.

```bash
# Interactive mode
python scripts/test_rag_compare.py --interactive --v2-only

# Built-in test prompts
python scripts/test_rag_compare.py --v2-only --k 10

# Real prompts from IaC-Eval dataset
python scripts/test_rag_compare.py --iac-eval 10 --v2-only
```

**Example output:**
```
======================================================================
PROMPT: Configure a Route 53 record with an Elastic Load Balancer reso...
======================================================================

  Necessary resources (6):
    [01] aws_route53_record
    [02] aws_route53_zone
    [03] aws_elb
    [04] aws_subnet
    [05] aws_security_group
    [06] aws_vpc
```

---

### Option C — Compare V1 vs V2 side by side

```bash
# Interactive mode
python scripts/test_rag_compare.py --interactive

# Built-in test prompts
python scripts/test_rag_compare.py --k 10

# Real prompts from IaC-Eval dataset
python scripts/test_rag_compare.py --iac-eval 10 --k 10
```

**Example output:**
```
  V1 (one-pass)                              V2 (two-pass / RAG-grounded)
  ---------------------------------------------  -----------------------------------------------
  [01] aws_route53_zone                      [01] aws_route53_record  ◄
  [02] aws_elb                               [02] aws_route53_zone
  [03] aws_lb                                [03] aws_elb
  ...

  V2 added : ['aws_route53_record']
  V1 only  : ['aws_lb']
```

The `◄` marker highlights resources V2 found that V1 missed.

---

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_BACKEND` | `"ollama"` | `"ollama"` or `"sentence_transformers"` |
| `OLLAMA_EMBED_MODEL` | `"nomic-embed-text"` | Embedding model (Ollama) |
| `ST_EMBED_MODEL` | `"all-MiniLM-L6-v2"` | Embedding model (SentenceTransformers) |
| `DEFAULT_TOP_K` | `15` | Default number of results |
| `CHUNK_MAX_CHARS` | `1500` | Max characters per doc chunk |

To change the LLM extraction model, edit `tools/resource_extractor.py`:

```python
EXTRACTOR_MODEL = "llama3.2:3b"   # change to llama3.1:8b for better accuracy
```

To change the V2 Pass 2 model, edit `tools/rag_retriever_v2.py`:

```python
PASS2_MODEL = "llama3.2:3b"
```

> **Important:** If you change the embedding model, you must re-ingest the docs:
> ```bash
> python scripts/ingest_tf_docs.py --clear
> ```

---

## Dataset

The system is evaluated against the [IaC-Eval](https://huggingface.co/datasets/autoiac-project/iac-eval) benchmark dataset (458 tasks, split=`test`). Only the `Prompt` column is used as input — no ground truth leakage.

---

## How V1 and V2 Differ

**V1** uses a single-pass pipeline:
1. Keyword taxonomy matches service names in the prompt
2. LLM supplements with resource types not covered by taxonomy
3. Semantic fill for remaining slots

**V2** adds a grounding step using the retrieved docs themselves:
1. LLM makes an initial resource list (Pass 1)
2. RAG retrieves the Argument Reference docs for those resources
3. LLM reads the docs — especially `(Required)` fields — and corrects its list (Pass 2)
4. Final retrieval using the corrected list

The key benefit of V2: Terraform docs explicitly state which other resources are required. For example, the `aws_db_subnet_group` doc says `subnet_ids (Required)`, which tells the LLM that `aws_subnet` must also be included — even if it was not in the original guess.
