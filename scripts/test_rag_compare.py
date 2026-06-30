"""
Side-by-side comparison of:
  V1 — current system  (taxonomy + LLM one-pass)
  V2 — Approach 4      (RAG-grounded two-pass)

Usage:
    python scripts/test_rag_compare.py                       # 6 built-in prompts, side-by-side
    python scripts/test_rag_compare.py --v2-only             # V2 resource names only
    python scripts/test_rag_compare.py --interactive         # type your own prompts
    python scripts/test_rag_compare.py --interactive --v2-only
    python scripts/test_rag_compare.py --iac-eval 5          # real IaC-Eval prompts
    python scripts/test_rag_compare.py --k 10                # change result count
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.rag_retriever import rag_retriever_rich
from tools.rag_retriever_v2 import rag_retriever_v2

SAMPLE_PROMPTS = [
    "Create an S3 bucket with versioning enabled and server-side encryption.",
    "Create a Route53 hosted zone named example.com with query logging to CloudWatch.",
    "Create an EC2 instance with a security group that allows HTTP and HTTPS traffic.",
    "Set up an IAM role for Lambda that can write to DynamoDB and publish to SNS.",
    "Create a VPC with two public and two private subnets across two availability zones, "
    "an internet gateway, and NAT gateway.",
    "Create an ECS Fargate service with an ALB, target group, CloudWatch log group, "
    "IAM task execution role, and auto-scaling policy.",
]


def _resource_names(results: list[dict]) -> list[str]:
    return [r["resource_type"] for r in results]


def print_comparison(prompt: str, v1: list[dict], v2: list[dict], debug: dict) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"PROMPT: {prompt[:width - 8]}{'...' if len(prompt) > width - 8 else ''}")
    print(f"{'=' * width}")

    v1_names = _resource_names(v1)
    v2_names = _resource_names(v2)
    max_rows = max(len(v1_names), len(v2_names))

    col = 45
    print(f"\n  {'V1 (one-pass)':<{col}}  V2 (two-pass / RAG-grounded)")
    print(f"  {'-' * col}  {'-' * col}")

    for i in range(max_rows):
        left  = f"[{i+1:02d}] {v1_names[i]}" if i < len(v1_names) else ""
        right = f"[{i+1:02d}] {v2_names[i]}" if i < len(v2_names) else ""
        marker = " ◄" if i < len(v2_names) and v2_names[i] not in v1_names else ""
        print(f"  {left:<{col}}  {right}{marker}")

    only_v2 = set(v2_names) - set(v1_names)
    only_v1 = set(v1_names) - set(v2_names)
    if only_v2:
        print(f"\n  V2 added : {sorted(only_v2)}")
    if only_v1:
        print(f"  V1 only  : {sorted(only_v1)}")
    print()


def print_v2_only(prompt: str, v2: list[dict], debug: dict) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"PROMPT: {prompt[:width - 8]}{'...' if len(prompt) > width - 8 else ''}")
    print(f"{'=' * width}")
    # Keep only resources the LLM explicitly identified as necessary (Pass 2 types).
    # This excludes semantic fill results which add noise.
    necessary_types = set(debug["pass2_types"])
    necessary = [r for r in v2 if r["resource_type"] in necessary_types]
    print(f"\n  Necessary resources ({len(necessary)}):")
    for i, r in enumerate(necessary, 1):
        print(f"    [{i:02d}] {r['resource_type']}")
    print()


def run_builtin(k: int, v2_only: bool) -> None:
    print(f"\nRunning {len(SAMPLE_PROMPTS)} prompts (k={k})...\n")
    for prompt in SAMPLE_PROMPTS:
        v2, debug = rag_retriever_v2(prompt, k=k)
        if v2_only:
            print_v2_only(prompt, v2, debug)
        else:
            v1 = rag_retriever_rich(prompt, k=k)
            print_comparison(prompt, v1, v2, debug)


def run_iac_eval(n: int, k: int, v2_only: bool) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets package not installed. Run: pip install datasets")
        sys.exit(1)

    print("Loading IaC-Eval dataset...")
    ds = load_dataset("autoiac-project/iac-eval", split="test")
    sample = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    print(f"Running {len(sample)} IaC-Eval prompts (k={k})...\n")
    for row in sample:
        prompt = row["Prompt"]
        difficulty = row.get("Difficulty", "?")
        v2, debug = rag_retriever_v2(prompt, k=k)
        print(f"[difficulty={difficulty}]")
        if v2_only:
            print_v2_only(prompt, v2, debug)
        else:
            v1 = rag_retriever_rich(prompt, k=k)
            print_comparison(prompt, v1, v2, debug)


def run_interactive(k: int, v2_only: bool) -> None:
    mode = "V2 only" if v2_only else "V1 vs V2 comparison"
    print(f"\n{mode} — interactive mode  (k={k})")
    print("Type your infrastructure prompt and press Enter.")
    print("Commands:  :k <number>  to change k  |  :quit  to exit\n")

    current_k = k
    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not prompt:
            continue
        if prompt.lower() in (":quit", ":exit", ":q"):
            print("Bye.")
            break
        if prompt.startswith(":k "):
            try:
                current_k = int(prompt.split()[1])
                print(f"k set to {current_k}")
            except ValueError:
                print("Usage: :k <number>")
            continue

        v2, debug = rag_retriever_v2(prompt, k=current_k)
        if v2_only:
            print_v2_only(prompt, v2, debug)
        else:
            v1 = rag_retriever_rich(prompt, k=current_k)
            print_comparison(prompt, v1, v2, debug)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare V1 (one-pass) vs V2 (RAG-grounded two-pass) retrieval"
    )
    parser.add_argument("--k", type=int, default=10,
                        help="Top-k results per query (default 10)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode: type prompts one by one")
    parser.add_argument("--iac-eval", type=int, metavar="N",
                        help="Sample N prompts from the real IaC-Eval dataset")
    parser.add_argument("--v2-only", action="store_true",
                        help="Show only V2 resource names (no V1 comparison)")
    args = parser.parse_args()

    if args.interactive:
        run_interactive(args.k, args.v2_only)
    elif args.iac_eval:
        run_iac_eval(args.iac_eval, args.k, args.v2_only)
    else:
        run_builtin(args.k, args.v2_only)


if __name__ == "__main__":
    main()
