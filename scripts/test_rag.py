"""
Test the RAG retriever against sample prompts representative of IaC-Eval tasks.

Usage:
    python scripts/test_rag.py                # run built-in prompts
    python scripts/test_rag.py --interactive  # type your own prompts (REPL mode)
    python scripts/test_rag.py --iac-eval 5  # sample 5 real IaC-Eval prompts
    python scripts/test_rag.py --stats        # print collection stats only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.rag_retriever import collection_stats, rag_retriever_rich

# A representative cross-section of IaC-Eval difficulty levels and services
SAMPLE_PROMPTS = [
    # Easy
    "Create an S3 bucket with versioning enabled and server-side encryption.",
    # Medium
    "Create a Route53 hosted zone named example.com with query logging to CloudWatch.",
    # Medium
    "Create an EC2 instance with a security group that allows HTTP and HTTPS traffic.",
    # Hard
    "Set up an IAM role for Lambda that can write to DynamoDB and publish to SNS.",
    # Hard
    "Create a VPC with two public and two private subnets across two availability zones, "
    "an internet gateway, and NAT gateway.",
    # Very Hard
    "Create an ECS Fargate service with an ALB, target group, CloudWatch log group, "
    "IAM task execution role, and auto-scaling policy.",
]


def print_results(prompt: str, results: list[dict], k: int) -> None:
    print(f"\n{'='*70}")
    print(f"PROMPT: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"{'='*70}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] resource={r['resource_type']}  section={r['section']}  "
              f"type={r['doc_type']}  score={r['score']:.4f}")
        print("-" * 50)
        snippet = r["content"][:400].replace("\n", " ")
        print(snippet + ("..." if len(r["content"]) > 400 else ""))


def run_builtin_tests(k: int) -> None:
    print(f"\nRunning {len(SAMPLE_PROMPTS)} sample prompts (k={k})...\n")
    for prompt in SAMPLE_PROMPTS:
        results = rag_retriever_rich(prompt, k=k)
        print_results(prompt, results, k)


def run_iac_eval_sample(n: int, k: int) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets package not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading IaC-Eval dataset...")
    ds = load_dataset("autoiac-project/iac-eval", split="test")
    sample = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    print(f"Testing {len(sample)} real IaC-Eval prompts (k={k})...\n")
    for row in sample:
        prompt = row["Prompt"]
        difficulty = row.get("Difficulty", "?")
        results = rag_retriever_rich(prompt, k=k)
        print(f"\n[difficulty={difficulty}]")
        print_results(prompt, results, k)


def run_interactive(k: int) -> None:
    print(f"\nTerraform RAG — interactive mode  (k={k})")
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

        results = rag_retriever_rich(prompt, k=current_k)
        print_results(prompt, results, current_k)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Terraform RAG retriever")
    parser.add_argument("--k", type=int, default=15, help="Top-k results per query (default 15)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode: enter prompts one by one")
    parser.add_argument("--iac-eval", type=int, metavar="N",
                        help="Sample N prompts from the real IaC-Eval dataset")
    parser.add_argument("--stats", action="store_true",
                        help="Print collection stats and exit")
    args = parser.parse_args()

    if args.stats:
        stats = collection_stats()
        print("\nCollection stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if args.interactive:
        run_interactive(args.k)
    elif args.iac_eval:
        run_iac_eval_sample(args.iac_eval, args.k)
    else:
        run_builtin_tests(args.k)


if __name__ == "__main__":
    main()
