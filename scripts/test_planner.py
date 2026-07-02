"""
Test the Planner Agent interactively or against IaC-Eval prompts.

Usage:
    python scripts/test_planner.py                  # 4 built-in prompts
    python scripts/test_planner.py --interactive    # type your own prompt
    python scripts/test_planner.py --iac-eval 5     # real IaC-Eval prompts
    python scripts/test_planner.py --verbose        # show LLM prompts/responses
    python scripts/test_planner.py --model llama3.1:8b  # use a different model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.planner_agent import PlannerAgent, PlannerOutput

SAMPLE_PROMPTS = [
    "Create an S3 bucket with versioning enabled and server-side encryption.",
    "Set up a VPC with two public subnets, an internet gateway, and a route table.",
    "Set up an IAM role for Lambda that can write to DynamoDB and publish to SNS.",
    "Create an ECS Fargate service with an ALB, CloudWatch log group, and IAM task execution role.",
]


def print_plan(output: PlannerOutput) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"PROMPT : {output.prompt[:width - 9]}{'...' if len(output.prompt) > width - 9 else ''}")
    print(f"{'=' * width}")
    print(f"SUMMARY: {output.summary}")
    print(f"RAG    : {'yes' if output.rag_docs_used else 'no'}")
    if output.reasoning:
        print(f"\nREASONING:")
        print(f"  {output.reasoning}")

    print(f"\nRESOURCES ({len(output.resources)}):")
    for r in output.resources:
        mode_tag = "[REFERENCE]" if r.mode == "reference" else "[CREATE]   "
        print(f"\n  {mode_tag} {r.resource_type}  name: {r.resource_name}")
        print(f"    Description : {r.description}")
        if r.mode == "reference":
            print(f"    → variable \"{r.resource_name}_id\" will be emitted (no resource block)")
        else:
            if r.depends_on:
                print(f"    Depends on  : {', '.join(r.depends_on)}")
            if r.config_hints:
                print(f"    Config hints:")
                for hint in r.config_hints:
                    print(f"      - {hint}")

    if output.deployment_order:
        print(f"\nDEPLOYMENT ORDER:")
        for i, rt in enumerate(output.deployment_order, 1):
            print(f"  {i:02d}. {rt}")

    if output.notes:
        print(f"\nNOTES:")
        for note in output.notes:
            print(f"  • {note}")

    if not output.resources:
        print("\n[!] No resources parsed. Raw LLM output:")
        print(output.raw_llm_output[:800])

    print()


def run_builtin(agent: PlannerAgent) -> None:
    print(f"\nRunning {len(SAMPLE_PROMPTS)} built-in prompts...\n")
    for prompt in SAMPLE_PROMPTS:
        output = agent.run(prompt)
        print_plan(output)


def run_iac_eval(agent: PlannerAgent, n: int) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets package not installed. Run: pip install datasets")
        sys.exit(1)

    print("Loading IaC-Eval dataset...")
    ds = load_dataset("autoiac-project/iac-eval", split="test")
    sample = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    print(f"Running Planner Agent on {len(sample)} IaC-Eval prompts...\n")
    for row in sample:
        prompt = row["Prompt"]
        difficulty = row.get("Difficulty", "?")
        print(f"[difficulty={difficulty}]")
        output = agent.run(prompt)
        print_plan(output)


def run_interactive(agent: PlannerAgent) -> None:
    print(f"\nPlanner Agent — interactive mode")
    print(f"Model: {agent.model}")
    print("Type your infrastructure prompt and press Enter.")
    print("Type :quit to exit.\n")

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

        output = agent.run(prompt)
        print_plan(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the IaC Planner Agent")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--iac-eval", type=int, metavar="N",
                        help="Run N prompts from the IaC-Eval dataset")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama model to use (overrides config)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print LLM prompts and raw responses")
    parser.add_argument("--rag-k", type=int, default=10,
                        help="Number of RAG results to retrieve (default 10)")
    args = parser.parse_args()

    kwargs = dict(rag_k=args.rag_k, verbose=args.verbose)
    if args.model:
        kwargs["model"] = args.model

    agent = PlannerAgent(**kwargs)
    print(f"Planner Agent ready  |  model: {agent.model}  |  RAG k: {args.rag_k}")

    if args.interactive:
        run_interactive(agent)
    elif args.iac_eval:
        run_iac_eval(agent, args.iac_eval)
    else:
        run_builtin(agent)


if __name__ == "__main__":
    main()
