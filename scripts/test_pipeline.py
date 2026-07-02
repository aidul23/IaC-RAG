"""
Full pipeline: Planner Agent → Generator Agent → Terraform .tf file

Usage:
    python scripts/test_pipeline.py --interactive          # type your own prompt
    python scripts/test_pipeline.py                        # 3 built-in prompts
    python scripts/test_pipeline.py --iac-eval 3           # real IaC-Eval prompts
    python scripts/test_pipeline.py --model llama3.1:8b    # use a better model
    python scripts/test_pipeline.py --verbose              # show LLM calls
    python scripts/test_pipeline.py --no-save              # skip saving .tf file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.generator_agent import GeneratorAgent, GeneratorOutput
from agents.planner_agent import PlannerAgent, PlannerOutput

SAMPLE_PROMPTS = [
    "Create an S3 bucket with versioning enabled and server-side encryption.",
    "Create a VPC with two public subnets, an internet gateway, and a route table.",
    "Creates an egress-only internet gateway associated with a specified VPC, "
    "allowing IPv6-enabled instances to connect to the internet without allowing "
    "inbound internet traffic.",
]


# ── Display helpers ───────────────────────────────────────────────────────────

def print_plan(plan: PlannerOutput) -> None:
    w = 70
    print(f"\n{'─' * w}")
    print(f"  STEP 1 — PLANNER")
    print(f"{'─' * w}")
    print(f"  Summary : {plan.summary}")
    print(f"  RAG used: {'yes' if plan.rag_docs_used else 'no'}")
    print(f"\n  Resources ({len(plan.resources)}):")
    for r in plan.resources:
        tag = "[REF]" if r.mode == "reference" else "[NEW]"
        deps = f"  depends on: {', '.join(r.depends_on)}" if r.depends_on else ""
        print(f"    {tag} {r.resource_type} \"{r.resource_name}\"{deps}")
    print(f"\n  Deployment order: {' → '.join(plan.deployment_order)}")


def print_generated(result: GeneratorOutput, save: bool) -> None:
    w = 70
    print(f"\n{'─' * w}")
    print(f"  STEP 2 — GENERATOR")
    print(f"{'─' * w}")
    print(f"  Resources generated: {result.resources_generated}")
    if save and result.output_file:
        print(f"  Saved to: {result.output_file}")
    print(f"\n{'─' * w}")
    print(f"  GENERATED TERRAFORM CODE")
    print(f"{'─' * w}\n")
    print(result.terraform_code)
    print(f"\n{'─' * w}")


def run_pipeline(
    prompt: str,
    planner: PlannerAgent,
    generator: GeneratorAgent,
    save: bool,
) -> None:
    w = 70
    print(f"\n{'=' * w}")
    print(f"  PROMPT: {prompt[:w - 10]}{'...' if len(prompt) > w - 10 else ''}")
    print(f"{'=' * w}")

    # Step 1 — Plan
    print("\n  Running Planner Agent...")
    plan = planner.run(prompt)
    print_plan(plan)

    # Step 2 — Generate
    print("\n  Running Generator Agent...")
    result = generator.run(plan)

    if not save and result.output_file:
        # Delete the file if --no-save was passed
        Path(result.output_file).unlink(missing_ok=True)
        result.output_file = ""

    print_generated(result, save=save)


# ── Run modes ─────────────────────────────────────────────────────────────────

def run_builtin(planner: PlannerAgent, generator: GeneratorAgent, save: bool) -> None:
    print(f"\nRunning {len(SAMPLE_PROMPTS)} built-in prompts through the pipeline...\n")
    for prompt in SAMPLE_PROMPTS:
        run_pipeline(prompt, planner, generator, save)


def run_iac_eval(
    planner: PlannerAgent, generator: GeneratorAgent, n: int, save: bool
) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets package not installed. Run: pip install datasets")
        sys.exit(1)

    print("Loading IaC-Eval dataset...")
    ds = load_dataset("autoiac-project/iac-eval", split="test")
    sample = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    print(f"Running pipeline on {len(sample)} IaC-Eval prompts...\n")
    for row in sample:
        difficulty = row.get("Difficulty", "?")
        print(f"[difficulty={difficulty}]")
        run_pipeline(row["Prompt"], planner, generator, save)


def run_interactive(
    planner: PlannerAgent, generator: GeneratorAgent, save: bool
) -> None:
    print(f"\nFull pipeline — interactive mode")
    print(f"Planner model  : {planner.model}")
    print(f"Generator model: {generator.model}")
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

        run_pipeline(prompt, planner, generator, save)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full IaC pipeline: Planner → Generator → .tf file"
    )
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--iac-eval", type=int, metavar="N",
                        help="Run N prompts from the IaC-Eval dataset")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama model for both agents (e.g. llama3.1:8b)")
    parser.add_argument("--planner-model", type=str, default=None,
                        help="Ollama model for the Planner Agent only")
    parser.add_argument("--generator-model", type=str, default=None,
                        help="Ollama model for the Generator Agent only")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save the generated .tf file to disk")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print LLM prompts and raw responses")
    args = parser.parse_args()

    # Resolve models — specific flags override the global --model flag
    planner_model  = args.planner_model  or args.model
    generator_model = args.generator_model or args.model

    planner_kwargs  = dict(verbose=args.verbose)
    generator_kwargs = dict(verbose=args.verbose)
    if planner_model:
        planner_kwargs["model"]  = planner_model
    if generator_model:
        generator_kwargs["model"] = generator_model

    planner   = PlannerAgent(**planner_kwargs)
    generator = GeneratorAgent(**generator_kwargs)

    print(f"Pipeline ready")
    print(f"  Planner model  : {planner.model}")
    print(f"  Generator model: {generator.model}")
    print(f"  Save output    : {'no' if args.no_save else 'yes → outputs/'}")

    save = not args.no_save

    if args.interactive:
        run_interactive(planner, generator, save)
    elif args.iac_eval:
        run_iac_eval(planner, generator, args.iac_eval, save)
    else:
        run_builtin(planner, generator, save)


if __name__ == "__main__":
    main()
