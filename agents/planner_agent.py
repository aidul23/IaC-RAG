"""
Planner Agent — first stage of the IaC generation pipeline.

Responsibilities:
  1. Receive a natural-language infrastructure prompt
  2. Decide whether to call the RAG tool for Terraform documentation
  3. Generate a structured plan: resources, dependencies, deployment order
  4. Return a PlannerOutput that the Generator Agent will use to write Terraform code

Flow:
  Prompt
    ├─► Step 1: Analyse — LLM decides if RAG is needed (almost always yes)
    ├─► Step 2: Retrieve — call terraform_docs_search tool with the prompt
    └─► Step 3: Plan — LLM reads docs + prompt → structured JSON plan
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from agents.tools import RAGTool
from config import PLANNER_MAX_TOKENS, PLANNER_MODEL, PLANNER_TEMPERATURE


# ── Output data structures ────────────────────────────────────────────────────

@dataclass
class ResourcePlan:
    """A single Terraform resource in the plan."""
    resource_type: str          # e.g. "aws_vpc"
    resource_name: str          # e.g. "main"
    description: str            # what this resource does in context of the prompt
    depends_on: list[str]       # list of resource_type values this depends on
    config_hints: list[str]     # key arguments / values inferred from prompt + docs
    mode: str = "create"        # "create" = new resource block | "reference" = existing, use variable


@dataclass
class PlannerOutput:
    """Structured output from the Planner Agent passed to the Generator Agent."""
    prompt: str                             # original user prompt
    summary: str                            # one-sentence description of the plan
    resources: list[ResourcePlan]           # ordered list of resources to create
    deployment_order: list[str]             # resource_types in creation order
    notes: list[str]                        # caveats, assumptions, provider hints
    rag_docs_used: bool = False             # whether RAG was called
    reasoning: str = ""                     # CoT reasoning the LLM produced
    raw_llm_output: str = ""               # full LLM response for debugging


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert Terraform and AWS infrastructure architect.
Your job is to create a detailed implementation plan from a natural-language
infrastructure request and relevant Terraform documentation.

Rules:
- Only include resources that are explicitly requested or strictly required
  (e.g. aws_subnet is required before aws_db_subnet_group)
- Use exact Terraform resource type names (aws_vpc, aws_s3_bucket, etc.)
- Choose concise, descriptive Terraform resource names (e.g. "main", "primary")
- List dependencies accurately — a resource depends_on what must exist first
- Extract config_hints from the prompt (e.g. "cidr_block: 10.0.0.0/16")
- deployment_order must be a valid topological sort (dependencies first)

CRITICAL — set the correct "mode" for every resource:
  "create"    : the prompt explicitly asks to CREATE this resource
                Example: "Create a VPC", "provision a bucket", "set up a gateway"
  "reference" : the resource already exists and is only REFERENCED by name/context
                Example: "associated with a specified VPC", "in the existing subnet",
                         "for the specified VPC", "attach to the given VPC"
                → referenced resources become Terraform input variables, NOT resource blocks

When in doubt between create and reference for foundational resources (aws_vpc,
aws_subnet, aws_security_group), choose "reference" unless the prompt explicitly
says to create one."""


_PLAN_PROMPT_TEMPLATE = """\
Infrastructure request:
{prompt}

Terraform documentation retrieved for this request:
{rag_context}

Before writing the JSON, reason through the request in four steps:

Step 1 — Resources: What resources are explicitly requested?
         What implicit dependencies are STRICTLY required (not nice-to-have)?
Step 2 — Create vs Reference: For each resource, decide its mode.
         REFERENCE clues: "specified", "existing", "given", "associated with a [noun]",
                          "for the [noun]", "attach to the [noun]"
         CREATE clues: "create", "set up", "provision", "deploy", "build", "launch"
         When in doubt for foundational resources (aws_vpc, aws_subnet), choose "reference".
Step 3 — Attributes: What key arguments does each resource need from the prompt/docs?
Step 4 — Connections: How do resources reference each other's IDs?

Now produce your JSON plan. Put your Step 1-4 reasoning in "reasoning" FIRST,
then fill the remaining fields. Use EXACTLY this structure (no extra fields, pure JSON):

{{
  "reasoning": "Step 1: prompt requests an egress-only gateway. Step 2: 'a specified VPC' means the VPC already exists → reference. The gateway itself is CREATE. Step 3: gateway needs vpc_id. Step 4: gateway uses var.selected_id (the VPC variable).",
  "summary": "one sentence describing what will be built",
  "resources": [
    {{
      "resource_type": "aws_vpc",
      "resource_name": "selected",
      "description": "Pre-existing VPC referenced by the request — not created here",
      "depends_on": [],
      "config_hints": ["vpc_id: var.selected_id"],
      "mode": "reference"
    }},
    {{
      "resource_type": "aws_egress_only_internet_gateway",
      "resource_name": "main",
      "description": "Egress-only internet gateway attached to the specified VPC",
      "depends_on": [],
      "config_hints": ["vpc_id: var.selected_id"],
      "mode": "create"
    }}
  ],
  "deployment_order": ["aws_egress_only_internet_gateway"],
  "notes": ["aws_vpc is a referenced existing resource; its ID is passed as var.selected_id"]
}}

mode rules:
- "create"    → full resource block generated
- "reference" → only a variable block generated (no resource block)
- deployment_order → list ONLY "create" resources in topological order

Output ONLY valid JSON — no markdown fences, no text outside the JSON object."""


# ── Agent ─────────────────────────────────────────────────────────────────────

class PlannerAgent(BaseAgent):
    """
    Plans the Terraform resources needed to fulfil an infrastructure prompt.

    Registered tools:
      - terraform_docs_search  (RAGTool, V2 two-pass retrieval)

    Usage:
        agent = PlannerAgent()
        output = agent.run("Create an S3 bucket with versioning...")
        print(output.resources)
    """

    SYSTEM_PROMPT = _SYSTEM_PROMPT

    def __init__(
        self,
        model: str = PLANNER_MODEL,
        rag_k: int = 10,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            tools=[RAGTool(k=rag_k)],
            temperature=PLANNER_TEMPERATURE,
            max_tokens=PLANNER_MAX_TOKENS,
            verbose=verbose,
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, prompt: str) -> PlannerOutput:
        """
        Execute the three-step planning pipeline.

        Step 1  Decide — check if the prompt needs documentation lookup
        Step 2  Retrieve — call RAG tool with the prompt
        Step 3  Plan — generate a structured JSON plan using docs + prompt
        """
        # Step 1: Decide whether to call RAG
        needs_rag = self._should_use_rag(prompt)

        # Step 2: Retrieve docs if needed (almost always true for infra prompts)
        rag_context = ""
        if needs_rag:
            if self.verbose:
                print(f"\n[PlannerAgent] calling terraform_docs_search...")
            rag_context = self.use_tool("terraform_docs_search", prompt)

        # Step 3: Generate the plan
        raw_output = self._generate_plan(prompt, rag_context)

        # Parse and return
        return self._parse_output(prompt, raw_output, rag_used=needs_rag)

    # ── Internal steps ────────────────────────────────────────────────────────

    def _should_use_rag(self, prompt: str) -> bool:
        """
        Ask the LLM whether the prompt requires Terraform documentation lookup.
        For infrastructure prompts this is almost always true; the check prevents
        unnecessary RAG calls for non-infrastructure inputs.
        """
        decision_prompt = (
            f"Does the following request need Terraform AWS documentation to plan correctly?\n\n"
            f"Request: {prompt}\n\n"
            f"Available tools:\n{self.list_tools()}\n\n"
            f"Answer with exactly one word: YES or NO."
        )
        response = self._call_llm(decision_prompt, system="You are a helpful assistant. Answer concisely.")
        return "no" not in response.strip().lower()

    def _generate_plan(self, prompt: str, rag_context: str) -> str:
        """Call the LLM with the prompt + RAG context and return raw JSON output."""
        if not rag_context:
            rag_context = "No documentation retrieved. Use your general Terraform knowledge."

        plan_prompt = _PLAN_PROMPT_TEMPLATE.format(
            prompt=prompt,
            rag_context=rag_context,
        )
        return self._call_llm(plan_prompt)

    def _parse_output(
        self,
        prompt: str,
        raw_output: str,
        rag_used: bool,
    ) -> PlannerOutput:
        """
        Parse the LLM JSON output into a PlannerOutput dataclass.
        Falls back to a minimal output if JSON parsing fails.
        """
        data = self._extract_json(raw_output)

        if not data or not isinstance(data, dict):
            # Parsing failed — return what we have with a warning
            return PlannerOutput(
                prompt=prompt,
                summary="(JSON parsing failed — see raw_llm_output)",
                resources=[],
                deployment_order=[],
                notes=["Warning: LLM did not return valid JSON. Try a larger model."],
                rag_docs_used=rag_used,
                raw_llm_output=raw_output,
            )

        resources: list[ResourcePlan] = []
        for r in data.get("resources", []):
            if not isinstance(r, dict) or "resource_type" not in r:
                continue
            raw_mode = r.get("mode", "create")
            mode = raw_mode if raw_mode in ("create", "reference") else "create"
            resources.append(ResourcePlan(
                resource_type=r.get("resource_type", ""),
                resource_name=r.get("resource_name", "main"),
                description=r.get("description", ""),
                depends_on=r.get("depends_on", []),
                config_hints=r.get("config_hints", []),
                mode=mode,
            ))

        return PlannerOutput(
            prompt=prompt,
            summary=data.get("summary", ""),
            resources=resources,
            deployment_order=data.get("deployment_order", [r.resource_type for r in resources]),
            notes=data.get("notes", []),
            rag_docs_used=rag_used,
            reasoning=data.get("reasoning", ""),
            raw_llm_output=raw_output,
        )
