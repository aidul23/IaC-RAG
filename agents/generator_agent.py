"""
Generator Agent — second stage of the IaC generation pipeline.

Responsibilities:
  1. Receive a PlannerOutput (structured resource plan)
  2. Build deterministic boilerplate (terraform{}, provider, variables) in Python
  3. Retrieve Terraform Argument Reference docs for each CREATE resource
  4. Ask the LLM to write only the resource{} blocks for CREATE resources
  5. Concatenate boilerplate + resource blocks → save to outputs/

Why this split?
  The LLM is unreliable when asked to write both boilerplate AND resource blocks
  in the same pass.  Common failures:
    - Emitting empty resource blocks for REFERENCE resources
    - Forgetting the terraform{} block
    - Inconsistent variable naming
  By generating boilerplate deterministically, we guarantee correctness for the
  scaffolding and let the LLM focus only on the part that requires domain
  knowledge: the resource arguments.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from agents.planner_agent import PlannerOutput, ResourcePlan
from agents.tools import RAGTool
from config import PLANNER_MAX_TOKENS, PLANNER_MODEL, PLANNER_TEMPERATURE

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
GENERATOR_MODEL = PLANNER_MODEL


# ── Output data structure ─────────────────────────────────────────────────────

@dataclass
class GeneratorOutput:
    """Output from the Generator Agent."""
    plan: PlannerOutput
    terraform_code: str
    resources_generated: list[str]
    output_file: str = ""


# ── LLM prompts (resource blocks only) ───────────────────────────────────────

_SYSTEM_PROMPT = """\
You are TerraformAI, an expert Terraform and AWS developer.
You will be given a list of AWS resources to write as Terraform HCL resource blocks.

Rules:
- Write ONLY resource "..." "..." { ... } blocks — nothing else
- Do NOT write terraform{}, provider{}, or variable{} blocks — those are already written
- Use exact resource type and name given (e.g. resource "aws_egress_only_internet_gateway" "main")
- Make sure the configuration is deployable — no undeclared references
- For any ID from a referenced (existing) resource, use var.<name>_id
  e.g. vpc_id = var.selected_id   subnet_id = var.subnet_id
- For tags always use: tags = merge(var.tags, {{ Name = "descriptive-name" }})
- Reference other CREATE resources by their HCL attribute path (e.g. aws_eip.nat.id)
- Supply inline default values for any resource argument that has a sensible default
- Write a short # comment above each resource block explaining its purpose
- Output ONLY valid HCL resource blocks — no markdown fences, no prose explanations"""


# One-shot example derived from the IaC-Eval few-shot template, adapted to our conventions.
# It shows: var.X_id for a referenced resource, merge(var.tags,...), cross-resource references.
_FEW_SHOT_EXAMPLE = """\
--- EXAMPLE (do not include in your output) ---
Request: Attach a NAT gateway to an existing public subnet.
Available reference variables: var.subnet_id (aws_subnet)

# Elastic IP for the NAT gateway
resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = merge(var.tags, { Name = "nat-eip" })
}

# NAT gateway placed in the referenced existing subnet
resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = var.subnet_id
  tags          = merge(var.tags, { Name = "nat-gateway" })
}
--- END EXAMPLE ---

"""


_RESOURCES_PROMPT = """\
{few_shot}Infrastructure request:
{prompt}

Resources to generate:
{resource_list}

Terraform Argument Reference documentation:
{rag_context}

Write HCL resource blocks for every resource listed above.
Do NOT include terraform{{}}, provider{{}}, or variable{{}} blocks — those are already written.
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class GeneratorAgent(BaseAgent):
    """
    Generates Terraform HCL from a PlannerOutput using a two-part strategy:

    Part A (Python)  — deterministic boilerplate:
        terraform{}  required_providers
        provider "aws" {}
        variable "region"
        variable "tags"
        variable "<name>_id"  for every REFERENCE resource

    Part B (LLM)  — resource blocks only:
        resource "aws_..." "..." { ... }  for every CREATE resource

    Final file = Part A + Part B
    """

    SYSTEM_PROMPT = _SYSTEM_PROMPT

    def __init__(
        self,
        model: str = GENERATOR_MODEL,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            tools=[RAGTool(k=5)],
            temperature=PLANNER_TEMPERATURE,
            max_tokens=PLANNER_MAX_TOKENS,
            verbose=verbose,
        )
        OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, plan: PlannerOutput) -> GeneratorOutput:
        """
        Step 1  Build boilerplate in Python (always correct, no LLM needed)
        Step 2  Fetch Argument Reference docs for CREATE resources
        Step 3  Ask LLM to write only the resource blocks
        Step 4  Combine boilerplate + resource blocks → save
        """
        boilerplate = self._build_boilerplate(plan)
        rag_context = self._fetch_docs(plan)
        resource_list = self._format_create_resources(plan)

        prompt_text = _RESOURCES_PROMPT.format(
            few_shot=_FEW_SHOT_EXAMPLE,
            prompt=plan.prompt,
            resource_list=resource_list,
            rag_context=rag_context,
        )
        raw = self._call_llm(prompt_text, timeout=180)
        resource_blocks = self._clean_hcl(raw)
        resource_blocks = self._remove_bad_blocks(resource_blocks, plan)

        tf_code = boilerplate + "\n\n" + resource_blocks

        output_path = self._save(tf_code, plan.prompt)

        return GeneratorOutput(
            plan=plan,
            terraform_code=tf_code,
            resources_generated=self._extract_resource_types(resource_blocks),
            output_file=str(output_path),
        )

    # ── Boilerplate builder (Python, not LLM) ─────────────────────────────────

    @staticmethod
    def _build_boilerplate(plan: PlannerOutput) -> str:
        """
        Build the terraform/provider/variable section programmatically.
        This is never delegated to the LLM — Python guarantees it is valid.
        """
        lines: list[str] = []

        # terraform{} block
        lines += [
            'terraform {',
            '  required_providers {',
            '    aws = {',
            '      source  = "hashicorp/aws"',
            '      version = "~> 5.0"',
            '    }',
            '  }',
            '}',
            '',
        ]

        # provider
        lines += [
            'provider "aws" {',
            '  region = var.region',
            '}',
            '',
        ]

        # region variable
        lines += [
            'variable "region" {',
            '  description = "AWS region to deploy resources"',
            '  type        = string',
            '  default     = "us-east-1"',
            '}',
            '',
        ]

        # tags variable — supports merge() pattern
        lines += [
            'variable "tags" {',
            '  description = "Additional tags applied to all resources"',
            '  type        = map(string)',
            '  default     = {}',
            '}',
            '',
        ]

        # One variable per REFERENCE resource
        seen: set[str] = set()
        for r in plan.resources:
            if r.mode != "reference":
                continue
            var_name = f"{r.resource_name}_id"
            if var_name in seen:
                continue
            seen.add(var_name)
            lines += [
                f'variable "{var_name}" {{',
                f'  description = "ID of the existing {r.resource_type}"',
                f'  type        = string',
                f'}}',
                '',
            ]

        return "\n".join(lines)

    # ── Doc fetching (CREATE resources only) ──────────────────────────────────

    def _fetch_docs(self, plan: PlannerOutput) -> str:
        """Retrieve Argument Reference docs for CREATE resources only."""
        from tools.rag_retriever import query_rag

        sections: list[str] = []
        seen: set[str] = set()

        for r in plan.resources:
            if r.mode == "reference":
                continue
            if r.resource_type in seen:
                continue
            seen.add(r.resource_type)

            hits = query_rag(
                plan.prompt,
                k=2,
                resource_types=[r.resource_type],
                sections=["Argument Reference"],
            )
            if hits:
                sections.append(f"### {r.resource_type}\n{hits[0]['content'][:800]}")

        return "\n\n".join(sections) if sections else "No documentation retrieved."

    # ── Resource list formatter (CREATE only) ─────────────────────────────────

    @staticmethod
    def _format_create_resources(plan: PlannerOutput) -> str:
        """
        Format only the CREATE resources for the LLM prompt.
        REFERENCE resources are not listed here — they are already handled
        as variables in the boilerplate.
        """
        lines: list[str] = []
        rt_to_resource = {r.resource_type: r for r in plan.resources}

        # Deployment order first
        ordered: list[ResourcePlan] = []
        for rt in plan.deployment_order:
            if rt in rt_to_resource and rt_to_resource[rt].mode == "create":
                ordered.append(rt_to_resource[rt])

        # Any CREATE resource not in deployment_order
        for r in plan.resources:
            if r.mode == "create" and r not in ordered:
                ordered.append(r)

        for i, r in enumerate(ordered, 1):
            lines.append(f'{i}. resource "{r.resource_type}" "{r.resource_name}"')
            lines.append(f"   Description : {r.description}")
            if r.depends_on:
                lines.append(f"   Depends on  : {', '.join(r.depends_on)}")
            if r.config_hints:
                lines.append(f"   Config hints: {'; '.join(r.config_hints)}")

        # Also remind the LLM which IDs are available as variables
        ref_vars = [
            f"var.{r.resource_name}_id  ({r.resource_type})"
            for r in plan.resources
            if r.mode == "reference"
        ]
        if ref_vars:
            lines.append("")
            lines.append("Available reference variables (use these for IDs):")
            for v in ref_vars:
                lines.append(f"  - {v}")

        return "\n".join(lines)

    # ── Post-processing ───────────────────────────────────────────────────────

    @staticmethod
    def _clean_hcl(raw: str) -> str:
        """Strip markdown code fences."""
        raw = re.sub(r"```(?:hcl|terraform)?\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
        return raw.strip()

    @staticmethod
    def _remove_bad_blocks(hcl: str, plan: PlannerOutput) -> str:
        """
        Safety net: remove any resource block the LLM generated for a REFERENCE
        resource.  A reference resource block has no key=value arguments inside
        it (or only comments), so it would be invalid Terraform anyway.
        """
        ref_types = {r.resource_type for r in plan.resources if r.mode == "reference"}
        if not ref_types:
            return hcl

        def _keep(match: re.Match) -> str:
            block = match.group(0)
            # Extract the resource type from the opening line
            type_match = re.search(r'resource\s+"(aws_[^"]+)"', block)
            if not type_match:
                return block
            rt = type_match.group(1)
            if rt not in ref_types:
                return block
            # It's a reference type — keep only if it has real argument assignments
            if re.search(r'^\s*\w+\s*=\s*\S', block, re.MULTILINE):
                return block
            return ""

        # Match resource blocks allowing one level of nested braces (enough for tags={})
        pattern = re.compile(
            r'resource\s+"aws_[^"]+"\s+"[^"]+"\s*\{(?:[^{}]|\{[^{}]*\})*\}',
            re.DOTALL,
        )
        return pattern.sub(_keep, hcl).strip()

    @staticmethod
    def _extract_resource_types(hcl: str) -> list[str]:
        """Parse resource types from HCL, preserving order, deduplicating."""
        matches = re.findall(r'resource\s+"(aws_[a-z_]+)"', hcl)
        seen: set[str] = set()
        return [m for m in matches if not (m in seen or seen.add(m))]  # type: ignore[func-returns-value]

    @staticmethod
    def _save(tf_code: str, prompt: str) -> Path:
        """Save to outputs/<slug>.tf, appending a counter if the file exists."""
        slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower())[:50].strip("_")
        path = OUTPUT_DIR / f"{slug}.tf"
        counter = 1
        while path.exists():
            path = OUTPUT_DIR / f"{slug}_{counter}.tf"
            counter += 1
        path.write_text(tf_code, encoding="utf-8")
        return path
