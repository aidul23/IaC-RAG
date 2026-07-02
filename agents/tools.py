"""
Tool definitions for IaC agents.

Each tool has a name, description, and a run() method.
The description is shown to the LLM so it can decide when to use the tool.
New tools can be added here and registered in any agent's tool list.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class BaseTool(ABC):
    """Base class for all agent tools."""

    name: str
    description: str

    @abstractmethod
    def run(self, query: str) -> str:
        """Execute the tool and return a plain-text result."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class RAGTool(BaseTool):
    """
    Searches the Terraform AWS provider documentation vector store.

    Returns relevant Argument Reference sections for the resource types
    identified in the query. Uses the two-pass RAG-grounded strategy (V2)
    for higher accuracy.
    """

    name = "terraform_docs_search"
    description = (
        "Search the Terraform AWS provider documentation. "
        "Use this tool when you need to know the exact arguments, "
        "required fields, or configuration options for any aws_* resource. "
        "Input: a natural-language infrastructure description or a resource name."
    )

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def run(self, query: str) -> str:
        from tools.rag_retriever_v2 import rag_retriever_v2

        results, debug = rag_retriever_v2(query, k=self.k)

        if not results:
            return "No relevant documentation found."

        # Keep only the resources LLM identified as necessary (no semantic noise)
        necessary_types = set(debug["pass2_types"]) or {r["resource_type"] for r in results}
        filtered = [r for r in results if r["resource_type"] in necessary_types]
        if not filtered:
            filtered = results[:self.k]

        lines = [f"Found documentation for {len(filtered)} resource(s):\n"]
        seen = set()
        for r in filtered:
            rt = r["resource_type"]
            if rt in seen:
                continue
            seen.add(rt)
            snippet = r["content"][:600].replace("\n", " ")
            lines.append(f"### {rt}\n{snippet}\n")

        return "\n".join(lines)
