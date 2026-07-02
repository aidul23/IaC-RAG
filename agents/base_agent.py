"""
Base agent class for all IaC pipeline agents.

Provides:
  - Configurable LLM backend (Ollama by default, easy to swap)
  - Tool registry — agents declare which tools they have access to
  - Single _call_llm() method all subclasses use
  - Structured logging of prompts and responses

To add a new agent: subclass BaseAgent, set SYSTEM_PROMPT, register tools,
and implement run(prompt) -> your output type.
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.tools import BaseTool
from config import OLLAMA_BASE_URL


class BaseAgent(ABC):
    """
    Abstract base for all agents in the IaC generation pipeline.

    Subclasses must define:
        SYSTEM_PROMPT  – str, sets the agent's role and behaviour
        run(prompt)    – executes the agent and returns a typed result
    """

    SYSTEM_PROMPT: str = "You are a helpful assistant."

    def __init__(
        self,
        model: str,
        tools: list[BaseTool] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        ollama_url: str = OLLAMA_BASE_URL,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.tools: dict[str, BaseTool] = {t.name: t for t in (tools or [])}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_url = ollama_url.rstrip("/")
        self.verbose = verbose

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, system: str | None = None, timeout: int = 120) -> str:
        """
        Send a prompt to the configured Ollama model and return the response text.
        Uses the system prompt defined on the class unless overridden.
        """
        system_prompt = system or self.SYSTEM_PROMPT
        full_prompt = f"{system_prompt}\n\n{prompt}"

        if self.verbose:
            print(f"\n[{self.__class__.__name__}] → {self.model}")
            print(f"  prompt ({len(full_prompt)} chars): {full_prompt[:200]}...")

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=timeout,
            )
            response.raise_for_status()
            text = response.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.ollama_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except requests.HTTPError as exc:
            raise RuntimeError(f"Ollama API error: {exc}") from exc

        if self.verbose:
            print(f"  response ({len(text)} chars): {text[:200]}...")

        return text

    # ── Tool helpers ──────────────────────────────────────────────────────────

    def use_tool(self, tool_name: str, query: str) -> str:
        """Execute a registered tool by name."""
        if tool_name not in self.tools:
            raise ValueError(
                f"Tool '{tool_name}' not registered. "
                f"Available: {list(self.tools.keys())}"
            )
        if self.verbose:
            print(f"\n[{self.__class__.__name__}] calling tool: {tool_name}")
        return self.tools[tool_name].run(query)

    def list_tools(self) -> str:
        """Return a formatted description of all registered tools (for LLM context)."""
        if not self.tools:
            return "No tools available."
        lines = []
        for tool in self.tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    # ── JSON parsing helper ───────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> dict | list | None:
        """
        Extract and parse the first JSON object or array from LLM output.
        Handles common formatting issues (markdown code fences, extra text).
        """
        import re

        # Strip markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", text).strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find a JSON object or array
        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        return None

    # ── Subclass interface ────────────────────────────────────────────────────

    @abstractmethod
    def run(self, prompt: str) -> Any:
        """Execute the agent on the given prompt and return a typed result."""
