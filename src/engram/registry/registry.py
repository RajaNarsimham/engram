"""Capability Registry (FR-R1..R3): pluggable skills / knowledge / tools.

Everything you add to Engram is a registered Capability with metadata + a routing
key. Add/remove a capability with NO code change — that's the plug-and-play half.
In agentic mode each capability is also exposed to the model as a TOOL (FR-O10).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable


class CapabilityKind(enum.Enum):
    SKILL = "skill"          # a LoRA adapter
    KNOWLEDGE = "knowledge"  # a RAG source / index
    TOOL = "tool"            # a callable function


@dataclass
class Capability:
    name: str
    kind: CapabilityKind
    description: str                      # what it does / when to use it (the model routes on this)
    handle: Any = None                    # adapter id | retriever | callable
    routing_key: list[float] | None = None   # frozen-feature key (FR-O3); NOT a trained classifier
    when_to_use: str = ""
    version: str = "0.1.0"
    eval_passed: bool = False             # must pass the eval-gate before serving (FR-E1)
    project: str = "default"              # tenant/project scope (FR-R4)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_tool_spec(self) -> dict[str, Any]:
        """OpenAI-style tool spec for agentic routing (the model picks by description)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"{self.description}\nWhen to use: {self.when_to_use}".strip(),
                "parameters": self.metadata.get("parameters", {"type": "object", "properties": {}}),
            },
        }


class Registry:
    """In-memory capability registry. (Persistence/multi-tenant store = Tier-1.)"""

    def __init__(self) -> None:
        self._caps: dict[tuple[str, str], Capability] = {}   # (project, name) -> Capability

    def register(self, cap: Capability) -> None:
        self._caps[(cap.project, cap.name)] = cap

    def unregister(self, name: str, project: str = "default") -> None:
        self._caps.pop((project, name), None)

    def get(self, name: str, project: str = "default") -> Capability | None:
        return self._caps.get((project, name))

    def list(self, *, project: str = "default", kind: CapabilityKind | None = None,
             live_only: bool = False) -> list[Capability]:
        out = [c for (p, _), c in self._caps.items() if p == project]
        if kind is not None:
            out = [c for c in out if c.kind == kind]
        if live_only:
            out = [c for c in out if c.eval_passed]
        return out

    def tool_specs(self, project: str = "default") -> list[dict[str, Any]]:
        """Tool specs for every LIVE capability — handed to the model in agentic mode."""
        return [c.as_tool_spec() for c in self.list(project=project, live_only=True)]
