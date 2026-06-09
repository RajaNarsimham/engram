"""Engram — a continual-learning harness for open-weight LLMs.

Bring your own base model; Engram adds a routed skill-library (LoRA adapters),
grounded retrieval (RAG), and an offline consolidation loop, behind a normal
OpenAI/Anthropic-compatible chat interface.

Public surface (Tier-0 core; see docs/REQUIREMENTS.md):
    BaseLLMDriver   - the interface every base-model backend implements
    Capability, Registry - pluggable skills / knowledge / tools
    Retriever       - RAG retrieval (expandable, relevance-gated)
    Orchestrator    - per-request runtime (gate -> retrieve -> route -> generate)
"""
__version__ = "0.0.1"

from engram.drivers.base import BaseLLMDriver, DriverCapabilities, ArchInfo
from engram.registry.registry import Capability, CapabilityKind, Registry

__all__ = [
    "__version__",
    "BaseLLMDriver",
    "DriverCapabilities",
    "ArchInfo",
    "Capability",
    "CapabilityKind",
    "Registry",
]
