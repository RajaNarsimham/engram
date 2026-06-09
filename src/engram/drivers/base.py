"""The Base-LLM Driver interface — the spine of Engram (FR-D1..D4).

Every base-model backend (PEFT, vLLM, Ollama, ...) implements this one interface.
The orchestrator talks ONLY to this — which is exactly what makes base models
plug-and-play. Capabilities are declared so the orchestrator can degrade
gracefully (e.g. an inference-only backend can't `train_lora`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence


@dataclass
class DriverCapabilities:
    """What a backend supports (FR-D3). Orchestrator degrades gracefully on these."""
    lora: bool = False            # can attach/serve LoRA adapters
    train_lora: bool = False      # can train adapters locally
    white_box: bool = False       # exposes logits/hidden states (for white-box distill)
    tool_use: bool = False        # base supports function/tool calling (needed for agentic mode)
    streaming: bool = True


@dataclass
class ArchInfo:
    """Architecture facts the platform needs (FR-D4).

    `induction_capable` is load-bearing: a base that cannot USE retrieved context
    (no induction-head behavior) makes RAG useless. Engram checks/flags this.
    """
    induction_capable: bool | None = None     # can it copy/attend to in-context info?
    layer_types: dict[str, int] = field(default_factory=dict)  # e.g. {"attention": 8, "linear_attn": 24}
    hidden_size: int | None = None
    num_layers: int | None = None
    notes: str = ""


@dataclass
class GenRequest:
    """A normalized generation request the orchestrator hands to a driver."""
    messages: list[dict[str, str]]            # chat messages (role/content)
    lora_ids: Sequence[str] = ()              # active skill adapter(s), if any
    tools: Sequence[dict[str, Any]] = ()      # tool specs (agentic mode)
    max_new_tokens: int = 512
    temperature: float = 0.0
    stream: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


class BaseLLMDriver(ABC):
    """Abstract base every backend implements. Keep this interface tiny and stable."""

    # ---- identity / introspection -------------------------------------------------
    @abstractmethod
    def capabilities(self) -> DriverCapabilities:
        """Declare what this backend supports (FR-D3)."""

    @abstractmethod
    def arch_info(self) -> ArchInfo:
        """Expose architecture facts incl. induction-capability (FR-D4)."""

    # ---- generation ---------------------------------------------------------------
    @abstractmethod
    def generate(self, req: GenRequest) -> Iterator[str]:
        """Stream generated text. The ONLY method the orchestrator must have."""

    # ---- adapters (skills) --------------------------------------------------------
    def load_lora(self, adapter_path: str, lora_id: str) -> None:
        """Register an adapter for serving. Override if `capabilities().lora`."""
        raise NotImplementedError("this backend does not support LoRA")

    def train_lora(self, examples: Iterable[dict[str, Any]], config: dict[str, Any]) -> str:
        """Train an adapter; return its path/id. Override if `capabilities().train_lora`.

        Consolidation requires this to be *context-preserving* (mix RAG-format
        examples) so a memorize-only adapter doesn't break retrieval use — a
        validated requirement (FR-C2).
        """
        raise NotImplementedError("this backend does not support adapter training")

    # ---- embeddings (routing / RAG keys) ------------------------------------------
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text (for routing keys / retrieval). May use a separate embedder."""
        raise NotImplementedError("this backend does not provide embeddings")
