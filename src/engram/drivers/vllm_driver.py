"""VLLMDriver — high-throughput multi-LoRA serving backend (FR-D2, elastic serving).

vLLM serves a base model with many hot-swappable LoRA adapters under continuous
batching — the serving half of an elastic deployment. It is inference-only:
training/consolidation stays on a PEFTDriver node; vLLM nodes load the resulting
adapters from the shared Store and serve them. Nodes are stateless (state lives in
S3/DynamoDB), so they scale horizontally behind a load balancer.

Requires Linux + CUDA + vLLM:  pip install "engram[vllm]"
NOTE: untested on Windows (vLLM is Linux-only); written against vLLM's documented API.
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence

from engram.drivers.base import ArchInfo, BaseLLMDriver, DriverCapabilities, GenRequest


class VLLMDriver(BaseLLMDriver):
    def __init__(self, model_id: str, max_lora_rank: int = 16,
                 gpu_memory_utilization: float = 0.9,
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2", **kw):
        try:
            from vllm import LLM
        except ImportError as e:  # pragma: no cover
            raise ImportError('VLLMDriver needs the vllm extra: pip install "engram[vllm]"') from e
        self.model_id = model_id
        self.llm = LLM(model=model_id, enable_lora=True, max_lora_rank=max_lora_rank,
                       gpu_memory_utilization=gpu_memory_utilization, **kw)
        self.tok = self.llm.get_tokenizer()
        self._loras: dict[str, tuple[int, str]] = {}     # lora_id -> (int_id, path)
        self._next_id = 1
        self._embed_model_name = embed_model
        self._embedder = None

    # ---- introspection -----------------------------------------------------------
    def capabilities(self) -> DriverCapabilities:
        # inference-only: serves + hot-swaps LoRAs, but does not train
        return DriverCapabilities(lora=True, train_lora=False, white_box=False,
                                  tool_use=True, streaming=False)

    def arch_info(self) -> ArchInfo:
        return ArchInfo(induction_capable=True,
                        notes="vLLM backend; assumes an induction-capable instruct base")

    # ---- adapters (serve-only; trained elsewhere) --------------------------------
    def load_lora(self, adapter_path: str, lora_id: str) -> None:
        if lora_id not in self._loras:
            self._loras[lora_id] = (self._next_id, adapter_path)
            self._next_id += 1

    def _prompt(self, messages, tools=()):
        kw = dict(add_generation_prompt=True, tokenize=False)
        if tools:
            kw["tools"] = list(tools)
        for extra in (dict(enable_thinking=False), dict()):
            try:
                return self.tok.apply_chat_template(messages, **kw, **extra)
            except TypeError:
                continue

    # ---- generation (continuous-batched; non-streaming in offline mode) ----------
    def generate(self, req: GenRequest) -> Iterator[str]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        prompt = self._prompt(req.messages, tools=req.tools)
        sp = SamplingParams(max_tokens=req.max_new_tokens,
                            temperature=req.temperature if req.temperature > 0 else 0.0)
        lora_req = None
        if req.lora_ids and req.lora_ids[0] in self._loras:
            iid, path = self._loras[req.lora_ids[0]]
            lora_req = LoRARequest(req.lora_ids[0], iid, path)
        out = self.llm.generate([prompt], sp, lora_request=lora_req)
        yield out[0].outputs[0].text

    # ---- training: not supported here (use a PEFTDriver node) --------------------
    def train_lora(self, examples: Iterable[dict[str, Any]], config: dict[str, Any]) -> str:
        raise NotImplementedError("vLLM is inference-only; train adapters on a PEFTDriver node")

    # ---- embeddings (routing / RAG keys) ----------------------------------------
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embed_model_name)
        return self._embedder.encode(list(texts), normalize_embeddings=True,
                                     show_progress_bar=False).tolist()
