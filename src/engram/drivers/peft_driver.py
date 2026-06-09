"""PEFTDriver — local transformers + PEFT backend (the Tier-0 reference driver).

Wraps an open-weight HF model (Qwen, Llama, Mistral, ...) and implements the
BaseLLMDriver spine. Bakes in the rules our experiments proved:
  - arch_info() detects induction-capability (presence of softmax-attention layers)
  - generate() disables 'thinking' mode (or answers truncate) and streams
  - LoRA targets all `language_model` Linears (works for hybrid linear-attn models too)
  - train_lora() masks the prompt, trains on completions; consolidation supplies the
    context-preserving example mix (FR-C2) so adapters don't break retrieval-use

Requires:  pip install "engram[peft]"   (and "engram[rag]" for embeddings)
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Iterable, Iterator, Sequence

import torch

from engram.drivers.base import ArchInfo, BaseLLMDriver, DriverCapabilities, GenRequest


class PEFTDriver(BaseLLMDriver):
    def __init__(self, model_id: str, device: str = "cuda:0", dtype=torch.bfloat16,
                 adapter_dir: str = "engram_store/adapters",
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from transformers import AutoTokenizer
        self.model_id = model_id
        self.device = device
        self.adapter_dir = adapter_dir
        os.makedirs(adapter_dir, exist_ok=True)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = self._load(model_id, dtype, device)
        self._stop_ids = self._stop_token_ids()
        self._pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self._is_peft = False
        self._adapters: set[str] = set()
        self._embed_model_name = embed_model
        self._embedder = None
        self._arch = None

    # ---- loading -----------------------------------------------------------------
    @staticmethod
    def _load(model_id, dtype, device):
        import transformers
        for cls in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
            try:
                return getattr(transformers, cls).from_pretrained(
                    model_id, dtype=dtype, device_map=device)
            except Exception:
                continue
        raise RuntimeError(f"could not load {model_id} with any Auto* class")

    def _stop_token_ids(self) -> list[int]:
        """End-of-turn stop tokens so chat generation halts at the assistant boundary
        (otherwise the model hallucinates further fake turns up to max_new_tokens)."""
        stop = [self.tok.eos_token_id]
        for t in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>"):
            try:
                i = self.tok.convert_tokens_to_ids(t)
                if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                    stop.append(i)
            except Exception:
                pass
        return list(dict.fromkeys(s for s in stop if s is not None))

    # ---- introspection -----------------------------------------------------------
    def capabilities(self) -> DriverCapabilities:
        return DriverCapabilities(lora=True, train_lora=True, white_box=True,
                                  tool_use=True, streaming=True)

    def arch_info(self) -> ArchInfo:
        if self._arch is not None:
            return self._arch
        types: dict[str, int] = {}
        for n, _ in self.model.named_modules():
            if n.endswith(".self_attn"):
                types["attention"] = types.get("attention", 0) + 1
            if n.endswith(".linear_attn"):
                types["linear_attn"] = types.get("linear_attn", 0) + 1
        cfg = getattr(self.model, "config", None)
        tcfg = getattr(cfg, "text_config", cfg)
        # induction-capable iff it has real softmax-attention layers (needed for RAG)
        self._arch = ArchInfo(
            induction_capable=types.get("attention", 0) > 0,
            layer_types=types,
            hidden_size=getattr(tcfg, "hidden_size", None),
            num_layers=getattr(tcfg, "num_hidden_layers", None),
            notes="induction_capable = has softmax-attention layers (RAG-usable)",
        )
        return self._arch

    # ---- chat templating ---------------------------------------------------------
    def _chat_ids(self, messages, tools=()):
        kw = dict(add_generation_prompt=True, tokenize=True, return_tensors="pt")
        if tools:
            kw["tools"] = list(tools)
        for extra in (dict(enable_thinking=False), dict()):   # disable thinking if supported
            try:
                out = self.tok.apply_chat_template(messages, **kw, **extra)
                return (out if torch.is_tensor(out) else out["input_ids"]).to(self.device)
            except TypeError:
                continue

    # ---- adapter management ------------------------------------------------------
    def _ensure_peft(self, targets, r=16, alpha=32):
        from peft import LoraConfig, get_peft_model
        if not self._is_peft:
            cfg = LoraConfig(target_modules=targets, r=r, lora_alpha=alpha,
                             lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            self.model = get_peft_model(self.model, cfg)
            self._is_peft = True

    def _lang_targets(self):
        return [n for n, m in self.model.named_modules()
                if isinstance(m, torch.nn.Linear) and ".language_model." in n]

    def load_lora(self, adapter_path: str, lora_id: str) -> None:
        from peft import PeftModel
        if not self._is_peft:
            self.model = PeftModel.from_pretrained(self.model, adapter_path, adapter_name=lora_id)
            self._is_peft = True
        else:
            self.model.load_adapter(adapter_path, adapter_name=lora_id)
        self._adapters.add(lora_id)

    def activate_lora(self, lora_id: str | None) -> None:
        if self._is_peft and lora_id:
            self.model.set_adapter(lora_id)

    # ---- generation (streaming) --------------------------------------------------
    def generate(self, req: GenRequest) -> Iterator[str]:
        from transformers import TextIteratorStreamer
        ids = self._chat_ids(req.messages, tools=req.tools)
        use_lora = bool(req.lora_ids) and self._is_peft
        if use_lora:
            self.model.set_adapter(req.lora_ids[0])
        self.model.eval()
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(input_ids=ids, max_new_tokens=req.max_new_tokens,
                      do_sample=req.temperature > 0, streamer=streamer,
                      eos_token_id=self._stop_ids, pad_token_id=self._pad_id)
        if req.temperature > 0:
            kwargs["temperature"] = req.temperature

        def _run():
            with torch.no_grad():
                if self._is_peft and not use_lora:
                    with self.model.disable_adapter():
                        self.model.generate(**kwargs)
                else:
                    self.model.generate(**kwargs)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for chunk in streamer:
            yield chunk
        t.join()

    # ---- adapter training (skills) ----------------------------------------------
    def train_lora(self, examples: Iterable[dict[str, Any]], config: dict[str, Any]) -> str:
        """examples: [{'messages': [...chat...]}]; loss on assistant completion only.
        config: {lora_id, steps, lr, r, alpha}. Returns the saved adapter id."""
        from peft import LoraConfig, get_peft_model
        lora_id = config.get("lora_id", f"skill_{int(time.time())}")
        # leaf-name targets match language Linears before AND after PEFT wrapping (suffix match)
        leaves = sorted({n.split(".")[-1] for n, m in self.model.named_modules()
                         if isinstance(m, torch.nn.Linear) and ".language_model." in n
                         and "lora" not in n.lower()})
        lcfg = LoraConfig(target_modules=leaves, r=config.get("r", 16),
                          lora_alpha=config.get("alpha", 32), lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM")
        if not self._is_peft:
            self.model = get_peft_model(self.model, lcfg, adapter_name=lora_id)
            self._is_peft = True
        elif lora_id not in self.model.peft_config:
            self.model.add_adapter(lora_id, lcfg)
        self.model.set_adapter(lora_id)   # only the active adapter trains

        def to_pair(ex):
            msgs = ex["messages"]
            prompt = msgs[:-1]
            p = self._chat_ids(prompt)
            full_kw = dict(tokenize=True, return_tensors="pt")
            f = self.tok.apply_chat_template(msgs, **full_kw)
            f = (f if torch.is_tensor(f) else f["input_ids"]).to(self.device)
            lab = f.clone(); lab[:, :p.shape[1]] = -100
            return f, lab

        examples = list(examples)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=config.get("lr", 3e-4), weight_decay=0.01)
        self.model.train(); self.model.enable_input_require_grads()
        import random
        rng = random.Random(0)
        for _ in range(config.get("steps", 400)):
            f, lab = to_pair(rng.choice(examples))
            loss = self.model(input_ids=f, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
        self.model.eval()
        path = os.path.join(self.adapter_dir, lora_id)
        self.model.save_pretrained(path, selected_adapters=[lora_id])
        self._adapters.add(lora_id)
        return lora_id

    # ---- embeddings (routing / RAG keys) ----------------------------------------
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embed_model_name, device=self.device)
        v = self._embedder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return v.tolist()
