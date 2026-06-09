<div align="center">

# 🧠 Engram

**A continual-learning harness for open-weight LLMs.**

*Bring your own base model. Engram adds a routed skill-library, grounded retrieval,
and an offline learning loop — so the model keeps learning after deployment,
behind a normal (OpenAI/Anthropic-compatible) chat interface.*

`Apache-2.0` · `status: early / research-validated` · `solo-built`

</div>

---

> **engram** *(n.)* — the physical trace a memory leaves in the brain.
> Engram is the trace your model forms from everything it's taught.

## Why

Today's LLMs are **trained once and frozen**. They can't absorb new knowledge
without an expensive retrain, can't fix a mistake permanently, and don't get
better at *your* problem over time. Retrieval (RAG) bolts facts onto the context;
fine-tuning bakes in skills but goes stale. Neither, alone, gives you a model that
**keeps learning**.

Engram is the missing harness that makes a base LLM **continually learning** by
combining three things that each do one job well:

| Layer | Does | Job |
|---|---|---|
| **Base LLM** (plug-and-play) | reasoning + the substrate that *uses* context | the engine |
| **Skill library** (LoRA adapters, routed) | internalized skills & behaviors | *how* it acts |
| **Retrieval** (RAG + knowledge graph) | exact, fresh facts | *what* it knows |
| **Consolidation loop** (offline) | turns interactions into new skills/knowledge | *learning* |

The headline feature: **teach it in the chat.** Mid-conversation, tell the
assistant *"internalize this document"* — Engram ingests it into retrieval
immediately and queues a skill-adapter to internalize it, all behind a normal
chat API.

## What we validated first (this isn't vibes)

Engram's design rules come from controlled experiments, then a real frontier
model. The non-obvious findings baked into the architecture:

- **RAG = retrieval + induction.** A base model must be able to *use* retrieved
  context (an induction-head capability). Open instruct models have it; Engram
  checks for it.
- **RAG beats fine-tuning for *fresh* facts** — on a real 4B model, retrieval
  answered brand-new facts at ~88% while a fine-tuned model scored 0% (it never
  saw them). **Facts → retrieval, skills → adapters.**
- **Naive LoRA+RAG interferes** — a memorize-only adapter makes the model ignore
  retrieved context. The fix (context-preserving adapter training) is built in.
- **Route with frozen features, not a trained classifier** — trained routers
  overfit and don't generalize to adapters added later.

See [`docs/`](docs/) for the experiments and the full requirements.

## Architecture

```
        OpenAI/Anthropic-compatible API   (streaming, multi-tenant)
                        │
            ┌────────  Orchestrator  ────────┐   ← agentic: the model drives via tools
            │ gate → retrieve → route →       │
            │ assemble → generate → cite      │
            └─────────────────────────────────┘
              │            │              │
   Capability Registry  Base-LLM Driver  Consolidation Engine (offline)
   (skills/knowledge/    (vLLM | PEFT |   (teach → train adapter → eval-gate
    tools as plugins)     Ollama)          → stage/canary → promote)
```

Everything talks to the **Base-LLM Driver** interface — that's what makes base
models plug-and-play. Everything you add (a skill, a knowledge source, a tool) is
a **registered capability**.

## Quickstart

```bash
pip install "engram[peft,rag]"        # + [serve] for the API, + [aws] for S3/DynamoDB
```

```python
from engram.core import Engram

eg = Engram("Qwen/Qwen3.5-4B")        # bring your own open-weight base model
eg.teach("Project Zephyr ships March 3rd, 2026, led by Dana Okoro.")  # teach in-band → RAG instant
print(eg.chat("When does Project Zephyr ship?").text())               # grounded answer + provenance
eg.consolidate()                      # internalize it as an eval-gated skill adapter
```

Run it as a **normal OpenAI-compatible server**:

```bash
engram serve --model Qwen/Qwen3.5-4B  # → http://127.0.0.1:8000/v1
```
```bash
curl localhost:8000/v1/teach -d '{"text":"Acme Q3 revenue was $4.2M."}'
curl localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"What was Acme Q3 revenue?"}]}'
```
Any OpenAI client/SDK works — point its base URL at `http://127.0.0.1:8000/v1`.

### Persistence

State (RAG index, skill registry, adapters) auto-persists and survives restart —
**local files by default**, or **AWS S3 + DynamoDB** when configured:

```bash
export ENGRAM_S3_BUCKET=my-bucket
export ENGRAM_DDB_TABLE=engram-registry   # optional; else the registry lives in S3
```

## Status & roadmap

Early and honest about it. Built solo; the research core is validated, the
platform is being assembled in tiers:

- **Tier 0 — core** *(in progress)*: driver interface, capability registry,
  orchestrator, retrieval, adapter train/route, consolidation + eval-gate.
  Runs on a single workstation.
- **Tier 1 — platform**: multi-tenancy, elastic serving, knowledge graph,
  connector framework.
- **Tier 2 — compliance**: SOC2 → HIPAA/GDPR/FedRAMP (design-for now).

## License

[Apache-2.0](LICENSE).
