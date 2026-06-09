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

### Multi-tenancy

Mint per-tenant API keys; each key is locked to its own **project namespace**, so
isolation is enforced from the key (not from the request body):

```python
key, tenant = eg.add_tenant(name="acme")    # plaintext key returned once; stored hashed
# clients send:  Authorization: Bearer <key>  →  every call scoped to tenant.project
```

Auth turns on automatically once any tenant exists (or force it with
`create_app(..., require_auth=True)`). Without tenants, the server runs open.

### Canary lifecycle

With `canary=True`, a newly-consolidated skill enters **canary** — served to only a
fraction of matching traffic — before you promote it to full:

```python
eg = Engram("Qwen/Qwen3.5-4B", canary=True)
eg.consolidate()              # new skill enters as canary (e.g. 10% of matching traffic)
eg.promote("skill_id")        # → live (100%)     eg.rollback("skill_id") → stop serving
```

Over the API: `POST /v1/promote` / `POST /v1/rollback`; `GET /v1/capabilities` shows
each skill's `status` and `served` count.

### Consolidation policy

By default consolidation is **manual** (`eg.consolidate()`). For hands-off operation,
run it in the background, off the request path:

```python
Engram(..., consolidation="threshold", consolidation_threshold=8)    # auto-run after 8 queued
Engram(..., consolidation="scheduled", consolidation_interval=3600)  # or hourly
```

A shared model-lock serializes training with serving, so background consolidation
never races generation on the same GPU.

### Elastic / horizontal scale

Split training and serving across node types that share one S3/DynamoDB store:

- a **PEFTDriver** node runs consolidation (trains skill adapters),
- **VLLMDriver** nodes serve the base + adapters under continuous batching
  (high-throughput multi-LoRA), loading adapters from the shared store,
- nodes are **stateless** — point them at the same `store=` and call `eg.reload()`
  to pick up skills/knowledge produced elsewhere, so serving scales behind a load balancer.

```python
# serving node (Linux + CUDA + vLLM):  pip install "engram[vllm]"
from engram.drivers.vllm_driver import VLLMDriver
eg = Engram(driver=VLLMDriver("Qwen/Qwen3.5-4B"), store={"type": "aws", "bucket": "..."})
```

(vLLM is inference-only and Linux-only — train on a PEFTDriver node.)

## Status & roadmap

Early and honest about it. Built solo; the research core is validated, the
platform is being assembled in tiers:

- **Tier 0 — core** ✅: driver interface, capability registry, orchestrator,
  retrieval, adapter train/route, consolidation + eval-gate, OpenAI-compatible
  server, pluggable persistence (local files or AWS S3 + DynamoDB).
- **Tier 1 — platform** *(core complete)*: ✅ document connectors (files/dirs/pdf/
  html/code) · ✅ agentic orchestration (streaming model-driven tool loop) · ✅ knowledge
  graph + GraphRAG (multi-hop retrieval) · ✅ multi-tenancy + API-key auth
  (per-tenant project isolation) · ✅ canary lifecycle (staged skill promotion)
  · ✅ elastic serving (vLLM backend + stateless multi-node on a shared store).
- **Tier 2 — compliance**: SOC2 → HIPAA/GDPR/FedRAMP (design-for now).

All Tier-0/1 capabilities above are validated end-to-end on a real model
(Qwen3.5-4B); see commit history. CI runs on every push (`pytest` + `ruff`).

## License

[Apache-2.0](LICENSE).
