# Product Requirements — Continual-Learning LLM Orchestration Platform

**Status:** DRAFT — requirements session in progress.
**Working name:** TBD
**One-liner:** Bring your own base LLM; the platform adds a routed skill-LoRA library,
RAG grounding, and an offline continual-learning loop — served behind a normal
(OpenAI-compatible) LLM interface.

> Legend: `[CONFIRMED]` decided · `[TBD]` needs the requirements session · `[DERIVED]`
> follows from our validated experiments (`nnet/` + `lora-rag/`).

---

## 0. Foundations — CONFIRMED (session 1)
- **Product form:** **Open-source framework** (self-hosted); optional managed/support tier later.
- **Primary persona:** **Enterprise AI/IT teams** (private data, governance, control).
- **Base models:** **Open-weights only** → the *full* LoRA + continual-learning loop is available everywhere; **no closed-API tier** (simpler driver matrix).
- **Compute:** **Hybrid** — must run single-node (workstation, e.g. RTX 5090) **and** scale to cluster/cloud.

**Cascaded requirements (derived from the foundations):**
- C-1 Permissive **OSS license** (Apache-2.0 or similar) to enable enterprise adoption.
- C-2 **Air-gap / offline operation** supported — no required external calls at runtime (enterprise private data).
- C-3 **Data locality** — all data (docs, LoRAs, logs) stays in the deployment; no phone-home/telemetry by default.
- C-4 Clean, documented **plugin SDK + driver SDK** — extensibility is the core OSS-framework value.
- C-5 **RBAC, audit logging, governance hooks** (enterprise).
- C-6 **Modular, pip-installable, config-driven**; minimal hard deps; scales single workstation → cluster.
- C-7 **Security**: secrets management, project/namespace isolation within a deployment, supply-chain hygiene.
- C-8 Open backends only in v1 (vLLM / transformers-PEFT / Ollama); no closed-API driver.

### Session 2 — runtime & lifecycle (CONFIRMED)
- **Orchestration:** **Agentic (model-driven)** — the base LLM drives via tool calls; RAG and skill-selection are exposed as **tools**.
- **Continual-learning trigger:** **Policy-configurable** (manual | scheduled | threshold), per project.
- **Promotion:** **Staged (canary) + configurable** — train → eval → staging/canary → monitor → promote (auto or human-approval per policy) → rollback.
- **Scale:** **Full-range elastic** — single workstation → large elastic multi-node cluster.

**Cascaded requirements (from Session 2):**
- C-9 **Agent runtime**: tool-calling protocol, loop control (max-steps, timeouts), safe tool execution, full agent-trace observability. Base must be **tool-use-capable** (add to `arch_info()` checks).
- C-10 **Capabilities exposed as tools**: RAG = `retrieve` tool, each skill-LoRA = a `use_skill` tool, each tool = a tool — the model routes by reading tool **descriptions**; frozen-feature router becomes an *optional accelerator/fallback*, not the only path.
- C-11 **Consolidation policy engine**: manual / scheduled / threshold, configurable per project.
- C-12 **Deployment lifecycle**: staging / canary / prod environments, monitoring, promotion gates, rollback, versioning of capabilities + base models.
- C-13 **Distributed/elastic serving**: multi-node, autoscaling, distributed RAG store, scaled multi-LoRA serving (vLLM/LoRAX), scheduler/queue, load-balancing.
- C-14 **Tension to flag:** elastic + air-gap + agentic + full-lifecycle = a *large* surface. Recommend internal phasing even though all features are required for "done" (see §10).

### Session 3 — scope, data & memory (CONFIRMED)
- **Use cases:** **ALL** — internal knowledge assistant, domain-expert agents, customer-facing assistants, dev/coding assistants, **+ general agentic domain-specific workflows** → a **general-purpose agentic platform**.
- **Isolation:** **Full multi-tenant** (hard boundaries, quotas, billing hooks) within one deployment.
- **Data sources:** **ALL** — documents/files, APIs/live connectors, code/repos, databases/structured.
- **Memory:** Conversation + long-term + **organizational knowledge graph**.

**Cascaded (Session 3):**
- C-15 General-purpose **agent/workflow definition** (define domain-specific agentic workflows), not just chat.
- C-16 **Guardrails & safety** (customer-facing): content filtering, tone/policy control, escalation/hand-off, PII handling.
- C-17 **Connector framework + SDK**: document parsing (PDF/docx/md/html), structured/DB retrieval (text-to-SQL or row-embed), code-aware chunking (AST/symbols), live connectors (Confluence/SharePoint/Jira/APIs) with **sync/refresh + incremental indexing**.
- C-18 **Knowledge-graph subsystem**: entity/relation extraction, KG storage, **GraphRAG** (KG-augmented retrieval), maintenance — plus per-user/project **long-term episodic memory** (CLS short/long).
- C-19 **Multi-tenancy everywhere**: per-tenant LoRAs / RAG / KG / memory / policies / quotas / usage metering / hard data isolation.
- C-20 **SCOPE MAGNITUDE (honest):** this is a *complete enterprise agentic AI platform*. Requirements capture the full vision as asked; the **build MUST be phased** (§10) — realistically a large, multi-quarter, multi-engineer effort. No single feature is droppable for "done," but they ship in dependency order.

### Session 4 — resourcing, compliance, success, phasing (CONFIRMED)
- **Resourcing:** **Solo / nights-and-weekends.**
- **Compliance target:** **Regulated** (HIPAA / GDPR / FedRAMP) — highest bar.
- **North-star metric:** **Time-to-add-a-capability** (register skill/source → eval → live; minutes/hours).
- **Phasing (stated):** **Capability-by-capability** to production-grade.

### ⚠ REALITY CHECK — scope ↔ resourcing (honest, must read)
The confirmed scope (complete enterprise agentic platform: multi-tenant + all-data + KG + agentic + elastic + OSS + **HIPAA/FedRAMP**) is, at full fidelity, a **multi-team, multi-year** effort. **Solo / part-time cannot deliver the full vision as stated.** Specifically:
- **FedRAMP/HIPAA certification** needs dedicated security/compliance staff, audits, budget — not achievable solo (you can *architect for* it, not *certify* it).
- **Full multi-tenancy + elastic distributed serving** is a platform/infra team's job.
- **KG subsystem + all live connectors + customer-facing guardrails** are each multi-person workstreams.
- Stated **capability-by-capability** phasing is the **slowest-to-value** path — the wrong fit for a solo builder who needs an integrated, demonstrable differentiator early.

**Recommended reconciliation — keep the vision, build in TIERS:**
- **Tier 0 — solo-buildable differentiated CORE** (single workstation / your 5090): driver interface + capability registry + agentic-lite orchestrator + RAG + LoRA train/route + simple consolidation + eval-gate. Delivers the **north-star (time-to-add-a-capability)** and the *unbuilt-anywhere* differentiator. **Shippable solo as OSS.**
- **Tier 1 — platform layer** (needs contributors/funding): multi-tenancy, elastic serving, full KG/GraphRAG, connector framework, lifecycle/canary at scale.
- **Tier 2 — compliance layer** (needs team + budget): SOC2 → HIPAA/GDPR/FedRAMP — **design-for now, certify later**.
- **Phasing recommendation (overrides 'capability-by-capability' for solo):** **spine-first → one vertical slice** (knowledge-assistant e2e) to prove the north-star, then expand. (Stated preference recorded; this is the realistic path.)

---

## 1. Vision, Goals & Success Metrics
- **1.1 Vision** — `[TBD]`
- **1.2 Product form / deployment model** — `[TBD: SaaS | self-hosted | OSS framework | internal]`
- **1.3 Primary value proposition** — the *integrated continual-learning loop* (routing +
  consolidation + eval-gated lifecycle) — the piece deep-research confirmed is **unbuilt**.
  RAG and LoRA-serving are reused plumbing, not the moat. `[CONFIRMED-direction]`
- **1.4 Success metrics** — `[TBD]` (e.g., fresh-knowledge accuracy vs retrain cost; time-to-add-a-skill; routing accuracy at library scale)

## 2. Personas & Use Cases
- **2.1 Primary persona** — `[TBD]`
- **2.2 Secondary personas** — `[TBD]`
- **2.3 Core use cases / workflows** — `[TBD]`

## 3. Functional Requirements

### 3.1 Base-LLM Driver & Model Lifecycle (plug-and-play)
- FR-D1 `[DERIVED]` Single driver interface: `generate / activate_lora / load_lora / train_lora / embed / capabilities() / arch_info()`. Orchestrator is backend-agnostic.
- FR-D2 `[DERIVED]` Backends: vLLM (multi-LoRA serving), transformers/PEFT (train+infer), Ollama/llama.cpp (infer), API (closed models, RAG-only tier).
- FR-D3 `[DERIVED]` `capabilities()` declares LoRA/white-box/training support → orchestrator degrades gracefully (tiers).
- FR-D4 `[DERIVED]` `arch_info()` exposes/tests **induction-capability** (can the base USE retrieved context); warn/block if not.
- FR-D5 `[TBD]` Supported base-model scope (open-only vs open+closed-API).
- FR-D6 `[TBD]` Model registration, versioning, hot-swap.

### 3.2 Capability Registry (LoRA / RAG / tool plugins)
- FR-R1 `[DERIVED]` Register capabilities with metadata: key/embedding, description, when-to-use, eval scores, version.
- FR-R2 `[DERIVED]` Capability types: skill-LoRA, knowledge-RAG-source, tool.
- FR-R3 `[DERIVED]` Add/remove capability with NO code change (registration only).
- FR-R4 `[TBD]` Multi-tenancy / isolation of libraries.

### 3.3 Orchestration Runtime
- FR-O1 `[DERIVED]` Per-request pipeline: gate → retrieve → route → assemble → generate → post-process.
- FR-O2 `[DERIVED]` **Retrieval relevance-gate**: skip retrieval if nothing passes threshold (irrelevant context HURTS — measured).
- FR-O3 `[DERIVED]` **Frozen-feature LoRA routing** (NOT a trained classifier — overfits/doesn't generalize, measured) + confidence threshold + base fallback.
- FR-O4 `[DERIVED]` Prompt assembler: context injection, window budgeting, chat template, **disable thinking-mode** where it truncates.
- FR-O5 `[DERIVED]` Division of labor enforced: facts→RAG, skills→LoRA.
- FR-O6 `[TBD]` Control policy: deterministic vs agentic (model-driven tool calls).

### 3.4 Continual-Learning Engine (offline)
- FR-C1 `[DERIVED]` Consolidation job: train/distill new skill-LoRAs, grow RAG index, prune, re-key router.
- FR-C2 `[DERIVED]` **Interference-safe** LoRA training/merge (context-preserving — naive merge breaks RAG/recall, measured & fixed).
- FR-C3 `[DERIVED]` Distillation pipeline: black-box (teacher outputs → LoRA), teacher-agnostic.
- FR-C4 `[CONFIRMED]` Trigger model: **policy-configurable** — manual | scheduled | threshold, per project (§0 S2).
- FR-C5 `[TBD]` Forgetting/retention policy as library grows.
- FR-C6 `[CONFIRMED]` **In-conversation teaching (in-band learning) — KEY USER-FACING FEATURE.**
  Through the chat interface (OpenAI/Anthropic-compatible), a user/agent can instruct the
  system to *internalize* data (documents, pasted text, examples, feedback). The agentic
  runtime exposes a `teach/internalize` tool that:
    (a) **immediately ingests to RAG** (chunk → embed → index → available the next turn);
    (b) **creates/queues a LoRA training + consolidation job** to internalize skill/knowledge
        (async, through the eval-gate + staged promotion);
    (c) returns an **in-conversation acknowledgement** with status (available-now via retrieval
        vs queued-for-internalization).
  - **Honest timing constraint:** RAG add is instant; **LoRA training CANNOT complete within a
    chat turn** (minutes–hours + eval) → "available now via retrieval, internalized after
    consolidation." Creation is triggered in-band; internalization completes asynchronously.
  - **Division of labor:** facts/knowledge → RAG (instant); skill/behavior → LoRA (async).
  - **Governance:** respects RBAC (who may teach), consolidation policy, and promotion gates
    (approval before a taught LoRA goes live); tenant data isolation.
  - **Tiering:** the teach-tool + RAG-instant + LoRA-queue is **Tier-0** (solo-buildable);
    approval/multi-tenant governance is Tier-1/2.

### 3.5 Evaluation & Quality Gates
- FR-E1 `[DERIVED]` A capability must PASS an eval before going live (seen/fresh-style benchmark, generalized).
- FR-E2 `[TBD]` Eval suite definition, regression gates, human-in-the-loop.

### 3.6 API, SDKs & Integration
- FR-A1 `[DERIVED]` OpenAI-compatible endpoint + streaming (drop-in for existing apps).
- FR-A2 `[TBD]` SDKs, auth, rate limiting, webhooks.

### 3.7 Memory & Conversation
- FR-M1 `[DERIVED]` Multi-turn conversation memory (short-term).
- FR-M2 `[TBD]` Long-term/episodic user memory; relation to RAG.

### 3.8 Management, Versioning & Admin
- FR-G1 `[TBD]` Lifecycle: register → eval → deploy → monitor → consolidate → retire; rollback.
- FR-G2 `[TBD]` Admin console / control plane.

### 3.9 Observability & Provenance
- FR-OB1 `[DERIVED]` Every answer logs which LoRA(s) + which docs were used (provenance, "why X").
- FR-OB2 `[TBD]` Tracing, cost/latency metrics, dashboards.

## 4. Non-Functional Requirements
- NFR-1 Performance/latency budget — `[TBD]`
- NFR-2 Throughput / concurrency — `[TBD]`
- NFR-3 Scalability (capabilities, tenants, requests) — `[TBD]`
- NFR-4 Reliability / availability / fallbacks — `[TBD]`
- NFR-5 Security & privacy (tenant isolation, data residency, secrets) — `[TBD]`
- NFR-6 Cost / GPU efficiency — `[TBD]`
- NFR-7 Extensibility (plugin SDK, custom drivers) — `[TBD]`

## 5. Deployment & Operations
- **5.1 Compute model** — `[TBD: user GPUs | platform cloud | both]`
- **5.2 Hosting / packaging** — `[TBD]`
- **5.3 GPU/serving management** — `[TBD]`

## 6. Data Requirements
- Data flows, storage, retention, governance, PII — `[TBD]`

## 7. Constraints & Assumptions
- **7.1 Hardware** — reference: RTX 5090 (32GB) + DGX Spark (128GB). `[CONFIRMED-dev]`
- **7.2 Legal** — distillation-from-API ToS; model licenses. `[DERIVED-risk]`
- **7.3 Team / timeline / budget** — `[TBD]`

## 8. Non-Goals / Out of Scope
- `[TBD]`

## 9. Open Questions & Risks
- OQ-1 `[DERIVED]` **Routing generalization at scale** — does frozen-feature routing hold for a large, continually-grown LoRA library? (the genuinely-open research risk / core moat)
- OQ-2 `[DERIVED]` Multi-LoRA interference at scale during consolidation.
- OQ-3 `[TBD]` ...
