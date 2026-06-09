"""OpenAI-compatible API server (FR-A1) — makes Engram look like a normal LLM.

Endpoints:
  GET  /healthz
  GET  /v1/models
  POST /v1/chat/completions   (OpenAI-compatible; stream + non-stream; provenance in `engram`)
  POST /v1/teach              (Engram: in-band learning, FR-C6)
  POST /v1/consolidate        (Engram: run the learning loop)
  GET  /v1/capabilities       (Engram: list registered skills/knowledge)

Single-GPU note: model access is serialized with a lock (adapters are stateful).
Needs the serve extra:  pip install "engram[serve]"
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Optional

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover
    raise ImportError('API server needs the serve extra: pip install "engram[serve]"') from e


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "engram"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = 0.0
    max_tokens: int = 512
    project: str = "default"          # Engram extension (multi-tenant later)
    agentic: bool = False             # Engram extension: model-driven tool loop


class TeachRequest(BaseModel):
    text: str
    project: str = "default"
    name: Optional[str] = None


class IngestRequest(BaseModel):
    path: str                              # server-side file or directory path
    project: str = "default"
    consolidate: bool = False


class PromoteRequest(BaseModel):
    name: str
    project: str = "default"


def _sse(cid, model, delta=None, finish=None, extra=None) -> str:
    ch = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
          "model": model, "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if extra:
        ch.update(extra)
    return f"data: {json.dumps(ch)}\n\n"


def create_app(engram=None, model_id: str | None = None, device: str = "cuda:0",
               require_auth: bool | None = None) -> "FastAPI":
    app = FastAPI(title="Engram", version="0.0.1",
                  description="Continual-learning harness for open-weight LLMs")
    app.state.eg = engram
    app.state.cfg = (model_id, device)

    def eg():
        if app.state.eg is None:
            from engram.core import Engram
            mid, dev = app.state.cfg
            app.state.eg = Engram(mid, device=dev)
        return app.state.eg

    def auth(authorization: str = Header(None)):
        """Resolve the tenant from the Bearer key. Required when tenants exist (or
        require_auth=True). Returns the Tenant, or None in open mode."""
        ts = eg().tenants
        need = require_auth if require_auth is not None else (len(ts) > 0)
        if not need:
            return None
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        tenant = ts.resolve(authorization.split(" ", 1)[1].strip())
        if tenant is None:
            raise HTTPException(status_code=401, detail="invalid api key")
        if not eg().quotas.allow_request(tenant):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        return tenant

    def scoped(tenant, requested):
        """A tenant can only touch its own project — derived from the key, not the body."""
        return tenant.project if tenant else requested

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list",
                "data": [{"id": "engram", "object": "model", "owned_by": "engram"}]}

    @app.get("/v1/capabilities")
    def capabilities(project: str = "default", tenant=Depends(auth)):
        project = scoped(tenant, project)
        return {"capabilities": [
            {"name": c.name, "kind": c.kind.value, "description": c.description,
             "status": c.status, "served": c.served}
            for c in eg().registry.list(project=project)]}

    @app.post("/v1/promote")
    def promote(req: PromoteRequest, tenant=Depends(auth)):
        c = eg().promote(req.name, project=scoped(tenant, req.project))
        return {"name": req.name, "status": c.status if c else "not_found"}

    @app.post("/v1/rollback")
    def rollback(req: PromoteRequest, tenant=Depends(auth)):
        c = eg().rollback(req.name, project=scoped(tenant, req.project))
        return {"name": req.name, "status": c.status if c else "not_found"}

    def _check_docs(tenant, project):
        if tenant and not eg().quotas.within_limit(
                tenant, "max_documents", eg().retriever(project).count()):
            raise HTTPException(status_code=429, detail="document quota exceeded")

    @app.post("/v1/teach")
    def teach(req: TeachRequest, tenant=Depends(auth)):
        project = scoped(tenant, req.project)
        _check_docs(tenant, project)
        return eg().teach(req.text, project=project, name=req.name)

    @app.post("/v1/ingest")
    def ingest(req: IngestRequest, tenant=Depends(auth)):
        project = scoped(tenant, req.project)
        _check_docs(tenant, project)
        with eg().model_lock:
            return eg().ingest(req.path, project=project, consolidate=req.consolidate)

    @app.post("/v1/consolidate")
    def consolidate(tenant=Depends(auth)):
        with eg().model_lock:
            return {"results": eg().consolidate()}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest, tenant=Depends(auth)):
        messages = [m.model_dump() for m in req.messages]
        model, cid = req.model, "chatcmpl-" + uuid.uuid4().hex[:24]
        project = scoped(tenant, req.project)

        if req.agentic and req.stream:        # streaming tool loop (content + tool events)
            def agent_stream():
                with eg().model_lock:
                    yield _sse(cid, model, delta={"role": "assistant"})
                    for ev in eg().run_stream(messages, project=project, max_new_tokens=req.max_tokens):
                        if ev["type"] == "content":
                            yield _sse(cid, model, delta={"content": ev["text"]})
                        elif ev["type"] == "tool_call":
                            yield _sse(cid, model, delta={}, extra={"engram": {"tool_call": ev["calls"]}})
                        elif ev["type"] == "tool_result":
                            yield _sse(cid, model, delta={},
                                       extra={"engram": {"tool_result": {"name": ev["name"],
                                                                         "result": ev["result"]}}})
                    yield _sse(cid, model, delta={}, finish="stop")
                    yield "data: [DONE]\n\n"
            return StreamingResponse(agent_stream(), media_type="text/event-stream")

        if req.agentic:                       # model-driven tool loop (non-streaming)
            with eg().model_lock:
                r = eg().run(messages, project=project, max_new_tokens=req.max_tokens)
            return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": r.answer},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "engram": {"trace": r.trace, "iterations": r.iterations}}

        if req.stream:
            def stream():
                with eg().model_lock:
                    ans = eg().chat(messages, project=project,
                                    max_new_tokens=req.max_tokens, temperature=req.temperature)
                    yield _sse(cid, model, delta={"role": "assistant"},
                               extra={"engram": {"provenance": ans.provenance}})
                    for piece in ans.stream:
                        yield _sse(cid, model, delta={"content": piece})
                    yield _sse(cid, model, delta={}, finish="stop")
                    yield "data: [DONE]\n\n"
            return StreamingResponse(stream(), media_type="text/event-stream")

        with eg().model_lock:
            ans = eg().chat(messages, project=project,
                            max_new_tokens=req.max_tokens, temperature=req.temperature)
            text = ans.text()
        return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "engram": {"provenance": ans.provenance}}

    return app
