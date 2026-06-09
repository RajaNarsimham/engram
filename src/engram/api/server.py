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
import threading
import time
import uuid
from typing import Optional

try:
    from fastapi import FastAPI
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


class TeachRequest(BaseModel):
    text: str
    project: str = "default"
    name: Optional[str] = None


def _sse(cid, model, delta=None, finish=None, extra=None) -> str:
    ch = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
          "model": model, "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if extra:
        ch.update(extra)
    return f"data: {json.dumps(ch)}\n\n"


def create_app(engram=None, model_id: str | None = None, device: str = "cuda:0") -> "FastAPI":
    app = FastAPI(title="Engram", version="0.0.1",
                  description="Continual-learning harness for open-weight LLMs")
    lock = threading.Lock()
    app.state.eg = engram
    app.state.cfg = (model_id, device)

    def eg():
        if app.state.eg is None:
            from engram.core import Engram
            mid, dev = app.state.cfg
            app.state.eg = Engram(mid, device=dev)
        return app.state.eg

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list",
                "data": [{"id": "engram", "object": "model", "owned_by": "engram"}]}

    @app.get("/v1/capabilities")
    def capabilities(project: str = "default"):
        return {"capabilities": [
            {"name": c.name, "kind": c.kind.value, "description": c.description, "live": c.eval_passed}
            for c in eg().registry.list(project=project)]}

    @app.post("/v1/teach")
    def teach(req: TeachRequest):
        return eg().teach(req.text, project=req.project, name=req.name)

    @app.post("/v1/consolidate")
    def consolidate():
        with lock:
            return {"results": eg().consolidate()}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        messages = [m.model_dump() for m in req.messages]
        model, cid = req.model, "chatcmpl-" + uuid.uuid4().hex[:24]

        if req.stream:
            def stream():
                with lock:
                    ans = eg().chat(messages, project=req.project,
                                    max_new_tokens=req.max_tokens, temperature=req.temperature)
                    yield _sse(cid, model, delta={"role": "assistant"},
                               extra={"engram": {"provenance": ans.provenance}})
                    for piece in ans.stream:
                        yield _sse(cid, model, delta={"content": piece})
                    yield _sse(cid, model, delta={}, finish="stop")
                    yield "data: [DONE]\n\n"
            return StreamingResponse(stream(), media_type="text/event-stream")

        with lock:
            ans = eg().chat(messages, project=req.project,
                            max_new_tokens=req.max_tokens, temperature=req.temperature)
            text = ans.text()
        return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "engram": {"provenance": ans.provenance}}

    return app
