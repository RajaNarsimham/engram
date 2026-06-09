"""Engram CLI.

    engram --version
    engram info
    engram serve --model Qwen/Qwen3.5-4B --port 8000   # OpenAI-compatible server
"""
from __future__ import annotations

import argparse

from engram import __version__


def main() -> None:
    ap = argparse.ArgumentParser(prog="engram", description="Continual-learning harness for LLMs")
    ap.add_argument("--version", action="version", version=f"engram {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("info", help="show package info")

    s = sub.add_parser("serve", help="run the OpenAI-compatible API server")
    s.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3.5-4B")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    args = ap.parse_args()

    if args.cmd == "serve":
        import uvicorn

        from engram.api.server import create_app
        app = create_app(model_id=args.model, device=args.device)
        uvicorn.run(app, host=args.host, port=args.port)
    else:  # info / default
        print(f"engram {__version__}")
        print("continual-learning harness for open-weight LLMs")
        print("status: Tier-0 — base + skill-LoRAs + RAG + teach/consolidate loop")


if __name__ == "__main__":
    main()
