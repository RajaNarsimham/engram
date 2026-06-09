"""Engram CLI (stub). `engram --version` / `engram info`.

The real CLI (serve, register, teach, consolidate) lands with the Tier-0 build.
"""
from __future__ import annotations

import argparse

from engram import __version__


def main() -> None:
    ap = argparse.ArgumentParser(prog="engram", description="Continual-learning harness for LLMs")
    ap.add_argument("--version", action="version", version=f"engram {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("info", help="show package info")
    args = ap.parse_args()

    if args.cmd == "info" or args.cmd is None:
        print(f"engram {__version__}")
        print("continual-learning harness for open-weight LLMs")
        print("docs/REQUIREMENTS.md  ·  status: early / Tier-0 in progress")


if __name__ == "__main__":
    main()
