"""Pluggable persistence (the `Store` interface) — same idea as the BaseLLMDriver:
the core never touches storage directly, so backends are swappable.

Two backends:
  FileStore  — the working directory IS the store (local files). Default; zero deps,
               air-gap friendly.
  AWSStore   — S3 for blobs (RAG indexes, adapter weights) + DynamoDB (or S3) for the
               registry. (see aws_store.py)

A `Store` persists three things:
  - the registry         (a list of capability dicts)
  - directory blobs       (a project's RAG index/docs, a skill's adapter weights)
The local working dir always holds live files the runtime reads (FAISS / adapters);
the Store syncs that working dir to/from the backend.
"""
from __future__ import annotations

import json
import os
import shutil
from abc import ABC, abstractmethod


class Store(ABC):
    @abstractmethod
    def push_registry(self, records: list[dict]) -> None: ...
    @abstractmethod
    def pull_registry(self) -> list[dict]: ...
    @abstractmethod
    def push_dir(self, key: str, local_dir: str) -> None: ...
    @abstractmethod
    def pull_dir(self, key: str, local_dir: str) -> bool: ...   # True if the key existed
    @abstractmethod
    def list_dirs(self, prefix: str) -> list[str]: ...          # immediate subkeys under prefix


class FileStore(Store):
    """The local working directory is the persistent store. push/pull are no-ops when
    the live files already live under `root` at the same key (the default layout)."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _reg(self) -> str:
        return os.path.join(self.root, "registry.json")

    def push_registry(self, records: list[dict]) -> None:
        with open(self._reg(), "w", encoding="utf-8") as f:
            json.dump(records, f)

    def pull_registry(self) -> list[dict]:
        p = self._reg()
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def push_dir(self, key: str, local_dir: str) -> None:
        dest = os.path.join(self.root, *key.split("/"))
        if os.path.isdir(local_dir) and os.path.abspath(local_dir) != os.path.abspath(dest):
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(local_dir, dest)

    def pull_dir(self, key: str, local_dir: str) -> bool:
        src = os.path.join(self.root, *key.split("/"))
        if os.path.abspath(local_dir) == os.path.abspath(src):
            return os.path.isdir(local_dir)
        if os.path.isdir(src):
            os.makedirs(os.path.dirname(local_dir) or ".", exist_ok=True)
            shutil.rmtree(local_dir, ignore_errors=True)
            shutil.copytree(src, local_dir)
            return True
        return False

    def list_dirs(self, prefix: str) -> list[str]:
        d = os.path.join(self.root, *prefix.split("/"))
        return sorted(n for n in os.listdir(d)) if os.path.isdir(d) else []
