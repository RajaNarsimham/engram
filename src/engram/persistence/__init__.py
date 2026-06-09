"""Persistence backends.

Auto-selection (when no explicit store is given):
  - if ENGRAM_S3_BUCKET is set in the environment -> AWSStore
      (+ optional ENGRAM_DDB_TABLE, ENGRAM_S3_PREFIX, AWS_REGION)
  - otherwise -> FileStore (local files; the default)

Or configure explicitly:
  Engram(..., store={"type": "aws", "bucket": "my-bkt", "table": "engram-registry"})
  Engram(..., store={"type": "file", "root": "/data/engram"})
  Engram(..., store=MyCustomStore())
"""
from __future__ import annotations

import os

from engram.persistence.store import FileStore, Store


def make_store(config=None, work_dir: str = "engram_store"):
    if config is not None and hasattr(config, "push_registry"):     # already a Store
        return config
    if config is None:                                              # auto from env
        bucket = os.getenv("ENGRAM_S3_BUCKET")
        if bucket:
            from engram.persistence.aws_store import AWSStore
            return AWSStore(work_dir, bucket=bucket, table=os.getenv("ENGRAM_DDB_TABLE"),
                            prefix=os.getenv("ENGRAM_S3_PREFIX", "engram/"),
                            region=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))
        return FileStore(work_dir)
    if isinstance(config, dict):
        t = config.get("type", "file")
        if t == "file":
            return FileStore(config.get("root", work_dir))
        if t in ("aws", "s3"):
            from engram.persistence.aws_store import AWSStore
            return AWSStore(work_dir, bucket=config["bucket"], table=config.get("table"),
                            prefix=config.get("prefix", "engram/"), region=config.get("region"))
    raise ValueError(f"unknown store config: {config!r}")


__all__ = ["Store", "FileStore", "make_store"]
