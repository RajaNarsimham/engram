"""AWSStore — S3 for directory blobs (RAG indexes, adapter weights) + DynamoDB
(or S3) for the registry. Local working dir is a cache; S3/DDB is the source of truth.

Registry items are stored as `{pk, data}` where `data` is the capability JSON — this
sidesteps DynamoDB's float/Decimal handling (routing_key vectors stay clean).

Needs the aws extra:  pip install "engram[aws]"   (boto3)
"""
from __future__ import annotations

import json
import os


class AWSStore:  # implements engram.persistence.store.Store (duck-typed to avoid import cost)
    def __init__(self, work_dir: str, bucket: str, table: str | None = None,
                 prefix: str = "engram/", region: str | None = None):
        try:
            import boto3
        except ImportError as e:  # pragma: no cover
            raise ImportError('AWSStore needs the aws extra: pip install "engram[aws]"') from e
        self.work_dir = work_dir
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.s3 = boto3.client("s3", region_name=region)
        self.table = boto3.resource("dynamodb", region_name=region).Table(table) if table else None

    def _k(self, key: str) -> str:
        return self.prefix + key

    # ---- registry ----------------------------------------------------------------
    def push_registry(self, records: list[dict]) -> None:
        if self.table is not None:
            for r in records:
                self.table.put_item(Item={"pk": f"{r.get('project', 'default')}#{r['name']}",
                                          "data": json.dumps(r)})
        else:
            self.s3.put_object(Bucket=self.bucket, Key=self._k("registry.json"),
                               Body=json.dumps(records).encode())

    def pull_registry(self) -> list[dict]:
        if self.table is not None:
            items, resp = [], self.table.scan()
            items += resp.get("Items", [])
            while "LastEvaluatedKey" in resp:
                resp = self.table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
                items += resp.get("Items", [])
            return [json.loads(i["data"]) for i in items if "data" in i]
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self._k("registry.json"))
            return json.loads(obj["Body"].read())
        except Exception:
            return []

    # ---- directory blobs ---------------------------------------------------------
    def push_dir(self, key: str, local_dir: str) -> None:
        if not os.path.isdir(local_dir):
            return
        for root, _, files in os.walk(local_dir):
            for fn in files:
                lp = os.path.join(root, fn)
                rel = os.path.relpath(lp, local_dir).replace("\\", "/")
                self.s3.upload_file(lp, self.bucket, self._k(f"{key}/{rel}"))

    def pull_dir(self, key: str, local_dir: str) -> bool:
        pfx = self._k(f"{key}/")
        found = False
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=pfx):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(pfx):]
                dest = os.path.join(local_dir, *rel.split("/"))
                os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                self.s3.download_file(self.bucket, obj["Key"], dest)
                found = True
        return found

    def list_dirs(self, prefix: str) -> list[str]:
        pfx = self._k(prefix.rstrip("/") + "/")
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=pfx, Delimiter="/")
        return sorted(cp["Prefix"][len(pfx):].rstrip("/") for cp in resp.get("CommonPrefixes", []))
