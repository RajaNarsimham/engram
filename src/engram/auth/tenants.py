"""Multi-tenancy + auth (FR-R4, FR-NFR security).

Per-tenant API keys. A key maps to a tenant whose `project` is their data
namespace — so isolation is enforced server-side from the *key*, never from an
untrusted request field. Keys are stored only as SHA-256 hashes; the plaintext is
returned once at creation.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field


def _hash(key: str) -> str:
    return hashlib.sha256((key or "").encode()).hexdigest()


@dataclass
class Tenant:
    tenant_id: str
    project: str                       # the namespace this tenant's data lives in
    name: str = ""
    quota: dict = field(default_factory=dict)


class TenantStore:
    def __init__(self):
        self._by_hash: dict[str, Tenant] = {}     # sha256(api_key) -> Tenant

    def create(self, name: str = "", project: str | None = None,
               tenant_id: str | None = None, quota: dict | None = None) -> tuple[str, Tenant]:
        """Mint a new tenant + API key. Returns (plaintext_key, tenant). Key shown once."""
        tid = tenant_id or "t_" + secrets.token_hex(6)
        key = "engram_" + secrets.token_hex(24)
        t = Tenant(tenant_id=tid, project=project or tid, name=name, quota=quota or {})
        self._by_hash[_hash(key)] = t
        return key, t

    def resolve(self, api_key: str) -> Tenant | None:
        return self._by_hash.get(_hash(api_key))

    def revoke(self, api_key: str) -> None:
        self._by_hash.pop(_hash(api_key), None)

    def __len__(self) -> int:
        return len(self._by_hash)

    # ---- persistence (hashes only — never plaintext keys) ------------------------
    def export(self) -> list[dict]:
        return [{"key_hash": h, "tenant_id": t.tenant_id, "project": t.project,
                 "name": t.name, "quota": t.quota} for h, t in self._by_hash.items()]

    def import_records(self, records) -> "TenantStore":
        for r in records or []:
            self._by_hash[r["key_hash"]] = Tenant(
                tenant_id=r["tenant_id"], project=r["project"],
                name=r.get("name", ""), quota=r.get("quota", {}))
        return self
