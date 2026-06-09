"""Per-tenant quotas (FR-NFR limits).

A tenant's `quota` dict may set:
  - "requests_per_min": sliding-window rate limit
  - "max_documents":    cap on RAG docs in the tenant's project
  - "max_skills":       cap on skills in the tenant's project
Rate state is in-memory per process (fine for a single node; a shared counter is a
later refinement). Resource caps are checked against live counts.
"""
from __future__ import annotations

import threading
import time
from collections import deque


class QuotaManager:
    def __init__(self):
        self._reqs: dict[str, deque] = {}      # tenant_id -> request timestamps (last 60s)
        self._lock = threading.Lock()

    def allow_request(self, tenant, now: float | None = None) -> bool:
        limit = (tenant.quota or {}).get("requests_per_min")
        if not limit:
            return True
        now = time.time() if now is None else now
        with self._lock:
            dq = self._reqs.setdefault(tenant.tenant_id, deque())
            while dq and dq[0] <= now - 60:
                dq.popleft()
            if len(dq) >= limit:
                return False
            dq.append(now)
            return True

    def within_limit(self, tenant, key: str, current: int) -> bool:
        """True if `current` is under the tenant's `key` cap (or no cap set)."""
        limit = (tenant.quota or {}).get(key)
        return limit is None or current < limit
