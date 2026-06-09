import pytest

from engram.auth.quota import QuotaManager
from engram.auth.tenants import Tenant
from engram.drivers.base import ArchInfo, BaseLLMDriver, DriverCapabilities


# ---- QuotaManager unit -------------------------------------------------------
def test_rate_limit_allows_then_blocks_then_recovers():
    q = QuotaManager()
    t = Tenant(tenant_id="t1", project="p", quota={"requests_per_min": 3})
    assert all(q.allow_request(t, 1000.0) for _ in range(3))   # 3 allowed in the window
    assert not q.allow_request(t, 1000.0)                      # 4th blocked
    assert q.allow_request(t, 1061.0)                          # window has slid -> allowed


def test_no_rate_limit_always_allows():
    q = QuotaManager()
    t = Tenant(tenant_id="t2", project="p", quota={})
    assert all(q.allow_request(t, 0.0) for _ in range(50))


def test_within_limit_resource_cap():
    q = QuotaManager()
    t = Tenant(tenant_id="t3", project="p", quota={"max_documents": 5})
    assert q.within_limit(t, "max_documents", 4) is True
    assert q.within_limit(t, "max_documents", 5) is False
    assert q.within_limit(t, "max_skills", 9999) is True       # no cap set -> allowed


# ---- server rate-limit enforcement -------------------------------------------
class MockDriver(BaseLLMDriver):
    adapter_dir = "."

    def capabilities(self):
        return DriverCapabilities(train_lora=False, tool_use=False)

    def arch_info(self):
        return ArchInfo(induction_capable=True)

    def generate(self, req):
        yield "x"

    def embed(self, texts):
        return [[0.1] for _ in texts]


def test_server_rate_limit_returns_429(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from engram.api.server import create_app
    from engram.core import Engram
    eg = Engram(driver=MockDriver(), store_dir=str(tmp_path), load_on_start=False)
    key, _ = eg.add_tenant(name="acme", quota={"requests_per_min": 2})
    c = TestClient(create_app(engram=eg))
    h = {"Authorization": f"Bearer {key}"}
    assert c.get("/v1/capabilities", headers=h).status_code == 200
    assert c.get("/v1/capabilities", headers=h).status_code == 200
    assert c.get("/v1/capabilities", headers=h).status_code == 429   # over the per-minute cap
