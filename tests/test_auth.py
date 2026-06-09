import pytest

from engram.auth.tenants import TenantStore
from engram.drivers.base import ArchInfo, BaseLLMDriver, DriverCapabilities
from engram.registry.registry import Capability, CapabilityKind


# ---- TenantStore -------------------------------------------------------------
def test_create_and_resolve():
    ts = TenantStore()
    key, t = ts.create(name="acme", project="acme_proj")
    assert key.startswith("engram_")
    assert ts.resolve(key).project == "acme_proj"
    assert ts.resolve("wrong-key") is None


def test_keys_stored_only_as_hash():
    ts = TenantStore()
    key, _ = ts.create()
    dumped = str(ts.export())
    assert key not in dumped            # plaintext key never persisted
    assert "key_hash" in dumped


def test_export_import_roundtrip():
    ts = TenantStore()
    key, _ = ts.create(name="x", project="p")
    ts2 = TenantStore().import_records(ts.export())
    assert ts2.resolve(key).project == "p"


def test_revoke():
    ts = TenantStore()
    key, _ = ts.create()
    ts.revoke(key)
    assert ts.resolve(key) is None


# ---- server auth + isolation (no model needed) -------------------------------
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


def _client(tmp_path, **kw):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from engram.api.server import create_app
    from engram.core import Engram
    eg = Engram(driver=MockDriver(), store_dir=str(tmp_path), load_on_start=False)
    return eg, TestClient(create_app(engram=eg, **kw))


def test_open_mode_when_no_tenants(tmp_path):
    _, c = _client(tmp_path)
    assert c.get("/v1/capabilities").status_code == 200      # no tenants -> open


def test_auth_required_once_a_tenant_exists(tmp_path):
    eg, c = _client(tmp_path)
    key, _ = eg.add_tenant(name="acme")
    assert c.get("/v1/capabilities").status_code == 401                              # no key
    assert c.get("/v1/capabilities", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/v1/capabilities", headers={"Authorization": f"Bearer {key}"}).status_code == 200


def test_tenant_is_scoped_to_its_own_project(tmp_path):
    eg, c = _client(tmp_path)
    key, t = eg.add_tenant(name="acme")
    eg.registry.register(Capability(name="mine", kind=CapabilityKind.KNOWLEDGE,
                                    description="d", project=t.project, eval_passed=True))
    eg.registry.register(Capability(name="theirs", kind=CapabilityKind.KNOWLEDGE,
                                    description="d", project="other_tenant", eval_passed=True))
    # even though we pass ?project=other_tenant, the key forces the tenant's own project
    r = c.get("/v1/capabilities?project=other_tenant", headers={"Authorization": f"Bearer {key}"})
    assert [x["name"] for x in r.json()["capabilities"]] == ["mine"]
