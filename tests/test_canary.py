import random

from engram.drivers.base import ArchInfo, BaseLLMDriver, DriverCapabilities
from engram.orchestrator.orchestrator import Orchestrator
from engram.registry.registry import Capability, CapabilityKind, Registry


class MockDriver(BaseLLMDriver):
    def capabilities(self):
        return DriverCapabilities()

    def arch_info(self):
        return ArchInfo()

    def generate(self, req):
        yield "ok"


def _emb(texts):
    return [[1.0, 0.0] for _ in texts]


def _skill(name, key, status="live", canary_pct=0.1):
    return Capability(name=name, kind=CapabilityKind.SKILL, description="d",
                      routing_key=key, status=status, canary_pct=canary_pct)


def test_canary_served_only_a_fraction():
    reg = Registry()
    reg.register(_skill("canary_skill", [1.0, 0.0], status="canary", canary_pct=0.3))
    o = Orchestrator(MockDriver(), reg, {}, _emb, route_threshold=0.5)
    o._rng = random.Random(42)
    routed = sum(1 for _ in range(400) if o._route("q", "default")[0] == "canary_skill")
    assert 80 < routed < 160          # ~30% of 400, generous band


def test_canary_bypass_falls_back_to_live_alternative():
    reg = Registry()
    reg.register(_skill("canary_skill", [1.0, 0.0], status="canary", canary_pct=0.0))  # never serve
    reg.register(_skill("live_skill", [0.99, 0.01], status="live"))                    # slightly lower
    o = Orchestrator(MockDriver(), reg, {}, _emb, route_threshold=0.5)
    assert o._route("q", "default")[0] == "live_skill"


def test_promote_and_rollback_status():
    reg = Registry()
    reg.register(_skill("s", [1.0, 0.0], status="canary"))
    assert reg.set_status("s", "live").status == "live"
    assert [c.name for c in reg.list(kind=CapabilityKind.SKILL, live_only=True)] == ["s"]
    reg.set_status("s", "rolled_back")
    assert reg.list(kind=CapabilityKind.SKILL, live_only=True) == []


def test_rolled_back_skill_not_routed():
    reg = Registry()
    reg.register(_skill("s", [1.0, 0.0], status="rolled_back"))
    o = Orchestrator(MockDriver(), reg, {}, _emb, route_threshold=0.5)
    assert o._route("q", "default")[0] is None


def test_status_survives_export_import():
    reg = Registry()
    reg.register(_skill("s", [1.0, 0.0], status="canary", canary_pct=0.25))
    reg2 = Registry().import_records(reg.export())
    c = reg2.get("s")
    assert c.status == "canary" and c.canary_pct == 0.25
