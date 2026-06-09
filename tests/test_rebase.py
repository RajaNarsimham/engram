from engram.drivers.base import ArchInfo, BaseLLMDriver, DriverCapabilities
from engram.registry.registry import Capability, CapabilityKind, Registry


class _Base(BaseLLMDriver):
    model_id = "base-x"
    adapter_dir = "."
    trains = False

    def capabilities(self):
        return DriverCapabilities(train_lora=self.trains)

    def arch_info(self):
        return ArchInfo(induction_capable=True, hidden_size=8, num_layers=2)

    def generate(self, req):
        yield "x"

    def embed(self, texts):
        return [[0.1] for _ in texts]


class DriverA(_Base):
    model_id = "base-a"


class DriverB(_Base):
    model_id = "base-b"


class DriverTrain(_Base):
    model_id = "base-c"
    trains = True


# ---- fingerprint -------------------------------------------------------------
def test_fingerprint_identifies_base():
    assert DriverA().fingerprint() == DriverA().fingerprint()
    assert DriverA().fingerprint() != DriverB().fingerprint()
    assert "base-a" in DriverA().fingerprint()


# ---- base_fp / source persist ------------------------------------------------
def test_capability_base_fp_and_source_roundtrip():
    r = Registry()
    r.register(Capability(name="s", kind=CapabilityKind.SKILL, description="d",
                          base_fp="base-a|h8|L2", source="the source text"))
    c = Registry().import_records(r.export()).get("s")
    assert c.base_fp == "base-a|h8|L2"
    assert c.source == "the source text"


# ---- base swap marks skills incompatible -------------------------------------
def test_swapping_base_marks_skill_incompatible(tmp_path):
    from engram.core import Engram
    store = str(tmp_path)
    a = Engram(driver=DriverA(), store_dir=store, load_on_start=False)
    a.registry.register(Capability(name="sk", kind=CapabilityKind.SKILL, description="d",
                                   routing_key=[0.1], handle="sk", eval_passed=True, status="live",
                                   base_fp=a.driver.fingerprint(), source="src"))
    a.save()

    b = Engram(driver=DriverB(), store_dir=store, load_on_start=True)   # different base
    assert b.registry.get("sk").status == "incompatible"
    assert [c.name for c in b.incompatible_skills()] == ["sk"]
    # incompatible skills are not servable
    assert b.registry.list(kind=CapabilityKind.SKILL, live_only=True) == []


def test_same_base_keeps_skill_live(tmp_path):
    from engram.core import Engram
    store = str(tmp_path)
    a = Engram(driver=DriverA(), store_dir=store, load_on_start=False)
    a.registry.register(Capability(name="sk", kind=CapabilityKind.SKILL, description="d",
                                   routing_key=[0.1], handle="sk", eval_passed=True, status="live",
                                   base_fp=a.driver.fingerprint(), source="src"))
    a.save()
    b = Engram(driver=DriverA(), store_dir=store, load_on_start=True)   # SAME base
    assert b.registry.get("sk").status == "live"


# ---- rebase queues incompatible-with-source, skips sourceless ----------------
def test_rebase_queues_sourced_skips_sourceless(tmp_path, monkeypatch):
    from engram.core import Engram
    eg = Engram(driver=DriverTrain(), store_dir=str(tmp_path), load_on_start=False)
    eg.registry.register(Capability(name="has_src", kind=CapabilityKind.SKILL, description="d",
                                    status="incompatible", source="source text"))
    eg.registry.register(Capability(name="no_src", kind=CapabilityKind.SKILL, description="d",
                                    status="incompatible", source=""))
    enq = []
    monkeypatch.setattr(eg._consolidator, "enqueue",
                        lambda text, project="default", name=None: enq.append(name) or name)
    monkeypatch.setattr(eg, "consolidate", lambda: [{"id": "has_src", "promoted": True}])
    res = eg.rebase()
    assert res["queued"] == ["has_src"]
    assert res["skipped_no_source"] == ["no_src"]
    assert res["retrained"] == ["has_src"]
    assert enq == ["has_src"]
