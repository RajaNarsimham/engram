import pytest

from engram.drivers.base import ArchInfo, BaseLLMDriver, DriverCapabilities, GenRequest
from engram.orchestrator.orchestrator import Orchestrator
from engram.registry.registry import Capability, CapabilityKind, Registry
from engram.retrieval.retriever import Hit


class MockDriver(BaseLLMDriver):
    def capabilities(self):
        return DriverCapabilities(lora=True, tool_use=True)

    def arch_info(self):
        return ArchInfo(induction_capable=True)

    def generate(self, req):
        self.last_req = req
        yield "ok"


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def retrieve(self, query, k):
        return self._hits


def _emb(texts):
    return [[1.0, 0.0] for _ in texts]   # query always points at [1,0]


# ---- driver interface --------------------------------------------------------
def test_mock_driver_satisfies_interface():
    d = MockDriver()
    assert d.capabilities().lora is True
    assert d.arch_info().induction_capable is True
    assert "".join(d.generate(GenRequest(messages=[{"role": "user", "content": "x"}]))) == "ok"


def test_unsupported_methods_raise():
    d = MockDriver()
    with pytest.raises(NotImplementedError):
        d.train_lora([], {})


# ---- orchestrator routing / assembly -----------------------------------------
def _skill(name, key):
    return Capability(name=name, kind=CapabilityKind.SKILL, description="d",
                      routing_key=key, eval_passed=True)


def test_routes_to_nearest_skill_above_threshold():
    reg = Registry()
    reg.register(_skill("match", [1.0, 0.0]))
    reg.register(_skill("other", [0.0, 1.0]))
    o = Orchestrator(MockDriver(), reg, {}, _emb, route_threshold=0.5)
    name, score = o._route("q", "default")
    assert name == "match" and score >= 0.5


def test_route_below_threshold_returns_none():
    reg = Registry()
    reg.register(_skill("other", [0.0, 1.0]))
    o = Orchestrator(MockDriver(), reg, {}, _emb, route_threshold=0.5)
    assert o._route("q", "default")[0] is None


def test_assemble_injects_context_as_system():
    o = Orchestrator(MockDriver(), Registry(), {}, _emb)
    msgs = o._assemble([{"role": "user", "content": "q"}], [Hit("a key fact", 0.9)])
    assert msgs[0]["role"] == "system" and "a key fact" in msgs[0]["content"]


def test_answer_retrieves_and_reports_provenance():
    o = Orchestrator(MockDriver(), Registry(), {"default": FakeRetriever([Hit("the fact", 0.9)])}, _emb)
    ans = o.answer([{"role": "user", "content": "q"}], project="default")
    assert ans.provenance["docs"] == ["the fact"]
    assert ans.text() == "ok"
