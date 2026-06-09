from engram.drivers.base import ArchInfo, BaseLLMDriver, DriverCapabilities
from engram.graph.extract import extract_triples
from engram.graph.store import KnowledgeGraph
from engram.orchestrator.agentic import AgenticOrchestrator
from engram.registry.registry import Registry


class _Driver(BaseLLMDriver):
    def __init__(self, outputs):
        self._outputs = list(outputs)

    def capabilities(self):
        return DriverCapabilities(tool_use=True)

    def arch_info(self):
        return ArchInfo(induction_capable=True)

    def generate(self, req):
        yield self._outputs.pop(0)


# ---- store -------------------------------------------------------------------
def test_query_one_hop():
    g = KnowledgeGraph()
    g.add("Mei Tanaka", "approves", "refunds")
    assert any("refunds" in f for f in g.query("who approves refunds?", hops=1))


def test_query_multi_hop():
    g = KnowledgeGraph()
    g.add("refunds", "approved by", "Mei Tanaka")
    g.add("Mei Tanaka", "works in", "Finance")
    joined = " ".join(g.query("tell me about refunds", hops=2))
    assert "Mei Tanaka" in joined and "Finance" in joined


def test_query_no_match_is_empty():
    g = KnowledgeGraph()
    g.add("A", "rel", "B")
    assert g.query("something entirely unrelated") == []


def test_add_many_skips_malformed():
    g = KnowledgeGraph()
    assert g.add_many([("a", "r", "b"), ("c", "r", "d"), ("bad",)]) == 2
    assert len(g) == 2


def test_save_load_roundtrip(tmp_path):
    g = KnowledgeGraph()
    g.add("Widget X", "weighs", "2.3 kg")
    g.save(str(tmp_path))
    g2 = KnowledgeGraph().load(str(tmp_path))
    assert len(g2) == 1
    assert any("2.3 kg" in f for f in g2.query("what about Widget X"))


# ---- extraction --------------------------------------------------------------
def test_extract_triples_parses_json():
    d = _Driver(['[["Mei Tanaka", "approves", "refunds"], ["Widget X", "weighs", "2.3 kg"]]'])
    triples = extract_triples(d, "...")
    assert ("Mei Tanaka", "approves", "refunds") in triples
    assert len(triples) == 2


def test_extract_triples_handles_garbage():
    assert extract_triples(_Driver(["sorry, no triples"]), "...") == []


# ---- agentic graph_search ----------------------------------------------------
def test_agentic_graph_search_tool():
    g = KnowledgeGraph()
    g.add("refunds", "approved by", "Mei Tanaka")
    driver = _Driver([
        '<tool_call>{"name": "graph_search", "arguments": {"entity": "refunds"}}</tool_call>',
        "Mei Tanaka approves refunds.",
    ])
    res = AgenticOrchestrator(driver, Registry(), {}, graphs={"default": g}).run(
        [{"role": "user", "content": "who approves refunds?"}], project="default")
    assert res.trace[0]["tool"] == "graph_search"
    assert any("Mei Tanaka" in f for f in res.trace[0]["result"])
