from engram.drivers.base import ArchInfo, BaseLLMDriver, DriverCapabilities
from engram.orchestrator.agentic import AgenticOrchestrator, AgentResult
from engram.registry.registry import Capability, CapabilityKind, Registry
from engram.retrieval.retriever import Hit


class ScriptedDriver(BaseLLMDriver):
    """Yields preset outputs, one per generate() call — drives the agent loop in tests."""

    def __init__(self, outputs):
        self._outputs = list(outputs)

    def capabilities(self):
        return DriverCapabilities(tool_use=True)

    def arch_info(self):
        return ArchInfo(induction_capable=True)

    def generate(self, req):
        yield self._outputs.pop(0)


class FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def retrieve(self, query, k):
        return [Hit(d, 0.9) for d in self._docs]


def test_parse_json_tool_call():
    text = 'ok <tool_call>{"name": "search_knowledge", "arguments": {"query": "x"}}</tool_call>'
    assert AgenticOrchestrator._parse(text) == [{"name": "search_knowledge", "arguments": {"query": "x"}}]


def test_parse_xml_tool_call():
    text = ("<tool_call>\n<function=search_knowledge>\n<parameter=query>\n"
            "meeting date\n</parameter>\n</function>\n</tool_call>")
    assert AgenticOrchestrator._parse(text) == [
        {"name": "search_knowledge", "arguments": {"query": "meeting date"}}]


def test_parse_xml_coerces_types():
    text = "<function=add><parameter=a>2</parameter><parameter=b>3</parameter></function>"
    assert AgenticOrchestrator._parse(text) == [{"name": "add", "arguments": {"a": 2, "b": 3}}]


def test_agent_searches_then_answers():
    driver = ScriptedDriver([
        '<tool_call>{"name": "search_knowledge", "arguments": {"query": "refund"}}</tool_call>',
        "Refunds take 14 days.",
    ])
    retr = {"default": FakeRetriever(["Refunds are processed within 14 business days."])}
    res = AgenticOrchestrator(driver, Registry(), retr).run(
        [{"role": "user", "content": "how long for refunds?"}], project="default")
    assert isinstance(res, AgentResult)
    assert res.answer == "Refunds take 14 days."
    assert res.trace[0]["tool"] == "search_knowledge"
    assert "14 business days" in res.trace[0]["result"][0]
    assert res.iterations == 2


def test_agent_answers_directly_when_no_tool_call():
    driver = ScriptedDriver(["The capital of France is Paris."])
    res = AgenticOrchestrator(driver, Registry(), {}).run([{"role": "user", "content": "capital?"}])
    assert res.answer == "The capital of France is Paris."
    assert res.trace == []
    assert res.iterations == 1


def test_registered_tool_is_called():
    driver = ScriptedDriver([
        '<tool_call>{"name": "add", "arguments": {"a": 2, "b": 3}}</tool_call>',
        "The sum is 5.",
    ])
    reg = Registry()
    reg.register(Capability(name="add", kind=CapabilityKind.TOOL, description="add two numbers",
                            handle=lambda a, b: a + b, eval_passed=True))
    res = AgenticOrchestrator(driver, reg, {}).run([{"role": "user", "content": "2+3?"}])
    assert res.trace[0]["result"] == 5
    assert res.answer == "The sum is 5."
