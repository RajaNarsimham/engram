from engram.registry.registry import Capability, CapabilityKind, Registry


def cap(name, project="default", live=True):
    return Capability(name=name, kind=CapabilityKind.SKILL, description=f"{name} skill",
                      routing_key=[0.1, 0.2, 0.3], when_to_use="x", eval_passed=live, project=project)


def test_register_and_live_filter():
    r = Registry()
    r.register(cap("a"))
    r.register(cap("b", live=False))
    assert {c.name for c in r.list()} == {"a", "b"}
    assert {c.name for c in r.list(live_only=True)} == {"a"}


def test_project_isolation():
    r = Registry()
    r.register(cap("a", project="p1"))
    r.register(cap("b", project="p2"))
    assert {c.name for c in r.list(project="p1")} == {"a"}
    assert {c.name for c in r.list(project="p2")} == {"b"}


def test_tool_specs_only_live():
    r = Registry()
    r.register(cap("planner"))
    r.register(cap("draft", live=False))
    names = [s["function"]["name"] for s in r.tool_specs()]
    assert names == ["planner"]


def test_export_import_roundtrip():
    r = Registry()
    r.register(cap("a"))
    r.register(cap("b"))
    r2 = Registry().import_records(r.export())
    assert {c.name for c in r2.list()} == {"a", "b"}
    assert r2.get("a").routing_key == [0.1, 0.2, 0.3]
    assert r2.get("a").kind == CapabilityKind.SKILL
