"""Agentic orchestrator (FR-O9/O10) — the model drives via tools.

Instead of the deterministic gate->retrieve->route->generate pipeline, the model
is given tools (a built-in `search_knowledge` + every registered TOOL capability)
and decides when to call them, in a loop, until it produces a final answer.

Tool-call protocol: native Hermes/Qwen `<tool_call>{...}</tool_call>`. Requires a
base with `capabilities().tool_use`. Falls back to a plain answer if no tool is called.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from engram.drivers.base import BaseLLMDriver, GenRequest
from engram.registry.registry import CapabilityKind, Registry

# Two tool-call dialects seen in the wild:
#   JSON (Hermes):  <tool_call>{"name": "...", "arguments": {...}}</tool_call>
#   XML (Qwen3.x):  <function=NAME><parameter=KEY>VALUE</parameter>...</function>
_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.S)


def _coerce(v: str):
    """Native XML params arrive as strings; recover ints/bools/json where possible."""
    try:
        return json.loads(v.strip())
    except Exception:                                               # noqa: BLE001
        return v.strip()

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": "Search the knowledge base for relevant passages. "
                       "Use this whenever the question needs facts you are unsure of.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "search query"}},
                       "required": ["query"]},
    },
}


@dataclass
class AgentResult:
    answer: str
    trace: list = field(default_factory=list)   # [{tool, args, result}, ...]
    iterations: int = 0


class AgenticOrchestrator:
    def __init__(self, driver: BaseLLMDriver, registry: Registry, retrievers: dict,
                 k: int = 3, max_iters: int = 4):
        self.driver = driver
        self.registry = registry
        self.retrievers = retrievers
        self.k = k
        self.max_iters = max_iters

    def _tools(self, project):
        tools = [SEARCH_TOOL]
        tools += [c.as_tool_spec()
                  for c in self.registry.list(project=project, kind=CapabilityKind.TOOL, live_only=True)]
        return tools

    def _execute(self, name, args, project):
        if name == "search_knowledge":
            r = self.retrievers.get(project)
            return [h.doc for h in r.retrieve(args.get("query", ""), self.k)] if r else []
        cap = self.registry.get(name, project=project)
        if cap and cap.kind == CapabilityKind.TOOL and callable(cap.handle):
            try:
                return cap.handle(**args)
            except Exception as e:                                  # noqa: BLE001
                return {"error": str(e)}
        return {"error": f"unknown tool: {name}"}

    @staticmethod
    def _parse(text):
        calls = []
        for m in _TOOL_RE.findall(text):                            # (a) JSON dialect
            try:
                d = json.loads(m)
            except Exception:                                       # noqa: BLE001
                continue
            if "name" in d:
                calls.append({"name": d["name"],
                              "arguments": d.get("arguments", d.get("parameters", {}))})
        for fn in _FUNC_RE.finditer(text):                          # (b) XML dialect
            args = {pm.group(1).strip(): _coerce(pm.group(2))
                    for pm in _PARAM_RE.finditer(fn.group(2))}
            calls.append({"name": fn.group(1).strip(), "arguments": args})
        return calls

    def run(self, messages, project: str = "default", max_new_tokens: int = 512) -> AgentResult:
        tools = self._tools(project)
        convo = list(messages)
        trace, out = [], ""
        for i in range(self.max_iters):
            out = "".join(self.driver.generate(
                GenRequest(messages=convo, tools=tools, max_new_tokens=max_new_tokens))).strip()
            calls = self._parse(out)
            if not calls:
                return AgentResult(answer=out, trace=trace, iterations=i + 1)
            convo.append({"role": "assistant", "content": "",
                          "tool_calls": [{"type": "function",
                                          "function": {"name": c["name"], "arguments": c["arguments"]}}
                                         for c in calls]})
            for c in calls:
                result = self._execute(c["name"], c["arguments"], project)
                convo.append({"role": "tool", "name": c["name"], "content": json.dumps(result)[:4000]})
                trace.append({"tool": c["name"], "args": c["arguments"], "result": result})
        # iteration budget spent — force a final answer with no tools
        out = "".join(self.driver.generate(
            GenRequest(messages=convo, max_new_tokens=max_new_tokens))).strip()
        return AgentResult(answer=out, trace=trace, iterations=self.max_iters)
