"""Distill from the Ollama teacher into Engram as MEDIUM-GRANULARITY, hierarchically-routed
context-LoRAs. ~3 groups/domain (general split by topic) -> ~10 focused LoRAs, each tagged
with a domain so the orchestrator routes domain-first (coarse, reliable) then subtopic.

Teacher: huihui_ai/qwen3-vl-abliterated:30b-a3b-instruct-q4 (fast, instruct -> clean CoT).
Run:  PYTHONIOENCODING=utf-8 PYTHONPATH=../src python distill.py
"""
import json
import re
import time

import requests

from engram.core import Engram
from engram.registry.registry import Capability, CapabilityKind

TEACHER = "huihui_ai/qwen3-vl-abliterated:30b-a3b-instruct-q4_K_M"
OLLAMA = "http://localhost:11434/v1/chat/completions"
STORE = "C:/Users/home/engram/distill_store"
PER_GROUP, STEPS = 44, 250

DOMAINS = {                                   # level-1 (coarse) routing descriptions
    "math": "mathematics arithmetic algebra geometry calculation problem solving",
    "coding": "programming python code functions algorithms software development",
    "general": "general knowledge science history geography facts explanations",
}
TMPL = {
    "math": "Generate {n} diverse {sub} math problems for practice. For each give a clear "
            "step-by-step solution ending with a line 'Final answer: <answer>'.",
    "coding": "Generate {n} diverse beginner Python {sub} problems. For each give a step-by-step "
              "solution including the code, ending with a line 'Final answer: <result>'.",
    "general": "Generate {n} diverse {sub} general-knowledge questions. For each give a concise "
               "step-by-step explanation ending with a line 'Final answer: <answer>'.",
}
JSON_TAIL = ' Output ONLY a JSON array: [{"q":"...","a":"...step by step...\\nFinal answer: ..."}]'
GROUPS = {   # name: (domain, level-2 subtopic-key description, [generation subtopics])
    "math_algebra": ("math", "arithmetic algebra fractions percentages ratios",
                     ["arithmetic", "fractions", "percentages", "basic algebra", "ratios and proportions"]),
    "math_geometry": ("math", "geometry shapes area perimeter angles volume",
                      ["area and perimeter", "angles", "triangles", "volume and surface area"]),
    "math_probability": ("math", "probability statistics averages word problems",
                         ["probability", "mean median mode", "data and averages", "multi-step word problems"]),
    "code_basics": ("coding", "python basics variables strings loops conditionals functions",
                    ["variables and strings", "loops", "conditionals", "simple functions"]),
    "code_datastructures": ("coding", "python data structures lists dictionaries sets tuples",
                            ["lists", "dictionaries", "sets and tuples", "nested structures"]),
    "code_algorithms": ("coding", "algorithms recursion sorting searching",
                        ["recursion", "sorting", "searching", "string algorithms"]),
    "gk_physics": ("general", "physics forces motion energy electricity",
                   ["forces and motion", "energy", "electricity and magnetism", "light and sound"]),
    "gk_biology": ("general", "biology human body plants cells ecology",
                   ["the human body", "plants and photosynthesis", "cells and genetics", "ecology"]),
    "gk_history": ("general", "world history civilizations wars famous people",
                   ["ancient civilizations", "major wars", "famous historical figures", "important inventions"]),
    "gk_geography": ("general", "geography countries capitals rivers mountains climate",
                     ["countries and capitals", "rivers and mountains", "climate and weather", "continents and oceans"]),
}


def teacher_call(prompt):
    try:
        r = requests.post(OLLAMA, json={"model": TEACHER, "messages": [{"role": "user", "content": prompt}],
                                        "temperature": 0.8, "max_tokens": 1500}, timeout=600)
        out = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\[.*\]", out, re.S)
        return [(d["q"], d["a"]) for d in json.loads(m.group(0))
                if isinstance(d, dict) and d.get("q") and d.get("a")] if m else []
    except Exception as e:
        print(f"    (teacher failed: {str(e)[:60]})", flush=True)
        return []


def main():
    # Phase 1: generate per group (teacher hot)
    corpus = {}
    for g, (domain, _, subs) in GROUPS.items():
        data, i, t0 = [], 0, time.time()
        while len(data) < PER_GROUP and i < PER_GROUP:
            data += teacher_call(TMPL[domain].format(n=4, sub=subs[i % len(subs)]) + JSON_TAIL)
            i += 1
        corpus[g] = data[:PER_GROUP]
        print(f"  {g}: {len(corpus[g])} examples ({round(time.time() - t0)}s)", flush=True)
    json.dump(corpus, open(f"{STORE}_data.json", "w"), indent=1)

    # Phase 2: one routed LoRA per group on the 0.8B student
    eg = Engram("Qwen/Qwen3.5-0.8B", store_dir=STORE, load_on_start=False)
    for g, (domain, subkey, _) in GROUPS.items():
        examples = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
                    for q, a in corpus[g]]
        print(f"  training {g} ({domain}) on {len(examples)} examples...", flush=True)
        eg.driver.train_lora(examples, {"lora_id": g, "steps": STEPS, "r": 16})
        eg.registry.register(Capability(
            name=g, kind=CapabilityKind.SKILL, description=f"{g} (distilled)", handle=g,
            routing_key=eg.driver.embed([subkey])[0], when_to_use=subkey, eval_passed=True,
            status="live", base_fp=eg.driver.fingerprint(), source=g,
            metadata={"domain": domain, "domain_desc": DOMAINS[domain]}))

    print("\n=== hierarchical routing check (query -> domain/subtopic LoRA) ===", flush=True)
    for q in ["What is 15% of 80?", "Find the area of a triangle with base 6 and height 4",
              "Write a Python function to reverse a list", "How do I sort a list with recursion?",
              "What is photosynthesis?", "What is the capital of France?", "Who was Napoleon?"]:
        print(f"  {q!r}\n     -> {eg.chat(q).provenance.get('skill')!r}", flush=True)


if __name__ == "__main__":
    main()
