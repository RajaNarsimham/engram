"""KnowledgeGraph (FR-C18) — a lightweight triple store + multi-hop retrieval.

Stores (subject, relation, object) facts and answers a query by finding entities
mentioned in it and walking k hops to gather connected facts. Dependency-free
(plain adjacency dict); persists as graph.json inside the project dir, so it rides
along with the RAG index through the Store.
"""
from __future__ import annotations

import json
import os
import re


class KnowledgeGraph:
    def __init__(self):
        self.triples: list[dict] = []           # {"s","r","o","meta"}
        self.adj: dict[str, set[int]] = {}       # entity(lower) -> triple indices

    def add(self, s: str, r: str, o: str, meta: dict | None = None) -> None:
        s, r, o = s.strip(), r.strip(), o.strip()
        if not (s and r and o):
            return
        idx = len(self.triples)
        self.triples.append({"s": s, "r": r, "o": o, "meta": meta or {}})
        self.adj.setdefault(s.lower(), set()).add(idx)
        self.adj.setdefault(o.lower(), set()).add(idx)

    def add_many(self, triples, meta: dict | None = None) -> int:
        n = 0
        for t in triples:
            if len(t) == 3:
                self.add(t[0], t[1], t[2], meta)
                n += 1
        return n

    @property
    def entities(self) -> set[str]:
        return set(self.adj.keys())

    def _fact(self, i: int) -> str:
        t = self.triples[i]
        return f'{t["s"]} {t["r"]} {t["o"]}.'

    def query(self, text: str, hops: int = 2, max_facts: int = 12) -> list[str]:
        """Facts connected (within `hops`) to entities whose name appears in `text`."""
        # whole-word/phrase match (substring would let "a" match "unrel-a-ted")
        frontier = {e for e in self.adj
                    if re.search(r"\b" + re.escape(e) + r"\b", text, re.IGNORECASE)}
        if not frontier:
            return []
        seen_ent, idxs = set(frontier), set()
        for _ in range(max(1, hops)):
            nxt = set()
            for e in frontier:
                for ti in self.adj.get(e, ()):
                    idxs.add(ti)
                    for nb in (self.triples[ti]["s"].lower(), self.triples[ti]["o"].lower()):
                        if nb not in seen_ent:
                            seen_ent.add(nb)
                            nxt.add(nb)
            frontier = nxt
            if not frontier:
                break
        return [self._fact(i) for i in sorted(idxs)][:max_facts]

    # ---- persistence (rides along in the project dir) ----------------------------
    def save(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, "graph.json"), "w", encoding="utf-8") as f:
            json.dump({"triples": self.triples}, f)

    def load(self, dirpath: str) -> "KnowledgeGraph":
        p = os.path.join(dirpath, "graph.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for t in json.load(f).get("triples", []):
                    self.add(t["s"], t["r"], t["o"], t.get("meta"))
        return self

    def __len__(self) -> int:
        return len(self.triples)
