"""The census co-occurrence graph: where swap targets are drawn from (proposal §7.6).

Definition (fixed here, before any checker runs on the bank):

    Nodes   every predicate p of the census record with at least one entry in
            ``position_support[p]``, and every class or datatype listed there.
    Edges   p - X, undirected, for each X in ``position_support[p]["subject"]`` or
            ``position_support[p]["object"]``; the position is kept on the edge but does not
            change the distance.

``position_support`` lists, per predicate position, the most specific classes (and literal
datatypes) at or over the census support threshold (check contract rule 33), so the graph holds
only patterns the data-derived ontology allows. The graph is bipartite, so distances between two
predicates or two classes are even; distance 2 means "shares a predicate" (classes) or "shares a
class or datatype" (predicates). Distances are never measured on a curated hierarchy: no subclass
edge of any source enters the graph.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

XSD = "http://www.w3.org/2001/XMLSchema#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def is_datatype(term: str) -> bool:
    return term.startswith(XSD) or term in (RDF + "langString", RDF + "HTML", RDF + "XMLLiteral")


@dataclass
class CooccurrenceGraph:
    predicates: set[str] = field(default_factory=set)
    classes: set[str] = field(default_factory=set)
    datatypes: set[str] = field(default_factory=set)
    adjacency: dict[str, set[str]] = field(default_factory=dict)
    positions: dict[tuple[str, str], set[str]] = field(default_factory=dict)  # (p, position) -> nodes
    _dist: dict[str, dict[str, int]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_census(cls, census: dict) -> CooccurrenceGraph:
        g = cls()
        for p, by_position in sorted(census["position_support"].items()):
            for position in ("subject", "object"):
                for x in sorted(by_position.get(position, {})):
                    g.predicates.add(p)
                    (g.datatypes if is_datatype(x) else g.classes).add(x)
                    g.adjacency.setdefault(p, set()).add(x)
                    g.adjacency.setdefault(x, set()).add(p)
                    g.positions.setdefault((p, position), set()).add(x)
        return g

    def kind(self, node: str) -> str | None:
        if node in self.predicates:
            return "predicate"
        if node in self.classes:
            return "class"
        if node in self.datatypes:
            return "datatype"
        return None

    def distances(self, source: str) -> dict[str, int]:
        if source not in self._dist:
            dist = {source: 0} if source in self.adjacency else {}
            queue = deque(dist)
            while queue:
                x = queue.popleft()
                for y in self.adjacency.get(x, ()):
                    if y not in dist:
                        dist[y] = dist[x] + 1
                        queue.append(y)
            self._dist[source] = dist
        return self._dist[source]

    def distance(self, a: str, b: str) -> int | None:
        return self.distances(a).get(b)

    def neighbours(self, predicate: str, position: str) -> set[str]:
        return set(self.positions.get((predicate, position), set()))

    def candidates(self, source: str, kind: str, hops: int) -> list[str]:
        """Nodes of ``kind`` ("predicate" or "class") at exactly ``hops`` from ``source``, sorted."""
        return self.candidates_from([source], kind, hops)

    def candidates_from(self, sources: list[str], kind: str, hops: int) -> list[str]:
        """Nodes of ``kind`` whose distance to the nearest of ``sources`` is exactly ``hops``."""
        nearest: dict[str, int] = {}
        for s in sources:
            for node, d in self.distances(s).items():
                if d < nearest.get(node, d + 1):
                    nearest[node] = d
        pool = self.predicates if kind == "predicate" else self.classes
        return sorted(n for n, d in nearest.items() if d == hops and n in pool and n not in sources)
