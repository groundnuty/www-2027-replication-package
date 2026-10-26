"""Offline subclass closure over a source's own parent links.

Every builder, the renderer and the profiler compute ancestors, roots and
depths here, from ``ClassInfo.parents`` (or ``PropertyInfo.parents``), so the
same hierarchy is read the same way everywhere. Cycles are tolerated: a node
is never its own ancestor.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping


class Hierarchy:
    def __init__(self, parents: Mapping[str, Iterable[str]]):
        # Parents that are not themselves nodes are added as nodes with no parents.
        self.parents: dict[str, tuple[str, ...]] = {}
        for node, ps in parents.items():
            self.parents[node] = tuple(sorted(set(ps) - {node}))
        for ps in list(self.parents.values()):
            for p in ps:
                self.parents.setdefault(p, ())
        self._anc: dict[str, frozenset[str]] = {}
        self._children: dict[str, list[str]] | None = None

    @property
    def nodes(self) -> set[str]:
        return set(self.parents)

    def ancestors(self, node: str) -> frozenset[str]:
        """Proper ancestors (not the node itself)."""
        if node in self._anc:
            return self._anc[node]
        seen: set[str] = set()
        stack = list(self.parents.get(node, ()))
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(self.parents.get(n, ()))
        seen.discard(node)
        out = frozenset(seen)
        self._anc[node] = out
        return out

    def ancestors_or_self(self, node: str) -> frozenset[str]:
        return self.ancestors(node) | {node}

    def equivalents(self, node: str) -> frozenset[str]:
        """The other members of the node's cycle (A below B and B below A): its proper ancestors that have it as an
        ancestor. Empty in an acyclic hierarchy."""
        return frozenset(a for a in self.ancestors(node) if node in self.ancestors(a))

    def children(self) -> dict[str, list[str]]:
        if self._children is None:
            ch: dict[str, list[str]] = {n: [] for n in self.parents}
            for n, ps in self.parents.items():
                for p in ps:
                    ch[p].append(n)
            for v in ch.values():
                v.sort()
            self._children = ch
        return self._children

    def descendants(self, node: str) -> set[str]:
        ch = self.children()
        seen: set[str] = set()
        stack = list(ch.get(node, ()))
        while stack:
            n = stack.pop()
            if n in seen or n == node:
                continue
            seen.add(n)
            stack.extend(ch.get(n, ()))
        return seen

    def roots(self, among: Iterable[str] | None = None) -> list[str]:
        nodes = self.parents if among is None else set(among)
        return sorted(n for n in nodes if not self.parents.get(n))

    def roots_of(self, node: str) -> frozenset[str]:
        """The top classes above a node (the node itself if it has no parents)."""
        return frozenset(a for a in self.ancestors_or_self(node) if not self.parents.get(a))

    def depths(self, among: Iterable[str] | None = None) -> dict[str, int]:
        """Shortest distance from a root, computed inside ``among`` (default: all nodes).

        ``among`` must be closed upwards (every ancestor of a member is a member).
        """
        keep = set(self.parents) if among is None else set(among)
        ch = self.children()
        depth: dict[str, int] = {}
        queue = deque()
        for r in sorted(n for n in keep if not any(p in keep for p in self.parents.get(n, ()))):
            depth[r] = 0
            queue.append(r)
        while queue:
            n = queue.popleft()
            for c in ch.get(n, ()):
                if c in keep and c not in depth:
                    depth[c] = depth[n] + 1
                    queue.append(c)
        # Members only reachable through a cycle get the largest depth seen plus one.
        rest = sorted(keep - set(depth))
        base = max(depth.values(), default=-1) + 1
        for n in rest:
            depth[n] = base
        return depth

    def breadth_first(self, among: Iterable[str] | None = None) -> list[str]:
        """Nodes ordered by depth from the roots, then by IRI: the one order every layer uses."""
        d = self.depths(among)
        return sorted(d, key=lambda n: (d[n], n))

    def upward_closure(self, nodes: Iterable[str]) -> set[str]:
        out: set[str] = set()
        for n in nodes:
            if n in self.parents:
                out |= self.ancestors_or_self(n)
        return out

    def minimal_general(self, nodes: Iterable[str]) -> list[str]:
        """The members of ``nodes`` that have no proper ancestor in ``nodes``. Members of one cycle are each
        other's ancestors; the cycle keeps one representative, its smallest IRI, so it is never dropped whole."""
        s = set(nodes)
        out = []
        for n in s:
            eq = self.equivalents(n)
            if not ((self.ancestors(n) - eq) & s) and not any(e < n for e in eq & s):
                out.append(n)
        return sorted(out)
