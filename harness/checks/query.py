"""Analysis of the agent's SPARQL query: what the checks read from it.

The query is parsed with rdflib's parser and translated to its algebra. The
supported fragment is frozen (§7.2): SELECT with basic graph patterns,
OPTIONAL, FILTER, aggregates and fixed-length property paths, plus GRAPH
(S0.1 ruling 19), and VALUES and FILTER equality as sources of constants
(ruling 5). MINUS and NOT EXISTS are negated parts. Every other construct is outside the fragment: it is named
in ``unsupported``, the variables it touches are ``tainted`` (their static
checks become unknown) and the footprint form does not run.

The pattern becomes a tree of groups. The root holds the required patterns;
each OPTIONAL is a child group, combined with its ancestors only (ruling 7);
each MINUS / NOT EXISTS is a negated child group, analysed on its own and
never combined with the outer part (S0.1 F12). Patterns inside unsupported
constructs go to detached groups, read only by vocabulary existence.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any

from rdflib import BNode, Literal, URIRef, Variable
from rdflib.paths import AlternativePath, InvPath, MulPath, NegatedPath, SequencePath
from rdflib.plugins.sparql.parserutils import CompValue

from harness.sparql_grammar import PARSER_LOCK, parse_query, translate_query

Term = Any  # an rdflib term: Variable, URIRef, Literal
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASSOF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"


@dataclass(eq=False)
class Edge:
    """A triple pattern: subject, predicate alternatives (one IRI usually,
    several for a|b, none for a variable predicate), object."""

    s: Term
    preds: tuple[str, ...]
    o: Term
    group: Group
    graph: Term = None  # the GRAPH term the pattern sits in (URIRef or Variable), or None
    pvar: Variable | None = None  # the variable of a variable predicate
    closure: str | None = None  # "+" / "*" for rdfs:subClassOf+ / *; "type*" for rdf:type/rdfs:subClassOf* (rule 19)
    negated: bool = False  # one IRI of a negated property set (!p): the pattern excludes it

    def path_text(self) -> str:
        """The predicate as SPARQL text, for an auxiliary query."""
        if self.closure == "type*":
            return f"<{RDF_TYPE}>/<{RDFS_SUBCLASSOF}>*"
        if self.closure:
            return f"<{RDFS_SUBCLASSOF}>{self.closure}"
        return f"<{self.preds[0]}>"

    def show(self) -> str:
        p = "|".join(f"<{x}>" for x in self.preds) or "?pred"
        return f"{_show(self.s)} {p} {_show(self.o)}"


@dataclass(eq=False)
class Group:
    kind: str  # "required", "optional", "negated", "detached"
    parent: Group | None
    gid: int
    edges: list[Edge] = field(default_factory=list)
    bindings: dict[Variable, set] = field(default_factory=dict)  # VALUES and FILTER equality constants
    children: list[Group] = field(default_factory=list)
    filtered: bool = False  # a FILTER applies inside the group

    def scope_groups(self) -> list[Group]:
        """This group and, for an OPTIONAL, its ancestors up to the required
        root or the negated group it sits in."""
        out = [self]
        g = self
        while g.kind == "optional" and g.parent is not None:
            g = g.parent
            out.append(g)
        return out

    def scope_edges(self) -> list[Edge]:
        return [e for g in self.scope_groups() for e in g.edges]

    def scope_bindings(self) -> dict[Variable, tuple]:
        """The constants each variable of the scope is bound to, in the order of their term text: the checks list
        a result per constant, and a set's order follows PYTHONHASHSEED (term text, not the plain string, which an
        IRI and a literal, or two literals differing in datatype or language, can share)."""
        out: dict[Variable, set] = {}
        for g in self.scope_groups():
            for v, cs in g.bindings.items():
                out.setdefault(v, set()).update(cs)
        return {v: tuple(sorted(cs, key=term_text)) for v, cs in out.items()}


@dataclass
class QueryAnalysis:
    text: str
    root: Group | None
    groups: list[Group]
    unsupported: list[str]
    tainted: set[Variable]
    fresh: set[Variable]  # path intermediates and blank nodes: never bound in result rows
    negation: bool
    aggregate: bool
    error: str | None = None
    projected: tuple[str, ...] = ()  # the SELECT clause's variables, sorted (all in scope for SELECT *)

    def scopes(self) -> list[Group]:
        """Groups whose scope the static checks analyse: the root, every
        OPTIONAL group, every negated group; nothing under a detached group."""
        return [g for g in self.groups if g.kind in ("required", "optional", "negated") and not _detached(g)]

    def positive(self) -> list[Group]:
        """The required root and the OPTIONAL groups hanging from it: the part
        of the pattern the footprint reads (never anything under a negation)."""
        return [g for g in self.groups if g.kind in ("required", "optional") and g.scope_groups()[-1].kind == "required"]

    def local_vars(self, g: Group) -> set[Variable]:
        """Variables of an OPTIONAL group that no other part of the positive
        pattern binds (ancestors, sibling OPTIONALs and their subtrees): the
        group matched in a row exactly when one of them is bound."""
        subtree: set[int] = set()
        stack = [g]
        while stack:
            h = stack.pop()
            subtree.add(h.gid)
            stack.extend(h.children)
        outer: set[Variable] = set()
        for h in self.groups:
            if h.gid in subtree or h.kind not in ("required", "optional"):
                continue
            for e in h.edges:
                outer |= _vars_of(e)
            outer |= set(h.bindings)
        own = set()
        for e in g.edges:
            own |= _vars_of(e)
        return {v for v in own - outer if v not in self.fresh}


def _detached(g: Group) -> bool:
    while g is not None:
        if g.kind == "detached":
            return True
        g = g.parent
    return False


def _vars_of(e: Edge) -> set[Variable]:
    return {t for t in (e.s, e.o) if isinstance(t, Variable)}


def _show(t: Term) -> str:
    if isinstance(t, Variable):
        return f"?{t}"
    if isinstance(t, URIRef):
        return f"<{t}>"
    return t.n3()


# Parse-tree names that are outside the frozen fragment.
_UNSUPPORTED_PARSE = {
    "ServiceGraphPattern": "SERVICE",
    "SubSelect": "subquery",
    "Bind": "BIND",
    "Builtin_EXISTS": "EXISTS",
    "PathNegatedPropertySet": "negated_property_set",
}


def _parse_tree_constructs(tree: Any) -> tuple[set[str], set[Variable]]:
    """Constructs outside the fragment, and the variables BIND assigns."""
    found: set[str] = set()
    bound: set[Variable] = set()

    def walk(x: Any) -> None:
        if isinstance(x, CompValue):
            if x.name in _UNSUPPORTED_PARSE:
                found.add(_UNSUPPORTED_PARSE[x.name])
            if x.name == "Bind" and isinstance(x.get("var"), Variable):
                bound.add(x.var)
            if x.name == "GroupOrUnionGraphPattern" and len(x.get("graph") or []) > 1:
                found.add("UNION")
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)) or hasattr(x, "asList"):
            for v in x:
                walk(v)

    walk(tree)
    return found, bound


class _Builder:
    def __init__(self) -> None:
        self.groups: list[Group] = []
        self.unsupported: set[str] = set()
        self.tainted: set[Variable] = set()
        self.fresh: set[Variable] = set()
        self.negation = False
        self.aggregate = False
        self._ids = itertools.count()
        self.graphs: list[Term] = []  # enclosing GRAPH terms
        self.bnodes: dict = {}  # pattern blank node -> its variable

    def group(self, kind: str, parent: Group | None) -> Group:
        g = Group(kind, parent, next(self._ids))
        if parent is not None:
            parent.children.append(g)
        self.groups.append(g)
        return g

    def term(self, t: Term) -> Term:
        if isinstance(t, BNode):  # a blank node in a pattern is a variable no row binds
            # named by first appearance, not by rdflib's random label, so reports are reproducible
            v = self.bnodes.setdefault(t, Variable(f"_bn{len(self.bnodes)}"))
            self.fresh.add(v)
            return v
        return t

    def fresh_var(self) -> Variable:
        v = Variable(f"_path{len(self.fresh)}")
        self.fresh.add(v)
        return v

    def taint(self, vars_: Any, construct: str) -> None:
        self.unsupported.add(construct)
        for v in vars_ or ():
            if isinstance(v, Variable):
                self.tainted.add(v)

    # -- patterns -------------------------------------------------------------

    def walk(self, node: Any, g: Group, top: bool = False) -> None:
        if not isinstance(node, CompValue):
            return
        name = node.name
        if name in ("SelectQuery", "AskQuery", "DescribeQuery", "ConstructQuery"):
            self.walk(node.p, g, top=True)
        elif name == "Project":
            if top:
                self.walk(node.p, g, top=True)
            else:  # a nested SELECT
                self.taint(node.get("PV"), "subquery")
                self.walk(node.p, self.group("detached", g))
        elif name in ("Distinct", "Reduced", "Slice", "OrderBy"):
            self.walk(node.p, g, top)
        elif name in ("Group", "AggregateJoin"):
            self.aggregate = True
            self.walk(node.p, g, top)
        elif name == "Extend":
            # an expression in the SELECT clause, a grouped variable, or a BIND
            # (BIND is outside the fragment; found in the parse tree)
            self.walk(node.p, g, top)
        elif name == "Filter":
            self.walk(node.p, g)
            g.filtered = True
            self.filter_expr(node.expr, g)
        elif name == "Join":
            self.walk(node.p1, g)
            self.walk(node.p2, g)
        elif name == "LeftJoin":
            self.walk(node.p1, g)
            child = self.group("optional", g)
            self.walk(node.p2, child)
            expr = node.get("expr")
            if isinstance(expr, CompValue) and expr.name != "TrueFilter":
                child.filtered = True
                self.filter_expr(expr, child)
        elif name == "Minus":
            self.negation = True
            self.walk(node.p1, g)
            self.walk(node.p2, self.group("negated", g))
        elif name == "Union":
            self.taint(node.get("_vars"), "UNION")
            d = self.group("detached", g)
            self.walk(node.p1, d)
            self.walk(node.p2, d)
        elif name == "Graph":
            # in the fragment (ruling 19): its patterns are checked as usual; type
            # lookups use the union of graphs, the re-run keeps the GRAPH clause
            self.graphs.append(node.term)
            self.walk(node.p, g)
            self.graphs.pop()
        elif name == "ServiceGraphPattern":
            # never analysed and never executed; its terms belong to another dataset
            self.taint(node.get("_vars"), "SERVICE")
        elif name == "BGP":
            for s, p, o in node.triples:
                self.path(self.term(s), p, self.term(o), g)
        elif name == "ToMultiSet":
            self.walk(node.p, g)
        elif name == "values":
            for row in node.res:
                for var, val in row.items():
                    if isinstance(val, (URIRef, Literal)):  # rdflib gives UNDEF as the string "UNDEF"
                        g.bindings.setdefault(var, set()).add(val)
        else:
            self.taint(node.get("_vars"), name)

    def edge(self, s: Term, preds: tuple[str, ...], o: Term, g: Group, pvar: Variable | None = None,
             closure: str | None = None) -> None:
        g.edges.append(Edge(s, preds, o, g, self.graphs[-1] if self.graphs else None, pvar, closure))

    def path(self, s: Term, p: Any, o: Term, g: Group) -> None:
        if isinstance(p, URIRef):
            self.edge(s, (str(p),), o, g)
        elif isinstance(p, Variable):
            self.edge(s, (), o, g, p)
        elif isinstance(p, InvPath):
            self.path(o, p.arg, s, g)
        elif isinstance(p, SequencePath) and len(p.args) == 2 and p.args[0] == URIRef(RDF_TYPE) and _is_subclass_star(p.args[1], "*"):
            # rdf:type/rdfs:subClassOf*: membership through the closure (rule 19)
            self.edge(s, (RDF_TYPE,), o, g, closure="type*")
        elif isinstance(p, SequencePath):
            nodes = [s] + [self.fresh_var() for _ in p.args[:-1]] + [o]
            for i, step in enumerate(p.args):
                self.path(nodes[i], step, nodes[i + 1], g)
        elif isinstance(p, AlternativePath):
            if all(isinstance(a, URIRef) for a in p.args):
                self.edge(s, tuple(str(a) for a in p.args), o, g)
            elif all(isinstance(a, InvPath) and isinstance(a.arg, URIRef) for a in p.args):
                self.edge(o, tuple(str(a.arg) for a in p.args), s, g)
            else:
                self.taint([s, o], "complex_alternative_path")
                d = self.group("detached", g)
                for a in p.args:
                    self.path(s, a, o, d)
        elif isinstance(p, MulPath) and p.mod in ("*", "+") and _is_subclass_star(p, p.mod):
            # rdfs:subClassOf+ / *: read as membership through the closure (rule 19, F28a)
            self.edge(s, (RDFS_SUBCLASSOF,), o, g, closure=p.mod)
        elif isinstance(p, MulPath):
            self.taint([s, o], "zero_or_one_path" if p.mod == "?" else "unbounded_path")
            self.path(s, p.path, o, self.group("detached", g))
        elif isinstance(p, NegatedPath):
            self.taint([s, o], "negated_property_set")
            d = self.group("detached", g)
            for a in p.args:
                inner = a.arg if isinstance(a, InvPath) else a
                if isinstance(inner, URIRef):
                    self.edge(s, (str(inner),), o, d)
                    d.edges[-1].negated = True
        else:
            self.taint([s, o], type(p).__name__)

    # -- filters --------------------------------------------------------------

    def filter_expr(self, expr: Any, g: Group) -> None:
        self.exists_parts(expr, g)
        for left, right in _equalities(expr):
            for var, const in ((left, right), (right, left)):
                if isinstance(var, Variable) and isinstance(const, (URIRef, Literal)):
                    g.bindings.setdefault(var, set()).add(const)

    def exists_parts(self, expr: Any, g: Group) -> None:
        if isinstance(expr, CompValue):
            if expr.name == "Builtin_NOTEXISTS":
                self.negation = True
                self.walk(expr.graph, self.group("negated", g))
                return
            if expr.name == "Builtin_EXISTS":
                self.taint(expr.get("_vars"), "EXISTS")
                self.walk(expr.graph, self.group("detached", g))
                return
            for v in expr.values():
                self.exists_parts(v, g)
        elif isinstance(expr, (list, tuple)):
            for v in expr:
                self.exists_parts(v, g)


def _is_subclass_star(p: Any, mod: str) -> bool:
    return isinstance(p, MulPath) and p.mod == mod and p.path == URIRef(RDFS_SUBCLASSOF)


def _equalities(expr: Any) -> list[tuple[Any, Any]]:
    """(left, right) of each `=` and sameTerm in the top-level conjunction."""
    if not isinstance(expr, CompValue):
        return []
    if expr.name == "ConditionalAndExpression":
        out = _equalities(expr.expr)
        for o in expr.get("other") or []:
            out += _equalities(o)
        return out
    if expr.name == "RelationalExpression" and expr.get("op") == "=":
        return [(expr.expr, expr.other)]
    if expr.name == "Builtin_sameTerm":
        return [(expr.arg1, expr.arg2)]
    return []


def analyse(text: str) -> QueryAnalysis:
    # rdflib's grammar is shared by the process and not thread-safe (harness/sparql_grammar.py, dress rehearsal F2):
    # the whole analysis holds the lock, the parse tree and the algebra read included; it is pure Python under the GIL
    with PARSER_LOCK:
        return _analyse(text)


def _analyse(text: str) -> QueryAnalysis:
    try:
        tree = parse_query(text)
        algebra = translate_query(tree).algebra
    except Exception as exc:  # rdflib raises several exception types on bad input
        return QueryAnalysis(text, None, [], [], set(), set(), False, False, error=f"{type(exc).__name__}: {exc}")
    b = _Builder()
    root = b.group("required", None)
    b.walk(algebra, root)
    if algebra.name != "SelectQuery":  # the frozen fragment is SELECT only
        b.unsupported.add(algebra.name.replace("Query", "").upper())
    constructs, bind_vars = _parse_tree_constructs(tree)
    b.unsupported |= constructs
    b.tainted |= bind_vars
    projected = tuple(sorted(str(v) for v in algebra.get("PV") or ())) if algebra.name == "SelectQuery" else ()
    return QueryAnalysis(
        text, root, b.groups, sorted(b.unsupported), b.tainted, b.fresh, b.negation, b.aggregate, projected=projected
    )


# -- the footprint re-run -----------------------------------------------------------

_IRIREF = re.compile(r'<[^<>"{}|^`\\\x00-\x20]*>')
_PROLOGUE = re.compile(r"""\A(?:\s+|\#[^\n]*\n?|BASE\s*<[^>]*>|PREFIX\s+[^\s:]*:\s*<[^>]*>)*""", re.IGNORECASE)
_DATASET = re.compile(r"\bFROM\s+(?:NAMED\s+)?(?:<[^>]*>|[A-Za-z_][\w.-]*:[\w.-]*|:[\w.-]*)", re.IGNORECASE)
_VALUES = re.compile(r"\bVALUES\s+(?:[?$]\w+|\((?:\s*[?$]\w+)*\s*\))\s*\{", re.IGNORECASE)


def _code_braces(text: str, start: int = 0):
    """Yield (index, char) of each brace outside strings, IRIs and comments."""
    i, n = start, len(text)
    while i < n:
        c = text[i]
        if c == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if c == "<":
            m = _IRIREF.match(text, i)
            if m:
                i = m.end()
                continue
        if c in "'\"":
            quote = c * 3 if text.startswith(c * 3, i) else c
            j = i + len(quote)
            while j < n and not text.startswith(quote, j):
                j += 2 if text[j] == "\\" else 1
            i = j + len(quote)
            continue
        if c in "{}":
            yield i, c
        i += 1


def _matching(text: str, open_at: int) -> int:
    depth = 0
    for i, c in _code_braces(text, open_at):
        depth += 1 if c == "{" else -1
        if depth == 0:
            return i
    raise ValueError("unbalanced braces")


def _masked(text: str) -> str:
    """The text with comments and string literals blanked out, offsets kept."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "#":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
            continue
        if c == "<":
            m = _IRIREF.match(text, i)
            if m:
                i = m.end()
                continue
        if c in "'\"":
            quote = c * 3 if text.startswith(c * 3, i) else c
            j = i + len(quote)
            while j < n and not text.startswith(quote, j):
                j += 2 if text[j] == "\\" else 1
            j = min(j + len(quote), n)
            out[i:j] = " " * (j - i)
            i = j
            continue
        i += 1
    return "".join(out)


def query_parts(text: str) -> tuple[str, str, str, str]:
    """(prologue, dataset clause, WHERE group, trailing VALUES block)."""
    open_at = next(i for i, c in _code_braces(text) if c == "{")
    close_at = _matching(text, open_at)
    head, group, tail = text[:open_at], text[open_at : close_at + 1], text[close_at + 1 :]
    prologue = head[: _PROLOGUE.match(head).end()]
    dataset = " ".join(m.group(0) for m in _DATASET.finditer(_masked(head)))  # never a FROM inside a comment
    values = ""
    m = _VALUES.search(_masked(tail))
    if m:
        end = _matching(tail, m.end() - 1)
        values = " " + tail[m.start() : end + 1]
    return prologue, dataset, group, values


_ORDER = re.compile(r"\bORDER\s+BY\s+(.+?)(?=\s+(?:LIMIT|OFFSET|VALUES)\b|\s*$)", re.IGNORECASE | re.DOTALL)
_LIMIT = re.compile(r"\bLIMIT\s+(\d+)", re.IGNORECASE)
_OFFSET = re.compile(r"\bOFFSET\s+(\d+)", re.IGNORECASE)


def modifiers(text: str) -> tuple[str, int | None, int]:
    """(ORDER BY expressions, LIMIT, OFFSET) after the WHERE group."""
    open_at = next(i for i, c in _code_braces(text) if c == "{")
    tail = _masked(text[_matching(text, open_at) + 1 :])
    m = _VALUES.search(tail)
    if m:  # a trailing VALUES block is not a modifier
        tail = tail[: m.start()] + tail[_matching(tail, m.end() - 1) + 1 :]
    order = _ORDER.search(tail)
    limit = _LIMIT.search(tail)
    offset = _OFFSET.search(tail)
    return (order.group(1).strip() if order else "", int(limit.group(1)) if limit else None,
            int(offset.group(1)) if offset else 0)


def rerun_query(text: str, limit: int, offset: int = 0, order: str = "") -> str:
    """The query's pattern with all its variables projected (§7.2): the WHERE
    group under SELECT *, with the prologue, the dataset clause, GRAPH clauses
    and a trailing VALUES block kept; grouping, aggregates and HAVING dropped.
    ORDER BY, and the window ``limit``/``offset``, are the caller's (rule 19)."""
    prologue, dataset, group, values = query_parts(text)
    page = (f" ORDER BY {order}" if order else "") + f" LIMIT {limit}" + (f" OFFSET {offset}" if offset else "")
    # a trailing VALUES block follows the solution modifiers in SPARQL's grammar
    return f"{prologue}SELECT * {dataset + ' ' if dataset else ''}WHERE {group}{page}{values}"


def term_text(t: Term) -> str:
    return f"?{t}" if isinstance(t, Variable) else (f"<{t}>" if isinstance(t, URIRef) else t.n3())


def edges_text(edges: list[Edge], predvars: dict | None = None) -> str:
    """Triple patterns for an auxiliary query, inside their GRAPH clauses; an
    alternative edge uses its predicate variable from ``predvars``."""
    by_graph: dict = {}
    for e in edges:
        if e.pvar is not None:
            p = f"?{e.pvar}"
        elif len(e.preds) == 1:
            p = e.path_text()
        else:
            p = f"?{predvars[e]}"
        by_graph.setdefault(e.graph, []).append(f"{term_text(e.s)} {p} {term_text(e.o)} .")
    parts = []
    for gr, triples in by_graph.items():
        body = " ".join(triples)
        parts.append(body if gr is None else f"GRAPH {term_text(gr)} {{ {body} }}")
    return " ".join(parts)
