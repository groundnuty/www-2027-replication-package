"""Footprint checks: class, domain and range, and disjointness on the evidence
the query touched (§7.2).

1. Re-run the query's pattern with all its variables projected (for an
   aggregate query, the pattern without grouping and aggregates), page by page.
2. Instantiate the pattern's triples from the rows. What a row cannot show is
   retrieved with auxiliary queries restricted to the row's bindings:
   * the hidden nodes of fixed-length paths and blank nodes in the pattern;
   * which alternative of ``a|b`` matched, so its own declarations apply;
   * whether an OPTIONAL group with no variable of its own matched.
   When that cannot be recovered (a filtered or nested-negation OPTIONAL, a
   blank node as a binding) the affected checks are unknown.
3. Fetch the membership assertions of the bound IRIs: rdf:type, or
   rdfs:subClassOf parents when the graph models its entities as classes
   (Rhea). The lookups query the default graph, which on our stores is the
   distinct union of the named graphs (stores/check_store.py checks it), so they
   never take a GRAPH scope (ruling 19); the re-run keeps the GRAPH clause.
4. The bound counts evidence triples: distinct instantiated pattern triples plus
   membership triples, at most 10,000 per output. Over it, or when any response
   is cut by a limit or a row cap, every footprint check is unknown.
5. Load the evidence and the closure facts into a local graph and decide each
   check with SPARQL ASK queries over it.

Negated parts are outside the footprint (ruling 17).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdflib import BNode, Graph, Namespace, URIRef, Variable
from rdflib import Literal as RLiteral

from harness import engine_suspect
from harness.checks.closure import (META_TYPES, RDFS_SUBCLASSOF, VACUOUS_CLASSES, Closure, Requirement,
                                    accepted_datatypes)
from harness.checks.outcomes import CLOSED_WORLD_MISMATCH, LOGICAL_CLASH, NOT_AXIOM, CheckResult, Outcome, fault_kind
from harness.checks.hierarchy import RDF_TYPE, membership_query
from harness.checks.query import Edge, Group, QueryAnalysis, edges_text, modifiers, query_parts, rerun_query, term_text
from harness.checks.static import implied_for, is_data_node, is_type_pattern, membership_predicate, visible_graphs
from harness.sources.model import ClassExpr, OntologySource
from harness.sparql_grammar import graph_query
from harness.terms import IRI, BNode as TBNode, Literal as TLiteral

EV = Namespace("urn:www2027:checks:ev#")
FOOTPRINT_CHECKS = ("class.footprint", "domain_range.footprint", "disjointness.footprint")
FETCH_CHUNK = 200
NODE_SAMPLE = 50  # per-node verdicts kept in a result's evidence
MAX_ROWS_FACTOR = 10  # re-run rows read at most: this many times the bound


class OverBound(Exception):
    """The evidence exceeds the bound, or a response was cut short."""


class QueryFailed(Exception):
    """The store refused or failed an evidence query."""


@dataclass(eq=False)
class Position:
    pid: int
    check_id: str
    key: str
    node: Variable
    group: Group
    edge: Edge
    pred: str  # the predicate whose declaration this is (the membership predicate for a type pattern)
    req: Requirement
    tautological: bool = False  # rule 18: a required positive type pattern already guarantees it

    @property
    def iri(self) -> URIRef:
        return URIRef(f"urn:www2027:checks:pos/{self.pid}")


@dataclass(eq=False)
class Unit:
    """Edges whose instantiation a row cannot show: linked by hidden variables,
    or an alternative path."""

    group: Group
    edges: list[Edge]
    endpoints: list[Variable]  # variables the rows bind
    hidden: list[Variable]  # path intermediates and pattern blank nodes
    predvars: dict = field(default_factory=dict)  # alternative edge -> variable for the matched predicate


@dataclass
class Plan:
    positions: list[Position] = field(default_factory=list)
    disj_vars: dict = field(default_factory=dict)  # variable -> implied classes
    na_reason: dict[str, str] = field(default_factory=dict)  # check id -> reason when nothing to test
    units: list[Unit] = field(default_factory=list)
    unit_of: dict = field(default_factory=dict)  # edge -> unit


def _edge_vars(e: Edge) -> set[Variable]:
    return {t for t in (e.s, e.o, e.pvar, e.graph) if isinstance(t, Variable)}


def plan(qa: QueryAnalysis, closure: Closure, source: OntologySource) -> Plan:
    member_pred = membership_predicate(source)
    p = Plan()
    vacuous = False
    unsupported = False
    # Rule 18: classes a required positive pattern asserts for each variable.
    asserted: dict = {}
    for e in qa.root.edges:
        if is_type_pattern(e, member_pred) and isinstance(e.s, Variable) and not is_data_node(closure, str(e.o)):
            asserted.setdefault(e.s, set()).add(str(e.o))  # rule 40: a data node asserts no class
    for g in qa.positive():
        comps: list[list[Edge]] = []
        for e in g.edges:
            hidden = {v for v in _edge_vars(e) if v in qa.fresh}
            if not hidden and (len(e.preds) <= 1 or e.pvar is not None):
                continue
            linked = [c for c in comps if hidden & {v for x in c for v in _edge_vars(x) if v in qa.fresh}]
            merged = [x for c in linked for x in c] + [e]
            comps = [c for c in comps if not any(c is l for l in linked)] + [merged]
        for comp in comps:
            allv = set().union(*(_edge_vars(e) for e in comp))
            unit = Unit(g, comp, sorted((v for v in allv if v not in qa.fresh), key=str),
                        sorted((v for v in allv if v in qa.fresh), key=str))
            for e in comp:
                if len(e.preds) > 1 and e.pvar is None:
                    unit.predvars[e] = Variable(f"_alt{len(p.units)}_{len(unit.predvars)}")
                p.unit_of[e] = unit
            p.units.append(unit)
        for e in g.edges:
            if is_type_pattern(e, member_pred):
                cls = str(e.o)
                # rule 40: a data node in the class position implies no class requirement
                if isinstance(e.s, Variable) and closure.known(cls) and not is_data_node(closure, cls):
                    p.positions.append(Position(len(p.positions), "class.footprint", f"?{e.s} a {cls}", e.s, g, e,
                                                member_pred, Requirement([ClassExpr.named(cls)]), True))
                continue
            for pred in e.preds:
                for node, side in ((e.s, "subject"), (e.o, "object")):
                    req = closure.requirement(pred, side)
                    vacuous = vacuous or req.vacuous
                    unsupported = unsupported or closure.unsupported_axiom(pred, side)
                    if req.any and isinstance(node, Variable):
                        taut = bool(req.exprs) and not (req.datatypes or req.enums or req.unsupported) and all(
                            any(closure.under(b, m) for b in asserted.get(node, ()) for m in x.members) for x in req.exprs)
                        p.positions.append(Position(len(p.positions), "domain_range.footprint", f"{side}-of {pred}",
                                                    node, g, e, pred, req, taut))
        for node, im in implied_for(g, closure, member_pred).items():
            if isinstance(node, Variable) and any(closure.touched(m) for x in im.exprs for m in x.members):
                entry = p.disj_vars.setdefault(node, {"implied": im.show(), "groups": [], "tautological": {},
                                                      "matched": set()})
                entry["groups"].append(g)
                # the classes the group's own scope asserts for the node: in a row
                # where the group matched, its type patterns hold by the match
                own = {str(e.o) for e in g.scope_edges() if is_type_pattern(e, member_pred) and e.s == node
                       and not is_data_node(closure, str(e.o))}
                entry["tautological"][g.gid] = all(
                    any(closure.under(b, m) for b in own for m in x.members) for x in im.exprs)
    if not any(x.check_id == "class.footprint" for x in p.positions):
        p.na_reason["class.footprint"] = "no_type_pattern"
    if not any(x.check_id == "domain_range.footprint" for x in p.positions):
        p.na_reason["domain_range.footprint"] = "unsupported_axiom" if unsupported else (
            "vacuous_axiom" if vacuous else "no_axiom")
    if not p.disj_vars:
        p.na_reason["disjointness.footprint"] = "no_disjointness"
    return p


def _to_rdflib(cell: Any):
    if isinstance(cell, IRI):
        return URIRef(cell.value)
    if isinstance(cell, TLiteral):
        return RLiteral(cell.lexical, lang=cell.lang, datatype=URIRef(cell.datatype) if cell.datatype else None)
    if isinstance(cell, TBNode):
        return BNode(cell.label)
    return None


def _node_id(n) -> str:
    return str(n) if isinstance(n, URIRef) else n.n3()


def _chunks(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _keys(pl: Plan, cid: str) -> list[str]:  # noqa: D103
    if cid == "disjointness.footprint":
        return [f"?{v}" for v in pl.disj_vars]
    return list(dict.fromkeys(x.key for x in pl.positions if x.check_id == cid))


class Budget:
    """Rule 27: one retrieval budget per output, 10,000 evidence triples shared
    by all online retrieval (constant types, footprint, provenance). The offline
    closure does not count."""

    def __init__(self, total: int):
        self.total = total
        self.used = 0

    @property
    def left(self) -> int:
        return max(self.total - self.used, 0)

    def take(self, n: int) -> None:
        self.used += n
        if self.used > self.total:
            raise OverBound("the output's retrieval budget")

    def exhaust(self) -> None:
        self.used = self.total + 1


def run(
    qa: QueryAnalysis,
    closure: Closure,
    source: OntologySource,
    client: Any,
    bound: int,
    hierarchy=None,
) -> tuple[list[CheckResult], dict]:
    """``closure``: the source's closure merged with the shared hierarchy (rule
    16, offline); ``bound``: what is left of the output's budget (rule 27)."""
    name = source.name
    ev_summary: dict[str, Any] = {}
    if engine_suspect.configured_suspect_classes():
        # design log (change-log entry 57): the engine-suspect stratum is the union of its parts, each recorded: here
        # the checked query and the footprint queries below (the runner records the agent's own queries); the
        # shape is the one the store is known to answer wrongly (harness/engine_suspect.py)
        ev_summary["engine_suspect"] = {"checked_query": engine_suspect.suspect(qa.text), "footprint": False}

    def every(outcome: Outcome, reason: str, pl: Plan | None) -> list[CheckResult]:
        """One result per planned check instance (or per check id without a
        plan), all with the same outcome; checks with nothing to test stay NA."""
        out = []
        for cid in FOOTPRINT_CHECKS:
            if pl is not None and cid in pl.na_reason:
                out.append(CheckResult(cid, name, Outcome.NOT_APPLICABLE, pl.na_reason[cid]))
            elif pl is not None:
                out += [CheckResult(cid, name, outcome, reason, k) for k in _keys(pl, cid)]
            else:
                out.append(CheckResult(cid, name, outcome, reason))
        return out

    if qa.unsupported:
        ev_summary["not_run"] = f"unsupported constructs: {', '.join(qa.unsupported)}"
        return every(Outcome.UNKNOWN, "unsupported_construct", None), ev_summary
    pl = plan(qa, closure, source)
    if len(pl.na_reason) == len(FOOTPRINT_CHECKS):
        return every(Outcome.NOT_APPLICABLE, "no_axiom", pl), ev_summary
    if client is None:
        return every(Outcome.UNKNOWN, "no_store", pl), ev_summary
    try:
        return _run(qa, closure, source, client, bound, pl, ev_summary, every, hierarchy), ev_summary
    except OverBound as exc:
        ev_summary["evidence_triples"] = f">{bound}"
        ev_summary["over_bound"] = str(exc)
        return every(Outcome.UNKNOWN, "bound", pl), ev_summary
    except QueryFailed as exc:
        ev_summary["query_error"] = str(exc)
        return every(Outcome.UNKNOWN, "query_failed", pl), ev_summary


def _instantiate(e: Edge, value) -> tuple | None:
    """The triple a row instantiates for a simple edge, or None (a closure path
    such as rdfs:subClassOf+ instantiates no single triple)."""
    if e.closure:
        return None
    s, o = value(e.s), value(e.o)
    p = value(e.pvar) if e.pvar is not None else (URIRef(e.preds[0]) if len(e.preds) == 1 else None)
    return (s, p, o) if None not in (s, p, o) else None


def _alt_values(edges: list[Edge], predvars: dict) -> str:
    return "".join(f" VALUES ?{predvars[e]} {{ {' '.join(f'<{p}>' for p in e.preds)} }}" for e in edges if e in predvars)


def _values_block(names: list[Variable], tuples) -> str:
    vals = " ".join("(" + " ".join(term_text(x) for x in tup) + ")" for tup in tuples)
    return f"VALUES ({' '.join(f'?{v}' for v in names)}) {{ {vals} }}"


def _in_values_order(tuples: set) -> list[tuple]:
    """Bound tuples in the order a VALUES block sends them: by their plain strings, as always, then, where those are
    equal (an IRI and a literal, or literals differing in datatype or language), by their term text. The plain strings
    alone left such ties in set iteration order, which follows PYTHONHASHSEED, so the query text, by which a recorded
    store log is served back, changed between processes."""
    return sorted(tuples, key=lambda t: (tuple(map(str, t)), tuple(map(term_text, t))))


def _run(qa, closure: Closure, source, client, bound: int, pl: Plan, ev_summary: dict, every,
         hierarchy=None) -> list[CheckResult]:
    name = source.name
    prologue, dataset, _, _ = query_parts(qa.text)
    dataset = dataset + " " if dataset else ""

    def ask_store(q: str):
        parts = ev_summary.get("engine_suspect")
        if parts is not None and not parts["footprint"] and engine_suspect.suspect(q):
            parts["footprint"] = True
        try:
            r = client.query(q)
        except Exception as exc:
            raise QueryFailed(str(exc)[:500]) from exc
        # an answer that arrived before the client's cut is complete (change-log entry 57); possibly_partial is
        # only described, in the report's store_queries evidence
        if getattr(r, "truncated", False):
            raise OverBound("a response was cut by the client's row cap")
        return r

    def row_value(row, col):
        def value(term):
            if isinstance(term, Variable):
                i = col.get(str(term))
                return None if i is None else _to_rdflib(row[i])
            return term
        return value

    # 1. the re-run, page by page; the bound counts distinct instantiated triples
    page, max_rows = bound + 1, MAX_ROWS_FACTOR * bound
    rows: list = []
    columns: list[str] = []
    root = next(g for g in qa.positive() if g.kind == "required")
    simple_root = [e for e in root.edges if e not in pl.unit_of]
    required_triples: set = set()
    # Rule 19: an aggregate query loses grouping, aggregates and HAVING; any
    # other keeps ORDER BY, LIMIT and OFFSET, so the evidence matches its rows.
    order, qlimit, qoffset = ("", None, 0) if qa.aggregate else modifiers(qa.text)
    pages = 0
    while True:
        start = page * pages
        if qlimit is not None and start >= qlimit:
            break
        lim = page if qlimit is None else min(page, qlimit - start)
        q = rerun_query(qa.text, lim, qoffset + start, order)
        if pages == 0:
            ev_summary["rerun_query"] = q
        r = ask_store(q)
        pages += 1
        columns = r.columns or columns
        col = {c: i for i, c in enumerate(columns)}
        rows.extend(r.rows)
        for row in r.rows:
            value = row_value(row, col)
            for e in simple_root:
                t = _instantiate(e, value)
                if t:
                    required_triples.add(t)
        if len(required_triples) > bound:
            raise OverBound("instantiated pattern triples")
        if len(r.rows) < lim:
            break
        if len(rows) > max_rows:
            raise OverBound(f"more than {max_rows} re-run rows")
    ev_summary.update(rerun_rows=len(rows), rerun_pages=pages)
    if not rows:
        ev_summary["evidence_triples"] = 0
        return every(Outcome.NOT_APPLICABLE, "empty_result", pl)
    col = {c: i for i, c in enumerate(columns)}
    values = [row_value(row, col) for row in rows]
    all_predvars = {e: v for u in pl.units for e, v in u.predvars.items()}

    # 2a. OPTIONAL groups with no variable of their own: did they match?
    opt_vars: dict[int, list[Variable]] = {}
    opt_found: dict[int, set | None] = {}
    memos = [dict() for _ in rows]

    def matched(i: int, g: Group):
        """True / False, or None when the match cannot be determined."""
        memo = memos[i]
        if g.gid in memo:
            return memo[g.gid]
        if g.kind == "required":
            res = True
        else:
            res = matched(i, g.parent)
            if res is True:
                local = qa.local_vars(g)
                if local:
                    res = any(values[i](v) is not None for v in local)
                elif opt_found.get(g.gid) is None:
                    res = None
                else:
                    res = tuple(values[i](v) for v in opt_vars[g.gid]) in opt_found[g.gid]
        memo[g.gid] = res
        return res

    opt_queries = 0
    for g in qa.positive():
        if g.kind != "optional" or qa.local_vars(g):
            continue
        vs = sorted(set().union(set(), *(_edge_vars(e) for e in g.edges)) - qa.fresh, key=str)
        opt_vars[g.gid] = vs
        if g.filtered or g.bindings or not g.edges or any(c.kind in ("negated", "detached") for c in g.children):
            opt_found[g.gid] = None  # its match depends on more than its triple patterns
            continue
        tuples, undetermined = set(), False
        for i in range(len(rows)):
            if matched(i, g.parent) is not True:
                continue
            tup = tuple(values[i](v) for v in vs)
            if any(x is None or isinstance(x, BNode) for x in tup):
                undetermined = True
            else:
                tuples.add(tup)
        if undetermined:
            opt_found[g.gid] = None
            continue
        found: set = set()
        body = edges_text(g.edges, all_predvars) + _alt_values(g.edges, all_predvars)
        if not vs:
            opt_queries += 1
            if ask_store(f"{prologue}SELECT * {dataset}WHERE {{ {body} }} LIMIT 1").rows:
                found.add(())
        else:
            names = " ".join(f"?{v}" for v in vs)
            for chunk in _chunks(_in_values_order(tuples), FETCH_CHUNK):
                q = f"{prologue}SELECT DISTINCT {names} {dataset}WHERE {{ {_values_block(vs, chunk)} {body} }}"
                opt_queries += 1
                ev_summary.setdefault("optional_query", q)
                found |= {tuple(_to_rdflib(c) for c in r2) for r2 in ask_store(q).rows}
        opt_found[g.gid] = found
    if opt_queries:
        ev_summary["optional_queries"] = opt_queries

    # 2b. instantiate the simple edges and the positions a row shows
    positive = qa.positive()
    triples: set = set()
    occupancy: dict[int, set] = {x.pid: set() for x in pl.positions}
    witness: dict[int, dict] = {x.pid: {} for x in pl.positions}  # node -> triples that put it there
    forced: dict[int, str] = {}
    bound_to: dict[Variable, set] = {v: set() for v in pl.disj_vars}
    simple_pos = [x for x in pl.positions if x.edge not in pl.unit_of]
    for i in range(len(rows)):
        state = {g.gid: matched(i, g) for g in positive}
        for g in positive:
            if state[g.gid] is True:
                for e in g.edges:
                    if e not in pl.unit_of:
                        t = _instantiate(e, values[i])
                        if t:
                            triples.add(t)
        for x in simple_pos:
            st = state[x.group.gid]
            if st is None:
                forced.setdefault(x.pid, "optional_undetermined")
            elif st:
                v = values[i](x.node)
                if v is not None:
                    occupancy[x.pid].add(v)
                    t = _instantiate(x.edge, values[i])
                    if t:
                        witness[x.pid].setdefault(v, set()).add(t)
        for var in bound_to:
            groups = [g for g in pl.disj_vars[var]["groups"] if state.get(g.gid) is True]
            if var not in qa.fresh and groups:
                v = values[i](var)
                if v is not None and not isinstance(v, RLiteral):
                    bound_to[var].add(v)
                    pl.disj_vars[var]["matched"].update(g.gid for g in groups)
    if len(triples) > bound:
        raise OverBound("instantiated pattern triples")

    # 2c. units: hidden path nodes and matched alternatives, from the rows' bindings
    by_edge: dict = {}
    for x in pl.positions:
        by_edge.setdefault(x.edge, []).append(x)
    path_queries = 0
    for unit in pl.units:
        unit_pos = [x for e in unit.edges for x in by_edge.get(e, [])]
        tuples, any_match = set(), False
        for i in range(len(rows)):
            st = matched(i, unit.group)
            if st is None:
                for x in unit_pos:
                    forced.setdefault(x.pid, "optional_undetermined")
                continue
            if not st:
                continue
            any_match = True
            tup = tuple(values[i](v) for v in unit.endpoints)
            if any(t is None for t in tup):
                continue
            if any(isinstance(t, BNode) for t in tup):  # a blank node cannot be sent back to the store
                for x in unit_pos:
                    forced.setdefault(x.pid, "blank_node")
                continue
            tuples.add(tup)
        if not tuples and not (any_match and not unit.endpoints):
            continue
        selected = unit.endpoints + unit.hidden + list(unit.predvars.values())
        names = " ".join(f"?{v}" for v in selected)
        body = edges_text(unit.edges, unit.predvars) + _alt_values(unit.edges, unit.predvars)
        if unit.endpoints:
            queries = [f"{prologue}SELECT DISTINCT {names} {dataset}WHERE {{ {_values_block(unit.endpoints, chunk)} {body} }}"
                       for chunk in _chunks(_in_values_order(tuples), FETCH_CHUNK)]
        else:
            queries = [f"{prologue}SELECT DISTINCT {names} {dataset}WHERE {{ {body} }}"]
        for q in queries:
            budget = bound - len(triples)
            q = f"{q} LIMIT {max(budget, 0) + 1}"
            path_queries += 1
            ev_summary.setdefault("path_query", q)
            r = ask_store(q)
            if len(r.rows) > budget:
                raise OverBound("path and alternative instantiations")
            ucol = {c: j for j, c in enumerate(r.columns)}
            for r2 in r.rows:
                uval = row_value(r2, ucol)
                for e in unit.edges:
                    s, o = uval(e.s), uval(e.o)
                    if e.pvar is not None:
                        p = uval(e.pvar)
                    elif len(e.preds) == 1:
                        p = URIRef(e.preds[0])
                    else:
                        p = uval(unit.predvars[e])
                    if None not in (s, p, o):
                        triples.add((s, p, o))
                    for x in by_edge.get(e, []):
                        if p is not None and str(p) == x.pred:  # the alternative that matched
                            v = uval(x.node)
                            if v is not None:
                                occupancy[x.pid].add(v)
                                if None not in (s, p, o):
                                    witness[x.pid].setdefault(v, set()).add((s, p, o))
                for var in unit.hidden:
                    if var in bound_to:
                        v = uval(var)
                        if v is not None and not isinstance(v, RLiteral):
                            bound_to[var].add(v)
                            pl.disj_vars[var]["matched"].add(unit.group.gid)
            if len(triples) > bound:
                raise OverBound("instantiated pattern triples")
    if path_queries:
        ev_summary["path_queries"] = path_queries
    used = len(triples)

    # 3. membership assertions (rule 35: rdf:type objects, and rdfs:subClassOf objects
    # inside the data graph), within what is left of the bound
    member_pred = membership_predicate(source)
    membership = {RDF_TYPE, RDFS_SUBCLASSOF}
    nodes = sorted({n for vs in occupancy.values() for n in vs if isinstance(n, URIRef)}
                   | {n for vs in bound_to.values() for n in vs if isinstance(n, URIRef)})
    types: dict[URIRef, set[str]] = {n: set() for n in nodes}
    via: dict[tuple, str] = {}  # (node, class) -> the membership predicate that stated it
    fetched = 0
    queries = []
    for chunk in _chunks(nodes, FETCH_CHUNK):
        budget = bound - used - fetched
        chunk_set = set(chunk)
        already = sum(1 for s, p_, _ in triples if s in chunk_set and str(p_) in membership)
        budget += already  # membership triples the pattern instantiated come back again, counted once
        if hierarchy is not None:
            q = f"{membership_query([str(n) for n in chunk], hierarchy.roles)} LIMIT {max(budget, 0) + 1}"
        else:
            q = (f"SELECT ?s ?t ?p WHERE {{ VALUES ?s {{ {' '.join(f'<{n}>' for n in chunk)} }} "
                 f"?s <{member_pred}> ?t BIND(<{member_pred}> AS ?p) }} LIMIT {max(budget, 0) + 1}")
        queries.append(q)
        r = ask_store(q)
        rows = [(s, t, p) for s, t, p in r.rows if isinstance(s, IRI) and isinstance(t, IRI) and isinstance(p, IRI)]
        new = [x for x in rows if (URIRef(x[0].value), URIRef(x[2].value), URIRef(x[1].value)) not in triples]
        fetched += len(new)  # a membership triple the pattern already instantiated is counted once
        if used + fetched > bound or len(r.rows) > budget:
            ev_summary.update(membership_queries=len(queries), membership_query=queries[0])
            raise OverBound("membership triples")
        for s, t, p in rows:
            types.setdefault(URIRef(s.value), set()).add(t.value)
            via[(URIRef(s.value), t.value)] = p.value
    # Rule 16: membership is resolved through the closure merged offline with the
    # shared hierarchy. A class used as a value (an untyped node the hierarchy
    # knows as a class or as a subclass of something) is its own membership (F30).
    if hierarchy is not None:
        for n, ts in types.items():
            if not (ts - META_TYPES - VACUOUS_CLASSES) and (str(n) in hierarchy.parents or str(n) in hierarchy.classes):
                ts.add(str(n))
    ev_summary.update(
        instantiated_triples=used,
        membership_triples=fetched,
        evidence_triples=used + fetched,
        membership_queries=len(queries),
        membership_query=queries[0] if queries else None,
    )

    # Rule 4: a blank node is untyped unless the slice types it: its membership
    # triples among the instantiated pattern triples. Read in term order: a class
    # stated by both membership predicates is shown (``via``) with the last one
    # read, which in set iteration order would follow PYTHONHASHSEED.
    blank_members = [(s, p_, o) for s, p_, o in triples
                     if isinstance(s, BNode) and str(p_) in membership and isinstance(o, URIRef)]
    for s, p_, o in sorted(blank_members, key=lambda t: tuple(x.n3() for x in t)):
        types.setdefault(s, set()).add(str(o))
        via[(s, str(o))] = str(p_)

    # 4. the evidence graph and the ASK decisions
    g = Graph()
    # Rule 20 with change-log entry 61, position support (a reading): a node fits a census position it holds in a
    # hierarchy graph the edge can be matched in, as the store would match it there.
    positions_of = getattr(hierarchy, "positions", None) if source.kind == "census" else None
    if positions_of is not None:
        for x in pl.positions:
            if x.check_id != "domain_range.footprint":
                continue
            side, vis = x.key.partition("-of ")[0], visible_graphs(x.edge, hierarchy)
            for v in occupancy.get(x.pid, ()):
                if isinstance(v, URIRef) and positions_of.held(str(v), x.pred, side, vis):
                    g.add((v, EV.fits, x.iri))
    for n, ts in types.items():
        for t in ts:
            if t in META_TYPES or t in VACUOUS_CLASSES:
                continue
            g.add((n, EV.type, URIRef(t)))
            if closure.known(t):  # placed by the merged closure (rule 16)
                for a in closure.ancestors(t):
                    g.add((URIRef(t), EV.under, URIRef(a)))
    # The merged closure's pairs hold no data node (rule 40), so the disjointness decision below, whose ?a and ?b
    # come only from ev:disjoint, never reads one as a class; ev:type and ev:under keep membership through them.
    for pair in closure.disjoint_pairs:
        a, b = sorted(pair)
        g.add((URIRef(a), EV.disjoint, URIRef(b)))
        g.add((URIRef(b), EV.disjoint, URIRef(a)))
    for x in pl.positions:
        for k, expr in enumerate(x.req.exprs):
            entry = URIRef(f"{x.iri}/c{k}")
            g.add((x.iri, EV.entry, entry))
            for m in expr.members:
                g.add((entry, EV.requires, URIRef(m)))
        # each declared datatype range must hold (rule 2); a census's are alternatives, one entry
        dt_entries = [x.req.datatypes] if x.req.any_of and x.req.datatypes else [[d] for d in x.req.datatypes]
        for k, ds in enumerate(dt_entries):
            entry = URIRef(f"{x.iri}/d{k}")
            g.add((x.iri, EV.dtentry, entry))
            for d in ds:
                for a in accepted_datatypes(d):
                    g.add((entry, EV.accepts, URIRef(a)))
        if x.req.any_of:
            g.add((x.iri, EV.anyOf, EV.yes))
        for k, enum in enumerate(x.req.enums):
            entry = URIRef(f"{x.iri}/e{k}")
            g.add((x.iri, EV.enumentry, entry))
            for j, v in enumerate(enum):
                opt = URIRef(f"{entry}/{j}")
                g.add((entry, EV.option, opt))
                g.add((opt, EV.value, RLiteral(v["lexical"], datatype=URIRef(v["datatype"]))))
                for a in accepted_datatypes(v["datatype"]):
                    g.add((opt, EV.accepts, URIRef(a)))
        for v in occupancy.get(x.pid, ()):
            if isinstance(v, RLiteral):
                g.add((x.iri, EV.atLiteral, v))
            else:
                g.add((v, EV.at, x.iri))
    for var, vs in bound_to.items():
        for v in vs:
            g.add((v, EV.boundTo, URIRef(f"urn:www2027:checks:var/{var}")))

    results: list[CheckResult] = []
    keys: dict[tuple[str, str], list[Position]] = {}
    for x in pl.positions:
        keys.setdefault((x.check_id, x.key), []).append(x)
    axioms = source.kind != "census"  # rule 10's kind is of axioms; a census has none (its data faults: NOT_AXIOM)
    for (cid, key), poss in keys.items():
        results.append(_decide_positions(g, cid, key, poss, occupancy, types, closure, name, forced, witness, axioms))
    for var, entry in pl.disj_vars.items():
        results.append(_decide_disjointness(g, var, entry, bound_to.get(var, set()), name, via, axioms))
    for cid in FOOTPRINT_CHECKS:
        if cid in pl.na_reason:
            results.append(CheckResult(cid, name, Outcome.NOT_APPLICABLE, pl.na_reason[cid]))
    return results


_PREFIX = f"PREFIX ev: <{EV}>\n"


def _ask(g: Graph, body: str, values: str) -> bool:
    return bool(graph_query(g, f"{_PREFIX}ASK {{ VALUES ?pos {{ {values} }} {body} }}").askAnswer)


def _select(g: Graph, body: str, values: str, var: str = "?n") -> set:
    return {r[0] for r in graph_query(g, f"{_PREFIX}SELECT DISTINCT {var} WHERE {{ VALUES ?pos {{ {values} }} {body} }}")}


_TYPED = "FILTER EXISTS { ?n ev:type ?t0 . ?t0 ev:under ?x0 }"
_FITS = "FILTER NOT EXISTS { ?n ev:fits ?pos }"  # entry 61: held in a visible hierarchy graph (census only)
_FLAG_CLASS = (f"?n ev:at ?pos . ?pos ev:entry ?e . {_TYPED} {_FITS} "
               "FILTER NOT EXISTS { ?n ev:type ?t . ?t ev:under ?c . ?e ev:requires ?c }")
# A census lists alternatives (rule 20, change-log entry 30): where it lists both classes and
# datatypes, a literal is judged by the datatypes alone and a resource by the classes alone.
_FLAG_LIT_AT_CLASS = "?pos ev:atLiteral ?n . ?pos ev:entry ?e . FILTER NOT EXISTS { ?pos ev:anyOf ev:yes ; ev:dtentry ?d0 }"
_FLAG_DATATYPE = "?pos ev:atLiteral ?n . ?pos ev:dtentry ?d . FILTER NOT EXISTS { ?d ev:accepts ?dt . FILTER(datatype(?n) = ?dt) }"
_FLAG_ENUM = ("?pos ev:atLiteral ?n . ?pos ev:enumentry ?en . FILTER NOT EXISTS { ?en ev:option ?o . ?o ev:value ?v . "
              "?o ev:accepts ?dt . FILTER(datatype(?n) = ?dt && ?n = ?v) }")
_FLAG_RES_AT_DT = ("?n ev:at ?pos . { ?pos ev:dtentry ?d } UNION { ?pos ev:enumentry ?d } "
                   "FILTER NOT EXISTS { ?pos ev:anyOf ev:yes ; ev:entry ?e0 }")
_UNTYPED = f"?n ev:at ?pos . ?pos ev:entry ?e . FILTER(isIRI(?n)) {_FITS} FILTER NOT EXISTS {{ ?n ev:type ?t }}"
_BLANK = "?n ev:at ?pos . ?pos ev:entry ?e . FILTER(isBlank(?n)) FILTER NOT EXISTS { ?n ev:type ?t }"
_FOREIGN = f"?n ev:at ?pos . ?pos ev:entry ?e . ?n ev:type ?t . {_FITS} FILTER NOT EXISTS {{ ?n ev:type ?t2 . ?t2 ev:under ?x }}"


def _n3(triple) -> str:
    return " ".join(t.n3() for t in triple)


def _decide_positions(g, cid, key, poss, occupancy, types, closure: Closure, source_name, forced, witness,
                      axioms: bool = True) -> CheckResult:
    values = " ".join(f"<{x.iri}>" for x in poss)
    flags = [
        ("outside_required_class", _FLAG_CLASS),
        ("literal_where_class_required", _FLAG_LIT_AT_CLASS),
        ("wrong_datatype", _FLAG_DATATYPE),
        ("value_not_in_enumeration", _FLAG_ENUM),
        ("resource_where_literal_required", _FLAG_RES_AT_DT),
    ]
    unknowns = [("untyped_node", _UNTYPED), ("blank_node", _BLANK), ("types_outside_source", _FOREIGN)]
    flag_hits = [(reason, body) for reason, body in flags if _ask(g, body, values)]
    unknown_hits = [(reason, body) for reason, body in unknowns if _ask(g, body, values)]
    forced_reasons = [forced[x.pid] for x in poss if x.pid in forced]
    verdicts: dict = {}
    for x in poss:
        for n in occupancy.get(x.pid, ()):
            verdicts[n] = "pass"
    for _, body in unknown_hits:
        for n in _select(g, body, values):
            verdicts[n] = "unknown"
    flagged_by: dict = {}  # node -> the flag reasons that selected it (rule 10's kind; never a verdict)
    for flag_reason, body in flag_hits:
        for n in _select(g, body, values):
            verdicts[n] = "flag"
            flagged_by.setdefault(n, set()).add(flag_reason)
    annotations: list[str] = []
    witnessed: list[str] = []
    kinds: set[str] = set()
    if not verdicts and not forced_reasons:  # e.g. only in an OPTIONAL group that matched in no row
        return CheckResult(cid, source_name, Outcome.NOT_APPLICABLE, "no_binding", key)
    if flag_hits:
        outcome, reason = Outcome.FLAG, flag_hits[0][0]
        # Rule 10: a data fault when the footprint witnesses a triple of the
        # graph that violates the axiom.
        witnessed = sorted({_n3(t) for x in poss for n, v in verdicts.items() if v == "flag"
                            for t in witness.get(x.pid, {}).get(n, ())})
        exprs = [e for x in poss for e in x.req.exprs]
        for n, v in verdicts.items():
            if v == "flag" and isinstance(n, URIRef):
                known = {t for t in types.get(n, ()) if closure.known(t)}
                failing = [e for e in exprs if not closure.falls_under(known, e)]
                if failing and all(not closure.disjoint_with_all(known, e) for e in failing):
                    annotations = ["open_world_accepts"]
                    break
        kinds = {_fault_kind(n, flagged_by[n], types, closure, exprs) for n, v in verdicts.items()
                 if v == "flag" and any(witness.get(x.pid, {}).get(n) for x in poss)}
    elif unknown_hits or forced_reasons:
        outcome, reason = Outcome.UNKNOWN, (unknown_hits[0][0] if unknown_hits else forced_reasons[0])
    else:
        outcome, reason = Outcome.PASS, "holds"
        if all(x.tautological for x in poss):
            # Rule 18: a required positive type pattern of the query asserts a
            # class under the required one, so on a store without reasoning the
            # check can only pass (always so for a type pattern's own class).
            reason = "tautological"
    counts = {k: sum(1 for v in verdicts.values() if v == k) for k in ("pass", "flag", "unknown")}
    shown = sorted(verdicts, key=lambda n: ({"flag": 0, "unknown": 1, "pass": 2}[verdicts[n]], _node_id(n)))[:NODE_SAMPLE]
    evidence = {
        "required": [r for x in poss for r in x.req.show()],
        "node_counts": counts,
        "nodes": {_node_id(n): verdicts[n] for n in shown},
    }
    if forced_reasons:
        evidence["undecided"] = sorted(set(forced_reasons))
    if witnessed:
        annotations = annotations + ["data_fault"]
        evidence["witnessed"] = witnessed[:20]
        evidence["data_fault_kind"] = fault_kind(kinds) if axioms else NOT_AXIOM
    return CheckResult(cid, source_name, outcome, reason, key, evidence=evidence, annotations=annotations)


def _fault_kind(n, reasons: set[str], types, closure: Closure, exprs) -> str:
    """Rule 10's kind of one witnessed flagged node (clarified 20 Oct, change-log entry 108). A node outside the
    required class is a closed-world mismatch unless one of the expressions it fails has every member disjoint with
    one of its asserted types: then the disjointness is witnessed, a logical clash. Every other flag reason (a wrong
    datatype, a value outside an enumeration, a literal where a class is required or a resource where a literal is) is
    a datatype clash, a logical clash: a literal is no individual, and an individual no literal."""
    if reasons != {"outside_required_class"}:
        return LOGICAL_CLASH
    known = {t for t in types.get(n, ()) if closure.known(t)}
    failing = [e for e in exprs if not closure.falls_under(known, e)]
    return LOGICAL_CLASH if any(closure.disjoint_with_all(known, e) for e in failing) else CLOSED_WORLD_MISMATCH


def _decide_disjointness(g, var, entry, nodes, source_name, via: dict, axioms: bool = True) -> CheckResult:
    implied = entry["implied"]
    v = f"<urn:www2027:checks:var/{var}>"
    flag = (f"{_PREFIX}ASK {{ ?n ev:boundTo {v} . ?n ev:type ?t1 . ?n ev:type ?t2 . "
            f"?t1 ev:under ?a . ?t2 ev:under ?b . ?a ev:disjoint ?b }}")
    untyped = f"{_PREFIX}ASK {{ ?n ev:boundTo {v} . FILTER(isIRI(?n)) FILTER NOT EXISTS {{ ?n ev:type ?t }} }}"
    blank = f"{_PREFIX}ASK {{ ?n ev:boundTo {v} . FILTER(isBlank(?n)) FILTER NOT EXISTS {{ ?n ev:type ?t }} }}"
    foreign = (f"{_PREFIX}ASK {{ ?n ev:boundTo {v} . ?n ev:type ?t . "
               f"FILTER NOT EXISTS {{ ?n ev:type ?t2 . ?t2 ev:under ?x }} }}")
    ev = {"implied": implied, "nodes": len(nodes)}
    if graph_query(g, flag).askAnswer:
        # ordered, so the sample (shown in retry feedback) is the same in every process and replay
        rows = graph_query(g, flag.replace("ASK {", "SELECT DISTINCT ?n ?t1 ?t2 WHERE {", 1)
                           + " ORDER BY ?n ?t1 ?t2 LIMIT 10")
        ev["witnessed"] = sorted({f"<{n}> <{via.get((n, str(t)), RDF_TYPE)}> <{t}>" for n, t1, t2 in rows for t in (t1, t2)})
        ev["data_fault_kind"] = LOGICAL_CLASH if axioms else NOT_AXIOM  # rule 10: a witnessed disjointness
        return CheckResult("disjointness.footprint", source_name, Outcome.FLAG, "disjoint_asserted_types", f"?{var}",
                           evidence=ev, annotations=["data_fault"])
    for reason, q in (("untyped_node", untyped), ("blank_node", blank), ("types_outside_source", foreign)):
        if graph_query(g, q).askAnswer:
            return CheckResult("disjointness.footprint", source_name, Outcome.UNKNOWN, reason, f"?{var}", evidence=ev)
    if not nodes:
        return CheckResult("disjointness.footprint", source_name, Outcome.NOT_APPLICABLE, "no_binding", f"?{var}", evidence=ev)
    # Rule 18 (as the contract's F22 reads it): when the variable's only class
    # constraints are its own type patterns, the pass is tautological.
    matched = entry["matched"] or set(entry["tautological"])
    reason = "tautological" if all(entry["tautological"].get(gid, False) for gid in matched) else "holds"
    return CheckResult("disjointness.footprint", source_name, Outcome.PASS, reason, f"?{var}", evidence=ev)
