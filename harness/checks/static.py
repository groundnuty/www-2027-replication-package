"""Static checks: read the parsed query alone, offline, with the closure (§7.2).

For each variable (and each constant used as a node) the checks collect the
classes implied by its type patterns and by the domain and range of the
predicates it occupies, and

* flag explicitly disjoint implied classes (``disjointness.static``);
* flag implied classes with no common named subclass in the closure
  (``no_common_subclass.static``, the switchable closed-world rule);
* flag a constant whose asserted types fall outside a class implied for its
  position (``class.static`` for type patterns, ``domain_range.static`` for
  predicate positions), including a literal of the wrong datatype (ruling 11);
* for a census source, flag a join whose predicate positions no class is
  supported at together, support counting upwards from the listed classes
  (``census_join.static``, rule 20 and change-log entry 30).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product

from rdflib import Literal as RLiteral
from rdflib import URIRef, Variable

from harness.checks.closure import (RDF_TYPE, RDFS_SUBCLASSOF, VACUOUS_CLASSES, XSD, Closure, Requirement,
                                    SearchCapExceeded, accepted_datatypes, is_builtin, is_datatype)
from harness.checks.numbers import close, number_of
from harness.checks.outcomes import CheckResult, Outcome, combine
from harness.checks.query import Edge, Group, QueryAnalysis, _show
from harness.sources.model import ClassExpr, OntologySource
from harness.terms import Literal as TLiteral

TypesOf = Callable[[list[str]], "dict[str, set[str]] | None"]


@dataclass
class Implied:
    exprs: list[ClassExpr] = field(default_factory=list)
    origins: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # classes outside the source, left to vocabulary existence
    entities: list[str] = field(default_factory=list)  # data nodes in class positions (rule 40): no requirement

    def add(self, expr: ClassExpr, origin: str, closure: Closure) -> None:
        # Rule 40 (change-log entry 49): a data node in a class position implies no class requirement, so an
        # expression naming one (alone, or as a member of a union, which it then leaves unconstrained) adds nothing.
        nodes = [m for m in expr.members if is_data_node(closure, m)]
        if nodes:
            self.entities += [m for m in nodes if m not in self.entities]
            return
        members = tuple(m for m in expr.members if closure.known(m))
        self.dropped += [m for m in expr.members if not closure.known(m)]
        if not members:
            return
        e = ClassExpr(members)
        if all(set(x.members) != set(members) for x in self.exprs):
            self.exprs.append(e)
            self.origins.append(origin)

    def show(self) -> list[str]:
        return ["|".join(e.members) for e in self.exprs]


def is_data_node(closure: Closure, c: str) -> bool:
    """Rule 32's data node, as the compiler's closure knows it (a closure built without the store knows none)."""
    return c in getattr(closure, "data_nodes", ())


# Change-log entry 70 (design ruling in the design log, rule 20): a position realised in a shared-hierarchy graph is
# supported by one hierarchy node realising it ("existence"); a data-graph position needs the census listing at the
# threshold. "threshold" applies the census threshold to hierarchy nodes as well (the reading not ruled), kept so the
# ruling stays a one-line change.
HIERARCHY_JOIN_EVIDENCE = "existence"
# Where the hierarchy route of a typed join reaches the hierarchy nodes realising a position (external reviewer 6 review of
# a01d0d7, item 2; ratified, entry 72). "position": through the subclass edges of the position's own graphs and of
# the graphs the type clause sees, the type clause needing members of its class in its own graphs: in the data graph
# the census listing at the threshold, in a hierarchy graph one node (entry 70's per-position rule; a type clause
# confined to the data graph beside a position realised in a hierarchy graph). "type"
# (round 7): through the type clause's graphs only, so a type clause confined to the data graph reaches no hierarchy
# node; the census route's own reading (entry 72, reading (b)) mirrored.
HIERARCHY_ROUTE_ANCESTRY = "position"


def visible_graphs(e: Edge, hierarchy) -> tuple[str, ...]:
    """Rule 20 with change-log entries 49 and 61: the hierarchy graphs in which the store can match an edge. Every
    one, unless a GRAPH clause names a graph: then that graph if it is a hierarchy graph, and none otherwise."""
    if hierarchy is None:
        return ()
    gs = tuple(hierarchy.roles.hierarchy_graphs)
    if e.graph is None or isinstance(e.graph, Variable):
        return gs
    return (str(e.graph),) if str(e.graph) in gs else ()


def sees_data(e: Edge, hierarchy) -> bool:
    """Whether the store can match the edge in the data graph, the graph the census describes: no GRAPH clause,
    GRAPH ?g, or GRAPH <data graph>. A pattern confined to another graph is matched by that graph's triples only, so
    the census's counts are no evidence for it (External reviews of 7e81a1e). Without a hierarchy (no graph roles) the
    census alone answers, as before."""
    if hierarchy is None or e.graph is None or isinstance(e.graph, Variable):
        return True
    return str(e.graph) == hierarchy.roles.data_graph


def _hierarchy_need(threshold: int) -> int:
    """How many hierarchy nodes must realise a position for a join's class (change-log entry 70)."""
    if HIERARCHY_JOIN_EVIDENCE == "existence":
        return 1
    if HIERARCHY_JOIN_EVIDENCE == "threshold":
        return threshold
    raise ValueError(f"unknown HIERARCHY_JOIN_EVIDENCE {HIERARCHY_JOIN_EVIDENCE!r}")


def membership_predicate(source: OntologySource) -> str:
    return "http://www.w3.org/2000/01/rdf-schema#subClassOf" if source.membership == "rdfs:subClassOf" else (
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    )


def is_type_pattern(e: Edge, member_pred: str) -> bool:
    """A type pattern under the source's membership; a variable class implies
    nothing (rule 6). rdfs:subClassOf+ / * on a subClassOf graph, and
    rdf:type/rdfs:subClassOf* on an rdf:type graph, are membership through the
    closure (rule 19)."""
    if not isinstance(e.o, URIRef) or e.preds != (member_pred,):
        return False
    if e.closure is None:
        return True
    return (e.closure == "type*") == (member_pred.endswith("#type"))


def position_requirement(closure: Closure, preds: tuple[str, ...], side: str) -> Requirement:
    """What the pattern statically requires on one side of an edge.

    For a|b the node meets the requirement of whichever alternative matched,
    which the static form cannot know: the union of their single class
    expressions when every alternative has one, and nothing otherwise. The
    footprint form reads the matched alternative instead.
    """
    if len(preds) == 1:
        return closure.requirement(preds[0], side)
    branches = []
    for p in preds:
        req = closure.requirement(p, side)
        if not req.exprs or req.datatypes or req.enums:
            return Requirement()  # some alternative requires no class: nothing is implied
        branches.append(req.exprs)
    # (E11 and E12) or (E21) ... = and over every choice of one entry per branch of their union
    exprs = []
    for pick in product(*branches):
        members = tuple(dict.fromkeys(m for e in pick for m in e.members))
        if all(set(members) != set(x.members) for x in exprs):
            exprs.append(ClassExpr(members))
    return Requirement(exprs)


def implied_for(scope: Group, closure: Closure, member_pred: str) -> dict:
    """Classes implied for each node of the scope."""
    out: dict = {}
    for e in scope.scope_edges():
        if is_type_pattern(e, member_pred):
            out.setdefault(e.s, Implied()).add(ClassExpr.named(str(e.o)), "type pattern", closure)
            continue
        if not e.preds:
            continue
        for node, side in ((e.s, "subject"), (e.o, "object")):
            for x in position_requirement(closure, e.preds, side).exprs:
                label = "domain" if side == "subject" else "range"
                out.setdefault(node, Implied()).add(x, f"{label} of {'|'.join(e.preds)}", closure)
    return out


def _key(node) -> str:
    return f"?{node}" if isinstance(node, Variable) else str(node)


def _verdicts(imp: Implied, closure: Closure) -> dict[str, tuple[str, str]]:
    """(outcome, reason) of each consistency check on one scope's implications.

    One implication is a pass the query itself guarantees (rule 18 static):
    "tautological" when it is the node's own type pattern, "vacuous" otherwise.
    """
    n = len(imp.exprs)
    if n == 0:
        return {"d": ("not_applicable", "no_axiom"), "n": ("not_applicable", "no_axiom")}
    if n == 1:
        r = "tautological" if imp.origins[0] == "type pattern" else "vacuous"
        return {"d": ("pass", r), "n": ("pass", r)}
    if any(closure.touched(m) for e in imp.exprs for m in e.members):
        disj = closure.exprs_disjoint(imp.exprs)
        if disj is None:  # too many combinations of union members to decide
            d = ("unknown", "bound")
        else:
            d = ("flag", "explicit_disjointness") if disj else ("pass", "holds")
    else:
        d = ("not_applicable", "no_disjointness")
    try:
        cns = closure.common_named_subclass(imp.exprs)
    except SearchCapExceeded as exc:  # the search's memory cap: undecided, so unknown, never a pass or a flag
        return {"d": d, "n": ("unknown", "bound"), "cap": exc.cap}
    return {"d": d, "n": ("pass", "holds") if cns else ("flag", "no_common_named_subclass")}


def _scope_label(scope: Group) -> str:
    """Rule 28: the part of the query a static check read."""
    if scope.scope_groups()[-1].kind == "negated":
        return "negated"
    return "optional" if scope.kind == "optional" else "required"


def _main_rank(outcome: str, reason: str, root: bool) -> int:
    if outcome == "pass" and reason == "holds":
        return 4
    if outcome == "pass":
        return 3 if root else 2
    return 0


def consistency_checks(qa: QueryAnalysis, closure: Closure, source: OntologySource) -> list[CheckResult]:
    """disjointness.static and no_common_subclass.static, per node.

    The required part decides the main result. A conflict that appears only in
    an OPTIONAL group on a variable the required part shares, only across
    OPTIONAL groups, or anywhere in a negated part (the pattern can never
    match), is a separate branch-local flag (rules 7, 17 and 23). A conflict on
    variables that appear only inside an OPTIONAL group is reported without a
    flag (rule 7, F27b).
    """
    member_pred = membership_predicate(source)
    root = qa.root
    root_nodes = {t for e in root.edges for t in (e.s, e.o)} | set(root.bindings)
    per_node: dict = {}  # node -> list of (scope, Implied)
    for scope in qa.scopes():
        for node, imp in implied_for(scope, closure, member_pred).items():
            if not isinstance(node, RLiteral):
                per_node.setdefault(node, []).append((scope, imp))
    results: list[CheckResult] = []
    for node, scoped in per_node.items():
        key = _key(node)
        tainted = node in qa.tainted
        for check, cid in (("d", "disjointness.static"), ("n", "no_common_subclass.static")):
            main: tuple | None = None  # (rank, outcome, reason, evidence)
            branch_flag = None
            capped = None  # the evidence of a scope whose no-common-subclass search stopped at its memory cap
            for scope, imp in scoped:
                verdicts = _verdicts(imp, closure)
                outcome, reason = verdicts[check]
                ev = {"implied": imp.show(), "origins": imp.origins, "scope": _scope_label(scope)}
                if check == "n" and len(imp.exprs) >= 2:
                    if "cap" in verdicts:
                        ev["common_subclass_cap"] = verdicts["cap"]  # the cap, in hierarchy nodes held
                        capped = capped or ev
                    else:
                        ev["common_subclass"] = closure.common_named_subclass(imp.exprs)
                is_root = scope is root
                if outcome == "flag" and not is_root:
                    negated = scope.scope_groups()[-1].kind == "negated"
                    if negated or node in root_nodes:
                        branch_flag = branch_flag or (reason, ev)
                        continue
                    outcome, reason = "pass", "branch_only_conflict_reported"
                rank = 5 if outcome == "flag" else (1 if outcome == "unknown" else _main_rank(outcome, reason, is_root))
                if main is None or rank > main[0]:
                    main = (rank, outcome, reason, ev)
            if main is None:
                # Every scope of the node conflicts and every conflict is branch-local: the node's
                # implications all sit inside OPTIONAL or negated parts that can never match (rules
                # 7 and 17). The branch-local flag below reports it; with branch-local flags replayed
                # off (rule 23) nothing outside the branch is left to test.
                ev = dict(branch_flag[1])
                main = (0, "not_applicable", "only_in_branch", ev)
            _, outcome, reason, ev = main
            if outcome != "flag" and tainted:
                outcome, reason = "unknown", "unsupported_construct"
            if outcome != "flag" and branch_flag is None:
                across = _across_optionals(scoped, closure, check)
                if across is not None and "common_subclass_cap" in across:
                    capped = capped or across
                elif across is not None:
                    branch_flag = (_verdicts_reason(check), across)
            if capped is not None and outcome == "pass":
                # Fail closed: a part of the query whose search stopped at the cap might conflict, so a pass read
                # elsewhere does not stand for the node; a flag still does.
                outcome, reason, ev = "unknown", "bound", capped
            scope = ev.get("scope", "required") if ev.get("scope") in ("required", "optional", "negated") else "required"
            results.append(CheckResult(cid, source.name, Outcome(outcome), reason, key, evidence=ev, scope=scope))
            if branch_flag is not None:
                reason, ev = branch_flag
                results.append(CheckResult(cid, source.name, Outcome.FLAG, reason, key, evidence=ev, branch_local=True,
                                           scope=ev.get("scope", "optional") if ev.get("scope") in ("optional", "negated")
                                           else "optional"))
    return results


def _verdicts_reason(check: str) -> str:
    return "explicit_disjointness" if check == "d" else "no_common_named_subclass"


def _across_optionals(scoped, closure: Closure, check: str):
    """Rule 7: implications of the required part and every OPTIONAL group of
    the positive pattern (never one inside a negated part), merged; a conflict
    only there is a branch-local flag."""
    optional = [imp for sc, imp in scoped if sc.kind == "optional" and sc.scope_groups()[-1].kind == "required"]
    if len(optional) < 2:
        return None
    merged = Implied()
    for imp in [imp for sc, imp in scoped if sc.kind == "required"] + optional:
        for e, o in zip(imp.exprs, imp.origins):
            merged.add(e, o, closure)
    if len(merged.exprs) < 2:
        return None
    verdicts = _verdicts(merged, closure)
    ev = {"implied": merged.show(), "origins": merged.origins, "scope": "across optional groups"}
    if check == "n" and "cap" in verdicts:  # undecided: the search stopped at its memory cap
        return {**ev, "common_subclass_cap": verdicts["cap"]}
    if verdicts[check][0] != "flag":
        return None
    return ev


def detached_constant_checks(qa: QueryAnalysis, closure: Closure, source: OntologySource) -> list[CheckResult]:
    """Constants inside a construct outside the fragment (UNION, a path with a
    repetition, ...) are not judged: unknown, never a silent pass."""
    member_pred = membership_predicate(source)
    out: dict[str, CheckResult] = {}
    detached = [g for g in qa.groups if g.kind == "detached"]
    for g in detached:
        for e in g.edges:
            if is_type_pattern(e, member_pred) and isinstance(e.s, URIRef):
                key = f"{e.s} a {e.o}"
                out.setdefault(key, CheckResult("class.static", source.name, Outcome.UNKNOWN, "unsupported_construct", key))
                continue
            for node, side in ((e.s, "subject"), (e.o, "object")):
                if isinstance(node, URIRef) and e.preds and position_requirement(closure, e.preds, side).any:
                    key = f"{node} {side}-of {'|'.join(e.preds)}"
                    out.setdefault(key, CheckResult("domain_range.static", source.name, Outcome.UNKNOWN,
                                                    "unsupported_construct", key))
    return list(out.values())


def literal_position_checks(qa: QueryAnalysis, closure: Closure, source: OntologySource) -> list[CheckResult]:
    """F29: a variable that a datatype range makes a literal cannot also be a
    subject, or a node a class is required of."""
    member_pred = membership_predicate(source)
    results: dict[str, CheckResult] = {}
    for scope in qa.scopes():
        literal_vars, resource_vars = set(), set()
        for e in scope.scope_edges():
            if isinstance(e.s, Variable):
                resource_vars.add(e.s)
            if is_type_pattern(e, member_pred) or not e.preds:
                continue
            req = position_requirement(closure, e.preds, "object")
            if isinstance(e.o, Variable) and (req.datatypes or req.enums) and not req.exprs:
                literal_vars.add(e.o)
            if isinstance(e.o, Variable) and req.exprs:
                resource_vars.add(e.o)
        for v in sorted(literal_vars & resource_vars, key=str):  # by name: a set's order follows PYTHONHASHSEED
            key = _key(v)
            if key in results and not results[key].branch_local:
                continue
            results[key] = CheckResult("domain_range.static", source.name, Outcome.FLAG, "literal_in_resource_position",
                                       key, evidence={"scope": scope.kind}, branch_local=scope is not qa.root,
                                       scope="required" if scope is qa.root else scope.kind)
    return list(results.values())


# -- constants --------------------------------------------------------------------


class IncompleteEvidence(RuntimeError):
    """A type lookup hit its limit or the client's row cap: the types are incomplete."""


@dataclass
class ConstantUse:
    const: object  # URIRef or rdflib Literal
    check_id: str  # class.static or domain_range.static
    position: str  # "a <C>", "subject-of <p>", "object-of <p>"
    req: Requirement
    in_root: bool = False  # used in the required part (else only in an OPTIONAL or negated part)
    edges: list = field(default_factory=list)  # the edges it is used in (their GRAPH clauses, entry 61)
    edge_roots: list = field(default_factory=list)  # per edge: whether it lies in the required part


def constant_uses(qa: QueryAnalysis, closure: Closure, source: OntologySource) -> list[ConstantUse]:
    member_pred = membership_predicate(source)
    uses: dict[tuple, ConstantUse] = {}

    def add(const, edge: Edge, side: str | None) -> None:
        if side is None:  # type pattern: the constant must be a member of the class
            cls = str(edge.o)
            k = (const, "class.static", f"a {cls}")
            if cls in VACUOUS_CLASSES:  # rule 1: owl:Thing says nothing
                req = Requirement(vacuous=True)
            elif is_data_node(closure, cls) or not closure.known(cls):
                # rule 40: an entity constant in the class position requires nothing; a class the source cannot
                # place is vocabulary existence's business
                req = Requirement()
            else:
                req = Requirement([ClassExpr.named(cls)])
            u = uses.setdefault(k, ConstantUse(const, "class.static", f"a {cls}", req))
        elif not edge.preds:
            return
        else:
            pos = f"{side}-of {'|'.join(edge.preds)}"
            k = (const, "domain_range.static", pos)
            u = uses.setdefault(k, ConstantUse(const, "domain_range.static", pos, position_requirement(closure, edge.preds, side)))
        u.in_root = u.in_root or scope is qa.root
        u.edges.append(edge)
        u.edge_roots.append(scope is qa.root)

    for scope in qa.scopes():
        edges = scope.scope_edges()
        bindings = scope.scope_bindings()
        for e in edges:
            if is_type_pattern(e, member_pred):
                if isinstance(e.s, URIRef):
                    add(e.s, e, None)
                for c in bindings.get(e.s, ()):
                    add(c, e, None)
                continue
            for node, side in ((e.s, "subject"), (e.o, "object")):
                if isinstance(node, URIRef) or (isinstance(node, RLiteral) and side == "object"):
                    add(node, e, side)
                for c in bindings.get(node, ()) if isinstance(node, Variable) else ():
                    add(c, e, side)
    return list(uses.values())


def judge_types(types: set[str], exprs: list[ClassExpr], closure: Closure) -> tuple[str, str, list[str]]:
    """Verdict on a typed resource: ("pass"|"flag"|"unknown", reason, annotations).

    A node passes an expression if any asserted type falls under any member.
    Types outside the source are left out; if no type remains the verdict is
    unknown, since the source cannot place the node.
    """
    if not types:
        return "unknown", "untyped_node", []
    known = {t for t in types if closure.known(t)}
    if not known:
        return "unknown", "types_outside_source", []
    failing = [e for e in exprs if not closure.falls_under(known, e)]
    if not failing:
        return "pass", "holds", []
    open_world = all(not closure.disjoint_with_all(known, e) for e in failing)
    return "flag", "outside_required_class", ["open_world_accepts"] if open_world else []


def literal_datatype(lit: RLiteral) -> str | None:
    """A plain literal is an xsd:string (RDF 1.1); a language-tagged one has none here."""
    return str(lit.datatype) if lit.datatype else (None if lit.language else XSD + "string")


def enum_allows(lit: RLiteral, enum: list[dict]) -> bool:
    """Ruling 14: the literal has a listed value's datatype and equals it by value."""
    dt = literal_datatype(lit)
    for v in enum:
        if dt not in accepted_datatypes(v["datatype"]):
            continue
        a = number_of(TLiteral(str(lit), dt))
        b = number_of(TLiteral(v["lexical"], v["datatype"]))
        if (a is not None and b is not None and close(a, b)) or (a is None and str(lit) == v["lexical"]):
            return True
    return False


def judge_literal(lit: RLiteral, req: Requirement) -> tuple[str, str]:
    dt = literal_datatype(lit)
    if req.any_of and req.datatypes:  # a census: any listed datatype fits (rule 20, change-log entry 30)
        return ("pass", "holds") if any(dt in accepted_datatypes(d) for d in req.datatypes) else ("flag", "wrong_datatype")
    if req.exprs:
        return "flag", "literal_where_class_required"
    for d in req.datatypes:
        if dt not in accepted_datatypes(d):
            return "flag", "wrong_datatype"
    for en in req.enums:
        if not enum_allows(lit, en):
            return "flag", "value_not_in_enumeration"
    return "pass", "holds"


def constant_iris(qa: QueryAnalysis, closure: Closure, source: OntologySource) -> list[str]:
    """The constants whose asserted types a static check needs."""
    return sorted({str(u.const) for u in constant_uses(qa, closure, source) if isinstance(u.const, URIRef) and u.req.exprs})


def constant_checks(
    qa: QueryAnalysis, closure: Closure, source: OntologySource, fetched: dict | None, fetch_error: str | None = None,
    hierarchy=None,
) -> list[CheckResult]:
    """``fetched``: asserted types per constant (None when they could not be
    read: ``fetch_error`` is "bound", "no_store" or the store's error).
    ``hierarchy``: the shared hierarchy, whose positions count as census
    position support (rule 20, change-log entry 61)."""
    uses = constant_uses(qa, closure, source)
    positions = getattr(hierarchy, "positions", None) if source.kind == "census" else None
    results: list[CheckResult] = []
    for u in uses:
        key = f"{_show(u.const) if not isinstance(u.const, URIRef) else str(u.const)} {u.position}"
        if not u.req.any:
            continue  # no axiom for this position; reported once below
        ev: dict = {"required": u.req.show()}
        ann: list[str] = []
        if isinstance(u.const, RLiteral):
            verdict, reason = judge_literal(u.const, u.req)
        elif (u.req.datatypes or u.req.enums) and not u.req.exprs:
            verdict, reason = "flag", "resource_where_literal_required"
        elif fetched is None:
            verdict = "unknown"
            reason = fetch_error if fetch_error in ("bound", "no_store") else "query_failed"
        else:
            types = fetched.get(str(u.const), set())
            ev["asserted_types"] = sorted(types)
            verdict, reason, ann = judge_types(types, u.req.exprs, closure)
        if verdict != "pass" and positions is not None and isinstance(u.const, URIRef) and u.check_id == "domain_range.static":
            # Entry 61, position support (a reading): a node fits a census position it holds in a hierarchy graph the
            # pattern can be matched in, as the store would match it there. Each occurrence needs its own graph's
            # support (external reviewer 6.1 review of 7e81a1e, item 1): the required ones, or every one of an OPTIONAL-only use.
            side, _, pred = u.position.partition("-of ")
            occurrences = [e for e, root in zip(u.edges, u.edge_roots, strict=True) if root or not u.in_root]
            held_by = [positions.held(str(u.const), pred, side, visible_graphs(e, hierarchy)) for e in occurrences]
            if held_by and all(held_by):
                verdict, reason, ann = "pass", "holds", []
                ev["hierarchy_position"] = sorted({g for h in held_by for g in h})
        if u.req.unsupported:  # rule 13: the left-out conjunct is not checked (NA, unsupported_axiom)
            ev["unsupported_axiom"] = True
        results.append(CheckResult(u.check_id, source.name, Outcome(verdict), reason, key, evidence=ev, annotations=ann,
                                   branch_local=verdict == "flag" and not u.in_root,
                                   scope="required" if u.in_root else "optional"))
    member_pred = membership_predicate(source)
    edges = [e for sc in qa.scopes() for e in sc.edges]
    has_type_pattern = any(is_type_pattern(e, member_pred) for e in edges)
    reqs = [position_requirement(closure, e.preds, side) for e in edges if e.preds and not is_type_pattern(e, member_pred)
            for side in ("subject", "object")]
    for check_id in ("class.static", "domain_range.static"):
        if any(r.check_id == check_id for r in results):
            continue
        if check_id == "class.static":
            mine = [u for u in uses if u.check_id == "class.static"]
            if any(u.req.vacuous for u in mine):
                reason = "vacuous_axiom"
            elif mine:
                reason = "no_axiom"
            else:
                reason = "no_constant" if has_type_pattern else "no_type_pattern"
        elif any(r.any for r in reqs):
            reason = "no_constant"
        elif any(closure.unsupported_axiom(p, side) for e in edges for p in e.preds for side in ("subject", "object")):
            reason = "unsupported_axiom"  # rules 3 and 13
        else:  # §7.2: NA when the source declares no axiom for the predicates used
            reason = "vacuous_axiom" if any(r.vacuous for r in reqs) else "no_axiom"
        results.append(CheckResult(check_id, source.name, Outcome.NOT_APPLICABLE, reason))
    return results


# -- census -----------------------------------------------------------------------------


def _direct_count(source: OntologySource, pred: str, cls: str, data_nodes=frozenset()) -> int | None:
    """The census's direct count for ``?x pred cls`` (rule 20), or None when the
    census cannot tell. Only counts the census prompt and generator also see.

    * The graph's membership predicate: the census record's ``type_support``.
      Rule 33: the census lists every class the data uses directly, so a class
      it does not list has a direct count of 0. A meta class the census keeps
      as a literal pattern (owl:Class, G8b) counts that pattern's support. A
      data node in the class position (rh:Compound_3249, rule 32) is an entity
      constant, not a class, and the census holds no count for it: None.
    * The other membership predicate (rdf:type on Rhea): the census's literal
      patterns (``reading="literal"``); a missing pattern means 0.
    """
    if pred == membership_predicate(source):
        if getattr(source, "type_support", None) is None or cls in data_nodes:
            return None
        if cls in source.type_support:
            return source.type_support[cls]
        return sum(p.support for p in source.patterns
                   if getattr(p, "reading", "membership") == "literal" and p.predicate == pred and p.object_class == cls)
    if pred in (RDF_TYPE, RDFS_SUBCLASSOF):
        if getattr(source, "type_support", None) is None and not any(
                getattr(p, "reading", "membership") == "literal" for p in source.patterns):
            return None  # an older census form without literal patterns
        return sum(p.support for p in source.patterns
                   if getattr(p, "reading", "membership") == "literal" and p.predicate == pred and p.object_class == cls)
    return None


def census_pattern_checks(qa: QueryAnalysis, source: OntologySource, data_nodes=frozenset(),
                          hierarchy=None) -> list[CheckResult]:
    """Rule 20: a single pattern ``?x p C`` (rdf:type or rdfs:subClassOf with a
    constant class) is statically empty, ``census_empty_pattern``, only when its
    direct count is 0. The threshold does not apply (F23, F23b). Emptiness is
    judged against what the store would match, the data graph together with the
    hierarchy graphs (change-log entries 49 and 61): a pattern whose class is the
    object of ``p`` in a hierarchy graph (``hierarchy.positions``) is not empty.
    A GRAPH clause narrows that to the graph it names (External review of a929c39,
    item 3): the data graph has no hierarchy triple, and a hierarchy graph only its
    own; nor does the census's count, a data-graph count, speak for a pattern
    confined to a hierarchy graph (external reviewer 6 review of 7e81a1e, item 1). Confined to
    a graph neither describes, the census cannot tell: not applicable,
    ``graph_outside_census``."""
    name = source.name
    positions = getattr(hierarchy, "positions", None)
    if source.kind != "census":
        return [CheckResult("census_pattern.static", name, Outcome.NOT_APPLICABLE, "not_census")]
    results: dict[str, CheckResult] = {}
    for scope in qa.scopes():
        for e in scope.edges:
            if len(e.preds) != 1 or e.preds[0] not in (RDF_TYPE, RDFS_SUBCLASSOF) or not isinstance(e.o, URIRef) or e.closure:
                continue
            pred = e.preds[0]
            # one result per occurrence and its graph (external reviewer 6 and 6.1 reviews of a01d0d7, item 1)
            key = f"{_key(e.s)} {pred} {e.o}" + (f" in GRAPH {_key(e.graph)}" if e.graph is not None else "")
            data, vis = sees_data(e, hierarchy), visible_graphs(e, hierarchy)
            n = _direct_count(source, pred, str(e.o), data_nodes) if data else 0
            held = positions.held(str(e.o), pred, "object", vis) if n == 0 and positions else ()
            if not data and not vis:
                res = CheckResult("census_pattern.static", name, Outcome.NOT_APPLICABLE, "graph_outside_census", key)
            elif n is None:
                res = CheckResult("census_pattern.static", name, Outcome.NOT_APPLICABLE, "no_type_census", key)
            elif held:
                ev = {"direct_count": 0, "hierarchy_graphs": list(held)}
                if pred == RDFS_SUBCLASSOF:
                    ev["shared_hierarchy_subclasses"] = True
                res = CheckResult("census_pattern.static", name, Outcome.PASS, "holds", key, evidence=ev)
            elif n == 0:
                res = CheckResult("census_pattern.static", name, Outcome.FLAG, "census_empty_pattern", key,
                                  evidence={"direct_count": 0} if data else {"hierarchy_graphs_seen": list(vis)},
                                  branch_local=scope is not qa.root)
            else:
                res = CheckResult("census_pattern.static", name, Outcome.PASS, "holds", key, evidence={"direct_count": n})
            # The same pattern in the same graph twice: the worst required occurrence, whatever the clause order; a
            # branch-local flag only where no required occurrence decides.
            if key not in results or _pattern_rank(res) > _pattern_rank(results[key]):
                results[key] = res
    return list(results.values()) or [CheckResult("census_pattern.static", name, Outcome.NOT_APPLICABLE, "no_axiom")]


def _pattern_rank(r: CheckResult) -> int:
    if r.branch_local:
        return 0
    return {Outcome.FLAG: 3, Outcome.PASS: 2}.get(r.outcome, 1)


def _untyped_fillers(source: OntologySource) -> dict[tuple[str, str], int]:
    """(predicate, "subject"|"object") -> untyped fillers, from the census
    record's ``untyped_fillers`` (rendered in the census prompt, rule 20)."""
    out: dict[tuple[str, str], int] = {}
    for pred, sides in (getattr(source, "untyped_fillers", None) or {}).items():
        for side, n in sides.items():
            out[(pred, side)] = n
    return out


def _census_predicates(source: OntologySource) -> set[str]:
    """The predicates a census join reads positions of (rule 20)."""
    own = {p for p in source.properties if not is_builtin(p)}
    if getattr(source, "position_support", None):
        return set(source.position_support) | own
    return {p.predicate for p in source.patterns if getattr(p, "reading", "membership") == "membership"} | own


def graph_outside_census_count(qa: QueryAnalysis, source: OntologySource, hierarchy) -> int:
    """Change-log entry 72: the triple patterns a census source's checks cannot judge because they are confined to a
    graph neither the census (the data graph) nor the shared hierarchy describes: single patterns of rule 20
    (``graph_outside_census``) and the type patterns and census positions census joins leave out. Each pattern once;
    0 for any other source."""
    if source.kind != "census" or hierarchy is None:
        return 0
    member_pred, preds = membership_predicate(source), _census_predicates(source)
    seen: set[int] = set()
    for scope in qa.scopes():
        for e in [*scope.edges, *scope.scope_edges()]:
            if id(e) in seen or sees_data(e, hierarchy) or visible_graphs(e, hierarchy):
                continue
            single = len(e.preds) == 1 and not e.closure
            if (is_type_pattern(e, member_pred)
                    or (single and e.preds[0] in (RDF_TYPE, RDFS_SUBCLASSOF) and isinstance(e.o, URIRef))
                    or (single and e.preds[0] in preds)):
                seen.add(id(e))
    return len(seen)


def census_checks(qa: QueryAnalysis, closure: Closure, source: OntologySource, data_nodes=frozenset(),
                  hierarchy=None, cache: dict | None = None) -> list[CheckResult]:
    """Rule 20, joins (clarified by change-log entry 30): support is counted
    upwards, so a class is supported at a predicate position when some class
    listed there (at or over the threshold) is at or below it; lying below a
    listed class is no support (Mammal 6 does not support Dog with 3). A typed
    variable flags ``census_low_support`` unless each of its query classes is
    supported at each of its positions; an untyped one, unless some class is
    supported at all of its positions (two or more). Unknown
    (``untyped_fillers``) when a position lists no class but has enough
    untyped fillers. A necessary condition: it over- and under-flags.

    Change-log entry 61: support is judged against what the store would match,
    the data graph and the hierarchy graphs. A class is also supported at a
    position when some hierarchy node strictly under it holds that position in a
    hierarchy graph the edge can be matched in (``visible_graphs``). Entry 70
    (design ruling in the design log): a join is supported when some class satisfies
    every required position, each by its own graph's rule: a data-graph position
    needs the census listing at the threshold, a shared-hierarchy position one
    hierarchy node realising it (``HIERARCHY_JOIN_EVIDENCE``); the hierarchy is an
    alternative to the listing, never added to its counts. GRAPH-confined
    occurrences use evidence from their own graph only (External reviews of 7e81a1e):
    each occurrence of a position is judged on its own; the census speaks only
    for an occurrence the data graph can match, and only when the class's type
    pattern can be matched there too; a type pattern confined to hierarchy graphs
    reaches the class's members through those graphs' subclass edges alone; an
    occurrence confined to a graph neither describes is left out. ``cache`` keeps
    the hierarchy's supported classes per position across outputs (one source)."""
    pattern_results = census_pattern_checks(qa, source, data_nodes, hierarchy)
    # A census in the current form carries position_support; one without literal patterns (a store with no
    # rdf:type triple) still has its joins checked (External review of b0c3920, item 1: the Mammal fixture).
    if source.kind != "census" or not (source.patterns or getattr(source, "position_support", None)):
        return [CheckResult("census_join.static", source.name, Outcome.NOT_APPLICABLE, "not_census"), *pattern_results]
    threshold = source.support_threshold or 1
    membership = [p for p in source.patterns if getattr(p, "reading", "membership") == "membership"]
    position_support = getattr(source, "position_support", None)
    direct: dict[tuple[str, str], set[str]] = {}
    census_preds = _census_predicates(source)
    if position_support:  # exact per-position totals (rule 20)
        literal_objects = set()
        for pred, sides in position_support.items():
            for side, counts in sides.items():
                for c, n in counts.items():
                    if is_datatype(c):
                        continue
                    if n >= threshold:
                        direct.setdefault((pred, side), set()).add(c)
            objects = position_support[pred].get("object", {})
            if objects and all(is_datatype(c) for c in objects):
                literal_objects.add(pred)
        # An object position whose literal datatypes are unknown (design ruling in the design log Oct 05:24) is no
        # constraint: where it lists no class it is read as a literal position, as it would be with its datatypes
        # listed, so it never flags a join and never counts as coverage.
        for pred, sides in (getattr(source, "position_datatype", None) or {}).items():
            listed = position_support.get(pred, {}).get("object", {})
            if sides.get("object") == "unknown" and not any(not is_datatype(c) for c in listed):
                literal_objects.add(pred)
    else:  # an older census form: per-pattern supports
        literal_objects = {p.predicate for p in membership if p.object_class is None} - {
            p.predicate for p in membership if p.object_class is not None}
        for p in membership:
            if p.support >= threshold:
                direct.setdefault((p.predicate, "subject"), set()).add(p.subject_class)
                if p.object_class:
                    direct.setdefault((p.predicate, "object"), set()).add(p.object_class)
    untyped = _untyped_fillers(source)
    up: dict[tuple[str, str], frozenset[str]] = {}

    def supported(pos: tuple[str, str]) -> frozenset[str]:
        """The classes supported at a position: the listed ones and their ancestors (rule 33's
        frontier; ancestors through the merged closure). RDF, RDFS and OWL terms never count, nor
        does a data node (rule 40), listed or reached as an ancestor (External review of a929c39,
        item 1); support still counts upwards through one to its classes (rule 35)."""
        if pos not in up:
            up[pos] = frozenset(a for c in direct.get(pos, ()) for a in closure.ancestors(c)
                                if a not in VACUOUS_CLASSES and not is_builtin(a) and a not in data_nodes)
        return up[pos]

    positions_of = getattr(hierarchy, "positions", None)
    cache = {} if cache is None else cache
    need = _hierarchy_need(threshold)
    all_graphs = frozenset(getattr(getattr(hierarchy, "roles", None), "hierarchy_graphs", ()))

    def counted(c: str) -> bool:
        return c not in data_nodes and c not in VACUOUS_CLASSES and not is_builtin(c)

    def hierarchy_supported(pos: tuple[str, str], graphs: tuple[str, ...]) -> frozenset[str]:
        """Entry 61: the classes with as many hierarchy nodes strictly under them (membership through the merged
        closure) holding the position in ``graphs`` as entry 70 needs; never a data node (rule 40), nor a W3C term."""
        if positions_of is None or not graphs:
            return frozenset()
        k = ("supported", pos, graphs, need)
        if k not in cache:
            occupants = [n for n in positions_of.occupants(pos[0], pos[1], graphs) if n not in data_nodes]
            if need == 1:
                above = closure.strictly_above(occupants)
            else:
                counts: Counter[str] = Counter()
                for n in occupants:
                    counts.update(closure.ancestors(n))
                    counts[n] -= 1  # strictly under: a node does not count for itself
                above = {a for a, m in counts.items() if m >= need}
            cache[k] = frozenset(a for a in above if counted(a))
        return cache[k]

    def confined_count(c: str, type_graphs: tuple[str, ...], pos: tuple[str, str], graphs: tuple[str, ...]) -> int:
        """For a type pattern confined to hierarchy graphs: the hierarchy nodes strictly under ``c`` through the
        subclass edges those graphs hold (External reviews of 7e81a1e: ancestry from the same graph) that hold the
        position in ``graphs``, counted up to the number needed."""
        k = ("confined", c, type_graphs, pos, graphs)
        if k not in cache:
            m = 0
            for n in hierarchy.descendants_within(c, type_graphs):
                if n not in data_nodes and positions_of.held(n, pos[0], pos[1], graphs):
                    m += 1
                    if m >= need:
                        break
            cache[k] = m
        return cache[k]

    def by_census(c: str, pos, occ, type_occs) -> bool:
        """The data-graph rule: the occurrence, and the class's type patterns, can be matched in the data graph, and
        the census lists a class at or below ``c`` there at the threshold."""
        return occ[0] and all(t[0] for t in type_occs) and c in supported(pos)

    data_members: frozenset[str] | None = None

    def has_members(c: str, t) -> bool:
        """The type clause's evidence in its own graphs. In the data graph it is data-graph evidence, so it needs the
        census listing at the threshold (entry 72's condition): a class at or below ``c`` with at least ``threshold``
        members (``class_support``, else ``type_support``), counted upwards. In a hierarchy graph, existence: a node
        typed ``c`` or a subclass of it there."""
        nonlocal data_members
        if t[0]:
            if data_members is None:
                counts = source.class_support if source.class_support is not None else (source.type_support or {})
                data_members = frozenset(a for k, m in counts.items() if m >= threshold for a in closure.ancestors(k)
                                         if counted(a))
            if c in data_members:
                return True
        return bool(t[1]) and positions_of is not None and any(
            positions_of.held(c, pred, "object", t[1]) for pred in {member_pred, RDFS_SUBCLASSOF})

    def by_hierarchy(c: str, pos, occ, type_occs) -> bool:
        """The shared-hierarchy rule (entry 70): hierarchy nodes under ``c`` realise the position in the graphs the
        occurrence sees, as many as needed. Under a type pattern that sees the data graph and every hierarchy graph
        they are reached through the merged closure; under a confined one, through the subclass edges of its graphs
        and, by default (``HIERARCHY_ROUTE_ANCESTRY``), of the occurrence's own, the type pattern then needing
        members of ``c`` in its own graphs."""
        if positions_of is None or not occ[1] or not type_occs:
            return False
        for t in type_occs:
            if t[0] and frozenset(t[1]) == all_graphs:
                if c not in hierarchy_supported(pos, occ[1]):
                    return False
                continue
            if HIERARCHY_ROUTE_ANCESTRY == "type":
                graphs = t[1]
            elif HIERARCHY_ROUTE_ANCESTRY == "position":
                if not has_members(c, t):
                    return False
                graphs = tuple(sorted(set(t[1]) | set(occ[1])))
            else:
                raise ValueError(f"unknown HIERARCHY_ROUTE_ANCESTRY {HIERARCHY_ROUTE_ANCESTRY!r}")
            if not graphs or confined_count(c, graphs, pos, occ[1]) < need:
                return False
        return True

    def untyped_classes(pos, occ) -> tuple[frozenset[str], frozenset[str]]:
        """For an untyped node: the classes the census supports at the position for this occurrence, and those the
        census or the hierarchy does."""
        census = supported(pos) if occ[0] else frozenset()
        hier = hierarchy_supported(pos, occ[1]) if occ[1] else frozenset()
        return census, census | hier

    member_pred = membership_predicate(source)
    per_var: dict = {}
    for scope in qa.scopes():
        # node -> {constraint: [one (sees the data graph, hierarchy graphs seen) per occurrence]}: each occurrence
        # needs its own graph's support (external reviewer 6.1 review of 7e81a1e, item 1)
        cons: dict = {}
        for e in scope.scope_edges():
            occ = (sees_data(e, hierarchy), visible_graphs(e, hierarchy))
            if not occ[0] and not occ[1]:
                continue  # confined to a graph neither the census nor the shared hierarchy describes
            if is_type_pattern(e, member_pred):
                if str(e.o) not in data_nodes:  # rule 40: a data node constrains the node to no class
                    cons.setdefault(e.s, {}).setdefault(("class", str(e.o)), []).append(occ)
                continue
            if len(e.preds) != 1 or e.preds[0] not in census_preds or e.closure:
                continue
            for node, side in ((e.s, "subject"), (e.o, "object")):
                if side == "object" and e.preds[0] in literal_objects:
                    continue
                cons.setdefault(node, {}).setdefault((e.preds[0], side), []).append(occ)
        for node, cs in cons.items():
            if isinstance(node, Variable):
                per_var.setdefault(node, []).append((scope, cs))
    results = []
    for var, scopes in per_var.items():
        main, branch = None, None
        for scope, cs in scopes:
            classes = sorted(v for k, v in cs if k == "class")
            positions = sorted((k, v) for k, v in cs if k != "class")
            # A join needs a position for a typed variable and two for an untyped one; type
            # patterns alone are no join (their conflicts are the consistency checks' business).
            if not positions or (not classes and len(positions) < 2):
                continue
            hier_used = False
            if classes:  # typed: each query class supported at each occurrence of each joined position, by the
                # census or by the hierarchy, each by its own graph's rule (entry 70)
                with_all = []
                for c in classes:
                    tocc = cs[("class", c)]
                    ok_c = True
                    for pos, occ in [(p_, o) for p_ in positions for o in cs[p_]]:
                        if by_census(c, pos, occ, tocc):
                            continue
                        if by_hierarchy(c, pos, occ, tocc):
                            hier_used = True
                            continue
                        ok_c = False
                        break
                    if ok_c:
                        with_all.append(c)
                ok = len(with_all) == len(classes)
                by_hierarchy_support = ok and hier_used
            else:  # untyped: some class supported at every occurrence of every joined position
                both = [untyped_classes(pos, occ) for pos in positions for occ in cs[pos]]
                census_only = frozenset.intersection(*(b[0] for b in both))
                with_all = sorted(census_only or frozenset.intersection(*(b[1] for b in both)))
                by_hierarchy_support = not census_only and bool(with_all)
                ok = bool(with_all)
            ev = {"positions": sorted(f"{v} {k}" for k, v in cs), "classes_with_all": with_all[:20],
                  "threshold": threshold, "scope": scope.kind}
            if by_hierarchy_support:
                ev["shared_hierarchy_support"] = True
            if ok:
                verdict = ("pass", "holds")
            elif any(any(o[0] for o in cs[pos]) and not direct.get(pos) and untyped.get(pos, 0) >= threshold
                     for pos in positions):
                verdict = ("unknown", "untyped_fillers")
            else:
                verdict = ("flag", "census_low_support")
            if verdict[0] == "flag" and scope is not qa.root:
                branch = branch or (verdict, ev)
                continue
            rank = {"flag": 3, "unknown": 2, "pass": 1}[verdict[0]]
            if main is None or rank > main[0]:
                main = (rank, verdict, ev)
        key = f"?{var}"
        if main is None and branch is None:
            results.append(CheckResult("census_join.static", source.name, Outcome.NOT_APPLICABLE, "no_join", key))
            continue
        if main is not None:
            (outcome, reason), ev = main[1], main[2]
            if outcome != "flag" and var in qa.tainted:
                outcome, reason = "unknown", "unsupported_construct"
            results.append(CheckResult("census_join.static", source.name, Outcome(outcome), reason, key, evidence=ev,
                                       scope=ev.get("scope", "required") if ev.get("scope") in ("optional", "negated")
                                       else "required"))
        if branch is not None:
            (outcome, reason), ev = branch
            # Rule 28: the scope the flag was read in, as for the consistency checks' branch-local flags.
            results.append(CheckResult("census_join.static", source.name, Outcome.FLAG, reason, key, evidence=ev,
                                       branch_local=True,
                                       scope=ev["scope"] if ev.get("scope") in ("optional", "negated") else "optional"))
    if not results:
        results.append(CheckResult("census_join.static", source.name, Outcome.NOT_APPLICABLE, "no_join"))
    return results + pattern_results
