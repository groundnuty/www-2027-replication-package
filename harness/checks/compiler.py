"""The check compiler: one source in, every check of the S0.1 contract out.

The same compiler serves every source (curated, census, generated, and later
UniProt and a15). The closure is computed once per source; each output (the
agent's checked SPARQL query and the rows it submitted) then gets one report
per source, with every check tagged by that source.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rdflib import URIRef

from harness.checks import footprint, numbers, static
from harness.checks.closure import (
    COMMON_SUBCLASS_NODE_CAP,
    META_TYPES,
    RDF_TYPE,
    VACUOUS_CLASSES,
    Closure,
)
from harness.checks.hierarchy import SCHEMA_VERSION as HIERARCHY_SCHEMA
from harness.checks.hierarchy import SharedHierarchy, membership_query
from harness.checks.outcomes import (
    INDEPENDENT_CHECKS,
    CheckResult,
    Outcome,
    OutputReport,
)
from harness.checks.query import QueryAnalysis, analyse
from harness.sources.model import ClassExpr, OntologySource
from harness.sources.origin import FlagOrigin
from harness.terms import IRI, Row

DEFAULT_BOUND = 10_000
# Rule 36 (change-log entry 25): the terms defined in the published RDF, RDFS, OWL 2 and XSD
# vocabularies, one list shared with the fault-bank labeller (scripts/build_w3c_terms.py).
W3C_TERMS_PATH = Path(__file__).resolve().parents[2] / "data" / "vocab" / "w3c-terms.json"
_W3C_TERMS: frozenset[str] | None = None


def w3c_terms() -> frozenset[str]:
    """The rule-36 list; a compiler refuses to start without it."""
    global _W3C_TERMS
    if _W3C_TERMS is None:
        if not W3C_TERMS_PATH.exists():
            raise FileNotFoundError(f"rule 36 needs the shared W3C term list at {W3C_TERMS_PATH}")
        doc = json.loads(W3C_TERMS_PATH.read_text())
        if doc.get("schema") != "w3c-terms/1" or not doc.get("terms"):
            raise ValueError(f"{W3C_TERMS_PATH} is not a w3c-terms/1 list")
        _W3C_TERMS = frozenset(doc["terms"])
    return _W3C_TERMS

ONTOLOGY_CHECKS = (
    "vocabulary.static", "class.static", "domain_range.static", "disjointness.static",
    "no_common_subclass.static", "census_join.static", "census_pattern.static", *footprint.FOOTPRINT_CHECKS,
)
# The order of the unknowns static checks become next to an unsupported construct. They were appended in set iteration
# order, which follows PYTHONHASHSEED, so reports differed between processes and a recorded checker decision could not
# be recomputed in another. This is the order PYTHONHASHSEED=0 gave (every static check is always present, and that
# seed orders the six alike whatever their insertion order), kept so that seed-0 reports stay byte-identical.
UNSUPPORTED_ORDER = (
    "disjointness.static", "no_common_subclass.static", "census_pattern.static", "census_join.static",
    "domain_range.static", "class.static",
)


def _unsupported_rank(check_id: str) -> tuple[int, str]:
    """A static check's place in ``UNSUPPORTED_ORDER``; a check id it lacks comes after them, by name."""
    if check_id in UNSUPPORTED_ORDER:
        return UNSUPPORTED_ORDER.index(check_id), ""
    return len(UNSUPPORTED_ORDER), check_id


class Client(Protocol):
    def query(self, query: str) -> Any: ...  # a harness.sparql.QueryResult


@dataclass
class Output:
    """What the checker reads from one attempt.

    ``query`` is the checked query (rule 22); static and footprint checks read
    it. ``result_rows`` are the rows it returned to the agent in the run.
    ``trajectory_rows`` holds the rows of every query the agent executed, one
    list per query; number traceability reads all of them (rule 21), and is
    unknown when the run recorded neither. Build an output from a
    run's trajectory with ``Output.from_trajectory``.
    """

    query: str
    answer: Sequence[Row]
    result_rows: Sequence[Row] | None = None  # as the agent received them; never re-run in their place
    role: str = "agent"  # "agent" or "gold"
    trajectory_rows: Sequence[Sequence[Row]] | None = None
    checked_status: str = "given"  # "aligned" / "fallback" when chosen from a trajectory
    store_accepted: bool | None = None  # did the store execute the checked query (None: unknown)

    @classmethod
    def from_trajectory(
        cls, executed: Sequence[tuple[str, Sequence[Row]]], answer: Sequence[Row], role: str = "agent"
    ) -> Output:
        """Rule 22: the checked query is the last executed query whose result
        rows contain all the submitted rows, under the scorer's matching
        ("aligned"); if there is none, the last executed query ("fallback").
        ``executed``: (query, rows) or (query, rows, ok) for each SPARQL query
        the agent ran, in order; ok is False for a query the store refused."""
        if not executed:
            raise ValueError("an output needs at least one executed query")
        entries = [(e[0], list(e[1] or []), e[2] if len(e) > 2 else True) for e in executed]
        chosen, status = entries[-1], "fallback"
        for entry in reversed(entries):
            if entry[2] and numbers.rows_contained(answer, entry[1]):
                chosen, status = entry, "aligned"
                break
        return cls(chosen[0], answer, chosen[1], role, [rows for _, rows, ok in entries if ok], status, chosen[2])


def number_traceability(name: str, output: Output) -> CheckResult:
    """Rule 8: number traceability, the ontology-independent layer every arm runs alike."""
    wanted = numbers.trace(output.answer, [[]])
    if wanted.outcome == "not_applicable":
        return CheckResult("number_traceability", name, Outcome.NOT_APPLICABLE, "no_numbers", "answer")
    # Rule 21: only rows the agent received count. They come from the run (its
    # trajectory, or the checked query's recorded result); the checker never
    # re-executes a query to stand in for them.
    if output.trajectory_rows is not None:
        row_sets = [list(r) for r in output.trajectory_rows]
    elif output.result_rows is not None:
        row_sets = [list(output.result_rows)]
    else:
        return CheckResult("number_traceability", name, Outcome.UNKNOWN, "received_rows_unavailable", "answer",
                           detail="the run recorded no result rows the agent received")
    t = numbers.trace(output.answer, row_sets)
    ev = {"answer_numbers": t.answer_numbers, "untraced": t.untraced, "via_row_count": t.via_row_count,
          "result_sets": len(row_sets), "result_rows": sum(len(r) for r in row_sets)}
    return CheckResult("number_traceability", name, Outcome(t.outcome), t.reason, "answer", evidence=ev)


class CheckCompiler:
    def __init__(self, source: OntologySource, hierarchy: SharedHierarchy, bound: int = DEFAULT_BOUND, *,
                 search_cap: int = COMMON_SUBCLASS_NODE_CAP):
        """``hierarchy``: the store's shared hierarchy, precomputed offline with an
        explicit list of excluded graphs (rules 16, 25, 29). A compiler without it
        refuses to start. ``bound``: the retrieval budget of one output (rule 27).
        ``search_cap``: the memory cap of one no-common-subclass search, in hierarchy
        nodes held (closure.COMMON_SUBCLASS_NODE_CAP)."""
        if not isinstance(hierarchy, SharedHierarchy):
            raise ValueError("rule 29: a run needs the shared hierarchy with its excluded graphs set explicitly")
        if source.kind == "census" and hierarchy.version != HIERARCHY_SCHEMA:
            # Rule 20 with change-log entries 61, 70 and 72: census support is judged against the data graph and the
            # hierarchy graphs, one graph at a time, so a census needs the positions each hierarchy graph holds and
            # which of its graphs hold each subclass edge: a v4 hierarchy. An older one would silently judge the
            # census alone, or pool the graphs.
            missing = [what for what, absent in (("records no hierarchy positions", hierarchy.positions is None),
                                                 ("does not record which of its graphs hold each subclass edge",
                                                  hierarchy.edge_graphs is None)) if absent]
            raise ValueError(f"rule 20 (change-log entry 61; entries 70 and 72): a census needs a {HIERARCHY_SCHEMA} "
                             f"shared hierarchy; this one is {hierarchy.version}"
                             + (f" and {' and '.join(missing)}" if missing else "")
                             + "; rebuild it (SharedHierarchy.build or build_streamed), or carry a complete v3 file "
                               "over with SharedHierarchy.to_v4, before checking a census")
        self.source = source
        self.hierarchy = hierarchy
        self.bound = bound
        # Per-axiom origin (design ruling in the design log Oct 08:04): copied into each flag's evidence, never read by
        # a verdict; only a source that records it has it.
        self._origin = FlagOrigin(source) if getattr(source, "origin", None) else None
        self.w3c = w3c_terms()  # rule 36
        self.base = Closure(source)
        # Rule 32, part (a) of the data-node test: a class the source declares is never a data node. Declared, not
        # merely named in a domain, range, disjointness axiom or pattern (External review of the sources snapshot,
        # item 4): a generated source records its explicit declarations apart from every class-position IRI.
        declared = getattr(source, "declared_classes", None)
        self.declared = frozenset(source.classes if declared is None else declared)
        self.data_nodes = hierarchy.data_nodes - self.declared
        # Rule 16: one closure from the shared hierarchy plus the source's own axioms, computed once.
        # Named classes for the common-subclass search: the source's, plus the hierarchy's classes, never a data
        # node. Membership follows the data graph's membership edges transitively, then the shared hierarchy
        # (rule 35). Rule 40: a data node in a class position implies no class requirement.
        # The search reads both edge maps upwards too: the hierarchy's own inverses, exact by construction
        # (closure_children inverts closure_parents; children was made from parents).
        self.closure = self.base.merged(hierarchy.closure_parents(), hierarchy.children(),
                                        hierarchy.classes - self.data_nodes, self.data_nodes,
                                        store_parents_inverse=hierarchy.closure_children(),
                                        store_children_inverse=hierarchy.parents, search_cap=search_cap)
        self._support_cache: dict = {}  # entry 61: the hierarchy's support per census position, computed once

    # -- evidence read before the static checks ----------------------------------------

    def _ask(self, client: Client):
        def ask(q: str):
            r = client.query(q)
            if getattr(r, "truncated", False):
                raise footprint.OverBound("a response was cut by the client's row cap")
            return r
        return ask

    def _fetch_types(self, iris: list[str], client: Client | None, type_map, budget) -> tuple[dict | None, str | None]:
        """Asserted membership types of constants, or (None, reason). They count
        against the output's budget (rule 27)."""
        if not iris:
            return {}, None
        if type_map is not None:
            out = {i: set(type_map.get(i, ())) for i in iris}
        elif client is None:
            return None, "no_store"
        else:
            out = {i: set() for i in iris}
            try:
                for k in range(0, len(iris), footprint.FETCH_CHUNK):
                    chunk = iris[k : k + footprint.FETCH_CHUNK]
                    # rule 35: rdf:type objects, and rdfs:subClassOf objects inside the data graph
                    q = f"{membership_query(chunk, self.hierarchy.roles)} LIMIT {budget.left + 1}"
                    r = self._ask(client)(q)
                    budget.take(len(r.rows))
                    for s, t, _p in r.rows:
                        if isinstance(s, IRI) and isinstance(t, IRI):
                            out.setdefault(s.value, set()).add(t.value)
            except footprint.OverBound:
                return None, "bound"
            except Exception as exc:  # the store refused or failed the query
                return None, str(exc)[:300]
        for iri, ts in out.items():  # a class used as a value is its own membership (F30)
            if not (ts - META_TYPES - VACUOUS_CLASSES) and (iri in self.hierarchy.parents or iri in self.hierarchy.classes):
                ts.add(iri)
        return out, None

    # -- vocabulary existence (static, rule 25) ------------------------------------------

    def _vocabulary(self, qa: QueryAnalysis) -> list[CheckResult]:
        """Rule 25: every predicate, and every other term, is in the source
        itself; a class term may also be a class of the shared hierarchy, known
        to every source alike. Offline: never a lookup in the store. Terms
        defined by the RDF, RDFS, OWL 2 and XSD vocabularies are exempt (rule 36,
        the shared list), and an IRI absent from the store and not declared by
        the source is a vocabulary term in a predicate or class position (rule 37);
        an absent subject or object constant falls under rule 5 instead, where it
        has no asserted types and so is unknown, never flagged. A data node
        (rule 32, see harness/checks/hierarchy.py) in a class position, such as
        rh:Compound_3249 in ``?p rdfs:subClassOf rh:Compound_3249``, is an entity
        constant (rule 5), not vocabulary; a class the source declares is checked."""
        name = self.source.name
        member_pred = static.membership_predicate(self.source)
        preds: set[str] = set()
        classes: set[str] = set()
        for g in qa.groups:
            for e in g.edges:
                # rule 36: only terms the W3C vocabularies define are exempt, not their whole namespaces
                preds.update(p for p in e.preds if p not in self.w3c)
                if (static.is_type_pattern(e, member_pred) or (e.preds == (RDF_TYPE,) and isinstance(e.o, URIRef))) \
                        and str(e.o) not in self.w3c:
                    classes.add(str(e.o))
        classes -= self.data_nodes  # data nodes in class positions are constants (rule 5), not vocabulary
        if not preds and not classes:
            return [CheckResult("vocabulary.static", name, Outcome.PASS, "no_terms")]
        out = []
        for t in sorted(preds | classes):
            if t in self.base.vocabulary:
                out.append(CheckResult("vocabulary.static", name, Outcome.PASS, "holds", t))
            elif t in classes and t in self.hierarchy.classes:
                out.append(CheckResult("vocabulary.static", name, Outcome.PASS, "holds", t,
                                       evidence={"term_of": "the shared hierarchy (rule 25)"}))
            else:
                out.append(CheckResult("vocabulary.static", name, Outcome.FLAG, "not_in_source", t))
        return out

    # -- number traceability (its own layer) -------------------------------------------

    def _numbers(self, qa: QueryAnalysis, output: Output, client: Client | None) -> CheckResult:
        return number_traceability(self.source.name, output)

    # -- provenance (a15 only, rule 24) -------------------------------------------------

    def _provenance(self, output: Output, client: Client | None, budget) -> CheckResult:
        """Rule 24: an existential ``cls ⊑ ∃prop.filler`` read as a required path:
        every answer value under ``cls`` must reach a ``filler`` node through
        ``prop``. A complete search with no path is a flag; a search cut by the
        output's budget is unknown; no applicable requirement is NA."""
        name = self.source.name
        if not self.source.graph.startswith("a15"):
            return CheckResult("provenance", name, Outcome.NOT_APPLICABLE, "no_provenance_model", "answer",
                               detail="the graph models no process-and-method provenance (a15 only)")
        if not self.source.existentials:
            return CheckResult("provenance", name, Outcome.NOT_APPLICABLE, "no_applicable_requirement", "answer")
        values = sorted({c.value for row in output.answer for c in row if isinstance(c, IRI)})
        if not values:
            return CheckResult("provenance", name, Outcome.NOT_APPLICABLE, "no_applicable_requirement", "answer")
        if client is None:
            return CheckResult("provenance", name, Outcome.UNKNOWN, "no_store", "answer")
        ask = self._ask(client)
        mc = self.closure
        try:
            vtypes, err = self._fetch_types(values, client, None, budget)
            if vtypes is None:
                raise footprint.OverBound(err or "types")
            applicable = [(v, ex) for v in values for ex in self.source.existentials
                          if mc.falls_under(vtypes[v], ClassExpr.named(ex.cls))]
            if not applicable:
                return CheckResult("provenance", name, Outcome.NOT_APPLICABLE, "no_applicable_requirement", "answer")
            fillers: dict[tuple[str, str], set[str]] = {}
            for prop in sorted({ex.prop for _, ex in applicable}):
                nodes = sorted({v for v, ex in applicable if ex.prop == prop})
                r = ask(f"SELECT ?n ?f WHERE {{ VALUES ?n {{ {' '.join(f'<{v}>' for v in nodes)} }} ?n <{prop}> ?f }} "
                        f"LIMIT {budget.left + 1}")
                budget.take(len(r.rows))
                for nd, f in r.rows:
                    if isinstance(nd, IRI) and isinstance(f, IRI):
                        fillers.setdefault((nd.value, prop), set()).add(f.value)
            all_f = sorted({f for fs in fillers.values() for f in fs})
            ftypes, err = self._fetch_types(all_f, client, None, budget) if all_f else ({}, None)
            if ftypes is None:
                raise footprint.OverBound(err or "filler types")
        except footprint.OverBound:
            return CheckResult("provenance", name, Outcome.UNKNOWN, "bound", "answer")
        verdicts = []
        for v, ex in applicable:
            fs = fillers.get((v, ex.prop), set())
            if any(mc.falls_under(ftypes.get(f, set()), ClassExpr.named(ex.filler)) for f in fs):
                verdicts.append(("pass", v, ex))
            elif any(not ftypes.get(f) for f in fs):
                verdicts.append(("unknown", v, ex))
            else:
                verdicts.append(("flag", v, ex))
        ev = {"requirements": [f"{v} {ex.prop} some {ex.filler}: {o}" for o, v, ex in verdicts][:20]}
        if any(o == "flag" for o, _, _ in verdicts):
            return CheckResult("provenance", name, Outcome.FLAG, "no_provenance_path", "answer", evidence=ev)
        if any(o == "unknown" for o, _, _ in verdicts):
            return CheckResult("provenance", name, Outcome.UNKNOWN, "untyped_node", "answer", evidence=ev)
        return CheckResult("provenance", name, Outcome.PASS, "holds", "answer", evidence=ev)

    def _with_origin(self, r: CheckResult) -> CheckResult:
        """A flag with ``origin`` in its evidence: the parts (files or graphs) of the axioms it rests on
        (harness.sources.origin.FlagOrigin); every other result as it is."""
        if r.outcome is not Outcome.FLAG:
            return r
        origin = self._origin.of(r.key, r.evidence, r.check_type, independent=r.check_id in INDEPENDENT_CHECKS)
        return dataclasses.replace(r, evidence={**r.evidence, "origin": origin})

    # -- one output ---------------------------------------------------------------------

    def check(
        self,
        output: Output,
        client: Client | None = None,
        type_map: dict[str, Iterable[str]] | None = None,
        qa: QueryAnalysis | None = None,
    ) -> OutputReport:
        qa = qa or analyse(output.query)
        name = self.source.name
        results: list[CheckResult] = []
        evidence: dict[str, Any] = {}
        timed = _Timed(client) if client is not None else None
        client = timed if timed is not None else client
        closure = self.closure
        budget = footprint.Budget(self.bound)  # rule 27: one budget for every online phase
        tooling: list[str] = []
        outside = 0  # census occurrences confined to a graph neither the census nor the hierarchy describes
        if qa.error:
            # The issue-4 ruling: a query the store executed but rdflib cannot parse
            # is a tooling limit (parser_mismatch), not a property of the agent or
            # the ontology; a query nobody can run is a parse error.
            evidence["parse_error"] = qa.error
            if output.store_accepted is not None:
                executed = output.store_accepted
            else:  # rows handed in mean the store answered it
                executed = output.result_rows is not None or output.trajectory_rows is not None
            if output.store_accepted is None and not executed and client is not None:
                # probe: does the store accept it? (runs the agent's query once, unbounded)
                try:
                    client.query(output.query)
                    executed = True
                except Exception:
                    executed = False
            reason = "parser_mismatch" if executed else "query_parse_error"
            if executed:
                tooling.append("parser_mismatch")
            results += [CheckResult(c, name, Outcome.UNKNOWN, reason) for c in ONTOLOGY_CHECKS]
        else:
            iris = static.constant_iris(qa, self.closure, self.source)
            types, type_error = self._fetch_types(iris, client, type_map, budget)
            evidence["constant_type_fetch"] = iris
            if type_error:
                evidence["constant_type_error"] = type_error
            results += self._vocabulary(qa)
            consistency = static.consistency_checks(qa, closure, self.source)
            for cid in ("disjointness.static", "no_common_subclass.static"):
                if not any(r.check_id == cid for r in consistency):
                    consistency.append(CheckResult(cid, name, Outcome.NOT_APPLICABLE, "no_axiom"))
            results += consistency
            const_results = static.constant_checks(qa, closure, self.source, types, type_error, self.hierarchy)
            literal = static.literal_position_checks(qa, closure, self.source)
            if literal:  # F29 flags replace the NA placeholder of domain_range.static
                const_results = [r for r in const_results if not (r.check_id == "domain_range.static" and r.key == "-")]
            results += const_results + literal + static.detached_constant_checks(qa, closure, self.source)
            results += static.census_checks(qa, self.closure, self.source, self.data_nodes, self.hierarchy,
                                            self._support_cache)
            outside = static.graph_outside_census_count(qa, self.source, self.hierarchy)
            if qa.unsupported:  # a static check with nothing decided is unknown, not NA, next to an unsupported construct
                static_ids = sorted({r.check_id for r in results if r.form == "static"} - {"vocabulary.static"},
                                    key=_unsupported_rank)
                for cid in static_ids:
                    mine = [r for r in results if r.check_id == cid]
                    if all(r.outcome is Outcome.NOT_APPLICABLE for r in mine) and qa.tainted:
                        results = [r for r in results if r.check_id != cid]
                        results.append(CheckResult(cid, name, Outcome.UNKNOWN, "unsupported_construct"))
            fp_results, fp_ev = footprint.run(qa, self.closure, self.source, client, budget.left, self.hierarchy)
            results += fp_results
            evidence.update(fp_ev)
            spent = fp_ev.get("evidence_triples")
            if isinstance(spent, int):
                budget.take(min(spent, budget.left))
            elif isinstance(spent, str):  # the footprint went over what was left
                budget.exhaust()
        results.append(self._numbers(qa, output, client))
        results.append(self._provenance(output, client, budget))
        evidence["budget_used"] = min(budget.used, budget.total + 1)
        if timed is not None and timed.log:
            evidence["store_queries"] = timed.log
            evidence["possibly_partial"] = any(q["possibly_partial"] for q in timed.log)
        coverage = dict(self.source.unsupported)
        sub = sum(
            1 for g in qa.groups for e in g.edges for p in e.preds
            if p in self.source.properties and self.source.properties[p].parents
        )
        if sub:  # rule 3: inheritance through rdfs:subPropertyOf is not applied
            coverage["rdfs:subPropertyOf (query predicates, not applied)"] = sub
        if self._origin is not None:
            results = [self._with_origin(r) for r in results]
        missing = None
        if output.result_rows is not None:  # rule 22: are the submitted rows among the checked query's results?
            missing = sum(1 for row in output.answer if not numbers.row_among(row, output.result_rows))
        return OutputReport(
            source=name,
            role=output.role,
            results=results,
            negation=qa.negation,
            unsupported_constructs=list(qa.unsupported),
            unsupported_coverage=coverage,
            evidence=evidence,
            checked_query=output.query,
            checked_query_status=output.checked_status,
            answer_rows_not_in_checked_result=missing,
            tooling_limits=tooling,
            graph_outside_census=outside,
        )


class _Timed:
    """The store client, keeping each answer's time and possibly_partial mark as descriptive evidence.

    Change-log entry 57 (design log): an answer that arrived before the client's cut is complete, so neither the
    time nor the mark changes an outcome; the analysis reports the outputs with a possibly partial answer (over
    the client's threshold) as a descriptive stratum. A truncated answer is still refused where it is read.
    """

    def __init__(self, client: Client):
        self._client = client
        self.log: list[dict[str, Any]] = []

    def query(self, query: str) -> Any:
        r = self._client.query(query)
        self.log.append({"seconds": round(float(getattr(r, "elapsed_s", 0.0) or 0.0), 3),
                         "possibly_partial": bool(getattr(r, "possibly_partial", False))})
        return r


def check_output(
    source: OntologySource,
    output: Output,
    client: Client | None = None,
    type_map: dict[str, Iterable[str]] | None = None,
    bound: int = DEFAULT_BOUND,
    *,
    hierarchy: SharedHierarchy,
) -> OutputReport:
    return CheckCompiler(source, hierarchy, bound).check(output, client, type_map)


def check_all(
    compilers: Sequence[CheckCompiler], output: Output, client: Client | None = None
) -> list[OutputReport]:
    """One report per source for the same output; the query is parsed once."""
    qa = analyse(output.query)
    return [c.check(output, client, qa=qa) for c in compilers]
