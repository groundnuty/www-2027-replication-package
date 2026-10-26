"""Answer-preserving rewrites: correct queries that differ from the gold (a control stratum).

Design ruling in the design log (change-log entry 39, pre-registration section 4.1): the false-flag rate
is half the untouched gold and half these rewrites, k per question, so a checker is also tested on
correct queries that are not the gold's own text. Each rewrite is drawn from store facts and the
shared hierarchy only, never from a checker source or the census, and is kept only when the
rewritten query, executed on the pinned store, returns exactly the gold rows under the question's
scoring rules. A rejected rewrite is recorded with its reason.

Operators, on a variable whose every answer binding (a plain projected column, an IRI in every gold
row) satisfies the added pattern, which is checked on the store before the rewrite runs:

    rewrite_add_type_specific   add a membership pattern (the graph's convention: rdf:type on
                                UniProt, rdfs:subClassOf on Rhea) at the most specific class every
                                binding has: its stored membership objects that are classes (not
                                RDF, RDFS or OWL meta classes, not rule 32 data nodes), closed up
                                the shared hierarchy; a class the query already asserts of the
                                variable is not added again
    rewrite_add_type_ancestor   the same at one strict shared-hierarchy ancestor of that class
    rewrite_transitive_class    replace an exact membership constraint ``?x m C`` by its transitive
                                form ``?x m/rdfs:subClassOf* C``
    rewrite_redundant_join      add ``?x p ?fresh`` for a predicate every binding has (not a
                                membership predicate, not one the query already uses on ?x), up to
                                three predicates per variable

An added class pattern is written in its direct form (``?x m C``) when every binding has that triple,
and otherwise in the path form (``?x m/rdfs:subClassOf* C``); stored data rarely holds a literal
membership triple to an ancestor, and a correct query that names an ancestor uses the path. The form
is recorded on each rewrite.

The draw: the candidates are fixed before any outcome is seen (deterministic per question, seeded);
each operator's list is shuffled with its own seed, and they are taken round robin in the order of
``REWRITES`` and executed one at a time until k are accepted, the candidates run out, or the attempt
cap is reached. The order does not depend on k, so the rewrites of a smaller k are a prefix of a
larger k's (``draw_rank``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from harness.faults.operators import META, rng_for, store_data_node
from harness.faults.query import (
    RDF_TYPE,
    RDFS_SUBCLASSOF,
    Edit,
    ParsedQuery,
    apply_edits,
    iri_text,
    parse,
)
from harness.faults.store import (
    Execution,
    Runner,
    SparqlOracle,
    answer_effect,
    query_sha,
)
from harness.score.rules import ScoringRules
from harness.terms import IRI

REWRITES = ("rewrite_add_type_specific", "rewrite_add_type_ancestor", "rewrite_transitive_class",
            "rewrite_redundant_join")
# Retired by ruling, kept in the code for the record and for exploratory use only when named explicitly.
RETIRED_REWRITES = {
    "rewrite_redundant_join": "artificial: dangling joins, often on record metadata (dev-only semantic audit; "
                              "design ruling, design log, change-log entry 41)",
}
ACTIVE_REWRITES = tuple(op for op in REWRITES if op not in RETIRED_REWRITES)
CONTROL = "answer_preserving_rewrite"
MEMBERSHIP = frozenset({RDF_TYPE, RDFS_SUBCLASSOF})
SUB = f"<{RDFS_SUBCLASSOF}>"
JOINS_PER_VARIABLE = 3
CHUNK = 100  # bindings per VALUES block


@dataclass
class RewriteSettings:
    k: int
    attempts: int  # executions per question, at most
    membership: str  # the graph's membership predicate
    shared_parents: dict[str, list[str]]  # the shared hierarchy only (rule 35)
    oracle: SparqlOracle  # store facts (rule 32's data-node test)
    runner: Runner
    seed: int
    operators: tuple[str, ...] = REWRITES  # the operators drawn from (the build passes ACTIVE_REWRITES)
    _anc: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    def ancestors(self, c: str) -> frozenset[str]:
        """c and every class above it in the shared hierarchy."""
        if c not in self._anc:
            seen, stack = {c}, [c]
            while stack:
                for p in self.shared_parents.get(stack.pop(), ()):
                    if p not in seen:
                        seen.add(p)
                        stack.append(p)
            self._anc[c] = frozenset(seen)
        return self._anc[c]


@dataclass
class Candidate:
    operator: str
    query: str
    edits: list[dict]
    selection: dict


def _values(bindings: list[str]) -> list[str]:
    return [" ".join(f"<{b}>" for b in bindings[i:i + CHUNK]) for i in range(0, len(bindings), CHUNK)]


def _rows(s: RewriteSettings, query: str) -> list:
    ex = s.runner.run(query)
    if ex.status != "ok" or ex.truncated:
        raise RuntimeError(f"rewrite store query failed: {ex.error_kind}: {ex.error}")
    return ex.rows or []


def _count(s: RewriteSettings, bindings: list[str], pattern: str) -> int:
    total = 0
    for block in _values(bindings):
        rows = _rows(s, f"SELECT (COUNT(DISTINCT ?b) AS ?n) WHERE {{ VALUES ?b {{ {block} }} {pattern} }}")
        total += int(rows[0][0].lexical) if rows and rows[0][0] is not None else 0
    return total


def _store_class(s: RewriteSettings, iri: str) -> bool:
    return not iri.startswith(META) and not store_data_node(s.oracle.facts(iri))


def _direct(s: RewriteSettings, bindings: list[str]) -> dict[str, set[str]]:
    """Each binding's stored membership objects that are classes."""
    out: dict[str, set[str]] = {b: set() for b in bindings}
    for block in _values(bindings):
        for row in _rows(s, f"SELECT DISTINCT ?b ?c WHERE {{ VALUES ?b {{ {block} }} ?b <{s.membership}> ?c "
                            "FILTER(isIRI(?c)) }"):
            if isinstance(row[0], IRI) and isinstance(row[1], IRI):
                out[row[0].value].add(row[1].value)
    classes = {c for cs in out.values() for c in cs}
    keep = {c for c in sorted(classes) if _store_class(s, c)}
    return {b: cs & keep for b, cs in out.items()}


def _predicates(s: RewriteSettings, bindings: list[str]) -> list[str]:
    """Predicates every binding is the subject of."""
    counts: dict[str, int] = {}
    for block in _values(bindings):
        for row in _rows(s, f"SELECT ?p (COUNT(DISTINCT ?b) AS ?n) WHERE {{ VALUES ?b {{ {block} }} ?b ?p ?o }} GROUP BY ?p"):
            if isinstance(row[0], IRI):
                counts[row[0].value] = counts.get(row[0].value, 0) + int(row[1].lexical)
    return sorted(p for p, n in counts.items() if n == len(bindings))


def _columns(q: ParsedQuery, gold: Execution) -> tuple[dict[str, list[str]], str | None]:
    """Answer bindings of each plain projected IRI column, or why there are none."""
    if not q.selects or q.selects[0].nested or not gold.columns or not gold.rows:
        return {}, "no_iri_column"
    sel = q.selects[0]
    plain = {"?" + q.tokens[i].text[1:] for i in sel.projection}
    out: dict[str, list[str]] = {}
    for j, name in enumerate(gold.columns):
        var = "?" + name
        if (sel.projection_all or var in plain) and all(row[j] is not None and isinstance(row[j], IRI) for row in gold.rows):
            out[var] = sorted({row[j].value for row in gold.rows})
    return out, (None if out else "no_iri_column")


def _fresh(q: ParsedQuery) -> str:
    written = {"?" + t.text[1:] for t in q.tokens if t.kind == "var"}
    n = 1
    while f"?rw{n}" in written:
        n += 1
    return f"?rw{n}"


def _insert(q: ParsedQuery, pattern: str) -> tuple[str, list[dict]]:
    brace = q.tokens[q.selects[0].where]
    edit = Edit(brace.end, brace.end, f" {pattern} .")
    return apply_edits(q.text, [edit]), [asdict(edit)]


def _membership_text(q: ParsedQuery, m: str) -> str:
    return "a" if m == RDF_TYPE else iri_text(q, m)


def candidates(question: dict, gold: Execution, s: RewriteSettings) -> tuple[list[Candidate], dict[str, str]]:
    """Every candidate rewrite of one gold query, and for each operator without one, why."""
    qid = question["id"]
    q = parse(question["gold_sparql"], lambda iri: _store_class(s, iri))
    out: list[Candidate] = []
    reasons: dict[str, str] = {}

    # rewrite_transitive_class: exact membership constraints on a class
    for slot in q.slots:
        if slot.role != "class" or slot.pattern is None:
            continue
        p = q.patterns[slot.pattern]
        if p.predicate not in MEMBERSHIP or p.modifier or p.in_path or p.complex_path or p.predicate_token is None:
            continue
        if sum(1 for o in q.patterns if o.predicate_token == p.predicate_token) != 1:
            continue  # a verb shared by several objects would change them all
        verb = q.tokens[p.predicate_token]
        edit = Edit(verb.start, verb.end, f"{verb.text}/{SUB}*")
        out.append(Candidate("rewrite_transitive_class", apply_edits(q.text, [edit]), [asdict(edit)],
                             {"variable": p.subject, "class": slot.iri, "predicate": p.predicate}))
    if not any(c.operator == "rewrite_transitive_class" for c in out):
        reasons["rewrite_transitive_class"] = "no_class_constraint"

    columns, why = _columns(q, gold)
    if why is None and q.selects[0].where is None:
        columns, why = {}, "no_where_group"
    elif why is None and q.tokens[q.selects[0].where + 1:] and _next_word(q, q.selects[0].where) == "SELECT":
        columns, why = {}, "where_is_subselect"
    var_ops = ("rewrite_add_type_specific", "rewrite_add_type_ancestor", "rewrite_redundant_join")
    if why is not None:
        reasons.update({op: why for op in var_ops})
        return out, reasons
    m_text = _membership_text(q, s.membership)
    fresh = _fresh(q)
    found = {op: False for op in var_ops}
    shortfall: dict[str, str] = {}
    for var, bindings in sorted(columns.items()):
        asserted = {p.object for p in q.patterns if p.subject == var and p.predicate in MEMBERSHIP
                    and not p.object.startswith(("?", "_:", "lit:"))}
        used = {p.predicate for p in q.patterns if p.subject == var}
        try:
            direct = _direct(s, bindings)
            shared = set.intersection(*({a for c in cs for a in s.ancestors(c)} for cs in direct.values())) - asserted
            specific = sorted(c for c in shared if not any(c in s.ancestors(d) - {d} for d in shared))
            if not specific:
                shortfall.setdefault("rewrite_add_type_specific", "no_shared_class")
                shortfall.setdefault("rewrite_add_type_ancestor", "no_shared_class")
            else:
                chosen = rng_for(s.seed, qid, f"rewrite:specific:{var}").choice(specific)
                c = _class_candidate(q, s, "rewrite_add_type_specific", var, bindings, direct, chosen, m_text,
                                     {"ties": len(specific)})
                if c:
                    out.append(c)
                    found["rewrite_add_type_specific"] = True
                above = sorted(s.ancestors(chosen) - {chosen} - asserted)
                if not above:
                    shortfall.setdefault("rewrite_add_type_ancestor", "no_shared_ancestor")
                else:
                    anc = rng_for(s.seed, qid, f"rewrite:ancestor:{var}").choice(above)
                    c = _class_candidate(q, s, "rewrite_add_type_ancestor", var, bindings, direct, anc, m_text,
                                         {"below": chosen, "ancestors": len(above)})
                    if c:
                        out.append(c)
                        found["rewrite_add_type_ancestor"] = True
            preds = [p for p in _predicates(s, bindings) if p not in MEMBERSHIP and p not in used]
            if not preds:
                shortfall.setdefault("rewrite_redundant_join", "no_shared_predicate")
            elif q.selects[0].projection_all:
                shortfall.setdefault("rewrite_redundant_join", "select_star")
            else:
                for pred in sorted(rng_for(s.seed, qid, f"rewrite:join:{var}").sample(preds, min(JOINS_PER_VARIABLE, len(preds)))):
                    text, edits = _insert(q, f"{var} {iri_text(q, pred)} {fresh}")
                    out.append(Candidate("rewrite_redundant_join", text, edits,
                                         {"variable": var, "predicate": pred, "fresh": fresh, "bindings": len(bindings),
                                          "shared_predicates": len(preds)}))
                    found["rewrite_redundant_join"] = True
        except RuntimeError as exc:  # a store failure on a fact query leaves this variable without candidates
            for op in var_ops:
                shortfall.setdefault(op, f"store_error:{str(exc)[:80]}")
    for op in var_ops:
        if not found[op]:
            reasons[op] = shortfall.get(op, "no_candidate")
    return out, reasons


def _next_word(q: ParsedQuery, i: int) -> str | None:
    for t in q.tokens[i + 1:]:
        if t.kind not in ("space", "comment"):
            return t.text.upper() if t.kind == "word" else None
    return None


def _class_candidate(q: ParsedQuery, s: RewriteSettings, op: str, var: str, bindings: list[str],
                     direct: dict[str, set[str]], cls: str, m_text: str, extra: dict) -> Candidate | None:
    """The added class pattern in its direct form when every binding has the triple, else the path
    form when every binding reaches the class through it; None when neither holds on the store."""
    if all(cls in cs for cs in direct.values()):
        form, pattern = "direct", f"{m_text} {iri_text(q, cls)}"
    elif _count(s, bindings, f"?b <{s.membership}>/{SUB}* <{cls}>") == len(bindings):
        form, pattern = "transitive", f"{m_text}/{SUB}* {iri_text(q, cls)}"
    else:
        return None
    text, edits = _insert(q, f"{var} {pattern}")
    return Candidate(op, text, edits, {"variable": var, "class": cls, "form": form, "bindings": len(bindings), **extra})


def order(qid: str, cands: list[Candidate], seed: int) -> list[Candidate]:
    """Round robin over the operators, each list shuffled with its own seed; independent of k."""
    lists = {}
    for op in REWRITES:
        mine = sorted((c for c in cands if c.operator == op), key=lambda c: c.query)
        rng_for(seed, qid, f"rewrite:order:{op}").shuffle(mine)
        lists[op] = mine
    out: list[Candidate] = []
    while any(lists.values()):
        for op in REWRITES:
            if lists[op]:
                out.append(lists[op].pop(0))
    return out


def draw(question: dict, gold: Execution, rules: ScoringRules, s: RewriteSettings) -> tuple[list[dict], list[dict], dict]:
    """The accepted rewrites (bank items without the question's fields), the rejected attempts, and
    the per-question report."""
    qid = question["id"]
    cands, reasons = candidates(question, gold, s)
    cands = [c for c in cands if c.operator in s.operators]
    queue = order(qid, cands, s.seed)
    accepted: list[dict] = []
    rejected: list[dict] = []
    tried = 0
    seen = {query_sha(question["gold_sparql"])}
    stop = "candidates_exhausted"
    for c in queue:
        if len(accepted) >= s.k:
            stop = "target_reached"
            break
        if tried >= s.attempts:
            stop = "attempt_cap"
            break
        sha = query_sha(c.query)
        if sha in seen:  # the same text twice (or the gold's): never a second control
            rejected.append({"question_id": qid, "operator": c.operator, "query_sha256": sha, "reason": "duplicate",
                             "selection": c.selection})
            continue
        seen.add(sha)
        tried += 1
        effect = answer_effect(gold, s.runner.run(c.query), rules)
        if effect["label"] == "answer_neutral":
            rank = len(accepted) + 1
            accepted.append({"id": f"{qid}#rewrite-{rank:02d}", "kind": "rewrite", "operator": c.operator,
                             "fault_class": None, "control": CONTROL, "bank": "control", "draw_rank": rank,
                             "query": c.query, "query_sha256": sha, "edits": c.edits, "selection": c.selection,
                             "answer_effect": effect})
        else:
            reason = "answer_changed" if effect["label"] == "answer_changing" else effect.get("reason", effect["label"])
            rejected.append({"question_id": qid, "operator": c.operator, "query_sha256": sha, "reason": reason,
                             "basis": effect.get("basis"), "selection": c.selection})
    else:
        if len(accepted) >= s.k:
            stop = "target_reached"
    counts = {op: sum(1 for c in cands if c.operator == op) for op in REWRITES}
    taken = {op: sum(1 for i in accepted if i["operator"] == op) + sum(1 for r in rejected if r["operator"] == op)
             for op in REWRITES}
    report: dict[str, Any] = {"question_id": qid, "candidates": counts, "reasons": dict(sorted(reasons.items())),
                              "tried": tried, "accepted": len(accepted), "stop": stop,
                              "accepted_by_operator": {op: sum(1 for i in accepted if i["operator"] == op) for op in REWRITES},
                              "untried": {op: counts[op] - taken[op] for op in REWRITES},
                              "operators": list(s.operators)}
    return accepted, rejected, report
