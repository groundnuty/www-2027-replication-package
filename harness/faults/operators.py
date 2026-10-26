"""Mutation operators over gold queries (proposal §7.6), neutral about every checker source.

Fault classes are defined by operator (design ruling in the design log, pre-registration §2b, change-log
entry 21), never by a check-based rule:

    vocabulary_absent   invented_predicate, invented_class, invented_constant: an identifier in the
                        same namespace that is absent from the store and from every term set
    type_wrong          swap_predicate, swap_class, swap_constant: a term replaced by another drawn
                        from the census co-occurrence graph (``cooccurrence``), at two distance
                        strata, near (d = 2) and far (d = 4), one draw each where both exist
    stale_in_data       stale_predicate, stale_class, stale_constant: an older or deprecated
                        identifier the graph itself records as replaced by the gold's term; such
                        terms are still in the store (exploratory class)
    meaning             agg_count_distinct, agg_min_max, agg_drop_group_key; and, as a descriptive
                        stratum (``DESCRIPTIVE``), type_consistent_constant

The type-consistent constant swap (design ruling (d) in the design log, change-log entry 63) replaces a
gold constant by another member of the constant's own class, so a type check cannot see it. It is a
descriptive stratum of the meaning class, outside C2 and outside C4's W: the manifest gives its
items a bank of their own (``descriptive``), so no set built from answer-changing items takes them.
Its draws, per distinct constant of the gold (every occurrence of the constant is replaced):

    own classes     the constant's rdf:type objects (any graph) and, when the graph's membership
                    predicate is rdfs:subClassOf, its data-graph subclass objects (rule 35); meta
                    classes and rule 32 data nodes are no class
    the class       the most specific own class that has another member: the fewest members other
                    than the gold, then the IRI (a Rhea compound's ChEBI class usually holds the
                    compound alone, so its SmallMolecule class is drawn from)
    the draw        up to ``TYPE_CONSISTENT_TRIES`` distinct members, the gold excluded, fixed by
                    the seeded stream before any outcome is seen; each is executed in turn and the
                    first answer-changing one is kept. An answer-neutral or rejected draw is a try
                    and is never an item (a neutral one would join the controls); a constant with no
                    answer-changing draw is skipped and counted, with every try recorded

Each source's conformance label is a separate outcome per mutation (``conformance``), and whether
the mutated pattern has a match in the store ("realisable in the data") is a descriptor
(``realisable_query``), not a class.

Source neutrality (review #1 of 3afbfa1): drawing reads no checker source, not even as a list of
known terms. It reads the shared hierarchy, the census co-occurrence graph and store facts only.
Whether a term in a class position is a class or a constant (``Context.is_class``) is decided by
the shared hierarchy and graph classes first, then by rule 32's store tests (b) and (c), which are
the same for every source; an IRI absent from the store is a class position's vocabulary term
(rule 37). Test (a), a source's own class declaration, is the labeller's, per source.

Constants that are data nodes are not nodes of the co-occurrence graph. A constant swap therefore
anchors on the constant's own classes in the store (rdf:type objects, and data-graph subclass
objects followed through data nodes, check contract rule 35), draws a class at the stratum's
distance from the nearest anchor, and takes a seeded member of that class from the store. A
constant that is itself a class node of the graph is swapped for another class node directly.

Determinism: every choice draws from ``Random(sha256(seed|question id|operator[@d]))``, so one
question's mutations do not depend on any other question (text fixes or removals elsewhere leave
them unchanged). Candidate lists are sorted before drawing; store samples use ORDER BY and OFFSET.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Protocol

from harness.faults.cooccurrence import CooccurrenceGraph
from harness.faults.query import (
    RDF_TYPE,
    RDFS_SUBCLASSOF,
    Edit,
    GroupKey,
    ParsedQuery,
    Pattern,
    Scope,
    Select,
    Slot,
    apply_edits,
    iri_text,
    parse,
)

SWAPS = ("swap_predicate", "swap_class", "swap_constant")
INVENTED = ("invented_predicate", "invented_class", "invented_constant")
STALE = ("stale_predicate", "stale_class", "stale_constant")
AGGREGATION = ("agg_count_distinct", "agg_min_max", "agg_drop_group_key")
OPERATORS = SWAPS + INVENTED + STALE + AGGREGATION
TYPE_CONSISTENT = "type_consistent_constant"
DESCRIPTIVE = (TYPE_CONSISTENT,)  # descriptive strata: outside C2 and outside C4's W (design ruling (d), entry 63)
DESCRIPTIVE_STRATUM = {TYPE_CONSISTENT: "type_consistent"}
ALL_OPERATORS = OPERATORS + DESCRIPTIVE
FAULT_CLASS = {**{op: "type_wrong" for op in SWAPS}, **{op: "vocabulary_absent" for op in INVENTED},
               **{op: "stale_in_data" for op in STALE}, **{op: "meaning" for op in AGGREGATION},
               TYPE_CONSISTENT: "meaning"}
TYPE_CONSISTENT_TRIES = 5
STRATA = {2: "near", 4: "far"}
STALE_KIND = {
    "obo_replaced_by": "deprecated_term",  # owl:deprecated with IAO_0100001 'term replaced by'
    "rhea_obsolete_comment": "obsolete_entity",  # status rh:Obsolete, replacement named in rdfs:comment
    "uniprot_replaces": "secondary_accession",  # up:replaces on the current entry
    "static_map": "deprecated_term",  # a deprecation map supplied with the build
}
INVENTED_TRIES = 8
CONSTANT_CLASS_TRIES = 64  # far classes are often ancestors with no direct member; keep drawing
META = ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/2002/07/owl#")  # meta classes such as owl:Class are never classes (rule 35)


class StoreOracle(Protocol):
    def absent(self, iri: str) -> bool: ...
    def classes_of(self, iri: str) -> tuple[str, ...]: ...
    def members(self, cls: str) -> int: ...
    def member(self, cls: str, offset: int) -> str | None: ...
    def stale(self, iri: str) -> tuple[tuple[str, str], ...]: ...
    def facts(self, iri: str) -> dict[str, bool]: ...
    def own_classes(self, iri: str, membership: str) -> tuple[str, ...]: ...
    def class_size(self, cls: str, membership: str, exclude: str) -> int: ...
    def class_member(self, cls: str, membership: str, exclude: str, offset: int) -> str | None: ...


def store_data_node(facts: dict[str, bool]) -> bool:
    """Rule 32's store tests (amended text, change-log entry 26): (b) a membership target or an
    entity modelled as a class, with a data-graph triple outside RDF, RDFS and OWL, and (c) never
    a predicate."""
    return bool((facts["membership_target"] or facts["modelled_as_class"]) and facts["data_subject"]
                and not facts["used_as_predicate"])


@dataclass
class Context:
    graph: CooccurrenceGraph
    oracle: StoreOracle
    vocabulary: frozenset[str]  # the shared hierarchy's classes and the co-occurrence graph's nodes
    named_classes: frozenset[str]  # the shared hierarchy's classes and the co-occurrence graph's classes
    seed: int
    distances: tuple[int, ...] = (2, 4)  # swap strata on the census co-occurrence graph
    stale_map: dict[str, list[str]] = field(default_factory=dict)
    membership: str = RDF_TYPE  # the graph's membership predicate (GraphProfile.membership)

    def is_class(self, iri: str) -> bool:
        """Class against constant for an IRI in a class position, from source-neutral facts only."""
        if iri.startswith(META):
            return False
        if iri in self.named_classes:
            return True
        return not store_data_node(self.oracle.facts(iri))  # an absent IRI has no facts: a class (rule 37)


@dataclass
class Mutation:
    question_id: str
    operator: str
    fault_class: str  # by operator: vocabulary_absent, type_wrong, stale_in_data, meaning
    query: str
    edits: list[dict]
    slot: dict | None
    replacement: str | None
    selection: dict
    site: dict
    variant: str | None = None
    distance: int | None = None  # swaps: the hop distance drawn at
    stratum: str | None = None  # swaps: near (d = 2) or far (d = 4)
    realisable_query: str | None = None  # ASK: does the mutated pattern have a match in the store?

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Skip:
    question_id: str
    operator: str
    reason: str  # why no mutation of this kind exists for this question
    rejected: bool = False  # True when a mutation was attempted and failed (counted as a rejection)
    distance: int | None = None  # swaps: the stratum that had no draw
    detail: dict | None = None  # the type-consistent swap: the constant, and its tries when they ran out


def rng_for(seed: int, question_id: str, operator: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{question_id}|{operator}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _slot_json(s: Slot) -> dict:
    return {"role": s.role, "context": s.context, "start": s.start, "end": s.end, "text": s.text, "iri": s.iri,
            "position": s.position, "negated": s.scope.negated, "optional": s.scope.optional}


def _replacement_text(q: ParsedQuery, slot: Slot, iri: str) -> str:
    """Keep the gold's spelling: a full IRI stays a full IRI, a prefixed name stays prefixed."""
    if q.tokens[slot.token].kind == "iri":
        return f"<{iri}>"
    return iri_text(q, iri)


def _sites(q: ParsedQuery, role: str) -> list[Slot]:
    return [s for s in q.slots if s.role == role]


# -- the site record the conformance labeller reads ------------------------------------------------
def _visible(anchor: Scope, o: Pattern) -> bool:
    """Whether pattern ``o`` constrains nodes read from scope ``anchor``. Rule 17: a negated part
    (MINUS, NOT EXISTS) is checked on its own patterns and adds no implications outside it, so a
    pattern counts only within the same negated part, or outside every negated part when the
    anchor is outside them. A UNION branch's patterns hold only for that branch's solutions, so a
    pattern counts only when it lies in the anchor's own chain of branches (its branch or an
    enclosing group). Subqueries need nothing here: the reader gives a subquery's unprojected
    variables keys of their own. Approximate path steps are left out."""
    return (not o.complex_path and o.scope.negation == anchor.negation
            and o.scope.branch == anchor.branch[: len(o.scope.branch)])


def node_info(q: ParsedQuery, term: str, k: int | None, ctx: Context, skip: int | None = None,
              anchor: Scope | None = None) -> dict:
    anchor = q.patterns[k].scope if k is not None else anchor
    classes: list[str] = []
    uses: list[dict] = []
    for j, p in enumerate(q.patterns):
        if j == skip or (anchor is not None and not _visible(anchor, p)):
            continue
        if p.subject == term and p.predicate in ("http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                                                 "http://www.w3.org/2000/01/rdf-schema#subClassOf"):
            # data nodes in a class position (Rhea's reaction hierarchy) are not classes (rule 32)
            if not p.object.startswith(("?", "_:", "lit:")) and ctx.is_class(p.object):
                classes.append(p.object)
            continue
        if p.predicate.startswith("?"):
            continue
        if p.subject == term:
            uses.append({"predicate": p.predicate, "position": "subject"})
        if p.object == term and p.predicate not in ("http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                                                    "http://www.w3.org/2000/01/rdf-schema#subClassOf"):
            uses.append({"predicate": p.predicate, "position": "object"})
    info = {"term": term, "classes": sorted(set(classes)), "uses": sorted(uses, key=lambda u: (u["predicate"], u["position"]))}
    if not term.startswith(("?", "_:", "lit:")):
        info["store_classes"] = sorted(ctx.oracle.classes_of(term))
    return info


def build_site(q: ParsedQuery, slot: Slot, new_iri: str, ctx: Context) -> dict:
    if slot.role == "predicate" and slot.pattern is not None:
        p = q.patterns[slot.pattern]
        return {"kind": "predicate", "term": new_iri, "original": slot.iri, "complex_path": p.complex_path,
                "subject": node_info(q, p.subject, slot.pattern, ctx, skip=slot.pattern),
                "object": node_info(q, p.object, slot.pattern, ctx, skip=slot.pattern)}
    if slot.role == "class" and slot.pattern is not None:
        p = q.patterns[slot.pattern]
        return {"kind": "class", "term": new_iri, "original": slot.iri,
                "membership": {"predicate": p.predicate, "modifier": p.modifier},
                "store_facts": ctx.oracle.facts(new_iri),  # rule 32's data-node test reads them
                "node": node_info(q, p.subject, slot.pattern, ctx, skip=slot.pattern)}
    # a constant: its uses are the patterns it fills (or the VALUES variable's uses)
    uses: list[dict] = []
    if slot.pattern is not None:
        p = q.patterns[slot.pattern]
        if p.predicate not in ("http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                               "http://www.w3.org/2000/01/rdf-schema#subClassOf") and not p.predicate.startswith("?"):
            uses.append({"predicate": p.predicate, "position": slot.position})
        elif slot.position == "object":
            uses.append({"predicate": p.predicate, "position": "membership_object"})
    elif slot.variable is not None:
        uses = node_info(q, slot.variable, None, ctx, anchor=slot.scope)["uses"]
        uses += [{"predicate": p.predicate, "position": "membership_object"} for p in q.patterns
                 if p.object == slot.variable and p.predicate in (RDF_TYPE, RDFS_SUBCLASSOF) and _visible(slot.scope, p)]
    node = {"term": new_iri, "classes": [], "uses": uses, "store_classes": sorted(ctx.oracle.classes_of(new_iri)),
            "in_store": not ctx.oracle.absent(new_iri)}
    if any(u["position"] == "membership_object" for u in uses):  # rules 32 and 37 read these in a class position
        node["store_facts"] = ctx.oracle.facts(new_iri)
    return {"kind": "constant", "term": new_iri, "original": slot.iri, "context": slot.context, "node": node}


# -- operators -------------------------------------------------------------------------------------
def _single_edit(q: ParsedQuery, slot: Slot, new_iri: str) -> tuple[str, list[Edit]]:
    edits = [Edit(slot.start, slot.end, _replacement_text(q, slot, new_iri))]
    return apply_edits(q.text, edits), edits


def _mutation(qid: str, op: str, q: ParsedQuery, slot: Slot, new_iri: str, ctx: Context, selection: dict,
              variant: str | None = None, distance: int | None = None) -> Mutation:
    text, edits = _single_edit(q, slot, new_iri)
    return Mutation(qid, op, FAULT_CLASS[op], text, [asdict(e) for e in edits], _slot_json(slot), new_iri,
                    selection, build_site(q, slot, new_iri, ctx), variant, distance,
                    STRATA.get(distance) if distance is not None else None, realisable_query(q, slot, new_iri))


def _swap(qid: str, op: str, q: ParsedQuery, ctx: Context, rng: random.Random, hops: int) -> Mutation | Skip:
    role = op.split("_", 1)[1]
    sites = _sites(q, role)
    if not sites:
        return Skip(qid, op, f"no_{role}_slot", distance=hops)
    rng.shuffle(sites)
    reasons: list[str] = []
    for slot in sites:
        if role in ("predicate", "class"):
            candidates = ctx.graph.candidates(slot.iri, role, hops)
            if not candidates:
                reasons.append("not_in_census_graph" if ctx.graph.kind(slot.iri) is None else "no_candidate_at_distance")
                continue
            new = rng.choice(candidates)
            selection = {"hops": hops, "candidates": len(candidates), "anchor": slot.iri}
            return _mutation(qid, op, q, slot, new, ctx, selection, None, hops)
        # a constant
        if ctx.graph.kind(slot.iri) == "class":
            candidates = ctx.graph.candidates(slot.iri, "class", hops)
            if not candidates:
                reasons.append("no_candidate_at_distance")
                continue
            new = rng.choice(candidates)
            selection = {"hops": hops, "candidates": len(candidates), "anchor": slot.iri, "draw": "class_node"}
            return _mutation(qid, op, q, slot, new, ctx, selection, None, hops)
        anchors = tuple(c for c in ctx.oracle.classes_of(slot.iri) if c in ctx.graph.classes)
        if not anchors:
            reasons.append("constant_without_census_class")
            continue
        classes = ctx.graph.candidates_from(list(anchors), "class", hops)
        if not classes:
            reasons.append("no_candidate_at_distance")
            continue
        order = classes[:]
        rng.shuffle(order)
        for cls in order[:CONSTANT_CLASS_TRIES]:
            n = ctx.oracle.members(cls)
            if n == 0:
                continue
            offset = rng.randrange(n)
            new = ctx.oracle.member(cls, offset)
            if new is None or new == slot.iri:
                continue
            selection = {"hops": hops, "candidates": len(classes), "anchor": list(anchors), "class": cls,
                         "members": n, "offset": offset, "draw": "store_member"}
            return _mutation(qid, op, q, slot, new, ctx, selection, None, hops)
        reasons.append("no_member_drawn")
    distinct = sorted(set(reasons))
    return Skip(qid, op, distinct[0] if len(distinct) == 1 else "no_applicable_site:" + ",".join(distinct), distance=hops)


_TAIL = re.compile(r"^(.*?)(\d+)$")


def invented_candidates(iri: str, rng: random.Random) -> list[str]:
    """Plausible identifiers in the same namespace: a different number for a numbered local name,
    otherwise a fixed set of morphs of the local name, in a seeded order."""
    cut = max(iri.rfind("/"), iri.rfind("#"))
    ns, local = iri[: cut + 1], iri[cut + 1 :]
    m = _TAIL.match(local)
    if m:
        # same width first; dense identifier spaces (Rhea's five-digit reactions, ChEBI) then get a
        # number one digit longer, as a mistyped or hallucinated identifier can be
        head, digits = m.groups()
        out: list[str] = []
        for width in (len(digits), len(digits) + 1):
            lo, hi = (10 ** (width - 1), 10**width - 1) if width > 1 else (1, 9)
            mine: list[str] = []
            for _ in range(INVENTED_TRIES * 4):
                cand = f"{ns}{head}{rng.randint(lo, hi):0{width}d}"
                if cand != iri and cand not in mine:
                    mine.append(cand)
                if len(mine) == INVENTED_TRIES:
                    break
            out.extend(mine)
        return out
    cap = local[:1].upper() + local[1:]
    morphs = [f"has{cap}", f"{local}Of", f"{local}s", f"is{cap}", f"{local}Type", f"{local}Value", f"{local}Id", f"main{cap}"]
    morphs = [f"{ns}{m_}" for m_ in morphs if f"{ns}{m_}" != iri]
    rng.shuffle(morphs)
    return morphs


def _invented(qid: str, op: str, q: ParsedQuery, ctx: Context, rng: random.Random) -> Mutation | Skip:
    role = op.split("_", 1)[1]
    sites = _sites(q, role)
    if not sites:
        return Skip(qid, op, f"no_{role}_slot")
    slot = rng.choice(sites)
    tried = invented_candidates(slot.iri, rng)
    for cand in tried:
        if cand in ctx.vocabulary or not ctx.oracle.absent(cand):
            continue
        selection = {"tried": tried.index(cand) + 1}
        return _mutation(qid, op, q, slot, cand, ctx, selection, "invented")
    return Skip(qid, op, "invented_present", rejected=True)


def _stale(qid: str, op: str, q: ParsedQuery, ctx: Context, rng: random.Random) -> Mutation | Skip:
    role = op.split("_", 1)[1]
    sites = _sites(q, role)
    if not sites:
        return Skip(qid, op, f"no_{role}_slot")
    rng.shuffle(sites)
    for slot in sites:
        found = list(ctx.oracle.stale(slot.iri)) + [(s, "static_map") for s in ctx.stale_map.get(slot.iri, [])]
        found = sorted(set(found))
        if not found:
            continue
        old, tag = rng.choice(found)
        selection = {"candidates": len(found), "source": tag}
        return _mutation(qid, op, q, slot, old, ctx, selection, STALE_KIND.get(tag, tag))
    return Skip(qid, op, "no_stale_term")


def _drop_with_space(q: ParsedQuery, first: int, last: int) -> Edit:
    """Remove tokens first..last and the whitespace token right before them."""
    start = q.tokens[first].start
    if first > 0 and q.tokens[first - 1].kind == "space":
        start = q.tokens[first - 1].start
    return Edit(start, q.tokens[last].end, "")


def _aggregation(qid: str, op: str, q: ParsedQuery, ctx: Context, rng: random.Random) -> Mutation | Skip:
    if op == "agg_count_distinct":
        sites = [a for a in q.aggregates if a.func == "COUNT"]
        if not sites:
            return Skip(qid, op, "no_count")
        a = rng.choice(sites)
        if a.distinct is not None:
            t = q.tokens[a.distinct]
            end = q.tokens[a.distinct + 1].end if a.distinct + 1 < len(q.tokens) and q.tokens[a.distinct + 1].kind == "space" else t.end
            edits = [Edit(t.start, end, "")]
            change = "COUNT(DISTINCT x) -> COUNT(x)"
        else:
            p = q.tokens[a.open_paren]
            edits = [Edit(p.end, p.end, "DISTINCT ")]
            change = "COUNT(x) -> COUNT(DISTINCT x)"
    elif op == "agg_min_max":
        sites = [a for a in q.aggregates if a.func in ("MIN", "MAX")]
        if not sites:
            return Skip(qid, op, "no_min_max")
        a = rng.choice(sites)
        t = q.tokens[a.token]
        other = "MAX" if a.func == "MIN" else "MIN"
        edits = [Edit(t.start, t.end, other if t.text.isupper() else other.lower())]
        change = f"{a.func} -> {other}"
    else:
        keys = [(sid, k) for sid, sel in enumerate(q.selects) for k in sel.group_keys]
        if not keys:
            return Skip(qid, op, "no_group_by")
        # a key the enclosing query reads cannot be dropped without changing that query too, so the
        # mutation would carry more than the recorded fault: not applicable (review #3 of 3afbfa1)
        used = [_used_outside(q, q.selects[sid], k) for sid, k in keys]
        outside = sorted({k.variable for (_, k), u in zip(keys, used, strict=True) if u})
        sites = [site for site, u in zip(keys, used, strict=True) if not u]
        if not sites:
            return Skip(qid, op, "group_key_used_outside")
        sid, key = rng.choice(sites)
        sel = q.selects[sid]
        edits = []
        if len(sel.group_keys) == 1:
            edits.append(_drop_with_space(q, sel.group_by, key.last))
        else:
            edits.append(_drop_with_space(q, key.token, key.last))
        if key.variable is not None:
            for i in sel.projection:
                if "?" + q.tokens[i].text[1:] == key.variable:
                    edits.append(_drop_with_space(q, i, i))
            hit = [it for it in sel.order_items if key.variable in it.variables]
            if hit and len(hit) == len(sel.order_items):
                edits.append(_drop_with_space(q, sel.order_by, sel.order_items[-1].last))
            else:
                edits.extend(_drop_with_space(q, it.first, it.last) for it in hit)
        change = f"GROUP BY key {key.variable or 'expression'} dropped (and from the projection and ORDER BY)"
        a = None
    text = apply_edits(q.text, edits)
    selection = {"sites": len(sites), "change": change}
    if op == "agg_drop_group_key" and outside:
        selection["keys_used_outside"] = outside
    return Mutation(qid, op, FAULT_CLASS[op], text, [asdict(e) for e in edits], None, None, selection,
                    {"kind": "aggregation"})


def _used_outside(q: ParsedQuery, sel: Select, key: GroupKey) -> bool:
    """A subquery's GROUP BY key that the subquery projects and the query names outside the
    subquery (in an outer projection, ORDER BY or pattern). Conservative: any outside occurrence of
    the name counts, which can only remove mutations."""
    if not sel.nested or key.variable is None or sel.end is None:
        return False
    if not (sel.projection_all or any("?" + q.tokens[i].text[1:] == key.variable for i in sel.projection)):
        return False
    return any(t.kind == "var" and "?" + t.text[1:] == key.variable and not sel.token <= i <= sel.end
               for i, t in enumerate(q.tokens) if i >= q.body_start)


def mutate(op: str, question: dict, q: ParsedQuery, ctx: Context, distance: int | None = None) -> Mutation | Skip:
    qid = question["id"]
    if op in SWAPS:
        d = ctx.distances[0] if distance is None else distance
        return _swap(qid, op, q, ctx, rng_for(ctx.seed, qid, f"{op}@d{d}"), d)
    rng = rng_for(ctx.seed, qid, op)
    if op in INVENTED:
        return _invented(qid, op, q, ctx, rng)
    if op in STALE:
        return _stale(qid, op, q, ctx, rng)
    if op in AGGREGATION:
        return _aggregation(qid, op, q, ctx, rng)
    raise ValueError(f"unknown operator {op!r}")


def generate(question: dict, ctx: Context, operators: tuple[str, ...] = OPERATORS) -> list[Mutation | Skip]:
    """One mutation (or the reason there is none) per operator for one gold query; swaps once per
    distance stratum."""
    q = parse(question["gold_sparql"], ctx.is_class)
    out: list[Mutation | Skip] = []
    for op in operators:
        if op in SWAPS:
            out.extend(mutate(op, question, q, ctx, d) for d in ctx.distances)
        else:
            out.append(mutate(op, question, q, ctx))
    return out


# -- the type-consistent constant swap (a descriptive stratum) ------------------------------------------
def _own_classes(ctx: Context, iri: str) -> list[str]:
    return sorted(c for c in ctx.oracle.own_classes(iri, ctx.membership)
                  if c != iri and not c.startswith(META) and ctx.is_class(c))


def _constant_site(q: ParsedQuery, slots: list[Slot], new_iri: str, ctx: Context) -> dict:
    """The site of a constant replaced at every occurrence: the uses of all its slots."""
    site = build_site(q, slots[0], new_iri, ctx)
    if len(slots) > 1:
        uses = {(u["predicate"], u["position"]) for s in slots for u in build_site(q, s, new_iri, ctx)["node"]["uses"]}
        site["node"]["uses"] = [{"predicate": p, "position": pos} for p, pos in sorted(uses)]
    return site


def type_consistent_swaps(question: dict, q: ParsedQuery, ctx: Context,
                          effect: Callable[[str], dict]) -> list[Mutation | Skip]:
    """One item, or the reason there is none, per distinct constant of the gold query (in order of
    first occurrence). ``effect`` labels a mutated query's answer effect against the gold
    (``store.answer_effect``): only an answer-changing draw is kept. All the question's draws come
    from one stream, ``rng_for(seed, question id, operator)``, as the other operators' do."""
    qid, op = question["id"], TYPE_CONSISTENT
    groups: dict[str, list[Slot]] = {}
    for s in q.slots:
        if s.role == "constant":
            groups.setdefault(s.iri, []).append(s)
    if not groups:
        return [Skip(qid, op, "no_constant_slot")]
    rng = rng_for(ctx.seed, qid, op)
    out: list[Mutation | Skip] = []
    for index, (iri, slots) in enumerate(groups.items(), 1):
        detail = {"constant": iri, "constant_index": index, "occurrences": len(slots)}
        classes = _own_classes(ctx, iri)
        if not classes:
            out.append(Skip(qid, op, "constant_without_class", detail=detail))
            continue
        usable = sorted((n, c) for c in classes if (n := ctx.oracle.class_size(c, ctx.membership, iri)) > 0)
        if not usable:
            out.append(Skip(qid, op, "no_other_member", detail={**detail, "classes": classes}))
            continue
        size, cls = usable[0]
        offsets = rng.sample(range(size), min(TYPE_CONSISTENT_TRIES, size))
        tries: list[dict] = []
        for offset in offsets:
            new = ctx.oracle.class_member(cls, ctx.membership, iri, offset)
            if new is None or new == iri:  # the store moved under the count: a spent try, never a mutation
                tries.append({"offset": offset, "replacement": new, "outcome": "no_member"})
                continue
            edits = [Edit(s.start, s.end, _replacement_text(q, s, new)) for s in slots]
            text = apply_edits(q.text, edits)
            got = effect(text)
            label = got["label"] if got["label"] != "rejected" else f"rejected:{got.get('reason')}"
            tries.append({"offset": offset, "replacement": new,
                          "query_sha256": hashlib.sha256(text.encode()).hexdigest(), "outcome": label})
            if label != "answer_changing":
                continue
            selection = {**detail, "classes": classes, "class": cls, "members": size, "offset": offset, "tries": tries,
                         "replacement_classes": _own_classes(ctx, new)}
            out.append(Mutation(qid, op, FAULT_CLASS[op], text, [asdict(e) for e in edits], _slot_json(slots[0]), new,
                                selection, _constant_site(q, slots, new, ctx), None, None, DESCRIPTIVE_STRATUM[op],
                                # one ASK replaces one occurrence: a repeated constant is not assessed
                                realisable_query(q, slots[0], new) if len(slots) == 1 else None))
            break
        else:
            out.append(Skip(qid, op, "no_answer_changing_draw",
                            detail={**detail, "classes": classes, "class": cls, "members": size, "tries": tries}))
    return out


# -- realisable in the data (a descriptor) ---------------------------------------------------------
def _term_text(term: str) -> str:
    if term.startswith("?"):
        return term
    if term.startswith("_:"):
        return "?_" + term[2:]
    if term.startswith("lit:"):
        return term[4:].replace("^^", "^^<", 1) + ">" if "^^" in term else term[4:]
    return f"<{term}>"


def realisable_query(q: ParsedQuery, slot: Slot, new_iri: str) -> str | None:
    """An ASK for the mutated pattern: the pattern holding the replaced term, with the term
    replaced, joined with the patterns of the same scope that share a node with it. None when the
    term sits outside a triple pattern (a FILTER constant, a VALUES member with no pattern), or in an
    approximate path step."""
    if slot.pattern is not None:
        seeds = [slot.pattern]
    elif slot.variable is not None:
        seeds = [j for j, p in enumerate(q.patterns) if slot.variable in (p.subject, p.object) and _visible(slot.scope, p)]
    else:
        return None
    if not seeds or any(q.patterns[k].complex_path for k in seeds):
        return None
    scope = q.patterns[seeds[0]].scope

    def nodes(p) -> set[str]:
        return {t for t in (p.subject, p.object) if t.startswith(("?", "_:"))}

    chosen = set(seeds)
    shared = set().union(*(nodes(q.patterns[k]) for k in seeds)) - ({slot.variable} if slot.pattern is None else set())
    for j, p in enumerate(q.patterns):
        if j not in chosen and p.scope == scope and not p.complex_path and nodes(p) & shared:
            chosen.add(j)
    blocks = []
    for j in sorted(chosen):
        p = q.patterns[j]
        s, pred, o = p.subject, p.predicate, p.object
        if j == slot.pattern:
            if slot.position == "predicate":
                pred = new_iri
            elif slot.position == "subject":
                s = new_iri
            else:
                o = new_iri
        elif slot.pattern is None:  # a VALUES constant: bind its variable to the replacement
            s = new_iri if s == slot.variable else s
            o = new_iri if o == slot.variable else o
        pred_text = pred if pred.startswith("?") else f"<{pred}>{p.modifier}"
        triple = f"{_term_text(s)} {pred_text} {_term_text(o)} ."
        blocks.append(f"GRAPH <{p.scope.graph}> {{ {triple} }}" if p.scope.graph else triple)
    return "ASK { " + " ".join(blocks) + " }"
