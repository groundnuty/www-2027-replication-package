"""One renderer for every source: ontology -> prompt text, at a token budget.

Proposal §7.3 and check contract v2, rule 16: axioms are ordered by a fixed
priority and truncated deterministically at a token budget that is the same
for every source; which layers survive is reported.

Three stages (check contract v2, rules 16 and 33):

* own: the source's own terms (every term that is not a shared-hierarchy class), through the four
  layers: on Rhea, the curated source's Rhea schema; the census's Rhea classes and all its predicates;
* own statements on shared-hierarchy terms: what the source itself says about shared-hierarchy classes
  (a census's counts, curated labels and comments, a generated ontology's own superclasses or
  disjointness), only for the shared terms in scope;
* shared hierarchy: the shared-hierarchy classes the data uses and their ancestors, with their shared
  superclasses, the same text for every source (``SharedContext``);
* imported ontology files (rule 16 as amended, change-log entry 101; only when the owner's namespaces are given,
  ``harness.sources.imported``): the statements of files from another publisher, which the own pass then leaves out,
  restricted to the terms the source's own axioms or the data use, and their ancestors.

The layers, in priority order within a pass:

1. classes and hierarchy: class declarations with their superclasses, then
   property declarations (with their kind) and their super-properties;
2. domain and range (a union is written ``A or B``; several domain lines for
   one property are intersected, as in RDFS; datatype ranges are read as "any
   of");
3. disjointness between named classes;
4. labels and comments.

A census also carries counts (check contract v2, rule 20), and every count it holds is rendered, so
that the checker never uses a count the prompt and the generator lack: in layer 1, each class line ends
with its ``class_support`` and ``type_support``, and the literal patterns follow the property lines; in
layer 2, after the domain and range lines, one line per predicate position gives its untyped fillers,
then one line per (predicate, position, class or datatype) gives its count, largest first. A legend
opens the census's own stage. Other sources have none of these, so nothing is added.

Within a layer, classes come breadth-first by depth from the roots of the
source's own (in-scope) hierarchy, then by IRI; properties likewise over the
sub-property hierarchy; disjoint pairs by the positions of their classes.
Existential restrictions are not a layer.

A census is ordered by descending support instead, within each layer of its own two passes, so that a cut keeps
its most supported part (design ruling in the design log, change-log entry 22). The legend stays first; then each
line by the count the record gives it, ties by the line's text: a class line by its ``class_support``; a
predicate's declaration by the largest count the record gives the predicate (its untyped fillers and listed
classes at either position); a literal pattern by its subjects; a domain line by the largest count listed at the
predicate's subject position, a range line at its object position; an untyped or count line by its count. The
shared hierarchy (the third pass) keeps its order, the same text for every source.

Truncation keeps the longest prefix of the item sequence (stage by stage, layer by
layer) whose full text, header and truncation note included,
fits the budget in ``tokens.count_tokens``; items are never cut in the middle.
The text never names the source or its kind (the manifest does), so two
sources with the same content render to the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from harness.sources.closure import Hierarchy
from harness.sources.imported import RULE as IMPORTED_RULE
from harness.sources.imported import Imported, Split
from harness.sources.model import ClassExpr, OntologySource
from harness.sources.scope import Scope, class_hierarchy, compute_scope, property_hierarchy
from harness.sources.tokens import count_tokens, tokenizer_info
from harness.sources.vocab import OWL, RDF, RDFS, SKOS, XSD

DEFAULT_BUDGET = 4000

LAYERS = (
    ("classes", "Classes and hierarchy"),
    ("domain_range", "Domain and range"),
    ("disjointness", "Disjointness"),
    ("labels", "Labels and comments"),
)
PASSES = (
    ("own", ""),
    ("own_shared", "The source's statements on shared-hierarchy terms: "),
    ("shared", "Shared hierarchy (classes the data uses and their ancestors): "),
    ("imported", "Imported ontology files (terms the source's own axioms or the data use, and their ancestors): "),
)
BASE_PASSES = 3  # own, own_shared, shared: every render has them; the imported pass only with the owner's namespaces

# Fixed for every source; only the prefixes a render uses are declared in it.
PREFIX_TABLE = (
    ("CHEBI", "http://purl.obolibrary.org/obo/CHEBI_"),
    ("GO", "http://purl.obolibrary.org/obo/GO_"),
    ("obo", "http://purl.obolibrary.org/obo/"),
    ("rh", "http://rdf.rhea-db.org/"),
    ("up", "http://purl.uniprot.org/core/"),
    ("taxon", "http://purl.uniprot.org/taxonomy/"),
    ("keywords", "http://purl.uniprot.org/keywords/"),
    ("location", "http://purl.uniprot.org/locations/"),
    ("pathway", "http://purl.uniprot.org/pathways/"),
    ("ec", "http://purl.uniprot.org/enzyme/"),
    ("chemrof", "https://w3id.org/chemrof/"),
    ("oboInOwl", "http://www.geneontology.org/formats/oboInOwl#"),
    ("sio", "http://semanticscience.org/resource/"),
    ("dcterms", "http://purl.org/dc/terms/"),
    ("pv", "http://ld.company.org/prod-vocab/"),  # the corporate KG's own preferred prefix (its vann declaration)
    ("skos", SKOS),
    ("rdf", RDF),
    ("rdfs", RDFS),
    ("owl", OWL),
    ("xsd", XSD),
    ("x", "http://example.org/opaque/"),  # opaque names of a decontaminated input
)
_BY_LENGTH = sorted(PREFIX_TABLE, key=lambda p: -len(p[1]))
_LOCAL = re.compile(r"[A-Za-z0-9_]([A-Za-z0-9_.-]*[A-Za-z0-9_-])?\Z")

HEADER = "Ontology of the graph, in four layers. IRIs use the prefixes declared below."
KIND_WORD = {"object": "object property", "datatype": "datatype property", "annotation": "annotation property", "unknown": "property"}


def compact(iri: str) -> tuple[str, str | None]:
    for prefix, ns in _BY_LENGTH:
        if iri.startswith(ns) and _LOCAL.match(iri[len(ns):]):
            return f"{prefix}:{iri[len(ns):]}", prefix
    return f"<{iri}>", None


@dataclass(frozen=True)
class Item:
    stage: int  # pass * len(LAYERS) + layer
    text: str
    prefixes: tuple[str, ...]


@dataclass
class Render:
    text: str
    manifest: dict


class _Writer:
    def __init__(self):
        self.prefixes: set[str] = set()

    def iri(self, iri: str) -> str:
        out, prefix = compact(iri)
        if prefix:
            self.prefixes.add(prefix)
        return out

    def expr(self, e: ClassExpr) -> str:
        return " or ".join(self.iri(m) for m in e.members)

    def take(self) -> tuple[str, ...]:
        out = tuple(sorted(self.prefixes))
        self.prefixes = set()
        return out


def _counts(w: _Writer, counts: dict[str, int]) -> str:
    return ", ".join(f"{w.iri(k)} {n}" for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _convention(src: OntologySource) -> str:
    return RDFS + "subClassOf" if src.membership == "rdfs:subClassOf" else RDF + "type"


@dataclass(frozen=True)
class SharedContext:
    """The shared hierarchy as every prompt renders it (rule 33): its classes the data uses and their
    ancestors, with their shared superclasses; and which source terms are shared-hierarchy classes."""

    hierarchy: Hierarchy
    terms: frozenset[str]

    @property
    def classes(self) -> frozenset[str]:
        return frozenset(self.hierarchy.nodes)


def _split(src: OntologySource, imported: Imported | None) -> Split | None:
    """The origin-based split of ``src``'s axioms, or None when nothing of it is imported (renders as before)."""
    return Split(src, imported) if imported is not None and imported.parts and src.origin else None


def build_items(src: OntologySource, scope: Scope, shared: SharedContext | None = None,
                imported: Imported | None = None) -> list[Item]:
    w = _Writer()
    split = _split(src, imported)
    items: list[Item] = []
    shared_terms = shared.terms if shared else frozenset()
    ch = class_hierarchy(src)
    class_order = [c for c in ch.breadth_first(scope.classes) if c in src.classes]
    ph = property_hierarchy(src)
    prop_order = [p for p in ph.breadth_first(scope.properties) if p in src.properties]
    census = src.class_support is not None
    pos = {c: i for i, c in enumerate(class_order)}

    supports: list[float] = []  # per item; a census is ordered by it (inf: the legend, kept first)
    pos_support = src.position_support or {}
    un = src.untyped_fillers or {}

    def add(stage: int, layer: int, text: str, support: float = 0) -> None:
        items.append(Item(stage * len(LAYERS) + layer, text, w.take()))
        supports.append(support)

    def at(p: str, position: str) -> int:
        return max(pos_support.get(p, {}).get(position, {}).values(), default=0)

    def predicate_support(p: str) -> int:
        return max(at(p, "subject"), at(p, "object"), *un.get(p, {}).values(), 0)

    def class_line(c: str, parents: list[str]) -> str:
        line = f"class {w.iri(c)}"
        if parents:
            line += " subClassOf " + ", ".join(w.iri(p) for p in parents)
        if census:
            line += f" [class_support {src.class_support.get(c, 0)}, type_support {(src.type_support or {}).get(c, 0)}]"
        return line

    own_classes = [c for c in class_order if c not in shared_terms and not (split and split.imported_class(c))]
    shared_classes = [c for c in class_order if c in shared_terms]
    own_props = [p for p in prop_order if not (split and split.imported_property(p))]

    # Stage 1: the source's own terms, through every layer.
    if census:
        add(0, 0, (
            "Counts (triples unless stated): class_support = nodes that are members of the class, counted up "
            f"the subclass chain; type_support = subjects of the literal {w.iri(_convention(src))} triple to the "
            "class; per predicate position, 'untyped' counts the fillers with no class, and the 'subject' and "
            "'object' lines give the most specific classes (and datatypes) whose count there meets the support "
            "threshold, counted up the shared hierarchy, so a shared superclass is given when only its subclasses "
            "together meet it; the superclasses of a given class also meet it and are not repeated; 'literal' "
            f"lines count subjects of an asserted triple. Support threshold {src.support_threshold}."
        ), float("inf"))
    for c in own_classes:
        add(0, 0, class_line(c, [p for p in src.classes[c].parents if p in scope.classes
                                 and not (split and split.imported_edge(c, p))]),
            (src.class_support or {}).get(c, 0))
    reach = _reach(src, split) if split else None
    if split:  # the owner's own superclass axioms on classes an imported file declares stay in the own pass, also on
        # one the restriction reaches outside the render scope (a shared-hierarchy class the data does not use)
        extra = [c for c in ch.breadth_first(reach[0]) if c not in pos]
        for c in [*class_order, *extra]:
            if (c not in shared_terms or c not in pos) and split.imported_class(c):
                ps = [p for p in src.classes[c].parents if (p in scope.classes or p in reach[0])
                      and not split.imported_edge(c, p)]
                if ps:
                    add(0, 0, class_line(c, ps))
    for p in own_props:
        info = src.properties[p]
        line = f"{KIND_WORD[info.kind]} {w.iri(p)}"
        sup = [q for q in info.parents if q in scope.properties]
        if sup:
            line += " subPropertyOf " + ", ".join(w.iri(q) for q in sup)
        add(0, 0, line, predicate_support(p))
    for pat in src.patterns:
        if pat.reading == "literal" and pat.object_class and pat.object_class not in shared_terms:
            add(0, 0, f"literal {w.iri(pat.predicate)} {w.iri(pat.object_class)}: {pat.support} subjects",
                pat.support)
    # Every domain and range axiom of an in-scope property is rendered whole.
    for p in prop_order:
        info = src.properties[p]
        for e in info.domain:
            if not (split and split.imported_expr("domain", p, e.members)):
                add(0, 1, f"{w.iri(p)} domain {w.expr(e)}", at(p, "subject"))
        for e in info.range:
            if not (split and split.imported_expr("range", p, e.members)):
                add(0, 1, f"{w.iri(p)} range {w.expr(e)}", at(p, "object"))
        dts = [d for d in info.datatype_range if not (split and split.imported_datatype(p, d))]
        if dts:
            add(0, 1, f"{w.iri(p)} range " + " or ".join(w.iri(d) for d in dts), at(p, "object"))
    if census:
        for p in prop_order:
            for position in ("subject", "object"):
                n = un.get(p, {}).get(position, 0)
                add(0, 1, f"{w.iri(p)} {position}: untyped {n}", n)
        counts = [(n, p, position, c) for p, d in pos_support.items() if p in scope.properties
                  for position, cs in d.items() for c, n in cs.items()]
        for n, p, position, c in sorted(counts, key=lambda x: (-x[0], x[1], x[2], x[3])):
            add(0, 1, f"{w.iri(p)} {position}: {w.iri(c)} {n}", n)
    for a, b in sorted((tuple(sorted((a, b), key=lambda x: pos[x])) for a, b in src.disjoint if a in pos and b in pos),
                       key=lambda ab: (pos[ab[0]], pos[ab[1]])):
        if (a not in shared_terms or b not in shared_terms) and not (split and split.imported_pair(a, b)):
            add(0, 2, f"{w.iri(a)} disjointWith {w.iri(b)}")
    for term, info in [(c, src.classes[c]) for c in own_classes] + [(p, src.properties[p]) for p in own_props]:
        for label in info.labels:
            add(0, 3, f"{w.iri(term)} label {json.dumps(label, ensure_ascii=False)}")
        for comment in info.comments:
            add(0, 3, f"{w.iri(term)} comment {json.dumps(comment, ensure_ascii=False)}")
    # Terms the source only describes keep their labels and comments here, after its classes and properties
    # (design ruling in the design log Oct 23:02); a source without any renders exactly as before.
    for term, d in sorted((src.described or {}).items()):
        for label in d.get("labels", ()):
            add(0, 3, f"{w.iri(term)} label {json.dumps(label, ensure_ascii=False)}")
        for comment in d.get("comments", ()):
            add(0, 3, f"{w.iri(term)} comment {json.dumps(comment, ensure_ascii=False)}")

    # Stage 2: the source's own statements on shared-hierarchy terms (superclasses the shared hierarchy
    # already has are left to stage 3).
    for c in shared_classes:
        given = shared.hierarchy.parents.get(c, ()) if shared else ()
        extra = [p for p in src.classes[c].parents if p in scope.classes and p not in given
                 and not (split and split.imported_edge(c, p))]  # an imported file's axioms wait for the imported pass
        if census or extra:
            add(1, 0, class_line(c, extra), (src.class_support or {}).get(c, 0))
    for pat in src.patterns:
        if pat.reading == "literal" and pat.object_class in shared_terms:
            add(1, 0, f"literal {w.iri(pat.predicate)} {w.iri(pat.object_class)}: {pat.support} subjects",
                pat.support)
    for a, b in sorted((tuple(sorted((a, b), key=lambda x: pos[x])) for a, b in src.disjoint if a in pos and b in pos),
                       key=lambda ab: (pos[ab[0]], pos[ab[1]])):
        if a in shared_terms and b in shared_terms and not (split and split.imported_pair(a, b)):
            add(1, 2, f"{w.iri(a)} disjointWith {w.iri(b)}")
    for c in shared_classes:
        if split and split.imported_class(c):
            continue  # an imported declaration's labels go with it, to the imported pass
        for label in src.classes[c].labels:
            add(1, 3, f"{w.iri(c)} label {json.dumps(label, ensure_ascii=False)}")
        for comment in src.classes[c].comments:
            add(1, 3, f"{w.iri(c)} comment {json.dumps(comment, ensure_ascii=False)}")

    # Stage 3: the shared hierarchy, the same for every source.
    if shared:
        for c in shared.hierarchy.breadth_first():
            ps = list(shared.hierarchy.parents.get(c, ()))
            add(2, 0, f"class {w.iri(c)}" + (" subClassOf " + ", ".join(w.iri(p) for p in ps) if ps else ""))
    if split:  # the imported files' statements, after the source's own layers (rule 16 as amended)
        _imported_items(src, split, scope, class_order, prop_order, own_classes, own_props, shared_classes, reach, w,
                        add)
    if census:  # the census's own two passes, by descending support within each layer; the shared pass as it is
        own = 2 * len(LAYERS)
        order = sorted(range(len(items)), key=lambda i: (items[i].stage, 0, -supports[i], items[i].text)
                       if items[i].stage < own else (items[i].stage, 1, i, ""))
        items = [items[i] for i in order]
    return items


def _reach(src: OntologySource, split: Split) -> tuple[set[str], set[str]]:
    """The classes and properties the imported pass may name: the terms the source's own axioms or the data use, and
    their ancestors (classes through the source's subclass axioms, properties through its sub-property axioms)."""
    seeds = split.own_terms() | split.imp.used
    return (class_hierarchy(src).upward_closure(t for t in seeds if t in src.classes),
            property_hierarchy(src).upward_closure(t for t in seeds if t in src.properties))


def _imported_items(src, split: Split, scope: Scope, class_order, prop_order, own_classes, own_props, shared_classes,
                    reach, w, add) -> None:
    """The imported pass: imported statements on the terms the source's own axioms or the data use, and on their
    ancestors in the source's hierarchy (classes through its subclass axioms, properties through its sub-property
    axioms), each layer in the own pass's order. A statement naming a term outside that restriction is left out.
    Sub-property axioms, which per-axiom origin does not record, go with the property's declaration."""
    stage = BASE_PASSES
    reach_c, reach_p = reach
    # In the own pass's order; a class the restriction reaches outside the render scope (a shared-hierarchy class
    # the data does not use, named by an owner's axiom) after them, in the hierarchy's order.
    in_scope = set(class_order)
    imp_classes = [c for c in class_order if split.imported_class(c) and c in reach_c]
    extra = [c for c in class_hierarchy(src).breadth_first(reach_c) if split.imported_class(c) and c not in in_scope]
    imp_classes += extra
    order = {c: i for i, c in enumerate([*class_order, *extra])}
    imp_props = [p for p in prop_order if split.imported_property(p) and p in reach_p]
    shown_c = set(own_classes) | set(imp_classes) | set(shared_classes)
    shown_p = set(own_props) | set(imp_props)
    shared = set(shared_classes)
    for c in imp_classes:
        ps = [p for p in src.classes[c].parents if p in shown_c and split.imported_edge(c, p)]
        if c not in shared:  # a shared-hierarchy class is declared in the shared pass
            add(stage, 0, f"class {w.iri(c)}" + (" subClassOf " + ", ".join(w.iri(p) for p in ps) if ps else ""))
        elif ps:
            add(stage, 0, f"class {w.iri(c)} subClassOf " + ", ".join(w.iri(p) for p in ps))
    for c in (c for c in class_order if not split.imported_class(c)):  # an imported file's axioms on the owner's classes
        ps = [p for p in src.classes[c].parents if p in shown_c and split.imported_edge(c, p)]
        if ps:
            add(stage, 0, f"class {w.iri(c)} subClassOf " + ", ".join(w.iri(p) for p in ps))
    for p in imp_props:
        info = src.properties[p]
        sup = [q for q in info.parents if q in scope.properties]
        add(stage, 0, f"{KIND_WORD[info.kind]} {w.iri(p)}" + (" subPropertyOf " + ", ".join(w.iri(q) for q in sup)
                                                              if sup else ""))
    for p in prop_order:
        if p not in shown_p:
            continue
        info = src.properties[p]
        for side in ("domain", "range"):
            for e in getattr(info, side):
                if split.imported_expr(side, p, e.members) and all(m in shown_c for m in e.members):
                    add(stage, 1, f"{w.iri(p)} {side} {w.expr(e)}")
        dts = [d for d in info.datatype_range if split.imported_datatype(p, d)]
        if dts:
            add(stage, 1, f"{w.iri(p)} range " + " or ".join(w.iri(d) for d in dts))
    for a, b in sorted((tuple(sorted((a, b), key=lambda x: order[x])) for a, b in src.disjoint
                        if a in shown_c and b in shown_c and split.imported_pair(a, b)),
                       key=lambda ab: (order[ab[0]], order[ab[1]])):
        add(stage, 2, f"{w.iri(a)} disjointWith {w.iri(b)}")
    for term, info in [(c, src.classes[c]) for c in imp_classes] + [(p, src.properties[p]) for p in imp_props]:
        for label in info.labels:
            add(stage, 3, f"{w.iri(term)} label {json.dumps(label, ensure_ascii=False)}")
        for comment in info.comments:
            add(stage, 3, f"{w.iri(term)} comment {json.dumps(comment, ensure_ascii=False)}")


N_STAGES = len(PASSES) * len(LAYERS)
IMPORTED_STAGES = range(BASE_PASSES * len(LAYERS), N_STAGES)


def _title(stage: int) -> str:
    pi, layer = divmod(stage, len(LAYERS))
    return f"{PASSES[pi][1]}{layer + 1}. {LAYERS[layer][1]}"


def _compose(items: list[Item], n: int, totals: list[int]) -> str:
    kept = items[:n]
    prefixes = sorted({p for it in kept for p in it.prefixes})
    ns = dict(PREFIX_TABLE)
    out = [HEADER] + [f"PREFIX {p}: <{ns[p]}>" for p in prefixes]
    stage = -1
    for it in kept:
        if it.stage != stage:
            stage = it.stage
            out.append("")
            out.append(f"## {_title(stage)}")
        out.append(it.text)
    if n < len(items):
        shown = [sum(1 for it in kept if it.stage == i) for i in range(N_STAGES)]
        parts = [f"{_title(i).lower()} {shown[i]} of {totals[i]} items"
                 for i in range(N_STAGES) if totals[i] and shown[i] < totals[i]]
        out.append("")
        out.append("[Truncated at the token budget; shown: " + "; ".join(parts) + ".]")
    return "\n".join(out) + "\n"


def own_content_text(src: OntologySource, scope: Scope | None = None, shared: SharedContext | None = None,
                     imported: Imported | None = None) -> str:
    """The source's own content alone, rendered in full: the first pass (its own terms through every layer), with
    the header and only the prefixes it uses, and nothing after it. The prompt budget of a graph is the curated
    source's measured this way, rounded up (design ruling in the design log, change-log entry 20). With the owner's
    namespaces, the imported pass counts too: entry 101 moves imported statements in the prompt, it does not change
    what the source states."""
    items = [it for it in build_items(src, scope or compute_scope(src, None), shared, imported)
             if it.stage < len(LAYERS) or it.stage in IMPORTED_STAGES]
    totals = [sum(1 for it in items if it.stage == i) for i in range(N_STAGES)]
    return _compose(items, len(items), totals)


def render(src: OntologySource, budget: int = DEFAULT_BUDGET, scope: Scope | None = None,
           shared: SharedContext | None = None, imported: Imported | None = None) -> Render:
    scope = scope or compute_scope(src, None)
    items = build_items(src, scope, shared, imported)
    split = _split(src, imported)
    totals = [sum(1 for it in items if it.stage == i) for i in range(N_STAGES)]

    def fits(n: int) -> bool:
        return count_tokens(_compose(items, n, totals)) <= budget

    if not fits(0):
        raise ValueError(f"budget {budget} is below the header and truncation note alone")
    if fits(len(items)):
        n = len(items)
    else:
        # Upper bound from per-item counts (each item costs at least its own tokens minus one).
        lo, hi, acc = 0, len(items), 0
        for i, it in enumerate(items):
            acc += max(count_tokens(it.text) - 1, 1)
            if acc > budget:
                hi = i + 1
                break
        while lo < hi:  # largest n in [lo, hi] that fits
            mid = (lo + hi + 1) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid - 1
        n = lo
    text = _compose(items, n, totals)
    kept = items[:n]
    layers = []
    for i in range(N_STAGES if split else BASE_PASSES * len(LAYERS)):  # the imported pass only where it applies
        pi, layer = divmod(i, len(LAYERS))
        shown = sum(1 for it in kept if it.stage == i)
        full_text = "\n".join([f"## {_title(i)}"] + [it.text for it in items if it.stage == i])
        status = "empty" if totals[i] == 0 else "full" if shown == totals[i] else "partial" if shown else "none"
        layers.append({
            "pass": PASSES[pi][0], "layer": layer + 1, "key": f"{PASSES[pi][0]}.{LAYERS[layer][0]}",
            "title": _title(i), "status": status, "items_total": totals[i], "items_shown": shown,
            "tokens_full_layer": count_tokens(full_text) if totals[i] else 0,
        })
    manifest = {
        "source": {"name": src.name, "kind": src.kind, "graph": src.graph, "digest": src.digest()},
        "scope": {
            "used_terms_digest": scope.used_digest,
            "classes": len(scope.classes),
            "properties": len(scope.properties),
            "shared_terms_in_scope": sum(1 for t in scope.classes if shared and t in shared.terms),
            "shared_hierarchy_classes": len(shared.classes) if shared else 0,
        },
        "budget_tokens": budget,
        "tokens": count_tokens(text),
        "tokenizer": tokenizer_info(),
        "truncated": n < len(items),
        "items_total": len(items),
        "items_shown": n,
        "layers": layers,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    if split:
        manifest["imported"] = {"rule": IMPORTED_RULE, "owner_namespaces": list(imported.owner_namespaces),
                                "parts": sorted(imported.parts), "used_terms": len(imported.used)}
    return Render(text, manifest)
