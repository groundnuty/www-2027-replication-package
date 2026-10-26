"""Profile of each source (proposal §7.6, "Ontology sources"; issue E0.3 criterion 5).

Counts are taken inside the source's scope (terms the data uses plus their
ancestors, ``scope``), the same restriction the renderer applies, so a large
imported ontology is profiled only where the data touches it. Whole-source
counts are given beside them. The share of the predicates the data uses (read
from the store) that have a declared domain and a declared range is computed
over every predicate used in the layout's graphs except the two membership
predicates (rdf:type, rdfs:subClassOf).
"""

from __future__ import annotations

from collections.abc import Iterable

from harness.sources.model import OntologySource
from harness.sources.scope import Scope

MEMBERSHIP_PREDICATES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
}


def _axiom_counts(src: OntologySource, classes: set[str] | frozenset[str], props: set[str] | frozenset[str]) -> dict:
    named_dom = union_dom = named_rng = union_rng = dt_rng = 0
    for p in props:
        info = src.properties[p]
        named_dom += sum(1 for e in info.domain if not e.is_union)
        union_dom += sum(1 for e in info.domain if e.is_union)
        named_rng += sum(1 for e in info.range if not e.is_union)
        union_rng += sum(1 for e in info.range if e.is_union)
        dt_rng += 1 if info.datatype_range else 0
    return {
        "classes": len(classes),
        "subclass_axioms": sum(sum(1 for q in src.classes[c].parents if q in classes) for c in classes),
        "properties": len(props),
        "domain_named": named_dom,
        "domain_union": union_dom,
        "range_named": named_rng,
        "range_union": union_rng,
        "datatype_ranges": dt_rng,
        "disjoint_pairs": sum(1 for a, b in src.disjoint if a in classes and b in classes),
        "existentials": sum(1 for e in src.existentials if e.cls in classes),
        "labelled_terms": sum(1 for c in classes if src.classes[c].labels) + sum(1 for p in props if src.properties[p].labels),
        "commented_terms": sum(1 for c in classes if src.classes[c].comments) + sum(1 for p in props if src.properties[p].comments),
    }


def profile(src: OntologySource, scope: Scope, used_predicates: Iterable[str]) -> dict:
    used = sorted(set(used_predicates) - MEMBERSHIP_PREDICATES)
    with_dom = [p for p in used if p in src.properties and src.properties[p].domain]
    with_rng = [p for p in used if p in src.properties and (src.properties[p].range or src.properties[p].datatype_range)]
    with_both = sorted(set(with_dom) & set(with_rng))
    declared = [p for p in used if p in src.properties]
    n = len(used)
    return {
        "source": src.name,
        "kind": src.kind,
        "digest": src.digest(),
        "in_scope": _axiom_counts(src, scope.classes, scope.properties),
        "whole_source": _axiom_counts(src, frozenset(src.classes), frozenset(src.properties)),
        "unsupported": dict(sorted(src.unsupported.items())),
        "unsupported_total": sum(src.unsupported.values()),
        "used_predicates": {
            "count": n,
            "declared": len(declared),
            "with_domain": len(with_dom),
            "with_range": len(with_rng),
            "with_domain_and_range": len(with_both),
            "share_with_domain": round(len(with_dom) / n, 4) if n else None,
            "share_with_range": round(len(with_rng) / n, 4) if n else None,
            "share_with_domain_and_range": round(len(with_both) / n, 4) if n else None,
        },
        "scope_used_terms_digest": scope.used_digest,
    }


ROWS = (
    ("Classes", "classes"),
    ("Subclass axioms", "subclass_axioms"),
    ("Properties", "properties"),
    ("Domain declarations, named class", "domain_named"),
    ("Domain declarations, union", "domain_union"),
    ("Range declarations, named class", "range_named"),
    ("Range declarations, union", "range_union"),
    ("Datatype ranges", "datatype_ranges"),
    ("Disjointness axioms (pairs)", "disjoint_pairs"),
    ("Existential restrictions", "existentials"),
    ("Terms with a label", "labelled_terms"),
    ("Terms with a comment", "commented_terms"),
)


def _undeclared_markdown(undeclared: dict) -> list[str]:
    out = ["", "## Terms the data uses that the curated source does not declare", "",
           undeclared["rule"] + ". Reported, never patched (check contract v2, rule 15).", ""]
    for title, key in (("Predicates", "predicates"), ("Classes used as rdf:type objects", "classes")):
        out.append(f"### {title}")
        out.append("")
        rows = [(g, t, n) for g, d in undeclared[key].items() for t, n in d.items()]
        if not rows:
            out.append("None.")
        else:
            out.append("| graph | term | triples |")
            out.append("|---|---|---|")
            out.extend(f"| {g} | {t} | {n:,} |" for g, t, n in sorted(rows, key=lambda r: (r[0], -r[2], r[1])))
        out.append("")
    return out


def to_markdown(profiles: list[dict], notes: list[str] | None = None, undeclared: dict | None = None,
                remarks: list[str] | None = None) -> str:
    """``remarks``: paragraphs on the used predicates, printed below their count."""
    names = [p["source"] for p in profiles]
    out = ["# Ontology source profiles", ""]
    out.append("Counts inside each source's scope (terms the data uses, plus their ancestors); whole-source counts in brackets where they differ.")
    out.append("")
    out.append("| | " + " | ".join(names) + " |")
    out.append("|---|" + "---|" * len(names))
    for title, key in ROWS:
        cells = []
        for p in profiles:
            a, b = p["in_scope"][key], p["whole_source"][key]
            cells.append(f"{a:,}" if a == b else f"{a:,} ({b:,})")
        out.append(f"| {title} | " + " | ".join(cells) + " |")
    cells = [f"{p['unsupported_total']:,}" for p in profiles]
    out.append("| Unsupported constructs (whole source) | " + " | ".join(cells) + " |")
    u = profiles[0]["used_predicates"]["count"] if profiles else 0
    for title, key in (("domain", "with_domain"), ("range", "with_range"), ("domain and range", "with_domain_and_range")):
        cells = []
        for p in profiles:
            up = p["used_predicates"]
            share = up[key] / up["count"] if up["count"] else 0
            cells.append(f"{up[key]}/{up['count']} ({share:.0%})")
        out.append(f"| Used predicates with a declared {title} | " + " | ".join(cells) + " |")
    out.append("")
    out.append(f"Used predicates: the {u} predicates of the store's graphs other than rdf:type and rdfs:subClassOf.")
    out.append("")
    for remark in remarks or ():
        out += [remark, ""]
    out.append("## Unsupported constructs, by source")
    out.append("")
    for p in profiles:
        items = ", ".join(f"{k} {v:,}" for k, v in p["unsupported"].items()) or "none"
        out.append(f"- **{p['source']}**: {items}")
    with_overlap = [p for p in profiles if p.get("overlap_with_curated")]
    if with_overlap:
        out += ["", "## Overlap with the curated ontology (generated sources)", "",
                "Terms in common: the curated ontology's own classes and properties that the generated source "
                "declares under the same IRI. Axioms in common: per axiom type, the share of the curated ontology's own "
                "axioms the generated source reproduces (subclass entailed in its own hierarchy; domain, range and "
                "disjointness stated identically); 'all' sums the axiom rows, terms excluded "
                "(`profile.overlap_with_curated`; the contamination diagnostic of design log).", "",
                "| | " + " | ".join(p["source"] for p in with_overlap) + " |",
                "|---|" + "---|" * len(with_overlap)]
        for key in ("terms", "subclass", "domain", "range", "datatype_range", "disjointness", "all"):
            cells = []
            for p in with_overlap:
                r = p["overlap_with_curated"][key]
                cells.append(f"{r['reproduced']}/{r['curated']}" + (f" ({r['share']:.0%})" if r["share"] is not None else ""))
            out.append(f"| {key} | " + " | ".join(cells) + " |")
    if undeclared:
        out.extend(_undeclared_markdown(undeclared))
    if notes:
        out.append("")
        out.append("## Notes")
        out.append("")
        out.extend(f"- {n}" for n in notes)
    out.append("")
    return "\n".join(out)


def overlap_with_curated(curated: OntologySource, generated: OntologySource, exclude: set[str] | frozenset[str] = frozenset()) -> dict:
    """Design log, criterion 5 (the contamination diagnostic): the share of the curated ontology's own axioms that a
    generated source reproduces. Curated axioms are counted over the curated source's own terms (``exclude``: the
    shared-hierarchy terms, which are evidence for every source and not the curated ontology's own):
      terms      the curated ontology's own classes and properties that the generated source declares (as a class
                 or property, under the same IRI);
      subclass   C subClassOf P between own classes: reproduced when P is among C's ancestors in the generated
                 source's own hierarchy (entailed there, not only stated);
      domain / range   (property, class expression): reproduced when the generated source states the same
                 expression (same member set) for that property;
      datatype range   (property, datatype): reproduced when the generated source lists that datatype;
      disjointness   an unordered pair: reproduced when the generated source states the pair.
    """
    from harness.sources.scope import class_hierarchy

    gh = class_hierarchy(generated)
    rows = {}

    def tally(name, axioms, hit):
        axioms = sorted(axioms)
        n = sum(1 for a in axioms if hit(a))
        rows[name] = {"curated": len(axioms), "reproduced": n, "share": round(n / len(axioms), 4) if axioms else None}

    own_c = {c for c in curated.classes if c not in exclude}
    own_p = {p for p in curated.properties if p not in exclude}
    tally("terms", own_c | own_p, lambda t: t in generated.classes or t in generated.properties)
    sub = {(c, p) for c in own_c for p in curated.classes[c].parents if p in own_c}
    tally("subclass", sub, lambda a: a[0] in gh.parents and a[1] in gh.ancestors(a[0]))
    for side in ("domain", "range"):
        ax = {(p, tuple(sorted(e.members))) for p in own_p for e in getattr(curated.properties[p], side)}
        tally(side, ax, lambda a, side=side: a[0] in generated.properties and any(
            tuple(sorted(e.members)) == a[1] for e in getattr(generated.properties[a[0]], side)))
    dt = {(p, d) for p in own_p for d in curated.properties[p].datatype_range}
    tally("datatype_range", dt, lambda a: a[0] in generated.properties and a[1] in generated.properties[a[0]].datatype_range)
    dj = {tuple(sorted(x)) for x in curated.disjoint if x[0] in own_c and x[1] in own_c}
    gdj = {tuple(sorted(x)) for x in generated.disjoint}
    tally("disjointness", dj, lambda a: a in gdj)
    total = sum(r["curated"] for k, r in rows.items() if k != "terms")
    hit = sum(r["reproduced"] for k, r in rows.items() if k != "terms")
    rows["all"] = {"curated": total, "reproduced": hit, "share": round(hit / total, 4) if total else None}
    return rows
