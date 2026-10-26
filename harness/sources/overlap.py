"""Change-log entry 92's per-layer overlap of a generated draft with the census and the curated source.

Source structure only (no checker or test outcome). The layers, as entry 92 reports them for UniProt:

* domain and range: the draft's (property, side, class) pairs, one per named member of each rdfs:domain or rdfs:range
  expression (a union counts each member; datatype ranges are not pairs). Reported: the share of the draft's pairs
  that are the census's ("100% of their pairs are in the census"), the share of the census's pairs the draft keeps
  ("they keep 87-90% of its pairs"), the Jaccard index with each, and the same against the curated source. This
  reading reproduces entry 92's published figures on the files it names (tests/test_sources_overlap.py);
* properties and classes: the Jaccard index of the IRI sets, and the class counts ("159-196, against 1,272");
* IRIs: the share of the draft's class and property IRIs that the census has as a class or a property;
* disjointness: the pairs each source states (the census has no disjointness layer).
"""

from __future__ import annotations

from harness.sources.model import OntologySource

RULE = ("change-log entry 92's per-layer overlap: (property, side, class) pairs over the named members of every "
        "domain and range expression, datatype ranges left out; property and class IRI sets; the draft's class and "
        "property IRIs found in the census; disjointness pairs. Shares are of the draft's set (in_*) or of the "
        "reference's (*_kept); Jaccard = |A and B| / |A or B|")


def dr_pairs(src: OntologySource) -> set[tuple[str, str, str]]:
    return {(p, side, m) for p, info in src.properties.items() for side in ("domain", "range")
            for e in getattr(info, side) for m in e.members}


def _share(n: int, d: int) -> float | None:
    return n / d if d else None


def _jaccard(a: set, b: set) -> float | None:
    return len(a & b) / len(a | b) if a | b else None


def overlap(draft: OntologySource, census: OntologySource, curated: OntologySource) -> dict:
    d, c, k = dr_pairs(draft), dr_pairs(census), dr_pairs(curated)
    dp, cp, kp = set(draft.properties), set(census.properties), set(curated.properties)
    dc, cc, kc = set(draft.classes), set(census.classes), set(curated.classes)
    return {
        "domain_range": {
            "draft": len(d), "census": len(c), "curated": len(k), "in_census": len(d & c), "in_curated": len(d & k),
            "share_in_census": _share(len(d & c), len(d)), "census_kept": _share(len(d & c), len(c)),
            "jaccard_census": _jaccard(d, c),
            "share_in_curated": _share(len(d & k), len(d)), "curated_kept": _share(len(d & k), len(k)),
            "jaccard_curated": _jaccard(d, k),
        },
        "properties": {"draft": len(dp), "census": len(cp), "curated": len(kp),
                       "jaccard_census": _jaccard(dp, cp), "jaccard_curated": _jaccard(dp, kp)},
        "classes": {"draft": len(dc), "census": len(cc), "curated": len(kc),
                    "jaccard_census": _jaccard(dc, cc), "jaccard_curated": _jaccard(dc, kc)},
        "iris_in_census": _share(len((dc | dp) & (cc | cp)), len(dc | dp)),
        "disjointness": {"draft": len(draft.disjoint), "census": len(census.disjoint), "curated": len(curated.disjoint)},
    }


ROWS = (
    ("domain and range pairs", lambda o: o["domain_range"]["draft"]),
    ("  in the census (share of the draft's)", lambda o: o["domain_range"]["share_in_census"]),
    ("  census pairs kept (share of the census's)", lambda o: o["domain_range"]["census_kept"]),
    ("  Jaccard with the census", lambda o: o["domain_range"]["jaccard_census"]),
    ("  in the curated source (share of the draft's)", lambda o: o["domain_range"]["share_in_curated"]),
    ("  curated pairs kept (share of the curated source's)", lambda o: o["domain_range"]["curated_kept"]),
    ("  Jaccard with the curated source", lambda o: o["domain_range"]["jaccard_curated"]),
    ("properties", lambda o: o["properties"]["draft"]),
    ("  Jaccard with the census", lambda o: o["properties"]["jaccard_census"]),
    ("  Jaccard with the curated source", lambda o: o["properties"]["jaccard_curated"]),
    ("classes", lambda o: o["classes"]["draft"]),
    ("  Jaccard with the census", lambda o: o["classes"]["jaccard_census"]),
    ("  Jaccard with the curated source", lambda o: o["classes"]["jaccard_curated"]),
    ("class and property IRIs in the census (share)", lambda o: o["iris_in_census"]),
    ("disjointness pairs", lambda o: o["disjointness"]["draft"]),
)


def _cell(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:.3f}" if isinstance(v, float) else f"{v:,}"


def to_markdown(rows: dict[str, dict], census: OntologySource, curated: OntologySource) -> str:
    names = list(rows)
    out = [f"| layer | {' | '.join(names)} |", "|---|" + "---:|" * len(names)]
    out += [f"| {label} | {' | '.join(_cell(get(rows[n])) for n in names)} |" for label, get in ROWS]
    first = rows[names[0]] if names else None
    if first is not None:
        out.append("")
        out.append(f"References: {census.name} with {first['domain_range']['census']:,} pairs, "
                   f"{first['properties']['census']:,} properties, {first['classes']['census']:,} classes, "
                   f"{first['disjointness']['census']:,} disjointness pairs; {curated.name} with "
                   f"{first['domain_range']['curated']:,} pairs, {first['properties']['curated']:,} properties, "
                   f"{first['classes']['curated']:,} classes, {first['disjointness']['curated']:,} disjointness pairs.")
    return "\n".join(out) + "\n"
