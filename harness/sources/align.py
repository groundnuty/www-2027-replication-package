"""The aligned curated source: a curated source with a reviewed SSSOM mapping applied to it.

The rule (design ruling in the design log Oct 12:54): every curated IRI that the reviewed mapping matches exactly
(``skos:exactMatch``) to a data IRI is replaced by that data IRI everywhere the source names a term: class and property
declarations, superclasses and super-properties, domain and range members, datatype and enumerated ranges, disjointness
pairs, existentials, declared and described terms, and the per-axiom origin record. A renamed term keeps its labels and
comments, which are text and stay as written. Terms the mapping does not match stay as published; data terms with no
mapping stay undeclared. Refused: a mapping file whose sha256 is not the reviewed file's; anything but a one-to-one
exact match; a pair whose kinds differ or are unknown (the curated side's kind from the source's declarations, class or
property; the data side's from the table's ``kind`` column, class or predicate); and a renaming onto a term the source
already names. Recorded: counts only (terms renamed, axioms touched by kind). Works on the source's JSON form, so a
field the curated builder adds later is carried unchanged unless it names terms, in which case it must be added here.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
from pathlib import Path

EXACT = "skos:exactMatch"
RULE = ("design ruling in the design log Oct 12:54: each curated IRI the reviewed SSSOM mapping matches exactly "
        "(skos:exactMatch) to a data IRI is renamed to it everywhere the source names a term, per-axiom origin "
        "included, keeping its labels and comments; unmapped terms stay as published; the mapping must be the reviewed "
        "file (sha256), one-to-one, kind-preserving (class to class, property to predicate) and collision-free")
# The data side's kind in the reviewed table, by the curated side's kind it must match.
KINDS = {"class": "class", "property": "predicate"}
AXIOM_KINDS = ("classes", "properties", "subclass", "subproperty", "disjoint", "domain", "range", "datatype_range",
               "range_values", "existentials")


class AlignmentRefused(ValueError):
    pass


def read_curie_map(path: Path) -> dict[str, str]:
    """The prefixes of an SSSOM curie_map.yaml in its flat form (``curie_map:`` then ``  prefix: IRI`` lines); any
    other form is refused rather than guessed."""
    out: dict[str, str] = {}
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines or lines[0].strip() != "curie_map:":
        raise AlignmentRefused("the curie map is not in the flat curie_map form")
    for ln in lines[1:]:
        if not ln.startswith(" ") or ":" not in ln:
            raise AlignmentRefused("the curie map has a line outside the flat curie_map form")
        prefix, iri = ln.strip().split(":", 1)
        out[prefix.strip()] = iri.strip().strip("'\"")
    return out


def _expand(curie: str, prefixes: dict[str, str]) -> str:
    if curie.startswith(("http://", "https://")):
        return curie
    prefix, _, local = curie.partition(":")
    if prefix not in prefixes:
        raise AlignmentRefused("a mapping CURIE uses a prefix the curie map lacks")
    return prefixes[prefix] + local


def read_mapping(tsv: Path, curie_map: Path, expected_sha256: str) -> tuple[dict[str, str], dict[str, str], dict]:
    """The renaming curated IRI -> data IRI from a reviewed SSSOM table (subject = data term, object = curated term),
    the data side's kind by curated IRI (the table's ``kind`` column, "" where it has none), and its record. Rows with no predicate are terms left unmapped; any predicate other than skos:exactMatch is
    refused, as is a term mapped more than once on either side. Where the table has an ``iri`` column (the data term's
    full IRI) it must equal the expanded subject."""
    raw = tsv.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != expected_sha256:
        raise AlignmentRefused(f"mapping file sha256 {sha} is not the reviewed {expected_sha256}")
    prefixes = read_curie_map(curie_map)
    text = raw.decode("utf-8")
    rows = list(csv.DictReader(io.StringIO("".join(ln for ln in text.splitlines(keepends=True)
                                                    if not ln.startswith("#"))), delimiter="\t"))
    rename: dict[str, str] = {}
    kinds: dict[str, str] = {}
    seen_data: dict[str, str] = {}
    unmapped = 0
    for r in rows:
        pred = (r.get("predicate_id") or "").strip()
        if not pred:
            unmapped += 1
            continue
        if pred != EXACT:
            raise AlignmentRefused(f"a row maps with a predicate other than {EXACT}, the only one applied")
        data, cur = _expand(r["subject_id"].strip(), prefixes), _expand(r["object_id"].strip(), prefixes)
        if r.get("iri") and r["iri"].strip() != data:
            raise AlignmentRefused("a row's iri column differs from its expanded subject_id")
        if cur in rename and rename[cur] == data:
            raise AlignmentRefused("a mapped pair has more than one row")
        if cur in rename:
            raise AlignmentRefused("a curated term is matched to more than one data term")
        if data in seen_data and seen_data[data] != cur:
            raise AlignmentRefused("a data term is matched to more than one curated term")
        rename[cur], seen_data[data] = data, cur
        kinds[cur] = (r.get("kind") or "").strip()
    record = {"rule": RULE, "mapping_sha256": sha, "curie_map_sha256": hashlib.sha256(curie_map.read_bytes()).hexdigest(),
              "exact_matches": len(rename), "rows_without_mapping": unmapped}
    return rename, kinds, record


def _origin_terms(origin: dict | None) -> set[str]:
    """Every term the per-axiom origin record names, at every level of its keys."""
    out: set[str] = set()
    for kind, value in (origin or {}).items():
        if kind in ("by", "parts"):
            continue
        stack = [(value, kind in ("domain", "range"))]
        while stack:
            node, split = stack.pop()
            if not isinstance(node, dict):
                continue
            for k, v in node.items():
                # a domain or range record's inner keys are member lists joined by "|"
                out |= set(k.split("|")) if split and not isinstance(v, dict) else {k}
                stack.append((v, split))
    return out


def _terms(src: dict) -> set[str]:
    """Every IRI the source names as a term (not in labels, comments or provenance), the origin record included."""
    out = set(src.get("classes", {})) | set(src.get("properties", {})) | set(src.get("described") or {})
    out |= set(src.get("declared_classes") or [])
    out |= {v["iri"] for v in src.get("classes", {}).values()} | {v["iri"] for v in src.get("properties", {}).values()}
    out |= _origin_terms(src.get("origin"))
    for info in src.get("classes", {}).values():
        out |= set(info.get("parents", []))
    for info in src.get("properties", {}).values():
        out |= set(info.get("parents", [])) | set(info.get("datatype_range", []))
        for side in ("domain", "range"):
            for e in info.get(side, []):
                out |= set(e["members"])
        out |= {v["datatype"] for v in info.get("range_values", []) if v.get("datatype")}
    for a, b in src.get("disjoint", []):
        out |= {a, b}
    for e in src.get("existentials", []):
        out |= {e["cls"], e["prop"], e["filler"]}
    for p in src.get("patterns", []):
        out |= {x for x in (p.get("subject_class"), p.get("predicate"), p.get("object_class"),
                            p.get("object_datatype")) if x}
    return out


def _origin(origin: dict, r) -> dict:
    out = {k: v for k, v in origin.items() if k in ("by", "parts")}

    def flat(d: dict) -> dict:
        return {r(k): v for k, v in d.items()}

    def nested(d: dict) -> dict:
        return {r(k): {r(k2): v for k2, v in inner.items()} for k, inner in d.items()}

    for kind, value in origin.items():
        if kind in ("by", "parts"):
            continue
        if kind in ("classes", "properties", "range_values"):
            out[kind] = flat(value)
        elif kind in ("subclass", "datatype_range"):
            out[kind] = nested(value)
        elif kind == "disjoint":
            pairs: dict[str, dict] = {}
            for a, inner in value.items():
                for b, parts in inner.items():
                    x, y = sorted((r(a), r(b)))
                    pairs.setdefault(x, {})[y] = parts
            out[kind] = pairs
        elif kind in ("domain", "range"):
            out[kind] = {r(p): {"|".join(sorted(r(m) for m in key.split("|"))): parts for key, parts in inner.items()}
                         for p, inner in value.items()}
        elif kind == "existentials":
            out[kind] = {r(c): {r(p): {r(f): parts for f, parts in fills.items()} for p, fills in props.items()}
                         for c, props in value.items()}
        else:
            raise AlignmentRefused("the origin record has a kind this alignment does not know")
    return out


def _axioms(src: dict) -> dict[str, set[tuple]]:
    """The source's axioms by kind, each a tuple of its terms: the frozen fragment's, as the per-axiom origin record
    counts them (harness.sources.origin), plus super-property edges."""
    out: dict[str, set[tuple]] = {k: set() for k in AXIOM_KINDS}
    declared = src.get("declared_classes")  # where present, only these are class declarations
    out["classes"] = {(c,) for c in (declared if declared is not None else src.get("classes", {}))}
    for c, info in src.get("classes", {}).items():
        out["subclass"] |= {(c, p) for p in info.get("parents", [])}
    for p, info in src.get("properties", {}).items():
        out["properties"].add((p,))
        out["subproperty"] |= {(p, q) for q in info.get("parents", [])}
        for side in ("domain", "range"):
            out[side] |= {(p, *sorted(e["members"])) for e in info.get(side, [])}
        out["datatype_range"] |= {(p, d) for d in info.get("datatype_range", [])}
        if info.get("range_values"):
            out["range_values"].add((p, *sorted(v.get("datatype") or "" for v in info["range_values"])))
    out["disjoint"] |= {tuple(sorted(pair)) for pair in src.get("disjoint", [])}
    out["existentials"] |= {(e["cls"], e["prop"], e["filler"]) for e in src.get("existentials", [])}
    return out


def _kind(src: dict, iri: str) -> str | None:
    """The curated side's kind from the source's declarations: "class", "property", or None (neither, or both)."""
    declared = src.get("declared_classes")  # where present, only these are declared classes (rule 32 (a))
    is_class = iri in declared if declared is not None else iri in src.get("classes", {})
    is_prop = iri in src.get("properties", {})
    return None if is_class == is_prop else ("class" if is_class else "property")


def align(src: dict, rename: dict[str, str], kinds: dict[str, str], mapping_record: dict,
          name: str = "aligned-curated") -> dict:
    """The aligned source (JSON form) of a curated source's JSON form: see the module docstring. ``kinds``: the data
    side's kind of each mapped curated IRI (``read_mapping``)."""
    mismatched = [c for c in rename if _kind(src, c) is None or KINDS[_kind(src, c)] != kinds.get(c)]
    if mismatched:
        raise AlignmentRefused(f"{len(mismatched)} mapped pairs have different or unknown kinds (class to class and "
                               "property to predicate only)")
    terms = _terms(src)
    applied = {t: rename[t] for t in terms if t in rename}
    unchanged = terms - set(applied)
    # Every target must be new to the source: a target the source names already, mapped itself or not (an identity
    # pair included), is a collision.
    clash = sorted(set(applied.values()) & terms)
    if clash:
        raise AlignmentRefused(f"the mapping renames a curated term onto a term the source already names ({len(clash)})")

    def r(iri):
        return applied.get(iri, iri) if isinstance(iri, str) else iri

    out = copy.deepcopy(src)
    out["name"] = name
    out["classes"] = {r(k): {**v, "iri": r(v["iri"]), "parents": [r(p) for p in v.get("parents", [])]}
                      for k, v in src.get("classes", {}).items()}
    props = {}
    for k, v in src.get("properties", {}).items():
        props[r(k)] = {
            **v, "iri": r(v["iri"]), "parents": [r(p) for p in v.get("parents", [])],
            "domain": [{"members": [r(m) for m in e["members"]]} for e in v.get("domain", [])],
            "range": [{"members": [r(m) for m in e["members"]]} for e in v.get("range", [])],
            "datatype_range": [r(d) for d in v.get("datatype_range", [])],
            "range_values": [{k2: (r(x2) if k2 == "datatype" else x2) for k2, x2 in x.items()}
                             for x in v.get("range_values", [])],
        }
    out["properties"] = props
    out["disjoint"] = sorted(sorted((r(a), r(b))) for a, b in src.get("disjoint", []))
    out["existentials"] = [{**e, "cls": r(e["cls"]), "prop": r(e["prop"]), "filler": r(e["filler"])}
                           for e in src.get("existentials", [])]
    named = ("subject_class", "predicate", "object_class", "object_datatype")
    out["patterns"] = [{k: (r(x) if k in named else x) for k, x in p.items()} for p in src.get("patterns", [])]
    if src.get("declared_classes") is not None:
        out["declared_classes"] = sorted(r(c) for c in src["declared_classes"])
    if src.get("described") is not None:
        out["described"] = {r(k): v for k, v in src["described"].items()}
    if src.get("origin") is not None:
        out["origin"] = _origin(src["origin"], r)
    before = _axioms(src)
    touched = {k: sum(1 for ax in before[k] if any(x in applied for x in ax)) for k in AXIOM_KINDS}
    out["provenance"] = {**src.get("provenance", {}), "alignment": {
        **mapping_record, "renamed_terms": len(applied), "source_terms_not_mapped": len(unchanged),
        "axioms": sum(len(v) for v in before.values()),
        "axioms_touched": {**touched, "total": sum(touched.values())}}}
    return out
