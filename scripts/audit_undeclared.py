"""Audit: terms of the curated ontology's namespace that the data uses but the ontology does not declare.

    uv run python scripts/audit_undeclared.py --endpoint http://localhost:7020/ --graph uniprot-sp-2026_03 \\
        --namespace http://purl.uniprot.org/core/ \\
        --questions ../e01/data/questions/uniprot-sp-2026_03/questions.jsonl \\
        --out data/sources/uniprot-sp-2026_03/undeclared-core.json

For every census graph (the store without the curated ontology's own graph), the predicates in
``--namespace`` and the classes in it used as rdf:type objects, with their triple counts; those the
curated ontology's graphs (``StoreLayout.curated_graphs``: core.owl on UniProt) do not declare (no
rdf:type to a class or property kind there) are listed, with the number of gold queries that use
each. Terms the ontology declares but marks owl:deprecated are listed apart. Nothing is patched
(check contract v2, rule 15).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.sources.vocab import LAYOUTS, PREFIXES, values  # noqa: E402
from harness.sparql import SparqlClient  # noqa: E402
from harness.terms import IRI  # noqa: E402

DECLARING_TYPES = (
    "http://www.w3.org/2002/07/owl#Class", "http://www.w3.org/2000/01/rdf-schema#Class",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property", "http://www.w3.org/2002/07/owl#ObjectProperty",
    "http://www.w3.org/2002/07/owl#DatatypeProperty", "http://www.w3.org/2002/07/owl#AnnotationProperty",
)
_PREFIX = re.compile(r"PREFIX\s+([A-Za-z][\w.-]*)?:\s*<([^>]*)>", re.I)
_IRI = re.compile(r"<([^<>\s\"]+)>")
_PNAME = re.compile(r"(?<![\w<\"'])([A-Za-z][\w.-]*)?:([A-Za-z_][\w.-]*)")


def query_iris(sparql: str) -> set[str]:
    """IRIs a SPARQL query mentions: full IRIs and prefixed names expanded by its own PREFIX lines."""
    prefixes = {p or "": ns for p, ns in _PREFIX.findall(sparql)}
    body = _PREFIX.sub(" ", sparql)
    body = re.sub(r'"(?:[^"\\]|\\.)*"', '""', body)  # drop string literals
    out = set(_IRI.findall(body))
    for p, local in _PNAME.findall(body):
        if (p or "") in prefixes:
            out.add(prefixes[p or ""] + local.rstrip("."))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--graph", required=True, choices=sorted(LAYOUTS))
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--questions", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    layout = LAYOUTS[args.graph]
    client = SparqlClient(args.endpoint, timeout_s=290)
    ns = args.namespace
    own = layout.curated_graphs

    declared: dict[str, str] = {}
    for s, t in client.query(PREFIXES + (
        f"SELECT ?s ?t WHERE {{ {values('g', own)} GRAPH ?g {{ ?s rdf:type ?t }} {values('t', DECLARING_TYPES)} "
        "FILTER isIRI(?s) }")).rows:
        declared.setdefault(s.value, t.value)
    deprecated = {r[0].value for r in client.query(PREFIXES + (
        f'SELECT ?s WHERE {{ {values("g", own)} GRAPH ?g {{ ?s owl:deprecated ?d }} FILTER(isIRI(?s) && STR(?d) = "true") }}')).rows}

    # One graph at a time: a single GROUP BY over every graph sorts hundreds of millions of rows.
    preds: dict[str, dict[str, int]] = {}
    types: dict[str, dict[str, int]] = {}
    for g in layout.census_graphs:
        for p, n in client.query(PREFIXES + (
            f'SELECT ?p (COUNT(*) AS ?n) WHERE {{ GRAPH <{g}> {{ ?s ?p ?o }} }} GROUP BY ?p')).rows:
            if isinstance(p, IRI) and p.value.startswith(ns):
                preds.setdefault(p.value, {})[g] = int(n.lexical)
        for t, n in client.query(PREFIXES + (
            f'SELECT ?t (COUNT(*) AS ?n) WHERE {{ GRAPH <{g}> {{ ?s rdf:type ?t }} }} GROUP BY ?t')).rows:
            if isinstance(t, IRI) and t.value.startswith(ns):
                types.setdefault(t.value, {})[g] = int(n.lexical)

    gold: dict[str, int] = {}
    qinfo = None
    if args.questions:
        raw = args.questions.read_bytes()
        qinfo = {"path": str(args.questions), "sha256": hashlib.sha256(raw).hexdigest(), "questions": 0}
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            qinfo["questions"] += 1
            for iri in query_iris(rec["gold_sparql"]):
                gold[iri] = gold.get(iri, 0) + 1

    def entry(term, per_graph):
        return {"term": term, "triples_by_graph": dict(sorted(per_graph.items())),
                "triples": sum(per_graph.values()), "gold_queries": gold.get(term, 0)}

    report = {
        "layout": layout.name,
        "namespace": ns,
        "declaring_graphs": list(own),
        "declaring_types": list(DECLARING_TYPES),
        "census_graphs": list(layout.census_graphs),
        "questions": qinfo,
        "declared_terms": len(declared),
        "undeclared_predicates": sorted((entry(p, d) for p, d in preds.items() if p not in declared),
                                        key=lambda e: -e["triples"]),
        "undeclared_classes": sorted((entry(t, d) for t, d in types.items() if t not in declared),
                                     key=lambda e: -e["triples"]),
        "deprecated_but_used": sorted((entry(t, d) for t, d in {**preds, **types}.items() if t in deprecated),
                                      key=lambda e: -e["triples"]),
        "used_predicates": len(preds),
        "used_classes": len(types),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
