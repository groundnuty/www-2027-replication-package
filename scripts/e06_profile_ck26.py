#!/usr/bin/env python3
"""E6.0: profile the Text2SPARQL 2026 corporate vocabulary (prod-vocab.ttl) and how the store's data uses it.

Offline:

    uv run python scripts/e06_profile_ck26.py --owl prod-vocab.ttl --out reports/e06

writes vocab_profile.json and vocab_profile.md (classes, properties, domain and range as named
classes or unions, disjointness, restrictions). With the store (stores/text2sparql-ck26):

    uv run python scripts/e06_profile_ck26.py --owl prod-vocab.ttl --out DIR --endpoint URL

also writes data_usage.json and data_usage.md: per predicate the data uses, triples per graph,
whether the vocabulary declares it, and its declared domain and range; and a class-usage table
(rdf:type objects in the instance graph, declared or not). The vocabulary graph is left out of the
data denominators. It reuses harness/owl_profile.py (the E0.2 profiler) with the corporate graphs.

prod-vocab.ttl is under /shared/projects/proj/eval-graphs/text2sparql/2026/repo/docs/participation/data/
on SiteA, not on the lab VM: copy it, and reach the served store with stores/tunnel.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.owl_profile import (  # noqa: E402
    curie,
    data_usage,
    load_graph,
    predicate_usage,
    profile_markdown,
    profile_ontology,
    usage_markdown,
)
from harness.sparql import SparqlClient  # noqa: E402

VOCAB_GRAPH = "http://ld.company.org/prod-vocab/"
INST_GRAPH = "http://ld.company.org/prod-instances/"
PV_NS = "http://ld.company.org/prod-vocab/"
NAME = "prod-vocab"

CLASS_QUERY = f"SELECT ?c (COUNT(DISTINCT ?s) AS ?n) WHERE {{ GRAPH <{INST_GRAPH}> {{ ?s a ?c }} }} GROUP BY ?c"
GRAPH_COUNT_QUERY = "SELECT ?g (COUNT(*) AS ?n) WHERE { GRAPH ?g { ?s ?p ?o } } GROUP BY ?g"


def class_usage(client: SparqlClient) -> dict:
    rows = [(c.value, int(n.lexical)) for c, n in client.query(CLASS_QUERY).rows]
    rows.sort(key=lambda r: (-r[1], r[0]))
    return {"graph": INST_GRAPH, "classes": [{"class": c, "instances": n} for c, n in rows]}


def class_usage_markdown(cu: dict, declared_classes: set[str]) -> str:
    L = [
        "",
        "## Classes the data uses",
        "",
        f"Distinct subjects per `rdf:type` object in `{cu['graph']}`.",
        "",
        "| Class | Instances | Declared in the vocabulary |",
        "|---|---|---|",
    ]
    for r in cu["classes"]:
        L.append(f"| {curie(r['class'])} | {r['instances']:,} | {'yes' if r['class'] in declared_classes else 'no'} |")
    return "\n".join(L) + "\n"


def declared_class_iris(g) -> set[str]:
    from rdflib import RDF, RDFS, URIRef
    from rdflib.namespace import OWL

    return {str(s) for t in (OWL.Class, RDFS.Class) for s in g.subjects(RDF.type, t) if isinstance(s, URIRef)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owl", required=True, help="path to prod-vocab.ttl")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--endpoint", help="SPARQL endpoint of the store, for the data-usage report")
    ap.add_argument("--timeout", type=float, default=120, help="per-query time-out in seconds (default 120; the server refuses more than its 300 s cap)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = load_graph(args.owl)
    profile = profile_ontology(g, args.owl)
    (out / "vocab_profile.json").write_text(json.dumps(profile, indent=1) + "\n")
    (out / "vocab_profile.md").write_text(profile_markdown(profile, None, ontology_name=NAME))
    print(f"wrote {out}/vocab_profile.json and vocab_profile.md")

    if args.endpoint:
        client = SparqlClient(args.endpoint, timeout_s=args.timeout)
        graph_counts = {r[0].value: int(r[1].lexical) for r in client.query(GRAPH_COUNT_QUERY).rows}
        usage = predicate_usage(client, [VOCAB_GRAPH, INST_GRAPH])
        report = data_usage(profile, usage, ontology_graph=VOCAB_GRAPH, namespace=PV_NS, namespace_scope="pv_namespace")
        cu = class_usage(client)
        declared_classes = declared_class_iris(g)
        report["store_graph_counts"] = graph_counts
        report["class_usage"] = cu
        (out / "data_usage.json").write_text(json.dumps(report, indent=1) + "\n")
        md = usage_markdown(report, ontology_name=NAME) + class_usage_markdown(cu, declared_classes)
        (out / "data_usage.md").write_text(md)
        print(f"wrote {out}/data_usage.json and data_usage.md (method {report['method']}, {len(report['errors'])} errors)")
        print("per-graph triples from the store:", json.dumps(graph_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
