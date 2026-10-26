#!/usr/bin/env python3
"""E0.2: profile UniProt's core.owl, and (with --endpoint) how the store's data uses it.

Offline:

    uv run python scripts/e02_profile_core.py --owl core.owl --out reports/e02

writes core_profile.json and core_profile.md. With the store:

    uv run python scripts/e02_profile_core.py --owl core.owl --out DIR --endpoint URL

also runs one GROUP BY over all named graphs (falling back to one query per
loaded graph if that fails) and writes data_usage.json and data_usage.md: per
predicate, triples per graph, whether core.owl declares it, and its declared
domain and range (named or anonymous), plus the share of triples covered.

If any requested graph fails or returns no triples, or a graph's
per-predicate counts do not sum to its independently counted triples, the
data-usage report is written marked PARTIAL (saying why) and the script exits
with code 2: its shares are not the store's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.owl_profile import (  # noqa: E402
    data_usage,
    load_graph,
    predicate_usage,
    profile_markdown,
    profile_ontology,
    raw_xml_counts,
    usage_markdown,
)
from harness.sib import LOADED_GRAPHS  # noqa: E402
from harness.sparql import SparqlClient  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owl", required=True, help="path to core.owl (RDF/XML)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--endpoint", help="SPARQL endpoint of the store, for the data-usage report")
    ap.add_argument("--timeout", type=float, default=600, help="per-query time-out in seconds (default 600)")
    ap.add_argument("--graphs", nargs="*", default=list(LOADED_GRAPHS), help="graphs for the per-graph fallback")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = load_graph(args.owl)
    profile = profile_ontology(g, args.owl)
    raw = raw_xml_counts(args.owl) if Path(args.owl).suffix in (".owl", ".rdf", ".xml") else None
    profile["raw_xml"] = raw
    (out / "core_profile.json").write_text(json.dumps(profile, indent=1) + "\n")
    (out / "core_profile.md").write_text(profile_markdown(profile, raw))
    print(f"wrote {out}/core_profile.json and core_profile.md")

    if args.endpoint:
        client = SparqlClient(args.endpoint, timeout_s=args.timeout)
        usage = predicate_usage(client, args.graphs)
        report = data_usage(profile, usage)
        (out / "data_usage.json").write_text(json.dumps(report, indent=1) + "\n")
        (out / "data_usage.md").write_text(usage_markdown(report))
        print(f"wrote {out}/data_usage.json and data_usage.md (method {report['method']}, {len(report['errors'])} errors)")
        if not report["complete"]:
            print(f"PARTIAL: missing graphs {report['missing_graphs']}, count mismatches {report['total_mismatches']}, "
                  f"totals error {report['totals_error']}; shares are not the store's", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
