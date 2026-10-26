"""Build one graph's fault bank (E1.3): mutations of every gold query, answer effects on the pinned
store, conformance labels per source, and the manifest with its counts.

One deterministic command; rerun it unchanged on the final inputs. It never freezes: freezing is
committing its output, whose hash and commit time are the freeze record.

    uv run python scripts/build_fault_bank.py --graph rhea-142 \\
        --questions data/questions/rhea-142/questions.jsonl \\
        --census data/sources/rhea-142/census-t5.json \\
        --shared-hierarchy data/sources/rhea-142/shared-hierarchy.json \\
        --source curated=data/sources/rhea-142/curated.json.gz --source census-t5=data/sources/rhea-142/census-t5.json \\
        --endpoint http://localhost:7019/ --out results/fault-bank/rhea-142 [--provisional]

The mutations depend on no ``--source``: they are drawn from the shared hierarchy, the census
co-occurrence graph and store facts only, so adding or removing a source adds or removes its
conformance column and nothing else. The store is identified first (``store_identity``) and
recorded in the manifest; the result cache in ``--out`` is reused only for the same store.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

from harness.faults.conformance import Labeller
from harness.faults.cooccurrence import CooccurrenceGraph
from harness.faults.hops import hop_profile
from harness.faults.manifest import (
    PROFILES,
    build_items,
    cap_wrong,
    check_extension,
    check_questions,
    exposed_path,
    mark_exposed,
    merge_question_sources,
    parse_exposed,
    sha256_file,
    summarise,
    tables_markdown,
    write_bank,
)
from harness.faults.operators import (
    DESCRIPTIVE,
    DESCRIPTIVE_STRATUM,
    FAULT_CLASS,
    OPERATORS,
    STRATA,
    TYPE_CONSISTENT_TRIES,
    Context,
)
from harness.faults.predictions import REGISTERED_PREDICTIONS
from harness.faults.rewrites import (
    ACTIVE_REWRITES,
    RETIRED_REWRITES,
    REWRITES,
    RewriteSettings,
)
from harness.faults.store import (
    PinnedStore,
    Runner,
    SparqlOracle,
    fetch_stats,
    store_identity,
)
from harness.sparql import SparqlClient

MAX_TIMEOUT_S = 290  # QLever 9bebc60 caps the time-out at 300 s (stores/OPERATIONS.md)


def load_json(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def mutation_terms(shared: dict, graph: CooccurrenceGraph) -> tuple[frozenset[str], frozenset[str]]:
    """The classes and the vocabulary the mutation engine knows, from source-neutral inputs only
    (review #1 of 3afbfa1): the shared hierarchy's classes and the census co-occurrence graph's
    nodes. No checker source enters, in any role."""
    named = set(shared.get("parents", {}))
    for ps in shared.get("parents", {}).values():
        named.update(ps)
    named |= graph.classes
    return frozenset(named), frozenset(named | graph.predicates | graph.datatypes)


def load_pinned(store: str) -> PinnedStore:
    return PinnedStore.load(store)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True, choices=sorted(PROFILES))
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--extension", type=Path,
                    help="the graph's bank-only extension (gold queries without question texts), pinned by its own sha256")
    ap.add_argument("--census", type=Path, required=True, help="the census record the co-occurrence graph is built from")
    ap.add_argument("--shared-hierarchy", type=Path, required=True)
    ap.add_argument("--source", action="append", default=[], metavar="NAME=PATH",
                    help="a source to label conformance against (labelling only: it never shapes a mutation)")
    ap.add_argument("--stale-map", type=Path, help="JSON {current IRI: [stale IRIs]} from deprecation records outside the store")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--distances", default="2,4", help="swap distance strata on the census co-occurrence graph (even)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--row-cap", type=int, default=10000)
    ap.add_argument("--provisional", action="store_true", help="mark the manifest provisional (inputs not final)")
    ap.add_argument("--freeze", action="store_true",
                    help="mark the manifest frozen (a final build of pinned inputs only); a directory holding a frozen "
                         "bank is never rewritten, and a rebuild that would change it is refused")
    ap.add_argument("--cache", type=Path, help="the store result cache (default: store-cache.jsonl in --out)")
    ap.add_argument("--unfrozen-questions", action="store_true",
                    help="allow a question file other than the graph's frozen set (provisional banks only); recorded")
    ap.add_argument("--rewrites", type=int, default=0, metavar="K",
                    help="answer-preserving rewrites to draw per question (a control stratum; 0: none)")
    ap.add_argument("--rewrite-attempts", type=int, default=24, help="rewrite executions per question, at most")
    ap.add_argument("--rewrite-operators", nargs="+", choices=list(REWRITES), default=list(ACTIVE_REWRITES),
                    help="operators to draw rewrites from (default: all but the retired ones, change-log entry 41)")
    ap.add_argument("--cap-wrong", type=int, metavar="N",
                    help="mark all but a seeded N answer-changing items per question subsampled_out (default: no cap)")
    ap.add_argument("--exposed", type=Path,
                    help="the items exposed to compiler outputs (default data/faults/<graph>/exposed-items.tsv, which "
                         "names each item's query, else exposed-items.txt, ids only)")
    ap.add_argument("--no-descriptive", action="store_true",
                    help="leave out the descriptive strata (the type-consistent constant swap); for checking that the "
                         "confirmatory bank does not depend on them")
    ap.add_argument("--agent-forms", type=Path,
                    help="agents' wrong constants by form, per graph (results/constants-check-20261019/"
                         "agent-constant-forms.json), shown beside the bank's replacement forms in tables.md")
    ap.add_argument("--note", action="append", default=[], help="free-text provenance (e.g. the commit an input came from)")
    ap.add_argument("--hop-profile", default="2,4,6", help="distances to profile offline ('' to skip)")
    args = ap.parse_args(argv)
    if args.timeout > MAX_TIMEOUT_S:
        ap.error(f"--timeout must be at most {MAX_TIMEOUT_S} s")
    if args.unfrozen_questions and not args.provisional:
        ap.error("--unfrozen-questions is for provisional banks only")
    if args.freeze and args.provisional:
        ap.error("a provisional bank cannot be frozen")
    distances = tuple(int(x) for x in args.distances.split(","))
    if any(d < 2 or d % 2 for d in distances):
        ap.error("--distances must be even and at least 2 (the graph is bipartite)")

    profile = PROFILES[args.graph]
    questions_sha = sha256_file(args.questions)
    questions_frozen = check_questions(profile, questions_sha, unfrozen=args.unfrozen_questions)
    if profile.extension_sha256 and not args.extension and not args.provisional:
        ap.error(f"{profile.name} pins a bank-only extension ({profile.extension_sha256[:8]}...): pass --extension")
    extension_sha = sha256_file(args.extension) if args.extension else None
    extension_pinned = (check_extension(profile, extension_sha, unfrozen=args.unfrozen_questions)
                        if args.extension else None)
    exposed_file = args.exposed or exposed_path(profile.name)
    exposed_list = parse_exposed(exposed_file.read_text()) if exposed_file.exists() else []
    frozen_rows = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    extension_rows = ([json.loads(line) for line in args.extension.read_text().splitlines() if line.strip()]
                      if args.extension else [])
    questions = merge_question_sources(frozen_rows, extension_rows)
    census = load_json(args.census)
    if census.get("graph") not in (None, profile.name):
        raise SystemExit(f"census graph {census.get('graph')!r} is not {profile.name!r}")
    shared = load_json(args.shared_hierarchy)
    loaded: dict[Path, dict] = {}

    def specs(values: list[str]) -> dict[str, tuple[Path, dict]]:
        out = {}
        for spec in values:
            name, _, path = spec.partition("=")
            p = Path(path)
            if p not in loaded:
                loaded[p] = load_json(p)
            out[name] = (p, loaded[p])
        return out

    sources = specs(args.source)
    stale_map = json.loads(args.stale_map.read_text()) if args.stale_map else {}

    graph = CooccurrenceGraph.from_census(census)
    named, vocabulary = mutation_terms(shared, graph)
    args.out.mkdir(parents=True, exist_ok=True)
    client = SparqlClient(args.endpoint, timeout_s=args.timeout, row_cap=args.row_cap, max_response_bytes=64 * 2**20)
    # live, never from the cache: which store answers, and that it answers
    identity = store_identity(client, load_pinned(profile.name), fetch_stats(args.endpoint))
    health = client.query("ASK { ?s ?p ?o }")
    if not health.boolean:
        raise SystemExit(f"store at {args.endpoint} is not answering")
    runner = Runner(client, cache_path=args.cache or args.out / "store-cache.jsonl", store=identity["key"])
    oracle = SparqlOracle(runner, profile.data_graph, named, profile.stale_sources)
    ctx = Context(graph, oracle, vocabulary, named, args.seed, distances, stale_map, profile.membership)
    descriptive = () if args.no_descriptive else DESCRIPTIVE
    labellers = {name: Labeller(src, shared) for name, (_, src) in sorted(sources.items())}

    rewrites = (RewriteSettings(args.rewrites, args.rewrite_attempts, profile.membership, shared.get("parents", {}),
                                oracle, runner, args.seed, tuple(args.rewrite_operators)) if args.rewrites > 0 else None)
    rewrite_log: list = []
    items, skips = build_items(questions, ctx, runner, labellers, OPERATORS, rewrites, rewrite_log, descriptive)
    exposure = mark_exposed(items, exposed_list)
    capped = cap_wrong(items, args.cap_wrong, args.seed)["summary"] if args.cap_wrong is not None else None
    header = {
        "graph": profile.name,
        "provisional": bool(args.provisional),
        "notes": args.note,
        "inputs": {
            "questions": {"path": str(args.questions), "sha256": questions_sha, "n": len(frozen_rows),
                          "frozen": questions_frozen},
            "extension": {"path": str(args.extension), "sha256": extension_sha, "n": len(extension_rows),
                          "pinned": extension_pinned} if args.extension else None,
            "census": {"path": str(args.census), "sha256": sha256_file(args.census)},
            "shared_hierarchy": {"path": str(args.shared_hierarchy), "sha256": sha256_file(args.shared_hierarchy)},
            "sources": {n: {"path": str(p), "sha256": sha256_file(p)} for n, (p, _) in sorted(sources.items())},
            "stale_map": {"path": str(args.stale_map), "sha256": sha256_file(args.stale_map)} if args.stale_map else None,
            "agent_forms": {"path": str(args.agent_forms), "sha256": sha256_file(args.agent_forms)} if args.agent_forms else None,
            "exposed": {"path": str(exposed_file), "sha256": sha256_file(exposed_file)} if exposed_file.exists() else None,
        },
        "store": {"endpoint": args.endpoint, "timeout_s": args.timeout, "row_cap": args.row_cap, "identity": identity},
        "parameters": {
            "seed": args.seed, "swap_distances": {str(d): STRATA.get(d, f"d{d}") for d in distances},
            "operators": list(OPERATORS), "fault_classes": FAULT_CLASS,
            "descriptive_operators": {op: {"fault_class": FAULT_CLASS[op], "stratum": DESCRIPTIVE_STRATUM[op],
                                           "tries": TYPE_CONSISTENT_TRIES, "bank": "descriptive",
                                           "rule": "a gold constant replaced, at every occurrence, by another member of "
                                                   "its own class (rdf:type, or the graph's membership predicate in the "
                                                   "data graph), the most specific own class with another member; up to "
                                                   f"{TYPE_CONSISTENT_TRIES} distinct seeded draws, the gold excluded, the "
                                                   "first answer-changing one kept; outside C2 and outside C4's W "
                                                   "(design ruling (d), design log, change-log entry 63)"}
                                      for op in descriptive},
            "fault_class_rule": "by operator (design ruling, design log; pre-registration 2b, change-log entry 21)",
            "realisable_descriptor": "ASK over the mutated pattern joined with same-scope patterns sharing a node",
            "mutation_terms": "shared hierarchy classes + census co-occurrence graph nodes (vocabulary for "
                              "invented-identifier absence) and store facts; no checker source in any role",
            "class_test": "in a class position: a shared-hierarchy or graph class is a class; otherwise a store data "
                          "node by rule 32 (b) and (c) is a constant, any other IRI (absent ones included, rule 37) a "
                          "class; RDF, RDFS and OWL meta terms are never classes (rule 35)",
            "scopes": "UNION branches and negated parts are their own scopes; a subquery's unprojected variables "
                      "are its own; a dropped GROUP BY key that an enclosing query names is not applicable",
            "rewriter": "token level (harness/faults/query.py); rdflib prepareQuery as the parse gate",
            "cooccurrence_graph": {"predicates": len(graph.predicates), "classes": len(graph.classes),
                                   "datatypes": len(graph.datatypes),
                                   "edges": sum(len(v) for v in graph.positions.values())},
            "stale_sources": list(profile.stale_sources), "stale_note": profile.stale_note,
            "rewrites": {"k": args.rewrites, "attempt_cap": args.rewrite_attempts,
                         "operators": list(args.rewrite_operators),
                         "retired": {op: why for op, why in RETIRED_REWRITES.items()
                                     if op not in args.rewrite_operators}} if rewrites else None,
            "membership": profile.membership,
            "cap_wrong": args.cap_wrong,
        },
        "summary": {**summarise(items, skips, sorted(sources), rewrite_log if rewrites else None,
                                args.rewrites, args.rewrite_attempts),
                    "exposed": exposure, **({"cap_wrong": capped} if capped else {})},
        "registered_predictions": REGISTERED_PREDICTIONS,
    }
    extra = {"rewrites-rejected.jsonl": [r for e in rewrite_log for r in e["rejected"]]} if rewrites else None
    written = write_bank(args.out, header, items, skips, extra, freeze=args.freeze)
    manifest = written.manifest
    if written.verified:  # a frozen bank, reproduced: nothing is rewritten
        print(json.dumps({"out": str(args.out), "verified_frozen": True, "items": manifest["files"]["items.jsonl"]}, indent=1))
        return 0
    agents = (json.loads(args.agent_forms.read_text())["graphs"].get(profile.name) if args.agent_forms else None)
    (args.out / "tables.md").write_text(tables_markdown(items, header["summary"], sorted(sources), skips, agents))
    if args.hop_profile:
        profile_d = tuple(int(x) for x in args.hop_profile.split(","))
        (args.out / "hop-profile.json").write_text(json.dumps(hop_profile(questions, ctx, profile_d), indent=1, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "items": manifest["files"]["items.jsonl"],
                      "rejections": manifest["summary"]["rejections_total"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
