#!/usr/bin/env python3
"""Offline replay matrix of the checker loop (design log, AC6, AC7, AC9).

Every source's checks (and the union arm, and the detection-switch variants)
on every record's initial output, with no model call. Writes, under OUT:

  matrix.jsonl.gz     one row per (record, arm, variant): status, fired checks,
                      feedback lines, traceability, checked-query status, strata
  store-log.jsonl.gz  per record, the store responses the checks read, so the
                      matrix replays with no store (--offline)
  accounting.json     six-cell accounting per model, cohort, arm and variant,
                      with strata; the cells are asserted to sum to N_out
  summary.md          the accounting as tables

    uv run python scripts/checker_replay.py --records results/pilot-lookup-20261018/sonnet/records.tar.gz \\
        --sources-dir data/sources/rhea-142 --hierarchy rhea-shared-hierarchy.json.gz \\
        --endpoint http://localhost:7019/ --out OUT
    uv run python scripts/checker_replay.py ... --offline OUT/store-log.jsonl.gz --out OUT2   # no store

Before checking, every record must load and write back byte for byte, and
replay offline through the agent loop (the gate of design log, AC8; a checker
cell's record replays its checker decision too).

With --obqc the matrix adds the two OBQC arms over the curated source (harness/baselines/obqc.py,
design log): "obqc", as published (the primary baseline), and "obqc-where", its WHERE-clause rules
1-6 (our variant, reported beside it). Both are static only, so they read no store.

Each --records path is one OUT. A dev grid's OUT directory (its manifest.json
pins its cells) is read as the analysis counts attempts (harness/checkloop/
grid.py): its initial cells' records, each under its cell as the cohort (not
its condition), and none of its branch cells' records, whose initial outputs
are their parents' (each is checked first: its parent must be in the OUT).
Every record's online checker decision (a grid's branch records hold them) is
still recomputed before the matrix, so leaving a branch record out of the
matrix never leaves its decision unverified.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from harness.agent.record import RunRecord
from harness.agent.replay import replay_attempt
from harness.baselines.obqc import obqc_compilers
from harness.checkloop.arms import hierarchy_sha256
from harness.checkloop.grid import GridError
from harness.checkloop.matrix import (
    MATRIX_VERSION,
    VARIANTS,
    Arms,
    accounting,
    load_attempts,
    load_raw_records,
    read_jsonl_gz,
    run_matrix,
    write_jsonl_gz,
)
from harness.checkloop.recompute import ArmUnavailable
from harness.checkloop.recompute import recompute_decision as _recompute_decision
from harness.checkloop.store_log import CHECKER_ROW_CAP, CHECKER_TIMEOUT_S
from harness.checks.compiler import CheckCompiler
from harness.checks.hierarchy import SharedHierarchy
from harness.sources.model import OntologySource
from harness.sparql import SparqlClient

DEFAULT_SOURCES = ("curated", "curated+doc", "census-t5", "generated-1", "generated-2", "generated-3")
DEFAULT_UNION = "union=curated,census-t5"


def source_path(directory: Path, name: str, files: dict[str, Path] | None = None) -> Path:
    if files and name in files:
        return files[name]
    for suffix in (".json.gz", ".json"):
        path = directory / f"{name}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"no source {name!r} in {directory}")


def load_source(directory: Path, name: str, files: dict[str, Path] | None = None) -> OntologySource:
    path = source_path(directory, name, files)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return OntologySource.from_json(json.load(f))


GRAPH_COUNTS = "SELECT ?g (COUNT(*) AS ?n) WHERE { GRAPH ?g { ?s ?p ?o } } GROUP BY ?g ORDER BY ?g"


def graph_counts(client) -> dict[str, int]:
    """The store's triples per named graph, recorded so that two runs can be compared."""
    r = client.query(GRAPH_COUNTS)
    return {g.value: int(n.lexical) for g, n in r.rows}


def gate(raws: list[str]) -> int:
    """Byte-for-byte round trip and offline replay of every record."""
    for raw in raws:
        rec = json.loads(raw)
        if RunRecord.from_json(rec).dumps() != raw:
            raise SystemExit(f"record {rec['run_id']} does not round-trip byte for byte")
        replay_attempt(rec)
    return len(raws)


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def summary_md(acc: dict, meta: dict) -> str:
    lines = ["# Checker replay matrix", "", (f"Records: {meta['records']}. Sources: {', '.join(meta['sources'])}. "
                                             f"Unions: {meta['unions']}. Variants: {', '.join(VARIANTS)}."), ""]
    lines += [f"Store: {meta['store']}" + (f"; triples per graph {meta['graph_counts']}" if meta.get("graph_counts")
                                           else "") + ".", ""]
    if meta.get("note"):
        lines += [meta["note"], ""]
    for model, cohorts in acc.items():
        for cohort, arms in cohorts.items():
            lines += [f"## {model}, cohort {cohort}", "",
                      ("| Arm | Variant | N_att | N_out | C_F | C_P (subst/vac) | C_U | W_F | W_P (subst/vac) | W_U | "
                       "flag precision | false-flag rate | recall | no query | evidence triples per output |"),
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for arm, variants in arms.items():
                for variant, a in variants.items():
                    c, p, r = a["cells"], a["passes"], a["rates"]
                    lines.append(
                        f"| {arm} | {variant} | {a['N_att']} | {a['N_out']} | {c['C_F']} | {c['C_P']} "
                        f"({p['C_P_substantive']}/{p['C_P_vacuous']}) | {c['C_U']} | {c['W_F']} | {c['W_P']} "
                        f"({p['W_P_substantive']}/{p['W_P_vacuous']}) | {c['W_U']} | {fmt(r['flag_precision'])} | "
                        f"{fmt(r['false_flag_rate'])} | {fmt(r['detection_recall'])} | {a.get('no_query', 0)} | "
                        f"{fmt(a.get('budget_used', 0) / a['N_out']) if a['N_out'] else '-'} |")
            lines.append("")
            lines += ["Primary variant, by checked-query status:", "",
                      "| Arm | Stratum | N_out | C_F | C_P | C_U | W_F | W_P | W_U |", "|---|---|---|---|---|---|---|---|---|"]
            for arm, variants in arms.items():
                for k, s in variants["primary"]["strata"]["checked_query_status"].items():
                    c = s["cells"]
                    lines.append(f"| {arm} | {k} | {s['N_out']} | {c['C_F']} | {c['C_P']} | {c['C_U']} | {c['W_F']} | "
                                 f"{c['W_P']} | {c['W_U']} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def recompute_decision(rec: dict, by_source: dict[str, CheckCompiler], curated: Callable[[], OntologySource]) -> bool:
    """Recompute a checker cell's decision from its record with no store and no model (AC8;
    harness/checkloop/recompute.py, which the replay check runs on every batch too): a source arm's
    from its recorded store responses, the OBQC arm's from the query alone, retry on empty's from the
    tool-call log, the judge's from its recorded reply. False for a record with no checked initial
    submission; SystemExit for an arm the sources cannot build."""
    try:
        return _recompute_decision(rec, by_source, curated)
    except ArmUnavailable as exc:
        raise SystemExit(str(exc)) from None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--records", type=Path, nargs="+", required=True)
    p.add_argument("--sources-dir", type=Path, required=True)
    p.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    p.add_argument("--source-file", action="append", default=[], metavar="NAME=PATH",
                   help="read source NAME from PATH instead of the sources directory")
    p.add_argument("--graph-counts", action="store_true", help="record the store's triples per graph (one query)")
    p.add_argument("--note", default="", help="a line for the summary and the metadata (e.g. which commits)")
    p.add_argument("--union", action="append", default=None, help="NAME=SRC1,SRC2 (default: union=curated,census-t5)")
    p.add_argument("--hierarchy", type=Path, required=True, help="the store's saved shared hierarchy (rule 29)")
    p.add_argument("--endpoint", default=None, help="the store the footprint checks read")
    p.add_argument("--timeout-s", type=float, default=CHECKER_TIMEOUT_S)
    p.add_argument("--offline", type=Path, default=None, help="a store-log.jsonl.gz to serve instead of the store")
    p.add_argument("--no-gate", action="store_true", help="skip the round-trip and replay gate")
    p.add_argument("--split", default=None, help="keep only records of this question split (e.g. dev)")
    p.add_argument("--obqc", action="store_true",
                   help="add the OBQC arms obqc and obqc-where over the curated source (design log)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    raws = list(load_raw_records(args.records))
    records = [json.loads(raw) for raw in raws]
    if args.split:
        keep = [i for i, r in enumerate(records) if r.get("split") == args.split]
        print(f"split {args.split}: {len(keep)} of {len(records)} records", file=sys.stderr)
        raws, records = [raws[i] for i in keep], [records[i] for i in keep]
    if not args.no_gate:
        print(f"gate: {gate(raws)} records round-trip byte for byte and replay", file=sys.stderr)
    try:  # each path one OUT: a grid's records under their cells, its branch records (no attempts) left out
        attempts, cohorts = load_attempts(args.records)
    except GridError as exc:
        raise SystemExit(f"refused: {exc}") from None
    kept = {r["run_id"] for r in records}
    attempts = [r for r in attempts if r["run_id"] in kept]
    if len(attempts) != len(records):
        print(f"matrix: {len(attempts)} attempt records ({len(records) - len(attempts)} grid branch records, whose "
              f"outputs are their parents', left out)", file=sys.stderr)
    hierarchy = SharedHierarchy.load(args.hierarchy)
    names = [n for n in args.sources.split(",") if n]
    files = {k: Path(v) for k, _, v in (x.partition("=") for x in args.source_file)}
    compilers = {}
    for name in names:
        t = time.time()
        compilers[name] = CheckCompiler(load_source(args.sources_dir, name, files), hierarchy)
        print(f"compiled {name} in {time.time() - t:.1f}s", file=sys.stderr)
    unions = {}
    for spec in args.union or [DEFAULT_UNION]:
        arm, _, srcs = spec.partition("=")
        unions[arm] = tuple(srcs.split(","))
    by_source = {c.source.name: c for c in compilers.values()}
    if args.obqc:  # OBQC reads the curated ontology (design ruling 3 of change-log entry 59)
        curated = by_source["curated"].source if "curated" in by_source else load_source(args.sources_dir, "curated",
                                                                                        files)
        compilers.update(obqc_compilers(curated))
    arms = Arms(compilers, unions)
    def curated():
        return by_source["curated"].source if "curated" in by_source else load_source(args.sources_dir, "curated",
                                                                                      files)

    decided = sum(recompute_decision(rec, by_source, curated) for rec in records)
    if decided:
        print(f"gate: {decided} checker decisions recomputed from their store logs", file=sys.stderr)

    logs_in = None
    live = None
    if args.offline:
        logs_in = {x["run_id"]: x["log"] for x in read_jsonl_gz(args.offline)}
    else:
        if not args.endpoint:
            raise SystemExit("--endpoint is required unless --offline is given")
        live = SparqlClient(args.endpoint, timeout_s=args.timeout_s, row_cap=CHECKER_ROW_CAP)
    counts = graph_counts(live) if args.graph_counts and live is not None else None

    t0 = time.time()

    def progress(i, rec):
        if i % 25 == 0:
            print(f"{i}/{len(records)} records, {time.time() - t0:.0f}s", file=sys.stderr)

    rows, logs = run_matrix(attempts, arms, live_client=live, store_logs=logs_in, progress=progress, cohorts=cohorts)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(args.out / "matrix.jsonl.gz", [{"version": MATRIX_VERSION, **r} for r in rows])
    write_jsonl_gz(args.out / "store-log.jsonl.gz", [{"run_id": k, "log": v} for k, v in sorted(logs.items())])
    acc = accounting(rows)
    meta = {"records": len(records), "attempts": len(attempts), "split": args.split, "sources": names, "unions": unions,
            "source_digests": {n: c.source.digest() for n, c in compilers.items()},
            "hierarchy_sha256": hierarchy_sha256(hierarchy), "hierarchy_file": args.hierarchy.name,
            "hierarchy_roles": hierarchy.to_json()["roles"], "store": "offline" if args.offline else args.endpoint,
            "graph_counts": counts, "note": args.note,
            "source_files": {n: {"path": str(source_path(args.sources_dir, n, files)),
                                 "sha256": hashlib.sha256(source_path(args.sources_dir, n, files).read_bytes()).hexdigest()}
                             for n in names}}
    (args.out / "accounting.json").write_text(json.dumps({"meta": meta, "accounting": acc}, indent=1,
                                                         sort_keys=True) + "\n")
    (args.out / "summary.md").write_text(summary_md(acc, meta))
    print(f"wrote {len(rows)} matrix rows for {len(attempts)} attempt records to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
