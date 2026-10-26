"""Compare the fault bank's conformance labeller with the check compiler on a stratified sample.

Issue E1.3's note: the study lead audits a sample of mutations before the freeze; every
disagreement between the separate labeller and the compiler is examined, not silently resolved.
This script imports both (the labeller module itself imports nothing from the harness); the
labeller's labels are read from the bank's items, the compiler runs here on each sampled mutated
query, with the mutated query's own result rows as the answer.

The labeller judges the mutated term; the compiler judges the whole output, so a flag the gold
query already raises (a gold-level false flag, such as a gold predicate the source does not
declare) is not the mutation's. The compiler therefore also runs on each sampled item's gold query,
and a flag counts as introduced by the mutation when its (check, reason, key) is not among the
gold's flags.

Agreement, per source:
    primary   labeller "violates"  <->  the compiler raises a flag the gold does not (any
              ontology-derived check)
    static    the same, static checks only (the labeller reads the query and the constants'
              classes, as the static layer does; footprint checks read the bound nodes as well)
    whole     labeller "violates"  <->  compiler output status "flagged" (reported for reference)

    uv run python scripts/compare_labeller_compiler.py --bank <bank dir> \\
        --source curated=<curated.json.gz> --source census-t5=<census-t5.json> \\
        --endpoint http://localhost:7019/ --hierarchy-cache <path.json.gz> --out <comparison.json>

Exposure (design ruling in the design log): a comparison shows compiler outputs on the items it samples,
so they leave the confirmatory population. On a confirmatory graph (``CONFIRMATORY_GRAPHS``) only dev
items are sampled; a test item is inspected only under ``--expose-test``. Every exposed item (on a
confirmatory graph its sampled test items; on an exploratory graph every sampled item, as the Rhea
list was built) is appended to ``data/faults/<graph>/exposed-items.txt`` before any compiler runs,
and the bank build reads that list.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from harness.checks.compiler import CheckCompiler, Output
from harness.checks.hierarchy import GraphRoles, SharedHierarchy
from harness.checks.outcomes import Outcome, OutputReport
from harness.faults.manifest import exposed_path
from harness.faults.store import (
    PinnedStore,
    Runner,
    StoreMismatch,
    fetch_stats,
    store_identity,
)
from harness.sources.model import OntologySource
from harness.sparql import SparqlClient

# (fault class, stratum) -> sample size; 100 in all, spread over operators within each stratum
QUOTAS = {("type_wrong", "near"): 25, ("type_wrong", "far"): 25, ("vocabulary_absent", None): 20,
          ("stale_in_data", None): 15, ("meaning", None): 15}
RHEA_ROLES = ("https://sparql.rhea-db.org/rhea", ("http://sparql.rhea-db.org/chebi",))
# The confirmatory banks (design ruling in the design log): their test items are the confirmatory population,
# so a comparison samples their dev items only. Rhea is exploratory.
CONFIRMATORY_GRAPHS = frozenset({"uniprot-sp-2026_03"})


def load_json(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def stratified_sample(items: list[dict], quotas: dict, seed: int, exclude: frozenset[str] = frozenset()) -> list[dict]:
    """Answer-changing mutations (the common denominator), per (class, stratum) quota, taken
    round-robin over operators in a seeded order; ``exclude`` keeps an earlier sample out, so a
    re-measurement is on fresh items."""
    out: list[dict] = []
    for (cls, stratum), n in quotas.items():
        pool = [i for i in items if i["kind"] == "mutation" and i["bank"] == "answer_changing"
                and i["fault_class"] == cls and i.get("stratum") == stratum and i["id"] not in exclude]
        by_op: dict[str, list[dict]] = defaultdict(list)
        for i in pool:
            by_op[i["operator"]].append(i)
        for op, group in by_op.items():
            group.sort(key=lambda i: hashlib.sha256(f"{seed}|{i['id']}".encode()).hexdigest())
        taken: list[dict] = []
        while len(taken) < n and any(by_op.values()):
            for op in sorted(by_op):
                if by_op[op] and len(taken) < n:
                    taken.append(by_op[op].pop(0))
        out.extend(sorted(taken, key=lambda i: i["id"]))
    return out


def record_exposure(path: Path, ids: list[str]) -> list[str]:
    """Append ``ids`` not yet listed to the exposure list, atomically: the whole list is written to a
    temporary file beside it and renamed over it, so a crash leaves the old list or the new one."""
    existing = path.read_text().splitlines() if path.exists() else []
    added = sorted(set(ids) - set(existing))
    if not added:
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", delete=False) as fh:
        fh.write("".join(f"{line}\n" for line in existing + added))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(fh.name, path)
    return added


def draw_sample(items: list[dict], graph: str, quotas: dict, seed: int, exclude: frozenset[str],
                expose_test: bool, exposed_file: Path) -> tuple[list[dict], list[str]]:
    """The sample and the item ids it newly exposes, recorded before the caller inspects anything."""
    confirmatory = graph in CONFIRMATORY_GRAPHS
    pool = items if expose_test or not confirmatory else [i for i in items if i.get("split") == "dev"]
    sample = stratified_sample(pool, quotas, seed, exclude)
    exposed = [i["id"] for i in sample if not confirmatory or i.get("split") == "test"]
    return sample, record_exposure(exposed_file, exposed)


def expose_test_requested(argv: list[str]) -> bool:
    return bool(build_parser().parse_known_args(argv)[0].expose_test)


def flag_keys(report: OutputReport) -> set[tuple[str, str, str]]:
    return {(r.check_id, r.reason, r.key) for r in report.ontology() if r.outcome is Outcome.FLAG}


def verdict(report: OutputReport, gold: OutputReport | None = None) -> dict:
    checks = report.ontology()
    flags = [r for r in checks if r.outcome is Outcome.FLAG]
    before = flag_keys(gold) if gold is not None else set()
    new = [r for r in flags if (r.check_id, r.reason, r.key) not in before]
    return {
        "status": report.status(),
        "static_flag": any(r.form == "static" for r in new),
        "footprint_flag": any(r.form == "footprint" for r in new),
        "new_flag": bool(new),
        "flags": sorted({f"{r.check_id}:{r.reason}" for r in new}),
        "gold_flags": sorted({f"{c}:{reason}" for c, reason, _ in before}),
        "unknown": sorted({f"{r.check_id}:{r.reason}" for r in checks if r.outcome is Outcome.UNKNOWN}),
    }


def diagnose(item: dict, source: str, label: str, violations: list[dict], v: dict) -> str:
    """One line: what each side saw."""
    lab = ", ".join(sorted({f"{x['constraint']}/{x['basis']}" for x in violations})) or "-"
    comp = ", ".join(v["flags"]) or ("unknown: " + ", ".join(v["unknown"]) if v["unknown"] else "-")
    if v.get("gold_flags"):
        comp += "; gold already flags " + ", ".join(v["gold_flags"])
    slot = (item.get("slot") or {}).get("text", "agg")
    return (f"{item['id']} [{source}] {slot} -> {item.get('replacement')}: labeller {label} ({lab}); "
            f"compiler {v['status']}, new flags: {comp}")


def expected_unknown(v: dict) -> bool:
    """The compiler left the mutation's checks unknown at its fragment boundary (rule 19:
    subqueries, BIND, UNION ...) and raised no new flag."""
    return v["status"] == "unknown" and not v["new_flag"] and any(u.endswith(":unsupported_construct") for u in v["unknown"])


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for source in sorted({r["source"] for r in rows}):
        mine = [r for r in rows if r["source"] == source]
        errors = [r for r in mine if r["agree"] is None and not r.get("expected_unknown")]
        unknowns = [r for r in mine if r.get("expected_unknown")]
        mine = [r for r in mine if r["agree"] is not None]
        by_class: dict[str, Counter] = defaultdict(Counter)
        for r in mine:
            by_class[r["class"]]["n"] += 1
            by_class[r["class"]]["agree"] += r["agree"]
            by_class[r["class"]]["agree_static"] += r["agree_static"]
            by_class[r["class"]]["agree_whole"] += r["agree_whole"]
        out[source] = {
            "n": len(mine),
            "compiler_errors": len(errors),
            "expected_unknowns": len(unknowns),
            "agree": sum(r["agree"] for r in mine),
            "agree_static": sum(r["agree_static"] for r in mine),
            "agree_whole": sum(r["agree_whole"] for r in mine),
            "confusion": dict(sorted(Counter(f"{r['label']} / {'new flag' if r['verdict']['new_flag'] else 'no new flag'}"
                                             for r in mine).items())),
            "by_class": {c: dict(v) for c, v in sorted(by_class.items())},
        }
    return out


def _output(query: str, runner: Runner, role: str) -> Output:
    ex = runner.run(query)
    rows = list(ex.rows or []) if ex.status == "ok" else []
    return Output(query=query, answer=rows, result_rows=rows, role=role, store_accepted=ex.status == "ok")


def compare_item(item: dict, compilers: dict[str, CheckCompiler], runner: Runner, client, gold_query: str) -> list[dict]:
    output = _output(item["query"], runner, "agent")
    gold = _output(gold_query, runner, "gold")
    out = []
    for name, compiler in compilers.items():
        try:
            v = verdict(compiler.check(output, client), compiler.check(gold, client))
        except Exception as exc:  # noqa: BLE001 - a compiler crash is a finding, recorded, never dropped
            v = {"status": "compiler_error", "error": f"{type(exc).__name__}: {exc}"[:300], "static_flag": False,
                 "footprint_flag": False, "new_flag": False, "flags": [], "gold_flags": [], "unknown": []}
        label = item["conformance"][name]
        violations = item.get("conformance_violations", {}).get(name, [])
        lab_v = label == "violates"
        row = {"id": item["id"], "source": name, "class": item["fault_class"] + (f"@{item['stratum']}" if item.get("stratum") else ""),
               "operator": item["operator"], "label": label, "verdict": v,
               "agree": lab_v == v["new_flag"], "agree_static": lab_v == v["static_flag"],
               "agree_whole": lab_v == (v["status"] == "flagged")}
        if v["status"] == "compiler_error":
            row["agree"] = row["agree_static"] = row["agree_whole"] = None
            row["diagnosis"] = f"{item['id']} [{name}] compiler error: {v['error']}"
        elif not row["agree"] and expected_unknown(v):
            # design ruling (design log): the compiler's fragment boundary (rule 19) is an expected
            # unknown, reported apart from agreement
            row["agree"] = row["agree_static"] = row["agree_whole"] = None
            row["expected_unknown"] = True
            row["diagnosis"] = diagnose(item, name, label, violations, v)
        elif not row["agree"]:
            row["diagnosis"] = diagnose(item, name, label, violations, v)
        out.append(row)
    return out


def gold_queries(items: list[dict]) -> dict[str, str]:
    """Each question's untouched gold query (rewrites are controls too, but never the gold)."""
    return {i["question_id"]: i["query"] for i in items if i["kind"] == "control" and i.get("control") != "answer_preserving_rewrite"}


def bank_store(bank: Path, client, endpoint: str) -> str:
    """The store key the bank's cache may be reused under: the store answering now must be the one
    the bank recorded (review #2 of 3afbfa1)."""
    manifest = json.loads((bank / "manifest.json").read_text())
    recorded = (manifest.get("store") or {}).get("identity")
    if recorded is None:
        raise StoreMismatch(f"{bank}: the manifest records no store identity; rebuild the bank")
    now = store_identity(client, PinnedStore.load(manifest["graph"]), fetch_stats(endpoint))
    if now["key"] != recorded["key"]:
        raise StoreMismatch(f"{bank} was built against {recorded['key']!r}; {endpoint} is {now['key']!r}")
    return now["key"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", type=Path)
    ap.add_argument("--source", action="append", metavar="NAME=PATH")
    ap.add_argument("--endpoint")
    ap.add_argument("--hierarchy-cache", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, help="a fresh seed per comparison (the 18 October one was 20261018)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--exclude", type=Path, action="append", default=[],
                    help="an earlier comparison's JSON: its sampled items are left out")
    ap.add_argument("--expose-test", action="store_true",
                    help="on a confirmatory graph, also sample test items; each one is recorded as exposed first")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    missing = [f"--{n.replace('_', '-')}" for n in ("bank", "source", "endpoint", "hierarchy_cache", "out", "seed")
               if getattr(args, n) is None]
    if missing:
        ap.error("required: " + ", ".join(missing))

    client = SparqlClient(args.endpoint, timeout_s=args.timeout)
    if args.hierarchy_cache.exists():
        hierarchy = SharedHierarchy.load(args.hierarchy_cache)
    else:  # the compiler's own shared hierarchy, built by its own code (rule 35 roles, as the Rhea smoke test)
        roles = GraphRoles(RHEA_ROLES[0], RHEA_ROLES[1], (), exclude_schema_subjects=True)
        hierarchy = SharedHierarchy.build(client, roles)
        hierarchy.save(args.hierarchy_cache)
    compilers = {}
    for spec in args.source:
        name, _, path = spec.partition("=")
        compilers[name] = CheckCompiler(OntologySource.from_json(load_json(Path(path))), hierarchy)
    items = [json.loads(line) for line in (args.bank / "items.jsonl").read_text().splitlines()]
    exclude = frozenset(i for path in args.exclude for i in json.loads(path.read_text())["sample"])
    graph = json.loads((args.bank / "manifest.json").read_text())["graph"]
    sample, newly_exposed = draw_sample(items, graph, QUOTAS, args.seed, exclude, args.expose_test, exposed_path(graph))
    runner = Runner(SparqlClient(args.endpoint, timeout_s=args.timeout, row_cap=10000),
                    cache_path=args.bank / "store-cache.jsonl", store=bank_store(args.bank, client, args.endpoint))
    golds = gold_queries(items)
    rows: list[dict] = []
    for item in sample:
        rows.extend(compare_item(item, compilers, runner, client, golds[item["question_id"]]))
    result = {"bank_items_sha256": json.loads((args.bank / "manifest.json").read_text())["files"]["items.jsonl"],
              "sources": {s.partition("=")[0]: hashlib.sha256(Path(s.partition("=")[2]).read_bytes()).hexdigest()
                          for s in args.source},
              "seed": args.seed, "excluded": len(exclude), "expose_test": args.expose_test,
              "newly_exposed": newly_exposed, "quotas": {f"{c}@{s}" if s else c: n for (c, s), n in QUOTAS.items()},
              "sample": [i["id"] for i in sample], "summary": summarise(rows),
              "disagreements": [r["diagnosis"] for r in rows if r["agree"] is False],
              "expected_unknowns": [r["diagnosis"] for r in rows if r.get("expected_unknown")],
              "compiler_errors": [r["diagnosis"] for r in rows if r["agree"] is None and not r.get("expected_unknown")],
              "rows": rows}
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
