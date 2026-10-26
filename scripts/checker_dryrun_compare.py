#!/usr/bin/env python3
"""Compare two checker dry runs (design log, ruling (c)): the replay matrix BEFORE and AFTER a
change of the compiler, the sources or the shared hierarchy, on the same records.

For each model and arm (primary variant): the six cells; the flags on correct (C) and wrong
(W) outputs by check; the false flags whose offending term is a data node of the AFTER
hierarchy (contract rule 32, amended in change-log entry 26); the disjointness flags of the
generated draws on correct outputs, labelled "axiom contradicted by the data" (ruling (e));
the outputs whose checked query or its status changed (rule 22); and, for the AFTER run,
the outputs with no query, the share of feedback lines capped at 10 listed items, the
evidence triples per output, and whether every union row's flags are exactly the union of
its sources' flags (rule 39).

    uv run python scripts/checker_dryrun_compare.py --before BEFORE --after AFTER \\
        --hierarchy AFTER/rhea-shared-hierarchy.json.gz --union union=curated,census-t5 \\
        --note "..." --out before-after.md

BEFORE and AFTER are run folders (their matrix.jsonl.gz is read) or matrix files.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from rdflib import URIRef, Variable

from harness.checkloop.matrix import read_jsonl_gz
from harness.checks.hierarchy import SharedHierarchy
from harness.checks.query import analyse

CELL = {"flagged": "F", "substantive_pass": "P", "vacuous_pass": "P", "unknown": "U"}
CAPPED = re.compile(r", and \d+ more")
CONTRADICTED = "axiom contradicted by the data"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASSOF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
NODE_KEYED = ("census_join.static", "no_common_subclass.static", "disjointness.static")


def matrix_path(path: Path) -> Path:
    return path / "matrix.jsonl.gz" if path.is_dir() else path


def primary(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["run_id"], r["arm"]): r for r in rows if r["variant"] == "primary"}


def cells(rows) -> Counter:
    c = Counter()
    for r in rows:
        if r["output"]:
            c[("C" if r["correct"] else "W") + "_" + CELL[r["status"]]] += 1
    return c


def class_constants(query: str, key: str) -> set[str]:
    """The IRIs the query puts in a class position of the node ``key`` (``?x a C``,
    ``?x rdfs:subClassOf C``, with + or * closures, and ``rdf:type/rdfs:subClassOf*``), read
    through the compiler's own query analysis, so PREFIX names resolve as full IRIs do."""
    qa = analyse(query)
    if qa.error:
        return set()
    return {str(e.o) for g in qa.groups for e in g.edges
            if isinstance(e.o, URIRef) and (e.closure or set(e.preds) & {RDF_TYPE, RDFS_SUBCLASSOF})
            and (f"?{e.s}" if isinstance(e.s, Variable) else str(e.s)) == key}


def offending_terms(check: dict, row: dict) -> set[str]:
    """The IRIs a fired check is about: the term (vocabulary), the constant (class, domain or
    range of a constant), the class of a single pattern (census pattern); for a check keyed
    on a node (census join, no common subclass, static disjointness), the constants in a class
    position of that node, which is where rule 32 puts a data node among the required classes."""
    cid, key = check["check_id"], check["key"]
    if cid == "vocabulary.static":
        return {key}
    if cid in ("class.static", "domain_range.static"):
        return {key.rsplit(" ", 2)[0]} if key.count(" ") >= 2 else set()
    if cid == "census_pattern.static":
        parts = key.split(" ")
        return {parts[-1]} if len(parts) == 3 else set()
    if cid in NODE_KEYED:
        return class_constants(row.get("checked_query") or "", key) if row.get("checked_query") else set()
    return set()


def table(head: list[str], body: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |"
                                                                         for r in body]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--before", type=Path, required=True, help="the BEFORE run folder or its matrix.jsonl.gz")
    p.add_argument("--after", type=Path, required=True, help="the AFTER run folder or its matrix.jsonl.gz")
    p.add_argument("--hierarchy", type=Path, required=True, help="the AFTER run's shared hierarchy (its data nodes)")
    p.add_argument("--union", action="append", default=["union=curated,census-t5"])
    p.add_argument("--title", default="Checker dry run: BEFORE and AFTER")
    p.add_argument("--note", action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    before = primary(read_jsonl_gz(matrix_path(args.before)))
    after = primary(read_jsonl_gz(matrix_path(args.after)))
    if {k for k in before} != {k for k in after}:
        raise SystemExit("the two matrices do not hold the same (record, arm) pairs")
    data_nodes = SharedHierarchy.load(args.hierarchy).data_nodes
    unions = {a: tuple(s.split(",")) for a, _, s in (u.partition("=") for u in dict.fromkeys(args.union))}
    models = sorted({r["model"] for r in after.values()})
    arms = sorted({r["arm"] for r in after.values()})
    out = [f"# {args.title}", ""] + [n for note in args.note for n in (note, "")]

    out += ["## Six cells per arm (primary variant)", ""]
    for model in models:
        body = []
        for arm in arms:
            b = cells(r for (_, a), r in before.items() if a == arm and r["model"] == model)
            a_ = cells(r for (_, a), r in after.items() if a == arm and r["model"] == model)
            body.append([arm] + [f"{b[k]} → {a_[k]}" for k in ("C_F", "C_P", "C_U", "W_F", "W_P", "W_U")])
        out += [f"### {model}", ""] + table(["Arm", "C_F", "C_P", "C_U", "W_F", "W_P", "W_U"], body) + [""]

    out += [("## Flags by check on correct (C) and wrong (W) outputs (primary; outputs with at least one flag of "
             "that check, BEFORE → AFTER)"), ""]
    for model in models:
        body = []
        for arm in arms:
            counts = {}
            for name, run in (("before", before), ("after", after)):
                c = Counter()
                for (_, a), r in run.items():
                    if a == arm and r["model"] == model and r["output"] and r["status"] == "flagged":
                        for cid in {f["check_id"] for f in r["fired_checks"]}:
                            c[(cid, "C" if r["correct"] else "W")] += 1
                counts[name] = c
            for cid in sorted({k[0] for c in counts.values() for k in c}):
                body.append([arm, cid] + [f"{counts['before'][(cid, cw)]} → {counts['after'][(cid, cw)]}"
                                          for cw in ("C", "W")])
        out += [f"### {model}", ""] + table(["Arm", "Check", "C", "W"], body) + [""]

    out += ["## False flags on a data node (rule 32 as amended, change-log entry 26)", "",
            (f"Correct outputs flagged by a check whose offending term is one of the AFTER hierarchy's "
             f"{len(data_nodes):,} data nodes, for both runs. For a census join, a no-common-subclass or a "
             f"static disjointness flag, the offending terms are the constants in a class position of the "
             f"flagged node; for vocabulary, the term (a data node there is used as a predicate, rule 25); "
             f"for class or domain and range, the constant (rule 5)."), ""]
    body = []
    for model in models:
        for arm in arms:
            vals = []
            for run in (before, after):
                n = Counter()
                for (_, a), r in run.items():
                    if a == arm and r["model"] == model and r["output"] and r["correct"] and r["status"] == "flagged":
                        hit = {f["check_id"] for f in r["fired_checks"] if offending_terms(f, r) & data_nodes}
                        for cid in hit:
                            n[cid] += 1
                        n["outputs"] += bool(hit)
                vals.append(n)
            if vals[0]["outputs"] or vals[1]["outputs"]:
                checks = sorted(k for v in vals for k in v if k != "outputs")
                detail = "; ".join(f"{c} {vals[0][c]} → {vals[1][c]}" for c in dict.fromkeys(checks))
                body.append([model, arm, f"{vals[0]['outputs']} → {vals[1]['outputs']}", detail])
    out += table(["Model", "Arm", "Correct outputs so flagged", "By check"], body) + [""]

    out += [f"## Disjointness flags of the generated draws on correct outputs: \"{CONTRADICTED}\" (ruling (e))", "",
            ("A correct output that a draw's disjointness flags shows nodes in classes the draw declares disjoint: "
             "the draw's axiom is contradicted by the data. Kept as flags (ruling (e))."), ""]
    body = []
    for model in models:
        for arm in [a for a in arms if a.startswith("generated")]:
            row = [model, arm]
            for run in (before, after):
                outputs = sum(1 for (_, a), r in run.items() if a == arm and r["model"] == model and r["output"]
                              and r["correct"] and any(f["check_id"].startswith("disjointness")
                                                       for f in r.get("fired_checks") or []))
                row.append(outputs)
            body.append(row[:2] + [f"{row[2]} → {row[3]}"])
    out += table(["Model", "Draw", f"Correct outputs flagged: {CONTRADICTED}"], body) + [""]

    out += ["## Checked query (rule 22, the scorer's own matching; review finding 3)", ""]
    body = []
    for model in models:
        arm = arms[0]
        keys = [k for k in after if k[1] == arm and after[k]["model"] == model and after[k]["output"]]
        moved = Counter((before[k]["checked_query_status"], after[k]["checked_query_status"]) for k in keys
                        if before[k]["checked_query_status"] != after[k]["checked_query_status"])
        query_changed = sum(1 for k in keys if before[k].get("checked_query") != after[k].get("checked_query"))
        body.append([model, len(keys), ", ".join(f"{a}→{b}: {n}" for (a, b), n in sorted(moved.items())) or "none",
                     query_changed])
    out += table(["Model", "Outputs", "Status changed", "Checked query changed"], body) + [""]

    out += ["## AFTER: no query, capped feedback lines, cost, and the union (rule 39)", ""]
    body = []
    for model in models:
        for arm in arms:
            rs = [r for (_, a), r in after.items() if a == arm and r["model"] == model and r["output"]]
            lines = [line for r in rs for line in r.get("fired") or []]
            capped = sum(1 for line in lines if CAPPED.search(line))
            share = f"{capped}/{len(lines)} ({capped / len(lines):.1%})" if lines else "0/0"
            triples = sum(r.get("budget_used") or 0 for r in rs)
            body.append([model, arm, sum(1 for r in rs if r.get("reason") == "no_query"), share,
                         f"{triples / len(rs):.1f}" if rs else "-"])
    out += table(["Model", "Arm", "No query (unknown, never retried)", "Feedback lines capped at 10 items",
                  "Evidence triples per output"], body) + [""]
    for arm, sources in unions.items():
        bad, n = 0, 0
        for (run_id, a), r in after.items():
            if a != arm or not r["output"]:
                continue
            n += 1
            singles = [after[(run_id, s)] for s in sources]
            fired = {tuple(sorted(f.items())) for f in r["fired_checks"]}
            expect = {tuple(sorted(f.items())) for s in singles for f in s["fired_checks"]}
            lines_ok = set(r["fired"]) == {line for s in singles for line in s["fired"]}
            cost_ok = (r.get("budget_used") or 0) == sum(s.get("budget_used") or 0 for s in singles)
            status_ok = (r["status"] == "flagged") == any(s["status"] == "flagged" for s in singles)
            bad += not (fired == expect and lines_ok and cost_ok and status_ok)
        out += [(f"Union `{arm}` = {' + '.join(sources)}: on {n} outputs, {bad} differ from the union of the "
                 f"single-source flags, feedback lines, flagged status and cost (rule 39). The replay matrix builds "
                 f"the union row from its sources' reports, so this is a consistency count of the matrix, not "
                 f"independent evidence; the online union arm is checked against its sources on recorded store "
                 f"responses in tests/test_checkloop_union_dryrun.py."), ""]
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
