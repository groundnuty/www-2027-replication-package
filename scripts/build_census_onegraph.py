"""Build the census of a store read through one graph by the one-graph plan, privately, with the ruled cross-checks.

    python -I scripts/build_census_onegraph.py --name <graph name> --endpoint <url> --user-file <f> \\
        --password-file <f> --graph-file <f> --gate-file <f> --result-cap-file <f> --pinned-count <n> \\
        --vocab-state <private json> --out <private directory>

Every query names the data graph (default-graph-uri, and named-graph-uri for the census code's own queries, which name
it inside GRAPH), runs with a 60 s client and 120 s server time-out, one at a time under the gate file
(``harness.query_gate``), is cached in <out>/cache.jsonl (a rerun resumes without resending), and is refused unsent if
it has the engine-suspect shape (``harness.engine_suspect``, every class of the vocabulary table as a suspect class).
An answer that reaches a configured row limit (--result-cap-file, private) is marked cut, cached or sent.

Steps:
 1. the store's triple count, against --pinned-count;
 2. a positive control: the smallest predicate's count by the plan and by the census code's own query must equal the
    vocabulary table's, or nothing else is sent;
 3. the census by ``harness.sources.census_onegraph`` at --thresholds and --offline-thresholds (and at 1, kept in
    memory for the cross-checks);
 4. cross-checks: with the vocabulary table's independent counts (per predicate, per class, per predicate and class),
    and with the census code's own queries (the listing, the direct and membership counts, and per predicate in
    increasing size, each query family until its first query that does not answer);
 5. the instance sample (``harness.sources.sample_firstm``), unless --no-sample;
 6. the store's triple count again.

Writes census-t<T>.json for --thresholds and census-t<T>.json.gz for --offline-thresholds (".incomplete" before the
extension when a position is unknown; a position whose literal datatypes alone are unknown is recorded in the census
itself, position_datatype), sample.json, sample-record.json and record.json, all mode 600 in a mode-700 directory.
Prints and logs counts, hashes and timings only; a failure's traceback goes to <out>/error.log.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from harness.query_gate import QueryGate  # noqa: E402
from harness.sources import census as census_code  # noqa: E402
from harness.sources import hierarchy_adapter as ha  # noqa: E402
from harness.sources.census_onegraph import POSITIONS, Queries, build_censuses_onegraph  # noqa: E402
from harness.sources.closure import Hierarchy  # noqa: E402
from harness.sources.membership import CensusInput, fetch_census_input  # noqa: E402
from harness.sources.query_cache import CachingClient, SuspectGuard  # noqa: E402
from harness.sources.sample_firstm import LADDER, sample_instances_firstm  # noqa: E402
from harness.sources.vocab import RDF_TYPE, RDFS_SUBCLASSOF, StoreLayout  # noqa: E402
from harness.sparql import SparqlError, VirtuosoClient, read_result_cap  # noqa: E402
from harness.terms import IRI  # noqa: E402

NOT_ANSWERED = frozenset({"timeout", "too_large", "transport", "http", "suspect"})


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(path, 0o600)


def write_json(path: Path, obj) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    write_private(path, text)
    return hashlib.sha256(text.encode()).hexdigest()


def write_json_gz(path: Path, obj) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write(text.encode("utf-8"))
    os.chmod(path, 0o600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_digest(src) -> str:
    """sha256 of the record without its provenance (what the census says, not how it was read)."""
    out = src.to_json()
    out.pop("provenance")
    return hashlib.sha256(json.dumps(out, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def code_commit() -> str:
    marker = ROOT / "COMMIT"
    if marker.exists():
        return marker.read_text().strip()
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


class Run:
    def __init__(self, out: Path):
        self.out = out
        self.t0 = time.monotonic()

    def say(self, msg: str) -> None:
        line = f"{now()} +{time.monotonic() - self.t0:.0f}s {msg}"
        print(line, flush=True)
        fd = os.open(self.out / "run.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as fh:
            fh.write(line + "\n")


def store_count(client) -> dict:
    t = time.monotonic()
    try:
        rows = client.query(Queries.store_count).rows
        return {"count": int(rows[0][0].lexical), "wall_s": round(time.monotonic() - t, 2), "at": now()}
    except SparqlError as exc:
        return {"count": None, "error": exc.kind, "at": now()}


def answer(client, query):
    """Rows, or None when the query did not answer (any other error is raised)."""
    try:
        result = client.query(query)
    except SparqlError as exc:
        if exc.kind in NOT_ANSWERED:
            return None
        raise
    return None if result.truncated else result.rows


def crosscheck_vocab(state: dict, t1, record: dict) -> dict:
    """The plan's counts against the vocabulary table's (counted independently of the plan)."""
    out = {"predicates": [0, 0], "classes": [0, 0], "predicate_classes": [0, 0]}  # [equal, different]
    totals = record["position_totals"]
    for p, n in state.get("predicates", {}).items():
        if p in (RDF_TYPE, RDFS_SUBCLASSOF) or p not in totals:
            continue
        out["predicates"][totals[p]["subject"] != n] += 1
    for c, n in state.get("classes", {}).items():
        if c in t1.type_support:
            out["classes"][t1.type_support[c] != n] += 1
    listed = set(t1.classes)
    unknown = {(u["predicate"], u["position"]) for u in record["unknown"]}
    for p, sides in state.get("predclasses", {}).items():
        if p not in t1.position_support:
            continue
        for position in POSITIONS:
            theirs = sides.get(position)
            if not isinstance(theirs, dict) or (p, position) in unknown:
                continue
            mine = t1.position_support[p][position]
            for c in listed:
                out["predicate_classes"][mine.get(c, 0) != theirs.get(c, 0)] += 1
    return {k: {"equal": v[0], "different": v[1]} for k, v in out.items()}


def crosscheck_code(client, layout: StoreLayout, t1, record: dict, run: Run) -> dict:
    """The plan's counts against the census code's own queries, where those answer."""
    res: dict = {"global": {}, "families": {}}

    def tally(family: str, outcome: str) -> None:
        f = res["families"].setdefault(family, {"equal": 0, "different": 0, "not_answered": 0, "skipped": 0})
        f[outcome] += 1

    listed = sorted(t1.classes)
    try:
        cin_cur = fetch_census_input(client, layout)
        res["global"]["census_input"] = "equal" if cin_cur.reading() == t1.provenance["membership_reading"] else "different"
    except SparqlError as exc:
        if exc.kind not in NOT_ANSWERED:
            raise
        res["global"]["census_input"] = "not_answered"
    try:
        listing = ha.fetch_listing(client, layout)
        mine = t1.provenance["listing"]
        same = (listing.listed == set(listed) and listing.primary_targets == mine["primary_targets"]
                and listing.secondary_targets == mine["secondary_targets"] and len(listing.data_nodes) == mine["data_nodes"])
        res["global"]["listing"] = "equal" if same else "different"
    except SparqlError as exc:
        if exc.kind not in NOT_ANSWERED:
            raise
        res["global"]["listing"] = "not_answered"
    cin = CensusInput(layout)
    q = census_code._Q(cin)
    shared = Hierarchy({c: [] for c in listed})
    rows = answer(client, q.direct(RDF_TYPE))
    if rows is None:
        res["global"]["type_support"] = "not_answered"
    else:
        direct = {r[0].value: int(r[1].lexical) for r in rows if r[0] is not None and isinstance(r[0], IRI)}
        metas = {x.object_class: x.support for x in t1.patterns if x.predicate == RDF_TYPE}
        same = ({c: direct.get(c, 0) for c in listed} == t1.type_support
                and all(direct.get(c) == n for c, n in metas.items()))
        res["global"]["type_support"] = "equal" if same else "different"
    rows = answer(client, q.direct(RDFS_SUBCLASSOF))
    if rows is None:
        res["global"]["literal_other"] = "not_answered"
    else:
        other = {r[0].value: int(r[1].lexical) for r in rows if r[0] is not None and isinstance(r[0], IRI)}
        mine = {x.object_class: x.support for x in t1.patterns if x.predicate == RDFS_SUBCLASSOF}
        res["global"]["literal_other"] = "equal" if {c: n for c, n in other.items() if c in set(listed)} == mine else "different"
    if listed:
        rows = answer(client, q.class_signatures(listed))
        if rows is None:
            res["global"]["class_support"] = "not_answered"
        else:
            sigs = [(census_code.signature(r[0].lexical), int(r[1].lexical)) for r in rows if r[0] is not None]
            rolled = census_code._roll_up(sigs, shared)
            res["global"]["class_support"] = ("equal" if {c: rolled.get(c, 0) for c in listed} == t1.class_support
                                              else "different")
    totals = record["position_totals"]
    unknown = {(u["predicate"], u["position"]) for u in record["unknown"]}
    dt_unknown = {d["predicate"] for d in record.get("datatype_unknown", [])}
    stopped: set[str] = set()
    order = sorted(totals, key=lambda p: (totals[p]["subject"], p))
    for i, p in enumerate(order, 1):
        if i == 1 or i % 10 == 0:
            run.say(f"  cross-check with the census code: predicate {i}/{len(order)}, stopped families {len(stopped)}")
        qp = census_code._Q(cin, p)
        got: dict = {}
        for family, query in (("position_total_subject", qp.position_total("subject")),
                              ("object_datatypes", qp.object_datatypes()),
                              ("position_total_object", qp.position_total("object"))):
            if family in stopped:
                tally(family, "skipped")
                continue
            rows = answer(client, query)
            if rows is None:
                stopped.add(family)
                tally(family, "not_answered")
                continue
            got[family] = rows
        if "position_total_subject" in got:
            n = sum(int(r[1].lexical) for r in got["position_total_subject"] if r[0] is not None)
            tally("position_total_subject", "equal" if n == totals[p]["subject"] else "different")
        if "position_total_object" in got and (p, "object") not in unknown:
            n = sum(int(r[1].lexical) for r in got["position_total_object"] if r[0] is not None)
            tally("position_total_object", "equal" if n == totals[p].get("object") else "different")
        if "object_datatypes" in got and p in dt_unknown:
            tally("object_datatypes", "skipped")  # the plan's datatypes are unknown there: nothing to compare
        elif "object_datatypes" in got and (p, "object") not in unknown:
            dts = {r[1].value: int(r[2].lexical) for r in got["object_datatypes"] if isinstance(r[1], IRI)}
            mine = {k: v for k, v in t1.position_support[p]["object"].items() if k not in set(listed)}
            tally("object_datatypes", "equal" if dts == mine else "different")
        for position in POSITIONS:
            family = f"position_signatures_{position}"
            if family in stopped or not listed:
                tally(family, "skipped")
                continue
            rows = answer(client, qp.position_signatures(position, listed))
            if rows is None:
                stopped.add(family)
                tally(family, "not_answered")
                continue
            if (p, position) in unknown:
                tally(family, "skipped")
                continue
            sigs = [(census_code.signature(r[1].lexical), int(r[2].lexical)) for r in rows if r[0] is not None]
            rolled = census_code._roll_up(sigs, shared)
            total = totals[p][position]
            mine_classes = {c: n for c, n in t1.position_support[p][position].items() if c in set(listed)}
            same = rolled == mine_classes and total - sum(n for _s, n in sigs) == t1.untyped_fillers[p][position]
            tally(family, "equal" if same else "different")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--user-file", required=True)
    ap.add_argument("--password-file", required=True)
    ap.add_argument("--graph-file", required=True)
    ap.add_argument("--gate-file", required=True)
    ap.add_argument("--result-cap-file", required=True, help="the store's own row limit (one integer; private)")
    ap.add_argument("--pinned-count", type=int, required=True)
    ap.add_argument("--vocab-state", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--thresholds", type=int, nargs="+", default=[census_code.DEFAULT_THRESHOLD])
    ap.add_argument("--offline-thresholds", type=int, nargs="*", default=[2, 20])
    ap.add_argument("--client-timeout-s", type=float, default=60.0)
    ap.add_argument("--ladder", type=int, nargs="+", default=list(LADDER))
    ap.add_argument("--no-sample", action="store_true")
    ap.add_argument("--no-code-crosscheck", action="store_true")
    ap.add_argument("--retry-timeouts", action="store_true")
    args = ap.parse_args()
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=True)
    os.chmod(args.out, 0o700)
    try:
        return build(args)
    except BaseException as exc:  # noqa: BLE001 - a traceback can carry the store's content: private file only
        with open(args.out / "error.log", "a") as fh:
            traceback.print_exc(file=fh)
        os.chmod(args.out / "error.log", 0o600)
        print(json.dumps({"error": type(exc).__name__, "at": now()}), flush=True)
        return 1


def build(args) -> int:
    run = Run(args.out)
    graph = Path(args.graph_file).expanduser().read_text().split()
    if len(graph) != 1:
        raise SystemExit("the graph file must hold exactly one IRI")
    graph = graph[0]
    user = Path(args.user_file).expanduser().read_text().strip()
    password = Path(args.password_file).expanduser().read_text().strip()
    cap = read_result_cap(args.result_cap_file)
    raw = VirtuosoClient(args.endpoint, graph, timeout_s=args.client_timeout_s,
                         http=httpx.Client(auth=httpx.DigestAuth(user, password)),
                         gate=QueryGate(args.gate_file), named_graphs=(graph,), result_cap=cap)
    del user, password
    state = json.loads(args.vocab_state.read_text())
    guard = SuspectGuard(raw, state.get("classes", {}).keys())
    # the row-limit check changes no answer and is applied to cached answers too, so it is not part of the cache key
    key = {k: v for k, v in raw.settings().items() if k != "result_cap_set"}
    client = CachingClient(guard, args.out / "cache.jsonl", key_settings=key, retry_timeouts=args.retry_timeouts,
                           result_cap=cap)
    layout = StoreLayout(name=args.name, data_graph=graph, graphs=(graph,), membership_predicate=RDF_TYPE)
    record: dict = {"started": now(), "code_commit": code_commit(), "endpoint": args.endpoint,
                    "graph_sha256": hashlib.sha256(graph.encode()).hexdigest(), "client": raw.settings(),
                    "pinned_count": args.pinned_count, "vocab_state_sha256": hashlib.sha256(args.vocab_state.read_bytes()).hexdigest()}
    run.say(f"start; code {record['code_commit']}")

    record["store_before"] = store_count(guard)
    record["store_before"]["equals_pinned"] = record["store_before"]["count"] == args.pinned_count
    run.say(f"store count before: equals pinned {record['store_before']['equals_pinned']}")
    if not record["store_before"]["equals_pinned"]:
        raise SystemExit("the store's count is not the pinned one; nothing else is sent")

    # Positive control before the plan: the smallest predicate, three ways.
    vocab_preds = {p: n for p, n in state["predicates"].items() if p not in (RDF_TYPE, RDFS_SUBCLASSOF)}
    p0 = min(vocab_preds, key=lambda p: (vocab_preds[p], p))
    plan_rows = answer(client, Queries.predicates)
    plan_n = None if plan_rows is None else next((int(r[1].lexical) for r in plan_rows
                                                   if isinstance(r[0], IRI) and r[0].value == p0), None)
    cur_rows = answer(client, census_code._Q(CensusInput(layout), p0).position_total("subject"))
    cur_n = None if cur_rows is None else sum(int(r[1].lexical) for r in cur_rows if r[0] is not None)
    record["positive_control"] = {"plan_equals_vocab": plan_n == vocab_preds[p0],
                                  "census_code_equals_vocab": cur_n == vocab_preds[p0],
                                  "predicate_sum_equals_pinned": plan_rows is not None and sum(
                                      int(r[1].lexical) for r in plan_rows if r[0] is not None) == args.pinned_count}
    run.say(f"positive control: {record['positive_control']}")
    if not all(record["positive_control"].values()):
        raise SystemExit("the positive control failed; the census is not built")

    thresholds = sorted(set(args.thresholds) | set(args.offline_thresholds) | {1})
    t = time.monotonic()
    built = build_censuses_onegraph(client, layout, thresholds, log=run.say)
    record["plan_wall_s"] = round(time.monotonic() - t, 1)
    t1 = built[1]
    plan = t1.provenance["plan"]
    record["plan"] = {"complete": plan["complete"], "unknown": len(plan["unknown"]),
                      "datatype_unknown": len(plan["datatype_unknown"]),
                      "invariants_checked": plan["invariants"]["checked"],
                      "invariants_failed": len(plan["invariants"]["failed"]),
                      "invariants_unchecked": len(plan["invariants"]["unchecked"]),
                      "store_properties_checked": len(plan["store_properties"]),
                      "store_properties_held": sum(bool(v) for v in plan["store_properties"].values()),
                      "queries": sum(plan["queries_by_family"].values())}
    record["plan_queries_by_family"] = plan["queries_by_family"]  # private record only: family names can hint at
    # what the store holds, so the printed summary has the total
    run.say(f"census built: complete {plan['complete']}, unknown {len(plan['unknown'])}, datatype unknown "
            f"{len(plan['datatype_unknown'])}, queries {sum(plan['queries_by_family'].values())}")

    record["crosscheck_vocab"] = crosscheck_vocab(state, t1, plan)
    run.say(f"cross-check with the vocabulary table: {record['crosscheck_vocab']}")
    if not args.no_code_crosscheck:
        t = time.monotonic()
        record["crosscheck_code"] = crosscheck_code(client, layout, t1, plan, run)
        record["crosscheck_code_wall_s"] = round(time.monotonic() - t, 1)
        run.say(f"cross-check with the census code: {record['crosscheck_code']}")

    outputs = {}
    suffix = "" if plan["complete"] else ".incomplete"
    for th in args.thresholds:
        src = built[th]
        outputs[f"census-t{th}"] = {"file_sha256": write_json(args.out / f"census-t{th}{suffix}.json", src.to_json()),
                                    "digest": src.digest(), "content_digest": content_digest(src),
                                    "classes": len(src.classes), "predicates": len(src.properties)}
    for th in args.offline_thresholds:
        src = built[th]
        outputs[f"census-t{th}"] = {"file_sha256": write_json_gz(args.out / f"census-t{th}{suffix}.json.gz", src.to_json()),
                                    "digest": src.digest(), "content_digest": content_digest(src)}
    record["outputs"] = outputs
    write_json(args.out / "record.json", record)  # the census is recorded even if a later step fails

    if not args.no_sample:
        main_t = census_code.DEFAULT_THRESHOLD
        t = time.monotonic()
        sample, srec = sample_instances_firstm(client, layout, built[main_t], ladder=tuple(args.ladder),
                                               recheck_client=guard, log=run.say)
        srec["wall_s"] = round(time.monotonic() - t, 1)
        payload = {"spec": sample.spec, "instances": sample.instances}
        record["sample"] = {"file_sha256": write_json(args.out / "sample.json", payload),
                            "record_sha256": write_json(args.out / "sample-record.json", srec),
                            "n_strata": sample.spec["n_strata"], "n_instances": sample.spec["n_instances"],
                            "candidates": srec["candidates"], "rules": srec["rules"], "pool_limit": srec["pool_limit"],
                            "pool_resolution": srec["pool_resolution"],
                            "unstable_pools": srec["unstable_pools"], "type_only_pools": srec["type_only_pools"],
                            "cut_answers": srec["cut_answers"],
                            "wall_s": srec["wall_s"]}
        run.say(f"sample: {record['sample']}")

    record["store_after"] = store_count(guard)
    record["store_after"]["equals_pinned"] = record["store_after"]["count"] == args.pinned_count
    record["client_stats"] = dict(client.stats, suspect_refused=guard.refused)
    record["timings"] = {"slowest_s": max((x["wall_s"] or 0 for x in raw.timings), default=0),
                         "possibly_partial": sum(1 for x in raw.timings if x.get("possibly_partial")),
                         "outcomes": {o: sum(1 for x in raw.timings if x["outcome"] == o)
                                      for o in sorted({str(x["outcome"]) for x in raw.timings})}}
    record["wall_s"] = round(time.monotonic() - run.t0, 1)
    record["finished"] = now()
    write_json(args.out / "record.json", record)
    summary = {k: record[k] for k in ("code_commit", "positive_control", "plan", "crosscheck_vocab", "outputs",
                                      "client_stats", "timings", "wall_s") if k in record}
    summary["store_before"] = {k: record["store_before"].get(k) for k in ("count", "equals_pinned")}
    summary["store_after"] = {k: record["store_after"].get(k) for k in ("count", "equals_pinned")}
    for k in ("crosscheck_code", "sample"):
        if k in record:
            summary[k] = record[k]
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
