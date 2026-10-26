#!/usr/bin/env python3
"""The store probe of the store time-out rule (change-log entry 149, item 2; harness/agent/store_fault.py): a copy's
fixed set of 20 dev gold queries that completed at its identity check, sent one at a time; it FAILS when more than 2
time out, or when a request still errors after 3 re-sends (section 4.3's re-runs; a time-out is what the probe
measures, so it is not re-sent).

    uv run python scripts/store_probe.py golds --store-json S --questions Q --out FILE [--split dev] [--timeout-s 60]
    uv run python scripts/store_probe.py build --gold-rerun FILE --questions Q --copy ID --store NAME --out SET \\
        [--private]
    uv run python scripts/store_probe.py run --store-json S --set SET --sha256 H --questions Q --out DIR \\
        [--copy ID] [--timeout-s 60]

``golds`` is the identity check's gold run for a copy read with the plain client (UniProt, Rhea, opaque Rhea,
corporate): each ``--split`` gold query of the questions file once, one at a time, written as {"summary", "results":
[{"id", "split", "outcome", "wall_s"}]} ("answered", "timeout:client" or "error:<kind>"), the shape of an a15 copy's
qualification gold re-run (its gold-rerun.json, which stays in the private tree). A gated (a15) store is refused here:
its identity check is its qualification. The run is refused unless the description's endpoint is on the port its copy
id names (an id ending in '-<port>'), and unless the store's ?cmd=stats answers at the start; the summary records the
copy, the endpoint and that answer's name-index and num-triples-normal (the store the run read, as it says).

``build`` picks the probe set from a gold run: the completed ``dev`` queries in the order of
sha256('20261021|<copy id>|<id>') (the copy's own seed; the copy id must be an approved one, store_fault.COPY_IDS or
its naming rule), the first 20 (store_fault.pick_probe_queries), with each gold query's sha256 (never its text) and the
sha256 of the gold run and of the questions file; it prints the set's sha256, which the frozen config binds and the
copy's qualification record holds before its first test attempt. With ``--private`` (an a15 copy) the set is written
mode 600 and refused inside a git work tree: it names a15 questions, so it stays in the private tree and only its
sha256 is published.

``run`` refuses a set whose sha256 is not ``--sha256`` and a query whose text no longer has the set's sha256, then
sends each query through the harness's own client of the copy's store description: the plain client at
``--timeout-s`` (the runs' --sparql-timeout), or, for a gated store, the store's own clients on its shared gate at the
store's own time-outs (harness/gated_store.py: one query at a time across processes, a timed-out query waited out
before the next, read-only credentials and the data graph from the private files). It writes DIR/rows.jsonl and
DIR/summary.json with counts and timings only (a request's index, outcome, tries, wall time and error kinds; never a
query, an id or an answer) and prints the summary. Exit 0 pass, 1 fail, 2 a usage or input error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.agent import store_fault as sf  # noqa: E402
from harness.sparql import SparqlError  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_questions(path: Path) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            out[str(q["id"])] = q
    return out


def in_git_tree(path: Path) -> bool:
    return any((p / ".git").exists() for p in [path.resolve().parent, *path.resolve().parent.parents])


STATS_FIELDS = ("name-index", "num-triples-normal")  # what a gold run records of the store it read (QLever ?cmd=stats)


def fetch_stats(endpoint: str) -> dict:
    """The store's ?cmd=stats (one light request; OSError when it cannot be reached or answers other than 200)."""
    import httpx

    try:
        resp = httpx.get(endpoint, params={"cmd": "stats"}, timeout=20.0)
    except httpx.HTTPError as exc:
        raise OSError(str(exc)) from exc
    if resp.status_code != 200:
        raise OSError(f"HTTP {resp.status_code}")
    return resp.json()


def plain_factory(endpoint: str, timeout_s: float) -> Callable[[], Any]:
    from harness.sparql import SparqlClient

    return lambda: SparqlClient(endpoint, timeout_s=timeout_s, row_cap=None)


def store_factory(store: dict, timeout_s: float) -> tuple[Callable[[], Any], bool]:
    """A factory of one client per request for the store description, and whether the store is gated."""
    from harness import gated_store as gsm

    gs = gsm.gated_store(store)
    if gs is None:
        return plain_factory(store["endpoint"], timeout_s), False
    gsm.check_private(gs, ROOT)
    make, _ = gsm.client_factory(gs, endpoint=gs.endpoint or store["endpoint"], timeout_s=gsm.client_timeout_s(gs),
                                 row_cap=None)
    return make, True


def send(make_client: Callable[[], Any], query: str, resends: int = sf.STORE_PROBE_RESENDS,
         clock: Callable[[], float] = time.monotonic) -> sf.ProbeRow | dict:
    """One probe request: a time-out ends it ("timeout"); any other failure is re-sent up to ``resends`` times, then
    is "error". Returns the outcome, tries, wall time of the last try and the error kinds met (no text)."""
    kinds: list[str] = []
    tries = 0
    wall = None
    while tries <= resends:
        tries += 1
        client = make_client()
        start = clock()
        try:
            client.query(query)
            return {"outcome": "ok", "tries": tries, "wall_s": round(clock() - start, 3), "error_kinds": kinds}
        except SparqlError as exc:
            wall = round(clock() - start, 3)
            if exc.kind == "timeout":
                return {"outcome": "timeout", "tries": tries, "wall_s": wall, "error_kinds": kinds}
            kinds.append(exc.kind)
        except Exception as exc:  # noqa: BLE001 - a failure of any kind is a failed try, re-sent
            wall = round(clock() - start, 3)
            kinds.append(type(exc).__name__)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    return {"outcome": "error", "tries": tries, "wall_s": wall, "error_kinds": kinds}


def run_probe(texts: list[str], make_client: Callable[[], Any], resends: int = sf.STORE_PROBE_RESENDS
              ) -> tuple[list[dict], dict]:
    rows = []
    for i, text in enumerate(texts, 1):
        rows.append({"i": i, **send(make_client, text, resends)})
    return rows, sf.judge_store_probe(rows)


def cmd_golds(args: argparse.Namespace) -> int:
    try:
        sf.probe_seed(args.copy)  # an approved copy id (or one the naming rule admits)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    store = json.loads(Path(args.store_json).read_text(encoding="utf-8"))
    if store.get("private_files") is not None:
        print("refused: a gated (a15) store's identity check is its qualification's gold re-run", file=sys.stderr)
        return 2
    if Path(args.out).exists():
        print(f"refused: {args.out} exists", file=sys.stderr)
        return 2
    endpoint = store.get("endpoint") or ""
    want = sf.copy_port(args.copy)
    if want is not None and urlsplit(endpoint).port != want:
        print(f"refused: copy {args.copy} is the store on port {want}, but the description's endpoint names port "
              f"{urlsplit(endpoint).port}", file=sys.stderr)
        return 2
    try:
        sf.check_copy_endpoint(args.copy, endpoint)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    try:  # the store this run reads, as it says at the start: its index name and triple count
        got = fetch_stats(endpoint)
        stats = {k: got[k] for k in STATS_FIELDS}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"refused: the store's ?cmd=stats cannot be read ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2
    make = plain_factory(endpoint, args.timeout_s)
    results = []
    for qid, q in read_questions(args.questions).items():
        if q.get("split") != args.split:
            continue
        r = send(make, q["gold_sparql"], resends=0)
        outcome = {"ok": "answered", "timeout": "timeout:client"}.get(r["outcome"]) or \
            f"error:{(r['error_kinds'] or ['unknown'])[-1]}"
        results.append({"id": qid, "split": q["split"], "outcome": outcome, "wall_s": r["wall_s"]})
    summary = {"questions": len(results), "answered": sum(r["outcome"] == "answered" for r in results),
               "timeouts": sum(r["outcome"].startswith("timeout:") for r in results),
               "errors": sum(r["outcome"].startswith("error:") for r in results), "split": args.split,
               "store": store.get("name"), "questions_sha256": sha256_file(args.questions), "copy": args.copy,
               "endpoint": endpoint, "stats": stats}
    Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


def gold_run_copy(gold: dict, path: Path) -> str | None:
    """The copy whose identity check a gold run is: the one its summary names (scripts/store_probe.py golds), or, for an
    a15 qualification gold re-run in the private tree (copies/<site>/gold-rerun.json), "a15-<site>"; else None."""
    named = (gold.get("summary") or {}).get("copy")
    if named:
        return named
    path = Path(path).resolve()
    if path.name == "gold-rerun.json" and path.parent.parent.name == "copies":
        return f"a15-{path.parent.name}"
    return None


def cmd_build(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists():
        print(f"refused: {out} exists", file=sys.stderr)
        return 2
    if args.copy.startswith("a15-") and not args.private:
        print("refused: an a15 copy's set is private (--private): it names a15 questions", file=sys.stderr)
        return 2
    if args.private and in_git_tree(out):
        print("refused: a private probe set never goes inside a git work tree", file=sys.stderr)
        return 2
    try:
        seed = sf.probe_seed(args.copy)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    gold = json.loads(Path(args.gold_rerun).read_text(encoding="utf-8"))
    named = gold_run_copy(gold, Path(args.gold_rerun))
    if named != args.copy:
        print(f"refused: the gold run is copy {named!r}'s identity check, not {args.copy}'s", file=sys.stderr)
        return 2
    if not args.copy.startswith("a15-"):  # a public gold run must have read the copy's registered endpoint
        try:
            sf.check_copy_endpoint(args.copy, (gold.get("summary") or {}).get("endpoint"))
        except ValueError as exc:
            print(f"refused: the gold run's endpoint: {exc}", file=sys.stderr)
            return 2
    questions = read_questions(args.questions)
    try:
        ids = sf.pick_probe_queries(gold.get("results") or [], seed)
        texts = {q: questions[q]["gold_sparql"] for q in ids}
    except (ValueError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    source = {"gold_rerun_sha256": sha256_file(args.gold_rerun), "questions_sha256": sha256_file(args.questions),
              "split": "dev", "completed": sum(1 for r in gold.get("results") or []
                                               if r.get("split") == "dev" and r.get("outcome") == "answered")}
    raw = sf.probe_set_bytes(sf.probe_set_doc(args.copy, args.store, seed, ids, texts, source))
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if args.private else 0o644)
    with os.fdopen(fd, "wb") as fh:
        fh.write(raw)
    print(json.dumps({"copy": args.copy, "store": args.store, "n": len(ids), "seed": seed,
                      "sha256": hashlib.sha256(raw).hexdigest(), "completed": source["completed"]}, sort_keys=True))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists():
        print(f"refused: {out} exists", file=sys.stderr)
        return 2
    try:
        doc = sf.load_store_probe_set(args.set, args.sha256)
        questions = read_questions(args.questions)
        texts = []
        for q in doc["queries"]:
            text = questions[q["id"]]["gold_sparql"]
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != q["sparql_sha256"]:
                raise ValueError(f"probe query {len(texts) + 1}: its text no longer has the set's sha256")
            texts.append(text)
        store = json.loads(Path(args.store_json).read_text(encoding="utf-8"))
        make, gated = store_factory(store, args.timeout_s)
    except (ValueError, KeyError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    started = time.time()
    rows, verdict = run_probe(texts, make)
    out.mkdir(parents=True)
    with (out / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    summary = {**verdict, "copy": args.copy or doc.get("copy"), "set_sha256": args.sha256, "gated": gated,
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
               "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "wall_s_max": max((r["wall_s"] or 0) for r in rows) if rows else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if verdict["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    golds = sub.add_parser("golds", help="a plain-client copy's identity-check gold run")
    golds.add_argument("--store-json", type=Path, required=True)
    golds.add_argument("--questions", type=Path, required=True)
    golds.add_argument("--copy", required=True, help="the copy whose identity check this is (an approved id)")
    golds.add_argument("--out", type=Path, required=True)
    golds.add_argument("--split", default="dev")
    golds.add_argument("--timeout-s", type=float, default=60.0)
    build = sub.add_parser("build", help="the copy's probe set from its identity-check gold run")
    build.add_argument("--gold-rerun", type=Path, required=True)
    build.add_argument("--questions", type=Path, required=True)
    build.add_argument("--copy", required=True)
    build.add_argument("--store", required=True, help="the store's name, e.g. uniprot-sp-2026_03")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--private", action="store_true", help="an a15 copy: mode 600, never in a git work tree")
    run = sub.add_parser("run", help="send the copy's probe set and judge it")
    run.add_argument("--store-json", type=Path, required=True)
    run.add_argument("--set", type=Path, required=True)
    run.add_argument("--sha256", required=True, help="the set's sha256 as the frozen config binds it")
    run.add_argument("--questions", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--copy", default=None)
    run.add_argument("--timeout-s", type=float, default=60.0, help="the plain client's time-out (--sparql-timeout)")
    args = ap.parse_args(argv)
    return {"golds": cmd_golds, "build": cmd_build, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
