"""The fault bank's production replay (dress rehearsal 2, B2; prereg section 3 "Bank metrics"): every bank source's
checks on every checked item of a split of the frozen bank, as ``matrix.run_bank`` checks them, in chunks that a run
can resume and that several processes can share, with each chunk's store log kept so that the whole matrix replays
offline with no store.

The run's directory OUT holds:

  plan.json              what the run checks and with what (``plan``): a resumed or sharded run must have exactly the
                         same plan (inputs, item ids, code, the store's identity), or it refuses
  chunks/NNNNN.json.gz   one chunk (``chunk`` items in the bank's order): its plan's digest, index and item ids, each
                         item's store log and the chunk's matrix rows, in one file written atomically
  progress/<shard>.jsonl what each process did, when (its start and end with the store's ?cmd=stats, every chunk)
  matrix.jsonl.gz        the merged matrix, every chunk in order (byte for byte the same however it was sharded)
  store-log.jsonl.gz     the merged store logs (``--offline`` replays them)
  run.json               the plan, the counts, the bank rates per arm and variant (descriptive), the progress

A chunk is read back only when its plan digest, index, item ids and rows (one per item, arm and variant) are the
plan's, so a chunk of another plan or a damaged one stops the merge. A live run stops, writing nothing for the chunk,
when the store did not answer (the recording client would otherwise turn an outage into "unknown" verdicts): a
transport failure or a client exception always stops it; an HTTP 5xx stops it unless it recurs on a second check of
the chunk while the store's ?cmd=stats identity is the plan's and a bounded canary query is answered, which makes it
that query's own failure, kept as the check's outcome and listed in run.json. A store whose ?cmd=stats identity is no
longer the plan's (restarted on another index) stops it too; a resume re-checks that chunk. Server time-outs are
checks' outcomes and are counted, not fatal. Plan creation and the merge hold OUT's lock; each shard K/N holds its
own. The merged matrix.jsonl.gz and store-log.jsonl.gz are the same bytes however the run was sharded or resumed
(run.json also holds the processes' progress logs, which are not).

The test split's bank verdicts are test outcomes, so ``require_freeze_deploy`` refuses split test unless the run is
confirmatory and the running code is the freeze tag's commit, by its commit and its tree digest recomputed from git's
objects (design ruling, design log, 20 Oct 15:19), and ``run_chunks`` refuses test items without that record.
"""

from __future__ import annotations

import base64
import fcntl
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import time
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.agent.replay_check import CODE_PATHS, CodeNotIdentified, records_sha256, require_identified_code
from harness.checkloop.ledger import LedgerError, frozen_bank
from harness.checkloop.matrix import BANK, MATRIX_VERSION, VARIANTS, Arms, bank_items, bank_rates, run_bank

PLAN_SCHEMA = "bank-replay-plan/2"
CHUNK_SCHEMA = "bank-replay-chunk/1"
RUN_SCHEMA = "bank-replay/1"
DEFAULT_SOURCES = ("curated", "census-t5", "generated-dc-1", "generated-dc-2", "generated-dc-3")
DEFAULT_CHUNK = 20
TEST_SPLIT = "test"
FETCH_TIMEOUT_S = 120
# ?cmd=stats fields that name the index a QLever server serves (its build and its counts; never its load)
STORE_IDENTITY = ("git-hash-index", "name-index", "num-permutations", "num-predicates-normal", "num-subjects-normal",
                  "num-objects-normal", "num-triples-normal")
# recorded query errors that mean the store did not answer: the run stops rather than record "unknown" verdicts
NO_ANSWER_KINDS = ("transport", "client", "gate")


class BankReplayError(Exception):
    """The run cannot go ahead as asked; the message says why."""


# -- what is checked ---------------------------------------------------------------------------------------------


def checked_items(manifest: Path, frozen_sha256: str, split: str,
                  questions: Sequence[str] | None = None) -> tuple[dict, list[dict]]:
    """(manifest, items): the checked items of ``split`` of the frozen bank (``matrix.bank_items``: untouched controls,
    rewrites, answer-changing and answer-neutral mutations, in the file's order), of the declared ``questions`` when
    given. The manifest must be the frozen one and its items the bytes it froze; a declared question with no checked
    item of the split, or one declared twice, is refused."""
    try:
        data, items = frozen_bank(manifest, frozen_sha256)
    except LedgerError as exc:
        raise BankReplayError(str(exc)) from None
    chosen = bank_items(items, split)
    if not chosen:
        raise BankReplayError(f"{manifest}: no checked item of split {split!r}")
    if questions is not None:
        if not questions or len(set(questions)) != len(questions):
            raise BankReplayError("a declared question subset is a non-empty list of distinct question ids")
        unknown = sorted(set(questions) - {i["question_id"] for i in chosen})
        if unknown:
            raise BankReplayError(f"{len(unknown)} declared question(s) have no checked item of split {split!r}, e.g. "
                                  f"{unknown[:3]}")
        keep = set(questions)
        chosen = [i for i in chosen if i["question_id"] in keep]
    return data, chosen


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def plan(*, manifest: Path, manifest_data: dict, split: str, items: Sequence[dict], questions: Sequence[str] | None,
         chunk: int, sources: dict[str, dict], unions: dict[str, Sequence[str]], hierarchy: dict, store: dict,
         code: dict, confirmatory: dict | None, shards: int = 1) -> dict:
    """What the run checks and with what; every field decides the matrix, so a resumed or sharded run must have it
    exactly (``open_run``). The item ids and the arms are listed in order (each chunk is checked against its slice and
    the arms' order, which a dict read back from plan.json would not keep), and the shard count is fixed, so that two
    runs sharded differently never write the same chunk."""
    if chunk < 1:
        raise BankReplayError("--chunk must be at least 1")
    if split == TEST_SPLIT and confirmatory is None:
        raise BankReplayError("a test-split plan needs the freeze deploy's record (require_freeze_deploy)")
    if shards < 1:
        raise BankReplayError("the shard count must be at least 1")
    ids = [i["id"] for i in items]
    return {
        "schema": PLAN_SCHEMA, "matrix_version": MATRIX_VERSION, "variants": list(VARIANTS),
        "manifest": {"path": str(manifest), "sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                     "items_sha256": (manifest_data.get("files") or {}).get("items.jsonl"),
                     "graph": manifest_data.get("graph")},
        "split": split,
        "questions": None if questions is None else {"n": len(set(questions)), "ids": sorted(set(questions)),
                                                     "sha256": sha256_json(sorted(set(questions)))},
        "items": {"n": len(ids), "ids_sha256": sha256_json(ids), "ids": ids},
        "chunk": chunk, "n_chunks": n_chunks(len(ids), chunk), "shards": shards,
        "arms": [*sources, *unions], "sources": sources, "unions": {k: list(v) for k, v in unions.items()},
        "hierarchy": hierarchy,
        "store": store, "code": {"git_commit": code.get("git_commit"), "tree_sha256": code.get("tree_sha256")},
        "confirmatory": confirmatory,
    }


def plan_sha256(the_plan: dict) -> str:
    return sha256_json(the_plan)


def n_chunks(n_items: int, chunk: int) -> int:
    return (n_items + chunk - 1) // chunk


def parse_shard(text: str) -> tuple[int, int]:
    """``K/N`` (0 <= K < N): this process takes the chunks whose index is K modulo N."""
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise BankReplayError(f"--shard {text!r} is not K/N") from None
    if not 0 <= k < n:
        raise BankReplayError(f"--shard {text!r}: K must be in 0..N-1")
    return k, n


@contextmanager
def locked(path: Path, *, wait: bool = True) -> Iterator[None]:
    """An exclusive flock on ``path`` for the block; with ``wait`` False, refused when another process holds it."""
    with open(path, "a+b") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            raise BankReplayError(f"another process holds {path}: that shard is running already") from None
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def open_run(out: Path, the_plan: dict) -> None:
    """Start OUT with the plan, or resume it (under OUT's lock): an OUT whose plan differs in any field is refused (its
    chunks were checked with something else)."""
    existing = next(a for a in (out, *out.parents) if a.exists())
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks").mkdir(exist_ok=True)
    (out / "progress").mkdir(exist_ok=True)
    # every new directory's entry survives a crash, as the chunks written into them do: fsync from OUT up to the
    # first directory that was already there
    created = [d for d in (out, *out.parents) if d != existing and existing in d.parents]
    for d in dict.fromkeys([out, *created, existing, out.parent]):  # OUT's parent always: another shard may have
        _fsync_dir(d)                                                 # created OUT and died before syncing it
    with locked(out / ".lock"):
        path = out / "plan.json"
        if path.is_file():
            found = json.loads(path.read_text(encoding="utf-8"))
            if found != the_plan:
                differ = sorted(k for k in set(found) | set(the_plan) if found.get(k) != the_plan.get(k))
                raise BankReplayError(f"{out} holds a run of another plan (it differs in {differ}); resume it with the "
                                      f"same inputs, code and store, or use another --out")
            return
        _write_atomic(path, (json.dumps(the_plan, indent=1, sort_keys=True) + "\n").encode("utf-8"))


# -- files -------------------------------------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(tmp: Path, path: Path) -> None:
    """Rename a written, fsynced file into place and fsync its directory, so a crash never leaves a renamed file
    without its bytes, or a written chunk unrecorded."""
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    _publish(tmp, path)


def _gzip(data: bytes) -> bytes:
    """Deterministic gzip (no time stamp, no file name)."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        gz.write(data)
    return buf.getvalue()


def _line(item: Any) -> bytes:
    return (json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def chunk_path(out: Path, index: int) -> Path:
    return out / "chunks" / f"{index:05d}.json.gz"


def chunk_done(out: Path, index: int) -> bool:
    return chunk_path(out, index).is_file()


def chunk_ids(the_plan: dict, index: int) -> list[str]:
    c = the_plan["chunk"]
    return the_plan["items"]["ids"][index * c:(index + 1) * c]


def read_chunk(out: Path, index: int, the_plan: dict) -> dict:
    """A done chunk, refused unless it is the plan's: its plan digest, index and item ids (the plan's slice), one log
    per item in order, and one row per item, arm and variant, in item order."""
    path = chunk_path(out, index)
    try:
        chunk = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise BankReplayError(f"{path}: not a readable chunk ({exc}); delete it and resume") from None
    ids = chunk_ids(the_plan, index)
    if (chunk.get("schema"), chunk.get("plan_sha256"), chunk.get("index"), chunk.get("item_ids")) != (
            CHUNK_SCHEMA, plan_sha256(the_plan), index, ids):
        raise BankReplayError(f"{path}: not chunk {index} of this plan (its schema, plan digest, index or items differ)")
    if [e.get("run_id") for e in chunk.get("logs", [])] != ids:
        raise BankReplayError(f"{path}: its store logs are not one per item in order")
    want = [(i, a, v) for i in ids for a in the_plan["arms"] for v in the_plan["variants"]]
    got = [(r.get("run_id"), r.get("arm"), r.get("variant")) for r in chunk.get("rows", [])]
    if got != want:
        raise BankReplayError(f"{path}: its rows are not one per item, arm and variant in order ({len(got)} rows, "
                              f"{len(want)} expected)")
    return chunk


def log_progress(out: Path, shard: tuple[int, int], event: dict) -> None:
    with open(out / "progress" / f"{shard[0]}-of-{shard[1]}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": utc_now(), **event}, sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# -- the store ---------------------------------------------------------------------------------------------------


def store_stats(endpoint: str, timeout_s: float = 30.0) -> dict[str, Any]:
    """The store's ``?cmd=stats`` (QLever: index name, triple and entity counts), read once; an error is recorded, not
    raised. A bounded read of the server's own counters, never a query."""
    url = endpoint + ("&" if "?" in endpoint else "?") + "cmd=stats"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:  # noqa: S310 - the store's own endpoint
            return {"url": url, "stats": json.loads(r.read(1 << 20).decode("utf-8"))}
    except Exception as exc:  # noqa: BLE001 - the caller decides
        return {"url": url, "error": f"{type(exc).__name__}: {exc}"[:300]}


def store_fingerprint(stats: dict[str, Any]) -> dict[str, Any]:
    """The index a store serves, from its ?cmd=stats (``STORE_IDENTITY``); refused when the stats were not read or lack
    a field."""
    if "stats" not in stats or not isinstance(stats["stats"], dict):
        raise BankReplayError(f"the store's ?cmd=stats could not be read ({stats.get('error')}): its identity is unknown")
    missing = [k for k in STORE_IDENTITY if k not in stats["stats"]]
    if missing:
        raise BankReplayError(f"the store's ?cmd=stats lacks {missing}: its identity is unknown")
    return {k: stats["stats"][k] for k in STORE_IDENTITY}


def no_answer(logs: dict[str, dict]) -> list[str]:
    """The recorded query errors that may mean the store did not answer (``NO_ANSWER_KINDS``, an HTTP 5xx), as
    ``item: kind: message`` lines. ``run_chunks`` asks the store whether it is up before deciding: down (or another
    index) stops the run; up, the chunk is checked once more, and failures that recur with the store up are the
    queries' own, kept as the checks' outcomes and counted."""
    found = []
    for item_id, log in logs.items():
        for e in log.get("entries", []):
            err = e.get("error")
            if err and (err.get("kind") in NO_ANSWER_KINDS
                        or (err.get("kind") == "http" and str(err.get("message", "")).startswith("HTTP 5"))):
                query = hashlib.sha256(str(e.get("query", "")).encode("utf-8")).hexdigest()[:16]
                found.append(f"{item_id}: {err.get('kind')}: query {query}: {str(err.get('message'))[:200]}")
    return found


def _failure_key(line: str) -> tuple[str, ...]:
    """A ``no_answer`` line's item, kind and query, without the message."""
    return tuple(line.split(": ", 3)[:3])


def timeouts(logs: dict[str, dict]) -> int:
    return sum(1 for log in logs.values() for e in log.get("entries", []) if (e.get("error") or {}).get("kind") == "timeout")


# -- the run -----------------------------------------------------------------------------------------------------


def run_chunks(out: Path, the_plan: dict, items: Sequence[dict], arms: Arms, graph: str, *,
               shard: tuple[int, int] = (0, 1), live_client: Any = None, store_logs: dict[str, dict] | None = None,
               check_store: Callable[[], None] | None = None, canary: Callable[[], None] | None = None,
               authorised: dict | None = None, max_chunks: int | None = None,
               say: Callable[[str], None] = lambda _: None) -> dict[str, int]:
    """Check this shard's chunks that are not done yet, each written atomically as one file. ``store_logs`` replays (no
    store); else ``live_client`` is read through a recording client per item, ``check_store`` (the store's identity,
    the plan's; raises otherwise) runs before each chunk is written, and a chunk whose store may not have answered
    (``no_answer``) is decided as the module says, ``canary`` (a bounded data query; raises when unanswered) proving
    the store answers data queries before an HTTP 5xx is kept as a query's own. ``items`` must be the plan's items.
    Test items need ``authorised`` (require_freeze_deploy's record). ``max_chunks`` stops after that many (an
    interrupted run, in tests). Returns counts."""
    if [i["id"] for i in items] != the_plan["items"]["ids"]:
        raise BankReplayError("the items are not the plan's")
    if list(arms.names) != the_plan["arms"]:
        raise BankReplayError(f"the arms {list(arms.names)} are not the plan's {the_plan['arms']} (names and order)")
    if shard[1] != the_plan["shards"]:
        raise BankReplayError(f"shard {shard[0]}/{shard[1]}: this OUT's plan is sharded {the_plan['shards']} ways")
    if any(i.get("split") == TEST_SPLIT for i in items) and (authorised is None or authorised != the_plan["confirmatory"]):
        raise BankReplayError("test-split items are checked only by a confirmatory run from the freeze deploy "
                              "(require_freeze_deploy's record, the plan's)")
    chunk = the_plan["chunk"]
    total = the_plan["n_chunks"]
    mine = [i for i in range(total) if i % shard[1] == shard[0]]
    digest = plan_sha256(the_plan)
    done = skipped = checked = waited = 0
    t0 = time.monotonic()
    with locked(out / f".shard-{shard[0]}-of-{shard[1]}.lock", wait=False):
        for index in mine:
            if chunk_done(out, index):
                skipped += 1
                continue
            if max_chunks is not None and done >= max_chunks:
                break
            part = items[index * chunk:(index + 1) * chunk]
            t = time.monotonic()
            kept: list[str] = []
            first: list[str] = []
            for attempt in (1, 2):
                try:
                    rows, logs = run_bank(part, arms, graph, live_client=live_client, store_logs=store_logs)
                except RuntimeError as exc:  # an offline replay whose log does not reproduce the checks
                    raise BankReplayError(str(exc)) from None
                failed = no_answer(logs) if store_logs is None else []
                if not failed:
                    break
                if check_store is None:
                    raise BankReplayError(f"chunk {index}: the store did not answer {len(failed)} quer(ies), first: "
                                          f"{failed[0]}; nothing written for the chunk")
                try:  # down, or another index: an outage, not the queries' own failures
                    check_store()
                except BankReplayError as exc:
                    raise BankReplayError(f"chunk {index}: the store did not answer {len(failed)} quer(ies) (first: "
                                          f"{failed[0]}) and is not up as the plan's store ({exc}); nothing written "
                                          f"for the chunk, resume once the store is back") from None
                if attempt == 1:
                    first = failed
                    say(f"chunk {index}: {len(failed)} store failure(s) with the store up, checking the chunk again")
                    continue
                # recurring with the store's identity the plan's: an HTTP 5xx is kept as that query's own failure only
                # when the same query of the same item failed so on the first check too and the store answers a data
                # query; a transport failure or client exception never is
                never = [f for f in failed if f.split(": ", 2)[1] in NO_ANSWER_KINDS]
                seen = {_failure_key(f) for f in first}  # item, kind and query; a body may differ between checks
                never += [f for f in failed if _failure_key(f) not in seen and f not in never]
                if never or canary is None:
                    raise BankReplayError(f"chunk {index}: the store did not answer {len(failed)} quer(ies) twice, first: "
                                          f"{(never or failed)[0]}; nothing written for the chunk")
                try:
                    canary()
                except Exception as exc:  # noqa: BLE001 - any failure of the canary means the store does not answer
                    raise BankReplayError(f"chunk {index}: {len(failed)} quer(ies) failed twice (first: {failed[0]}) "
                                          f"and the store does not answer a canary query ({exc}); nothing written for "
                                          f"the chunk") from None
                kept = failed  # the queries' own failures, recorded as the checks' outcomes
            if store_logs is None and check_store is not None:
                check_store()
            n_timeouts = timeouts(logs)
            took = time.monotonic() - t
            meta = {"seconds": round(took, 2), "store_timeouts": n_timeouts, "query_failures": kept}
            # the event first: every written chunk has one (a crash before the write repeats it, harmlessly)
            log_progress(out, shard, {"event": "chunk", "index": index, "items": len(part), **meta})
            body = {"schema": CHUNK_SCHEMA, "plan_sha256": digest, "index": index, "item_ids": [i["id"] for i in part],
                    "logs": [{"run_id": i["id"], "log": logs[i["id"]]} for i in part],
                    "rows": [{"version": MATRIX_VERSION, **r} for r in rows], "meta": meta}
            _write_atomic(chunk_path(out, index), _gzip(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()))
            done += 1
            checked += len(part)
            waited += n_timeouts
            say(f"chunk {index + 1}/{total}: {len(part)} items in {took:.0f}s; this shard {checked} items in "
                f"{time.monotonic() - t0:.0f}s ({(time.monotonic() - t0) / max(checked, 1):.1f} s/item)"
                + (f"; {n_timeouts} store time-outs" if n_timeouts else "")
                + (f"; {len(kept)} query failures kept as outcomes" if kept else ""))
    return {"chunks": total, "mine": len(mine), "checked_chunks": done, "skipped_chunks": skipped,
            "checked_items": checked, "store_timeouts": waited}


def all_done(out: Path, total: int) -> bool:
    return all(chunk_done(out, i) for i in range(total))


def merge(out: Path, the_plan: dict) -> dict[str, Any]:
    """matrix.jsonl.gz and store-log.jsonl.gz from every chunk in order (each read back only as the plan's,
    ``read_chunk``), and run.json, under OUT's lock: the same bytes however the run was sharded or resumed."""
    total = the_plan["n_chunks"]
    with locked(out / ".lock"):
        missing = [i for i in range(total) if not chunk_done(out, i)]
        if missing:
            raise BankReplayError(f"{len(missing)} of {total} chunks are not done yet (e.g. {missing[:5]}): run the "
                                  f"remaining shards first")
        slim: list[dict] = []
        metas: list[dict] = []
        rows_tmp, log_tmp = out / f".matrix.{os.getpid()}.tmp", out / f".store-log.{os.getpid()}.tmp"
        n_rows = n_items = 0
        with open(rows_tmp, "wb") as raw_rows, open(log_tmp, "wb") as raw_log, \
                gzip.GzipFile(filename="", mode="wb", fileobj=raw_rows, mtime=0) as gz_rows, \
                gzip.GzipFile(filename="", mode="wb", fileobj=raw_log, mtime=0) as gz_log:
            for index in range(total):
                chunk = read_chunk(out, index, the_plan)
                metas.append(chunk.get("meta") or {})
                for entry in chunk["logs"]:
                    n_items += 1
                    gz_log.write(_line(entry))
                for row in chunk["rows"]:
                    n_rows += 1
                    gz_rows.write(_line(row))
                    slim.append({k: row.get(k) for k in ("cohort", "output", "status", "arm", "variant", "correct",
                                                         "fault_class", "bank_item", "result_rows")})
        for tmp in (rows_tmp, log_tmp):
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
        _publish(rows_tmp, out / "matrix.jsonl.gz")
        _publish(log_tmp, out / "store-log.jsonl.gz")
        # the counts from the chunks themselves; the progress logs are the processes' own account, kept beside them
        progress = {p.stem: [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
                    for p in sorted((out / "progress").glob("*.jsonl"))}
        run = {"schema": RUN_SCHEMA, "plan_sha256": plan_sha256(the_plan), "plan": the_plan, "items": n_items,
               "rows": n_rows, "files": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                                         for name in ("matrix.jsonl.gz", "store-log.jsonl.gz")},
               "chunk_seconds": round(sum(m.get("seconds", 0) for m in metas), 1),
               "store_timeouts": sum(m.get("store_timeouts", 0) for m in metas),
               "query_failures": [f for m in metas for f in m.get("query_failures", [])],
               "bank_rates": bank_rates(r for r in slim if r["cohort"] == BANK), "progress": progress}
        _write_atomic(out / "run.json", (json.dumps(run, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    return run


def read_store_logs(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {e["run_id"]: e["log"] for e in map(json.loads, f) if e}


# -- the test split: a confirmatory deploy of the freeze tag only ------------------------------------------------


def _git(repo: Path, *args: str, text: bool = True, timeout: float = 60) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text, timeout=timeout,
                              check=False, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    except (OSError, subprocess.SubprocessError) as exc:
        raise BankReplayError(f"git cannot be run in {repo} ({exc})") from None


def git_tree_sha256(repo: Path, commit: str) -> str:
    """The code tree digest of ``commit`` from git's objects alone (no working tree): replay_check.tree_sha256's
    definition, over CODE_PATHS (no __pycache__, no .pyc or .pyo), each file's sha256 by its blob."""
    listed = _git(repo, "ls-tree", "-r", "-z", commit, "--", *CODE_PATHS)
    if listed.returncode != 0:
        raise BankReplayError(f"git ls-tree {commit[:12]} failed in {repo}: {listed.stderr.strip()[:200]}")
    blobs: dict[str, str] = {}
    for entry in listed.stdout.split("\0"):
        if not entry:
            continue
        meta, name = entry.split("\t", 1)
        mode, kind, obj = meta.split()
        if "__pycache__" in name.split("/") or name.endswith((".pyc", ".pyo")):
            continue
        if kind != "blob" or mode == "120000":
            raise BankReplayError(f"{name}: a {kind} of mode {mode} in {commit[:12]}, not a regular file")
        blobs[name] = obj
    batch = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch"],
                           input="".join(f"{o}\n" for o in blobs.values()).encode(), capture_output=True, check=False,
                           timeout=600)
    if batch.returncode != 0:
        raise BankReplayError(f"git cat-file failed in {repo}")
    data, digests, pos = batch.stdout, {}, 0
    for name in blobs:  # each object: "<sha> blob <size>\n<bytes>\n", in the order asked
        header_end = data.index(b"\n", pos)
        size = int(data[pos:header_end].split()[2])
        digests[name] = hashlib.sha256(data[header_end + 1:header_end + 1 + size]).hexdigest()
        pos = header_end + 1 + size + 1
    return records_sha256(digests)


def require_freeze_deploy(split: str, *, confirmatory: bool, freeze_repo: Path | None, freeze_tag: str | None,
                          code_root: Path | None = None, fetch: bool = True) -> dict | None:
    """None for a split other than test. For the test split (its bank verdicts are test outcomes, design ruling,
    design log): refused unless ``confirmatory`` and the running code is the freeze tag's: the code is identified and clean
    (replay_check.require_identified_code: a deploy's files are its TREE, bound to its COMMIT), ``freeze_tag`` is a tag
    of ``freeze_repo`` as its origin has it now (fetched, forced; a failed fetch refuses), the tag's commit is the
    code's commit, and the tree digest of the tag's commit computed from git's objects is the code's tree digest
    recomputed from its files. The command line adds the frozen pre-registration (the test grid's gate). Runs before
    anything is read or any store is contacted. Returns what the plan records (the tag, its commit and tree; the time
    of the fetch is the progress log's, so that every shard's plan is the same)."""
    if split != TEST_SPLIT:
        return None
    if not confirmatory:
        raise BankReplayError("the test split's bank verdicts are test outcomes: they run only in the confirmatory run, "
                              "with --confirmatory, from a deploy of the freeze tag (--freeze-repo, --freeze-tag)")
    if freeze_repo is None or not freeze_tag:
        raise BankReplayError("a confirmatory test-split run needs --freeze-repo and --freeze-tag: the running code must "
                              "be the freeze tag's commit")
    try:
        code = require_identified_code(root=code_root)
    except CodeNotIdentified as exc:
        raise BankReplayError(str(exc)) from None
    if code.get("git_dirty") is not False:
        raise BankReplayError(f"the running code ({str(code.get('git_commit'))[:12]}) has uncommitted changes or cannot "
                              f"be told clean: the test split runs from a clean deploy of the freeze tag only")
    ref = f"refs/tags/{freeze_tag}"
    if fetch:
        got = _git(freeze_repo, "fetch", "--quiet", "--no-tags", "origin", f"+{ref}:{ref}", timeout=FETCH_TIMEOUT_S)
        if got.returncode != 0:
            raise BankReplayError(f"could not fetch {ref} from {freeze_repo}'s origin ({' '.join(got.stderr.split())[:200]}): "
                                  f"the freeze tag is read from the origin as it is now, never a local copy")
    tag = _git(freeze_repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    if tag.returncode != 0:
        raise BankReplayError(f"{freeze_tag} is not a tag of {freeze_repo}")
    commit = tag.stdout.strip()
    if commit != code.get("git_commit"):
        raise BankReplayError(f"the running code is commit {str(code.get('git_commit'))[:12]}, not the freeze tag "
                              f"{freeze_tag}'s {commit[:12]}")
    tree = git_tree_sha256(freeze_repo, commit)
    if tree != code.get("tree_sha256"):
        raise BankReplayError(f"the running code's tree digest {str(code.get('tree_sha256'))[:12]} (from its files) is "
                              f"not the freeze tag's {tree[:12]} (from git's objects)")
    return {"freeze_tag": freeze_tag, "commit": commit, "tree_sha256": tree}


# The freeze gate from a manifest (lead, 21 Oct): no git clone where the job runs (SiteA's project space is
# group-readable and the experiments repository's objects hold the held-out blobs). On the VM the tag and the frozen
# pre-registration are verified as the git gates verify them (make_freeze_manifest, scripts/bank_freeze_manifest.py);
# where the job runs, the deploy is checked against the manifest (require_freeze_manifest).
FREEZE_MANIFEST_SCHEMA = "bank-freeze-manifest/1"
PREREG_PATH = "paper/preregistration.md"
CONFIG_PATH = "data/analysis/prereg-config.json"
_FROZEN_LINE = re.compile(r"^Status: frozen$", re.MULTILINE)  # the test grid's gate's exact whole line


def _frozen_bytes(data: bytes, what: str) -> None:
    if not _FROZEN_LINE.search(data.decode("utf-8", errors="replace").replace("\r\n", "\n")):
        raise BankReplayError(f"{what} has no line 'Status: frozen': the pre-registration is not frozen there")


def make_freeze_manifest(freeze_repo: Path, freeze_tag: str, prereg: dict, prereg_bytes: bytes, signed_off_by: str,
                         *, fetch: bool = True) -> dict:
    """The freeze manifest of the bank job, made where git is (the VM): ``freeze_tag`` as ``freeze_repo``'s origin has
    it now (fetched, forced; a failed fetch refuses), its commit, its tree digest from git's objects (git_tree_sha256)
    and the sha256 of its config (CONFIG_PATH at the tag); the frozen pre-registration as the test grid's gate returned
    it (``prereg``: its commit and preregistration_sha256) with its bytes, which must have that sha256 and the exact
    line 'Status: frozen'; who signed it off and when."""
    if not (signed_off_by or "").strip():
        raise BankReplayError("the freeze manifest is signed off by someone: give the name")
    digest = hashlib.sha256(prereg_bytes).hexdigest()
    if digest != prereg.get("preregistration_sha256"):
        raise BankReplayError(f"the pre-registration bytes have sha256 {digest[:12]}, not the frozen pre-registration's "
                              f"{str(prereg.get('preregistration_sha256'))[:12]} (the gate's)")
    _frozen_bytes(prereg_bytes, f"the pre-registration at {str(prereg.get('commit'))[:12]}")
    ref = f"refs/tags/{freeze_tag}"
    if fetch:
        got = _git(freeze_repo, "fetch", "--quiet", "--no-tags", "origin", f"+{ref}:{ref}", timeout=FETCH_TIMEOUT_S)
        if got.returncode != 0:
            raise BankReplayError(f"could not fetch {ref} from {freeze_repo}'s origin ({' '.join(got.stderr.split())[:200]})")
    tag = _git(freeze_repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    if tag.returncode != 0:
        raise BankReplayError(f"{freeze_tag} is not a tag of {freeze_repo}")
    commit = tag.stdout.strip()
    config = _git(freeze_repo, "show", f"{commit}:{CONFIG_PATH}", text=False)
    if config.returncode != 0:
        raise BankReplayError(f"{CONFIG_PATH} is not in {commit[:12]}")
    return {"schema": FREEZE_MANIFEST_SCHEMA, "freeze_tag": freeze_tag, "commit": commit,
            "tree_sha256": git_tree_sha256(freeze_repo, commit),
            "config_sha256": hashlib.sha256(config.stdout).hexdigest(),
            "preregistration": {"commit": prereg.get("commit"), "path": PREREG_PATH, "sha256": digest,
                                "bytes_b64": base64.b64encode(prereg_bytes).decode()},
            "signed_off_by": signed_off_by.strip(), "made_at": utc_now()}


def require_freeze_manifest(split: str, *, confirmatory: bool, manifest: Path | None, frozen_config: Path,
                            manifest_sha256: str | None = None, code_root: Path | None = None) -> dict | None:
    """require_freeze_deploy's checks against a freeze manifest instead of git (make_freeze_manifest): None for a split
    other than test; for the test split refused unless ``confirmatory``, the running code is identified and clean, its
    commit is the manifest's and its tree digest recomputed from its files is the manifest's (from git's objects, on
    the VM), the config the run reads (``frozen_config``) has the manifest's sha256, and the pre-registration bytes the
    manifest carries have its recorded sha256 and the exact line 'Status: frozen'. The manifest is authenticated first
    by ``manifest_sha256``, its sha256 as the VM computed it right after verifying the tag and the freeze, given on the
    job's command line (the submission, held outside the copied archive): a manifest replaced in the archive, or an
    old one, is refused before any field of it is read (External review of 1f11585). Runs before anything is read or any
    store is contacted. Returns what the plan records (as the git gate and the pre-registration's gate give it, with
    the manifest's sha256)."""
    if split != TEST_SPLIT:
        return None
    if not confirmatory:
        raise BankReplayError("the test split's bank verdicts are test outcomes: they run only in the confirmatory run, "
                              "with --confirmatory, from a deploy of the freeze tag")
    if manifest is None:
        raise BankReplayError("a confirmatory test-split run needs --freeze-manifest (or --freeze-repo and --freeze-tag "
                              "with --prereg-repo and --prereg-commit): the running code must be the freeze tag's")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256 or ""):
        raise BankReplayError("a confirmatory test-split run from a freeze manifest needs --freeze-manifest-sha256: the "
                              "manifest's sha256 as the VM computed it (the trust anchor, held outside the archive)")
    raw = Path(manifest).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise BankReplayError(f"{manifest} has sha256 {hashlib.sha256(raw).hexdigest()[:12]}, not the "
                              f"{manifest_sha256[:12]} the VM wrote and the submission names: refused unread")
    try:
        m = json.loads(raw)
    except ValueError as exc:
        raise BankReplayError(f"{manifest}: not JSON ({exc})") from None
    if not isinstance(m, dict) or m.get("schema") != FREEZE_MANIFEST_SCHEMA:
        raise BankReplayError(f"{manifest}: schema {m.get('schema') if isinstance(m, dict) else None!r}, not "
                              f"{FREEZE_MANIFEST_SCHEMA!r}")
    pre = m.get("preregistration")
    if not isinstance(pre, dict) or not all(isinstance(m.get(k), str) for k in ("freeze_tag", "commit", "tree_sha256",
                                                                                 "config_sha256")) \
            or not all(isinstance(pre.get(k), str) for k in ("commit", "sha256", "bytes_b64")):
        raise BankReplayError(f"{manifest}: a field is missing or not a string (freeze_tag, commit, tree_sha256, "
                              f"config_sha256; preregistration commit, sha256, bytes_b64)")
    try:
        code = require_identified_code(root=code_root)
    except CodeNotIdentified as exc:
        raise BankReplayError(str(exc)) from None
    if code.get("git_dirty") is not False:
        raise BankReplayError(f"the running code ({str(code.get('git_commit'))[:12]}) has uncommitted changes or cannot "
                              f"be told clean: the test split runs from a clean deploy of the freeze tag only")
    tag, commit, tree = m.get("freeze_tag"), str(m.get("commit")), str(m.get("tree_sha256"))
    if commit != code.get("git_commit"):
        raise BankReplayError(f"the running code is commit {str(code.get('git_commit'))[:12]}, not the freeze tag "
                              f"{tag}'s {commit[:12]}")
    if tree != code.get("tree_sha256"):
        raise BankReplayError(f"the running code's tree digest {str(code.get('tree_sha256'))[:12]} (from its files) is "
                              f"not the freeze manifest's {tree[:12]} (from git's objects)")
    config_sha = hashlib.sha256(Path(frozen_config).read_bytes()).hexdigest()
    if config_sha != m.get("config_sha256"):
        raise BankReplayError(f"the config {frozen_config} (sha256 {config_sha[:12]}) is not the freeze manifest's "
                              f"{str(m.get('config_sha256'))[:12]} (the tag's {CONFIG_PATH})")
    try:
        data = base64.b64decode(pre["bytes_b64"], validate=True)
    except ValueError as exc:  # binascii.Error is a ValueError
        raise BankReplayError(f"{manifest}: the pre-registration bytes do not decode ({exc})") from None
    if hashlib.sha256(data).hexdigest() != pre.get("sha256"):
        raise BankReplayError(f"{manifest}: the pre-registration bytes do not have the recorded sha256 "
                              f"{str(pre.get('sha256'))[:12]}")
    _frozen_bytes(data, f"the pre-registration in {manifest}")
    return {"freeze_tag": tag, "commit": commit, "tree_sha256": tree,
            "preregistration": {"commit": pre.get("commit"), "preregistration_sha256": pre.get("sha256")},
            "freeze_manifest_sha256": hashlib.sha256(raw).hexdigest()}
