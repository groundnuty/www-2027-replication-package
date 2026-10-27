#!/usr/bin/env python3
"""Counts-only cost figures of the UniProt Tier 1 checker retries (H3), per source, pooled over models and per model.

    python3 -I cost_h3.py --out LABEL=OUTDIR [--complete LABEL=FILE] ... --ledger LEDGER.jsonl
                          --draws LABEL=DIR ... --out-json FILE --out-txt FILE

Blinding (the design log): this tool never reads or prints a runner log, a record's outcome, score, verdict, answer,
transcript, tool call, message or any correctness field. Every file is read through a window that answers only for the
locations listed below (``Forbidden`` for any other location), and a parsed document is dropped as soon as its few allowed values
are copied out. Nothing is printed per record: the output holds aggregates only (n, median, p90) and fixed exclusion tallies.

Allowed in a run record (RecordView): ``model``, ``wall_s``, ``load.arm``, ``checker.branched_from``,
``checker.settings.mode``, ``checker.settings.arm.name``, ``checker.retry.started_s``, ``checker.retry.turns``,
``runner.infra_try_run_ids`` (the run ids of the earlier tries of the attempt that failed on infrastructure: bookkeeping
the runner writes, used only to pick the final record among files that share a claim digest), whether
``checker`` and ``checker.retry`` are present (a bool each), and in each ``tokens.per_turn`` entry ``turn``, ``started_s``,
``input_tokens``, ``output_tokens``, ``cache_read_tokens``, ``cache_write_tokens``, ``latency_s``; the per-turn entries are
also hashed whole (never shown) to check that a branch record's replayed prefix is its parent's. Allowed in a run's
manifest.json: ``schema`` and the names of the branch arms (under the phases key of its first section).
Allowed in a ledger row: ``graph``, ``model``, ``cohort``, ``source``, ``use``, ``evidence_triples``, ``check_s``; in its
header ``schema`` and ``split``. Allowed in a draw record (draw-N.json): ``draw``, ``attempt``, ``status`` (as ok or not),
``model_returned``, ``cache_hit``, ``cost_usd``, ``usage`` (token counts) and ``prices_usd_per_mtok``; in batches.json the
batch ``group``, ``draws``, ``submitted_utc``, ``ended_utc``, ``collected`` and ``count.input_tokens``.

Definitions are in the output's ``definitions`` member and in the README.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone

SCHEMA = "cost-h3/1"
RECORD_NAME = re.compile(r"[0-9a-f]{40}\.[^./]+\.json")
SAFE_NAME = re.compile(r"[A-Za-z0-9_.+:/-]{1,80}")
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
SAFE_LABEL = re.compile(r"[A-Za-z0-9_.-]{1,80}")  # an OUT's or draws' label on the command line (OUT names hold dots)
NOT_ESTIMABLE = "not estimable"
NOT_RECORDED = "not recorded"
TURN_FIELDS = ("turn", "started_s", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "latency_s")
EPS = 1e-6


class Forbidden(Exception):
    """A location outside the allowlist was asked for."""


class Bad(Exception):
    """A malformed input of the command line or a document that cannot be used (no detail is printed)."""


MISSING = object()


def is_count(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def safe_str(x, pattern=SAFE_NAME):
    return x if isinstance(x, str) and pattern.fullmatch(x) else None


def canonical_digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def norm_model(name):
    """A model's key in every table: its id as the records and ledger rows write it, lower-cased with every run of
    other characters as one '-' (so 'deepseek-ai/DeepSeek-V4-Flash-0731' and 'deepseek-ai-deepseek-v4-flash-0731' agree);
    None when the name is missing or not of a safe shape."""
    if not safe_str(name):
        return None
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or None


# -- the windows: the only code that touches a parsed document ---------------------------------------------------


class View:
    """A window on one parsed JSON document. ``get`` answers for the listed scalar locations only, ``present`` for the listed
    presence locations only (a bool: the value is a JSON object), ``keys`` for the listed key locations only (sorted names of the
    object there). Any other location raises Forbidden, and a location that holds an object or a list is never returned."""

    SCALARS: frozenset = frozenset()
    PRESENT: frozenset = frozenset()
    KEYS: frozenset = frozenset()

    def __init__(self, doc):
        self._doc = doc

    def _walk(self, path):
        cur = self._doc
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return MISSING
            cur = cur[key]
        return cur

    def get(self, path):
        path = tuple(path)
        if path not in self.SCALARS:
            raise Forbidden("location")
        cur = self._walk(path)
        if cur is MISSING or isinstance(cur, (dict, list)):
            return None
        return cur

    def present(self, path) -> bool:
        path = tuple(path)
        if path not in self.PRESENT:
            raise Forbidden("location")
        return isinstance(self._walk(path), dict)

    def keys(self, path) -> list:
        path = tuple(path)
        if path not in self.KEYS:
            raise Forbidden("location")
        cur = self._walk(path)
        return sorted(k for k in cur if isinstance(k, str)) if isinstance(cur, dict) else []


def _turn_value(entry: dict, name: str):
    v = entry.get(name)
    return None if isinstance(v, (dict, list)) else v


class RecordView(View):
    SCALARS = frozenset({
        ("model",), ("wall_s",), ("load", "arm"), ("checker", "branched_from"), ("checker", "settings", "mode"),
        ("checker", "settings", "arm", "name"), ("checker", "retry", "started_s"), ("checker", "retry", "turns"),
    })
    PRESENT = frozenset({("checker",), ("checker", "retry")})

    def per_turn_entries(self):
        """The record's ``tokens.per_turn`` list, or None. Private to this class: callers get ``turns`` and the digests."""
        tokens = self._doc.get("tokens") if isinstance(self._doc, dict) else None
        per_turn = tokens.get("per_turn") if isinstance(tokens, dict) else None
        return per_turn if isinstance(per_turn, list) and all(isinstance(e, dict) for e in per_turn) else None

    def infra_try_ids(self) -> frozenset:
        """The run ids in ``runner.infra_try_run_ids`` (strings of a safe shape only)."""
        runner = self._doc.get("runner") if isinstance(self._doc, dict) else None
        ids = runner.get("infra_try_run_ids") if isinstance(runner, dict) else None
        return frozenset(i for i in ids if safe_str(i, SAFE_ID)) if isinstance(ids, list) else frozenset()

    def turn_count(self):
        entries = self.per_turn_entries()
        return None if entries is None else len(entries)

    def turns(self, start: int = 0) -> list:
        """The allowed fields of the per-turn entries from index ``start`` on (a list of small dicts)."""
        entries = self.per_turn_entries() or []
        return [{name: _turn_value(e, name) for name in TURN_FIELDS} for e in entries[start:]]

    def digest(self, upto: int | None = None):
        """sha256 of the first ``upto`` per-turn entries (all by default) as the cost tool of 24 Oct hashes them."""
        entries = self.per_turn_entries()
        if entries is None:
            return None
        return canonical_digest(entries if upto is None else entries[:upto])


class ManifestView(View):
    SCALARS = frozenset({("schema",)})
    KEYS = frozenset({("fingerprint", "phases", "branch", "arms")})


class LedgerView(View):
    SCALARS = frozenset({("graph",), ("model",), ("cohort",), ("source",), ("use",), ("evidence_triples",), ("check_s",)})


class LedgerHeaderView(View):
    SCALARS = frozenset({("schema",), ("split",)})


class DrawView(View):
    SCALARS = frozenset({
        ("draw",), ("attempt",), ("model_returned",), ("cache_hit",), ("cost_usd",), ("status",),
        ("usage", "input_tokens"), ("usage", "output_tokens"), ("usage", "cache_read_input_tokens"),
        ("usage", "cache_creation_input_tokens"), ("usage", "cache_creation", "ephemeral_5m_input_tokens"),
        ("usage", "cache_creation", "ephemeral_1h_input_tokens"),
        ("prices_usd_per_mtok", "input"), ("prices_usd_per_mtok", "output"), ("prices_usd_per_mtok", "cache_read"),
        ("prices_usd_per_mtok", "cache_write_1h"), ("prices_usd_per_mtok", "batch_discount"),
    })


class BatchesView(View):
    SCALARS = frozenset({("count", "input_tokens")})

    def batches(self) -> list:
        """Each batch's group, draws, submitted_utc, ended_utc and collected (nothing else)."""
        raw = self._doc.get("batches") if isinstance(self._doc, dict) else None
        out = []
        for b in raw if isinstance(raw, list) else []:
            if not isinstance(b, dict):
                continue
            draws = b.get("draws")
            out.append({
                "group": safe_str(b.get("group"), SAFE_ID),
                "draws": [d for d in draws if is_count(d)] if isinstance(draws, list) else [],
                "submitted_utc": safe_str(b.get("submitted_utc")),
                "ended_utc": safe_str(b.get("ended_utc")),
                "collected": b.get("collected") is True,
            })
        return out


# -- quantiles ---------------------------------------------------------------------------------------------------


def percentile(sorted_values: list, q: float) -> float:
    """Linear interpolation between order statistics (R type 7, numpy's default, statistics.quantiles 'inclusive'):
    position (n - 1) * q of the sorted values; one value is its own percentile."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("empty")
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summary(values: list) -> dict:
    vals = sorted(values)
    if not vals:
        return {"n": 0, "median": NOT_ESTIMABLE, "p90": NOT_ESTIMABLE}
    return {"n": len(vals), "median": round(percentile(vals, 0.5), 4), "p90": round(percentile(vals, 0.9), 4)}


# -- run records -------------------------------------------------------------------------------------------------


class Part:
    """What is kept of one record: the allowed values only."""

    __slots__ = ("model", "wall_s", "cell", "mode", "arm_name", "parent_id", "has_checker", "has_retry",
                 "retry_started_s", "retry_turns", "n_turns", "turns", "whole_digest", "prefix_digest", "infra_ids")

    def __init__(self):
        for name in self.__slots__:
            setattr(self, name, None)
        self.has_checker = self.has_retry = False
        self.infra_ids = frozenset()


def load_part(path: str) -> Part:
    """The allowed values of the record file given as the first argument. The parsed record is a local of this frame and is cleared in
    ``finally``; nothing of it but the Part leaves."""
    doc = None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        v = RecordView(doc)
        p = Part()
        p.model = safe_str(v.get(("model",)))
        p.wall_s = v.get(("wall_s",))
        p.cell = safe_str(v.get(("load", "arm")))
        p.has_checker = v.present(("checker",))
        p.n_turns = v.turn_count()
        if not p.has_checker:
            p.whole_digest = v.digest()
            return p
        p.mode = safe_str(v.get(("checker", "settings", "mode")), SAFE_ID)
        p.arm_name = safe_str(v.get(("checker", "settings", "arm", "name")))
        p.parent_id = safe_str(v.get(("checker", "branched_from")), SAFE_ID)
        p.infra_ids = v.infra_try_ids()
        p.has_retry = v.present(("checker", "retry"))
        if p.has_retry:
            p.retry_started_s = v.get(("checker", "retry", "started_s"))
            p.retry_turns = v.get(("checker", "retry", "turns"))
            if is_count(p.retry_turns) and p.n_turns is not None and 0 <= p.retry_turns < p.n_turns:
                p.prefix_digest = v.digest(p.n_turns - p.retry_turns)
            if p.n_turns is not None:
                p.turns = tuple(tuple(t[name] for name in TURN_FIELDS) for t in v.turns())
        return p
    finally:
        doc = None


TI = {name: i for i, name in enumerate(TURN_FIELDS)}


class Row:
    """One retried answer with a verified attribution: its model, cell and the values of its two series."""

    __slots__ = ("model", "cell", "tokens", "latency_s", "model_call_s", "n_live")

    def __init__(self, model, cell, tokens, latency_s, model_call_s, n_live):
        self.model, self.cell, self.tokens, self.latency_s, self.model_call_s = model, cell, tokens, latency_s, model_call_s
        self.n_live = n_live


CELL = re.compile(r"(fb|dep):([A-Za-z0-9_.+-]{1,60})|gen:any")


def evaluate(part: Part, parents: dict, pinned: set):
    """('row', Row) or ('excluded', reason) for one retried branch record. ``parents``: run id -> (turn count, digest of
    its whole per-turn list) for the plain records of the same OUT."""
    cell = part.cell
    m = CELL.fullmatch(cell) if cell else None
    if m is None:
        return "excluded", "cell_malformed"
    if cell not in pinned:
        return "excluded", "cell_not_pinned"
    if cell == "gen:any":
        consistent = part.mode == "generic"
    else:
        consistent = part.mode == "feedback" and part.arm_name == m.group(2)
    if not consistent:
        return "excluded", "cell_mismatch"
    model = norm_model(part.model)
    if not model:
        return "excluded", "model_missing"
    # retry.turns may be 0: the first live request failed (an infrastructure failure of the retry); such a retried answer
    # has no live tokens and its latency is the time the failed request took
    if not (is_count(part.retry_turns) and is_num(part.retry_started_s) and is_num(part.wall_s)
            and part.n_turns is not None and part.turns is not None and len(part.turns) == part.n_turns):
        return "excluded", "retry_malformed"
    n_prefix = part.n_turns - part.retry_turns
    if n_prefix < 1:
        return "excluded", "count_mismatch"
    turns = part.turns
    if any(turns[i][TI["turn"]] != i + 1 for i in range(part.n_turns)):
        return "excluded", "turn_numbering"
    last_prefix_start = turns[n_prefix - 1][TI["started_s"]]
    if not (is_num(last_prefix_start) and last_prefix_start <= part.retry_started_s + EPS):
        return "excluded", "boundary_mismatch"
    for t in turns[n_prefix:]:
        if not is_num(t[TI["started_s"]]) or t[TI["started_s"]] < part.retry_started_s - EPS:
            return "excluded", "boundary_mismatch"
    parent = parents.get(part.parent_id) if part.parent_id else None
    if parent is None:
        return "excluded", "parent_absent"
    if parent[0] != n_prefix or part.prefix_digest is None or parent[1] != part.prefix_digest:
        return "excluded", "parent_prefix_mismatch"
    live = turns[n_prefix:]
    tokens = None
    if all(is_count(t[TI["input_tokens"]]) and is_count(t[TI["output_tokens"]]) for t in live):
        def total(name):
            return sum(v for v in (t[TI[name]] for t in live) if is_count(v))
        tokens = {"input": total("input_tokens"), "output": total("output_tokens"),
                  "cache_read": total("cache_read_tokens"), "cache_write": total("cache_write_tokens")}
        tokens["io"] = tokens["input"] + tokens["output"]
        tokens["all"] = tokens["io"] + tokens["cache_read"] + tokens["cache_write"]
    latency = part.wall_s - part.retry_started_s
    latency = latency if latency >= -EPS else None
    call_vals = [t[TI["latency_s"]] for t in live]
    model_call = sum(call_vals) if all(is_num(v) and v >= 0 for v in call_vals) else None
    return "row", Row(model, cell, tokens, None if latency is None else max(latency, 0.0), model_call, len(live))


def read_manifest(out_dir: str):
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    try:
        v = ManifestView(doc)
        if v.get(("schema",)) != "pilot-grid-manifest/1":
            raise Bad()
        return set(v.keys(("fingerprint", "phases", "branch", "arms")))
    finally:
        doc = None


def process_out(out_dir: str, tally: dict):
    """(candidates, named): the verified rows of one OUT as (run id, Row), and the run ids that any checker record of the OUT
    names as an earlier try. Claim-digest groups are resolved per OUT (OUTs may share a digest for different attempts);
    the drop of an earlier try named in another OUT, and the run-id dedup across OUTs, are made by the caller over all
    OUTs. ``tally`` collects fixed-name counters (aggregates over all OUTs)."""
    pinned = read_manifest(out_dir)
    rec_dir = os.path.join(out_dir, "records")
    names = sorted(n for n in os.listdir(rec_dir) if RECORD_NAME.fullmatch(n))
    parents = {}
    checker_parts = {}  # claim digest -> [(run id, Part)] of the checker records
    for name in names:
        run_id = name[41:-5]
        try:
            part = load_part(os.path.join(rec_dir, name))
        except Forbidden:
            raise  # a bug in this tool, never a data condition: stop, never count it as an unreadable file
        except Exception:
            tally["unreadable"] = tally.get("unreadable", 0) + 1
            continue
        if not part.has_checker:
            if part.n_turns is not None and part.whole_digest is not None:
                parents[run_id] = (part.n_turns, part.whole_digest)
            continue
        checker_parts.setdefault(name[:40], []).append((run_id, part))
    # One attempt (one claim digest) counts once. The runner writes the failed tries of an attempt to another directory
    # and names them in the final record's runner.infra_try_run_ids; several checker records in records/ under one claim
    # digest are resolved the same way: the one record that no other of the group names as an earlier try. A group with
    # none or several such records is ambiguous and is left out whole (counted in the diagnostics).
    named_all = frozenset().union(*(part.infra_ids for group in checker_parts.values() for _, part in group))
    final = []
    for claim, group in sorted(checker_parts.items()):
        if len(group) == 1:
            final.append(group[0])
            continue
        tally["claim_digest_groups_over_one_record"] = tally.get("claim_digest_groups_over_one_record", 0) + 1
        named = frozenset().union(*(part.infra_ids for _, part in group))
        left = [(run_id, part) for run_id, part in group if run_id not in named]
        if len(left) == 1:
            final.append(left[0])
        else:
            tally["claim_digest_groups_ambiguous_left_out"] = tally.get("claim_digest_groups_ambiguous_left_out", 0) + 1
    cands = []
    for run_id, part in final:
        if not part.has_retry:
            continue
        kind, result = evaluate(part, parents, pinned)
        if kind == "row":
            cands.append((run_id, result))
        else:
            tally["excluded_" + result] = tally.get("excluded_" + result, 0) + 1
    return cands, named_all


def merge_outs(per_out: list, tally: dict) -> list:
    """The rows of all included OUTs. A record that any record of any OUT names as an earlier try of its attempt is dropped
    wherever it sits (the final record of an attempt may be in one OUT and its earlier try, copied, in another); a run id
    counts once. Both steps run over all OUTs after each is read, so the result does not depend on their order."""
    named = frozenset().union(*(n for _, n in per_out)) if per_out else frozenset()
    rows, seen = [], set()
    for cands, _ in per_out:
        for run_id, row in cands:
            if run_id in named:
                tally["dropped_named_as_earlier_try"] = tally.get("dropped_named_as_earlier_try", 0) + 1
            elif run_id in seen:
                tally["duplicate_run_ids"] = tally.get("duplicate_run_ids", 0) + 1
            else:
                seen.add(run_id)
                rows.append(row)
    return rows


# -- aggregation -------------------------------------------------------------------------------------------------

SERIES = (
    ("io_tokens", lambda r: None if r.tokens is None else r.tokens["io"]),
    ("all_tokens", lambda r: None if r.tokens is None else r.tokens["all"]),
    # the same without the retried answers whose first live request failed (0 is a floor for them: that request's own
    # tokens are not recorded)
    ("io_tokens_excluding_zero_live_turns", lambda r: None if r.tokens is None or r.n_live == 0 else r.tokens["io"]),
    ("all_tokens_excluding_zero_live_turns", lambda r: None if r.tokens is None or r.n_live == 0 else r.tokens["all"]),
    ("input_tokens", lambda r: None if r.tokens is None else r.tokens["input"]),
    ("output_tokens", lambda r: None if r.tokens is None else r.tokens["output"]),
    ("cache_read_tokens", lambda r: None if r.tokens is None else r.tokens["cache_read"]),
    ("cache_write_tokens", lambda r: None if r.tokens is None else r.tokens["cache_write"]),
    ("latency_s", lambda r: r.latency_s),
    ("model_call_s", lambda r: r.model_call_s),
)


def series_block(rows: list) -> dict:
    out = {"n_retried": len(rows)}
    for name, get in SERIES:
        out[name] = summary([v for v in (get(r) for r in rows) if v is not None])
    return out


def aggregate_cells(rows: list) -> dict:
    """{cell: {"all": block, "models": {model: block}}}, plus the same for the sources (fb and dep merged by arm name)."""
    def group(key_of):
        cells = {}
        for r in rows:
            for k in key_of(r):
                cells.setdefault(k, []).append(r)
        return {k: {"all": series_block(v), "models": {m: series_block([r for r in v if r.model == m])
                                                      for m in sorted({r.model for r in v})}}
                for k, v in sorted(cells.items())}
    by_cell = group(lambda r: [r.cell])
    by_source = group(lambda r: ["generic" if r.cell == "gen:any" else r.cell.split(":", 1)[1]])
    return {"by_cell": by_cell, "by_source": by_source}


# -- ledger ------------------------------------------------------------------------------------------------------

LEDGER_SOURCE = re.compile(r"[A-Za-z0-9_.+-]{1,60}|generic_retry/[A-Za-z0-9_.+-]{1,60}")


def process_ledger(path: str, graph: str, expect_sha256: str | None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    if expect_sha256 is not None and sha != expect_sha256:
        raise Bad()
    values: dict = {}
    tally = {"rows_other_graph": 0, "rows_bank": 0, "rows_variant_source": 0, "rows_unknown_use": 0,
             "evidence_not_recorded": 0, "check_s_not_none": 0, "rows_read_for_graph": 0}
    check_vals = []
    with open(path, encoding="utf-8") as fh:
        header = LedgerHeaderView(json.loads(fh.readline()))
        schema = header.get(("schema",))
        split = header.get(("split",))
        if schema not in ("checker-ledger/2", "checker-ledger/1"):
            raise Bad()
        for line in fh:
            if not line.strip():
                continue
            row = LedgerView(json.loads(line))
            if row.get(("graph",)) != graph:
                tally["rows_other_graph"] += 1
                continue
            model_raw, cohort, source, use = (row.get(("model",)), row.get(("cohort",)), row.get(("source",)),
                                              row.get(("use",)))
            if cohort == "bank" or model_raw == "bank":
                tally["rows_bank"] += 1
                continue
            if not (isinstance(source, str) and LEDGER_SOURCE.fullmatch(source)):
                tally["rows_variant_source"] += 1  # a switch-off variant "<arm>/<variant>" or a source of no safe name
                continue
            if use not in ("replay", "checker", "both"):
                tally["rows_unknown_use"] += 1
                continue
            tally["rows_read_for_graph"] += 1
            check_s = row.get(("check_s",))
            if check_s is not None:
                tally["check_s_not_none"] += 1
                if is_num(check_s):
                    check_vals.append(check_s)
            ev = row.get(("evidence_triples",))
            if not is_count(ev):
                tally["evidence_not_recorded"] += 1
                continue
            model = norm_model(model_raw) or "unknown-model"
            values.setdefault(("replay" if use == "replay" else "online", source), {}).setdefault(model, []).append(ev)
    if tally["rows_read_for_graph"] == 0:
        raise Bad()  # no row of the graph: the graph name or the ledger is wrong
    result = {"replay": {}, "online": {}}
    for (kind, source), per_model in sorted(values.items()):
        every = [v for vals in per_model.values() for v in vals]
        result[kind][source] = {"all": summary(every), "models": {m: summary(vals) for m, vals in sorted(per_model.items())}}
    if check_vals:
        check = {"check_s": summary(check_vals), "note": "check_s is recorded for some rows"}
    else:
        check = {"check_s": NOT_RECORDED}
    return {"sha256": sha, "schema": schema, "split": split, "graph": graph, "evidence_triples": result,
            "tally": tally, **check}


# -- draws -------------------------------------------------------------------------------------------------------


def parse_utc(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


DRAW_FINAL = re.compile(r"draw-\d+\.json")
DRAW_ATTEMPT = re.compile(r"draw-\d+\.attempt-\d+\.json")  # an earlier, failed or retired attempt of a draw (paid too)


def draw_files(d: str, pattern=DRAW_FINAL):
    return sorted(n for n in os.listdir(d) if pattern.fullmatch(n))


def _count_or_none(x):
    return x if is_count(x) else None


def _num_or_none(x):
    return x if is_num(x) else None


def read_draw(path: str, kind: str) -> dict:
    """One draw record's allowed values, each type-checked (a value of another type reads as None)."""
    with open(path, encoding="utf-8") as fh:
        v = DrawView(json.load(fh))
    usage = {k: _count_or_none(v.get(("usage", k))) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                              "cache_creation_input_tokens")}
    usage["cache_creation_5m_tokens"] = _count_or_none(v.get(("usage", "cache_creation", "ephemeral_5m_input_tokens")))
    usage["cache_creation_1h_tokens"] = _count_or_none(v.get(("usage", "cache_creation", "ephemeral_1h_input_tokens")))
    hit = v.get(("cache_hit",))
    return {
        "kind": kind, "draw": _count_or_none(v.get(("draw",))), "attempt": _count_or_none(v.get(("attempt",))),
        "status_ok": v.get(("status",)) == "ok", "model_returned": safe_str(v.get(("model_returned",))),
        "cache_hit": hit if isinstance(hit, bool) else None,
        "usage": usage, "cost_usd": _num_or_none(v.get(("cost_usd",))),
        "prices_usd_per_mtok": {k: _num_or_none(v.get(("prices_usd_per_mtok", k)))
                                for k in ("input", "output", "cache_read", "cache_write_1h", "batch_discount")},
    }


def process_draws(label: str, d: str) -> dict:
    draws = [read_draw(os.path.join(d, n), "final") for n in draw_files(d)]
    earlier = [read_draw(os.path.join(d, n), "earlier_attempt") for n in draw_files(d, DRAW_ATTEMPT)]
    out = {"label": label, "draws": draws, "earlier_attempts": earlier,
           "total_cost_usd": round(sum(x["cost_usd"] for x in draws if is_num(x["cost_usd"])), 4) if draws else None,
           "total_cost_usd_with_earlier_attempts": round(sum(x["cost_usd"] for x in draws + earlier if is_num(x["cost_usd"])), 4)
           if draws else None}
    path = os.path.join(d, "batches.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            bv = BatchesView(json.load(fh))
        batches = []
        for b in bv.batches():
            t0, t1 = parse_utc(b["submitted_utc"]), parse_utc(b["ended_utc"])
            batches.append({"group": b["group"], "draws": b["draws"], "collected": b["collected"],
                            "wall_s": None if t0 is None or t1 is None else (t1 - t0).total_seconds()})
        out["batches"] = batches
        out["input_tokens_counted_before_submission"] = _count_or_none(bv.get(("count", "input_tokens")))
    sup = os.path.join(d, "superseded")
    if os.path.isdir(sup):
        paid = 0
        for sub in sorted(os.listdir(sup)):
            sd = os.path.join(sup, sub)
            if os.path.isdir(sd):
                paid += len(draw_files(sd)) + len(draw_files(sd, DRAW_ATTEMPT))
        out["superseded_draw_records"] = paid
    return out


# -- text table --------------------------------------------------------------------------------------------------


def fmt(x):
    if isinstance(x, float):
        return f"{x:.1f}" if abs(x) >= 100 else f"{x:.2f}"
    return str(x)


def label_of(key: str) -> str:
    kind, _, name = key.rpartition(":")
    shown = "census t5" if name == "census-t5" else ("draft " + name.split("-", 1)[1] if name.startswith("generated-") else name)
    if key == "gen:any":
        return "gen generic"
    return f"{kind} {shown}" if kind else shown


def text_table(result: dict) -> str:
    lines = []
    ret = result["retry"]
    lines.append("RETRY COST per retried answer (live retry turns only; medians and p90; n = retried answers)")
    lines.append("tokens: io = input + output as recorded (cache excluded); all = io + cache read + cache write. "
                 "latency_s = wall_s - checker.retry.started_s.")
    head = f"{'cell':<22}{'model':<42}{'n':>5}{'io med':>10}{'io p90':>10}{'all med':>11}{'all p90':>11}{'lat med':>9}{'lat p90':>9}"
    for title, block in (("BY CELL (fb = feedback on natural parents; gen = generic; dep = own-source prompted parents)",
                          ret["by_cell"]), ("BY SOURCE (fb and dep merged)", ret["by_source"])):
        lines += ["", title, head]
        for key, cellblock in block.items():
            rows = [("all", cellblock["all"])] + list(cellblock["models"].items())
            for model, b in rows:
                lines.append(f"{label_of(key):<22}{model:<42}{b['n_retried']:>5}{fmt(b['io_tokens']['median']):>10}"
                             f"{fmt(b['io_tokens']['p90']):>10}{fmt(b['all_tokens']['median']):>11}"
                             f"{fmt(b['all_tokens']['p90']):>11}{fmt(b['latency_s']['median']):>9}{fmt(b['latency_s']['p90']):>9}")
    led = result["ledger"]
    lines += ["", "EVIDENCE TRIPLES per checked answer (ledger rows; n, median, p90)"]
    for kind in ("replay", "online"):
        lines.append(f"  rows of use = {'replay' if kind == 'replay' else 'checker/both (online, separate from replay)'}")
        for source, b in led["evidence_triples"][kind].items():
            lines.append(f"    {source:<26}{'all':<42}n={b['all']['n']:<8}median={fmt(b['all']['median'])}  p90={fmt(b['all']['p90'])}")
            for model, s in b["models"].items():
                lines.append(f"    {'':<26}{model:<42}n={s['n']:<8}median={fmt(s['median'])}  p90={fmt(s['p90'])}")
    lines += ["", f"check time per checked answer: {led['check_s']}   (ledger rows with check_s not null: "
                  f"{led['tally']['check_s_not_none']})",
              f"census build time: {result['setup']['census_build_s']}"]
    lines += ["", "SETUP: generation usage of each draft (draw records)"]
    for d in result["setup"]["draws"]:
        lines.append(f"  {d['label']}: total cost_usd {d['total_cost_usd']} (with earlier attempts "
                     f"{d['total_cost_usd_with_earlier_attempts']})")
        for x in d["draws"]:
            u = x["usage"]
            lines.append(f"    draw {x['draw']} (attempt {x['attempt']}, ok={x['status_ok']}): input {u['input_tokens']}, "
                         f"output {u['output_tokens']}, cache read {u['cache_read_input_tokens']}, cache write "
                         f"{u['cache_creation_input_tokens']}, USD {x['cost_usd']}")
        for x in d["earlier_attempts"]:
            u = x["usage"]
            lines.append(f"    earlier attempt of draw {x['draw']} (attempt {x['attempt']}, ok={x['status_ok']}): input "
                         f"{u['input_tokens']}, output {u['output_tokens']}, USD {x['cost_usd']}")
        for b in d.get("batches", []):
            lines.append(f"    batch group {b['group']} draws {b['draws']}: submit to end {b['wall_s']} s")
        if "superseded_draw_records" in d:
            lines.append(f"    superseded draw records: {d['superseded_draw_records']}")
    return "\n".join(lines) + "\n"


# -- command line ------------------------------------------------------------------------------------------------


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        print("error: bad command line (no detail is printed)", file=sys.stderr)
        sys.exit(2)


def labelled(values: list) -> list:
    out = []
    for v in values or []:
        label, sep, path = v.partition("=")
        if not sep or not SAFE_LABEL.fullmatch(label) or not path:
            raise Bad()
        out.append((label, path))
    return out


def self_sha256() -> str:
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


DEFINITIONS = {
    "retried_answer": "a branch record whose checker.retry is present (the checker flagged the initial answer and a live retry ran, or was started: retry.turns may be 0 when the first live request failed, such an answer has no live tokens and its latency is the time that request took; diagnostics count them); the answer is counted once, in the cell of the record. A claim digest names one attempt: where several checker records in records/ share one, the record no other names in runner.infra_try_run_ids is the attempt's, and a group that does not resolve to exactly one is left out and counted in the diagnostics",
    "model_keys": "the model id of the record or ledger row, lower-cased with every run of other characters as '-'",
    "live_retry_turns": "the per-turn entries after the replayed prefix; the prefix is the parent record's whole per_turn list (checked equal by hash, parent found by checker.branched_from in the same OUT); also required: the count equals checker.retry.turns, turn numbers run 1..N, and the prefix turns start no later than checker.retry.started_s while every live turn starts no earlier",
    "zero_live_turns": "a retried answer whose first live request failed has retry.turns 0: its tokens are 0 here, which is a floor (the failed request's own tokens are not recorded); the *_excluding_zero_live_turns series leave these answers out",
    "io_tokens": "sum over the live retry turns of input_tokens + output_tokens as recorded. input_tokens excludes cached prompt tokens only where the provider reports them (Anthropic always; an OpenAI-compatible provider when usage.prompt_tokens_details.cached_tokens is reported, harness/agent/providers.py normalise_*_usage); where it does not, input_tokens is the whole prompt. io_tokens is therefore NOT comparable across providers that differ in cache reporting; all_tokens is",
    "all_tokens": "io_tokens + the live turns' cache_read_tokens + cache_write_tokens (not reported = 0): input + cache read + cache write is the whole prompt on every provider, so this total is comparable across providers; the tokens of provider requests that failed or were retried inside a step are not in per_turn and not counted",
    "latency_s": "wall_s - checker.retry.started_s of the branch record: on a branch record both come from the prefix clock, retry.started_s is the first live clock read after the replayed initial phase and the check, so the interval is the live retry phase only (model calls, tool calls, store waits, query-gate waits and the runner's start-gate wait before the first live request included; model_call_s is the model requests alone). A negative value (clock inconsistency) is left out of the latency series and counted in the diagnostics",
    "model_call_s": "sum of the live retry turns' latency_s (the model requests' own latency, no tool or store time)",
    "median_p90": "linear interpolation between order statistics (type 7, statistics.quantiles method inclusive); n = 0 is 'not estimable'; n = 1 gives that value",
    "evidence_triples": "ledger field evidence_triples, one value per ledger row (an initial output checked by one source); replay rows (use = replay) cover every checked output of every source; online rows (use = checker or both) are the arms actually run and are never pooled with replay rows; union = sum of its sources' budgets; generic_retry/<arm> = the curated checks",
    "check_s": "CheckedOutput.check_s is never filled: reported as 'not recorded'",
    "census_build_s": "not recorded",
}


def main() -> int:
    ap = QuietParser()
    ap.add_argument("--out", action="append", default=[], help="LABEL=OUTDIR of a run OUT")
    ap.add_argument("--complete", action="append", default=[],
                    help="LABEL=FILE: the OUT is read only if this file exists (its completion marker or audit)")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--ledger-sha256", default=None)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--draws", action="append", default=[], help="LABEL=DIR of a draws directory")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-txt", required=True)
    args, extra = ap.parse_known_args()
    if extra:
        ap.error("")
    outs = labelled(args.out)
    complete = dict(labelled(args.complete))
    tally: dict = {}
    per_out = []
    included, excluded = [], []
    for label, path in outs:
        marker = complete.get(label)
        if marker is None or not os.path.isfile(marker):
            excluded.append({"label": label, "reason": "phase not complete: excluded"})
            continue
        st = os.stat(marker)
        per_out.append(process_out(path, tally))
        included.append({"label": label, "basis_file": os.path.basename(marker),
                         "basis_mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    rows = merge_outs(per_out, tally)
    ledger = process_ledger(args.ledger, args.graph, args.ledger_sha256)
    draws = [process_draws(label, d) for label, d in labelled(args.draws)]
    result = {
        "schema": SCHEMA,
        "tool_sha256": self_sha256(),
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "definitions": DEFINITIONS,
        "outs_included": included,
        "outs_excluded": excluded,
        "retry": {**aggregate_cells(rows),
                  "diagnostics": dict(sorted({**tally, "retried_with_zero_live_turns": sum(1 for r in rows if r.n_live == 0),
                                              "retried_with_negative_latency_left_out_of_latency":
                                              sum(1 for r in rows if r.latency_s is None)}.items()))},
        "ledger": ledger,
        "setup": {"draws": draws, "census_build_s": NOT_RECORDED},
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(args.out_json, flags, 0o600), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
        fh.write("\n")
    with os.fdopen(os.open(args.out_txt, flags, 0o600), "w", encoding="utf-8") as fh:
        fh.write(text_table(result))
    print(json.dumps({"ok": True, "tool_sha256": result["tool_sha256"]}))
    return 0


def guarded() -> int:
    """main(), with every uncaught exception turned into one fixed line: no traceback, so no location, name or record
    text can reach the console."""
    try:
        return main()
    except SystemExit:
        raise
    except BaseException:
        print("error: the figures could not be computed (no detail is printed)")
        return 1


if __name__ == "__main__":
    sys.exit(guarded())
