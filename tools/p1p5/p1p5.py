#!/usr/bin/env python3
"""P1-P5 on the dev checker replay (the design log; preregistration section 6): counts and rates only.

    python3 -I p1p5.py --config CONFIG.json --out OUTDIR

Reads dev inputs only: replay matrices (checker-replay-matrix/1 rows), a dev bank replay matrix with its plan and
run record, and the interim judge calibration (selection, truth, calls, tables). Every matrix row must carry the split
named in the config ("dev"); a row of any other split stops the run (a test-split file is never accepted).

Writes OUTDIR/p1p5-dev.json and OUTDIR/p1p5-dev.txt. It prints no verdict labels: the quantities each prediction
names (rates, differences, ratios) are computed, and nothing says whether a prediction holds. Exit 0 when every
input check and every reconciliation matches and the frozen-source comparison is equal or not checked (its tree is absent: NOT CHECKED, not a failure), 1 when one does not (outputs are still written), 2 on bad input.

Definitions (also in the package README):
- Only the "primary" variant is read. A row with output false has no status and enters no denominator.
- F = flagged. P = substantive_pass + vacuous_pass. U = unknown (in the denominator, not a flag).
- W = wrong outputs (output true, correct false); C = correct outputs.
- recall = W_F / (W_F + W_P + W_U); false-flag rate = C_F / (C_F + C_P + C_U): ratios of pooled counts
  (preregistration section 3, "Weighting"). No intervals: the preregistration prescribes none for P1-P5.
- The union's status is the union arm's own matrix row; it is never recomputed from its members.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

TOOL_VERSION = "p1p5/1"
NOT_CHECKED = "NOT CHECKED (sources not released)"   # the frozen-source comparison when its tree is absent: reported, never counted as a failure
STATUS_LETTER = {"flagged": "F", "substantive_pass": "P", "vacuous_pass": "P", "unknown": "U"}
PRIMARY = "primary"
NOT_REPLAYED = "not replayed"
NOT_ESTIMABLE = "not estimable"
# the sources the registered text names; one that a replay lacks prints NOT_REPLAYED (never 0)
SOURCE_UNIVERSE = ("curated", "curated+doc", "census-t2", "census-t5", "census-t20", "generated-1", "generated-2",
                   "generated-3", "generated-dc-1", "generated-dc-2", "generated-dc-3")
# P2 (preregistration section 6): the meaning-heavy UniProt templates, and Rhea's type_members
P2_FOCUS = {"uniprot-sp-2026_03": ("aggregation", "numeric_filter"), "rhea-142": ("type_members",)}
AGENT_SHORT = {"claude-sonnet-5-5": "Sonnet", "Qwen/Qwen3.8-27B": "Qwen", "claude-haiku-5-5": "Haiku"}
QWEN = "Qwen/Qwen3.8-27B"
GRAPH_SHORT = {"uniprot-sp-2026_03": "uniprot", "rhea-142": "rhea"}
ALL_GRAPHS = "all"


class InputError(Exception):
    """An input is malformed or breaks a stated rule (exit 2)."""


# -- files ---------------------------------------------------------------------------------------------------------

def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def read_jsonl(path: Path) -> Iterator[dict]:
    with open_text(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_json(path: Path) -> Any:
    with open_text(path) as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- ratios ---------------------------------------------------------------------------------------------------------

def ratio(num: int, den: int) -> dict:
    return {"num": num, "den": den, "rate": (num / den) if den else None}


def fmt_ratio(r: Any) -> str:
    if r is None:
        return "-"
    if isinstance(r, str):
        return r
    if r["rate"] is None:
        return f"{r['num']}/{r['den']} = {NOT_ESTIMABLE}"
    return f"{r['num']}/{r['den']} = {r['rate']:.3f}"


def diff(a: dict | None, b: dict | None) -> float | None:
    """a.rate - b.rate when both are estimable."""
    if not isinstance(a, dict) or not isinstance(b, dict) or a["rate"] is None or b["rate"] is None:
        return None
    return a["rate"] - b["rate"]


# -- replay matrices -------------------------------------------------------------------------------------------------

@dataclass
class Unit:
    graph: str
    run_id: str
    question_id: str
    template: str | None
    model: str
    cohort: str | None
    output: bool
    correct: bool | None
    arms: dict[str, str | None] = field(default_factory=dict)  # arm -> F/P/U (None for a no-output row)
    per_source: dict[str, dict[str, str]] = field(default_factory=dict)  # arm -> {member: status} (union rows)


def build_units(rows: Iterable[dict], expected_split: str) -> dict[tuple[str, str], Unit]:
    """One Unit per (graph, run_id) from the primary-variant rows. Every row's split must be expected_split."""
    units: dict[tuple[str, str], Unit] = {}
    for r in rows:
        if r.get("split") != expected_split:
            raise InputError(f"a row of split {r.get('split')!r} (expected {expected_split!r}): run {r.get('run_id')}")
        if r.get("variant") != PRIMARY:
            continue
        key = (r["graph"], r["run_id"])
        out = bool(r["output"])
        correct = r.get("correct")
        status = r.get("status")
        if out:
            if status not in STATUS_LETTER:
                raise InputError(f"{key}: output true with status {status!r} (arm {r['arm']})")
            if correct is None:
                raise InputError(f"{key}: output true with no correctness")
        else:
            if status is not None:
                raise InputError(f"{key}: no output but status {status!r} (arm {r['arm']})")
        u = units.get(key)
        if u is None:
            u = Unit(r["graph"], r["run_id"], r["question_id"], r.get("template"), r["model"], r.get("cohort"), out,
                     correct if out else None)
            units[key] = u
        elif (u.question_id, u.template, u.model, u.output, u.correct) != (
                r["question_id"], r.get("template"), r["model"], out, correct if out else None):
            raise InputError(f"{key}: arms disagree on the output's attributes (arm {r['arm']})")
        if r["arm"] in u.arms:
            raise InputError(f"{key}: arm {r['arm']} appears twice")
        u.arms[r["arm"]] = STATUS_LETTER[status] if out else None
        if out and r.get("per_source"):
            u.per_source[r["arm"]] = dict(r["per_source"])
    by_graph: dict[str, set] = {}
    for u in units.values():
        by_graph.setdefault(u.graph, set()).add(frozenset(u.arms))
    for g, sets in by_graph.items():
        if len(sets) != 1:
            raise InputError(f"graph {g}: outputs do not all carry the same arms")
    return units


def project(rows: Iterable[dict], fields: tuple[str, ...]) -> Iterator[dict]:
    for r in rows:
        yield {k: r.get(k) for k in fields}


MATRIX_FIELDS = ("run_id", "graph", "question_id", "template", "split", "model", "cohort", "variant", "arm", "output",
                 "correct", "status", "per_source")


def arms_of(units: Iterable[Unit]) -> list[str]:
    for u in units:
        return list(u.arms)
    return []


def outputs(units: Iterable[Unit]) -> list[Unit]:
    return [u for u in units if u.output]


def cells(units: Iterable[Unit], arm: str) -> dict[str, int]:
    """The six cells of one arm over the units with an output."""
    c = {k: 0 for k in ("C_F", "C_P", "C_U", "W_F", "W_P", "W_U")}
    for u in units:
        if not u.output:
            continue
        c[("C_" if u.correct else "W_") + u.arms[arm]] += 1
    return c


def rates_block(units: list[Unit], arm: str) -> dict:
    c = cells(units, arm)
    w = c["W_F"] + c["W_P"] + c["W_U"]
    cc = c["C_F"] + c["C_P"] + c["C_U"]
    return {**c, "W": w, "C": cc, "recall": ratio(c["W_F"], w), "false_flag": ratio(c["C_F"], cc)}


def wrong(units: Iterable[Unit]) -> list[Unit]:
    return [u for u in units if u.output and not u.correct]


def questions(units: Iterable[Unit]) -> int:
    return len({u.question_id for u in units})


# -- the dev bank ----------------------------------------------------------------------------------------------------

BANK_FIELDS = ("run_id", "graph", "split", "variant", "arm", "output", "correct", "status", "result_rows", "cohort")


def bank_strata(result_rows: Any) -> str:
    if isinstance(result_rows, bool) or not isinstance(result_rows, int):
        return "no_row_count"
    return "empty" if result_rows == 0 else "returns_rows"


def bank_counts(rows: Iterable[dict], expected_split: str) -> dict[str, dict[str, dict[str, int]]]:
    """arm -> stratum ('all', 'returns_rows', 'empty', 'no_row_count') -> six cells. W = correct false (the
    answer-changing items); every row of the bank is an output."""
    seen: set[tuple[str, str]] = set()
    out: dict[str, dict[str, dict[str, int]]] = {}
    items_by_arm: dict[str, set] = {}
    for r in rows:
        if r.get("split") != expected_split:
            raise InputError(f"a bank row of split {r.get('split')!r} (expected {expected_split!r})")
        if r.get("variant") != PRIMARY:
            continue
        if r.get("cohort") != "bank":
            raise InputError(f"a bank row of cohort {r.get('cohort')!r}")
        if r.get("status") not in STATUS_LETTER or r.get("correct") is None:
            raise InputError(f"bank item {r.get('run_id')}: status {r.get('status')!r}, correct {r.get('correct')!r}")
        key = (r["arm"], r["run_id"])
        if key in seen:
            raise InputError(f"bank item {r['run_id']}: arm {r['arm']} appears twice")
        seen.add(key)
        items_by_arm.setdefault(r["arm"], set()).add(r["run_id"])
        letter = STATUS_LETTER[r["status"]]
        cell = ("C_" if r["correct"] else "W_") + letter
        for stratum in ("all", bank_strata(r.get("result_rows"))):
            c = out.setdefault(r["arm"], {}).setdefault(stratum, {k: 0 for k in
                                                                  ("C_F", "C_P", "C_U", "W_F", "W_P", "W_U")})
            c[cell] += 1
    if len({frozenset(s) for s in items_by_arm.values()}) > 1:
        raise InputError("the bank arms do not all hold the same items")
    return out


def cells_ratios(c: dict[str, int]) -> dict:
    w = c["W_F"] + c["W_P"] + c["W_U"]
    cc = c["C_F"] + c["C_P"] + c["C_U"]
    return {**c, "W": w, "C": cc, "recall": ratio(c["W_F"], w), "false_flag": ratio(c["C_F"], cc)}


# -- the judge -------------------------------------------------------------------------------------------------------

@dataclass
class JudgeData:
    truth: dict[str, str]
    selection: dict[str, dict]
    verdict: dict[tuple[str, str], str]  # (key, judge model) -> reject / accept / unparsed / error
    raw: dict[tuple[str, str], str]  # (key, judge model) -> the raw verdict string, "<unparsed>" or "<failed>"


def load_judge(truth_path: Path, selection_path: Path, calls_path: Path) -> JudgeData:
    truth = {}
    for r in read_jsonl(truth_path):
        if r["key"] in truth:
            raise InputError(f"truth: {r['key']} twice")
        truth[r["key"]] = r["truth"]
    selection = {}
    for o in read_json(selection_path)["outputs"]:
        if o["key"] in selection:
            raise InputError(f"selection: {o['key']} twice")
        selection[o["key"]] = o
    verdict: dict[tuple[str, str], str] = {}
    raw_of: dict[tuple[str, str], str] = {}
    for c in read_jsonl(calls_path):
        pair = (c["item"]["key"], c["judge_model"])
        if pair in verdict:
            raise InputError(f"two judge calls for {pair}")
        j = c.get("judge")
        raw = "<failed>" if j is None else (j.get("verdict") or "<unparsed>")
        if raw not in ("accept", "reject", "<failed>", "<unparsed>"):
            raise InputError(f"judge call {pair}: verdict {raw!r} is neither accept nor reject")
        raw_of[pair] = raw
        verdict[pair] = "error" if j is None else (j.get("verdict") or "unparsed")
    return JudgeData(truth, selection, verdict, raw_of)


def judge_cell(keys: list[str], judge: str, jd: JudgeData) -> dict:
    """Reject recall of one judge on the given (wrong) outputs; complete only when it judged all of them."""
    judged = [k for k in keys if (k, judge) in jd.verdict]
    n = len(judged)
    rej = sum(1 for k in judged if jd.verdict[(k, judge)] == "reject")
    unp = sum(1 for k in judged if jd.verdict[(k, judge)] in ("unparsed", "error"))
    base = {"of": len(keys), "judged": n, "reject": rej, "unparsed_or_failed": unp}
    if n == 0:
        return {**base, "coverage": "not run", "recall": None}
    if n == len(keys):
        return {**base, "coverage": "complete", "recall": ratio(rej, n)}
    return {**base, "coverage": f"partial {n} of {len(keys)}", "recall": None,
            "recall_on_judged_subset": ratio(rej, n)}


def judge_role(agent: str, judge: str) -> str:
    cross = {"claude-sonnet-5-5": QWEN, QWEN: "claude-sonnet-5-5"}
    if judge == agent:
        return "self"
    if cross.get(agent) == judge:
        return "cross"
    return "haiku" if judge == "claude-haiku-5-5" else "other"


# -- the analysis ----------------------------------------------------------------------------------------------------

def source_rows(units: list[Unit], arms: list[str]) -> dict[str, Any]:
    return {a: rates_block(units, a) for a in arms}


def pool_arms(units_by_graph: dict[str, list[Unit]]) -> list[str]:
    sets = [set(arms_of(us)) for us in units_by_graph.values() if us]
    common = set.intersection(*sets) if sets else set()
    first = arms_of(next(iter(units_by_graph.values())))
    return [a for a in first if a in common]


def graph_groups(units: list[Unit]) -> dict[str, list[Unit]]:
    g: dict[str, list[Unit]] = {}
    for u in units:
        g.setdefault(u.graph, []).append(u)
    out = dict(sorted(g.items()))
    if len(out) > 1:
        out[ALL_GRAPHS] = list(units)
    return out


def arms_for(group_name: str, group: list[Unit], by_graph: dict[str, list[Unit]]) -> list[str]:
    if group_name == ALL_GRAPHS:
        return pool_arms({g: us for g, us in by_graph.items() if g != ALL_GRAPHS})
    return arms_of(group)


def not_replayed(arms: list[str], union_names: list[str]) -> list[str]:
    return [s for s in SOURCE_UNIVERSE if s not in arms and s not in union_names]


def nr(pop: "Population", arms: list[str]) -> dict[str, str]:
    """The sources of the registered universe that this replay lacks, each as the literal 'not replayed'."""
    union_names = [a for gg in pop.unions.values() for a in gg]
    return {s: NOT_REPLAYED for s in not_replayed(arms, union_names)}


def rate_ratio(real: float | None, bank_rate: float | None) -> Any:
    """real recall / bank recall; 'not estimable' when either is not estimable or the bank recall is 0."""
    if real is None or not bank_rate:
        return NOT_ESTIMABLE
    return real / bank_rate


@dataclass
class Population:
    name: str
    title: str
    registered: bool
    units: dict[tuple[str, str], Unit]
    digests: dict[str, dict[str, str]]  # graph -> arm -> source digest
    unions: dict[str, dict[str, list[str]]]  # graph -> union arm -> members
    result: dict[str, str]  # run_id -> final checked query result (rows / empty / refused), when known
    judge: JudgeData | None
    judges: list[str]
    accounting: dict[str, dict]  # graph -> accounting.json content (for the reconciliation)
    design_log: dict | None = None  # design_log_summary.json content (for the reconciliation)
    excluded: dict[str, int] = field(default_factory=dict)  # model -> attempts left out of the analysed set
    kind: str = "registered"  # registered / sensitivity / supplementary
    dropped: list = field(default_factory=list)  # Units removed by drop_runs (sensitivity populations)
    judge_full: "JudgeData | None" = None  # the judge data before the dropped outputs were removed
    result_full: dict = field(default_factory=dict)  # run_id -> checked-query result, before the removal


RESULT_CLASSES = ("rows", "empty", "refused")


def p1(pop: Population, bank: dict | None) -> dict:
    """Each source's recall on real wrong outputs beside its bank recall on non-empty items (same source digest)."""
    out: dict[str, Any] = {}
    by_graph = {g: us for g, us in graph_groups(list(pop.units.values())).items() if g != ALL_GRAPHS}
    for g, us in by_graph.items():
        w_all = wrong(us)
        by_result: dict[str, list[Unit]] | None = None
        if pop.result:
            missing = [u.run_id for u in w_all if u.run_id not in pop.result]
            if missing:
                raise InputError(f"{len(missing)} wrong outputs of {g} have no checked-query result in the judge "
                                 f"selection (first: {missing[0]})")
            bad = sorted({pop.result[u.run_id] for u in w_all} - set(RESULT_CLASSES))
            if bad:
                raise InputError(f"unknown checked-query result classes {bad}")
            by_result = {c: [u for u in w_all if pop.result[u.run_id] == c] for c in RESULT_CLASSES}
        w_rows = None if by_result is None else by_result["rows"]
        rows = []
        for arm in arms_of(us):
            real_all = rates_block(us, arm)["recall"]
            entry: dict[str, Any] = {"source": arm, "real_wrong_all": real_all,
                                     "real_wrong_by_agent": {m: rates_block([u for u in us if u.model == m], arm)["recall"]
                                                             for m in sorted({u.model for u in us})}}
            if w_rows is not None:
                entry["real_wrong_result_rows"] = ratio(sum(1 for u in w_rows if u.arms[arm] == "F"), len(w_rows))
            else:
                entry["real_wrong_result_rows"] = "n/a (no checked-query result)"
            entry["bank"] = bank_pair(pop, bank, g, arm)
            b = entry["bank"]
            if isinstance(b, dict):
                bank_rate = b["returns_rows"]["recall"]["rate"]
                entry["ratio_real_all_over_bank_returns_rows"] = rate_ratio(real_all["rate"], bank_rate)
                rr = entry["real_wrong_result_rows"]
                entry["ratio_real_result_rows_over_bank_returns_rows"] = (
                    None if not isinstance(rr, dict) else rate_ratio(rr["rate"], bank_rate))
            else:
                entry["ratio_real_all_over_bank_returns_rows"] = None
                entry["ratio_real_result_rows_over_bank_returns_rows"] = None
            rows.append(entry)
        out[g] = {"wrong_outputs": len(w_all), "wrong_questions": questions(w_all),
                  "wrong_with_result_rows": None if w_rows is None else len(w_rows),
                  "wrong_by_result": None if by_result is None else {c: len(v) for c, v in by_result.items()},
                  "wrong_by_result_by_agent": None if by_result is None else {
                      m: {c: sum(1 for u in v if u.model == m) for c, v in by_result.items()}
                      for m in sorted({u.model for u in w_all})},
                  "sources": rows, "not_replayed": nr(pop, arms_of(us))}
    return out


def bank_pair(pop: Population, bank: dict | None, graph: str, arm: str) -> Any:
    """The bank arm with the same source digest on the same graph; otherwise NOT_REPLAYED (never another source)."""
    if bank is None or bank["graph"] != graph:
        return NOT_REPLAYED + " (no dev bank replay for this graph)"
    dig = pop.digests.get(graph, {}).get(arm)
    if dig is None:
        return NOT_REPLAYED + " (no source digest for this arm)"
    match = sorted(a for a, d in bank["digests"].items() if d == dig and a in bank["counts"])
    if not match:
        return NOT_REPLAYED + " (no bank arm with this source digest)"
    if len(match) > 1:
        raise InputError(f"two bank arms {match} share the digest of {arm}")
    a = match[0]
    c = bank["counts"][a]
    return {"bank_arm": a, "source_digest": dig,
            "returns_rows": cells_ratios(c.get("returns_rows", {k: 0 for k in
                                                                 ("C_F", "C_P", "C_U", "W_F", "W_P", "W_U")})),
            "all_items": cells_ratios(c.get("all", {k: 0 for k in ("C_F", "C_P", "C_U", "W_F", "W_P", "W_U")}))}


def p2(pop: Population) -> dict:
    out: dict[str, Any] = {}
    by_graph = {g: us for g, us in graph_groups(list(pop.units.values())).items() if g != ALL_GRAPHS}
    for g, us in by_graph.items():
        arms = arms_of(us)
        tmpl = sorted({u.template for u in us if u.template})
        table = {}
        for t in tmpl:
            sel = [u for u in us if u.template == t]
            table[t] = {"wrong_outputs": len(wrong(sel)), "wrong_questions": questions(wrong(sel)),
                        "sources": {a: _w_only(rates_block(sel, a)) for a in arms}}
        focus_t = P2_FOCUS.get(g, ())
        focus = {}
        if focus_t:
            sel = [u for u in us if u.template in focus_t]
            focus = {"templates": list(focus_t), "templates_present": [t for t in focus_t if t in table],
                     "wrong_outputs": len(wrong(sel)), "wrong_questions": questions(wrong(sel)),
                     "sources": {a: _w_only(rates_block(sel, a)) for a in arms}}
        out[g] = {"focus": focus, "templates": table, "not_replayed": nr(pop, arms)}
    return out


def _w_only(b: dict) -> dict:
    return {k: b[k] for k in ("W", "W_F", "W_P", "W_U", "recall")}


def p3(pop: Population) -> dict:
    out: dict[str, Any] = {}
    groups = graph_groups(list(pop.units.values()))
    for g, us in groups.items():
        arms = arms_for(g, us, groups)
        sources = {}
        for a in arms:
            b = rates_block(us, a)
            sources[a] = {k: b[k] for k in ("C", "C_F", "C_P", "C_U", "false_flag")}
            by_agent = {}
            for m in sorted({u.model for u in us}):
                ba = rates_block([u for u in us if u.model == m], a)
                by_agent[m] = {k: ba[k] for k in ("C", "C_F", "false_flag")}
            sources[a]["by_agent"] = by_agent
        out[g] = {"correct_outputs": sum(1 for u in us if u.output and u.correct), "sources": sources,
                  "not_replayed": nr(pop, arms)}
    return out


def p4(pop: Population) -> dict:
    out: dict[str, Any] = {}
    groups = graph_groups(list(pop.units.values()))
    for g, us in groups.items():
        arms = arms_for(g, us, groups)
        members_by_union: dict[str, list[str]] = {}
        for gg in ([g] if g != ALL_GRAPHS else list(pop.unions)):
            for un, mem in pop.unions.get(gg, {}).items():
                if members_by_union.setdefault(un, mem) != mem:
                    raise InputError(f"union {un} has different members on different graphs")
        block = {}
        for un, members in members_by_union.items():
            if un not in arms:
                continue
            ub = rates_block(us, un)
            singles = [a for a in arms if a not in members_by_union]
            rows = {}
            for s in singles:
                sb = rates_block(us, s)
                wr = wrong(us)
                co = [u for u in us if u.output and u.correct]
                rows[s] = {
                    "recall": sb["recall"], "false_flag": sb["false_flag"],
                    "union_minus_recall": diff(ub["recall"], sb["recall"]),
                    "union_minus_false_flag": diff(ub["false_flag"], sb["false_flag"]),
                    "wrong_flagged_by_union_not_by_source": sum(1 for u in wr if u.arms[un] == "F" and u.arms[s] != "F"),
                    "wrong_flagged_by_source_not_by_union": sum(1 for u in wr if u.arms[s] == "F" and u.arms[un] != "F"),
                    "correct_flagged_by_union_not_by_source": sum(1 for u in co if u.arms[un] == "F" and u.arms[s] != "F"),
                    "correct_flagged_by_source_not_by_union": sum(1 for u in co if u.arms[s] == "F" and u.arms[un] != "F"),
                    "is_member": s in members}
            block[un] = {"members": members, "union": {k: ub[k] for k in ("W", "W_F", "C", "C_F", "recall", "false_flag")},
                         "singles": rows,
                         "best_member": best_of([s for s in singles if s in members], rows, un, ub),
                         "best_single_replayed": best_of(singles, rows, un, ub),
                         "tie_rule": "highest recall; ties: lower false-flag rate, then source name"}
        out[g] = {"unions": block, "not_replayed": nr(pop, arms)}
    return out


def best_of(cands: list[str], rows: dict, un: str, ub: dict) -> Any:
    ok = [s for s in cands if rows[s]["recall"]["rate"] is not None]
    if not ok:
        return NOT_ESTIMABLE
    top = max(rows[s]["recall"]["rate"] for s in ok)
    tied = sorted((s for s in ok if rows[s]["recall"]["rate"] == top),
                  key=lambda s: (rows[s]["false_flag"]["rate"] if rows[s]["false_flag"]["rate"] is not None else 2, s))
    s = tied[0]
    return {"source": s, "tied_on_recall": tied, "union_minus_recall": rows[s]["union_minus_recall"],
            "union_minus_false_flag": rows[s]["union_minus_false_flag"]}


def p5(pop: Population) -> dict:
    """Per agent: every judge's recall beside every source's recall on the same wrong outputs."""
    if pop.judge is None:
        return {"status": "no judge data for this population"}
    jd = pop.judge
    out: dict[str, Any] = {}
    groups = graph_groups(list(pop.units.values()))
    for agent in sorted({u.model for u in pop.units.values()}):
        per_graph = {}
        for g, us in groups.items():
            arms = arms_for(g, us, groups)
            w = [u for u in wrong(us) if u.model == agent]
            keys = [f"output:{u.run_id}" for u in w]
            judges = {}
            for j in pop.judges:
                jc = judge_cell(keys, j, jd)
                judged_units = [u for u in w if (f"output:{u.run_id}", j) in jd.verdict]
                jc["role"] = judge_role(agent, j)
                jc["sources_on_the_judged_outputs"] = {
                    a: ratio(sum(1 for u in judged_units if u.arms[a] == "F"), len(judged_units)) for a in arms}
                judges[j] = jc
            per_graph[g] = {"wrong_outputs": len(w), "wrong_questions": questions(w),
                            "not_replayed": nr(pop, arms),
                            "sources_on_all_wrong": {a: ratio(sum(1 for u in w if u.arms[a] == "F"), len(w))
                                                     for a in arms},
                            "judges": judges}
        out[agent] = per_graph
    return out


def judge_coverage(pop: Population) -> dict:
    if pop.judge is None:
        return {}
    jd = pop.judge
    keys = [f"output:{u.run_id}" for u in pop.units.values() if u.output]
    out = {}
    for j in pop.judges:
        judged = [k for k in keys if (k, j) in jd.verdict]
        by_agent = {}
        for agent in sorted({u.model for u in pop.units.values() if u.output}):
            ak = [f"output:{u.run_id}" for u in pop.units.values() if u.output and u.model == agent]
            by_agent[agent] = {"outputs": len(ak), "judged": sum(1 for k in ak if (k, j) in jd.verdict)}
        in_pop = set(keys)
        raw_counts = collections.Counter(v for (k, jj), v in jd.raw.items() if jj == j and k in in_pop)
        out[j] = {"calls_for_population_outputs": len(judged), "outputs_in_population": len(keys),
                  "outputs_judged": len(judged), "by_agent": by_agent,
                  "status": "not run" if not judged else ("complete" if len(judged) == len(keys) else "partial"),
                  "raw_verdicts": dict(sorted(raw_counts.items()))}
    return out


# -- population check ------------------------------------------------------------------------------------------------

def population_check(pop: Population, registered: dict | None) -> dict:
    us = list(pop.units.values())
    outs = outputs(us)
    wr = wrong(us)
    res: dict[str, Any] = {
        "attempts": len(us), "no_output": len(us) - len(outs), "outputs": len(outs), "wrong": len(wr),
        "correct": len(outs) - len(wr),
        "wrong_by_agent": dict(sorted(collections.Counter(u.model for u in wr).items())),
        "outputs_by_agent": dict(sorted(collections.Counter(u.model for u in outs).items())),
        "by_graph": {g: {"attempts": len(gu), "no_output": sum(1 for u in gu if not u.output),
                         "outputs": len(outputs(gu)), "wrong": len(wrong(gu))}
                     for g, gu in graph_groups(us).items() if g != ALL_GRAPHS},
        "cohorts": dict(collections.Counter(u.cohort for u in us)),
        "models_left_out_of_the_analysed_set_attempts": pop.excluded,
    }
    res["union_consistency"] = union_consistency(pop)
    if pop.dropped:
        d_out = [u for u in pop.dropped if u.output]
        d_wr = [u for u in d_out if not u.correct]
        res["dropped"] = {"records": len(pop.dropped), "no_output": len(pop.dropped) - len(d_out),
                          "outputs": len(d_out), "wrong": len(d_wr),
                          "wrong_by_agent": dict(sorted(collections.Counter(u.model for u in d_wr).items())),
                          "outputs_by_agent": dict(sorted(collections.Counter(u.model for u in d_out).items())),
                          "graphs": sorted({u.graph for u in pop.dropped})}
        if registered:
            res["registered_minus_dropped"] = {
                "outputs": {"registered": registered["outputs"], "dropped": len(d_out), "observed": len(outs),
                            "equal": len(outs) == registered["outputs"] - len(d_out)},
                "wrong": {"registered": registered["wrong"], "dropped": len(d_wr), "observed": len(wr),
                          "equal": len(wr) == registered["wrong"] - len(d_wr)},
                "wrong_by_agent": {a: {"registered": n, "dropped": res["dropped"]["wrong_by_agent"].get(a, 0),
                                       "observed": res["wrong_by_agent"].get(a, 0),
                                       "equal": res["wrong_by_agent"].get(a, 0)
                                       == n - res["dropped"]["wrong_by_agent"].get(a, 0)}
                                   for a, n in registered["wrong_by_agent"].items()}}
            res["registered_minus_dropped"]["all_equal"] = (
                res["registered_minus_dropped"]["outputs"]["equal"] and res["registered_minus_dropped"]["wrong"]["equal"]
                and all(v["equal"] for v in res["registered_minus_dropped"]["wrong_by_agent"].values()))
    if registered and pop.registered:
        def cmp(reg, obs):
            return {"registered": reg, "observed": obs, "difference": obs - reg, "equal": obs == reg}
        res["against_registered"] = {
            "outputs": cmp(registered["outputs"], len(outs)),
            "wrong": cmp(registered["wrong"], len(wr)),
            "wrong_by_agent": {a: cmp(n, res["wrong_by_agent"].get(a, 0))
                               for a, n in registered["wrong_by_agent"].items()}}
        res["against_registered"]["all_equal"] = (
            res["against_registered"]["outputs"]["equal"] and res["against_registered"]["wrong"]["equal"]
            and all(v["equal"] for v in res["against_registered"]["wrong_by_agent"].values()))
    elif registered:
        res["against_registered"] = {
            "note": "a sensitivity of the registered population" if pop.kind == "sensitivity"
            else "not the registered population",
            "outputs": {"registered": registered["outputs"], "observed": len(outs),
                        "difference": len(outs) - registered["outputs"]},
            "wrong": {"registered": registered["wrong"], "observed": len(wr),
                      "difference": len(wr) - registered["wrong"]}}
    if pop.judge is not None:
        jd = pop.judge
        mk = {f"output:{u.run_id}": u for u in outs}
        only_m = sorted(set(mk) - set(jd.truth))
        only_t = sorted(set(jd.truth) - set(mk))
        both = set(mk) & set(jd.truth)
        label_dis = sum(1 for k in both if (jd.truth[k] == "wrong") != (not mk[k].correct))
        sel_dis = sum(1 for k in both if k in jd.selection and jd.selection[k].get("model") != mk[k].model)
        sel_q_dis = sum(1 for k in both if k in jd.selection and jd.selection[k].get("question_id") not in
                        (None, mk[k].question_id))
        stray = {j: sum(1 for (k, jj) in jd.verdict if jj == j and k not in mk) for j in pop.judges}
        res["against_judge_calibration"] = {
            "keys_removed_with_the_dropped_records": len(pop.judge_full.truth) - len(jd.truth) if pop.judge_full else 0,
            "truth_keys": len(jd.truth), "selection_keys": len(jd.selection), "matrix_outputs": len(mk),
            "only_in_matrix": len(only_m), "only_in_truth": len(only_t),
            "truth_label_disagrees_with_matrix_correct": label_dis,
            "selection_model_disagrees_with_matrix_model": sel_dis,
            "selection_question_id_disagrees_with_matrix": sel_q_dis,
            "selection_keys_equal_truth_keys": set(jd.selection) == set(jd.truth),
            "selection_keys_equal_matrix_outputs": set(jd.selection) == set(mk),
            "judge_calls_for_keys_outside_the_population": stray,
            "equal": (not only_m and not only_t and label_dis == 0 and sel_dis == 0 and sel_q_dis == 0
                      and set(jd.selection) == set(jd.truth) and set(jd.selection) == set(mk)
                      and not any(stray.values()))}
    return res


def union_consistency(pop: Population) -> dict:
    """Input check, not an input to any rate: on outputs whose union row lists its members' statuses, each listed
    status equals the member arm's own row (the union row itself is what P4 reads)."""
    checked = disagree = 0
    for u in pop.units.values():
        for un, ps in u.per_source.items():
            if un not in pop.unions.get(u.graph, {}):
                continue
            for member, st in ps.items():
                if member in u.arms:
                    checked += 1
                    disagree += STATUS_LETTER.get(st) != u.arms[member]
    return {"member_statuses_checked": checked, "disagreeing": disagree,
            "available": checked > 0}


# -- reconciliations (non-circular: other code computed the comparison numbers) --------------------------------------

def full_units(pop: Population) -> list[Unit]:
    """The analysed units plus the dropped ones: what accounting.json and tables.json cover."""
    return list(pop.units.values()) + list(pop.dropped)


def reconcile_accounting(pop: Population) -> dict:
    """The six cells per (graph, model, cohort, arm) against the replay's own accounting.json (match/mismatch)."""
    compared = mism = 0
    detail = []
    for g, acc in sorted(pop.accounting.items()):
        us = [u for u in full_units(pop) if u.graph == g]
        for model in sorted({u.model for u in us}):
            for cohort in sorted({str(u.cohort) for u in us}):
                sel = [u for u in us if u.model == model and str(u.cohort) == cohort]
                for arm in arms_of(sel):
                    try:
                        theirs = acc["accounting"][model][cohort][arm][PRIMARY]
                    except KeyError:
                        compared += 1
                        mism += 1
                        detail.append({"graph": g, "model": model, "arm": arm, "problem": "absent in accounting"})
                        continue
                    mine = cells(sel, arm)
                    want = theirs["cells"]
                    n_out = len(outputs(sel))
                    compared += 1
                    if any(mine[k] != want[k] for k in mine) or theirs["N_out"] != n_out:
                        mism += 1
                        detail.append({"graph": g, "model": model, "arm": arm, "problem": "cells differ"})
    return {"compared": compared, "mismatches": mism, "match": mism == 0 and compared > 0, "detail": detail}


def reconcile_design_log(pop: Population, summary: dict) -> dict:
    """Pooled counts per source and template over the summary's group P against design_log_summary.json (the analysis's
    own code), and the attempt count."""
    group = set(summary["groups"]["P"])
    compared = mism = 0
    detail = []
    compared += 1
    n_all = len(pop.units) + sum(pop.excluded.values())
    if summary["attempts"] != n_all:
        mism += 1
        detail.append({"problem": "attempts differ", "summary": summary["attempts"], "units": n_all})
    for src, per_t in sorted(summary["per_template_P"].items()):
        for t, theirs in sorted(per_t.items()):
            us = [u for u in pop.units.values() if u.model in group and u.template == t]
            compared += 1
            if not us or src not in us[0].arms:
                mism += 1
                detail.append({"source": src, "template": t, "problem": "no such arm or template here"})
                continue
            b = rates_block(us, src)
            if (b["W"], b["W_F"], b["C"], b["C_F"]) != (theirs["W"], theirs["W_F"], theirs["C"], theirs["C_F"]):
                mism += 1
                detail.append({"source": src, "template": t, "problem": "counts differ"})
    return {"compared": compared, "mismatches": mism, "match": mism == 0 and compared > 0, "detail": detail}


# -- reconciliation of the PRINTED blocks against counts another program produced -----------------------------------

QUAD = ("W", "W_F", "C", "C_F")


def quad_of_cells(c: dict) -> dict[str, int]:
    return {"W": c["W_F"] + c["W_P"] + c["W_U"], "W_F": c["W_F"], "C": c["C_F"] + c["C_P"] + c["C_U"], "C_F": c["C_F"]}


def reference_counts(pop: Population) -> tuple[dict, dict, str] | None:
    """(cells_ref, tmpl_ref, name): W, W_F, C, C_F per (graph, model, arm) and per (graph, model, arm, template), read
    from accounting.json (the replay's own accounting) or, for a population with none, design_log_summary.json (the
    analysis's own code). For design_log the template counts are pooled over the summary's group (model is None)."""
    cells_ref: dict[tuple, dict] = {}
    tmpl_ref: dict[tuple, dict] = {}
    if pop.accounting:
        for g, acc in pop.accounting.items():
            for model, per_cohort in acc["accounting"].items():
                for _cohort, per_arm in per_cohort.items():
                    for arm, per_var in per_arm.items():
                        a = per_var[PRIMARY]
                        if (g, model, arm) in cells_ref:
                            raise InputError(f"accounting.json: {(g, model, arm)} appears under two cohorts")
                        cells_ref[(g, model, arm)] = quad_of_cells(a["cells"])
                        for t, d in a["strata"]["template"].items():
                            tmpl_ref[(g, model, arm, t)] = quad_of_cells(d["cells"])
        for u in pop.dropped:  # the file covers the dropped records: take their counts out
            if not u.output:
                continue
            w_or_c = "C" if u.correct else "W"
            for arm, st in u.arms.items():
                for ref_, key in ((cells_ref, (u.graph, u.model, arm)), (tmpl_ref, (u.graph, u.model, arm, u.template))):
                    ref_[key][w_or_c] -= 1
                    if st == "F":
                        ref_[key][w_or_c + "_F"] -= 1
        return cells_ref, tmpl_ref, "accounting.json" + (" less the dropped records" if pop.dropped else "")
    if pop.design_log is not None:
        graphs = {u.graph for u in pop.units.values()}
        if len(graphs) != 1:
            raise InputError("design_log_summary.json reconciliation needs a one-graph population")
        g = next(iter(graphs))
        for arm, src in pop.design_log["sources"].items():
            for model, d in src["by_model"].items():
                cells_ref[(g, model, arm)] = {k: d["counts"][k] for k in QUAD}
        for arm, per_t in pop.design_log["per_template_P"].items():
            for t, d in per_t.items():
                tmpl_ref[(g, None, arm, t)] = {k: d[k] for k in QUAD}
        return cells_ref, tmpl_ref, "design_log_summary.json"
    return None


def reconcile_printed(pop: Population, pr: dict) -> dict | None:
    """Every W / W_F / C / C_F behind the printed P1-P5 numbers (per graph, agent and source, the pooled rows, the P2
    template cells and focus sums, the P4 union and single counts) against the reference counts. None when the
    population has no reference."""
    ref = reference_counts(pop)
    if ref is None:
        return None
    cells_ref, tmpl_ref, ref_name = ref
    agents = sorted({u.model for u in pop.units.values()})
    compared = 0
    detail: list[dict] = []
    graphs = sorted({u.graph for u in pop.units.values()})
    expected_groups = graphs + ([ALL_GRAPHS] if len(graphs) > 1 else [])
    for name, got, want in (("P1", set(pr["P1"]), set(graphs)), ("P2", set(pr["P2"]), set(graphs)),
                            ("P3", set(pr["P3"]), set(expected_groups)), ("P4", set(pr["P4"]), set(expected_groups))):
        compared += 1
        if got != want:
            detail.append({"where": f"{name} graph groups", "printed": sorted(got), "reference": sorted(want)})
    if pr["P5"].get("status") is None:
        for agent, per_graph in pr["P5"].items():
            compared += 1
            if set(per_graph) != set(expected_groups):
                detail.append({"where": f"P5 graph groups of agent {agent}", "printed": sorted(per_graph),
                               "reference": sorted(expected_groups)})

    def refsum(g: str, arm: str, models: list[str]) -> dict | None:
        total = {k: 0 for k in QUAD}
        for m in models:
            for gg in ([g] if g != ALL_GRAPHS else sorted({k[0] for k in cells_ref})):
                got = cells_ref.get((gg, m, arm))
                if got is None:
                    return None
                for k in QUAD:
                    total[k] += got[k]
        return total

    arms_known = {k[2] for k in cells_ref}

    def tsum(g: str, arm: str, templates: list[str]) -> dict | None:
        """Template counts summed over the analysed agents. A template the reference has no entry for counts as zero
        (an agent with no output there); any printed count above that is then a mismatch. An arm the reference does
        not know at all is a mismatch."""
        if arm not in arms_known:
            return None
        total = {k: 0 for k in QUAD}
        for t in templates:
            if (g, None, arm, t) in tmpl_ref:
                parts = [tmpl_ref[(g, None, arm, t)]]
            else:
                parts = [tmpl_ref.get((g, m, arm, t), {k: 0 for k in QUAD}) for m in agents]
            for x in parts:
                for k in QUAD:
                    total[k] += x[k]
        return total

    def chk(where: str, got: tuple, want: tuple | None) -> None:
        nonlocal compared
        compared += 1
        if want is None or tuple(got) != tuple(want):
            detail.append({"where": where, "printed": list(got), "reference": None if want is None else list(want)})

    def pair(r: dict) -> tuple[int, int]:
        return r["num"], r["den"]

    for g, gr in pr["P1"].items():
        w0 = refsum(g, gr["sources"][0]["source"], agents)
        chk(f"P1 {g} wrong outputs", (gr["wrong_outputs"],), None if w0 is None else (w0["W"],))
        for e in gr["sources"]:
            arm = e["source"]
            for m, r in e["real_wrong_by_agent"].items():
                w = refsum(g, arm, [m])
                chk(f"P1 {g} {arm} agent {m}", pair(r), None if w is None else (w["W_F"], w["W"]))
            w = refsum(g, arm, list(e["real_wrong_by_agent"]))
            chk(f"P1 {g} {arm} all wrong", pair(e["real_wrong_all"]), None if w is None else (w["W_F"], w["W"]))
            rr = e["real_wrong_result_rows"]
            if isinstance(rr, dict):  # a subset of the wrong outputs: bounded by the reference's totals
                compared += 1
                if not (rr["num"] <= rr["den"] <= (w or {"W": -1})["W"] and rr["num"] <= e["real_wrong_all"]["num"]):
                    detail.append({"where": f"P1 {g} {arm} non-empty subset", "printed": pair(rr)})
    for g, gr in pr["P2"].items():
        for t, tr in gr["templates"].items():
            for arm, b in tr["sources"].items():
                w = tsum(g, arm, [t])
                chk(f"P2 {g} {t} {arm}", (b["W_F"], b["W"]), None if w is None else (w["W_F"], w["W"]))
        f = gr["focus"]
        if f:
            for arm, b in f["sources"].items():
                w = tsum(g, arm, list(f["templates_present"]))
                chk(f"P2 {g} focus {arm}", (b["W_F"], b["W"]), None if w is None else (w["W_F"], w["W"]))
    for g, gr in pr["P3"].items():
        c0 = refsum(g, next(iter(gr["sources"])), agents)
        chk(f"P3 {g} correct outputs", (gr["correct_outputs"],), None if c0 is None else (c0["C"],))
        for arm, b in gr["sources"].items():
            for m, ba in b["by_agent"].items():
                w = refsum(g, arm, [m])
                chk(f"P3 {g} {arm} agent {m}", (ba["C_F"], ba["C"]), None if w is None else (w["C_F"], w["C"]))
            w = refsum(g, arm, list(b["by_agent"]))
            chk(f"P3 {g} {arm} all correct", (b["C_F"], b["C"]), None if w is None else (w["C_F"], w["C"]))
    for g, grp in pr["P4"].items():
        for un, ub in grp["unions"].items():
            wu = refsum(g, un, agents)
            u = ub["union"]
            chk(f"P4 {g} {un} union", (u["W_F"], u["W"], u["C_F"], u["C"]),
                None if wu is None else (wu["W_F"], wu["W"], wu["C_F"], wu["C"]))
            for s_, r in ub["singles"].items():
                ws = refsum(g, s_, agents)
                chk(f"P4 {g} {s_} single", (pair(r["recall"]) + pair(r["false_flag"])),
                    None if ws is None else (ws["W_F"], ws["W"], ws["C_F"], ws["C"]))
                if wu is not None and ws is not None:  # the one-sided counts tie to the reference totals
                    chk(f"P4 {g} {un} vs {s_} one-sided wrong counts",
                        (r["wrong_flagged_by_union_not_by_source"] - r["wrong_flagged_by_source_not_by_union"],
                         r["correct_flagged_by_union_not_by_source"] - r["correct_flagged_by_source_not_by_union"]),
                        (wu["W_F"] - ws["W_F"], wu["C_F"] - ws["C_F"]))
    if pr["P5"].get("status") is None:
        for agent, per_graph in pr["P5"].items():
            for g, gr in per_graph.items():
                w0 = refsum(g, next(iter(gr["sources_on_all_wrong"])), [agent])
                chk(f"P5 {g} agent {agent} wrong outputs", (gr["wrong_outputs"],), None if w0 is None else (w0["W"],))
                for arm, r in gr["sources_on_all_wrong"].items():
                    w = refsum(g, arm, [agent])
                    chk(f"P5 {g} {arm} agent {agent}", pair(r), None if w is None else (w["W_F"], w["W"]))
                for j, jc in gr["judges"].items():
                    if jc["coverage"] == "complete":  # the judged set is the whole wrong set: same source counts
                        compared += 1
                        if jc["sources_on_the_judged_outputs"] != gr["sources_on_all_wrong"]:
                            detail.append({"where": f"P5 {g} agent {agent} judge {j}: judged-subset sources differ"})
    return {"against": ref_name, "compared": compared, "mismatches": len(detail), "match": not detail and compared > 0,
            "detail": detail[:50]}


def reconcile_nonempty(pop: Population, pr: dict, tables: dict) -> dict:
    """The wrong-output counts by final checked result (rows / empty / refused) per graph and agent against
    tables.json table 2 (the calibration's own tables.py, which takes graph and agent from the selection)."""
    short = {"claude-sonnet-5-5": "Sonnet", "claude-haiku-5-5": "Haiku", QWEN: "Qwen"}
    judge = next((short.get(j, j) for j in pop.judges
                  if any(r["judge"] == short.get(j, j) for r in tables["table_2_by_result"])), None)
    ref: dict[tuple, int] = collections.Counter()
    for r in tables["table_2_by_result"]:
        if r["judge"] == judge:
            ref[(r["graph"], r["agent"], r["result"])] += r["wrong"]
    for u in pop.dropped:  # the tables cover the dropped records: take their wrong outputs out
        if u.output and not u.correct:
            ref[(GRAPH_SHORT.get(u.graph, u.graph), u.model, pop.result_full[u.run_id])] -= 1
    compared = 0
    detail: list[dict] = []
    for g, gr in pr["P1"].items():
        if gr["wrong_by_result_by_agent"] is None:
            continue
        rows_total = sum(ref.get((GRAPH_SHORT.get(g, g), m, "rows"), 0) for m in gr["wrong_by_result_by_agent"])
        compared += 1
        if gr["wrong_with_result_rows"] != rows_total:
            detail.append({"graph": g, "what": "wrong_with_result_rows", "printed": gr["wrong_with_result_rows"],
                           "tables": rows_total})
        for e in gr["sources"]:
            rr = e["real_wrong_result_rows"]
            compared += 1
            if not isinstance(rr, dict) or rr["den"] != rows_total:
                detail.append({"graph": g, "what": f"non-empty denominator of {e['source']}",
                               "printed": rr if not isinstance(rr, dict) else rr["den"], "tables": rows_total})
        for m, per_class in gr["wrong_by_result_by_agent"].items():
            for c, n in per_class.items():
                compared += 1
                if ref.get((GRAPH_SHORT.get(g, g), m, c), 0) != n:
                    detail.append({"graph": g, "agent": m, "result": c, "printed": n,
                                   "tables": ref.get((GRAPH_SHORT.get(g, g), m, c), 0)})
    return {"against": f"tables.json table_2_by_result (judge {judge})", "compared": compared,
            "mismatches": len(detail), "match": not detail and compared > 0, "detail": detail}


def reconcile_p1_bank(pr: dict, bank: dict) -> dict:
    """The bank numbers printed beside each real recall against the bank run's own bank_rates, and each printed ratio
    against the division of the two printed recalls."""
    compared = 0
    detail: list[dict] = []
    for g, gr in pr["P1"].items():
        for e in gr["sources"]:
            b = e["bank"]
            if not isinstance(b, dict):
                continue
            theirs = bank["bank_rates"].get(b["bank_arm"], {}).get(PRIMARY, {})
            for stratum, key in (("returns_rows", "returns_rows"), ("all", "all_items")):
                t = theirs.get(stratum)
                x = b[key]
                compared += 1
                if t is None or (x["W"], x["W_F"], x["C"], x["C_F"]) != (t["W"], t["W_F"], t["C"], t["C_F"]):
                    detail.append({"graph": g, "source": e["source"], "stratum": stratum, "printed":
                                   [x["W"], x["W_F"], x["C"], x["C_F"]], "bank_rates": t})
            bank_rate = b["returns_rows"]["recall"]["rate"]
            for field, real in (("ratio_real_all_over_bank_returns_rows", e["real_wrong_all"]),
                                ("ratio_real_result_rows_over_bank_returns_rows", e["real_wrong_result_rows"])):
                if not isinstance(real, dict):
                    continue
                compared += 1
                want = NOT_ESTIMABLE if real["rate"] is None or not bank_rate else real["rate"] / bank_rate
                got = e[field]
                same = (got == want) if isinstance(want, str) or isinstance(got, str) else abs(got - want) < 1e-12
                if not same:
                    detail.append({"graph": g, "source": e["source"], "field": field, "printed": got, "recomputed": want})
    return {"compared": compared, "mismatches": len(detail), "match": not detail and compared > 0, "detail": detail}


def reconcile_bank(bank: dict) -> dict:
    """Bank recall counts per arm and stratum against the bank run's own bank_rates."""
    compared = mism = 0
    detail = []
    for arm, strata in bank["counts"].items():
        for stratum in ("all", "returns_rows", "empty"):
            theirs = bank["bank_rates"].get(arm, {}).get(PRIMARY, {}).get(stratum)
            mine = strata.get(stratum)
            if theirs is None and mine is None:
                continue
            compared += 1
            if theirs is None or mine is None:
                mism += 1
                detail.append({"arm": arm, "stratum": stratum, "problem": "one side absent"})
                continue
            b = cells_ratios(mine)
            if (b["W"], b["W_F"], b["C"], b["C_F"]) != (theirs["W"], theirs["W_F"], theirs["C"], theirs["C_F"]):
                mism += 1
                detail.append({"arm": arm, "stratum": stratum, "problem": "counts differ"})
    return {"compared": compared, "mismatches": mism, "match": mism == 0 and compared > 0, "detail": detail}


def reconcile_judge(pop: Population, tables: dict) -> dict:
    """Judge counts per (graph, agent, judge) against the interim tables.json, summed over rounds."""
    jd = pop.judge_full or pop.judge
    short = {"claude-sonnet-5-5": "Sonnet", "claude-haiku-5-5": "Haiku", QWEN: "Qwen"}
    theirs: dict[tuple, dict] = {}
    for row in tables["table_1_by_graph_round_agent_judge"]:
        key = (row["graph"], row["agent"], row["judge"])
        t = theirs.setdefault(key, {"wrong": 0, "reject": 0, "unparsed": 0, "right": 0, "false_reject": 0})
        t["wrong"] += row["wrong"]
        t["right"] += row["right"]
        if row.get("recall"):
            t["reject"] += row["recall"]["reject"]
            t["unparsed"] += row["recall"]["unparsed"]
        if row.get("false_reject"):
            t["false_reject"] += row["false_reject"]["reject"]
    compared = mism = 0
    detail = []
    seen = set()
    everything = full_units(pop)
    for g in sorted({u.graph for u in everything}) + [ALL_GRAPHS]:
        for agent in sorted({u.model for u in everything}):
            for j in pop.judges:
                key = (GRAPH_SHORT.get(g, g), agent, short.get(j, j))
                if key not in theirs:
                    continue
                seen.add(key)
                us = [u for u in everything if (g == ALL_GRAPHS or u.graph == g) and u.model == agent
                      and u.output]
                w = [f"output:{u.run_id}" for u in us if not u.correct]
                r = [f"output:{u.run_id}" for u in us if u.correct]
                mine = {"wrong": len(w), "right": len(r),
                        "reject": sum(1 for k in w if jd.verdict.get((k, j)) == "reject"),
                        "unparsed": sum(1 for k in w if jd.verdict.get((k, j)) in ("unparsed", "error")),
                        "false_reject": sum(1 for k in r if jd.verdict.get((k, j)) == "reject")}
                compared += 1
                if mine != theirs[key]:
                    mism += 1
                    detail.append({"key": list(key), "problem": "counts differ"})
    missing = sorted(set(theirs) - seen)
    if missing:
        mism += len(missing)
        detail += [{"key": list(k), "problem": "in tables.json, not computed"} for k in missing]
    return {"compared": compared, "mismatches": mism, "match": mism == 0 and compared > 0, "detail": detail}


def frozen_source_check(inputs: dict, frozen_dir: Path) -> list[dict]:
    """The replay's source files (sha256 from its inputs.json) against the same-named files in a frozen tree."""
    out = []
    for g, gd in sorted(inputs["graphs"].items()):
        for s, sd in sorted(gd["sources"].items()):
            cand = [p for p in (frozen_dir / g).glob(f"{s}.*") if p.name.split(".")[0] == s]
            frozen = {p.name: sha256_file(p) for p in sorted(cand)}
            out.append({"graph": g, "source": s, "replay_file_sha256": sd["sha256"], "frozen_files": frozen,
                        "equal": sd["sha256"] in frozen.values()})
    return out


# -- text ------------------------------------------------------------------------------------------------------------

def fmt_num(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    return f"{x:.3f}"


def render_text(res: dict) -> str:
    L: list[str] = []
    add = L.append

    def nr_lines(d: dict, indent: str = "    ") -> None:
        for a, b in d.items():
            add(f"{indent}{a:<16}{b}")

    add(f"P1-P5 on the dev checker replay ({res['tool']['version']}); counts and rates only; no verdict labels.")
    add("recall = W_F/(W_F+W_P+W_U) over wrong outputs; false-flag = C_F/(C_F+C_P+C_U) over correct outputs; "
        "U counts in the denominator and is not a flag; primary variant; no intervals (none prescribed for P1-P5).")
    add("")
    for name, pr in res["populations"].items():
        add("=" * 100)
        add(f"POPULATION {name}: {pr['title']}")
        pc = pr["population_check"]
        add(f"  attempts {pc['attempts']}, no output {pc['no_output']}, outputs {pc['outputs']} "
            f"(wrong {pc['wrong']}, correct {pc['correct']}); wrong by agent {pc['wrong_by_agent']}")
        if pc["models_left_out_of_the_analysed_set_attempts"]:
            add(f"  models left out of this population (attempts): {pc['models_left_out_of_the_analysed_set_attempts']}")
        if pc.get("dropped"):
            dr = pc["dropped"]
            add(f"  DROPPED from the registered population: {dr['records']} records ({dr['outputs']} outputs, "
                f"{dr['no_output']} without output; wrong {dr['wrong']}: {dr['wrong_by_agent']}; graphs {dr['graphs']})")
            rm = pc.get("registered_minus_dropped")
            if rm:
                add(f"  registered minus dropped: outputs {rm['outputs']['registered']} - {rm['outputs']['dropped']} = "
                    f"{rm['outputs']['registered'] - rm['outputs']['dropped']} (observed {rm['outputs']['observed']}), wrong "
                    f"{rm['wrong']['registered']} - {rm['wrong']['dropped']} = "
                    f"{rm['wrong']['registered'] - rm['wrong']['dropped']} (observed {rm['wrong']['observed']}): "
                    f"equal={rm['all_equal']}")
        ar = pc.get("against_registered")
        if ar and "all_equal" in ar:
            add(f"  registered {ar['outputs']['registered']} outputs / {ar['wrong']['registered']} wrong "
                f"(Sonnet {ar['wrong_by_agent'].get('claude-sonnet-5-5', {}).get('registered')}, "
                f"Qwen {ar['wrong_by_agent'].get(QWEN, {}).get('registered')}): "
                f"outputs diff {ar['outputs']['difference']:+d}, wrong diff {ar['wrong']['difference']:+d}, "
                f"equal={ar['all_equal']}")
        elif ar and pc.get("dropped"):
            pass  # the arithmetic above states it
        elif ar:
            add(f"  NOT the registered population: outputs {ar['outputs']['difference']:+d} and wrong "
                f"{ar['wrong']['difference']:+d} against 1,445 / 201")
        uc = pc["union_consistency"]
        add(f"  union rows' member statuses against the member arms' own rows: checked "
            f"{uc['member_statuses_checked']}, disagreeing {uc['disagreeing']}")
        jc = pc.get("against_judge_calibration")
        if jc:
            add(f"  against the judge calibration: matrix outputs {jc['matrix_outputs']}, truth keys {jc['truth_keys']}, "
                f"only in matrix {jc['only_in_matrix']}, only in truth {jc['only_in_truth']}, label disagreements "
                f"{jc['truth_label_disagrees_with_matrix_correct']}, equal={jc['equal']}")
        add("")
        add("P1  recall on real wrong outputs beside bank recall on non-empty items (same source digest)")
        for g, gr in pr["P1"].items():
            add(f"  [{g}] wrong outputs {gr['wrong_outputs']} ({gr['wrong_questions']} questions); by final checked "
                f"result {gr['wrong_by_result'] if gr['wrong_by_result'] is not None else 'n/a'}; non-empty = result rows: "
                f"{'n/a' if gr['wrong_with_result_rows'] is None else gr['wrong_with_result_rows']}")
            add(f"    {'source':<16}{'real, all wrong':<21}{'real, result rows':<31}{'bank, non-empty':<22}"
                f"{'real(all)/bank':<17}{'real(rows)/bank':<18}bank arm")
            for e in gr["sources"]:
                b = e["bank"]
                if isinstance(b, dict):
                    bs, arm_s = fmt_ratio(b["returns_rows"]["recall"]), b["bank_arm"]
                    r1, r2 = e["ratio_real_all_over_bank_returns_rows"], e["ratio_real_result_rows_over_bank_returns_rows"]
                    rts = (fmt_num(r1), fmt_num(r2))
                else:
                    bs, arm_s, rts = "-", b, ("-", "-")
                add(f"    {e['source']:<16}{fmt_ratio(e['real_wrong_all']):<21}{fmt_ratio(e['real_wrong_result_rows']):<31}"
                    f"{bs:<22}{rts[0]:<17}{rts[1]:<18}{arm_s}")
            nr_lines(gr["not_replayed"])
        add("")
        add("P2  recall per source by template (wrong outputs)")
        for g, gr in pr["P2"].items():
            f = gr["focus"]
            if f:
                absent = [t for t in f["templates"] if t not in f["templates_present"]]
                add(f"  [{g}] focus {f['templates']}: wrong {f['wrong_outputs']} ({f['wrong_questions']} questions)"
                    + (f"; template(s) absent from this replay: {absent}" if absent else ""))
                for a, b in f["sources"].items():
                    add(f"    {a:<16}{fmt_ratio(b['recall'])}")
            for t, tr in gr["templates"].items():
                cells_s = "  ".join(f"{a}={fmt_ratio(b['recall'])}" for a, b in tr["sources"].items())
                add(f"    {t:<22} wrong {tr['wrong_outputs']:>3}  {cells_s}")
            nr_lines(gr["not_replayed"])
        add("")
        add("P3  false-flag rate on correct outputs")
        for g, gr in pr["P3"].items():
            add(f"  [{g}] correct outputs {gr['correct_outputs']}")
            for a, b in gr["sources"].items():
                add(f"    {a:<16}{fmt_ratio(b['false_flag'])}")
            for a, b in gr["not_replayed"].items():
                add(f"    {a:<16}{b}")
        add("")
        add("P4  the union against its single sources (union minus source; positive = union higher)")
        for g, grp in pr["P4"].items():
            gr = grp["unions"]
            for un, ub in gr.items():
                add(f"  [{g}] {un} = {'+'.join(ub['members'])}: recall {fmt_ratio(ub['union']['recall'])}, "
                    f"false-flag {fmt_ratio(ub['union']['false_flag'])}")
                for s, r in ub["singles"].items():
                    dr = "-" if r["union_minus_recall"] is None else f"{r['union_minus_recall']:+.3f}"
                    df = "-" if r["union_minus_false_flag"] is None else f"{r['union_minus_false_flag']:+.3f}"
                    add(f"    vs {s:<16}recall {fmt_ratio(r['recall']):<20}false-flag {fmt_ratio(r['false_flag']):<20}"
                        f"d_recall {dr:<8}d_false_flag {df:<8}"
                        f"(wrong only union {r['wrong_flagged_by_union_not_by_source']}, only source "
                        f"{r['wrong_flagged_by_source_not_by_union']})")
                for k in ("best_member", "best_single_replayed"):
                    b = ub[k]
                    if isinstance(b, dict):
                        add(f"    {k}: {b['source']} (tied on recall: {b['tied_on_recall']}); d_recall "
                            f"{b['union_minus_recall']:+.3f}; d_false_flag {b['union_minus_false_flag']:+.3f}"
                            if b["union_minus_recall"] is not None and b["union_minus_false_flag"] is not None else
                            f"    {k}: {b['source']} (tied on recall: {b['tied_on_recall']}); difference not estimable")
                    else:
                        add(f"    {k}: {b}")
            nr_lines(grp["not_replayed"])
        add("")
        add("P5  judge recall beside every source's recall on the same wrong outputs")
        p5r = pr["P5"]
        if "status" in p5r:
            add(f"  {p5r['status']}")
        else:
            for agent, per_graph in p5r.items():
                for g, gr in per_graph.items():
                    add(f"  agent {AGENT_SHORT.get(agent, agent)} [{g}] wrong outputs {gr['wrong_outputs']} "
                        f"({gr['wrong_questions']} questions)")
                    for a, r in gr["sources_on_all_wrong"].items():
                        add(f"    source {a:<16}{fmt_ratio(r)}")
                    for a, b in gr["not_replayed"].items():
                        add(f"    source {a:<16}{b}")
                    for j, jc in gr["judges"].items():
                        rec = jc["recall"] if jc["recall"] is not None else jc.get("recall_on_judged_subset")
                        tag = "" if jc["recall"] is not None else (" (judged subset only)" if rec else "")
                        add(f"    judge  {AGENT_SHORT.get(j, j):<16}role {jc['role']:<6}{jc['coverage']:<18}"
                            f"{fmt_ratio(rec) if rec else 'not run (no calls for this judge in the input)'}{tag}  "
                            f"unparsed/failed {jc['unparsed_or_failed']}")
        cov = pr["judge_coverage"]
        if cov:
            add("  judge coverage of the population's outputs:")
            for j, c in cov.items():
                add(f"    {j}: {c['status']}, calls {c['calls_for_population_outputs']}, outputs judged {c['outputs_judged']} of "
                    f"{c['outputs_in_population']}; raw verdicts {c['raw_verdicts']}")
        add("")
    add("=" * 100)
    add("RECONCILIATIONS (match / mismatch only)")
    for k, v in res["reconciliation"].items():
        if isinstance(v, dict) and "compared" in v:
            add(f"  {k}: compared {v['compared']}, mismatches {v['mismatches']}, "
                f"{'MATCH' if v['match'] else 'MISMATCH'}")
        elif isinstance(v, list):
            add(f"  {k}: " + "; ".join(f"{e['graph']}/{e['source']}={'equal' if e['equal'] else 'DIFFERENT'}" for e in v))
        elif isinstance(v, str):
            add(f"  {k}: {v}")
    add(f"RESULT {res['result']}")
    return "\n".join(L) + "\n"


# -- driver ----------------------------------------------------------------------------------------------------------

def resolve(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else base / q


def load_population(cfg_pop: dict, base: Path, split: str, shas: dict, judge_cfg: dict | None) -> Population:
    rows: list[dict] = []
    digests: dict[str, dict[str, str]] = {}
    unions: dict[str, dict[str, list[str]]] = {}
    accounting: dict[str, dict] = {}
    for m in cfg_pop["matrices"]:
        mp = resolve(base, m["matrix"])
        shas[str(mp)] = sha256_file(mp)
        part = list(project(read_jsonl(mp), MATRIX_FIELDS))
        graphs = {r["graph"] for r in part}
        if len(graphs) != 1:
            raise InputError(f"{mp}: expected one graph, found {sorted(map(str, graphs))}")
        g = graphs.pop()
        rows += part
        ap = resolve(base, m["accounting"])
        shas[str(ap)] = sha256_file(ap)
        acc = read_json(ap)
        digests[g] = dict(acc["meta"]["source_digests"])
        unions[g] = {k: list(v) for k, v in (acc["meta"].get("unions") or {}).items()}
        if m.get("reconcile_accounting", True):
            accounting[g] = acc
    units = build_units(rows, split)
    dropped: list[Unit] = []
    drop_cfg = cfg_pop.get("drop_runs")
    if drop_cfg:  # a sensitivity: the records the audit's own population file labels with one frame label
        dp = resolve(base, drop_cfg["file"])
        shas[str(dp)] = sha256_file(dp)
        ids = {r["run_id"] for r in read_jsonl(dp)
               if r.get("frame") == drop_cfg["frame"] and r.get("graph") == drop_cfg["graph"]}
        if len(ids) != drop_cfg["expect_records"]:
            raise InputError(f"drop_runs: {len(ids)} records carry the frame label {drop_cfg['frame']!r} on "
                             f"{drop_cfg['graph']}, expected {drop_cfg['expect_records']}")
        by_id = {u.run_id: k for k, u in units.items() if u.graph == drop_cfg["graph"]}
        absent = sorted(ids - set(by_id))
        if absent:
            raise InputError(f"drop_runs: {len(absent)} labelled records are not in the matrix (first: {absent[0]})")
        dropped = [units.pop(by_id[i]) for i in sorted(ids)]
    excluded: dict[str, int] = {}
    e77 = None
    if cfg_pop.get("design_log_summary"):
        ep = resolve(base, cfg_pop["design_log_summary"])
        shas[str(ep)] = sha256_file(ep)
        e77 = read_json(ep)
    if cfg_pop.get("model_group"):  # analyse only the models of one group of the design_log summary
        group = set(e77["groups"][cfg_pop["model_group"]])
        excluded = dict(sorted(collections.Counter(u.model for u in units.values() if u.model not in group).items()))
        units = {k: u for k, u in units.items() if u.model in group}
    jd = None
    jd_full = None
    result: dict[str, str] = {}
    result_full: dict[str, str] = {}
    judges: list[str] = []
    if judge_cfg:
        paths = {k: resolve(base, judge_cfg[k]) for k in ("truth", "selection", "calls")}
        for p in paths.values():
            shas[str(p)] = sha256_file(p)
        jd = load_judge(paths["truth"], paths["selection"], paths["calls"])
        jd_full = jd
        judges = list(judge_cfg["judges"])
        if dropped:
            gone = {f"output:{u.run_id}" for u in dropped if u.output}
            if not gone <= set(jd.truth):
                raise InputError("drop_runs: a dropped output is not in the judge calibration")
            jd = JudgeData({k: v for k, v in jd.truth.items() if k not in gone},
                           {k: v for k, v in jd.selection.items() if k not in gone},
                           {p: v for p, v in jd.verdict.items() if p[0] not in gone},
                           {p: v for p, v in jd.raw.items() if p[0] not in gone})
        no_result = [k for k, o in jd.selection.items() if "result" not in o]
        if no_result:
            raise InputError(f"{len(no_result)} judge selection entries carry no checked-query result "
                             f"(first: {no_result[0]})")
        result = {k.split(":", 1)[1]: o["result"] for k, o in jd.selection.items()}
        result_full = {k.split(":", 1)[1]: o["result"] for k, o in jd_full.selection.items() if "result" in o}
    return Population(cfg_pop["name"], cfg_pop["title"], bool(cfg_pop.get("registered")), units, digests, unions,
                      result, jd, judges, accounting, e77, excluded, cfg_pop.get("kind", "registered"), dropped,
                      jd_full, result_full)


def load_bank(cfg_bank: dict, base: Path, split: str, shas: dict) -> dict:
    mp, pp, rp = (resolve(base, cfg_bank[k]) for k in ("matrix", "plan", "run"))
    for p in (mp, pp, rp):
        shas[str(p)] = sha256_file(p)
    plan = read_json(pp)
    run = read_json(rp)
    rows = list(project(read_jsonl(mp), BANK_FIELDS))
    graphs = {r["graph"] for r in rows}
    if graphs != {cfg_bank["graph"]}:
        raise InputError(f"bank graphs {sorted(map(str, graphs))}, expected {cfg_bank['graph']}")
    if plan.get("split") != split:
        raise InputError(f"bank plan split {plan.get('split')!r}")
    return {"graph": cfg_bank["graph"], "counts": bank_counts(rows, split),
            "digests": {a: d["digest"] for a, d in plan["sources"].items()}, "bank_rates": run["bank_rates"],
            "plan_code": plan.get("code"), "items": run.get("items")}


def run(cfg: dict, base: Path) -> dict:
    split = cfg["expected_split"]
    shas: dict[str, str] = {}
    bank = load_bank(cfg["bank"], base, split, shas) if cfg.get("bank") else None
    registered = cfg.get("registered")
    res: dict[str, Any] = {"tool": {"name": "p1p5.py", "version": TOOL_VERSION}, "expected_split": split,
                           "populations": {}, "reconciliation": {}}
    ok = True
    for cp in cfg["populations"]:
        pop = load_population(cp, base, split, shas, cp.get("judge"))
        pc = population_check(pop, registered)
        pr = {"title": pop.title, "registered_population": pop.registered, "population_check": pc,
              "P1": p1(pop, bank), "P2": p2(pop), "P3": p3(pop), "P4": p4(pop), "P5": p5(pop),
              "judge_coverage": judge_coverage(pop), "source_digests": pop.digests, "unions": pop.unions}
        res["populations"][pop.name] = pr
        if pop.accounting:
            acc = reconcile_accounting(pop)
            res["reconciliation"][f"{pop.name}: six cells against accounting.json"] = acc
            ok &= acc["match"]
        ok &= pc["union_consistency"]["disagreeing"] == 0
        rp = reconcile_printed(pop, pr)
        if rp is not None:
            res["reconciliation"][f"{pop.name}: printed P1-P5 blocks against {rp['against']}"] = rp
            ok &= rp["match"]
        if pop.design_log is not None:
            re77 = reconcile_design_log(pop, pop.design_log)
            res["reconciliation"][f"{pop.name}: pooled counts against design_log_summary.json"] = re77
            ok &= re77["match"]
        if pop.registered and "against_registered" in pc:
            ok &= pc["against_registered"]["all_equal"]
        if "registered_minus_dropped" in pc:
            ok &= pc["registered_minus_dropped"]["all_equal"]
        if "against_judge_calibration" in pc:
            ok &= pc["against_judge_calibration"]["equal"]
        if cp.get("judge") and cp["judge"].get("tables"):
            tp = resolve(base, cp["judge"]["tables"])
            shas[str(tp)] = sha256_file(tp)
            tables = read_json(tp)
            rj = reconcile_judge(pop, tables)
            res["reconciliation"][f"{pop.name}: judge counts against tables.json"] = rj
            ok &= rj["match"]
            rn = reconcile_nonempty(pop, pr, tables)
            res["reconciliation"][f"{pop.name}: non-empty denominators against tables.json table 2"] = rn
            ok &= rn["match"]
    if bank:
        rb = reconcile_bank(bank)
        res["reconciliation"]["dev bank: counts against the bank run's bank_rates"] = rb
        ok &= rb["match"]
        for pname, pr_ in res["populations"].items():
            rpb = reconcile_p1_bank(pr_, bank)
            if rpb["compared"]:
                res["reconciliation"][f"{pname}: P1 bank columns and ratios against bank_rates"] = rpb
                ok &= rpb["match"]
        res["bank"] = {"graph": bank["graph"], "arms": sorted(bank["counts"]), "source_digests": bank["digests"],
                       "code": bank["plan_code"], "items": bank["items"]}
    fs = cfg.get("frozen_sources")
    if fs:
        ip = resolve(base, fs["inputs"])
        shas[str(ip)] = sha256_file(ip)
        fdir = resolve(base, fs["dir"])
        if fdir.is_dir():   # the frozen tree is present: the comparison is as strict as before (any source that differs fails the run)
            chk = frozen_source_check(read_json(ip), fdir)
            ok &= all(e["equal"] for e in chk)
        else:   # the unreleased sources are not part of the package: this one comparison is reported as not checked and is not a failure
            chk = NOT_CHECKED
        res["reconciliation"]["source files against the frozen tree"] = chk
        res["frozen_sources_dir"] = fs["dir"]
    res["inputs_sha256"] = dict(sorted(shas.items()))
    res["result"] = "OK" if ok else "FAIL"
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        res = run(cfg, args.config.resolve().parent)
    except (InputError, KeyError, ValueError) as e:
        print(f"INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    res["tool"]["sha256"] = sha256_file(Path(__file__))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "p1p5-dev.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "p1p5-dev.txt").write_text(render_text(res), encoding="utf-8")
    for k, v in res["reconciliation"].items():
        if isinstance(v, dict):
            print(f"{k}: compared {v['compared']}, mismatches {v['mismatches']}")
    print(f"RESULT {res['result']}")
    return 0 if res["result"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
