#!/usr/bin/env python3
"""The re-validation of scorer-rules-v6 on dev (change-log entry 127), the same way as entry 120's check: every dev
record rescored, dev only.

    python -I scorer_entry127_check.py rescore --code ROOT --sources SOURCES.json --registered V5_RESCORE.jsonl.gz \\
        --judged AUDIT_COMPARISON.jsonl.gz --judged CHANGED.jsonl.gz --out OUT
    python -I scorer_entry127_check.py items --sources SOURCES.json --audit-code DIR --out OUT
    python -I scorer_entry127_check.py report --out OUT

``rescore`` reads the dev runs named in SOURCES.json (as scorer_entry120_check reads them: an archive is opened only
after its run's manifest says split dev and names the records' question set) and scores each record through the
runner's own path (loop.score_outcome against harness.score.question_gold) with the code under ``--code`` (its frozen
pin set to the candidate's digest in this process only):

* v4 (every v5 rule off), which must reproduce each record's stored score exactly;
* v5 as registered (the v6 kind rule off), which must equal the registered v5 rescoring (``--registered``: the
  entry-120 check's rescore.jsonl.gz, made by the v5 scorer 3d6b1e6a; exact flag, F1 and notes);
* v6 (the rules as they now read).

A verdict "changes" when v6's exact flag differs from v5's. A v5 rescue is a record v4 scores wrong and v5 exact: it
is kept when v6 scores it exact and lost otherwise. ``--judged`` files name the judge's earlier verdicts (prompt p2) by
run id (the audit's comparison rows' ``judge``, the entry-120 check's changed rows' ``verdict``); ``items`` builds a
judge input, exactly as the audit does, for every changed verdict the judge has not ruled on. ``report`` writes the
counts: v6 against v5 and v4 per model and graph, the rescues kept and lost, and the changes' verdicts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorer_entry120_check as base  # noqa: E402  (the same readers: dev sources, records, the code import)

write_jsonl_gz, read_jsonl_gz = base.write_jsonl_gz, base.read_jsonl_gz


def judged_verdicts(paths: list[Path]) -> dict[str, str]:
    """run id -> the judge's verdict from earlier p2 runs (later files win; "yes (audit)" reads as yes)."""
    out: dict[str, str] = {}
    for path in paths:
        for row in read_jsonl_gz(path):
            verdict = row.get("verdict", row.get("judge"))
            if verdict in ("yes", "no", "unclear", "yes (audit)"):
                out[row["run_id"]] = "yes" if verdict == "yes (audit)" else verdict
    return out


def flipping_rules(paths: list[Path]) -> dict[str, str]:
    """run id -> the v5 rule that alone flips its verdict against v4 (the entry-120 check's changed rows' ``rule``)."""
    return {row["run_id"]: row["rule"] for path in paths for row in read_jsonl_gz(path) if row.get("rule")}


def rules_read_back(ScoringRules, stored) -> bool:
    """Whether a stored score's rules object reads back (from_json) as to_json writes it: every version refuses a field
    it would not write (External review of 13741b8), so a stored object it refuses could not be reproduced."""
    try:
        return ScoringRules.from_json(stored).to_json() == stored
    except (KeyError, TypeError, ValueError):
        return False


def cmd_rescore(a: argparse.Namespace) -> int:
    loop, ScoringRules, question_gold, candidate = base.import_code(a.code)
    registered = {r["run_id"]: r for r in read_jsonl_gz(a.registered)}
    judged = judged_verdicts(a.judged or [])
    flipped_by = flipping_rules(a.judged or [])
    rows, mismatches = [], []
    for src, questions in base.dev_sources(a.sources):
        for name, rec in base.iter_records(src["tars"]):
            outcome, score = rec.get("outcome") or {}, rec.get("score")
            if outcome.get("failure_type") in base.UNSCORED_FAILURES or not isinstance(score, dict):
                continue
            q = questions[rec["question_id"]]
            if (rec.get("transcript") or [{}])[0].get("text") != q["question"]:
                raise SystemExit(f"{name}: the first message is not the question file's text")
            rules, gold = ScoringRules(**q["scoring_rules"]), question_gold(q)
            v4 = loop.score_outcome(outcome, rec.get("final_rows"), gold, rules.as_v4().to_json())
            v5 = loop.score_outcome(outcome, rec.get("final_rows"), gold,
                                    replace(rules, extra_columns_by_kind=False).to_json())
            v6 = loop.score_outcome(outcome, rec.get("final_rows"), gold, rules.to_json())
            reg = registered.get(rec["run_id"])
            if v4 != score:
                mismatches.append({"record": name, "what": "v4", "recorded": score, "now": v4})
            if reg is None or (reg["v5_exact"], reg["v5_f1"], reg["v5_notes"]) != (v5["exact"], v5["f1"], v5["notes"]):
                mismatches.append({"record": name, "what": "v5", "registered": reg and {k: reg[k] for k in (
                    "v5_exact", "v5_f1", "v5_notes")}, "now": {k: v5[k] for k in ("exact", "f1", "notes")}})
            rows.append({
                "graph": src["graph"], "model": base.ROUTE_MODELS.get(rec["model"], rec["model"]),
                "run_id": rec["run_id"], "record": name, "question_id": rec["question_id"],
                "status": outcome.get("status"), "v4_exact": v4["exact"], "v5_exact": v5["exact"],
                "v6_exact": v6["exact"], "v4_f1": v4["f1"], "v5_f1": v5["f1"], "v6_f1": v6["f1"], "v6_notes": v6["notes"],
                "v4_reproduces": v4 == score, "v5_matches_registered": reg is not None and (
                    reg["v5_exact"], reg["v5_f1"], reg["v5_notes"]) == (v5["exact"], v5["f1"], v5["notes"]),
                "changed_vs_v5": v6["exact"] != v5["exact"], "judged": judged.get(rec["run_id"]),
                "rule": flipped_by.get(rec["run_id"]), "stored_rules_read_back": rules_read_back(ScoringRules, score.get(
                    "rules")),
            })
    a.out.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(a.out / "rescore.jsonl.gz", rows)
    write_jsonl_gz(a.out / "mismatches.jsonl.gz", mismatches)
    rescues = [r for r in rows if r["v5_exact"] and not r["v4_exact"]]
    summary = {
        "candidate_scorer_digest": candidate, "code": str(a.code), "records": len(rows),
        "v4_reproduces_recorded": sum(r["v4_reproduces"] for r in rows),
        "stored_rules_read_back": sum(r["stored_rules_read_back"] for r in rows),
        "accepted_reports": sum(r["status"] == "submitted" for r in rows),
        "accepted_reports_v4_reproduced": sum(r["status"] == "submitted" and r["v4_reproduces"] for r in rows),
        "v5_matches_registered": sum(r["v5_matches_registered"] for r in rows),
        "changed_vs_v5": sum(r["changed_vs_v5"] for r in rows),
        "v6_exact_where_v5_wrong": sum(r["v6_exact"] and not r["v5_exact"] for r in rows),
        "v6_against_v4": {"gained": sum(r["v6_exact"] and not r["v4_exact"] for r in rows),
                          "lost": sum(r["v4_exact"] and not r["v6_exact"] for r in rows)},
        "v5_rescues": len(rescues), "v5_rescues_kept": sum(r["v6_exact"] for r in rescues),
        "v5_rescues_lost": sum(not r["v6_exact"] for r in rescues),
        "f1_lowered_vs_v4": sum(r["v6_f1"] < r["v4_f1"] for r in rows),
    }
    (a.out / "rescore-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def cmd_items(a: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(a.audit_code).resolve()))
    import scorer_audit as sa

    rows = read_jsonl_gz(a.out / "rescore.jsonl.gz")
    wanted = {r["run_id"]: r for r in rows if r["changed_vs_v5"] and r["judged"] is None}
    nonce, items = secrets.token_hex(8), []
    for src, questions in base.dev_sources(a.sources):
        for name, rec in base.iter_records(src["tars"]):
            row = wanted.get(rec["run_id"])
            if row is None or row["graph"] != src["graph"]:
                continue
            view = sa.final_output(rec)
            if not view["paired"]:
                raise SystemExit(f"{name}: its calls do not pair with its transcript")
            text = sa.render_item(questions[rec["question_id"]], view)
            cid = sa.custom_id(nonce, src["graph"], rec["run_id"])
            items.append({**{k: row[k] for k in ("graph", "model", "run_id", "question_id", "status")},
                          "klass": "wrong" if not row["v5_exact"] else "exact",
                          "label": sa.outcome_label(rec.get("outcome") or {}), "exact": row["v5_exact"],
                          "f1": row["v5_f1"], "custom_id": cid, "view": view, "user_text": text,
                          "user_sha256": sa.sha256_text(text), "order": sa.sha256_text(f"{a.seed}|{cid}")})
    if len(items) != len(wanted):
        raise SystemExit(f"{len(wanted)} changed records wanted, {len(items)} found")
    items.sort(key=lambda it: it["order"])
    write_jsonl_gz(a.out / "items.jsonl.gz", items)
    ids = "".join(sorted(f"{it['graph']}\t{it['model']}\t{it['run_id']}\n" for it in items))
    (a.out / "items-ids.txt").write_text(ids)
    (a.out / "items.json").write_text(json.dumps({"nonce": nonce, "seed": a.seed, "items": len(items),
                                                  "ids_sha256": hashlib.sha256(ids.encode()).hexdigest()},
                                                 indent=1) + "\n")
    print(f"{len(items)} judge items")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    rows = read_jsonl_gz(a.out / "rescore.jsonl.gz")
    new: dict[str, str] = {}
    if (a.out / "calls.jsonl").exists() and (a.out / "items.jsonl.gz").exists():
        by_cid = {it["custom_id"]: it["run_id"] for it in read_jsonl_gz(a.out / "items.jsonl.gz")}
        for line in open(a.out / "calls.jsonl", encoding="utf-8"):
            c = json.loads(line)
            if c.get("prompt_id") == "p2" and "verdict" in c:
                new[by_cid[c["custom_id"]]] = c["verdict"]["semantically_correct"]
    per_model: dict[tuple[str, str], Counter] = defaultdict(Counter)
    verdicts: Counter = Counter()
    rescued: dict[str, Counter] = defaultdict(Counter)  # v6's rescues against v4, by the rule that alone flips each
    for r in rows:
        c = per_model[(r["graph"], r["model"])]
        c["records"] += 1
        c["v4"] += r["v4_exact"]
        c["v5"] += r["v5_exact"]
        c["v6"] += r["v6_exact"]
        if r["v5_exact"] and not r["v4_exact"]:
            c["kept" if r["v6_exact"] else "lost"] += 1
        if r["v6_exact"] and not r["v4_exact"]:
            rescued[r.get("rule") or "unattributed"][r["judged"] or new.get(r["run_id"]) or "not judged"] += 1
        if r["changed_vs_v5"]:
            verdict = r["judged"] or new.get(r["run_id"]) or "not judged"
            verdicts[("v5 exact, v6 wrong" if r["v5_exact"] else "v5 wrong, v6 exact", verdict)] += 1
    lines = ["| graph | model | records | exact v4 | exact v5 (registered) | exact v6 | v5 rescues kept | v5 rescues lost |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (graph, model), c in sorted(per_model.items()):
        lines.append(f"| {graph} | {model} | {c['records']} | {c['v4']} | {c['v5']} | {c['v6']} | {c['kept']} | "
                     f"{c['lost']} |")
    lines += ["", "| change against v5 | judge (p2) | records |", "|---|---|---:|"]
    lines += [f"| {k[0]} | {k[1]} | {n} |" for k, n in sorted(verdicts.items())]
    # the false rate as entry 120 reported it: the judge's "no" over every rescue in the row, judged or not
    lines += ["", ("| rule (alone flips it) | v6 rescues against v4 | judge yes | unclear | no | not judged | "
                   "false rate (no) |"), "|---|---:|---:|---:|---:|---:|---:|"]
    for rule, c in sorted(rescued.items()):
        rate = f"{100 * c['no'] / sum(c.values()):.1f}%"
        lines.append(f"| {rule} | {sum(c.values())} | {c['yes']} | {c['unclear']} | {c['no']} | {c['not judged']} | "
                     f"{rate} |")
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("rescore")
    s.add_argument("--code", required=True, type=Path)
    s.add_argument("--sources", required=True, type=Path)
    s.add_argument("--registered", required=True, type=Path)
    s.add_argument("--judged", action="append", type=Path)
    s.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("items")
    s.add_argument("--sources", required=True, type=Path)
    s.add_argument("--audit-code", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--seed", type=int, default=20261021)
    s = sub.add_parser("report")
    s.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    return {"rescore": cmd_rescore, "items": cmd_items, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
