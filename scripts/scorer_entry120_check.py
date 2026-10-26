#!/usr/bin/env python3
"""The out-of-sample check of scorer-rules-v5 before it is registered (change-log entry 120; lead's checks (a),
(b), (d)): every dev record rescored, dev only.

    python -I scorer_entry120_check.py rescore --code ROOT --sources SOURCES.json --audit AUDIT_OUT --out OUT
    python -I scorer_entry120_check.py items   --sources SOURCES.json --audit-code DIR --out OUT
    python -I scorer_entry120_check.py report  --out OUT --audit AUDIT_OUT

``rescore`` reads every record of the dev runs named in SOURCES.json (an archive is opened only after its run's
manifest says split dev and names the records' question set; a question line is parsed only when it carries the dev
mark) and scores it through ``harness.agent.loop.score_outcome``, the runner's own path, with the code under ``--code``
(the candidate scorer; its digest is recorded, and the loop's frozen-digest pin is set to it in this process only,
since the pin moves in a separate commit once the check is accepted):

* v4: the question's rules with every v5 rule off. It must reproduce each record's recorded score exactly (the whole
  score object), which shows that the v4 semantics are untouched and that the plain reading of every accepted report
  is the recorded one (check (b));
* v5: the question's rules as they now read (every v5 rule on), against ``question_gold`` (the anchors' names);
* each v5 rule alone, to say which rule changed a verdict.

A verdict "changes" when the v5 score's exact flag differs from the recorded one. ``items`` builds, for every changed
record the scorer audit's judge did not already call correct (AUDIT_OUT/comparison.jsonl.gz, prompt p2), a judge input
exactly as the audit does (scorer_audit.final_output and render_item from ``--audit-code``: same blinding), ready for
``scorer_audit.py run``. ``report`` joins the verdicts and writes the counts: per rule, the changed verdicts the judge
calls correct, unclear or wrong (in the audit's selection, and out of it), and the per-model changes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import secrets
import sys
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

UNSCORED_FAILURES = frozenset({"infrastructure", "harness_defect", "model_changed"})
RULES = ("extra_columns", "anchor_names", "temporal_strings")
ROUTE_MODELS = {
    "claude-sonnet-5-5": "claude-sonnet-5-5", "claude-haiku-5-5": "claude-haiku-5-5",
    "claude-opus-5-5": "claude-opus-5-5", "Qwen/Qwen3.8-27B": "qwen3.8-27b", "zai-org/GLM-5.3-Flash": "glm-5.3-flash",
    "google/gemma-4-31B": "gemma-4-31b", "Qwen/Qwen3-Coder-30B-A3B-Instruct": "qwen3-coder-30b-a3b",
    "deepseek-ai/DeepSeek-V4-Flash-0731": "deepseek-v4-flash-0731", "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
    "Qwen/Qwen3.6-35B-A3B": "qwen3.6-35b-a3b",
}
_DEV_MARK = re.compile(r'"split"\s*:\s*"dev"')


def write_jsonl_gz(path: Path, rows) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dev_questions(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() and _DEV_MARK.search(line):
                q = json.loads(line)
                if q.get("split") != "dev":
                    raise SystemExit(f"{path}: a dev-marked line of split {q.get('split')!r}")
                out[q["id"]] = q
    return out


def dev_sources(sources_path: Path) -> Iterator[tuple[dict, dict[str, dict]]]:
    for src in json.loads(Path(sources_path).read_text()):
        qpath = Path(src["questions"])
        if hashlib.sha256(qpath.read_bytes()).hexdigest() != src["questions_sha256"]:
            raise SystemExit(f"{qpath}: not the question set {src['questions_sha256']}")
        shared = json.loads(Path(src["manifest"]).read_text())["fingerprint"]["shared"]
        if shared["grid"]["split"] != "dev" or shared["grid"]["graph"] != src["graph"] or \
                shared["questions_sha256"] != src["questions_sha256"]:
            raise SystemExit(f"{src['manifest']}: not a dev run of {src['graph']} on {src['questions_sha256']}")
        yield src, load_dev_questions(qpath)


def iter_records(paths: Sequence[str]) -> Iterator[tuple[str, dict]]:
    for path in paths:
        with tarfile.open(path, "r:gz") as tf:
            for m in tf:
                if m.isfile() and m.name.startswith("records/") and m.name.endswith(".json"):
                    rec = json.load(tf.extractfile(m))
                    if rec.get("split") != "dev":
                        raise SystemExit(f"{path}: {m.name} is of split {rec.get('split')!r}")
                    yield m.name, rec


def import_code(code: Path):
    sys.path.insert(0, str(code.resolve()))
    from harness.agent import loop
    from harness.score import ScoringRules, digest, question_gold

    if not Path(loop.__file__).resolve().is_relative_to(code.resolve()):
        raise SystemExit(f"imported {loop.__file__}, not the code under {code}")
    candidate = digest.scorer_digest(digest.ROOT)
    loop.FROZEN_SCORER_DIGEST = candidate  # the candidate scorer, in this process only (module doc)
    return loop, ScoringRules, question_gold, candidate


def cmd_rescore(a: argparse.Namespace) -> int:
    loop, ScoringRules, question_gold, candidate = import_code(a.code)
    audit = {r["run_id"]: r for r in read_jsonl_gz(Path(a.audit) / "comparison.jsonl.gz")}
    out_rows, mismatches = [], []
    for src, questions in dev_sources(a.sources):
        for name, rec in iter_records(src["tars"]):
            outcome, score = rec.get("outcome") or {}, rec.get("score")
            if outcome.get("failure_type") in UNSCORED_FAILURES or not isinstance(score, dict):
                continue
            q = questions[rec["question_id"]]
            if (rec.get("transcript") or [{}])[0].get("text") != q["question"]:
                raise SystemExit(f"{name}: the first message is not the question file's text")
            base = ScoringRules(**q["scoring_rules"])
            gold = question_gold(q)
            v4 = loop.score_outcome(outcome, rec.get("final_rows"), gold, base.as_v4().to_json())
            if v4 != score:
                mismatches.append({"record": name, "recorded": score, "v4": v4})
            v5 = loop.score_outcome(outcome, rec.get("final_rows"), gold, base.to_json())
            alone = {}
            for rule in RULES:
                only = ScoringRules(**{**q["scoring_rules"], **dict.fromkeys(RULES, False), rule: True})
                alone[rule] = loop.score_outcome(outcome, rec.get("final_rows"), gold, only.to_json())["exact"]
            judged = audit.get(rec["run_id"])
            out_rows.append({
                "graph": src["graph"], "model": ROUTE_MODELS.get(rec["model"], rec["model"]), "run_id": rec["run_id"],
                "record": name, "question_id": rec["question_id"], "status": outcome.get("status"),
                "recorded_exact": score["exact"], "recorded_f1": score["f1"], "v4_reproduces": v4 == score,
                "v5_exact": v5["exact"], "v5_f1": v5["f1"], "v5_notes": v5["notes"], "alone": alone,
                "changed": v5["exact"] != score["exact"], "f1_changed": v5["f1"] != score["f1"],
                "audit_judge": judged["judge"] if judged else None,
                "anchors": isinstance(gold, dict) and gold.get("rule") == "anchor_names",
            })
    a.out.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(a.out / "rescore.jsonl.gz", out_rows)
    write_jsonl_gz(a.out / "v4-mismatches.jsonl.gz", mismatches)
    summary = {
        "candidate_scorer_digest": candidate, "code": str(a.code), "records": len(out_rows),
        "v4_reproduces_recorded": sum(r["v4_reproduces"] for r in out_rows),
        "v4_mismatches": len(mismatches),
        "accepted_reports": sum(r["status"] == "submitted" for r in out_rows),
        "accepted_reports_reproduced": sum(r["status"] == "submitted" and r["v4_reproduces"] for r in out_rows),
        "changed_exact": sum(r["changed"] for r in out_rows),
        "exact_lost": sum(r["recorded_exact"] and not r["v5_exact"] for r in out_rows),
        "f1_changed_without_exact_change": sum(r["f1_changed"] and not r["changed"] for r in out_rows),
        "f1_lowered": sum(r["v5_f1"] < r["recorded_f1"] for r in out_rows),
        "questions_with_anchor_names": len({(r["graph"], r["question_id"]) for r in out_rows if r["anchors"]}),
    }
    (a.out / "rescore-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def attribution(row: dict) -> str:
    rules = [rule for rule in RULES if row["alone"][rule]]
    return "+".join(rules) if rules else "combined"


def cmd_items(a: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(a.audit_code).resolve()))
    import scorer_audit as sa

    rows = read_jsonl_gz(a.out / "rescore.jsonl.gz")
    wanted = {r["run_id"]: r for r in rows if r["changed"] and r["audit_judge"] != "yes"}
    nonce, items = secrets.token_hex(8), []
    for src, questions in dev_sources(a.sources):
        for name, rec in iter_records(src["tars"]):
            row = wanted.get(rec["run_id"])
            if row is None or row["graph"] != src["graph"]:
                continue
            view = sa.final_output(rec)
            if not view["paired"]:
                raise SystemExit(f"{name}: its calls do not pair with its transcript")
            q = questions[rec["question_id"]]
            text = sa.render_item(q, view)
            cid = sa.custom_id(nonce, src["graph"], rec["run_id"])
            items.append({**{k: row[k] for k in ("graph", "model", "run_id", "question_id", "status")},
                          "klass": "wrong", "label": sa.outcome_label(rec.get("outcome") or {}),
                          "exact": row["recorded_exact"], "f1": row["recorded_f1"], "rule": attribution(row),
                          "custom_id": cid, "view": view, "user_text": text, "user_sha256": sa.sha256_text(text),
                          "order": sa.sha256_text(f"{a.seed}|{cid}")})
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
    calls: dict[str, dict] = {}
    if (a.out / "calls.jsonl").exists():
        for line in open(a.out / "calls.jsonl", encoding="utf-8"):
            c = json.loads(line)
            if c.get("prompt_id") == "p2" and ("verdict" in c or c["custom_id"] not in calls):
                calls[c["custom_id"]] = c
    items = {it["run_id"]: it for it in read_jsonl_gz(a.out / "items.jsonl.gz")} if \
        (a.out / "items.jsonl.gz").exists() else {}
    per_rule: dict[tuple[str, str], Counter] = defaultdict(Counter)
    per_model: dict[tuple[str, str], Counter] = defaultdict(Counter)
    joined = []
    for r in rows:
        per_model[(r["graph"], r["model"])]["records"] += 1
        per_model[(r["graph"], r["model"])]["recorded_exact"] += r["recorded_exact"]
        per_model[(r["graph"], r["model"])]["v5_exact"] += r["v5_exact"]
        if not r["changed"]:
            continue
        in_audit = r["audit_judge"] is not None
        if r["audit_judge"] == "yes":
            verdict = "yes (audit)"
        else:
            it = items.get(r["run_id"])
            call = calls.get(it["custom_id"]) if it else None
            verdict = (call or {}).get("verdict", {}).get("semantically_correct") or "not judged"
        rule = attribution(r)
        per_rule[(rule, "in audit selection" if in_audit else "out of sample")][verdict] += 1
        joined.append({**r, "rule": rule, "verdict": verdict, "in_audit": in_audit})
    write_jsonl_gz(a.out / "changed.jsonl.gz", joined)
    lines = ["| rule (alone flips it) | records | changed | judge yes | unclear | no | not judged | false rate (no) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (rule, where), c in sorted(per_rule.items()):
        yes = c["yes"] + c["yes (audit)"]
        n = sum(c.values())
        lines.append(f"| {rule} | {where} | {n} | {yes} | {c['unclear']} | {c['no']} | {c['not judged']} | "
                     f"{c['no'] / n:.1%} |")
    lines += ["", "| graph | model | records | exact (recorded, v4) | exact (v5) | change |", "|---|---|---:|---:|---:|---:|"]
    for (graph, model), c in sorted(per_model.items()):
        lines.append(f"| {graph} | {model} | {c['records']} | {c['recorded_exact']} | {c['v5_exact']} | "
                     f"{c['v5_exact'] - c['recorded_exact']:+d} |")
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("rescore")
    s.add_argument("--code", required=True, type=Path)
    s.add_argument("--sources", required=True, type=Path)
    s.add_argument("--audit", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("items")
    s.add_argument("--sources", required=True, type=Path)
    s.add_argument("--audit-code", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--seed", type=int, default=20261021)
    s = sub.add_parser("report")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--audit", required=True, type=Path)
    a = p.parse_args(argv)
    return {"rescore": cmd_rescore, "items": cmd_items, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
