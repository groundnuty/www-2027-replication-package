#!/usr/bin/env python3
"""Re-read and re-score the stored probe (a) replies of a closed-book probe run under another answer
format (design log): no model call, no store.

    uv run python scripts/rescore_probes.py RUN_DIR --out OUT [--graph G ...] [--answer-format rows-json-v2]
        [--code-note TEXT]

RUN_DIR is a run_probes.py ``--out`` directory (one ``<graph>/`` per graph). For each graph, every
probe (a) record in ``<graph>/records/`` is

1. rebuilt under the answer format it was made with (replay.replay_probe, checked): the current code
   must reproduce its stored outcome, answer and score exactly, so the "before" figures (read from the
   stored records) and the "after" figures differ by the answer format alone. One record that does not
   rebuild stops the graph before anything is written (exit 3);
2. read again under ``--answer-format`` from its stored reply and re-scored against the gold rows it
   carries (replay.reparse_answer).

Written to ``OUT/<graph>/`` (refused, exit 2, when it exists; the run directory is only read):

* ``rescore.jsonl.gz``: one line per probe (a) record, keyed by the record's ``run_id``, with the
  record's path and sha256, the stored label, decline and score ("before"), and the new answer
  entry, outcome, final rows and score ("after");
* ``rescore-manifest.json``: the answer format and decline rule, the sha256 of the record set read
  (the replay check's ``records_sha256``), the code in use, the counts;
* ``summary.json``: per model, over the attempts (``failed:infrastructure`` and ``failed:harness``
  are outside them, as in probe_summary.py): parsed (label ``answered``, empty rows included), exact,
  declined (rule ``declined-v1``) and unparseable (label ``unparseable``, phrase declines included),
  before and after; the exact rate clustered by base question with its 95% bootstrap interval
  (harness/pilot_report.py, as probe_summary.py); the answers parsed only by the new format's
  leniency (form ``embedded_json``), those parsed although cut by the output cap, and the
  unparseable replies holding a brace and a decline phrase (rule ``declined-v1`` never calls them
  declines); and the probe (a) replies still waiting in ``replies/`` for their record (no gold
  there: parsed and declined only).

Probe (b) records are not read: its query extraction is unchanged.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agent.providers import OUTPUT_LIMIT_STOPS  # noqa: E402
from harness.agent.record import utc_now  # noqa: E402
from harness.agent.replay_check import code_commit, current_records_sha256  # noqa: E402
from harness.pilot_report import bootstrap_interval, per_question_means  # noqa: E402
from harness.probe.parse import (  # noqa: E402
    ANSWER_FORMAT,
    ANSWER_FORMATS,
    DECLINE_PATTERNS,
    DECLINE_RULE,
    declined,
    parse_answer,
)
from harness.probe.record import ProbeRecord  # noqa: E402
from harness.probe.replay import replay_probe, reparse_answer  # noqa: E402
from harness.probe.run import Reply, step_from_reply  # noqa: E402

UNSCORED = ("infrastructure", "harness")
MANIFEST_SCHEMA = "probe-rescore/1"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _view(outcome: dict, answer: dict | None, score: dict | None) -> dict:
    return {"label": outcome["label"], "declined": (answer or {}).get("declined"), "form": (answer or {}).get("form"),
            "exact": (score or {}).get("exact"), "f1": (score or {}).get("f1")}


def _counts(views: list[dict]) -> dict:
    return {"parsed": sum(v["label"] == "answered" for v in views), "exact": sum(v["exact"] is True for v in views),
            "declined": sum(v["declined"] is not None for v in views),
            "unparseable": sum(v["label"] == "unparseable" for v in views)}


def _exact_rate(att: list[dict], views: list[dict]) -> tuple[float | None, tuple | None]:
    """The exact rate clustered by base question (each one's attempts averaged first), with its
    95% percentile-bootstrap interval over base questions (probe_summary.py's method)."""
    means = per_question_means([{"base_question_id": ln["base_question_id"], "exact": v["exact"]}
                                for ln, v in zip(att, views)], lambda r: float(r["exact"] is True))
    values = list(means.values())
    return (statistics.fmean(values) if values else None), bootstrap_interval(values)


def summarize(lines: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for ln in lines:
        by_model[ln["model"]].append(ln)
    out = {}
    for model in sorted(by_model):
        group = by_model[model]
        att = [ln for ln in group if ln["failure_type"] not in UNSCORED]
        before, after = [ln["before"] for ln in att], [_view(**ln["after_view"]) for ln in att]
        rates = {"before": _exact_rate(att, before), "after": _exact_rate(att, after)}
        out[model] = {
            "attempts": len(att),
            "infra_failures": sum(ln["failure_type"] == "infrastructure" for ln in group),
            "harness_failures": sum(ln["failure_type"] == "harness" for ln in group),
            "changed": sum(ln["changed"] for ln in att),
            "before": _counts(before), "after": _counts(after),
            "before_declined_by_rule": dict(sorted(Counter(v["declined"] for v in before if v["declined"]).items())),
            "after_declined_by_rule": dict(sorted(Counter(v["declined"] for v in after if v["declined"]).items())),
            "after_forms": dict(sorted(Counter(v["form"] for v in after
                                               if v["label"] == "answered" and v["form"] != "json").items())),
            "after_parsed_though_truncated": sum(ln["truncated"] and v["label"] == "answered"
                                                 for ln, v in zip(att, after)),
            "after_unparseable_brace_with_decline_phrase": sum(ln["brace_phrase"] and v["label"] == "unparseable"
                                                               and v["declined"] is None for ln, v in zip(att, after)),
            "exact_rate": {side: rates[side][0] for side in rates},
            "exact_ci95": {side: rates[side][1] for side in rates},
        }
    return out


def waiting_replies(graph_dir: Path, done: set[tuple], answer_format: str, iri_rule: str) -> dict:
    """Probe (a) replies kept in replies/ with no record yet: parsed and declined under the records'
    format (rows-json-v1 here) and under ``answer_format``."""
    out: dict[str, dict] = {}
    for path in sorted((graph_dir / "replies").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        _, probe_id, qid, rep = d["attempt"]
        model = d["route"]["model"]
        if not probe_id.startswith("closed-book-answer") or (qid, model, probe_id, rep) in done:
            continue
        c = out.setdefault(model, {"n": 0, "before": {"parsed": 0, "declined": 0},
                                   "after": {"parsed": 0, "declined": 0}})
        c["n"] += 1
        reply = Reply.from_json(d["reply"])
        step = step_from_reply(reply, d["route"]["provider"])
        if reply.provider_failure is not None or step is None or step.refusal is not None:
            continue
        truncated = step.stop_reason in OUTPUT_LIMIT_STOPS
        for side, fmt in (("before", "rows-json-v1"), ("after", answer_format)):
            rows, _, _ = parse_answer(step.text, iri_rule, fmt)
            c[side]["parsed"] += rows is not None
            c[side]["declined"] += declined(step.text, rows, truncated) is not None
    return out


def rescore_graph(graph_dir: Path, out: Path, answer_format: str, code_note: str | None,
                  say=print) -> int:
    if out.exists():
        say(f"error: {out} exists; a re-score is never written over")
        return 2
    lines, done, mismatches, iri_rules = [], set(), [], set()
    for path in sorted((graph_dir / "records").glob("*.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        done.add((data["question_id"], data["model"], data["prompt"]["id"], data["repetition"]))
        if data["probe"] != "answer":
            continue
        rel = path.relative_to(graph_dir).as_posix()
        try:
            replay_probe(data, check=True)
        except Exception as exc:  # noqa: BLE001 - every record is checked, every failure reported
            mismatches.append(f"{rel}: {type(exc).__name__}: {str(exc)[:300]}")
            continue
        rec = ProbeRecord.from_json(data)
        iri_rules.add(rec.config["iri_rule"])
        text = next((t.get("text") or "" for t in rec.transcript if t.get("role") == "assistant"), "")
        new = reparse_answer(rec, answer_format)
        before = _view(rec.outcome, rec.answer, rec.score)
        after_view = {"outcome": new.outcome, "answer": new.answer, "score": new.score}
        lines.append({
            "run_id": rec.run_id, "record": rel, "record_sha256": _sha256(raw), "model": rec.model,
            "question_id": rec.question_id, "base_question_id": rec.base_question_id,
            "repetition": rec.repetition, "failure_type": rec.outcome["failure_type"],
            "truncated": bool(rec.outcome.get("truncated")),
            # the case declined-v1 leaves out: a brace in the reply and a decline phrase (a science reading)
            "brace_phrase": "{" in text and any(pattern.search(text) for _, pattern in DECLINE_PATTERNS),
            "answer_format": {"before": rec.config["answer_format"], "after": answer_format},
            "changed": _view(**after_view) != before, "before": before, "after_view": after_view,
            "after": {"answer": new.answer, "outcome": new.outcome, "final_rows": new.final_rows,
                      "score": new.score},
        })
    if mismatches:
        say(f"error: {graph_dir.name}: {len(mismatches)} probe (a) record(s) do not rebuild under their own "
            f"answer format; nothing written:")
        for m in mismatches[:20]:
            say(f"  {m}")
        return 3
    summary = summarize(lines)
    waiting = waiting_replies(graph_dir, done, answer_format, next(iter(iri_rules), "registered_schemes_v3"))
    out.mkdir(parents=True)
    with gzip.open(out / "rescore.jsonl.gz", "wt", encoding="utf-8") as fh:
        for ln in lines:
            fh.write(json.dumps({k: v for k, v in ln.items() if k != "after_view"}, ensure_ascii=False) + "\n")
    manifest = {
        "schema": MANIFEST_SCHEMA, "created_at": utc_now(), "graph": graph_dir.name,
        "source": str(graph_dir), "answer_format": answer_format, "decline_rule": DECLINE_RULE,
        "source_records_sha256": current_records_sha256(graph_dir),
        "replay_check_records_sha256": _replay_check_sha(graph_dir),
        "n_rescored": len(lines), "v1_rebuild": {"n": len(lines), "n_ok": len(lines)},
        "code": code_commit(), "code_note": code_note,
        "files": {"rescore.jsonl.gz": _sha256((out / "rescore.jsonl.gz").read_bytes())},
    }
    (out / "rescore-manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"graph": graph_dir.name, "answer_format": answer_format,
                                                  "models": summary, "replies_without_record": waiting},
                                                 indent=1), encoding="utf-8")
    for model, s in summary.items():
        say(f"{graph_dir.name} {model}: attempts {s['attempts']} parsed {s['before']['parsed']}->{s['after']['parsed']}"
            f" exact {s['before']['exact']}->{s['after']['exact']} declined {s['before']['declined']}->"
            f"{s['after']['declined']} (changed {s['changed']})")
    return 0


def _replay_check_sha(graph_dir: Path) -> str | None:
    path = graph_dir / "replay-check.json"
    return json.loads(path.read_text(encoding="utf-8")).get("records_sha256") if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--graph", nargs="+", default=None)
    p.add_argument("--answer-format", choices=ANSWER_FORMATS, default=ANSWER_FORMAT)
    p.add_argument("--code-note", default=None, help="how the code in use relates to a commit (recorded)")
    args = p.parse_args(argv)
    graphs = args.graph or sorted(d.name for d in args.run_dir.iterdir() if (d / "records").is_dir())
    rc = 0
    for g in graphs:
        rc = max(rc, rescore_graph(args.run_dir / g, args.out / g, args.answer_format, args.code_note))
    return rc


if __name__ == "__main__":
    sys.exit(main())
