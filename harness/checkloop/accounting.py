"""Accounting per output (design log, AC7; proposal §7.6, row "Accounting").

Each initial output is right (C) or wrong (W) by the scorer, and flagged (F),
passed (P: substantive or vacuous pass, reported split) or unknown (U) at output
level under contract v2's four-way status. The six counts sum to N_out, the
outputs that were checked; N_att adds the attempts with no output (failures and
abstentions), which count as wrong in accuracy but sit outside the matrix.

For retried outputs: fixed (W -> C), corrupted (C -> W), still wrong, still
right, all by the scorer; a retry that ends with no answer (abstained, failed,
capped) is wrong. "The retry changed the answer" is never "fixed".

Every ratio whose denominator is zero is reported as "not estimable", never 0.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

NOT_ESTIMABLE = "not estimable"
CELLS = ("C_F", "C_P", "C_U", "W_F", "W_P", "W_U")
_LETTER = {"flagged": "F", "substantive_pass": "P", "vacuous_pass": "P", "unknown": "U"}


@dataclass
class OutputRow:
    run_id: str
    correct: bool | None  # the initial answer by the scorer; None with no output
    status: str | None  # the arm's output status; None with no output
    output: bool = True  # the attempt produced an output that was checked (counts in N_out)
    checked_query_status: str = "given"  # aligned | fallback | none
    template: str | None = None
    negation: bool = False
    any_unknown: bool = False
    unsupported: tuple[str, ...] = ()
    no_query: bool = False  # no executed query: unknown, never retried (rule 22, entry 30)
    retried: bool = False
    retry_correct: bool | None = None  # the retried answer by the scorer; None when the retry gave no answer
    retry_changed: bool = False  # for audit only: never decides "fixed"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["unsupported"] = list(self.unsupported)
        return out


def _ratio(num: float, den: float) -> float | str:
    return num / den if den else NOT_ESTIMABLE


def _cells(rows: Sequence[OutputRow]) -> dict[str, Any]:
    outputs = [r for r in rows if r.output]
    cells = dict.fromkeys(CELLS, 0)
    passes = {"C_P_substantive": 0, "C_P_vacuous": 0, "W_P_substantive": 0, "W_P_vacuous": 0}
    for r in outputs:
        if r.correct is None or r.status not in _LETTER:
            raise ValueError(f"output {r.run_id} has no scorer label or no status: it cannot be placed in a cell")
        cw = "C" if r.correct else "W"
        cells[f"{cw}_{_LETTER[r.status]}"] += 1
        if r.status in ("substantive_pass", "vacuous_pass"):
            passes[f"{cw}_P_{r.status.split('_')[0]}"] += 1
    return {"N_att": len(rows), "N_out": len(outputs), "cells": cells, "passes": passes,
            "no_query": sum(1 for r in outputs if r.no_query)}


def account(rows: Iterable[OutputRow], strata: Sequence[str] = ()) -> dict[str, Any]:
    rows = list(rows)
    out = _cells(rows)
    c = out["cells"]
    retried = [r for r in rows if r.output and r.retried]
    fixed = sum(1 for r in retried if not r.correct and r.retry_correct)
    corrupted = sum(1 for r in retried if r.correct and not r.retry_correct)
    out["retry"] = {
        "retried": len(retried),
        "fixed": fixed,
        "corrupted": corrupted,
        "still_wrong": sum(1 for r in retried if not r.correct and not r.retry_correct),
        "still_right": sum(1 for r in retried if r.correct and r.retry_correct),
        "retry_without_answer": sum(1 for r in retried if r.retry_correct is None),
    }
    final = sum(1 for r in rows if (r.retry_correct if r.retried else r.correct))
    initial = sum(1 for r in rows if r.output and r.correct)
    out["accuracy"] = {"initial": _ratio(initial, len(rows)), "end_to_end": _ratio(final, len(rows)),
                       "change": _ratio(final - initial, len(rows))}
    out["rates"] = {
        "flag_precision": _ratio(c["W_F"], c["W_F"] + c["C_F"]),
        "false_flag_rate": _ratio(c["C_F"], c["C_F"] + c["C_P"] + c["C_U"]),
        "detection_recall": _ratio(c["W_F"], c["W_F"] + c["W_P"] + c["W_U"]),
        "error_among_certified_passes": _ratio(c["W_P"], c["W_P"] + c["C_P"]),
        "error_not_intervened": _ratio(c["W_P"] + c["W_U"], c["W_P"] + c["C_P"] + c["W_U"] + c["C_U"]),
        "unknown_rate": _ratio(c["C_U"] + c["W_U"], out["N_out"]),
        "fix_rate": _ratio(fixed, sum(1 for r in retried if not r.correct)),
        "corruption_rate": _ratio(corrupted, sum(1 for r in retried if r.correct)),
        "repair_yield": _ratio(fixed - corrupted, out["N_out"]) if retried else NOT_ESTIMABLE,
    }
    if strata:
        out["strata"] = {}
        for key in strata:
            groups: dict[str, list[OutputRow]] = defaultdict(list)
            for r in rows:
                value = getattr(r, key) if hasattr(r, key) else r.extra.get(key)
                values = list(value) if isinstance(value, (list, tuple)) else [value]
                for v in values or ["none"]:
                    groups[str(v)].append(r)
            out["strata"][key] = {k: _cells(v) for k, v in sorted(groups.items())}
    return out


def online_row(rec: dict) -> OutputRow:
    """The accounting row of one checker-cell record (agent-run/7): the initial answer's
    cell, and for a retried one the retried answer by the scorer. A retry that ends
    without an answer (abstained, failed, capped) is wrong; a record whose retry failed
    on infrastructure must be re-run, not counted."""
    chk = rec.get("checker")
    if not chk:
        raise ValueError(f"record {rec.get('run_id')} is not a checker cell's record")
    initial, retry = chk["initial"], chk["retry"]
    output = initial["outcome"]["status"] == "submitted"
    v = initial.get("verdict") or {}
    row = OutputRow(run_id=rec["run_id"], correct=(initial.get("score") or {}).get("exact") if output else None,
                    status=v.get("status") if output else None, output=output,
                    checked_query_status=initial.get("checked_query_status") or "none", template=rec.get("template"),
                    any_unknown=bool(v.get("any_unknown")), no_query=output and v.get("reason") == "no_query")
    if retry is not None:
        if rec["outcome"].get("failure_type") == "infrastructure":
            raise ValueError(f"record {rec['run_id']}: the retry failed on infrastructure; re-run it")
        row.retried = True
        answered = rec["outcome"]["status"] == "submitted"
        row.retry_correct = bool((rec.get("score") or {}).get("exact")) if answered else None
        row.retry_changed = not answered or rec.get("final_rows") != initial.get("final_rows")
    return row
