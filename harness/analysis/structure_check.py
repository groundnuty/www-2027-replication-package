"""Compare a real bank's source-free structure with the structure the calibration ran on (design ruling, design log,
after change-log entry 41).

The calibration first runs on the 210-question bank's per-question structure, resampled within template x tier strata
and scaled to N. When the real bank-only extension exists, its structure is compared with that one. It must be
recalibrated at the real structure if ANY check fails (and if the number of questions differs):
- **Per-template means:** each template's mean of each count is within +-10% of the resampled structure's
  (relative to it).
- **90th percentiles:** each count's overall 90th percentile is within +-1 item.
- **Rewrites:** the share of questions with k_q >= 1 is within +-5 percentage points.

**The counts per question:**
- W_q: answer-changing items;
- C_q: controls other than rewrites, i.e. the untouched gold plus answer-neutral mutations;
- T_q: type-wrong items;
- k_q: rewrites, min(available, 3) without rewrite_redundant_join (entry 41).

A template whose resampled mean is 0 for a count needs the real mean to be 0 as well.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

COUNTS = ("W", "C", "T", "k")
MEAN_TOLERANCE = 0.10  # relative, per template and count
P90_TOLERANCE = 1.0  # items
SHARE_TOLERANCE = 0.05  # the share of questions with k_q >= 1
JOIN = "rewrite_redundant_join"
REWRITE_CAP = 3


def structure_from_items(items: Iterable[Mapping[str, Any]], split: str, cap: int = REWRITE_CAP,
                         exclude_operators: Sequence[str] = (JOIN,)) -> list[dict[str, Any]]:
    """Per-question source-free counts of a bank's split, from its items (id, question, split, kind, bank, fault class,
    template, tier, operator), sorted by question id."""
    items = [i for i in items if i["split"] == split]
    counts: dict[str, Counter] = {}
    meta: dict[str, tuple[str, str]] = {}
    for i in items:
        q = i["question_id"]
        c = counts.setdefault(q, Counter())
        if i.get("kind") == "control" and i.get("bank") == "control":
            meta[q] = (str(i.get("template")), str(i.get("tier")))
        if i.get("kind") == "rewrite":
            if i.get("operator") not in exclude_operators:
                c["k_available"] += 1
        elif i["bank"] == "answer_changing":
            c["W"] += 1
            c["T"] += i.get("fault_class") == "type_wrong"
        elif i["bank"] in ("control", "answer_neutral"):
            c["C"] += 1
        if q not in meta and i.get("template"):
            meta[q] = (str(i.get("template")), str(i.get("tier")))
    return [{"question": q, "template": meta[q][0], "tier": meta[q][1], "W": c["W"], "C": c["C"], "T": c["T"],
             "k": min(cap, c["k_available"])} for q, c in sorted(counts.items())]


def _p90(values: Sequence[float]) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.9, method="inverted_cdf"))


def compare(real: Sequence[Mapping[str, Any]], resampled: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every check of the ruling, and whether the calibration must be redone at the real structure."""
    checks = [{"check": "number of questions", "real": len(real), "resampled": len(resampled),
               "pass": len(real) == len(resampled)}]
    templates = sorted({r["template"] for r in resampled} | {r["template"] for r in real})
    for t in templates:
        a = [r for r in real if r["template"] == t]
        b = [r for r in resampled if r["template"] == t]
        for count in COUNTS:
            ma = float(np.mean([r[count] for r in a])) if a else None
            mb = float(np.mean([r[count] for r in b])) if b else None
            if ma is None or mb is None:
                ok = False
            elif mb == 0:
                ok = ma == 0
            else:
                ok = abs(ma - mb) <= MEAN_TOLERANCE * mb + 1e-12
            checks.append({"check": f"mean {count} in template {t}", "real": ma, "resampled": mb, "pass": ok})
    for count in COUNTS:
        pa, pb = _p90([r[count] for r in real]), _p90([r[count] for r in resampled])
        checks.append({"check": f"90th percentile of {count}", "real": pa, "resampled": pb,
                       "pass": abs(pa - pb) <= P90_TOLERANCE + 1e-12})
    sa = float(np.mean([r["k"] >= 1 for r in real]))
    sb = float(np.mean([r["k"] >= 1 for r in resampled]))
    checks.append({"check": "share of questions with k >= 1", "real": sa, "resampled": sb,
                   "pass": abs(sa - sb) <= SHARE_TOLERANCE + 1e-12})
    return {"recalibrate": not all(c["pass"] for c in checks), "failed": [c["check"] for c in checks if not c["pass"]],
            "checks": checks}
