#!/usr/bin/env python3
"""E6.1a: write the scoring rulings for the held-out corporate questions (heldout/text2sparql-ck26/scoring.json).

    uv run python scripts/e06_ck26_scoring.py --endpoint http://localhost:7021/

The rulings are the study lead's in the design log (2026-10-18), recorded in prose in heldout/text2sparql-ck26/README.md:

* ck26-29, ck26-46, ck26-50 (ORDER BY ... LIMIT with a tie at a cut and no tie-break): scored **tie-aware**. Every row
  of the window whose ordering value no row outside the window shares is required; for each value that rows inside
  and outside the window share (a tie at the OFFSET, the LIMIT, or both), the answer must hold ``slots`` rows of that
  group (``slots`` = the window positions that value fills), any members. Extra or missing rows count as usual.
* ck26-32 (ROUND(AVG(..) * 100) / 100 over decimals, which QLever computes in doubles): either the store's rows or the
  exact decimal rows are gold.
* ck26-35: the published -0.01 against the store's -0.009999999999998 is covered by the numeric tolerance.
* ck26-29 is excluded from the primary held-out score (study lead's ruling relayed 2026-10-19, an erratum): its text
  asks for the 6th to the 10th most expensive items, its published query takes ranks 11 to 15 (OFFSET 10 LIMIT 5). Its
  entry keeps the tie computation for the published query.

The tie groups are computed from the store: the gold query without its LIMIT and OFFSET (29, 50), or, for 46, the
supplier averages recomputed in exact decimal arithmetic from the per-product values (QLever's double averages of
identical values can split a tie that exact arithmetic keeps). The exact rows of 32 are recomputed the same way, with
SPARQL ROUND (halves away from zero for these positive values). The published files and the stored results are only
read; this script writes scoring.json only. Like everything under heldout/, the file is test-only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.sparql import SparqlClient
from harness.terms import row_to_json

REPO = Path(__file__).resolve().parents[1]
HELDOUT = REPO / "heldout" / "text2sparql-ck26"
XSD = "http://www.w3.org/2001/XMLSchema#"
RULINGS = "study lead, design log, 2026-10-18 (heldout/text2sparql-ck26/README.md, 'Scoring rulings')"
# Questions left out of the primary held-out score, with the reason (their stored data and entries stay).
EXCLUDED = {
    "ck26-29": {
        "ruling": "study lead's ruling, relayed by the lead 2026-10-19 (an erratum)",
        "reason": (
            "the text asks for the hardware items that rank from the 6th to the 10th most expensive, but the published "
            "gold query takes ranks 11 to 15 (ORDER BY DESC(?price) LIMIT 5 OFFSET 10): text and gold disagree, so the "
            "item is not scored in the primary held-out score; this entry's tie computation is for the published query"
        ),
    },
}
TIE_RULE = (
    "tie-aware: every row of the window [offset, offset + limit) of the full ordering whose ordering value no row "
    "outside the window shares is required; for each tied group (a value shared by rows inside and outside the window) "
    "the answer must hold 'slots' of its members, any of them; rows beyond these are extra, missing ones missing, as usual"
)


def _dump(row: Any) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tie_spec(rows: Sequence[Any], key: Callable[[Any], Any], offset: int, limit: int) -> dict[str, Any]:
    """Required rows and tied groups of the window [offset, offset + limit) of ``rows`` ordered by descending ``key``.

    ``rows`` may come in any order; rows with equal keys are interchangeable, so their order never matters.
    """
    ordered = sorted(rows, key=lambda r: (key(r), _dump(r)), reverse=True)
    window, outside = ordered[offset : offset + limit], ordered[:offset] + ordered[offset + limit :]
    outside_keys = {key(r) for r in outside}
    required = [r for r in window if key(r) not in outside_keys]
    groups: OrderedDict[Any, dict[str, Any]] = OrderedDict()
    for r in window:
        k = key(r)
        if k in outside_keys:
            groups.setdefault(k, {"key": str(k), "slots": 0, "members": [m for m in ordered if key(m) == k]})["slots"] += 1
    return {"required": required, "tied_groups": list(groups.values())}


def _strip_limit_offset(query: str) -> tuple[str, int, int]:
    limit = re.search(r"\bLIMIT\s+(\d+)", query, re.IGNORECASE)
    offset = re.search(r"\bOFFSET\s+(\d+)", query, re.IGNORECASE)
    stripped = re.sub(r"\b(LIMIT|OFFSET)\s+\d+", "", query, flags=re.IGNORECASE).rstrip() + "\n"
    return stripped, int(offset.group(1)) if offset else 0, int(limit.group(1)) if limit else 0


def _number(cell: dict[str, Any]) -> Decimal:
    return Decimal(cell["value"])


def _decimal_cell(value: Decimal) -> dict[str, Any]:
    text = format(value.normalize(), "f") if value != value.to_integral() else format(value.quantize(Decimal(1)), "f")
    return {"datatype": XSD + "decimal", "type": "literal", "value": text}


def _sparql_round(x: Decimal) -> Decimal:
    """SPARQL ROUND (fn:round): to the nearest integer, halves towards positive infinity."""
    return (x + Decimal("0.5")).to_integral_value(rounding="ROUND_FLOOR")


def build(client: SparqlClient, heldout: Path = HELDOUT) -> dict[str, Any]:
    queries = {q["id"]: q for q in json.loads((heldout / "queries.json").read_text())["questions"]}
    out: dict[str, Any] = OrderedDict()

    for qid, col in (("ck26-29", 2), ("ck26-50", 1)):
        gold = queries[qid]["query"]
        full, offset, limit = _strip_limit_offset(gold)
        res = client.query(full)
        rows = [row_to_json(r) for r in res.rows]
        spec = tie_spec(rows, lambda r, c=col: _number(r[c]), offset, limit)
        out[qid] = {
            "rule": "tie_aware",
            "query_sha256": queries[qid]["query_sha256"],
            "columns": res.columns,
            "order": f"descending by column {res.columns[col]!r}",
            "offset": offset,
            "limit": limit,
            "full_ordering_rows": len(rows),
            **spec,
        }

    # 46: exact decimal averages per supplier from the per-product values (the gold's own pattern).
    per_product = client.query(
        "PREFIX pv: <http://ld.company.org/prod-vocab/>\n"
        "SELECT ?result ?reliabilityIndex WHERE { ?hardware pv:reliabilityIndex ?reliabilityIndex . "
        "?hardware pv:hasSupplier ?result . }"
    )
    sums: OrderedDict[str, list[Decimal]] = OrderedDict()
    for r in per_product.rows:
        cells = row_to_json(r)
        sums.setdefault(cells[0]["value"], []).append(_number(cells[1]))
    rows46 = [
        [{"type": "iri", "value": s}, _decimal_cell(sum(v) / len(v))] for s, v in sorted(sums.items())
    ]
    _, offset, limit = _strip_limit_offset(queries["ck26-46"]["query"])
    out["ck26-46"] = {
        "rule": "tie_aware",
        "query_sha256": queries["ck26-46"]["query_sha256"],
        "columns": ["result", "averageReliabilityIndex"],
        "order": "descending by the exact decimal average of each supplier's reliability values",
        "offset": offset,
        "limit": limit,
        "full_ordering_rows": len(rows46),
        **tie_spec(rows46, lambda r: _number(r[1]), offset, limit),
    }

    # 32: the store's rows (as stored) or the exact ones.
    stored = _read_result(heldout / "results" / "ck26-32.json.gz")
    amounts = client.query(
        "PREFIX pv: <http://ld.company.org/prod-vocab/>\n"
        "SELECT ?sup ?name ?amt WHERE { ?sup a pv:Supplier ; pv:name ?name . ?hw pv:hasSupplier ?sup ; "
        "pv:price/pv:amount ?amt ; pv:price/pv:currency \"EUR\" . }"
    )
    groups: OrderedDict[tuple[str, str], list[Decimal]] = OrderedDict()
    for r in amounts.rows:
        cells = row_to_json(r)
        groups.setdefault((cells[0]["value"], cells[1]["value"]), []).append(_number(cells[2]))
    exact = sorted(
        ([{"type": "literal", "value": name}, _decimal_cell(_sparql_round(sum(v) / len(v) * 100) / 100)] for (_, name), v in groups.items()),
        key=_dump,
    )
    store_rows = sorted(stored["rows"], key=_dump)
    differing = sum(1 for a, b in zip(store_rows, exact) if Decimal(a[1]["value"]) != Decimal(b[1]["value"]))
    out["ck26-32"] = {
        "rule": "either_value",
        "query_sha256": queries["ck26-32"]["query_sha256"],
        "columns": stored["columns"],
        "note": (
            "QLever computes AVG over the decimal amounts in double precision before ROUND; either the store's rows or "
            "the exact decimal rows are gold, as a whole answer"
        ),
        "rows_differing": differing,
        "alternatives": {"store": store_rows, "exact": exact},
    }
    out["ck26-35"] = {
        "rule": "numeric_tolerance",
        "query_sha256": queries["ck26-35"]["query_sha256"],
        "note": "the store's -0.009999999999998 against the published -0.01: equal within the scorer's numeric tolerance",
    }
    for qid, exclusion in EXCLUDED.items():
        out[qid] = {**out[qid], "primary_score": "excluded", "exclusion": exclusion}
    return {
        "rulings": RULINGS,
        "tie_rule": TIE_RULE,
        "store": "text2sparql-ck26",
        "excluded_from_primary": sorted(EXCLUDED),
        "questions": out,
    }


def _read_result(path: Path) -> dict[str, Any]:
    import gzip

    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--out", type=Path, default=HELDOUT / "scoring.json")
    args = ap.parse_args(argv)
    doc = build(SparqlClient(args.endpoint, timeout_s=60))
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")
    for qid, q in doc["questions"].items():
        extra = ""
        if q["rule"] == "tie_aware":
            extra = f"required {len(q['required'])}, tied groups " + ", ".join(
                f"{g['key']} ({g['slots']} of {len(g['members'])})" for g in q["tied_groups"]
            )
        elif q["rule"] == "either_value":
            extra = f"{q['rows_differing']} of {len(q['alternatives']['store'])} rows differ"
        if q.get("primary_score") == "excluded":
            extra += " (excluded from the primary score)"
        print(f"{qid}: {q['rule']} {extra}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
