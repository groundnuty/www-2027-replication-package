"""Generate template questions with gold answers from a pinned store.

The output is deterministic: the same seed and the same store give a
byte-identical ``questions.jsonl``. Two measured quantities would break that,
so they are kept out of the question file:

- ``gold_query_seconds`` in each question is the measured time rounded up to a
  whole second (every gold query here runs well under a second, so it reads 1);
  the measured times go to ``manifest.json``.
- The generation time is in ``manifest.json`` only.

With a profile (the two-tier sets), query times never decide which questions
are selected: a time-out of any query fails the run, and so does a selected
gold over the time cap (``max_seconds``, applied to the first, cold run).
Each selected gold is then run twice more, back to back, and the record's
whole-second field comes from the fastest run: a first run can be slow on a
cold cache (one round-2 gold took 3.5-3.8 s whenever it was the first of its
shape since the cache was refilled, 0.05 s warm), which would make the bytes
depend on the cache state. A gold that is not under one second even warm
fails the run (WARM_GOLD_SECONDS), so a written file never depends on timing.
The cold times are in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from harness.qgen.base import (
    DISCRIMINATIONS,
    GRAPH_ID,
    ID_PREFIX,
    Candidate,
    Profile,
    QuestionTemplate,
    TemplateError,
    _substitute,
    resolve_templates,
)
from harness.qgen.readings import (
    ReadingError,
    check_readings,
    enforce,
    graph_unscoped,
    membership_swapped,
    nests_graph,
    neutral_text,
)
from harness.qgen.sampling import TWO_VALUED_SHARE, Accepted, Selector, quotas, rng_for, shuffled_groups
# The audit rules' value cap (a15; two-valued variants) is sampling's; the UniProt line keeps its own value_cap below.
from harness.qgen.sampling import VALUE_SHARE as AUDIT_VALUE_SHARE
from harness.qgen.sampling import value_cap as audit_value_cap
from harness.qgen.text_guard import database_names
from harness.qgen.split import DEV, TEST, assign_splits
from harness.qgen.templates import TEMPLATES
from harness.sparql import QueryResult, SparqlClient, SparqlError
from harness.terms import IRI, XSD, BNode, Literal, Row, row_from_json, row_to_json

GENERATOR_VERSION = "0.2.0"
STORE_IDENTITY = {
    "engine": "QLever 9bebc60",
    "image_digest": "sha256:dc93f8df62de83bff35374ea3f27af0461aa5453c643c7088cb9c64f71c6315c",
}
MIN_ROWS = 1
MAX_ROWS = 50
MAX_SECONDS = 10.0
# With a profile, every selected gold is re-run WARM_RUNS times back to back and must run under
# WARM_GOLD_SECONDS on its fastest run (see the module docstring).
WARM_RUNS = 2
WARM_GOLD_SECONDS = 1.0
NUMERIC_DATATYPES = frozenset(
    XSD + t
    for t in (
        "integer", "int", "long", "short", "byte", "decimal", "double", "float",
        "nonNegativeInteger", "nonPositiveInteger", "negativeInteger", "positiveInteger",
        "unsignedInt", "unsignedLong", "unsignedShort", "unsignedByte",
    )
)


@dataclass(frozen=True)
class Gold:
    sparql: str
    rows: list[list[dict[str, str] | None]]
    seconds: float
    # DISCRIMINATIONS computed so far (True, False, or None when not applicable or unknown).
    discriminates: dict[str, bool | None] = field(default_factory=dict)
    # A raised variant's step check (step_outcome), None for other variants.
    step: dict[str, Any] | None = None


@dataclass
class VariantStats:
    candidates: int = 0
    entities: int = 0
    evaluated: int = 0
    accepted: int = 0
    rejected: Counter = field(default_factory=Counter)
    # Entities per answer-shape stratum in the pool, and questions per stratum sampled.
    pool_strata: Counter = field(default_factory=Counter)
    sampled_strata: Counter = field(default_factory=Counter)
    # The variant's share of its tier's questions (two-tier profiles; None otherwise).
    quota: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "entities": self.entities,
            "evaluated": self.evaluated,
            "accepted": self.accepted,
            **({"quota": self.quota} if self.quota is not None else {}),
            "rejected": dict(sorted(self.rejected.items())),
        }

    def strata_json(self) -> dict[str, Any] | None:
        """Pool share versus sampled share per shape stratum; None when the variant has no strata."""
        if set(self.pool_strata) <= {""}:
            return None
        pool, sampled = sum(self.pool_strata.values()), sum(self.sampled_strata.values())
        return {
            s: {
                "pool": self.pool_strata[s],
                "pool_share": round(self.pool_strata[s] / pool, 4) if pool else None,
                "sampled": self.sampled_strata[s],
                "sampled_share": round(self.sampled_strata[s] / sampled, 4) if sampled else None,
            }
            for s in sorted(self.pool_strata)
        }


@dataclass
class Generation:
    records: list[dict[str, Any]]
    stats: dict[str, dict[str, VariantStats]]
    measured_seconds: dict[str, float]
    store_triples: dict[str, int]
    overlaps: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {"nested": [], "intersecting": []})
    # Raised variants' step checks by question id (calibration round 2).
    step_checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    # With a profile: each selected gold's faster time of two runs (the record's whole-second field).
    warm_seconds: dict[str, float] = field(default_factory=dict)
    # Comparison variants (round 2): per question the winner (or binding reference) and its mention slot.
    comparisons: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Rounds that keep test questions off an earlier round's dev anchors: the rule, the redrawn questions and shares.
    prior: dict[str, Any] | None = None
    # Notes from Profile.prepare (e.g. the size and digest of a sampling frame).
    prepared: dict[str, Any] = field(default_factory=dict)
    # Profile.confirm_golds: per template and variant, golds confirmed and replaced, and the slowest confirmation.
    confirmation: dict[str, Any] = field(default_factory=dict)


def pool_rows(result: QueryResult) -> list[dict[str, Any]]:
    """Candidate-query rows as plain strings (IRI value or literal lexical form)."""
    out = []
    for row in result.rows:
        values: dict[str, Any] = {}
        for col, cell in zip(result.columns, row):
            if isinstance(cell, IRI):
                values[col] = cell.value
            elif isinstance(cell, Literal):
                values[col] = cell.lexical
            elif cell is None:
                values[col] = None
            else:
                raise TemplateError(f"candidate query returned a blank node in {col}")
        out.append(values)
    return out


def canonical_rows(rows: Iterable[Row]) -> list[list[dict[str, str] | None]]:
    """Rows in the term JSON form, sorted by their canonical JSON (store order is not relied on)."""
    as_json = [row_to_json(r) for r in rows]
    return sorted(as_json, key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False))


def check_row_kinds(columns: Sequence[Any], rows: Iterable[Row]) -> None:
    """Every cell is bound and of its column's kind; otherwise the template is wrong."""
    for row in rows:
        if len(row) != len(columns):
            raise TemplateError(f"row has {len(row)} cells for {len(columns)} answer columns")
        for col, cell in zip(columns, row):
            if cell is None or isinstance(cell, BNode):
                raise TemplateError(f"column {col.name} has an unbound or blank cell")
            if col.kind == "iri" and not isinstance(cell, IRI):
                raise TemplateError(f"column {col.name} expects an IRI, got {cell!r}")
            if col.kind == "literal" and not isinstance(cell, Literal):
                raise TemplateError(f"column {col.name} expects a literal, got {cell!r}")
            if col.kind == "number" and not (isinstance(cell, Literal) and cell.datatype in NUMERIC_DATATYPES):
                raise TemplateError(f"column {col.name} expects a number, got {cell!r}")


def row_set(rows: Iterable[list[dict[str, str] | None]]) -> frozenset[str]:
    """Gold rows (term JSON form) as a set of canonical strings, for comparing answers."""
    return frozenset(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)


def _is_number(cell: Any) -> bool:
    return isinstance(cell, dict) and cell.get("type") == "literal" and cell.get("datatype") in NUMERIC_DATATYPES


def answer_key(rows: Sequence[list[dict[str, str] | None]], anchors: Sequence[dict[str, Any]] = ()) -> frozenset[str]:
    """The answer as compared by the overlap guards (profiles with audit_rules; ported from feat/qgen).

    A one-row answer of numbers only (a count, a difference) is identified with
    the entities the question names, so two questions whose counts happen to
    be equal are not taken for the same answer (audit of 2026-10-18 08:50,
    fix 1). Every other answer is its set of rows.
    """
    if len(rows) == 1 and rows[0] and all(_is_number(c) for c in rows[0]):
        named = sorted(str(a["entity"]) for a in anchors)
        return frozenset({json.dumps({"anchors": named, "row": rows[0]}, sort_keys=True, ensure_ascii=False)})
    return row_set(rows)


def _cell_text(cell: Any) -> str:
    return "" if cell is None else str(cell.get("value"))


def audit_answer_value(rows: Sequence[list[dict[str, str] | None]]) -> str:
    """The answer value of a single-row question, as compared by the value-diversity rule."""
    return " | ".join(_cell_text(c) for c in rows[0])


def named_entities(params: Mapping[str, Any]) -> list[dict[str, str]]:
    """The entities a question names: its IRI-valued parameters, as answer_key's anchors."""
    return [{"entity": v} for _, v in sorted(params.items()) if isinstance(v, str) and v.startswith(("http://", "https://"))]


VALUE_RULE = (
    "value diversity (design ruling in the design log): in each (variant, split) stratum of n single-value questions "
    f"(one-row answers: counts, differences, single literals), at most max(1, ceil({AUDIT_VALUE_SHARE} * n)) questions "
    "share one answer value (n=3 -> 1, n=7 -> 2, n=9 -> 3, n=20 -> 5); a two-valued variant is exempt but keeps "
    f"each value at most {TWO_VALUED_SHARE:.0%} of a stratum. Sampling caps each variant's values at the same "
    "formula over its quota, and the split spreads equal values over dev and test; generation fails if a stratum "
    "breaks the rule"
)


def audit_value_strata(records: Sequence[dict[str, Any]], templates: Sequence[type[QuestionTemplate]]) -> dict[str, Any]:
    """Per single-value variant and split: n, the cap, and the histogram of answer values."""
    out: dict[str, Any] = OrderedDict()
    for t in templates:
        for v in t.variants:
            if not v.single_row:
                continue
            rows = [r for r in records if r["template"] == t.id and r["params"]["variant"] == v.name]
            if not rows:
                continue
            entry: dict[str, Any] = {"tier": v.tier, "two_valued": v.two_valued}
            for split in (DEV, TEST):
                c = Counter(audit_answer_value(r["gold_rows"]) for r in rows if r["split"] == split)
                n = sum(c.values())
                entry[split] = {
                    "questions": n,
                    "cap": audit_value_cap(n, v.two_valued),
                    "max_share_count": max(c.values(), default=0),
                    "values": dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))),
                }
            out[f"{t.id}/{v.name}"] = entry
    return out


def check_audit_value_diversity(records: Sequence[dict[str, Any]], templates: Sequence[type[QuestionTemplate]]) -> dict[str, Any]:
    """Raise if a (variant, split) stratum has more questions sharing one value than its cap allows."""
    strata = audit_value_strata(records, templates)
    failures = [
        (key, split, d[split]["max_share_count"], d[split]["cap"])
        for key, d in strata.items()
        for split in (DEV, TEST)
        if d[split]["max_share_count"] > d[split]["cap"]
    ]
    if failures:
        raise TemplateError(f"value diversity broken (variant, split, most common count, cap): {failures}")
    return strata


CONFIRM_RULE = (
    "gold confirmation (design ruling (a), design log): every gold is confirmed by a differently written query, its "
    "variant's confirm_pattern (another route to the same answer, written by the template file; never the gold's "
    "own pattern counted or wrapped, which shares any fault), "
    "run on a client with a far higher time-out; the answers must agree exactly, row for row. A candidate whose "
    "gold disagrees is rejected ('unconfirmed') and sampling takes the next under the same rules. A time-out of a "
    "confirmation fails the run, as does a disagreement whose confirmation may be partial, so query times never "
    "decide the selection. A gold or confirmation of the shape the store is known to answer wrongly "
    "(harness/engine_suspect.py) fails the run."
)


def _group_text(pattern: str) -> str:
    """The pattern's WHERE group, whitespace collapsed (for comparing two patterns' bodies)."""
    m = re.search(r"\bWHERE\s*\{", pattern, re.IGNORECASE)
    if m is None:
        return " ".join(pattern.split())
    depth, start = 0, m.end() - 1
    for i in range(start, len(pattern)):
        if pattern[i] == "{":
            depth += 1
        elif pattern[i] == "}":
            depth -= 1
            if depth == 0:
                return " ".join(pattern[start : i + 1].split())
    return " ".join(pattern[start:].split())


def confirm_problem(variant: Any) -> str | None:
    """Why a variant's confirmation query cannot confirm its gold, or None."""
    if variant.confirm_pattern is None:
        return "has no confirmation query"
    if _group_text(variant.gold_pattern) in " ".join(variant.confirm_pattern.split()):
        return "confirms with the gold's own pattern (counted or wrapped, it shares any fault)"
    return None


def check_confirm_forms(templates: Sequence[type[QuestionTemplate]]) -> None:
    problems = [f"{t.id}/{v.name} {why}" for t in templates for v in t.variants if (why := confirm_problem(v))]
    if problems:
        raise TemplateError(f"gold confirmation impossible: {problems}")


class OverlapError(ValueError):
    """A dev question's answer equals, contains or is contained in a test question's."""


def split_overlaps(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Dev/test pairs of one template whose gold rows overlap.

    "nested": one answer set equals or contains the other, so knowing the dev
    answer gives away part of the test answer for the same rows. Generation
    refuses these. "intersecting": the sets share rows but neither contains
    the other (e.g. a common cofactor in two reactions); these are reported.
    """
    nested: list[dict[str, Any]] = []
    intersecting: list[dict[str, Any]] = []
    by_template: dict[str, list[dict[str, Any]]] = OrderedDict()
    for r in records:
        by_template.setdefault(r["template"], []).append(r)
    for template, rows in by_template.items():
        dev = [(r, row_set(r["gold_rows"])) for r in rows if r["split"] == DEV]
        test = [(r, row_set(r["gold_rows"])) for r in rows if r["split"] == TEST]
        for d, ds in dev:
            for t, ts in test:
                shared = ds & ts
                if not shared:
                    continue
                pair = {
                    "template": template,
                    "dev": d["id"],
                    "test": t["id"],
                    "dev_variant": d["params"]["variant"],
                    "test_variant": t["params"]["variant"],
                    "shared_rows": len(shared),
                    "dev_rows": len(ds),
                    "test_rows": len(ts),
                }
                (nested if ds <= ts or ts <= ds else intersecting).append(pair)
    return {"nested": nested, "intersecting": intersecting}


def _single_value(rows: Sequence[list[dict[str, str] | None]]) -> bool:
    return len(rows) == 1 and bool(rows[0]) and all(_is_number(c) for c in rows[0])


def split_overlaps_audit(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Dev/test pairs whose actual gold-row sets overlap, across all templates and tiers (audit_rules; ported from feat/qgen).

    The sets compared are the gold rows themselves (External review of 355f04e,
    finding 3), not ``answer_key``, which sampling uses for diversity.
    "nested": one answer set equals or contains the other, in any two templates
    (External reviews of 6e5ce1a); generation refuses these. One-row single values
    are left out of it: the study lead ruled that a shared value says nothing
    about another question (guessability is the value cap's job). Two equal ones
    in one template are listed as "equal_single_value". "intersecting": the sets
    share rows but neither contains the other, listed within a template with the
    shared values (e.g. a compound common to two reactions) and counted across
    templates.
    """
    nested: list[dict[str, Any]] = []
    intersecting: list[dict[str, Any]] = []
    equal_values: list[dict[str, Any]] = []
    cross_template_intersecting = 0
    keyed = [(r, row_set(r["gold_rows"]), _single_value(r["gold_rows"])) for r in records]
    dev = [k for k in keyed if k[0]["split"] == DEV]
    test = [k for k in keyed if k[0]["split"] == TEST]
    for d, ds, d_single in dev:
        for t, ts, t_single in test:
            shared = ds & ts
            if not shared:
                continue
            same_template = d["template"] == t["template"]
            pair = {
                "template": d["template"] if same_template else f"{d['template']}|{t['template']}",
                "dev": d["id"],
                "test": t["id"],
                "dev_variant": d["params"]["variant"],
                "test_variant": t["params"]["variant"],
                "shared_rows": len(shared),
                "dev_rows": len(ds),
                "test_rows": len(ts),
            }
            if d_single or t_single:
                if same_template and ds == ts:
                    pair["value"] = " | ".join(_cell_text(c) for c in d["gold_rows"][0])
                    equal_values.append(pair)
            elif ds <= ts or ts <= ds:
                nested.append(pair)
            elif same_template:
                pair["shared_values"] = sorted(" | ".join(_cell_text(c) for c in json.loads(row)) for row in shared)
                intersecting.append(pair)
            else:
                cross_template_intersecting += 1
    return {
        "nested": nested,
        "intersecting": intersecting,
        "equal_single_value": equal_values,
        "cross_template_intersecting": cross_template_intersecting,
    }


DISCRIMINATION_RULE = (
    "direct_vs_transitive: swapping rdfs:subClassOf and rdfs:subClassOf+ in the gold (not the master-reaction "
    "type test) changes the rows; graph_scope: dropping the GRAPH clauses changes the rows; null when not "
    "applicable or the variant query failed"
)
# Two-tier profiles test the other membership reading itself (the variant's
# direct_vs_transitive reading), because a store with a materialised closure
# answers a plain rdfs:subClassOf like the transitive path.
DISCRIMINATION_RULE_READINGS = (
    "direct_vs_transitive: the gold under the other membership reading changes the rows. For a question about "
    "everything below an anchor, the other reading is the anchor and its immediate children only, by the data's "
    "immediate-parent link (skos:narrowerTransitive in the taxonomy) or, without one, the subclasses with no other "
    "subclass of the anchor in between (GO); for a question about immediate children it is the whole subtree. Where "
    "a variant declares no such reading, rdfs:subClassOf and rdfs:subClassOf+ are swapped. graph_scope: dropping the "
    "GRAPH clauses changes the rows. null when not applicable"
)


def discrimination_summary(
    records: Sequence[dict[str, Any]], templates: Sequence[type[QuestionTemplate]], readings_based: bool = False
) -> dict[str, Any]:
    def tally(rows: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
        values = [r[field_name] for r in rows]
        known = [v for v in values if v is not None]
        return {
            "true": sum(1 for v in known if v),
            "false": sum(1 for v in known if not v),
            "not_applicable": len(values) - len(known),
            "share_true": round(sum(1 for v in known if v) / len(known), 4) if known else None,
        }

    out: dict[str, Any] = OrderedDict()
    for t in templates:
        rows = [r for r in records if r["template"] == t.id]
        out[t.id] = {
            "preferred": {v.name: list(v.prefer) for v in t.variants if v.prefer},
            "direct_vs_transitive": tally(rows, "discriminates_direct_vs_transitive"),
            "graph_scope": tally(rows, "discriminates_graph_scope"),
            "by_variant": {
                v.name: {
                    "direct_vs_transitive": tally([r for r in rows if r["params"]["variant"] == v.name], "discriminates_direct_vs_transitive"),
                    "graph_scope": tally([r for r in rows if r["params"]["variant"] == v.name], "discriminates_graph_scope"),
                }
                for v in t.variants
            },
        }
    return {
        "rule": DISCRIMINATION_RULE_READINGS if readings_based else DISCRIMINATION_RULE,
        "by_template": out,
    }


def readings_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per alternative reading: questions checked, answers changed, scope clauses present."""
    by_reading: dict[str, Counter] = OrderedDict()
    by_template: dict[str, dict[str, Counter]] = OrderedDict()
    for r in records:
        for name, res in sorted(r.get("alternative_readings", {}).items()):
            for target in (by_reading.setdefault(name, Counter()), by_template.setdefault(r["template"], OrderedDict()).setdefault(name, Counter())):
                target["checked"] += 1
                target["changed"] += res["changed"] is True
                target["unknown"] += res["changed"] is None
                target["clause_present"] += res["clause"]
                target["changed_without_clause"] += res["changed"] is not False and not res["clause"]
    return {
        "rule": (
            "for every question each alternative reading (organism with or without strains and sub-taxa, narrower or "
            "broader hierarchy terms, direct against transitive, is-a against part-of, ...) is run on the store; if it "
            "changes the answer, the question text must carry the matching scope clause, or generation fails"
        ),
        "by_reading": {k: dict(v) for k, v in by_reading.items()},
        "by_template": {t: {k: dict(v) for k, v in d.items()} for t, d in by_template.items()},
    }


def cross_template_nested(records: Sequence[dict[str, Any]]) -> int:
    """Pairs of questions from different templates whose answers are equal or nested."""
    sets = [(r["template"], row_set(r["gold_rows"])) for r in records]
    n = 0
    for i, (ti, si) in enumerate(sets):
        for tj, sj in sets[i + 1 :]:
            if ti != tj and (si <= sj or sj <= si):
                n += 1
    return n


def entity_reuse(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Entities (IRI-valued parameters) that more than one question names, and whether they straddle the split."""
    by_entity: dict[str, list[dict[str, Any]]] = OrderedDict()
    for r in records:
        for value in r["params"].values():
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                by_entity.setdefault(value, []).append(r)
    shared = {e: rs for e, rs in by_entity.items() if len(rs) > 1}
    straddling = {e: rs for e, rs in shared.items() if len({r["split"] for r in rs}) > 1}
    return {
        "entities": len(by_entity),
        "in_more_than_one_question": len(shared),
        "straddling_dev_test": len(straddling),
        "shared": {
            e: [{"id": r["id"], "split": r["split"]} for r in rs] for e, rs in sorted(shared.items())
        },
    }


def engine_sensitive_summary(records: Sequence[dict[str, Any]], templates: Sequence[type[QuestionTemplate]]) -> dict[str, Any]:
    reasons: dict[str, str] = {}
    for t in templates:
        for v in t.variants:
            for probe in v.engine_probes:
                reasons[probe.reason] = probe.description
    by_template: dict[str, Any] = OrderedDict()
    for t in templates:
        counts = Counter(code for r in records if r["template"] == t.id for code in r["engine_sensitive"])
        if counts:
            by_template[t.id] = {
                "questions": sum(1 for r in records if r["template"] == t.id and r["engine_sensitive"]),
                "by_reason": dict(sorted(counts.items())),
                DEV: sum(1 for r in records if r["template"] == t.id and r["engine_sensitive"] and r["split"] == DEV),
                TEST: sum(1 for r in records if r["template"] == t.id and r["engine_sensitive"] and r["split"] == TEST),
            }
    return {
        "rule": "a question is tagged when a natural query for it, run on this engine, does not give the gold rows",
        "reasons": dict(sorted(reasons.items())),
        "total": sum(1 for r in records if r["engine_sensitive"]),
        "by_template": by_template,
    }


def row_count_distribution(records: Sequence[dict[str, Any]], templates: Sequence[type[QuestionTemplate]]) -> dict[str, Any]:
    out: dict[str, Any] = OrderedDict()
    for t in templates:
        out[t.id] = OrderedDict()
        for v in t.variants:
            counts = sorted(r["gold_row_count"] for r in records if r["template"] == t.id and r["params"]["variant"] == v.name)
            if not counts:
                continue
            out[t.id][v.name] = {
                "distribution": {str(n): c for n, c in sorted(Counter(counts).items())},
                "min": counts[0],
                "median": counts[len(counts) // 2],
                "max": counts[-1],
                "multi_row": sum(n > 1 for n in counts),
                "single_row": sum(n == 1 for n in counts),
            }
    return out


TEMPORAL_DATATYPES = frozenset(XSD + t for t in ("date", "dateTime", "time", "gYear", "gYearMonth", "duration"))
# How a question states its column order: "the A, then the B", or in plain
# words "each A with its B" / "that A with its B".
COLUMN_ORDER_MARKER = re.compile(r", then (the|its) |\b(each|that)\b[^.?]*\bwith (its|the|that|a)\b")


def value_family(cell: Any) -> str:
    """The scorer's value family of a cell: iri, number, boolean, temporal, string, or other:<datatype>."""
    if isinstance(cell, IRI):
        return "iri"
    if isinstance(cell, BNode) or cell is None:
        return "blank"
    if cell.datatype in NUMERIC_DATATYPES:
        return "number"
    if cell.datatype == XSD + "boolean":
        return "boolean"
    if cell.datatype in TEMPORAL_DATATYPES:
        return "temporal"
    if cell.datatype in (None, XSD + "string") or cell.lang:
        return "string"
    return "other:" + cell.datatype


def same_family_columns(columns: Sequence[Any], rows: Iterable[Row]) -> bool:
    """Whether two answer columns share a value family (so the scorer will not permute them)."""
    families: list[set[str]] = [set() for _ in columns]
    for row in rows:
        for i, cell in enumerate(row):
            families[i].add(value_family(cell))
    for i, col in enumerate(columns):
        if not families[i]:  # no gold rows: judge from the declared kind
            families[i].add({"iri": "iri", "number": "number"}.get(col.kind, "string"))
    return any(families[i] & families[j] for i in range(len(columns)) for j in range(i + 1, len(columns)))


def discrimination(
    client: SparqlClient, name: str, sparql: str, mode: str, gold_rows: list, alternative: str | None = None,
    strict_timeouts: bool = False,
) -> bool | None:
    """Whether the variant query for ``name`` gives other rows than the gold; None if n.a. or it fails.

    ``alternative`` is the gold under the other membership reading when the
    variant spells it out (Variant.membership_alternative); otherwise
    rdfs:subClassOf is swapped with a regex. ``strict_timeouts``: a time-out
    raises instead of giving None, so query time cannot decide a preference.
    """
    if name == "direct_vs_transitive":
        alt = alternative if alternative is not None and mode != "n.a." else membership_swapped(sparql, mode)
    else:
        alt = graph_unscoped(sparql)
    if alt is None or alt == sparql:
        return None
    try:
        return canonical_rows(client.query(alt).rows) != gold_rows
    except SparqlError as exc:
        if strict_timeouts and exc.kind == "timeout":
            raise
        return None


def step_outcome(client: SparqlClient, variant: Any, params: Mapping[str, Any], gold_rows: list) -> dict[str, Any]:
    """Whether a raised variant's step matters for one question (Variant.step_check); see base.StepCheck."""
    check = variant.step_check
    out: dict[str, Any] = {"rule": check.rule}
    matters = True
    if check.undo:
        changed = []
        for pattern, rename in check.undo:
            undone = {**params, **{k: params[v] for k, v in rename.items()}}
            sparql = variant._prefixes() + _substitute(pattern, {**undone, **variant._graphs()})
            changed.append(canonical_rows(client.query(sparql).rows) != gold_rows)
        out["undo_changed"] = changed
        if check.steps:
            out["undo_steps"] = list(check.steps)
        matters = all(changed) if check.mode == "every" else any(changed)
    if check.fan_out is not None:
        sparql = variant._prefixes() + _substitute(check.fan_out, {**params, **variant._graphs()})
        by_near: dict[str, set[str]] = OrderedDict()
        for row in client.query(sparql).rows:
            cells = row_to_json(row)
            by_near.setdefault(json.dumps(cells[0], sort_keys=True), set()).add(json.dumps(cells[1:], sort_keys=True))
        items = set().union(*by_near.values()) if by_near else set()
        covered = any(v == items for v in by_near.values())
        out["fan_out"] = len(by_near)
        out["one_entity_covers_all"] = covered
        matters = matters and len(by_near) >= 2 and not covered
    out["matters"] = matters
    return out


# Value diversity (the design ruling in the design log, as in the Rhea set): in each
# (variant, split) stratum of n one-row questions, at most max(1, ceil(n / 4))
# share one answer value. The UniProt set keeps every answer apart anyway
# (global answer uniqueness), so this can only fail if that rule is broken.
VALUE_SHARE = 0.25


def value_cap(n: int) -> int:
    return 0 if n <= 0 else max(1, math.ceil(VALUE_SHARE * n - 1e-9))


def answer_value(rows: Sequence[list[dict[str, str] | None]]) -> str:
    """The value of a one-row answer, as compared by the value-diversity rule."""
    return " | ".join((c or {}).get("value", "") for c in rows[0])


def value_strata(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per (template/variant) and split: one-row questions, the cap and the histogram of answer values."""
    keys = OrderedDict((f"{r['template']}/{r['params']['variant']}", r.get("tier")) for r in sorted(records, key=lambda r: r["id"]))
    out: dict[str, Any] = OrderedDict()
    for key, tier in sorted(keys.items()):
        entry: dict[str, Any] = {"tier": tier}
        for split in (DEV, TEST):
            rows = [r for r in records if f"{r['template']}/{r['params']['variant']}" == key and r["split"] == split]
            c = Counter(answer_value(r["gold_rows"]) for r in rows if r["gold_row_count"] == 1)
            n = sum(c.values())
            entry[split] = {
                "questions": len(rows),
                "one_row": n,
                "cap": value_cap(n),
                "max_share_count": max(c.values(), default=0),
                "values": dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))),
            }
        out[key] = entry
    return out


def check_value_diversity(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Raise if a (variant, split) stratum has more one-row questions sharing a value than its cap."""
    strata = value_strata(records)
    broken = [
        (key, split, d[split]["max_share_count"], d[split]["cap"])
        for key, d in strata.items()
        for split in (DEV, TEST)
        if d[split]["max_share_count"] > d[split]["cap"]
    ]
    if broken:
        raise TemplateError(f"value diversity broken (variant, split, most common count, cap): {broken}")
    return strata


def engine_reasons(
    client: SparqlClient, variant: Any, params: dict[str, Any], gold_rows: list, strict_timeouts: bool = False
) -> list[str]:
    """Reason codes of the variant's engine probes whose natural query does not give the gold rows.

    ``strict_timeouts``: a probe time-out raises instead of counting as a reason (the bank-only extension).
    """
    reasons = []
    for probe in variant.engine_probes:
        try:
            rows = canonical_rows(client.query(variant.probe_sparql(probe, params)).rows)
        except SparqlError as exc:
            if strict_timeouts and exc.kind == "timeout":
                raise
            rows = None
        if rows != gold_rows:
            reasons.append(probe.reason)
    return sorted(reasons)


def seconds_bucket(seconds: float) -> int:
    """Whole seconds, rounded up, at least 1: stable across runs for fast queries."""
    return max(1, math.ceil(seconds))


def store_triples(client: SparqlClient) -> dict[str, int]:
    result = client.query("SELECT ?g (COUNT(*) AS ?n) WHERE { GRAPH ?g { ?s ?p ?o } } GROUP BY ?g")
    return {row[0].value: int(row[1].lexical) for row in result.rows}


# --------------------------------------------------------------------------
# Comparison variants (round 2, study lead's conditions on experiments#12)


def balanced_targets(n: int, k: int, rng: Any) -> list[int]:
    """Mention slots for the winners of ``n`` questions over ``k`` slots: as even as n allows, in seeded order."""
    targets = [slot for slot in range(k) for _ in range(n // k)]
    extra = list(range(k))
    rng.shuffle(extra)
    targets += extra[: n % k]
    rng.shuffle(targets)
    return targets


def place_winner(params: Mapping[str, Any], slots: Sequence[Sequence[str]], winner: int, target: int, rng: Any) -> dict[str, Any]:
    """The parameters with the winning entity moved to mention slot ``target``, the others in a seeded order."""
    entities = [tuple(params[p] for p in slot) for slot in slots]
    others = [e for i, e in enumerate(entities) if i != winner]
    rng.shuffle(others)
    order = others[:target] + [entities[winner]] + others[target:]
    out = dict(params)
    for slot, entity in zip(slots, order):
        out.update(zip(slot, entity))
    return out


def _balance_slots(
    picked: list, qids: list[str], splits: Mapping[str, str], seed: int, client: SparqlClient
) -> tuple[list, dict[str, dict[str, Any]]]:
    """Comparison questions with the winner's mention slot balanced within each (variant, split).

    The entities' order in the text (their parameter slots) is a seeded permutation; the gold rows do not
    change (checked on the store), and the step check is re-run with the new slots.
    """
    groups: dict[tuple[str, str, str], list[int]] = OrderedDict()
    for k, (t, acc) in enumerate(picked):
        v = t.variant(acc.candidate.variant)
        if v.slots:
            groups.setdefault((t.id, v.name, splits[qids[k]]), []).append(k)
    out, info = list(picked), {}
    for (tid, name, split), ks in groups.items():
        t = picked[ks[0]][0]
        v = t.variant(name)
        targets = balanced_targets(len(ks), len(v.slots), rng_for(seed, "slots", tid, name, split))
        for k, target in zip(ks, targets):
            acc = picked[k][1]
            params, gold = acc.candidate.params, acc.payload
            winner = v.winner(params, gold.rows, gold.step)
            winning = params[v.slots[winner][0]]
            new = place_winner(params, v.slots, winner, target, rng_for(seed, "slot-order", qids[k]))
            sparql = v.gold_sparql(new)
            if canonical_rows(client.query(sparql).rows) != gold.rows:
                raise TemplateError(f"{qids[k]}: reordering the compared entities changed the gold rows")
            step = step_outcome(client, v, new, gold.rows) if v.step_check is not None else gold.step
            if step is not None and not step["matters"]:
                raise TemplateError(f"{qids[k]}: the step check fails after reordering the compared entities")
            out[k] = (t, replace(acc, candidate=replace(acc.candidate, params=new), payload=replace(gold, sparql=sparql, step=step)))
            info[qids[k]] = {"winner": winning, "winner_slot": v.slots[target][0], "split": split}
    return out, info


def comparison_summary(records: Sequence[dict[str, Any]], templates: Sequence[type[QuestionTemplate]], info: Mapping[str, Any]) -> dict[str, Any]:
    """Per comparison variant and split: how often the winner (or binding reference) sits in each mention slot."""
    by_variant: dict[str, Any] = OrderedDict()
    for t in templates:
        for v in t.variants:
            if not v.slots:
                continue
            entry: dict[str, Any] = {}
            for split in (DEV, TEST):
                rows = [r for r in records if r["template"] == t.id and r["params"]["variant"] == v.name and r["split"] == split]
                hist = Counter(info[r["id"]]["winner_slot"] for r in rows)
                entry[split] = {
                    "questions": len(rows),
                    "winner_slot_histogram": {slot[0]: hist.get(slot[0], 0) for slot in v.slots},
                }
            by_variant[f"{t.id}/{v.name}"] = entry
    return by_variant


def check_comparisons(summary: Mapping[str, Any]) -> None:
    """Raise if the winner's mention slots in a comparison stratum differ by more than one."""
    broken = [
        (key, split, d["winner_slot_histogram"])
        for key, entry in summary.items()
        for split, d in entry.items()
        if d["questions"] and max(d["winner_slot_histogram"].values()) - min(d["winner_slot_histogram"].values()) > 1
    ]
    if broken:
        raise TemplateError(f"winner mention slots unbalanced (variant, split, histogram): {broken}")


def generate(
    gold_client: SparqlClient,
    seed: int,
    per_template: int,
    templates: Sequence[type[QuestionTemplate]] = TEMPLATES,
    pool_client: SparqlClient | None = None,
    max_rows: int = MAX_ROWS,
    max_seconds: float = MAX_SECONDS,
    min_preferred_share: float = 1.0,
    profile: Profile | None = None,
    prior_records: Sequence[Mapping[str, Any]] | None = None,
    confirm_client: SparqlClient | None = None,
) -> Generation:
    """Sample ``per_template`` questions for each template and compute their gold answers.

    ``gold_client`` should time out at ``max_seconds``; ``pool_client`` (default:
    the same client) runs the candidate queries, which can take longer.
    ``min_preferred_share``: share of each template's questions that must have
    their variant's preferred properties when the pool allows (1.0: all).
    ``profile``: the graph's id, id prefix, tiers and knobs; None keeps the
    original single-tier Rhea behaviour (and output) unchanged. With a
    profile, its structural steps (the delta knobs) choose each compositional
    variant's form, and query times never select: a time-out fails the run,
    as does a selected gold slower than ``max_seconds``.
    ``prior_records``: the questions of the earlier calibration round whose dev
    anchors the active round's test questions avoid (``Profile.prior_dev_rounds``);
    required exactly when the round pins one, and checked against its sha256.
    ``confirm_client``: the client with a far higher time-out that runs the
    confirmation queries (required when the profile confirms golds).
    """
    prior = _prior_round(profile, templates, prior_records)
    if profile is not None and profile.steps:
        templates = resolve_templates(templates, profile)
    if profile is not None and profile.resolve_variants is not None:
        templates = profile.resolve_variants(templates, profile.knobs)
    audit = bool(profile and profile.audit_rules)
    # The UniProt line's timing rules (strict time-outs, warm re-runs, the slow-gold check, comparison slots and its
    # value-diversity check). A profile with audit_rules (a15) has its own rules for all of these instead, as on
    # feat/a15, and its records carry no query times.
    strict = profile is not None and not audit
    pool_client = pool_client or gold_client
    graph_id = profile.graph_id if profile else GRAPH_ID
    id_prefix = profile.id_prefix if profile else ID_PREFIX
    two_tier = bool(profile and profile.two_tier)
    knobs = dict(profile.knobs) if profile else {}
    forbid_names = bool(profile and profile.forbid_database_names)
    extra_names = tuple(profile.forbidden_names) if profile else ()
    unique_anchors = bool(profile and profile.unique_anchors_per_template)
    if profile is not None and profile.require_step_checks:
        missing = [f"{t.id}/{v.name}" for t in templates for v in t.variants if v.tier == "compositional" and v.step_check is None]
        if missing:
            raise TemplateError(f"compositional variants without a step check: {missing}")
    # Shared across templates when the profile asks for set-wide answer uniqueness.
    global_answers: list | None = [] if profile and profile.global_answer_uniqueness else None
    confirm = None
    confirmation: dict[str, Any] = {}
    if profile is not None and profile.confirm_golds:
        if confirm_client is None:
            raise TemplateError("the profile confirms every gold: pass a confirm_client")
        check_confirm_forms(templates)
        confirm = (confirm_client, confirmation, profile.engine_suspect)
    triples = profile.store_count(pool_client) if profile and profile.store_count else store_triples(pool_client)
    prepared = profile.prepare(pool_client, seed, knobs) if profile and profile.prepare else {}
    stats: dict[str, dict[str, VariantStats]] = OrderedDict()
    picked: list[tuple[type[QuestionTemplate], Accepted]] = []
    # The selector and its position for each pick, so a pick can be redrawn once its split is known.
    slots: list[tuple[Selector, int]] = []
    for template in templates:
        if not two_tier:
            selectors: list[Selector] = []
            stats[template.id], accepted = _sample_template(
                template, gold_client, pool_client, seed, per_template, max_rows, max_seconds, min_preferred_share,
                selector_out=selectors,
                confirm=confirm,
            )
            picked.extend((template, a) for a in accepted)
            slots.extend((selectors[0], n) for n in range(len(accepted)))
            continue
        # Two tiers: each tier fills its own count; entities and answers stay
        # unique across both tiers of the template.
        used_keys: set[str] = set()
        used_answers: list = global_answers if global_answers is not None else []
        # One anchor entity per template (both tiers): the entities each candidate names by label.
        used_anchors: set[str] | None = set() if unique_anchors else None
        anchors_of = (lambda c, t=template: anchor_entities(t.variant(c.variant), c.params)) if unique_anchors else None
        stats[template.id] = OrderedDict()
        for tier, count in profile.tier_counts(per_template).items():
            variants = [v for v in template.variants if v.tier == tier]
            if not variants or count <= 0:
                raise TemplateError(f"{template.id}: no {tier} variants for {count} questions")
            selectors = []
            tier_stats, accepted = _sample_template(
                template, gold_client, pool_client, seed, count, max_rows, max_seconds, min_preferred_share,
                variants=variants, knobs=knobs, used=(used_keys, used_answers), forbid_names=forbid_names,
                strict_timeouts=strict, selector_out=selectors, audit=audit, confirm=confirm, extra_names=extra_names,
                anchors=(anchors_of, used_anchors) if unique_anchors else None,
            )
            stats[template.id].update(tier_stats)
            picked.extend((template, a) for a in accepted)
            slots.extend((selectors[0], n) for n in range(len(accepted)))

    # Ids and splits depend only on the pick order and the variants, so they are known before the records.
    counters: Counter = Counter()
    qids = []
    for template, acc in picked:
        counters[template.id] += 1
        qids.append(f"{id_prefix}-{template.id}-{counters[template.id]:03d}")

    def value_fields(template: type[QuestionTemplate], acc: Accepted) -> dict[str, Any]:
        # audit_rules: the split spreads equal answer values of a single-value variant over dev and test.
        v = template.variant(acc.candidate.variant)
        if not (audit and v.single_row):
            return {}
        return {"value": audit_answer_value(acc.payload.rows), "two_valued": v.two_valued}

    splits = assign_splits(
        [
            {
                "template": template.id,
                "base_question_id": qid,
                "stratum": acc.candidate.variant,
                **({"tier": template.variant(acc.candidate.variant).tier} if two_tier else {}),
                **value_fields(template, acc),
            }
            for (template, acc), qid in zip(picked, qids)
        ],
        seed,
    )
    redrawn: list[dict[str, Any]] = []
    if prior is not None:
        # After every template is sampled, so a redrawn answer is checked against the whole set and nothing else moves.
        redrawn = _redraw_on_prior_anchors(picked, slots, qids, splits, prior["anchors"], stats)
    if strict:
        slow = [(t.id, a.candidate.variant, round(a.payload.seconds, 2)) for t, a in picked if a.payload.seconds >= max_seconds]
        if slow:
            raise TemplateError(f"selected gold queries at or over the time cap of {max_seconds:g} s: {slow[:5]}")
    comparisons: dict[str, dict[str, Any]] = {}
    if strict:
        picked, comparisons = _balance_slots(picked, qids, splits, seed, gold_client)
    version = profile.generator_version(GENERATOR_VERSION) if profile is not None else GENERATOR_VERSION

    records = []
    measured: dict[str, float] = {}
    step_checks: dict[str, dict[str, Any]] = {}
    warm: dict[str, float] = {}
    texts: set[str] = set()
    for (template, acc), qid in zip(picked, qids):
        variant = template.variant(acc.candidate.variant)
        gold: Gold = acc.payload
        text = variant.question(acc.candidate.params)
        if text in texts:
            raise TemplateError(f"duplicate question text: {text}")
        texts.add(text)
        if forbid_names and database_names(text, extra_names):
            raise TemplateError(f"{qid}: question text names a database {database_names(text, extra_names)}: {text}")
        measured[qid] = round(gold.seconds, 4)
        seconds = gold.seconds
        if strict:
            seconds = min([seconds, *(gold_client.query(gold.sparql).elapsed_s for _ in range(WARM_RUNS))])
            warm[qid] = round(seconds, 4)
            if seconds >= WARM_GOLD_SECONDS:
                raise TemplateError(f"{qid}: the gold takes {seconds:.2f} s even when warm, so its whole-second field would depend on timing")
        if gold.step is not None:
            step_checks[qid] = gold.step
        elif profile is not None and profile.require_step_checks and variant.tier == "compositional":
            raise TemplateError(f"{qid}: a compositional question without a step check")
        if (strict or audit) and nests_graph(gold.sparql):
            raise TemplateError(f"{qid}: the gold nests a GRAPH clause inside another")
        for anchor in variant.anchors:
            n = anchor_count(gold_client, variant, anchor, acc.candidate.params)
            if n != 1:
                raise TemplateError(f"{qid}: the {anchor.kind} label anchor matches {n} entities, not one")
        engine_sensitive = engine_reasons(gold_client, variant, acc.candidate.params, gold.rows, strict_timeouts=audit)
        same_family = same_family_columns(variant.answer_columns, [row_from_json(r) for r in gold.rows])
        if same_family and not COLUMN_ORDER_MARKER.search(text):
            raise TemplateError(
                f"{template.id}/{variant.name}: two answer columns share a value family, so the question must state "
                f"their order (\"the A, then the B\"): {text}"
            )
        mode = variant.mode(template)
        disc = {
            name: gold.discriminates[name]
            if name in gold.discriminates
            else discrimination(
                gold_client, name, gold.sparql, mode, gold.rows, variant.alternative_sparql(acc.candidate.params),
                strict_timeouts=strict,
            )
            for name in DISCRIMINATIONS
        }
        record = {
                "id": qid,
                "base_question_id": qid,
                "template": template.id,
                "graph": graph_id,
                "split": None,
                "question": text,
                "answer_columns": [c.to_json() for c in variant.answer_columns],
                "gold_sparql": gold.sparql,
                "gold_rows": gold.rows,
                "gold_row_count": len(gold.rows),
                **({} if audit else {"gold_query_seconds": seconds_bucket(seconds)}),
                "scoring_rules": dict(variant.scoring),
                "membership_mode": mode,
                "hops": variant.hops,
                "params": {"variant": variant.name, **acc.candidate.params},
                "engine_sensitive": engine_sensitive,
                "discriminates_direct_vs_transitive": disc["direct_vs_transitive"],
                "discriminates_graph_scope": disc["graph_scope"],
                "same_family_columns": same_family,
                "generator_version": version,
                "seed": seed,
        }
        if two_tier:
            record = _with_tier(record, variant, acc.candidate.params)
            clause_text = neutral_text(text, acc.candidate.params) if audit else text
            readings = check_readings(gold_client, variant, acc.candidate.params, gold.sparql, gold.rows, clause_text)
            enforce(qid, readings)
            if "direct_vs_transitive" in readings:
                # The reading is the stronger test where the closure is materialised in the data.
                record["discriminates_direct_vs_transitive"] = readings["direct_vs_transitive"].changed
            record = _insert_after(record, "same_family_columns", "alternative_readings", {k: v.to_json() for k, v in readings.items()})
        records.append(record)
    for r in records:
        r["split"] = splits[r["base_question_id"]]
    records.sort(key=lambda r: r["id"])
    if strict:
        check_value_diversity(records)
        check_comparisons(comparison_summary(records, templates, comparisons))
    if audit:
        check_audit_value_diversity(records, templates)
    if unique_anchors:
        check_unique_anchors(records, templates)
    overlaps = split_overlaps_audit(records) if audit else split_overlaps(records)
    if overlaps["nested"]:
        raise OverlapError(f"dev and test answers are nested: {overlaps['nested'][:5]}")
    summary = None
    if prior is not None:
        check_prior_anchors(records, templates, prior["anchors"])
        summary = prior_summary(records, templates, prior, redrawn)
    return Generation(records, stats, measured, triples, overlaps, step_checks, warm, comparisons, summary,
                      prepared, confirmation)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _confirmed(template: Any, variant: Any, c: Candidate, sparql: str, rows: list, confirm: tuple) -> bool:
    """Run the variant's differently written gold; True when it gives exactly the same rows (CONFIRM_RULE)."""
    client, summary, suspect = confirm
    confirm_sparql = variant.confirm_sparql(c.params)
    if suspect is not None and (suspect(sparql) or suspect(confirm_sparql)):
        raise TemplateError(f"{template.id}/{variant.name}: a gold or confirmation query has the engine-suspect shape")
    result = client.query(confirm_sparql)  # a time-out raises: it fails the run
    entry = summary.setdefault(template.id, {}).setdefault(variant.name, {"confirmed": 0, "replaced": 0, "max_seconds": 0.0})
    entry["max_seconds"] = max(entry["max_seconds"], round(result.elapsed_s, 3))
    if canonical_rows(result.rows) == rows:
        entry["confirmed"] += 1
        summary.setdefault("_confirmed_golds", set()).add(_sha(sparql))
        return True
    if getattr(result, "possibly_partial", False):
        raise TemplateError(f"{template.id}/{variant.name}: a confirmation disagrees with its gold and may be partial "
                            f"({result.elapsed_s:.0f} s)")
    entry["replaced"] += 1
    return False


def _with_tier(record: dict[str, Any], variant: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """The record with ``tier`` and ``difficulty`` inserted after ``hops`` (two-tier profiles only)."""
    out: dict[str, Any] = {}
    for key, value in record.items():
        out[key] = value
        if key == "hops":
            out["tier"] = variant.tier
            out["difficulty"] = {**dict(variant.difficulty), **{k: params[k] for k in variant.knob_params}}
    return out


def _insert_after(record: dict[str, Any], after: str, key: str, value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in record.items():
        out[k] = v
        if k == after:
            out[key] = value
    return out


def anchor_count(client: SparqlClient, variant: Any, anchor: Any, params: Mapping[str, Any]) -> int:
    """How many entities of the anchor's kind carry the question's label."""
    rows = client.query(variant.anchor_sparql(anchor, params)).rows
    return int(rows[0][0].lexical) if rows and rows[0][0] is not None else 0


def _sample_template(
    template: type[QuestionTemplate],
    gold_client: SparqlClient,
    pool_client: SparqlClient,
    seed: int,
    per_template: int,
    max_rows: int,
    max_seconds: float,
    min_preferred_share: float = 1.0,
    variants: Sequence[Any] | None = None,
    knobs: Mapping[str, Any] | None = None,
    used: tuple[set, list] | None = None,
    forbid_names: bool = False,
    strict_timeouts: bool = False,
    selector_out: list[Selector] | None = None,
    audit: bool = False,
    confirm: tuple[SparqlClient, dict[str, Any], Any] | None = None,
    entities: Any = None,
    exclude: Any = None,
    exclude_why: str = "excluded",
    extra_names: tuple[str, ...] = (),
    anchors: tuple[Any, set[str]] | None = None,
) -> tuple[dict[str, VariantStats], list[Accepted]]:
    variants = list(template.variants if variants is None else variants)
    vstats: dict[str, VariantStats] = OrderedDict((v.name, VariantStats()) for v in variants)
    weights = [int((knobs or {}).get("variant_weights", {}).get(v.name, v.weight)) for v in variants]
    if strict_timeouts:
        for v, q in zip(variants, quotas(per_template, weights)):
            vstats[v.name].quota = q
    pools = []
    for variant in variants:
        if knobs:
            pool = pool_client.query(variant.candidate_sparql(knobs))
            # audit_rules: a slow pool query fails the run (never changes the selection); some stores return a
            # long join incomplete without flagging a time-out
            limit = knobs.get("pool_max_seconds")
            if audit and limit is not None and pool.elapsed_s > float(limit):
                raise TemplateError(f"{template.id}/{variant.name}: the pool query took {pool.elapsed_s:.0f} s (> {limit} s)")
            candidates = variant.candidates(pool_rows(pool), knobs)
        else:
            candidates = variant.candidates(pool_rows(pool_client.query(variant.candidate_sparql())))
        for c in candidates:
            c.params.update({k: knobs[k] for k in variant.knob_params})
        vstats[variant.name].candidates = len(candidates)
        vstats[variant.name].entities = len({c.key for c in candidates})
        vstats[variant.name].pool_strata = Counter({c.key: c.stratum for c in candidates}.values())
        pools.append(shuffled_groups(candidates, rng_for(seed, "sample", template.id, variant.name)))

    def reject(c: Candidate, why: str) -> None:
        vstats[c.variant].rejected[why] += 1

    def evaluate(c: Candidate):
        variant = template.variant(c.variant)
        s = vstats[c.variant]
        s.evaluated += 1
        if forbid_names and database_names(variant.question(c.params), extra_names):
            reject(c, "database_name")
            return None
        for anchor in variant.anchors:
            if anchor_count(gold_client, variant, anchor, c.params) != 1:
                reject(c, "ambiguous_anchor")
                return None
        sparql = variant.gold_sparql(c.params)
        if audit and nests_graph(sparql):
            raise TemplateError(f"{template.id}/{variant.name}: the gold nests a GRAPH clause inside another")
        if audit:
            result = gold_client.query(sparql)  # a time-out raises: it fails the run, never picks another question
            limit = (knobs or {}).get("gold_max_seconds")
            if limit is not None and result.elapsed_s > float(limit):
                raise TemplateError(f"{template.id}/{variant.name}: a gold query took {result.elapsed_s:.0f} s (> {limit} s)")
        else:
            try:
                result = gold_client.query(sparql)
            except SparqlError as exc:
                if exc.kind == "timeout" and not strict_timeouts:
                    reject(c, "timeout")
                    return None
                raise
            # With strict time-outs, query time never selects; generate() fails on a slow selected gold.
            if result.elapsed_s >= max_seconds and not strict_timeouts:
                reject(c, "timeout")
                return None
        if audit and variant.single_row and result.total_rows > 1:
            reject(c, "not_single_row")
            return None
        if result.total_rows < MIN_ROWS:
            reject(c, "empty")
            return None
        if result.total_rows > max_rows:
            reject(c, "too_many_rows")
            return None
        check_row_kinds(variant.answer_columns, result.rows)
        rows = canonical_rows(result.rows)
        if len({json.dumps(r, sort_keys=True) for r in rows}) != len(rows):
            raise TemplateError(f"{template.id}/{variant.name}: gold has duplicate rows")
        if confirm is not None and not _confirmed(template, variant, c, sparql, rows, confirm):
            reject(c, "unconfirmed")
            return None
        step = None
        if variant.step_check is not None:
            step = step_outcome(gold_client, variant, c.params, rows)
            if not step["matters"]:
                reject(c, "step_not_load_bearing")
                return None
        gold = Gold(sparql, rows, result.elapsed_s, step=step)
        for name in variant.prefer:
            gold.discriminates[name] = discrimination(
                gold_client, name, sparql, variant.mode(template), rows, variant.alternative_sparql(c.params),
                strict_timeouts=strict_timeouts,
            )
        tier = sum(1 for name in variant.prefer if gold.discriminates[name])
        return gold, (answer_key(rows, named_entities(c.params)) if audit else row_set(rows)), tier

    value_args: dict[str, Any] = {}
    if audit:
        # Value diversity: a single-value variant takes at most value_cap(quota) questions with one answer value,
        # so the split can keep every (variant, split) stratum within its cap.
        weights = [int((knobs or {}).get("variant_weights", {}).get(v.name, v.weight)) for v in variants]
        variant_quota = quotas(per_template, weights)
        by_name = {v.name: v for v in variants}
        value_args = {
            "value_of": lambda c, gold: audit_answer_value(gold.rows) if by_name[c.variant].single_row else None,
            "value_caps": [audit_value_cap(q, v.two_valued) if v.single_row else None for v, q in zip(variants, variant_quota)],
        }

    selector = Selector(
        pools,
        per_template,
        evaluate,
        on_skip=reject,
        weights=weights,
        top_tiers=[len(v.prefer) for v in variants],
        nonpreferred_budget=math.floor((1 - min_preferred_share) * per_template + 1e-9),
        used_keys=used[0] if used else None,
        used_answers=used[1] if used else None,
        entities=entities,
        exclude=exclude,
        exclude_why=exclude_why,
        **value_args,
        anchors_of=anchors[0] if anchors else None,
        used_anchors=anchors[1] if anchors else None,
    )
    accepted = list(selector.run())
    if selector_out is not None:
        selector_out.append(selector)
    for a in accepted:
        vstats[a.candidate.variant].accepted += 1
        vstats[a.candidate.variant].sampled_strata[a.candidate.stratum] += 1
    return vstats, accepted


# --------------------------------------------------------------------------
# An earlier round's dev anchors (design ruling in the design log)

PRIOR_RULE = (
    "no test question names an anchor entity that a dev question of the same variant named in calibration round "
    "{prior}: those dev questions were the calibration pool, whose results were seen (design ruling in the design log and "
    "12). A question's anchor entities are the entities its text names by label (its variant's anchors), read from its "
    "parameters as IRIs (anchor_params); entities are compared, not parameter names, so a round-1 organism and a round-2 "
    "taxon of the same variant clash when they are the same taxon. Dev questions may reuse them; questions of other "
    "variants are listed under other_variants, not redrawn. The split is assigned after sampling, so a test question "
    "that breaks the rule is redrawn in place once every template is sampled: its id, variant and split stay, its "
    "variant's seeded pool continues where it stopped (every sampling rule applies, the answer checked against the "
    "whole set, plus this rule), and the dropped candidate's entity and answer stay used, so no other question moves. "
    "Generation fails if a test question still names such an anchor, or if the round-{prior} file is missing or not "
    "the pinned one."
)


def anchor_entities(variant: Any, params: Mapping[str, Any]) -> frozenset[str]:
    """The IRIs of the entities a question names by label (its variant's anchors), read from its parameters."""
    unnamed = [a.kind for a in variant.anchors if not a.param]
    if unnamed:
        raise TemplateError(f"{variant.name}: anchors without an entity parameter: {unnamed}")
    return frozenset(str(params[a.param]) for a in variant.anchors)


def prior_dev_anchors(
    prior_records: Iterable[Mapping[str, Any]], templates: Sequence[type[QuestionTemplate]]
) -> dict[str, dict[str, list[str]]]:
    """"template/variant" -> anchor IRI -> ids of the earlier round's dev questions of that variant naming it.

    ``templates`` are the earlier round's (its variant forms name the anchor parameters).
    """
    by_id = {t.id: t for t in templates}
    out: dict[str, dict[str, list[str]]] = OrderedDict()
    for r in sorted(prior_records, key=lambda r: r["id"]):
        if r["split"] != DEV:
            continue
        variant = by_id[r["template"]].variant(r["params"]["variant"])
        entry = out.setdefault(f"{r['template']}/{variant.name}", OrderedDict())
        for iri in sorted(anchor_entities(variant, r["params"])):
            entry.setdefault(iri, []).append(r["id"])
    return out


def _test_anchors(records: Iterable[Mapping[str, Any]], templates: Sequence[type[QuestionTemplate]]):
    by_id = {t.id: t for t in templates}
    for r in sorted(records, key=lambda r: r["id"]):
        if r["split"] == TEST:
            key = f"{r['template']}/{r['params']['variant']}"
            yield r, key, anchor_entities(by_id[r["template"]].variant(r["params"]["variant"]), r["params"])


def prior_conflicts(
    records: Iterable[Mapping[str, Any]], templates: Sequence[type[QuestionTemplate]], prior: Mapping[str, Mapping[str, list[str]]]
) -> list[dict[str, Any]]:
    """Test questions naming an anchor that an earlier round's dev question of the same variant named."""
    out = []
    for r, key, anchors in _test_anchors(records, templates):
        seen = prior.get(key, {})
        shared = sorted(anchors & set(seen))
        if shared:
            out.append({"id": r["id"], "variant": key, "shared": {iri: list(seen[iri]) for iri in shared}})
    return out


def prior_other_variant_shares(
    records: Iterable[Mapping[str, Any]], templates: Sequence[type[QuestionTemplate]], prior: Mapping[str, Mapping[str, list[str]]]
) -> list[dict[str, Any]]:
    """Test questions naming an anchor of an earlier round's dev question of another variant (reported, not redrawn)."""
    owners: dict[str, list[tuple[str, str]]] = {}
    for key, entry in prior.items():
        for iri, ids in entry.items():
            owners.setdefault(iri, []).extend((key, i) for i in ids)
    out = []
    for r, key, anchors in _test_anchors(records, templates):
        shared = {iri: sorted(i for k, i in owners.get(iri, []) if k != key) for iri in sorted(anchors)}
        shared = {iri: ids for iri, ids in shared.items() if ids}
        if shared:
            out.append({"id": r["id"], "variant": key, "shared": shared})
    return out


def check_prior_anchors(
    records: Iterable[Mapping[str, Any]], templates: Sequence[type[QuestionTemplate]], prior: Mapping[str, Mapping[str, list[str]]]
) -> None:
    """Raise if a test question names an anchor of an earlier round's dev question of the same variant."""
    conflicts = prior_conflicts(records, templates, prior)
    if conflicts:
        raise TemplateError(
            "test questions name an anchor of an earlier round's dev question of the same variant: "
            f"{[(c['id'], sorted(c['shared'])) for c in conflicts[:5]]}"
        )


def _prior_round(
    profile: Profile | None, templates: Sequence[type[QuestionTemplate]], prior_records: Sequence[Mapping[str, Any]] | None
) -> dict[str, Any] | None:
    """The earlier round whose dev anchors the active round's test questions avoid, checked against its pin; None if none."""
    pin = profile.prior_dev_round() if profile is not None else None
    if pin is None:
        if prior_records is not None:
            raise TemplateError("earlier-round questions were given, but the active round avoids no earlier round's dev anchors")
        return None
    rnd, sha = pin
    if prior_records is None:
        raise TemplateError(
            f"calibration round {profile.calibration_round()} keeps its test questions off the round-{rnd} dev anchors: "
            f"pass the round-{rnd} questions (sha256 {sha})"
        )
    got = hashlib.sha256(questions_jsonl(prior_records).encode("utf-8")).hexdigest()
    if got != sha:
        raise TemplateError(f"the round-{rnd} questions given have sha256 {got}, not the pinned {sha}")
    earlier = replace(profile, knobs={**profile.knobs, **profile.calibration_rounds[rnd]})
    prior_templates = resolve_templates(templates, earlier) if profile.steps else tuple(templates)
    return {
        "round": rnd,
        "sha256": sha,
        "anchors": prior_dev_anchors(prior_records, prior_templates),
        "dev_questions": sum(1 for r in prior_records if r["split"] == DEV),
    }


def _redraw_on_prior_anchors(
    picked: list[tuple[type[QuestionTemplate], Accepted]],
    slots: Sequence[tuple[Selector, int]],
    qids: Sequence[str],
    splits: Mapping[str, str],
    prior: Mapping[str, Mapping[str, list[str]]],
    stats: Mapping[str, Mapping[str, VariantStats]],
) -> list[dict[str, Any]]:
    """Redraw, in place, every test pick that names an earlier round's dev anchor of its variant; what was redrawn."""
    redrawn = []
    for k, ((template, acc), qid) in enumerate(zip(picked, qids)):
        key = f"{template.id}/{acc.candidate.variant}"
        excluded = frozenset(prior.get(key, {}))
        if splits[qid] != TEST or not excluded:
            continue
        variant = template.variant(acc.candidate.variant)
        shared = anchor_entities(variant, acc.candidate.params) & excluded
        if not shared:
            continue
        selector, n = slots[k]
        new = selector.redraw(
            n, lambda c, v=variant, ex=excluded: bool(anchor_entities(v, c.params) & ex), why="prior_dev_anchor"
        )
        picked[k] = (template, new)
        s = stats[template.id][acc.candidate.variant]
        s.sampled_strata[acc.candidate.stratum] -= 1
        s.sampled_strata[new.candidate.stratum] += 1
        redrawn.append({
            "id": qid,
            "variant": key,
            "dropped": sorted(anchor_entities(variant, acc.candidate.params)),
            "shared": {iri: list(prior[key][iri]) for iri in sorted(shared)},
            "drawn": sorted(anchor_entities(variant, new.candidate.params)),
        })
    return redrawn


def prior_summary(
    records: Sequence[dict[str, Any]], templates: Sequence[type[QuestionTemplate]], prior: Mapping[str, Any], redrawn: list[dict[str, Any]]
) -> dict[str, Any]:
    """The manifest's record of the rule: the earlier round, the anchor parameters, the redrawn questions, other shares."""
    anchors = prior["anchors"]
    return {
        "rule": PRIOR_RULE.format(prior=prior["round"]),
        "prior_round": prior["round"],
        "prior_questions_sha256": prior["sha256"],
        "scope": "test questions against the earlier round's dev questions of the same variant (template/variant)",
        "anchor_params": {f"{t.id}/{v.name}": [a.param for a in v.anchors] for t in templates for v in t.variants},
        "prior_dev_questions": prior["dev_questions"],
        "prior_dev_anchor_entities": sum(len(e) for e in anchors.values()),
        "redrawn": redrawn,
        "test_questions_sharing": len(prior_conflicts(records, templates, anchors)),
        "other_variants": {
            "note": (
                "test questions naming an anchor that an earlier-round dev question of another variant named (any "
                "template): reported, not redrawn, like the cross-template reuse under entity_reuse"
            ),
            "questions": prior_other_variant_shares(records, templates, anchors),
        },
    }


# --------------------------------------------------------------------------
# One anchor entity per template (round-2 rule, applied when the profile asks)

UNIQUE_ANCHORS_RULE = (
    "no anchor entity (an entity a question names by label, read from its variant's anchor parameters) appears in two "
    "questions of one template, across both tiers and both splits: sampling skips a candidate naming an anchor that an "
    "accepted question of the template already names ('anchor_used'), and generation fails if the written set still "
    "has one. So no test question names an anchor of a dev question of its template, let alone of its variant"
)


def anchor_reuse(records: Sequence[Mapping[str, Any]], templates: Sequence[type[QuestionTemplate]]) -> dict[str, Any]:
    """Per template: questions, anchor entities, and the anchors named by more than one question (with their ids)."""
    by_id = {t.id: t for t in templates}
    out: dict[str, Any] = OrderedDict()
    for t in templates:
        owners: dict[str, list[str]] = OrderedDict()
        rows = sorted((r for r in records if r["template"] == t.id), key=lambda r: r["id"])
        for r in rows:
            for a in sorted(anchor_entities(by_id[t.id].variant(r["params"]["variant"]), r["params"])):
                owners.setdefault(a, []).append(r["id"])
        reused = {a: ids for a, ids in owners.items() if len(ids) > 1}
        dev_ids = {r["id"] for r in rows if r["split"] == DEV}
        out[t.id] = {
            "questions": len(rows),
            "anchor_entities": len(owners),
            "reused": reused,
            "test_naming_a_dev_anchor": sum(
                1 for ids in owners.values() if any(i in dev_ids for i in ids) and any(i not in dev_ids for i in ids)
            ),
        }
    return out


def check_unique_anchors(records: Sequence[Mapping[str, Any]], templates: Sequence[type[QuestionTemplate]]) -> None:
    """Raise if an anchor entity appears in two questions of one template."""
    broken = [(t, a, ids) for t, d in anchor_reuse(records, templates).items() for a, ids in d["reused"].items()]
    if broken:
        raise TemplateError(f"anchor entities named by two questions of one template: {broken[:5]}")


STEP_CHECK_RULE = (
    "every compositional question's steps must matter, checked on the store when it is drawn (rejection "
    "'step_not_load_bearing'): populations and constraints - the gold with any one population, constraint, negation or "
    "limit dropped (undo patterns, all must change the rows); comparisons - dropping the winner changes the answer and "
    "the compared values are pairwise distinct; signed differences - each of the two terms alone gives another value; "
    "paths - the answer comes through at least two entities at the anchor end of the path and no single one of them "
    "accounts for every answer item (fan-out)"
)


def step_checks_summary(gen: Generation, templates: Sequence[type[QuestionTemplate]] = ()) -> dict[str, Any]:
    """The step-check rule, each variant's declared steps, its outcome per question, and per-variant counts (profiles
    without structural steps)."""
    by_variant = Counter(
        next(f"{r['template']}/{r['params']['variant']}" for r in gen.records if r["id"] == q) for q in gen.step_checks
    )
    comp = [r for r in gen.records if r.get("tier") == "compositional"]
    declared = {
        f"{t.id}/{v.name}": {
            "rule": v.step_check.rule,
            "mode": v.step_check.mode if v.step_check.undo else None,
            "undo_steps": list(v.step_check.steps),
            "fan_out": v.step_check.fan_out is not None,
        }
        for t in templates
        for v in t.variants
        if v.step_check is not None
    }
    return {
        "rule": STEP_CHECK_RULE,
        "compositional_questions": len(comp),
        "questions_checked": len(gen.step_checks),
        "all_matter": all(c["matters"] for c in gen.step_checks.values()),
        **({"declared_steps": declared} if declared else {}),
        "by_variant": dict(sorted(by_variant.items())),
        "by_question": dict(sorted(gen.step_checks.items())),
    }


# --------------------------------------------------------------------------
# Output


def questions_jsonl(records: Iterable[dict[str, Any]]) -> str:
    lines = [json.dumps(r, ensure_ascii=False) for r in sorted(records, key=lambda r: r["id"])]
    return "\n".join(lines) + "\n"


def build_manifest(
    gen: Generation,
    *,
    seed: int,
    per_template: int,
    endpoint: str,
    questions_text: str,
    templates: Sequence[type[QuestionTemplate]] = TEMPLATES,
    generated_at: str | None = None,
    profile: Profile | None = None,
    confirm_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if profile is not None and profile.steps:
        templates = resolve_templates(templates, profile)
    if profile is not None and profile.resolve_variants is not None:
        templates = profile.resolve_variants(templates, profile.knobs)
    by_template: dict[str, Any] = OrderedDict()
    for t in templates:
        rows = [r for r in gen.records if r["template"] == t.id]
        by_template[t.id] = {
            "total": len(rows),
            DEV: sum(r["split"] == DEV for r in rows),
            TEST: sum(r["split"] == TEST for r in rows),
            "by_variant": dict(Counter(r["params"]["variant"] for r in rows)),
        }
        if profile and profile.two_tier:
            by_template[t.id]["by_tier"] = {
                tier: {
                    "total": sum(1 for r in rows if r["tier"] == tier),
                    DEV: sum(1 for r in rows if r["tier"] == tier and r["split"] == DEV),
                    TEST: sum(1 for r in rows if r["tier"] == tier and r["split"] == TEST),
                }
                for tier in profile.tier_counts(per_template)
            }
    times = list(gen.measured_seconds.values())
    extra: dict[str, Any] = {}
    if profile is not None:
        extra["profile"] = {
            "two_tier": profile.two_tier,
            "lookup_share": profile.lookup_share,
            "tier_counts_per_template": profile.tier_counts(per_template),
            "knobs": dict(profile.knobs),
            "notes": dict(profile.notes),
        }
        if profile.steps:
            extra["calibration_round"] = profile.calibration_round()
            extra["structural_steps"] = structural_steps_summary(gen, templates, profile)
        if gen.prior is not None:
            extra["prior_round_anchors"] = gen.prior
        if profile.unique_anchors_per_template:
            extra["anchor_reuse"] = {"rule": UNIQUE_ANCHORS_RULE, "by_template": anchor_reuse(gen.records, templates)}
        if not profile.steps and gen.step_checks:
            extra["step_checks"] = step_checks_summary(gen, templates)
        if not profile.steps and gen.comparisons:
            extra["comparisons"] = {
                "rule": (
                    "in each comparison variant the entities' mention order is a seeded permutation that puts the winner "
                    "in each mention slot as evenly as each (variant, split) allows; compared values are pairwise distinct"
                ),
                "by_variant": comparison_summary(gen.records, templates, gen.comparisons),
                "by_question": dict(sorted(gen.comparisons.items())),
            }
        revision = profile.revisions.get(profile.calibration_round())
        if revision is not None and revision.get("applies_to") == {"seed": seed, "per_template": per_template}:
            extra["revisions"] = dict(revision)
        if profile.audit_rules:
            extra["value_diversity"] = {"rule": VALUE_RULE, "strata": audit_value_strata(gen.records, templates)}
            extra["timing_rule"] = (
                "query times never decide the selection: a time-out of any query fails the run, and measured gold "
                "times are kept here only (gold_query_seconds_measured), not in questions.jsonl"
            )
            extra["gold_rows_sha256"] = hashlib.sha256(
                "\n".join(json.dumps([r["id"], r["gold_rows"]], ensure_ascii=False) for r in gen.records).encode("utf-8")
            ).hexdigest()
            extra["prepared"] = dict(gen.prepared)
        if profile.confirm_golds:
            by_variant = {t: dict(vs) for t, vs in gen.confirmation.items() if t != "_confirmed_golds"}
            done = gen.confirmation.get("_confirmed_golds", set())
            extra["gold_confirmation"] = {
                "rule": CONFIRM_RULE,
                "confirm_client": dict(confirm_settings or {}),
                "evaluated": sum(e["confirmed"] + e["replaced"] for t in by_variant.values() for e in t.values()),
                "confirmed": sum(e["confirmed"] for t in by_variant.values() for e in t.values()),
                "replaced": sum(e["replaced"] for t in by_variant.values() for e in t.values()),
                # questions whose own gold query was confirmed (looked up by its hash, not inferred)
                "questions_confirmed": sum(_sha(r["gold_sparql"]) in done for r in gen.records),
                "questions": len(gen.records),
                "by_variant": by_variant,
            }
        if profile.two_tier:
            extra["alternative_readings"] = {**readings_summary(gen.records), "clauses": reading_clauses(templates)}
            extra["cross_template_nested_pairs"] = cross_template_nested(gen.records)
            extra["counts_by_tier"] = {
                tier: {
                    "total": sum(1 for r in gen.records if r["tier"] == tier),
                    DEV: sum(1 for r in gen.records if r["tier"] == tier and r["split"] == DEV),
                    TEST: sum(1 for r in gen.records if r["tier"] == tier and r["split"] == TEST),
                }
                for tier in profile.tier_counts(per_template)
            }
    manifest = {
        "graph": profile.graph_id if profile else GRAPH_ID,
        "generator_version": profile.generator_version(GENERATOR_VERSION) if profile is not None else GENERATOR_VERSION,
        "seed": seed,
        "per_template": per_template,
        "endpoint": endpoint,
        **extra,
        "store": {
            **(dict(profile.store_identity) if profile else STORE_IDENTITY),
            "triples_per_graph": dict(sorted(gen.store_triples.items())),
        },
        "generated_at": generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "questions_file": "questions.jsonl",
        "questions_sha256": hashlib.sha256(questions_text.encode("utf-8")).hexdigest(),
        **({"gold_rows_sha256": gold_rows_sha256(gen.records)} if profile is not None else {}),
        "filters": {"min_rows": MIN_ROWS, "max_rows": MAX_ROWS, "max_query_seconds": MAX_SECONDS},
        "split": {
            "dev_fraction": 0.3,
            "min_dev_per_template": 3,
            "grouped_by": "base_question_id",
            "stratified_by": "variant (largest remainder within each template)",
        },
        "counts": {
            "total": len(gen.records),
            DEV: sum(r["split"] == DEV for r in gen.records),
            TEST: sum(r["split"] == TEST for r in gen.records),
            "by_template": by_template,
        },
        "gold_query_seconds_measured": {
            "note": (
                "questions.jsonl carries no query times, so the file is byte-identical across runs"
                if profile is not None and profile.audit_rules
                else "questions.jsonl rounds these up to whole seconds so the file is byte-identical across runs"
            ),
            "max": max(times) if times else None,
            "by_question": dict(sorted(gen.measured_seconds.items())),
            **(
                {
                    "warm_rule": (
                        f"each selected gold is re-run {WARM_RUNS} times back to back; the record's whole-second field "
                        f"comes from the fastest run, which must be under {WARM_GOLD_SECONDS:g} s or the run fails (a "
                        "first run can be slow on a cold cache); by_question holds the first (cold) runs"
                    ),
                    "warm_max": max(gen.warm_seconds.values()) if gen.warm_seconds else None,
                    "warm_by_question": dict(sorted(gen.warm_seconds.items())),
                }
                if gen.warm_seconds
                else {}
            ),
        },
        "row_counts": row_count_distribution(gen.records, templates),
        **(
            {
                "single_value_histograms": {
                    "rule": (
                        "value diversity: in each (variant, split) stratum of n one-row questions, at most "
                        "max(1, ceil(n / 4)) share one answer value; generation fails otherwise"
                    ),
                    "by_variant": value_strata(gen.records),
                }
            }
            if profile is not None and not profile.audit_rules  # audit profiles report value_diversity instead
            else {}
        ),
        "shape_strata": {
            "note": "sampling alternates between answer shapes, so multi-valued answers are over-sampled; shares compare the candidate pool (entities) with the questions sampled",
            "by_variant": {
                t: {v: s.strata_json() for v, s in vs.items() if s.strata_json() is not None}
                for t, vs in gen.stats.items()
                if any(s.strata_json() is not None for s in vs.values())
            },
        },
        "engine_sensitive": engine_sensitive_summary(gen.records, templates),
        "discrimination": discrimination_summary(gen.records, templates, readings_based=bool(profile and profile.two_tier)),
        "entity_reuse": entity_reuse(gen.records),
        "same_family_columns": {
            "rule": (
                "the scorer permutes answer columns only across value families, so a question with two columns of "
                "one family states their order (\"the A, then the B\"); generation fails otherwise"
            ),
            "by_variant": dict(
                sorted(
                    Counter(f"{r['template']}/{r['params']['variant']}" for r in gen.records if r["same_family_columns"]).items()
                )
            ),
        },
        "split_overlap": {
            "rule": (
                "across all templates and tiers, no dev answer may equal, contain or be contained in a test answer; "
                "one-row numeric answers are left out (their guessability is the value cap's job) and equal ones "
                "within a template are listed; answers that only share some rows are listed"
                if profile is not None and profile.audit_rules
                else "within a template, no dev answer may equal, contain or be contained in a test answer "
                "(sampling skips nested answers); answers that only share some rows are listed"
            ),
            "nested_pairs": len(gen.overlaps["nested"]),
            "intersecting_pairs": len(gen.overlaps["intersecting"]),
            "intersecting_by_template": dict(Counter(p["template"] for p in gen.overlaps["intersecting"])),
            "intersecting": gen.overlaps["intersecting"],
            **{k: gen.overlaps[k] for k in ("equal_single_value", "cross_template_intersecting") if k in gen.overlaps},
        },
        "sampling": {
            t: {v: s.to_json() for v, s in vs.items()} for t, vs in gen.stats.items()
        },
        "templates": [t.to_json(extended=bool(profile and profile.two_tier)) for t in templates],
    }
    for section, rule in (profile.rule_texts if profile is not None else {}).items():
        manifest[section]["rule"] = rule
    return manifest


STEP_RULES = {
    "axis_precedence": (
        "each compositional variant is raised along exactly one axis, read from its round-1 difficulty declaration "
        "in this order: a set operation -> populations; a comparison of entities -> populations (the entities "
        "compared); negation -> negation depth; grouping (GROUP BY with HAVING or ORDER BY / LIMIT) -> grouping "
        "constraints; otherwise (paths, distinct counts across a join) -> path hops. A variant's other constructs "
        "keep their round-1 values"
    ),
    "hops": (
        "path hops are the edges of the gold pattern that join the anchors and the answer: a property path or "
        "rdfs:subClassOf* counts as one edge; type tests, up:reviewed and the label edges that resolve an anchor do "
        "not count; a label edge that gives an answer column does. The hop step adds one edge at the anchor end: an "
        "organism anchor becomes a taxon and its subtree (rdfs:subClassOf*), a disease anchor becomes a keyword its "
        "disease entries refer to (rdfs:seeAlso)"
    ),
    "populations": "one more population of the kind the round-1 form names last (one more keyword, GO term, protein or reference protein)",
    "grouping_constraints": "one more HAVING constraint on the grouped count: at most having_max (the same knob for every grouping variant)",
    "negation_depth": "one more negation nested inside the existing one (every activity linked: NOT EXISTS an activity with NOT EXISTS a reaction)",
    "load_bearing": (
        "every raised question's step must matter, checked on the store (rejection 'step_not_load_bearing'): set "
        "operations - dropping any one population of the raised kind changes the gold rows; comparisons - only that "
        "the compared values are pairwise distinct (no ties): round 2 redrew its items, so no entity is 'the added "
        "one' (study lead's ruling on experiments#12); the winner's mention slot is balanced instead; "
        "grouping - the round-1 form (the step undone) gives other rows; negation - both the round-1 form and the "
        "positive reading without the nested negation (some activity linked) give other rows; hops - answer items come through "
        "at least two entities at the near end of the added edge and no single one of them accounts for all"
    ),
}


def structural_steps_summary(gen: Generation, templates: Sequence[type[QuestionTemplate]], profile: Profile) -> dict[str, Any]:
    """The step table (variant, axis, round-1 and round-2 values), the rules, per-question checks and shortfalls."""
    applied = {f"{t.id}/{v.name}" for t in templates for v in t.variants}
    by_variant: dict[str, Any] = OrderedDict()
    for key, step in profile.steps.items():
        raised_now = profile.delta(step.axis) == 1
        by_variant[key] = {**step.to_json(), "applied": raised_now and key in applied}
    shortfalls = [
        {"variant": f"{t}/{v}", "quota": s.quota, "accepted": s.accepted}
        for t, vs in gen.stats.items()
        for v, s in vs.items()
        if s.quota is not None and s.accepted < s.quota
    ]
    checks = gen.step_checks
    raised = {k for k, d in by_variant.items() if d["applied"]}
    comp = [r for r in gen.records if f"{r['template']}/{r['params']['variant']}" in raised]
    dev, test = sum(r["split"] == DEV for r in comp), sum(r["split"] == TEST for r in comp)
    return {
        "knobs": {f"{a}_delta": profile.delta(a) for a in sorted({s.axis for s in profile.steps.values()})},
        "rules": STEP_RULES,
        "changed_questions": {
            "note": (
                f"every compositional question is in its raised form, so all {dev + test} changed against round 1: the "
                f"{dev} dev questions (the calibration pool) and the {test} test questions alike. The test questions use "
                "the same template settings as the calibrated dev questions; no test question has been run"
            ),
            "compositional": {DEV: dev, TEST: test},
        },
        "by_variant": by_variant,
        "step_checks": {
            "questions": len(checks),
            "by_variant": dict(sorted(Counter(
                next(r["template"] + "/" + r["params"]["variant"] for r in gen.records if r["id"] == q) for q in checks
            ).items())),
            "by_question": dict(sorted(checks.items())),
        },
        "shortfalls": shortfalls,
        "comparisons": {
            "rule": (
                "in each comparison variant the entities' mention order is a seeded permutation that puts the winner (for "
                "heavier_than_reference the binding reference, whose heaviest sequence sets the limit) in each mention "
                "slot as evenly as each (variant, split) allows; compared values are pairwise distinct (no ties). The "
                "draw lists entities in accession order, which follows entry age, so mention order would otherwise be "
                "a shortcut (study lead's audit, experiments#12)"
            ),
            "by_variant": comparison_summary(gen.records, templates, gen.comparisons),
            "by_question": dict(sorted(gen.comparisons.items())),
        },
    }


def reading_clauses(templates: Sequence[type[QuestionTemplate]]) -> dict[str, Any]:
    """Each variant's alternative readings with the exact clause each must find in the question text."""
    return {
        f"{t.id}/{v.name}": [{"reading": r.name, "description": r.description, "clause": r.clause} for r in v.readings]
        for t in templates
        for v in t.variants
        if v.readings
    }


def gold_rows_sha256(records: Sequence[dict[str, Any]]) -> str:
    """The gold-row hash (as on the Rhea set): sha256 of one JSON line [id, gold_rows] per record, in id order."""
    lines = (json.dumps([r["id"], r["gold_rows"]], ensure_ascii=False) for r in sorted(records, key=lambda r: r["id"]))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def write_outputs(out_dir: Path, gen: Generation, **manifest_args: Any) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = questions_jsonl(gen.records)
    qpath = out_dir / "questions.jsonl"
    qpath.write_text(text, encoding="utf-8")
    manifest = build_manifest(gen, questions_text=text, **manifest_args)
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return qpath, mpath
