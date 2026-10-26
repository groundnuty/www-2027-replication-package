"""Generate template questions with gold answers from a pinned store.

Each template's questions come in two tiers: about one third "lookup" (the
audited templates of generator 0.2) and two thirds "compositional" (chained or
combined steps; ``harness.qgen_rhea.compositional``). The lookup tier is sampled
first and does not depend on the difficulty knobs, so it stays the same across
calibration rounds; the compositional tier is sampled next, sharing the lookup
tier's used entities and answers, so no answer of a template nests in another
whichever tier or split it lands in.

The output is deterministic: the same seed, knobs and store give a
byte-identical ``questions.jsonl``. No measured quantity is written to it
(External review of 355f04e, finding 4): gold query times and the generation time
are in ``manifest.json`` only, which is not part of the determinism check.
Query times never decide which questions are selected either: a selected gold
over the hard cap (HARD_GOLD_SECONDS), or any query time-out, fails the run.
Each selected gold is run twice more, back to back; the manifest keeps the
first (cold) time and the faster warm time, and a gold that is not under
MAX_GOLD_SECONDS even warm fails the run.

Calibration round 2 (``harness.qgen_rhea.compositional_r2``): the knobs'
``<axis>_delta`` raise every active compositional variant one structural step.
From round 2 on, every template's lookup tier is sampled before any
compositional tier, so the compositional draw cannot reach the lookup tier
through the shared answers (round 1 keeps its interleaved order, which the
frozen file was drawn with; its lookup tier is the same either way, a test
checks it). Records carry the generator version of their round.
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

from harness.qgen_rhea.base import (
    STRUCTURAL_AXES,
    CHEBI_GRAPH,
    COMPOSITIONAL,
    HOMONYM_KIND,
    HOMONYM_NOTES,
    DATABASE_PHRASES,
    DISCRIMINATIONS,
    FORMS,
    GRAPH_ID,
    ID_PREFIX,
    LOOKUP,
    RHEA_GRAPH,
    TIERS,
    Candidate,
    Knobs,
    QuestionTemplate,
    TemplateError,
    Variant,
    _substitute,
    database_names_in,
)
from harness.qgen_rhea.compositional_r2 import (
    BOUNDS_RULE,
    COLUMN_RENAMES,
    ROUND_1_BOUNDS,
    STEP_RULES,
    STEPS,
    resolve_templates,
    step_of,
)
from harness.qgen_rhea.readings import (
    RHEA_READING_NAMES,
    ReadingError,
    check_readings,
    nests_graph,
    neutral_text,
    reading_base,
    reading_changes,
)
from harness.qgen_rhea.sampling import TWO_VALUED_SHARE, VALUE_SHARE, Accepted, quotas, rng_for, select, shuffled_groups, value_cap
from harness.qgen_rhea.split import DEV, TEST, assign_splits, dev_count
from harness.qgen_rhea.templates import TEMPLATES
from harness.qgen_rhea.text_guard import AGENT_VISIBLE_FIELDS, record_database_names
from harness.sparql import QueryResult, SparqlClient, SparqlError
from harness.terms import IRI, XSD, BNode, Literal, Row, row_from_json, row_to_json

# The generator version each calibration round's records carry: round 1 is the frozen file 00e84760...
# Round 2 is 0.4.3 (study lead, design log): 0.4.0 named the earlier round-2 sets (a0024f4, e892710), 0.4.1 the set
# 4050bfe5... (3ea978e) and 0.4.2 the set 475670e8... (01eeb51); (id, generator_version) stays a key.
ROUND_VERSIONS = {1: "0.3.0", 2: "0.4.3"}
GENERATOR_VERSION = ROUND_VERSIONS[2]
STORE_IDENTITY = {
    "engine": "QLever 9bebc60",
    "image_digest": "sha256:dc93f8df62de83bff35374ea3f27af0461aa5453c643c7088cb9c64f71c6315c",
}
MIN_ROWS = 1
MAX_ROWS = 50
# Query times never decide which questions are selected (External reviews of 6e5ce1a): a gold that takes longer
# than HARD_GOLD_SECONDS fails the run instead, and a time-out of any query fails it too. MAX_SECONDS is the
# clients' time-out (a backstop at the hard cap); MAX_GOLD_SECONDS is what the integration test asks of each
# shipped gold on a warm store.
HARD_GOLD_SECONDS = 60.0
MAX_SECONDS = HARD_GOLD_SECONDS
MAX_GOLD_SECONDS = 2.0
# Round 1 of the calibration (ruling in the design log): the knobs' defaults. Round 2 raises every axis one step.
DEFAULT_KNOBS = Knobs()
ROUND_2_KNOBS = replace(DEFAULT_KNOBS, **{f"{a}_delta": 1 for a in STRUCTURAL_AXES})
CALIBRATION_ROUNDS = {1: DEFAULT_KNOBS, 2: ROUND_2_KNOBS}
# Each selected gold is re-run this many times back to back (warm times in the manifest).
WARM_RUNS = 2
# The agent settings each round runs with (study lead's confirmation of round 2 in the design log): round 1's
# caps (30 turns, 900 s) belong to round 1 only; round 2 runs with the frozen caps of change-log entry 23.
AGENT_SETTINGS = {
    1: {"max_turns": 30, "max_seconds": 900},
    2: {"max_turns": 60, "max_seconds": 1800, "max_output_tokens_per_turn": "16k"},
}


def generator_version(knobs: Knobs) -> str:
    """The version a setting's records carry: its calibration round's, or the current one outside the rounds."""
    return ROUND_VERSIONS.get(knobs.calibration_round(), GENERATOR_VERSION)
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
    # Size of the population the variant's filter acts on (compositional variants), or None.
    population: int | None = None
    # A raised variant's step check (step_outcome), None for other variants.
    step: dict[str, Any] | None = None


@dataclass
class VariantStats:
    candidates: int = 0
    entities: int = 0
    evaluated: int = 0
    accepted: int = 0
    rejected: Counter = field(default_factory=Counter)
    # Candidates dropped from the pool because a label names another entity of its kind.
    ambiguous_labels: int = 0
    # Candidates kept under the within-kind rule that the strict rule of 355f04e dropped.
    kept_by_kind_rule: int = 0
    # Entities per answer-shape stratum in the pool, and questions per stratum sampled.
    pool_strata: Counter = field(default_factory=Counter)
    sampled_strata: Counter = field(default_factory=Counter)
    # The variant's share of its tier's questions.
    quota: int | None = None
    # Accepted questions without all of the variant's preferred properties (taken because the preferred pool ran out).
    non_preferred: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            **({"quota": self.quota} if self.quota is not None else {}),
            "candidates": self.candidates,
            "entities": self.entities,
            "dropped_ambiguous_label": self.ambiguous_labels,
            "kept_by_kind_rule": self.kept_by_kind_rule,
            "evaluated": self.evaluated,
            "accepted": self.accepted,
            "accepted_non_preferred": self.non_preferred,
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
    overlaps: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {"nested": [], "intersecting": [], "equal_single_value": []}
    )
    knobs: Knobs = DEFAULT_KNOBS
    tier_counts: dict[str, int] = field(default_factory=dict)
    anchor_check: dict[str, Any] = field(default_factory=dict)
    reading_check: dict[str, Any] = field(default_factory=dict)
    # Raised variants' step checks by question id (calibration round 2).
    step_checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Comparison variants (round 2): per question the winner (or binding reference) and its mention slot.
    comparisons: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Each selected gold's faster time of WARM_RUNS back-to-back runs after the first.
    warm_seconds: dict[str, float] = field(default_factory=dict)
    # The previous-round anchor rule (check_test_anchors) and its input; None without a previous round (round 1).
    test_anchor_check: dict[str, Any] | None = None


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
    """The answer as compared by the overlap guards.

    A one-row answer of numbers only (a count, a difference) is identified with
    the entities the question names, so two questions whose counts happen to
    be equal are not taken for the same answer (audit of 2026-10-18 08:50,
    fix 1). Every other answer is its set of rows.
    """
    if len(rows) == 1 and rows[0] and all(_is_number(c) for c in rows[0]):
        named = sorted(str(a["entity"]) for a in anchors)
        return frozenset({json.dumps({"anchors": named, "row": rows[0]}, sort_keys=True, ensure_ascii=False)})
    return row_set(rows)


class OverlapError(ValueError):
    """A dev question's answer equals, contains or is contained in a test question's."""


def _cell_text(cell: Any) -> str:
    return "" if cell is None else str(cell.get("value"))


def _single_value(rows: Sequence[list[dict[str, str] | None]]) -> bool:
    return len(rows) == 1 and bool(rows[0]) and all(_is_number(c) for c in rows[0])


def split_overlaps(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Dev/test pairs whose actual gold-row sets overlap, across all templates and tiers.

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


def discrimination_summary(records: Sequence[dict[str, Any]], templates: Sequence[type[QuestionTemplate]]) -> dict[str, Any]:
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
            "by_tier": {
                tier: {
                    "direct_vs_transitive": tally([r for r in rows if r["tier"] == tier], "discriminates_direct_vs_transitive"),
                    "graph_scope": tally([r for r in rows if r["tier"] == tier], "discriminates_graph_scope"),
                }
                for tier in TIERS
            },
            "by_variant": {
                v.name: {
                    "direct_vs_transitive": tally([r for r in rows if r["params"]["variant"] == v.name], "discriminates_direct_vs_transitive"),
                    "graph_scope": tally([r for r in rows if r["params"]["variant"] == v.name], "discriminates_graph_scope"),
                }
                for v in t.variants
                if any(r["params"]["variant"] == v.name for r in rows)
            },
        }
    return {
        "rule": (
            "direct_vs_transitive: swapping rdfs:subClassOf and rdfs:subClassOf+ in the gold (not the master-reaction "
            "type test) changes the rows; graph_scope: dropping the GRAPH clauses changes the rows; null when not "
            "applicable or the variant query failed"
        ),
        "by_template": out,
    }


def entity_reuse(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Entities named by more than one question, and whether they straddle the split."""
    by_entity: dict[str, list[dict[str, Any]]] = OrderedDict()
    for r in records:
        for e in sorted({a["entity"] for a in r.get("anchors", ())}):
            by_entity.setdefault(e, []).append(r)
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
                "tier": v.tier,
                "distribution": {str(n): c for n, c in sorted(Counter(counts).items())},
                "min": counts[0],
                "median": counts[len(counts) // 2],
                "max": counts[-1],
                "multi_row": sum(n > 1 for n in counts),
                "single_row": sum(n == 1 for n in counts),
            }
    return out


def answer_value(rows: Sequence[list[dict[str, str] | None]]) -> str:
    """The answer value of a single-row question, as compared by the value-diversity rule."""
    return " | ".join(_cell_text(c) for c in rows[0])


VALUE_RULE = (
    "value diversity (design ruling in the design log): in each (variant, split) stratum of n single-value questions "
    f"(one-row answers: counts, differences, single literals), at most max(1, ceil({VALUE_SHARE} * n)) questions "
    "share one answer value (n=3 -> 1, n=7 -> 2, n=9 -> 3, n=20 -> 5); a two-valued variant is exempt but keeps "
    f"each value at most {TWO_VALUED_SHARE:.0%} of a stratum. Sampling caps each variant's values at the same "
    "formula over its quota, and the split spreads equal values over dev and test; generation fails if a stratum "
    "breaks the rule"
)


def value_strata(records: Sequence[dict[str, Any]], templates: Sequence[type[QuestionTemplate]]) -> dict[str, Any]:
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
                c = Counter(answer_value(r["gold_rows"]) for r in rows if r["split"] == split)
                n = sum(c.values())
                entry[split] = {
                    "questions": n,
                    "cap": value_cap(n, v.two_valued),
                    "max_share_count": max(c.values(), default=0),
                    "values": dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))),
                }
            out[f"{t.id}/{v.name}"] = entry
    return out


def check_value_diversity(records: Sequence[dict[str, Any]], templates: Sequence[type[QuestionTemplate]]) -> dict[str, Any]:
    """Raise if a (variant, split) stratum has more questions sharing one value than its cap allows."""
    strata = value_strata(records, templates)
    failures = [
        (key, split, d[split]["max_share_count"], d[split]["cap"])
        for key, d in strata.items()
        for split in (DEV, TEST)
        if d[split]["max_share_count"] > d[split]["cap"]
    ]
    if failures:
        raise TemplateError(f"value diversity broken (variant, split, most common count, cap): {failures}")
    return strata


def row_count_by_tier(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for tier in TIERS:
        counts = sorted(r["gold_row_count"] for r in records if r["tier"] == tier)
        if counts:
            out[tier] = {
                "distribution": {str(n): c for n, c in sorted(Counter(counts).items())},
                "min": counts[0],
                "median": counts[len(counts) // 2],
                "max": counts[-1],
                "single_row": sum(n == 1 for n in counts),
            }
    return out


TEMPORAL_DATATYPES = frozenset(XSD + t for t in ("date", "dateTime", "time", "gYear", "gYearMonth", "duration"))
# How a question states its column order: "the A, then the B" (lookup tier) or
# "each A with its B" / "each A with each of its B" (compositional tier).
COLUMN_ORDER_MARKER = re.compile(r", then (?:the|its) |\beach\b[^.?;]*? with (?:each of )?(?:its|their) ")


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


def check_stated_order(where: str, text: str, same_family: bool) -> None:
    """Two same-family columns need their order stated in the question (scorer-rules-v4)."""
    if same_family and not COLUMN_ORDER_MARKER.search(text):
        raise TemplateError(
            f"{where}: two answer columns share a value family, so the question must state their order "
            f"(\"the A, then the B\" or \"each A with its B\"): {text}"
        )


def reading_summary(records: Sequence[dict[str, Any]], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per reading: questions checked, answers changed (or unknown), and clauses present among those."""
    by_reading: dict[str, Any] = OrderedDict()
    bases = list(RHEA_READING_NAMES) + sorted(
        {reading_base(n) for res in results.values() for n in res} - set(RHEA_READING_NAMES)
    )
    by_id = {r["id"]: r for r in records}
    for base in bases:
        # Per question: checked if any reading of this kind applied (one per participation or counted
        # variable); changed if any changed the answer; the clause present if every changing one had its clause.
        per_q = {
            qid: [x for n, x in res.items() if reading_base(n) == base] for qid, res in results.items()
        }
        checked = [qid for qid, xs in per_q.items() if xs]
        changed = [qid for qid in checked if any(x.changed is not False for x in per_q[qid])]
        by_reading[base] = {
            "checked": len(checked),
            "answer_changed": sum(1 for qid in changed if any(x.changed for x in per_q[qid])),
            "change_unknown": sum(1 for qid in changed if any(x.changed is None for x in per_q[qid])),
            "clause_present": sum(1 for qid in changed if all(x.clause for x in per_q[qid] if x.changed is not False)),
            "alternatives_checked": sum(len(xs) for xs in per_q.values()),
            "by_tier": {tier: sum(by_id[qid]["tier"] == tier for qid in changed) for tier in TIERS},
            "by_variant": dict(sorted(Counter(by_id[qid]["params"]["variant"] for qid in changed).items())),
        }
    return {
        "rule": (
            "every question's gold is rewritten into the alternative readings a reasonable reader could take "
            "(harness/qgen_rhea/readings.py) and run on the store; when a reading changes the answer, the question "
            "states the matching scope (its clause pattern, matched with labels replaced by tokens), or generation "
            "fails. one_side is tried left-only and right-only for every participation (side variable) and "
            "non_distinct_count for every counted variable; a row counts a question once per reading kind"
        ),
        "questions_with_a_changing_reading": sum(1 for r in records if r["scope_readings_changed"]),
        "engine_note": (
            "no gold or alternative query nests a GRAPH clause inside another: QLever 9bebc60 evaluates the inner "
            "one against the outer graph (questions whose natural query does so are tagged qlever_nested_graph)"
        ),
        "by_reading": by_reading,
    }


def check_no_database_names(where: str, text: str) -> None:
    """Question text names no database (ruling in the design log)."""
    found = database_names_in(text)
    if found:
        raise TemplateError(f"{where}: question text names a database ({', '.join(found)}): {text}")


def check_agent_visible(record: Mapping[str, Any]) -> None:
    """No agent-visible field of a record names a database (study lead's audit, design log; calibration round 2
    on): the text, the answer-column names and the scoring rules (text_guard.AGENT_VISIBLE_FIELDS)."""
    found = record_database_names(record, database_names_in)
    if found:
        raise TemplateError(f"{record.get('id')}: agent-visible fields name a database: {found}")


def reading_notes(
    client: SparqlClient, variant: Variant, params: Mapping[str, Any], gold_sparql: str, gold_rows: list
) -> dict[str, bool]:
    """The note flags of a question: each reading with a note_param (and a note on the variant) that changes the
    answer sets its flag, so the text states that scope exactly where it matters; a failed query fails the run."""
    flags: dict[str, bool] = {}
    for reading in variant.readings:
        if reading.note_param is None or reading.note_param not in variant.notes:
            continue
        changed = reading_changes(client, reading, variant, params, gold_sparql, gold_rows)
        if changed is None:
            raise TemplateError(f"{variant.name}: reading {reading.name} could not be run")
        if changed is True:
            flags[reading.note_param] = True
    return flags


def membership_swapped(sparql: str, mode: str) -> str | None:
    """The gold with its membership relation swapped (direct <-> transitive); None for "n.a."

    The master-reaction type test (rdfs:subClassOf rh:Reaction) is not the
    relation a question asks about and is left alone.
    """
    if mode == "transitive":
        return sparql.replace("rdfs:subClassOf+", "rdfs:subClassOf")
    if mode == "direct":
        return re.sub(r"rdfs:subClassOf(?![+*])(?=\s+(?!rh:Reaction\b))", "rdfs:subClassOf+", sparql)
    return None


def graph_unscoped(sparql: str) -> str | None:
    """The gold without its GRAPH clauses (so it reads the union graph); None if it has none."""
    out = re.sub(r"GRAPH\s+<[^>]*>\s*", "", sparql)
    return out if out != sparql else None


def discrimination(client: SparqlClient, name: str, sparql: str, mode: str, gold_rows: list) -> bool | None:
    """Whether the variant query for ``name`` gives other rows than the gold; None if n.a. or it fails."""
    alt = membership_swapped(sparql, mode) if name == "direct_vs_transitive" else graph_unscoped(sparql)
    if alt is None or alt == sparql:
        return None
    try:
        return canonical_rows(client.query(alt).rows) != gold_rows
    except SparqlError as exc:
        if exc.kind == "timeout":  # a time-out must not decide a preference (or the sample)
            raise
        return None


def engine_reasons(client: SparqlClient, variant: Any, params: dict[str, Any], gold_rows: list) -> list[str]:
    """Reason codes of the variant's engine probes whose natural query does not give the gold rows."""
    reasons = []
    for probe in variant.engine_probes:
        try:
            rows = canonical_rows(client.query(variant.probe_sparql(probe, params)).rows)
        except SparqlError as exc:
            if exc.kind == "timeout":
                raise
            rows = None
        if rows != gold_rows:
            reasons.append(probe.reason)
    return sorted(reasons)


def store_triples(client: SparqlClient) -> dict[str, int]:
    result = client.query("SELECT ?g (COUNT(*) AS ?n) WHERE { GRAPH ?g { ?s ?p ?o } } GROUP BY ?g")
    return {row[0].value: int(row[1].lexical) for row in result.rows}


def lookup_count(per_template: int) -> int:
    """Lookup questions per template: one third rounded half up, at least one (43 -> 14, 10 -> 3)."""
    return max(1, math.floor(per_template / 3 + 0.5)) if per_template > 0 else 0


# --------------------------------------------------------------------------
# Label anchors: a label names exactly one entity of its kind

_NAMING = "<http://rdf.rhea-db.org/name>|<http://www.w3.org/2000/01/rdf-schema#label>"
RH_NAME = "http://rdf.rhea-db.org/name"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
# The kind of entity a (graph, naming property) pair names.
LABEL_KINDS = {
    (RHEA_GRAPH, RH_NAME): "compound",
    (CHEBI_GRAPH, RDFS_LABEL): "chemical_class",
    (RHEA_GRAPH, RDFS_LABEL): "reaction",
}
LABEL_RULE = (
    "every entity a question names by label is the only entity of its kind with that exact string: compound "
    "entries by rh:name in the reaction graph, chemical classes by rdfs:label in the class hierarchy, reactions "
    "by rdfs:label in the reaction graph; a compound entry and a chemical class may share a label, and the "
    "question then says which one it means (a note after the label)"
)


def label_kind(graph: str, prop: str) -> str:
    return LABEL_KINDS.get((graph, prop), "other")


def _sparql_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass
class LabelIndex:
    """Strings that name more than one subject (via rh:name or rdfs:label, in any graph), by kind.

    ``subjects[label][kind]`` is the set of subjects of that kind named by the
    label. A label not in the index names a single subject.
    """

    subjects: dict[str, dict[str, set[str]]] = field(default_factory=dict)

    @classmethod
    def from_store(cls, client: SparqlClient) -> "LabelIndex":
        query = f"""SELECT ?s ?e ?g ?p WHERE {{
  {{ SELECT ?s WHERE {{ GRAPH ?g1 {{ ?e1 {_NAMING} ?l1 . }} BIND(STR(?l1) AS ?s) }} GROUP BY ?s HAVING (COUNT(DISTINCT ?e1) > 1) }}
  {{ SELECT DISTINCT ?s ?e ?g ?p WHERE {{ VALUES ?p {{ <{RH_NAME}> <{RDFS_LABEL}> }} GRAPH ?g {{ ?e ?p ?l . }} BIND(STR(?l) AS ?s) }} }}
}}
ORDER BY ?s ?e ?g ?p"""
        index = cls()
        for row in pool_rows(client.query(query)):
            index.add(row["s"], row["e"], label_kind(row["g"], row["p"]))
        return index

    def add(self, label: str, entity: str, kind: str) -> None:
        self.subjects.setdefault(label, {}).setdefault(kind, set()).add(entity)

    def ok(self, label: str, entity: str, kind: str) -> bool:
        """The label names this entity and no other entity of its kind; other kinds only as the homonym kind allows."""
        by_kind = self.subjects.get(label)
        if by_kind is None:
            return True
        if by_kind.get(kind, {entity}) != {entity}:
            return False
        allowed = {kind, HOMONYM_KIND.get(kind)}
        return set(by_kind) <= allowed

    def strictly_ok(self, label: str, entity: str) -> bool:
        """The rule of generator 0.3.0 at 355f04e: the label names no other subject at all."""
        by_kind = self.subjects.get(label)
        return by_kind is None or set().union(*by_kind.values()) == {entity}

    def homonym(self, label: str, kind: str) -> bool:
        """Whether the label also names an entity of the other kind (compound entry against chemical class)."""
        other = HOMONYM_KIND.get(kind)
        return bool(other and self.subjects.get(label, {}).get(other))


def check_label_anchors(client: SparqlClient, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Every label anchor is the only entity of its kind with that label, and says so when the other kind shares it.

    Independent of the index used for sampling: one query looks up each label
    as an exact string in every graph. Raises on any failure.
    """
    labels = sorted({a["name"] for r in records for a in r["anchors"] if a["named_by"] == "label"})
    found = LabelIndex()
    for start in range(0, len(labels), 200):
        batch = labels[start:start + 200]
        query = f"""SELECT ?l ?e ?g ?p WHERE {{
  VALUES ?l {{ {" ".join(_sparql_string(x) for x in batch)} }}
  VALUES ?p {{ <{RH_NAME}> <{RDFS_LABEL}> }}
  GRAPH ?g {{ ?e ?p ?l . }}
}}"""
        for row in pool_rows(client.query(query)):
            found.add(row["l"], row["e"], label_kind(row["g"], row["p"]))
    failures = []
    checked = homonyms = 0
    by_kind: Counter = Counter()
    for r in records:
        for a in r["anchors"]:
            if a["named_by"] != "label":
                continue
            checked += 1
            by_kind[a["kind"]] += 1
            kinds = found.subjects.get(a["name"], {})
            if kinds.get(a["kind"]) != {a["entity"]} or not found.ok(a["name"], a["entity"], a["kind"]):
                failures.append((r["id"], a["name"], {k: sorted(v) for k, v in kinds.items()}))
            elif found.homonym(a["name"], a["kind"]) != a["homonym"]:
                failures.append((r["id"], a["name"], "homonym note does not match the store"))
            elif a["homonym"]:
                homonyms += 1
                if HOMONYM_NOTES[a["kind"]] not in r["question"]:
                    failures.append((r["id"], a["name"], "homonym note missing from the text"))
    if failures:
        raise TemplateError(f"label anchors that fail the rule: {failures[:5]}")
    return {
        "rule": LABEL_RULE,
        "label_anchors_checked": checked,
        "by_kind": dict(sorted(by_kind.items())),
        "shared_with_the_other_kind": homonyms,
        "identifier_anchors": sum(1 for r in records for a in r["anchors"] if a["named_by"] != "label"),
    }


# --------------------------------------------------------------------------
# Calibration round 2: step checks and comparison mention slots


def _cell_value(cell: Any) -> Any:
    if isinstance(cell, IRI):
        return cell.value
    if isinstance(cell, Literal) and cell.datatype in NUMERIC_DATATYPES:
        number = float(cell.lexical)
        return int(number) if number.is_integer() else number
    return None if cell is None else str(getattr(cell, "lexical", cell))


def step_outcome(client: SparqlClient, variant: Any, params: Mapping[str, Any], gold_rows: list) -> dict[str, Any]:
    """Whether a raised variant's step matters for one question (base.StepCheck). Time-outs raise."""
    check = variant.step_check
    out: dict[str, Any] = {"rule": check.rule}
    matters = True

    def render(pattern: str) -> str:
        return variant._prefixes() + _substitute(pattern, {**params, **variant._graphs()})

    if check.distinct is not None:
        values: dict[Any, Any] = {}
        conflicting: set[Any] = set()
        for row in client.query(render(check.distinct)).rows:
            entity, value = _cell_value(row[0]), _cell_value(row[1])
            if entity in values and values[entity] != value:  # two values for one entity: no single winner
                conflicting.add(entity)
            values.setdefault(entity, value)
        named = [params[slot[0]] for slot in variant.slots] or [params[p] for p in check.compared]
        expected = set(named) if named else set(values)
        distinct = (
            not conflicting
            and set(values) == expected
            and len(values) == (len(named) or len(values))
            and None not in values.values()
            and len(set(values.values())) == len(values)
        )
        out["values"] = dict(sorted(values.items()))
        if conflicting:
            out["conflicting"] = sorted(conflicting)
        out["distinct"] = distinct
        matters = matters and distinct
    if check.undo:
        changed = [canonical_rows(client.query(render(pattern)).rows) != gold_rows for pattern in check.undo]
        out["undo_changed"] = changed
        matters = matters and (all(changed) if check.mode == "every" else any(changed))
    if check.fan_out is not None:
        by_near: dict[str, set[str]] = OrderedDict()
        for row in client.query(render(check.fan_out)).rows:
            cells = row_to_json(row)
            by_near.setdefault(json.dumps(cells[0], sort_keys=True), set()).add(json.dumps(cells[1:], sort_keys=True))
        items = set().union(*by_near.values()) if by_near else set()
        covered = any(v == items for v in by_near.values())
        out["fan_out"] = len(by_near)
        out["one_entity_covers_all"] = covered
        matters = matters and len(by_near) >= 2 and not covered
    out["matters"] = matters
    return out


def balanced_targets(n: int, k: int, rng: Any) -> list[int]:
    """Mention slots (0-based) for the winners of ``n`` questions over ``k`` slots: as even as n allows, in seeded order."""
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


def winner_slot(variant: Any, params: Mapping[str, Any], values: Mapping[Any, Any]) -> tuple[int, Any]:
    """(slot index, entity) of the winner: the entity with the highest ("max") or lowest ("min") value."""
    pick = max if variant.winner == "max" else min
    entity = pick(sorted(values), key=lambda e: values[e])
    return next(i for i, slot in enumerate(variant.slots) if params[slot[0]] == entity), entity


def balance_slots(
    picked: list, qids: Sequence[str], splits: Mapping[str, str], seed: int, client: SparqlClient
) -> tuple[list, dict[str, dict[str, Any]]]:
    """Comparison questions with the winner's mention slot balanced within each (variant, split).

    The compared entities' order in the text is a seeded permutation. The gold
    reads the entities as a set (or by entity, not by slot), so its rows do not
    change: checked on the store whenever the gold text changes, and the step
    check is re-run with the new order.
    """
    groups: dict[tuple[str, str, str], list[int]] = OrderedDict()
    for k, (t, acc) in enumerate(picked):
        if t.variant(acc.candidate.variant).slots:
            groups.setdefault((t.id, acc.candidate.variant, splits[qids[k]]), []).append(k)
    out, info = list(picked), {}
    for (tid, name, split), ks in groups.items():
        variant = picked[ks[0]][0].variant(name)
        targets = balanced_targets(len(ks), len(variant.slots), rng_for(seed, "slots", tid, name, split))
        for k, target in zip(ks, targets):
            template, acc = picked[k]
            params, gold = acc.candidate.params, acc.payload
            winner, entity = winner_slot(variant, params, gold.step["values"])
            new = place_winner(params, variant.slots, winner, target, rng_for(seed, "slot-order", qids[k]))
            sparql = variant.gold_sparql(new)
            if sparql != gold.sparql and canonical_rows(client.query(sparql).rows) != gold.rows:
                raise TemplateError(f"{qids[k]}: reordering the compared entities changed the gold rows")
            step = step_outcome(client, variant, new, gold.rows)
            if not step["matters"] or winner_slot(variant, new, step["values"]) != (target, entity):
                raise TemplateError(f"{qids[k]}: the step check or the winner changed after reordering the compared entities")
            out[k] = (template, replace(acc, candidate=replace(acc.candidate, params=new), payload=replace(gold, sparql=sparql, step=step)))
            info[qids[k]] = {
                "variant": f"{tid}/{name}",
                "split": split,
                "winner": entity,
                "winner_slot": target + 1,
                "values": step["values"],
                **({"binding_is_the_anchor": entity == new["entity"]} if "entity" in new and variant.winner == "min" else {}),
            }
    return out, info


def comparison_summary(records: Sequence[dict[str, Any]], templates: Sequence[type[QuestionTemplate]], info: Mapping[str, Any]) -> dict[str, Any]:
    """Per comparison variant and split: how often the winner (or binding reference) sits in each mention slot."""
    out: dict[str, Any] = OrderedDict()
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
                    "winner_slot_histogram": {str(slot): hist.get(slot, 0) for slot in range(1, len(v.slots) + 1)},
                }
                if v.winner == "min" and rows:
                    entry[split]["binding_is_the_anchor"] = sum(bool(info[r["id"]].get("binding_is_the_anchor")) for r in rows)
            out[f"{t.id}/{v.name}"] = entry
    return out


def check_comparisons(summary: Mapping[str, Any]) -> None:
    """Raise if the winner's mention slots in a comparison stratum differ by more than one question."""
    broken = [
        (key, split, d["winner_slot_histogram"])
        for key, entry in summary.items()
        for split, d in entry.items()
        if d["questions"] and max(d["winner_slot_histogram"].values()) - min(d["winner_slot_histogram"].values()) > 1
    ]
    if broken:
        raise TemplateError(f"winner mention slots unbalanced (variant, split, histogram): {broken}")


# The previous round's dev items (round 2: round 1's) were the calibration pool, whose results were seen, so no test
# item may derive from one (design ruling in the design log). An item's anchors are the entities its variant's Anchor
# declarations take from the params (each Anchor's entity_param): the compounds, classes and reactions it names by
# label and the EC numbers it names by identifier, as the record's "anchors" list gives them.
PREVIOUS_DEV_ANCHOR = "round_1_dev_anchor"
TEST_ANCHOR_RULE = (
    "no test item may derive from a round-1 dev item (the calibration pool, whose results were seen): the anchor "
    "entities of every round-1 dev question, of either tier and any template, are skipped when a round-2 "
    "compositional candidate names one, and generation fails if any test item (either tier) shares an anchor with a "
    "round-1 dev item. The split is drawn after the selection (it is keyed on the ids, which follow the picks), so "
    "every compositional candidate may become a test item and the skip applies to all of them, dev items included. "
    "An item's anchors are the entities its variant's Anchor declarations take from the params (anchor_params "
    "below), i.e. the entities of the record's anchors list"
)
SAMPLER_RULE_CHANGE = {
    "rule": (
        "a variant short of its quota takes its own set-aside (non-preferred) candidates, best tier first, before a "
        "sibling variant covers the rest of its shortfall with preferred candidates; set-aside candidates of any "
        "variant still come last (sampling.select)"
    ),
    "before": "a sibling's preferred candidates covered the shortfall first, and the variant's own set-aside "
              "candidates were used only if the siblings could not",
    "made_after": (
        "a generation outcome, with no model outcome involved: the round-2 set of a0024f4 (questions 879d5a9e) gave "
        "absence_negation/unused_child_classes 0 of its 14 (71 candidates passed every check but lacked the "
        "variant's preferred properties), and its sibling reactions_without_partner took 29 against a quota of 15; "
        "no round-2 question had been run"
    ),
    "reason": (
        "a template's compositional tier kept to one variant, and the new readings of the raised variant ran on no "
        "question; a variant's own candidates that pass every check are closer to its quota than a sibling's"
    ),
    "ruling": "study lead, design log",
    "effect_on_round_1": (
        "none: every round-1 variant of both tiers filled its quota from preferred candidates, so the round-1 "
        "setting still reproduces 00e84760 byte for byte"
    ),
}


def anchor_entities(variant: Variant, params: Mapping[str, Any]) -> frozenset[str]:
    """The entities a question names (its variant's anchors, read from the params)."""
    return frozenset(str(a["entity"]) for a in variant.anchor_list(params))


def previous_dev_anchors(previous: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """The anchor entities of every dev record of a previous round, of either tier and any template."""
    return frozenset(str(a["entity"]) for r in previous if r["split"] == DEV for a in r.get("anchors", ()))


def check_test_anchors(records: Sequence[Mapping[str, Any]], dev_anchors: frozenset[str]) -> dict[str, int]:
    """Fail when a test record shares an anchor with a previous round's dev item."""
    tests = [r for r in records if r["split"] == TEST]
    shared = [(r["id"], sorted(_anchor_set(r) & dev_anchors)) for r in tests if _anchor_set(r) & dev_anchors]
    if shared:
        raise TemplateError(f"test items share an anchor with a previous-round dev item: {shared[:5]}")
    return {"test_items_checked": len(tests), "test_items_sharing": 0}


# --------------------------------------------------------------------------
# Generation


def generate(
    gold_client: SparqlClient,
    seed: int,
    per_template: int,
    templates: Sequence[type[QuestionTemplate]] = TEMPLATES,
    pool_client: SparqlClient | None = None,
    max_rows: int = MAX_ROWS,
    max_seconds: float = HARD_GOLD_SECONDS,
    min_preferred_share: float = 1.0,
    knobs: Knobs = DEFAULT_KNOBS,
    lookup_per_template: int | None = None,
    previous_round: Sequence[Mapping[str, Any]] | None = None,
) -> Generation:
    """Sample ``per_template`` questions for each template and compute their gold answers.

    ``lookup_per_template`` (default: one third, ``lookup_count``) of them come
    from the lookup tier, the rest from the compositional tier under ``knobs``.
    ``gold_client`` should time out at ``MAX_SECONDS``. Query times never decide
    the selection: after it, a selected gold over ``max_seconds`` (the hard cap)
    fails the run, as does any time-out. ``pool_client`` (default: the same
    client) runs the candidate queries, which can take longer.
    ``min_preferred_share``: share of each template's questions that must have
    their variant's preferred properties when the pool allows (1.0: all).
    ``previous_round``: the previous calibration round's records, required
    from round 2 on. No test item may share an anchor with one of its dev
    items (the calibration pool, whose results were seen): compositional
    candidates naming such an anchor are skipped, and generation fails if a
    test item of either tier still names one (check_test_anchors).
    """
    if knobs.calibration_round() >= 2 and previous_round is None:
        raise TemplateError(
            f"calibration round {knobs.calibration_round()} needs the previous round's records (previous_round): "
            "no test item may share an anchor with a previous-round dev item"
        )
    dev_anchors = previous_dev_anchors(previous_round) if previous_round is not None else frozenset()
    templates = resolve_templates(templates, knobs)
    pool_client = pool_client or gold_client
    triples = store_triples(pool_client)
    labels = LabelIndex.from_store(pool_client)
    n_lookup = lookup_count(per_template) if lookup_per_template is None else lookup_per_template
    counts = {LOOKUP: n_lookup, COMPOSITIONAL: per_template - n_lookup}
    stats: dict[str, dict[str, VariantStats]] = OrderedDict()
    picked: list[tuple[type[QuestionTemplate], Accepted]] = []
    # Answers are kept apart across all templates (no answer equals, contains or sits inside another), so the
    # dev/test nesting check, which spans templates, cannot fail on a sampled set. Entities repeat across templates.
    used_answers: list[frozenset] = []
    used_keys: dict[str, set[str]] = {t.id: set() for t in templates}
    for template in templates:
        stats[template.id] = OrderedDict()
    if knobs.calibration_round() == 1:  # the order the frozen round-1 file was drawn in
        order = [(t, tier) for t in templates for tier in TIERS]
    else:  # every lookup tier first, so no compositional answer can change a lookup pick
        order = [(t, tier) for tier in TIERS for t in templates]
    for template, tier in order:
        variants = template.tier_variants(tier, knobs)
        if not variants or counts[tier] <= 0:
            continue
        vstats, accepted = _sample_tier(
            template, variants, gold_client, pool_client, seed, counts[tier], max_rows, max_seconds,
            min_preferred_share, knobs, labels, used_keys[template.id], used_answers,
            excluded_anchors=dev_anchors if tier == COMPOSITIONAL else frozenset(),
        )
        stats[template.id].update(vstats)
        picked.extend((template, a) for a in accepted)

    slow = [(template.id, acc.candidate.variant, round(acc.payload.seconds, 2)) for template, acc in picked
            if acc.payload.seconds > max_seconds]
    if slow:
        raise TemplateError(f"selected gold queries over the hard cap of {max_seconds:g} s: {slow[:5]}")
    by_variant = {(t.id, v.name): v for t in templates for v in t.variants}
    # Ids and splits depend only on the picks, so they are known before the records (the comparison
    # variants balance their mention slots within each split).
    counters: Counter = Counter()
    qids_in_order = []
    for template, _ in picked:  # per-template counters: lookup picks come first within a template in any order
        counters[template.id] += 1
        qids_in_order.append(f"{ID_PREFIX}-{template.id}-{counters[template.id]:03d}")

    def split_picks() -> dict[str, str]:
        return assign_splits(
            [
                {
                    "template": f"{template.id}/{template.variant(acc.candidate.variant).tier}",
                    "base_question_id": qid,
                    "stratum": acc.candidate.variant,
                    **(
                        {"value": answer_value(acc.payload.rows), "two_valued": by_variant[(template.id, acc.candidate.variant)].two_valued}
                        if by_variant[(template.id, acc.candidate.variant)].single_row
                        else {}
                    ),
                }
                for (template, acc), qid in zip(picked, qids_in_order)
            ],
            seed,
        )

    # Comparison variants need the splits before their texts; otherwise the split comes after the guards, as in
    # round 1 (so a missing clause is reported before a split that cannot keep the value caps).
    splits: dict[str, str] | None = None
    comparisons: dict[str, dict[str, Any]] = {}
    if any(t.variant(a.candidate.variant).slots for t, a in picked):
        splits = split_picks()
        picked, comparisons = balance_slots(picked, qids_in_order, splits, seed, gold_client)
    version = generator_version(knobs)

    records = []
    measured: dict[str, float] = {}
    warm: dict[str, float] = {}
    step_checks: dict[str, dict[str, Any]] = {}
    texts: set[str] = set()
    reading_results: dict[str, dict[str, Any]] = {}
    unguarded: list[tuple[str, str, list[str]]] = []
    for (template, acc), qid in zip(picked, qids_in_order):
        variant = template.variant(acc.candidate.variant)
        gold: Gold = acc.payload
        # A reading with a note (approved_only, round 2) puts its sentence in the text only where it changes the
        # answer; the picks are made before, so a note never changes which questions are drawn.
        params = {**acc.candidate.params, **reading_notes(gold_client, variant, acc.candidate.params, gold.sparql, gold.rows)}
        text = variant.question(params)
        if text in texts:
            raise TemplateError(f"duplicate question text: {text}")
        texts.add(text)
        where = f"{template.id}/{variant.name}"
        check_no_database_names(where, text)
        measured[qid] = round(gold.seconds, 4)
        # Warm times are run metadata only (the records hold no timing), so they never decide anything.
        warm[qid] = round(min(gold_client.query(gold.sparql).elapsed_s for _ in range(WARM_RUNS)), 4)
        if gold.step is not None:
            step_checks[qid] = gold.step
        engine_sensitive = engine_reasons(gold_client, variant, params, gold.rows)
        same_family = same_family_columns(variant.answer_columns, [row_from_json(r) for r in gold.rows])
        check_stated_order(where, text, same_family)
        mode = template.mode_of(variant)
        disc = {
            name: gold.discriminates[name]
            if name in gold.discriminates
            else discrimination(gold_client, name, gold.sparql, mode, gold.rows)
            for name in DISCRIMINATIONS
        }
        for reading in variant.readings:
            alt = reading.alternative(variant, params, gold.sparql)
            if alt is not None and nests_graph(alt):
                raise TemplateError(f"{where}: reading {reading.name} nests a GRAPH clause inside another")
        results = check_readings(
            gold_client, variant, params, gold.sparql, gold.rows, neutral_text(text, params, variant.anchors),
        )
        unknown = [name for name, x in results.items() if x.changed is None]
        if unknown:  # a reading query that failed (or timed out) must not leave the guard undecided
            raise TemplateError(f"{qid}: alternative readings {unknown} could not be run")
        reading_results[qid] = results
        missing = [name for name, x in results.items() if x.changed is not False and not x.clause]
        if missing:
            unguarded.append((qid, where, missing))
        changed = [name for name, x in results.items() if x.changed]
        records.append(
            {
                "id": qid,
                "base_question_id": qid,
                "template": template.id,
                "tier": variant.tier,
                "graph": GRAPH_ID,
                "split": None,
                "question": text,
                "answer_columns": [c.to_json() for c in variant.answer_columns],
                "gold_sparql": gold.sparql,
                "gold_rows": gold.rows,
                "gold_row_count": len(gold.rows),
                "scoring_rules": dict(variant.scoring),
                "membership_mode": mode,
                "hops": variant.hops,
                "forms": list(variant.forms),
                "features": dict(variant.features),
                "anchors": variant.anchor_list(params),
                "params": {"variant": variant.name, **params},
                "engine_sensitive": engine_sensitive,
                "discriminates_direct_vs_transitive": disc["direct_vs_transitive"],
                "discriminates_graph_scope": disc["graph_scope"],
                "same_family_columns": same_family,
                "scope_readings_changed": sorted(changed),
                "generator_version": version,
                "seed": seed,
            }
        )
    if knobs.calibration_round() != 1:  # round 1 keeps its frozen column names (chebi_label, chebi_class)
        for r in records:
            check_agent_visible(r)
    if unguarded:
        by_variant = Counter((where, name) for _, where, names in unguarded for name in names)
        raise ReadingError(
            "questions with a reading that changes the answer and no clause stating the scope: "
            + "; ".join(f"{w} [{n}] x{c}" for (w, n), c in sorted(by_variant.items()))
            + f" (e.g. {unguarded[0][0]})"
        )
    reading_check = reading_summary(records, reading_results)
    anchor_check = check_label_anchors(pool_client, records)
    if splits is None:
        splits = split_picks()
    for r in records:
        r["split"] = splits[r["base_question_id"]]
    records.sort(key=lambda r: r["id"])
    test_anchor_check = None
    if previous_round is not None:
        test_anchor_check = {
            "previous_round_sha256": hashlib.sha256(questions_jsonl(previous_round).encode("utf-8")).hexdigest(),
            "previous_dev_anchor_entities": len(dev_anchors),
            **check_test_anchors(records, dev_anchors),
        }
    check_value_diversity(records, templates)
    check_comparisons(comparison_summary(records, templates, comparisons))
    overlaps = split_overlaps(records)
    if overlaps["nested"]:
        raise OverlapError(f"dev and test answers are nested: {overlaps['nested'][:5]}")
    return Generation(
        records, stats, measured, triples, overlaps, knobs, counts, anchor_check, reading_check,
        step_checks, comparisons, warm, test_anchor_check,
    )


def _sample_tier(
    template: type[QuestionTemplate],
    variants: Sequence[Variant],
    gold_client: SparqlClient,
    pool_client: SparqlClient,
    seed: int,
    total: int,
    max_rows: int,
    max_seconds: float,
    min_preferred_share: float,
    knobs: Knobs,
    labels: LabelIndex,
    used_keys: set[str],
    used_answers: list[frozenset],
    excluded_anchors: frozenset[str] = frozenset(),
) -> tuple[dict[str, VariantStats], list[Accepted]]:
    vstats: dict[str, VariantStats] = OrderedDict((v.name, VariantStats()) for v in variants)
    pools = []
    for variant in variants:
        candidates = variant.candidates(pool_rows(pool_client.query(variant.candidate_sparql())), knobs)
        kept = []
        for c in candidates:
            named = [(a, a.describe(c.params)) for a in variant.anchors if a.label_param]
            if not all(labels.ok(d["name"], d["entity"], a.kind) for a, d in named):
                vstats[variant.name].ambiguous_labels += 1
                continue
            if not all(labels.strictly_ok(d["name"], d["entity"]) for _, d in named):
                vstats[variant.name].kept_by_kind_rule += 1
            flags = {a.homonym_param: True for a, d in named if labels.homonym(d["name"], a.kind)}
            kept.append(replace(c, params={**c.params, **flags}) if flags else c)
        vstats[variant.name].candidates = len(kept)
        vstats[variant.name].entities = len({c.key for c in kept})
        vstats[variant.name].pool_strata = Counter({c.key: c.stratum for c in kept}.values())
        pools.append(shuffled_groups(kept, rng_for(seed, "sample", template.id, variant.name)))

    by_name = {v.name: v for v in variants}

    def reject(c: Candidate, why: str) -> None:
        vstats[c.variant].rejected[why] += 1

    def evaluate(c: Candidate):
        variant = by_name[c.variant]
        # Skipped when drawn, not dropped from the pool, so the seeded draw order of every other candidate stays.
        if excluded_anchors and anchor_entities(variant, c.params) & excluded_anchors:
            reject(c, PREVIOUS_DEV_ANCHOR)
            return None
        if c.excluded:  # e.g. a retained round-1 bound (compositional_r2.ROUND_1_BOUND)
            reject(c, c.excluded)
            return None
        s = vstats[c.variant]
        s.evaluated += 1
        sparql = variant.gold_sparql(c.params)
        result = gold_client.query(sparql)  # a time-out raises: it fails the run, never picks another question
        if variant.single_row:
            low, high = 1, 1
        elif variant.tier == COMPOSITIONAL:
            low, high = knobs.min_rows, min(knobs.max_rows, max_rows)
        else:
            low, high = MIN_ROWS, max_rows
        if result.total_rows < low:
            reject(c, "empty" if result.total_rows == 0 else "too_few_rows")
            return None
        if result.total_rows > high:
            reject(c, "too_many_rows")
            return None
        check_row_kinds(variant.answer_columns, result.rows)
        rows = canonical_rows(result.rows)
        if len({json.dumps(r, sort_keys=True) for r in rows}) != len(rows):
            raise TemplateError(f"{template.id}/{variant.name}: gold has duplicate rows")
        if nests_graph(sparql):
            raise TemplateError(f"{template.id}/{variant.name}: the gold nests a GRAPH clause inside another")
        population = None
        population_sparql = variant.population_sparql(c.params)
        if population_sparql is not None:
            population = int(gold_client.query(population_sparql).rows[0][0].lexical)
            if population < knobs.min_population:
                reject(c, "small_population")
                return None
            if not variant.single_row and len({json.dumps(r[0], sort_keys=True) for r in rows}) >= population:
                reject(c, "filter_removes_nothing")
                return None
        step = None
        if variant.step_check is not None:
            step = step_outcome(gold_client, variant, c.params, rows)
            if not step["matters"]:
                reject(c, "step_not_load_bearing")
                return None
        gold = Gold(sparql, rows, result.elapsed_s, population=population, step=step)
        for name in variant.prefer:
            gold.discriminates[name] = discrimination(gold_client, name, sparql, template.mode_of(variant), rows)
        tier = sum(1 for name in variant.prefer if gold.discriminates[name])
        return gold, answer_key(rows, variant.anchor_list(c.params)), tier

    # Value diversity: a single-value variant takes at most value_cap(quota) questions with one answer value,
    # so the split can keep every (variant, split) stratum within its cap.
    variant_quota = quotas(total, [v.weight for v in variants])
    for v, q in zip(variants, variant_quota):
        vstats[v.name].quota = q
    accepted = select(
        pools,
        total,
        evaluate,
        on_skip=reject,
        weights=[v.weight for v in variants],
        top_tiers=[len(v.prefer) for v in variants],
        nonpreferred_budget=math.floor((1 - min_preferred_share) * total + 1e-9),
        used_keys=used_keys,
        used_answers=used_answers,
        value_of=lambda c, gold: answer_value(gold.rows) if by_name[c.variant].single_row else None,
        value_caps=[value_cap(q, v.two_valued) if v.single_row else None for v, q in zip(variants, variant_quota)],
    )
    for a in accepted:
        vstats[a.candidate.variant].accepted += 1
        vstats[a.candidate.variant].sampled_strata[a.candidate.stratum] += 1
        if a.tier < len(by_name[a.candidate.variant].prefer):
            vstats[a.candidate.variant].non_preferred += 1
    return vstats, accepted


# --------------------------------------------------------------------------
# Output


def questions_jsonl(records: Iterable[dict[str, Any]]) -> str:
    lines = [json.dumps(r, ensure_ascii=False) for r in sorted(records, key=lambda r: r["id"])]
    return "\n".join(lines) + "\n"


def _shared_value_counts(pairs: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(v for p in pairs for v in p.get("shared_values", ()))
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def build_manifest(
    gen: Generation,
    *,
    seed: int,
    per_template: int,
    endpoint: str,
    questions_text: str,
    templates: Sequence[type[QuestionTemplate]] = TEMPLATES,
    generated_at: str | None = None,
    calibration_round: int | None = None,
    previous: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The manifest; ``previous``: the previous calibration round's records, compared with these ones."""
    templates = resolve_templates(templates, gen.knobs)
    round_ = gen.knobs.calibration_round() if calibration_round is None else calibration_round
    if calibration_round is not None and calibration_round != gen.knobs.calibration_round():
        raise TemplateError(f"calibration round {calibration_round} does not match the knobs (round {gen.knobs.calibration_round()})")
    by_template: dict[str, Any] = OrderedDict()
    for t in templates:
        rows = [r for r in gen.records if r["template"] == t.id]
        by_template[t.id] = {
            "total": len(rows),
            DEV: sum(r["split"] == DEV for r in rows),
            TEST: sum(r["split"] == TEST for r in rows),
            "by_tier": {
                tier: {
                    "total": sum(r["tier"] == tier for r in rows),
                    DEV: sum(r["tier"] == tier and r["split"] == DEV for r in rows),
                    TEST: sum(r["tier"] == tier and r["split"] == TEST for r in rows),
                }
                for tier in TIERS
            },
            "by_variant": dict(Counter(r["params"]["variant"] for r in rows)),
        }
    times = list(gen.measured_seconds.values())
    intersecting = gen.overlaps["intersecting"]
    shared_counts = _shared_value_counts(intersecting)
    return {
        "graph": GRAPH_ID,
        "generator_version": generator_version(gen.knobs),
        "calibration_round": round_,
        "agent_settings": {
            **AGENT_SETTINGS.get(round_, {}),
            "models": "both (Sonnet and Qwen)",
            "note": (
                "the agent caps this round's runs use (study lead's confirmation of round 2 in the design log): round 2 "
                "runs with the frozen caps of change-log entry 23 (60 turns, 1,800 s, 16k output tokens per turn) for "
                "both models; round 1's caps (30 turns, 900 s) belong to round 1 only. The templates freeze after "
                "round 2, whatever it gives"
            ),
        },
        "seed": seed,
        "per_template": per_template,
        "tiers": {
            "per_template": dict(gen.tier_counts),
            "totals": {tier: sum(r["tier"] == tier for r in gen.records) for tier in TIERS},
            "lookup_share": round(sum(r["tier"] == LOOKUP for r in gen.records) / len(gen.records), 4) if gen.records else None,
            "rule": (
                "lookup: the audited generator-0.2 templates, sampled first and independent of the knobs; "
                "compositional: chained or combined steps, sampled under the knobs, sharing the lookup tier's "
                "used entities and answers"
            ),
        },
        "knobs": {
            "settings": gen.knobs.to_json(),
            "calibration_round": round_,
            "active_variants": {
                t.id: [v.name for v in t.tier_variants(COMPOSITIONAL, gen.knobs)] for t in templates
            },
            "inactive_variants": {
                t.id: [v.name for v in t.tier_variants(COMPOSITIONAL) if not gen.knobs.allows(v.features)]
                for t in templates
            },
            "features_by_variant": {
                f"{t.id}/{v.name}": {"forms": list(v.forms), **dict(v.features), "hops": v.hops}
                for t in templates
                for v in t.tier_variants(COMPOSITIONAL)
            },
        },
        "hops_convention": (
            "hops count edges from the named entity to the answer node, not the label used to name it. A compound "
            "reaches a reaction in 3 edges: compound <- participant (rh:compound), participant <- side "
            "(rh:contains), side <- reaction (rh:side), because the reaction graph models each participation as a "
            "participant node on a side node. So reaction -> compounds is 3 hops, compound -> EC number 4, and "
            "compound -> compounds on the other side of its reactions 6; a ChEBI hierarchy step or rh:chebi link is "
            "one edge each"
        ),
        "endpoint": endpoint,
        "store": {**STORE_IDENTITY, "triples_per_graph": dict(sorted(gen.store_triples.items()))},
        "generated_at": generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "questions_file": "questions.jsonl",
        "questions_sha256": hashlib.sha256(questions_text.encode("utf-8")).hexdigest(),
        "gold_rows_sha256": hashlib.sha256(
            "\n".join(json.dumps([r["id"], r["gold_rows"]], ensure_ascii=False) for r in gen.records).encode("utf-8")
        ).hexdigest(),
        "filters": {
            "min_rows": MIN_ROWS,
            "max_rows": MAX_ROWS,
            "compositional_rows": [gen.knobs.min_rows, gen.knobs.max_rows],
            "hard_cap_seconds": HARD_GOLD_SECONDS,
            "timing_rule": (
                "query times never decide the selection; a selected gold over the hard cap, or any time-out, "
                "fails the run; measured times are run metadata below"
            ),
        },
        "split": {
            "dev_fraction": 0.3,
            "min_dev_per_group": 3,
            "grouped_by": "base_question_id",
            "split_within": "each (template, tier)",
            "stratified_by": (
                "variant (largest remainder within each template and tier); in single-value variants equal answer "
                "values are also spread over dev and test so neither side exceeds the value-diversity cap"
            ),
            "dev_per_template_and_tier": {
                tier: dev_count(n) for tier, n in gen.tier_counts.items()
            },
        },
        "counts": {
            "total": len(gen.records),
            DEV: sum(r["split"] == DEV for r in gen.records),
            TEST: sum(r["split"] == TEST for r in gen.records),
            "by_tier": {
                tier: {
                    "total": sum(r["tier"] == tier for r in gen.records),
                    DEV: sum(r["tier"] == tier and r["split"] == DEV for r in gen.records),
                    TEST: sum(r["tier"] == tier and r["split"] == TEST for r in gen.records),
                }
                for tier in TIERS
            },
            "by_template": by_template,
        },
        "forms": {
            "note": "compositional forms per question (a question can have several)",
            "by_form": {f: sum(f in r["forms"] for r in gen.records) for f in FORMS},
            "by_template": {
                t.id: {f: sum(f in r["forms"] for r in gen.records if r["template"] == t.id) for f in FORMS}
                for t in templates
            },
        },
        "label_anchors": gen.anchor_check,
        "alternative_readings": gen.reading_check,
        "database_names": {
            "rule": (
                "question text names no database: the words Rhea, ChEBI, PubMed and UniProt (any case, whole "
                "words, so also accessions such as RHEA:... and CHEBI:...) fail generation; the opaque mapper's "
                "phrases all contain one of them"
            ),
            "phrases": list(DATABASE_PHRASES),
            **(
                {
                    "agent_visible_fields": {
                        "fields": list(AGENT_VISIBLE_FIELDS),
                        "rule": (
                            "from calibration round 2 on, no agent-visible field of a record names a database: the text, "
                            "the answer-column names (split into words at _, - and case changes; a lower-case 'go' "
                            "counts) and the scoring rules; the runner renders only the text (harness/agent/prompts."
                            "user_message), the others are guarded so no prompt or checker message can leak a name"
                        ),
                        "renamed_columns": {k: dict(v) for k, v in COLUMN_RENAMES.items()},
                    }
                }
                if round_ != 1
                else {}
            ),
        },
        "gold_query_seconds_measured": {
            "note": (
                "run metadata, not part of the determinism check: questions.jsonl holds no timing, so it is "
                "byte-identical across runs; a selected gold over filters.hard_cap_seconds fails the run"
            ),
            "max": max(times) if times else None,
            "by_question": dict(sorted(gen.measured_seconds.items())),
            "warm_rule": (
                f"each selected gold is re-run {WARM_RUNS} times back to back after the first (cold) run; warm is the "
                "faster of those; neither decides anything"
            ),
            "warm_max": max(gen.warm_seconds.values()) if gen.warm_seconds else None,
            "warm_by_question": dict(sorted(gen.warm_seconds.items())),
        },
        "structural_steps": structural_steps_summary(gen, templates),
        "sampler_rule_change": SAMPLER_RULE_CHANGE,
        "test_anchor_exclusion": test_anchor_summary(gen, templates),
        **({"changes_from_previous_round": round_changes(gen.records, previous)} if previous is not None else {}),
        "row_counts": row_count_distribution(gen.records, templates),
        "row_counts_by_tier": row_count_by_tier(gen.records),
        "single_value_histograms": {
            "rule": VALUE_RULE,
            "formula": "cap(n) = max(1, ceil(0.25 * n)) per (variant, split); two-valued: max(1, floor(0.6 * n))",
            "by_variant": value_strata(gen.records, templates),
        },
        "shape_strata": {
            "note": "sampling alternates between answer shapes, so multi-valued answers are over-sampled; shares compare the candidate pool (entities) with the questions sampled",
            "by_variant": {
                t: {v: s.strata_json() for v, s in vs.items() if s.strata_json() is not None}
                for t, vs in gen.stats.items()
                if any(s.strata_json() is not None for s in vs.values())
            },
        },
        "engine_sensitive": engine_sensitive_summary(gen.records, templates),
        "discrimination": discrimination_summary(gen.records, templates),
        "entity_reuse": entity_reuse(gen.records),
        "same_family_columns": {
            "rule": (
                "the scorer permutes answer columns only across value families, so a question with two columns of "
                "one family states their order (\"the A, then the B\" or \"each A with its B\"); generation fails "
                "otherwise"
            ),
            "by_variant": dict(
                sorted(
                    Counter(f"{r['template']}/{r['params']['variant']}" for r in gen.records if r["same_family_columns"]).items()
                )
            ),
        },
        "split_overlap": {
            "cross_template_intersecting_pairs": gen.overlaps.get("cross_template_intersecting", 0),
            "rule": (
                "dev and test gold-row sets are compared as they are, across all templates and both tiers: no dev "
                "answer may equal, contain or be contained in a test answer. One-row single values are left out of "
                "this (design ruling: guessability is handled by the value cap); equal ones within a template are "
                "listed under equal_single_value. Answers that share some rows are listed within a template with "
                "the shared values and counted across templates. Sampling skips nested answers across all "
                "templates, comparing one-row values by (entities, value)"
            ),
            "nested_pairs": len(gen.overlaps["nested"]),
            "equal_single_value_pairs": len(gen.overlaps["equal_single_value"]),
            "equal_single_value_by_template": dict(Counter(p["template"] for p in gen.overlaps["equal_single_value"])),
            "equal_single_value": gen.overlaps["equal_single_value"],
            "intersecting_pairs": len(intersecting),
            "intersecting_by_template": dict(Counter(p["template"] for p in intersecting)),
            "max_shared_rows": max((p["shared_rows"] for p in intersecting), default=0),
            "shared_values": shared_counts,
            "intersecting": intersecting,
        },
        "sampling": {
            t: {v: s.to_json() for v, s in vs.items()} for t, vs in gen.stats.items()
        },
        "templates": [t.to_json() for t in templates],
    }


def structural_steps_summary(gen: Generation, templates: Sequence[type[QuestionTemplate]]) -> dict[str, Any]:
    """The step table (round-1 variant, axis, round-1 and round-2 values), rules, per-question checks, comparisons."""
    active = {(t.id, v.name) for t in templates for v in t.tier_variants(COMPOSITIONAL, gen.knobs)}
    by_variant: dict[str, Any] = OrderedDict()
    for key, step in STEPS.items():
        tid = key.split("/")[0]
        by_variant[key] = {**step.to_json(), "applied": bool(gen.knobs.delta(step.axis)) and (tid, step.raised.name) in active}
    raised = {(k.split("/")[0], d["raised_variant"]) for k, d in by_variant.items() if d["applied"]}
    comp = [r for r in gen.records if (r["template"], r["params"]["variant"]) in raised]
    checks_by_variant: dict[str, Any] = OrderedDict()
    for key, step in STEPS.items():
        tid = key.split("/")[0]
        if (tid, step.raised.name) not in raised:
            continue
        outcomes = [gen.step_checks[r["id"]] for r in comp if (r["template"], r["params"]["variant"]) == (tid, step.raised.name)]
        rejected = gen.stats.get(tid, {}).get(step.raised.name)
        entry: dict[str, Any] = {
            "rule": step.raised.step_check.rule,
            "questions": len(outcomes),
            "matters": sum(o["matters"] for o in outcomes),
            "rejected_step_not_load_bearing": rejected.rejected.get("step_not_load_bearing", 0) if rejected else 0,
        }
        if any("fan_out" in o for o in outcomes):
            fans = sorted(o["fan_out"] for o in outcomes)
            entry["near_end_entities"] = {"min": fans[0], "median": fans[len(fans) // 2], "max": fans[-1]}
        if any("undo_changed" in o for o in outcomes):
            entry["undo_changed_all"] = sum(all(o["undo_changed"]) for o in outcomes)
            entry["undo_changed_some"] = sum(any(o["undo_changed"]) for o in outcomes)
        if any("distinct" in o for o in outcomes):
            entry["distinct_values"] = sum(o["distinct"] for o in outcomes)
        checks_by_variant[key] = entry
    shortfalls = [
        {"variant": f"{t}/{v}", "quota": s.quota, "accepted": s.accepted}
        for t, vs in gen.stats.items()
        for v, s in vs.items()
        if s.quota is not None and s.accepted < s.quota
    ]
    dev, test = sum(r["split"] == DEV for r in comp), sum(r["split"] == TEST for r in comp)
    return {
        "knobs": {f"{a}_delta": gen.knobs.delta(a) for a in STRUCTURAL_AXES},
        "having_max": gen.knobs.having_max,
        "rules": STEP_RULES,
        "changed_questions": {
            "note": (
                f"every compositional question is in its raised form, so all {dev + test} changed against round 1: "
                f"the {dev} dev questions (the calibration pool) and the {test} test questions alike, because the test "
                "questions use the same template settings as the calibrated dev questions; no test question has been "
                "run. The lookup tier keeps round 1's picks; its records differ from round 1 only in generator_version, "
                "the renamed answer columns and the status note where approved_only changes a lookup answer "
                "(changes_from_previous_round.lookup.differing_fields)"
            ),
            "compositional": {DEV: dev, TEST: test},
        } if raised else {"note": "no structural step applied (round 1)"},
        "by_variant": by_variant,
        "inactive_round_1_variants": {
            t.id: [v.name for v in t.tier_variants(COMPOSITIONAL) if v.step_check is None and not gen.knobs.allows(v.features)]
            for t in templates
        },
        "step_checks": {
            "summary_by_variant": checks_by_variant,
            "by_question": dict(sorted(gen.step_checks.items())),
        },
        "shortfalls": shortfalls,
        "comparisons": {
            "rule": (
                "in each comparison variant that asks for a winner the compared entities' mention order is a seeded "
                "permutation that puts the winner (partners_with_lower_charge: the binding reference, the one with the "
                "lower charge) in each mention slot as evenly as each (variant, split) allows; the compared values are "
                "pairwise distinct (no ties). The candidate draw lists entities in IRI "
                "order, which follows accession order, so mention order would otherwise be a shortcut (the UniProt "
                "audit found mention order = accession order in 58 of 58 cases). The gold reads the entities as a set "
                "or by entity, never by slot. citation_count_difference asks for no winner (code review of a0024f4): "
                "like round 1 it names its subject, the most-cited reaction, first, and reads the other two as one "
                "set, so it has no mention slot to balance; its counts are still pairwise distinct"
            ),
            "by_variant": comparison_summary(gen.records, templates, gen.comparisons),
            "by_question": dict(sorted(gen.comparisons.items())),
            # Design ruling in the design log: no slot balancing applies here.
            "no_slot_balancing": {
                f"{t.id}/{v.name}": (
                    "the subject (the most-cited reaction) is named first by the wording, as in round 1, and the other "
                    "two are read as one set, so no mention slot carries the answer"
                )
                for t in templates
                for v in t.tier_variants(COMPOSITIONAL, gen.knobs)
                if "comparison" in v.forms and not v.slots and (t.id, v.name) in raised
            },
        },
        "round_1_bounds": {"rule": BOUNDS_RULE, "by_bound": list(ROUND_1_BOUNDS)} if raised else {},
    }


def test_anchor_summary(gen: Generation, templates: Sequence[type[QuestionTemplate]]) -> dict[str, Any]:
    """The previous-round anchor rule: its source, the anchor parameters per variant, skipped candidates, the check."""
    def params_of(tier: str) -> dict[str, list[str]]:
        return {f"{t.id}/{v.name}": [a.entity_param for a in v.anchors] for t in templates for v in t.tier_variants(tier, gen.knobs)}

    if gen.test_anchor_check is None:
        return {"rule": TEST_ANCHOR_RULE, "applied": False, "note": "no previous round (round 1)"}
    return {
        "rule": TEST_ANCHOR_RULE,
        "applied": True,
        **gen.test_anchor_check,
        "skipped_candidates": {
            f"{t}/{v}": s.rejected[PREVIOUS_DEV_ANCHOR]
            for t, vs in gen.stats.items()
            for v, s in vs.items()
            if s.rejected.get(PREVIOUS_DEV_ANCHOR)
        },
        "anchor_params": params_of(COMPOSITIONAL),
        "lookup_anchor_params": params_of(LOOKUP),
    }


def _anchor_set(record: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(str(a["entity"]) for a in record.get("anchors", ()))


def round_changes(records: Sequence[dict[str, Any]], previous: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """This round against the previous one: changed texts and golds, the lookup tier, stepped against redrawn items.

    A compositional item is "stepped" when some item of its round-1 variant (same template) names a subset of
    its anchors (the raised question extends that item: same anchor, or one more population); otherwise it is
    "redrawn" (new anchors). Ids are not compared, since an id says nothing about the item behind it.
    """
    prev_by_id = {r["id"]: r for r in previous}

    def without_version(r: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in r.items() if k != "generator_version"}

    lookup = [r for r in records if r["tier"] == LOOKUP]
    lookup_differs = sorted(r["id"] for r in lookup if r["id"] not in prev_by_id or without_version(prev_by_id[r["id"]]) != without_version(r))
    comp = [r for r in records if r["tier"] == COMPOSITIONAL]
    text_changed = {s: sorted(r["id"] for r in comp if r["split"] == s and (r["id"] not in prev_by_id or prev_by_id[r["id"]]["question"] != r["question"])) for s in (DEV, TEST)}
    gold_changed = sum(1 for r in comp if r["id"] not in prev_by_id or prev_by_id[r["id"]]["gold_rows"] != r["gold_rows"])
    by_variant: dict[str, Any] = OrderedDict()
    redrawn: dict[str, list[str]] = {DEV: [], TEST: []}
    stepped: dict[str, list[dict[str, str]]] = {DEV: [], TEST: []}
    for r in comp:
        found = step_of(r["template"], r["params"]["variant"])
        round_1_variant = found[0].split("/")[1] if found else r["params"]["variant"]
        key = f"{r['template']}/{round_1_variant}"
        sources = [
            p for p in previous
            if p["tier"] == COMPOSITIONAL and p["template"] == r["template"] and p["params"]["variant"] == round_1_variant
            and _anchor_set(p) <= _anchor_set(r)
        ]
        entry = by_variant.setdefault(key, {"raised_variant": r["params"]["variant"], "stepped": 0, "redrawn": 0})
        if sources:
            entry["stepped"] += 1
            source = min(sources, key=lambda p: p["id"])
            stepped[r["split"]].append({"id": r["id"], "from": source["id"], "from_split": source["split"]})
        else:
            entry["redrawn"] += 1
            redrawn[r["split"]].append(r["id"])
    return {
        "rule": (
            "a compositional item is stepped when an item of its round-1 variant (same template) names a subset of its "
            "anchors, otherwise redrawn (new anchors); round 2 draws its items afresh under the raised forms, so most "
            "are redrawn and no paired round-1 against round-2 contrast is claimed. ec_compounds cannot be stepped: its "
            "hop moves the anchor from an EC number to a reaction"
        ),
        "previous_questions_sha256": hashlib.sha256(questions_jsonl(previous).encode("utf-8")).hexdigest(),
        "lookup": {
            "records": len(lookup),
            "identical_except_generator_version": len(lookup) - len(lookup_differs),
            "differing": lookup_differs,
            # Round 2's allowed differences (study lead, design log): renamed answer columns, and the status note
            # (question, params.any_status, scope_readings_changed) where approved_only changes a lookup answer.
            "differing_fields": {
                r["id"]: sorted(k for k in set(r) | set(prev_by_id[r["id"]]) if k != "generator_version"
                                and r.get(k) != prev_by_id[r["id"]].get(k))
                for r in lookup if r["id"] in lookup_differs and r["id"] in prev_by_id
            },
        },
        "compositional": {
            "records": len(comp),
            "text_changed": {s: len(v) for s, v in text_changed.items()},
            "gold_rows_changed": gold_changed,
            "stepped": sum(e["stepped"] for e in by_variant.values()),
            "redrawn": sum(e["redrawn"] for e in by_variant.values()),
            "by_variant": by_variant,
        },
        "text_changed_ids": text_changed,
        "redrawn_ids": redrawn,
        "stepped_ids": stepped,
        # Round-1 dev items are the calibration pool, whose results were seen: a test item stepped from one is flagged.
        "test_stepped_from_round_1_dev": [x["id"] for x in stepped[TEST] if x["from_split"] == DEV],
        "id_note": (
            "ids are reused for other items across rounds (an id names a position in the file, not an item): anything "
            "keyed by id must also key on questions_sha256 or generator_version"
        ),
        "ids_shared_with_previous_round": {
            s: sum(1 for r in comp if r["split"] == s and r["id"] in prev_by_id) for s in (DEV, TEST)
        },
    }


def write_outputs(out_dir: Path, gen: Generation, **manifest_args: Any) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = questions_jsonl(gen.records)
    qpath = out_dir / "questions.jsonl"
    qpath.write_text(text, encoding="utf-8")
    manifest = build_manifest(gen, questions_text=text, **manifest_args)
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return qpath, mpath
