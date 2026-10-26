"""A bank-only extension of a frozen question set (science approval in the design log; rulings in the design log).

The confirmatory fault-bank contrasts need more base questions than the frozen agent-facing set has. The extension
draws more gold queries from the frozen set's templates and knobs (UniProt: calibration round 2, the last round),
with a new seed, for the fault bank only. Its records carry no question text (``"question": null``), so no text
guard, reading guard or text audit applies; they never reach an agent.

What the extension shares with the frozen set, so the bank (frozen plus extension) stays one population:

- Set-wide answer uniqueness, seeded with every frozen answer: no extension answer equals, contains or sits inside
  any other answer in the bank, in any template.
- One entity per template across the whole bank (design ruling in the design log): each template's used entities are
  seeded with every frozen question's key and declared anchors, and an extension candidate naming any used entity
  is skipped; a check after generation fails if an extension question shares an entity with any other bank question
  of its template. Frozen questions that already share an anchor among themselves are listed, not changed.
- The generator's sampling, quota rule, split rule and query-level checks (``_sample_template``), unchanged; a
  tier that runs short is made up by the other tier of its template (change-log entry 52).

A sidecar cluster map (clusters.json) assigns every bank question, frozen and extension, to the connected component
of shared declared anchors within its template, for the clustered analysis.

Determinism: the same frozen file, seed and store give a byte-identical extension file. Query times never select
(any time-out fails the run) and records carry no time field; the measured times are in the manifest only.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.qgen.base import (
    Profile,
    QuestionTemplate,
    TemplateError,
    resolve_templates,
)
from harness.qgen.generate import (
    GENERATOR_VERSION,
    MAX_ROWS,
    MAX_SECONDS,
    MIN_ROWS,
    VariantStats,
    _prior_round,
    _sample_template,
    anchor_count,
    anchor_entities,
    check_prior_anchors,
    check_value_diversity,
    engine_reasons,
    gold_rows_sha256,
    prior_summary,
    questions_jsonl,
    row_count_distribution,
    row_set,
    store_triples,
)
from harness.qgen.readings import nests_graph
from harness.qgen.sampling import Accepted, Selector, rng_for
from harness.qgen.split import DEV, DEV_FRACTION, MIN_DEV, TEST, allocate, dev_count
from harness.sparql import SparqlClient

EXTENSION_TAG = "ext1"
EXTENSION_VERSION = "0.3.0-ext1"
EXTENSION_QUESTIONS_FILE = "questions-ext.jsonl"
CLUSTERS_FILE = "clusters.json"
CLUSTER_MAP_SCHEMA = "cluster-map/2"  # the format feat/analysis reads
PROPOSED_SEED = 20271019
# N_test = 1,260 bank test questions, 180 per template: the frozen set's 30 plus 150 new. 214 new per template give
# 64 dev and 150 test (SPLIT_RULE); with the planned tiers, lookup 71 (21 dev, 50 test) and compositional 143 (43, 100).
PER_TEMPLATE = 214
# The frozen agent-facing sets the extension may seed from: round 2 of the UniProt set (0b247dd).
FROZEN_QUESTIONS_SHA256 = {"uniprot-sp-2026_03": "17eef284d7c89c64e5da42245b0b09f21215d2ebd01e086670057baae36c9184"}

ID_RULE = (
    f"extension ids are '<id_prefix>{EXTENSION_TAG}-<template>-<NNN>', numbered from 001 within each template in pick order. "
    "Frozen ids are '<id_prefix>-<template>-<NNN>': the character after the id prefix is '-' there and 'e' here, so "
    "no extension id can equal a frozen id, whatever either set's size; the build also checks it"
)
ENTITY_RULE = (
    "no two bank questions of one template (frozen or extension) name the same entity: a question's entities are "
    "its key (the variant's key parameter) and its declared anchors (the entities its variant names, as IRIs); "
    "every frozen question's entities seed its template's used set before sampling, an extension candidate naming "
    "a used entity is skipped ('anchor_used' or 'entity_used'), and a check after generation fails if an extension "
    "question shares an entity with any other bank question of its template. Frozen pairs that already share an "
    "anchor are listed under checks.shared_entities.frozen_only: the frozen file is not changed (design ruling on "
    "design log: base questions are independent clusters)"
)
GUARDS = {
    "applied": {
        "note": "the study lead's query-level list (design log)",
        "execution_without_time_out": (
            f"strict: a time-out of any gold, anchor, step or preference query fails the run, and so does a selected "
            f"gold at or over {MAX_SECONDS:g} s on its first (cold) run; query time never selects"
        ),
        "non_empty_rows": f"a gold with fewer than {MIN_ROWS} row is rejected ('empty')",
        "unique_anchors": "every anchor label matches exactly one entity, checked when sampling and again per record",
        "set_wide_answer_uniqueness": (
            "no answer equals, contains or sits inside another anywhere in the bank (seeded with every frozen answer; "
            "checked again over the whole bank after generation)"
        ),
        "value_cap": (
            "in each (variant, split) stratum of n one-row questions at most max(1, ceil(n / 4)) share a value: "
            "checked on the extension alone and on the extension merged with the frozen set"
        ),
        "no_nested_answers": "within a template and across templates, as part of answer uniqueness",
        "no_nested_graph": "a gold that nests a GRAPH clause inside another fails the run",
        "round_2_step_checks": "every raised variant's step must matter on the store ('step_not_load_bearing')",
    },
    "kept_generator_rules": {
        "max_gold_rows": f"a gold with more than {MAX_ROWS} rows is rejected ('too_many_rows'); fixed (design ruling in the design log)",
        "one_entity_per_template": "extended to every declared anchor across the whole bank; fixed (design ruling in the design log)",
        "quota_rule": (
            "1/3 lookup, 2/3 compositional per template, equal variant quotas within a tier; a variant that runs dry "
            "is made up by the other variants of its tier (the skew is accepted, ruling in the design log; see counts and "
            "bank_mix), and a tier that still runs short is made up by the other tier of its template (change-log "
            "entry 52; see tier_mix)"
        ),
        "preferred_properties": "candidates with their variant's preferred discriminations are drawn first, as in the generator",
        "gold_shape": "every cell bound and of its column's kind, no duplicate rows (template errors fail the run)",
        "prior_dev_anchors": "see settings.prior_dev_anchors (kept: design ruling in the design log, change-log entry 52)",
    },
    "not_applied": {
        "note": "text-level rules: the records carry no question text and never reach an agent",
        "database_names": "the question-text guard against database names",
        "duplicate_text": "the duplicate question-text check",
        "alternative_readings": "the reading guard and its scope clauses",
        "column_order_marker": "the same-family column order statement in the text",
        "winner_slot_balance": (
            "comparison variants keep the draw's mention order: the winner's mention slot matters only in a question "
            "text, so the extension does not permute it (gold rows never depend on it)"
        ),
        "warm_rule": (
            "the warm re-runs exist so the record's whole-second time field does not depend on the cache; the "
            "extension's records have no time field"
        ),
    },
}


def extension_id_prefix(profile: Profile) -> str:
    return f"{profile.id_prefix}{EXTENSION_TAG}"


def question_entities(variant: Any, params: Mapping[str, Any]) -> frozenset[str]:
    """The entities a question names: its key and its declared anchors (IRIs)."""
    return frozenset({str(params[variant.key_param])}) | anchor_entities(variant, params)


def frozen_seeds(
    frozen_records: Sequence[Mapping[str, Any]], templates: Sequence[type[QuestionTemplate]]
) -> tuple[list[frozenset[str]], dict[str, set[str]]]:
    """Every frozen answer (set-wide uniqueness) and each template's used entities (keys and anchors)."""
    by_id = {t.id: t for t in templates}
    answers = [row_set(r["gold_rows"]) for r in frozen_records]
    entities: dict[str, set[str]] = OrderedDict()
    for r in frozen_records:
        variant = by_id[r["template"]].variant(r["params"]["variant"])
        entities.setdefault(r["template"], set()).update(question_entities(variant, r["params"]))
    return answers, entities


def shared_entities(
    records: Sequence[Mapping[str, Any]], templates: Sequence[type[QuestionTemplate]]
) -> list[dict[str, Any]]:
    """Entities that more than one question of a template names: {template, entity, ids}."""
    by_id = {t.id: t for t in templates}
    owners: dict[tuple[str, str], list[str]] = {}
    for r in sorted(records, key=lambda r: r["id"]):
        variant = by_id[r["template"]].variant(r["params"]["variant"])
        for entity in sorted(question_entities(variant, r["params"])):
            owners.setdefault((r["template"], entity), []).append(r["id"])
    return [{"template": t, "entity": e, "ids": ids} for (t, e), ids in sorted(owners.items()) if len(ids) > 1]


def nested_answer_pairs(
    frozen_records: Sequence[Mapping[str, Any]], ext_records: Sequence[Mapping[str, Any]]
) -> list[tuple[str, str]]:
    """Pairs with at least one extension question whose answers are equal or nested, in any templates."""
    frozen = [(r["id"], row_set(r["gold_rows"])) for r in frozen_records]
    ext = [(r["id"], row_set(r["gold_rows"])) for r in sorted(ext_records, key=lambda r: r["id"])]
    out = []
    for i, (a, sa) in enumerate(ext):
        for b, sb in [*frozen, *ext[i + 1 :]]:
            if sa <= sb or sb <= sa:
                out.append((a, b))
    return out


CLUSTER_RULE = (
    "clusters are the connected components of shared declared anchors within a template: two questions of one "
    "template that name the same anchor entity (an IRI among their variants' declared anchors; keys that the text "
    "does not name do not count) are in one cluster, transitively; a question that shares no anchor is its own "
    "cluster. The cluster id is the smallest question id of the component. The bank (frozen and extension) and the "
    "frozen set alone give the same clusters for every frozen question: no extension question may share an anchor "
    "(checked when generating)"
)


def anchor_clusters(
    records: Sequence[Mapping[str, Any]], templates: Sequence[type[QuestionTemplate]]
) -> dict[str, str]:
    """Question id -> cluster id: connected components of shared declared anchors within a template."""
    by_id = {t.id: t for t in templates}
    parent = {r["id"]: r["id"] for r in records}

    def find(q: str) -> str:
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    first: dict[tuple[str, str], str] = {}
    for r in sorted(records, key=lambda r: r["id"]):
        variant = by_id[r["template"]].variant(r["params"]["variant"])
        for entity in sorted(anchor_entities(variant, r["params"])):
            other = first.setdefault((r["template"], entity), r["id"])
            a, b = find(other), find(r["id"])
            if a != b:
                parent[max(a, b)] = min(a, b)  # the root stays the smallest id of the component
    return {q: find(q) for q in sorted(parent)}


def bank_clusters(
    frozen_records: Sequence[Mapping[str, Any]],
    ext_records: Sequence[Mapping[str, Any]],
    templates: Sequence[type[QuestionTemplate]],
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    """The frozen set's clusters, the bank's, and the test split's figures; fails if the extension moves a frozen
    question to another cluster (it cannot while no extension question shares an anchor: checked, not assumed)."""
    frozen_clusters = anchor_clusters(frozen_records, templates)
    clusters = anchor_clusters([*frozen_records, *ext_records], templates)
    moved = sorted(q for q, c in frozen_clusters.items() if clusters[q] != c)
    if moved:
        raise TemplateError(f"the extension joins frozen clusters: {moved[:5]}")
    test_clusters = {
        "frozen": analysed_split_clusters(frozen_records, frozen_clusters, templates),
        "bank": analysed_split_clusters([*frozen_records, *ext_records], clusters, templates),
    }
    return frozen_clusters, clusters, test_clusters


def analysed_split_clusters(
    records: Sequence[Mapping[str, Any]], clusters: Mapping[str, str], templates: Sequence[type[QuestionTemplate]]
) -> dict[str, Any]:
    """The test split's clusters: as the map gives them ("map", what the analysis reads), and recomputed over the
    test questions alone ("recomputed"; fewer joins when only a dev question links two test questions)."""
    test = [r for r in records if r["split"] == TEST]
    return {
        "map": cluster_summary({r["id"]: clusters[r["id"]] for r in test}),
        "recomputed": cluster_summary(anchor_clusters(test, templates)),
    }


def cluster_summary(clusters: Mapping[str, str]) -> dict[str, Any]:
    sizes = Counter(Counter(clusters.values()).values())
    return {
        "questions": len(clusters),
        "clusters": len(set(clusters.values())),
        "size_distribution": {str(k): sizes[k] for k in sorted(sizes)},
    }


PRIOR_EXCLUSION_RULE = (
    "no extension question names an anchor entity that a dev question of the same variant named in calibration "
    "round 1; such candidates are skipped at selection, dev included (stricter than the agent-facing rule, under which "
    "dev questions may reuse them), so no question is redrawn after the split"
)


def check_no_prior_anchors(
    records: Sequence[Mapping[str, Any]], templates: Sequence[type[QuestionTemplate]],
    prior: Mapping[str, Mapping[str, list[str]]],
) -> None:
    """Raise if any extension question, in any split, names an earlier round's dev anchor of its own variant."""
    by_id = {t.id: t for t in templates}
    bad = []
    for r in sorted(records, key=lambda r: r["id"]):
        excluded = prior.get(f"{r['template']}/{r['params']['variant']}", {})
        shared = anchor_entities(by_id[r["template"]].variant(r["params"]["variant"]), r["params"]) & set(excluded)
        if shared:
            bad.append((r["id"], sorted(shared)))
    if bad:
        raise TemplateError(f"extension questions name an earlier round's dev anchor of their variant: {bad[:5]}")


SPLIT_RULE = (
    "per template, 30% of its new questions go to dev (rounded half up, at least 3), so every template gets the same "
    "dev and test counts (214 new: 64 dev, 150 test); the template's dev count is shared over its tiers by largest "
    "remainder (each tier within one of 30%), then each tier's over its variants by largest remainder; within a "
    "variant the base ids are sorted, shuffled with a generator seeded from (seed, template, variant), and the first "
    "go to dev (design ruling in the design log). The frozen set rounded each tier on its own"
)


def tier_dev_counts(
    sizes: Mapping[str, int], dev_fraction: float = DEV_FRACTION, min_dev: int = MIN_DEV
) -> dict[str, int]:
    """Dev questions per tier of one template: the template's dev count shared over its tiers by largest remainder."""
    return allocate(dev_count(sum(sizes.values()), dev_fraction, min_dev), dict(sizes))


def extension_splits(
    questions: Sequence[Mapping[str, str]], seed: int, dev_fraction: float = DEV_FRACTION, min_dev: int = MIN_DEV
) -> dict[str, str]:
    """Map each base_question_id to "dev" or "test", exact per template (SPLIT_RULE).

    ``questions`` carry ``template``, ``base_question_id``, ``stratum`` (the variant) and ``tier``.
    """
    groups: dict[str, dict[str, dict[str, set[str]]]] = OrderedDict()
    owner: dict[str, tuple[str, str, str]] = {}
    for q in questions:
        template, base, stratum, tier = q["template"], q["base_question_id"], q["stratum"], q["tier"]
        if owner.setdefault(base, (template, stratum, tier)) != (template, stratum, tier):
            raise ValueError(f"base question {base} appears under two templates, variants or tiers")
        groups.setdefault(template, OrderedDict()).setdefault(tier, OrderedDict()).setdefault(stratum, set()).add(base)
    out: dict[str, str] = {}
    for template in sorted(groups):
        tiers = groups[template]
        tier_dev = tier_dev_counts({t: sum(len(b) for b in strata.values()) for t, strata in tiers.items()},
                                   dev_fraction, min_dev)
        for tier in sorted(tiers):
            strata = tiers[tier]
            shares = allocate(tier_dev[tier], {s: len(b) for s, b in strata.items()})
            for stratum in sorted(strata):
                bases = sorted(strata[stratum])
                rng_for(seed, "split", template, stratum).shuffle(bases)
                for i, base in enumerate(bases):
                    out[base] = DEV if i < shares[stratum] else TEST
    return out


def bank_checks(
    frozen_records: Sequence[Mapping[str, Any]],
    ext_records: Sequence[Mapping[str, Any]],
    templates: Sequence[type[QuestionTemplate]],
) -> dict[str, Any]:
    """The whole-bank checks; raise TemplateError on a violation that involves an extension question."""
    counts = Counter(r["id"] for r in ext_records)
    repeated = sorted(i for i, n in counts.items() if n > 1)
    if repeated:
        raise TemplateError(f"extension ids used more than once: {repeated[:5]}")
    frozen_ids = {r["id"] for r in frozen_records}
    clash = sorted(r["id"] for r in ext_records if r["id"] in frozen_ids)
    if clash:
        raise TemplateError(f"extension ids collide with frozen ids: {clash[:5]}")
    ext_ids = {r["id"] for r in ext_records}
    shares = shared_entities([*frozen_records, *ext_records], templates)
    involving = [s for s in shares if ext_ids.intersection(s["ids"])]
    if involving:
        raise TemplateError(
            "bank questions of one template share an entity: "
            f"{[(s['template'], s['entity'], s['ids']) for s in involving[:5]]}"
        )
    nested = nested_answer_pairs(frozen_records, ext_records)
    if nested:
        raise TemplateError(f"answers nested or equal across the bank: {nested[:5]}")
    ext_strata = check_value_diversity(list(ext_records))
    merged_strata = check_value_diversity([*frozen_records, *ext_records])
    return {
        "shared_entities": {
            "rule": ENTITY_RULE,
            "involving_extension": [],
            "frozen_only": [s for s in shares if not ext_ids.intersection(s["ids"])],
        },
        "nested_answer_pairs": {
            "rule": "no extension answer equals, contains or sits inside any other bank answer, in any template",
            "count": 0,
        },
        "value_diversity": {
            "rule": "per (variant, split): at most max(1, ceil(n / 4)) of n one-row questions share a value",
            "extension": _value_peaks(ext_strata),
            "merged_with_frozen": _value_peaks(merged_strata),
        },
    }


def _value_peaks(strata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: {split: {k: d[split][k] for k in ("questions", "one_row", "cap", "max_share_count")} for split in (DEV, TEST)}
        for key, d in strata.items()
    }


@dataclass
class Extension:
    records: list[dict[str, Any]]
    stats: dict[str, dict[str, VariantStats]]
    measured_seconds: dict[str, float]
    store_triples: dict[str, int]
    profile: Profile
    templates: tuple[type[QuestionTemplate], ...]
    seed: int
    per_template: int
    frozen: dict[str, Any]
    settings: dict[str, Any]
    checks: dict[str, Any]
    # Per template: planned and drawn questions per tier, and the cross-tier make-up (None when not needed).
    tier_mix: dict[str, dict[str, Any]]
    # Question id -> cluster id over the bank (frozen and extension), and over the frozen set alone.
    clusters: dict[str, str]
    frozen_clusters: dict[str, str]
    # The test split's clusters (analysed_split_clusters), frozen alone and bank.
    test_clusters: dict[str, Any]
    # The frozen records and the profile's (resolved) templates, for the cluster map.
    frozen_records: list[dict[str, Any]]
    bank_templates: tuple[type[QuestionTemplate], ...]
    prior: dict[str, Any] | None = None


def generate_extension(
    gold_client: SparqlClient,
    frozen_records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    per_template: int,
    profile: Profile,
    templates: Sequence[type[QuestionTemplate]] | None = None,
    pool_client: SparqlClient | None = None,
    prior_records: Sequence[Mapping[str, Any]] | None = None,
    prior_dev_anchors: bool = True,
    max_seconds: float = MAX_SECONDS,
) -> Extension:
    """Draw ``per_template`` new bank-only gold queries per template, seeded from the frozen set.

    ``frozen_records`` must serialise to the graph's pin in ``FROZEN_QUESTIONS_SHA256`` (never a caller's hash) and
    carry the generator version of the profile's knobs (the frozen set's templates and knobs). ``templates``
    (default: the profile's) may be a subset of the profile's, each once, for tests; the frozen set seeds from all
    of the profile's templates. ``prior_dev_anchors``: exclude at selection, in every split, any candidate naming
    an anchor of an earlier calibration round's dev question of its variant (PRIOR_EXCLUSION_RULE; the round the
    profile pins, ``prior_records`` its file); False drops the rule.

    A tier that cannot fill its count after its own variants make up each other is made up by the other tier of
    the same template (design ruling in the design log, change-log entry 52; recorded in ``tier_mix``); a template
    that cannot reach ``per_template`` as a whole fails the run.
    """
    if not profile.two_tier:
        raise TemplateError("the extension needs a two-tier profile")
    pin = FROZEN_QUESTIONS_SHA256.get(profile.graph_id)
    if pin is None:
        raise TemplateError(f"no frozen set pinned for an extension of {profile.graph_id}")
    got = hashlib.sha256(questions_jsonl(frozen_records).encode("utf-8")).hexdigest()
    if got != pin:
        raise TemplateError(f"the frozen questions have sha256 {got}, not the pinned {pin}")
    version = profile.generator_version(GENERATOR_VERSION)
    versions = sorted({str(r["generator_version"]) for r in frozen_records})
    if versions != [version]:
        raise TemplateError(
            f"the profile's knobs give generator version {version}, but the frozen questions carry {versions}: "
            "the extension uses the frozen set's templates and knobs"
        )
    prior = None
    if prior_dev_anchors:
        prior = _prior_round(profile, profile.templates, prior_records)
    elif prior_records is not None:
        raise TemplateError("earlier-round questions were given, but the earlier round's dev-anchor rule is dropped")

    def resolved(ts: Sequence[type[QuestionTemplate]]) -> tuple[type[QuestionTemplate], ...]:
        return resolve_templates(ts, profile) if profile.steps else tuple(ts)

    bank = resolved(profile.templates)
    run = resolved(templates if templates is not None else profile.templates)
    run_ids = [t.id for t in run]
    repeated = sorted({i for i in run_ids if run_ids.count(i) > 1})
    if repeated:
        raise TemplateError(f"templates given more than once: {repeated}")
    foreign = sorted(set(run_ids) - {t.id for t in bank})
    if foreign:
        raise TemplateError(f"templates not in the profile: {foreign}")
    answers, seeded = frozen_seeds(frozen_records, bank)
    pool_client = pool_client or gold_client
    knobs = dict(profile.knobs)
    triples = store_triples(pool_client)
    used_answers: list = list(answers)  # set-wide: shared by every template and tier
    stats: dict[str, dict[str, VariantStats]] = OrderedDict()
    picked: list[tuple[type[QuestionTemplate], Accepted]] = []
    tier_mix: dict[str, dict[str, Any]] = OrderedDict()
    shortfalls = []
    planned = profile.tier_counts(per_template)
    for template in run:
        used_keys = set(seeded.get(template.id, ()))  # keys and anchors, shared by both tiers

        def entities(c, t=template):
            return question_entities(t.variant(c.variant), c.params)

        exclude = None
        if prior is not None:
            def exclude(c, t=template, anchors=prior["anchors"]):
                # The earlier round's dev anchors of the candidate's own variant, in every split (option A).
                return not anchor_entities(t.variant(c.variant), c.params).isdisjoint(anchors.get(f"{t.id}/{c.variant}", {}))

        stats[template.id] = OrderedDict()
        tier_selector: dict[str, Selector] = {}
        drawn: dict[str, int] = {}
        for tier, count in planned.items():
            variants = [v for v in template.variants if v.tier == tier]
            if not variants or count <= 0:
                raise TemplateError(f"{template.id}: no {tier} variants for {count} questions")
            selectors: list[Selector] = []
            tier_stats, accepted = _sample_template(
                template, gold_client, pool_client, seed, count, MAX_ROWS, max_seconds,
                variants=variants, knobs=knobs, used=(used_keys, used_answers), forbid_names=False,
                strict_timeouts=True, selector_out=selectors, entities=entities, exclude=exclude,
                exclude_why="prior_dev_anchor",
            )
            stats[template.id].update(tier_stats)
            picked.extend((template, a) for a in accepted)
            tier_selector[tier], drawn[tier] = selectors[0], len(accepted)
        # Cross-tier make-up: after both tiers are drawn, so the other tier's own count is filled first.
        short = {tier: planned[tier] - drawn[tier] for tier in planned if drawn[tier] < planned[tier]}
        made_up = None
        if len(short) == 1:
            (tier, missing), = short.items()
            other = next(t for t in planned if t != tier)
            selector = tier_selector[other]
            extra = selector.more(missing)
            for a in extra:
                s = stats[template.id][a.candidate.variant]
                s.accepted += 1
                s.sampled_strata[a.candidate.stratum] += 1
                picked.append((template, a))
            drawn[other] += len(extra)
            made_up = {"by_tier": other, "for_tier": tier, "count": len(extra)}
        tier_mix[template.id] = {"planned": dict(planned), "drawn": dict(drawn), "made_up": made_up}
        if sum(drawn.values()) < per_template:
            shortfalls.append({
                "template": template.id, "drawn": dict(drawn), "made_up": made_up,
                "by_variant": {v: (s.quota, s.accepted) for v, s in stats[template.id].items()},
            })
    if shortfalls:
        raise TemplateError(f"extension short of {per_template} per template (variant: (quota, drawn)): {shortfalls}")

    prefix = extension_id_prefix(profile)
    counters: Counter = Counter()
    qids = []
    for template, _ in picked:
        counters[template.id] += 1
        qids.append(f"{prefix}-{template.id}-{counters[template.id]:03d}")
    splits = extension_splits(
        [
            {"template": t.id, "base_question_id": qid, "stratum": a.candidate.variant, "tier": t.variant(a.candidate.variant).tier}
            for (t, a), qid in zip(picked, qids)
        ],
        seed,
    )
    slow = [(q, round(a.payload.seconds, 2)) for (_, a), q in zip(picked, qids) if a.payload.seconds >= max_seconds]
    if slow:
        raise TemplateError(f"selected gold queries at or over the time cap of {max_seconds:g} s: {slow[:5]}")

    records = []
    measured: dict[str, float] = {}
    for (template, acc), qid in zip(picked, qids):
        variant = template.variant(acc.candidate.variant)
        gold = acc.payload
        params = acc.candidate.params
        measured[qid] = round(gold.seconds, 4)
        if nests_graph(gold.sparql):
            raise TemplateError(f"{qid}: the gold nests a GRAPH clause inside another")
        for anchor in variant.anchors:
            n = anchor_count(gold_client, variant, anchor, params)
            if n != 1:
                raise TemplateError(f"{qid}: the {anchor.kind} label anchor matches {n} entities, not one")
        records.append({
            "id": qid,
            "base_question_id": qid,
            "template": template.id,
            "variant": variant.name,
            "graph": profile.graph_id,
            "split": splits[qid],
            "tier": variant.tier,
            "question": None,
            "answer_columns": [c.to_json() for c in variant.answer_columns],
            "gold_sparql": gold.sparql,
            "gold_rows": gold.rows,
            "gold_row_count": len(gold.rows),
            "scoring_rules": dict(variant.scoring),
            "membership_mode": variant.mode(template),
            "hops": variant.hops,
            "difficulty": {**dict(variant.difficulty), **{k: params[k] for k in variant.knob_params}},
            "params": {"variant": variant.name, **params},
            "engine_sensitive": engine_reasons(gold_client, variant, params, gold.rows, strict_timeouts=True),
            "generator_version": EXTENSION_VERSION,
            "seed": seed,
        })
    records.sort(key=lambda r: r["id"])
    checks = bank_checks(frozen_records, records, bank)
    frozen_clusters, clusters, test_clusters = bank_clusters(frozen_records, records, bank)
    summary = None
    if prior is not None:
        check_prior_anchors(records, run, prior["anchors"])
        check_no_prior_anchors(records, run, prior["anchors"])
        summary = {**prior_summary(records, run, prior, []), "rule": PRIOR_EXCLUSION_RULE,
                   "mechanism": "excluded at selection in every split (rejection 'prior_dev_anchor'); nothing redrawn"}

    frozen_counts: dict[str, dict[str, Counter]] = OrderedDict()
    for r in frozen_records:
        frozen_counts.setdefault(r["template"], OrderedDict()).setdefault(r["params"]["variant"], Counter())[r["split"]] += 1
    prior_pin = profile.prior_dev_round()
    settings = {
        "max_gold_rows": {"value": MAX_ROWS, "fixed": True, "ruling": "kept, design ruling in the design log"},
        "min_gold_rows": {"value": MIN_ROWS},
        "max_gold_seconds": {"value": max_seconds, "strict_time_outs": True},
        "entity_uniqueness": {
            "scope": "whole bank (frozen and extension), per template: keys and anchors",
            "fixed": True,
            "ruling": "design ruling in the design log",
        },
        "answer_uniqueness": {"scope": "whole bank, set-wide (every template)", "seeded_answers": len(answers)},
        "prior_dev_anchors": {
            "kept": prior_dev_anchors,
            "applied": prior is not None,
            "pinned_round": None if prior_pin is None else {"round": prior_pin[0], "questions_sha256": prior_pin[1]},
            "status": "kept: design ruling in the design log (change-log entry 52); applied at selection (option A, design log)",
            "rule": PRIOR_EXCLUSION_RULE,
        },
        "split": {
            "rule": SPLIT_RULE,
            "dev_fraction": DEV_FRACTION,
            "min_dev": MIN_DEV,
            "note": "drawn after selection with the extension's seed; per template and tier: counts.by_template.*.by_tier",
        },
        "tier_counts_per_template": profile.tier_counts(per_template),
        "cross_tier_make_up": {
            "rule": "a tier that cannot fill its count after its own variants make up each other is made up by the other "
                    "tier of the same template (that tier's pools continue, round-robin over its variants); a template "
                    "that cannot reach its count as a whole fails the run",
            "ruling": "design ruling in the design log (change-log entry 52)",
        },
    }
    frozen_info = {
        "sha256": pin,
        "questions": len(frozen_records),
        "generator_version": version,
        "seeded": {
            "answers": len(answers),
            "entities_by_template": {t.id: len(seeded.get(t.id, ())) for t in bank},
        },
        "by_variant": {t: {v: {DEV: c[DEV], TEST: c[TEST]} for v, c in vs.items()} for t, vs in frozen_counts.items()},
    }
    return Extension(records, stats, measured, triples, profile, run, seed, per_template, frozen_info, settings,
                     checks, tier_mix, clusters, frozen_clusters, test_clusters,
                     [dict(r) for r in frozen_records], bank, summary)


# --------------------------------------------------------------------------
# Output


def cluster_map_json(
    records: Sequence[Mapping[str, Any]],
    templates: Sequence[type[QuestionTemplate]],
    *,
    frozen_sha256: str,
    extension_sha256: str,
) -> str:
    """The cluster-map/2 file the analysis reads (feat/analysis: harness/analysis/prereg.py, ClusterMap.load): per
    bank question (frozen and extension, dev included), its template, its declared anchors (sorted IRIs, possibly
    none) and its cluster id, so that the loader can recompute the components from the file itself."""
    by_id = {t.id: t for t in templates}
    clusters = anchor_clusters(records, templates)
    questions = {}
    for r in sorted(records, key=lambda r: r["id"]):
        variant = by_id[r["template"]].variant(r["params"]["variant"])
        questions[r["id"]] = {
            "template": r["template"],
            "anchors": sorted(anchor_entities(variant, r["params"])),
            "cluster": clusters[r["id"]],
        }
    return json.dumps({
        "schema": CLUSTER_MAP_SCHEMA,
        "rule": CLUSTER_RULE,
        "frozen_questions_sha256": frozen_sha256,
        "extension_questions_sha256": extension_sha256,
        "questions": questions,
    }, indent=1, ensure_ascii=False) + "\n"


def clusters_json(ext: Extension, questions_text: str) -> str:
    """The extension's cluster map (cluster-map/2) over the frozen set and the extension."""
    return cluster_map_json(
        [*ext.frozen_records, *ext.records], ext.bank_templates, frozen_sha256=ext.frozen["sha256"],
        extension_sha256=hashlib.sha256(questions_text.encode("utf-8")).hexdigest(),
    )


def clusters_section(
    clusters_text: str, frozen_clusters: Mapping[str, str], clusters: Mapping[str, str], test_clusters: Mapping[str, Any]
) -> dict[str, Any]:
    """The manifest's cluster fields."""
    return {
        "file": CLUSTERS_FILE,
        "schema": CLUSTER_MAP_SCHEMA,
        "sha256": hashlib.sha256(clusters_text.encode("utf-8")).hexdigest(),
        "rule": CLUSTER_RULE,
        "frozen": cluster_summary(frozen_clusters),
        "bank": cluster_summary(clusters),
        "test": {
            "note": "the analysed split (test questions only): 'map' as clusters.json gives them, 'recomputed' "
                    "over the test questions alone (it splits clusters that only a dev question joins)",
            **test_clusters,
        },
    }


def refresh_cluster_map(out_dir: Path, frozen_records: Sequence[Mapping[str, Any]], profile: Profile) -> Path:
    """Rewrite clusters.json and the manifest's cluster fields of a written extension from its records; the bank
    file must be the one its manifest records, and is not changed. No store query."""
    out_dir = Path(out_dir)
    text = (out_dir / EXTENSION_QUESTIONS_FILE).read_text(encoding="utf-8")
    mpath = out_dir / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    got = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if got != manifest["questions_sha256"]:
        raise TemplateError(f"{EXTENSION_QUESTIONS_FILE} has sha256 {got}, not its manifest's {manifest['questions_sha256']}")
    pin = FROZEN_QUESTIONS_SHA256.get(profile.graph_id)
    frozen_sha = hashlib.sha256(questions_jsonl(frozen_records).encode("utf-8")).hexdigest()
    if pin is None or frozen_sha != pin:
        raise TemplateError(f"the frozen questions have sha256 {frozen_sha}, not the pinned {pin}")
    bank = resolve_templates(profile.templates, profile) if profile.steps else tuple(profile.templates)
    records = [json.loads(line) for line in text.splitlines()]
    frozen_clusters, clusters, test_clusters = bank_clusters(frozen_records, records, bank)
    cluster_text = cluster_map_json([*frozen_records, *records], bank, frozen_sha256=pin, extension_sha256=got)
    cpath = out_dir / CLUSTERS_FILE
    cpath.write_text(cluster_text, encoding="utf-8")
    manifest["clusters"] = clusters_section(cluster_text, frozen_clusters, clusters, test_clusters)
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return cpath


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {DEV: sum(r["split"] == DEV for r in rows), TEST: sum(r["split"] == TEST for r in rows)}


def build_extension_manifest(
    ext: Extension,
    *,
    endpoint: str,
    questions_text: str,
    generated_at: str | None = None,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    profile = ext.profile
    by_template: dict[str, Any] = OrderedDict()
    bank_mix: dict[str, Any] = OrderedDict()
    for t in ext.templates:
        rows = [r for r in ext.records if r["template"] == t.id]
        by_template[t.id] = {
            "total": len(rows),
            **_split_counts(rows),
            "by_tier": {
                tier: {"total": sum(r["tier"] == tier for r in rows), **_split_counts([r for r in rows if r["tier"] == tier])}
                for tier in profile.tier_counts(ext.per_template)
            },
            "by_variant": {
                v.name: {
                    "tier": v.tier,
                    "quota": ext.stats[t.id][v.name].quota,
                    "total": sum(r["variant"] == v.name for r in rows),
                    **_split_counts([r for r in rows if r["variant"] == v.name]),
                }
                for v in t.variants
            },
        }
        bank_mix[t.id] = OrderedDict()
        for v in t.variants:
            frozen = ext.frozen["by_variant"].get(t.id, {}).get(v.name, {DEV: 0, TEST: 0})
            new = _split_counts([r for r in rows if r["variant"] == v.name])
            bank = {DEV: frozen[DEV] + new[DEV], TEST: frozen[TEST] + new[TEST]}
            bank_mix[t.id][v.name] = {"frozen": dict(frozen), "extension": new, "bank": {**bank, "total": bank[DEV] + bank[TEST]}}
    times = list(ext.measured_seconds.values())
    clusters_text = clusters_json(ext, questions_text)
    return {
        "graph": profile.graph_id,
        "generator_version": EXTENSION_VERSION,
        "mode": "bank-only extension: gold queries for the fault bank, no question texts, never shown to an agent",
        "seed": ext.seed,
        "per_template": ext.per_template,
        "endpoint": endpoint,
        "frozen": ext.frozen,
        "id_scheme": {"prefix": extension_id_prefix(profile), "rule": ID_RULE},
        "settings": ext.settings,
        "guards": GUARDS,
        "profile": {
            "calibration_round": profile.calibration_round(),
            "lookup_share": profile.lookup_share,
            "knobs": dict(profile.knobs),
        },
        "store": {**dict(profile.store_identity), "triples_per_graph": dict(sorted(ext.store_triples.items()))},
        "generated_at": generated_at or datetime.now(UTC).replace(microsecond=0).isoformat(),
        "questions_file": EXTENSION_QUESTIONS_FILE,
        "questions_sha256": hashlib.sha256(questions_text.encode("utf-8")).hexdigest(),
        "gold_rows_sha256": gold_rows_sha256(ext.records),
        "counts": {"total": len(ext.records), **_split_counts(ext.records), "by_template": by_template},
        "bank_mix": bank_mix,
        "tier_mix": ext.tier_mix,
        "clusters": clusters_section(clusters_text, ext.frozen_clusters, ext.clusters, ext.test_clusters),
        "checks": ext.checks,
        **({"prior_round_anchors": ext.prior} if ext.prior is not None else {}),
        "gold_query_seconds_measured": {
            "note": "first (cold) run of each selected gold; the records carry no time field",
            "max": max(times) if times else None,
            "total": round(sum(times), 4),
            "by_question": dict(sorted(ext.measured_seconds.items())),
            **({"generation_wall_seconds": round(wall_seconds, 1)} if wall_seconds is not None else {}),
        },
        "row_counts": row_count_distribution(ext.records, ext.templates),
        "sampling": {t: {v: s.to_json() for v, s in vs.items()} for t, vs in ext.stats.items()},
        "templates": [t.to_json(extended=True) for t in ext.templates],
    }


def write_extension(
    out_dir: Path,
    ext: Extension,
    *,
    endpoint: str,
    frozen_path: Path | None = None,
    generated_at: str | None = None,
    wall_seconds: float | None = None,
) -> tuple[Path, Path]:
    """Write questions-ext.jsonl, clusters.json and manifest.json; never into the frozen set's directory."""
    out_dir = Path(out_dir)
    if frozen_path is not None and out_dir.resolve() == Path(frozen_path).resolve().parent:
        raise ValueError(f"refusing to write the extension into the frozen set's directory {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    text = questions_jsonl(ext.records)
    qpath = out_dir / EXTENSION_QUESTIONS_FILE
    qpath.write_text(text, encoding="utf-8")
    (out_dir / CLUSTERS_FILE).write_text(clusters_json(ext, text), encoding="utf-8")
    manifest = build_extension_manifest(ext, endpoint=endpoint, questions_text=text, generated_at=generated_at,
                                        wall_seconds=wall_seconds)
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return qpath, mpath
