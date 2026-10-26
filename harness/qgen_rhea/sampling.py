"""Deterministic sampling of candidates, independent of the store.

Randomness comes only from ``random.Random`` instances seeded with a SHA-256
of the run seed and a purpose string, never from ``hash()`` (salted per
process) or global state. Given the same ordered candidate pools and the same
answers from ``evaluate``, the selection is the same on every run.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import OrderedDict
from itertools import zip_longest
from dataclasses import dataclass
from typing import Any, Callable, Generic, Hashable, Sequence, TypeVar

from harness.qgen_rhea.base import Candidate

T = TypeVar("T")


def rng_for(seed: int, *parts: str) -> random.Random:
    """A generator for one purpose, e.g. ``rng_for(seed, "sample", "type_members", "chebi_descendants")``."""
    material = ":".join([str(seed), *parts]).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


# Value diversity (design ruling in the design log): in a (variant, split) stratum of n
# single-value questions, at most value_cap(n) may share one answer value:
# ceil(0.25 * n), at least 1 (n=3 -> 1, n=7 -> 2, n=9 -> 3, n=20 -> 5). A
# two-valued variant is exempt but keeps each value at most 60% of a stratum.
VALUE_SHARE = 0.25
TWO_VALUED_SHARE = 0.6


def value_cap(n: int, two_valued: bool = False) -> int:
    """Questions of a stratum of ``n`` that may share one answer value."""
    if n <= 0:
        return 0
    if two_valued:
        return max(1, math.floor(TWO_VALUED_SHARE * n + 1e-9))
    return max(1, math.ceil(VALUE_SHARE * n - 1e-9))


def quotas(total: int, weights: Sequence[int]) -> list[int]:
    """Split ``total`` over variants in proportion to their weights, leftovers to the first ones."""
    if not weights or any(w <= 0 for w in weights):
        raise ValueError("need at least one variant, each with a positive weight")
    whole = sum(weights)
    out = [total * w // whole for w in weights]
    for i in range(total - sum(out)):
        out[i % len(out)] += 1
    return out


def shuffled_groups(candidates: Sequence[Candidate], rng: random.Random) -> list[list[Candidate]]:
    """Group candidates by key (pool order), shuffle the keys, then each key's options.

    An entity with several options (e.g. thresholds) is drawn once, like an
    entity with one option, so entities with many options are not favoured.
    When the candidates carry strata (answer shapes), the keys are shuffled
    within each stratum and the strata are interleaved in sorted order, so
    the draw alternates between shapes until a stratum runs out. Candidates
    with a higher priority (a hint from the candidate query that a preferred
    property is likely) come before all candidates of lower priority.
    """
    groups: "OrderedDict[str, list[Candidate]]" = OrderedDict()
    for c in candidates:
        groups.setdefault(c.key, []).append(c)
    by_priority: "dict[int, dict[str, list[str]]]" = {}
    for key, options in groups.items():
        by_priority.setdefault(options[0].priority, {}).setdefault(options[0].stratum, []).append(key)
    order: list[str] = []
    for priority in sorted(by_priority, reverse=True):
        by_stratum = by_priority[priority]
        shuffled = []
        for stratum in sorted(by_stratum):
            keys = list(by_stratum[stratum])
            rng.shuffle(keys)
            shuffled.append(keys)
        order.extend(k for batch in zip_longest(*shuffled) for k in batch if k is not None)
    out = []
    for key in order:
        options = list(groups[key])
        rng.shuffle(options)
        out.append(options)
    return out


@dataclass(frozen=True)
class Accepted(Generic[T]):
    candidate: Candidate
    payload: T
    tier: int = 0


# evaluate(candidate) -> (payload, answer) or (payload, answer, tier) to accept,
# or None to reject. The answer is the set of gold rows in a hashable,
# canonical form; the tier counts the preferred properties the candidate has.
Evaluate = Callable[[Candidate], "tuple | None"]


def select(
    pools: Sequence[Sequence[Sequence[Candidate]]],
    total: int,
    evaluate: Evaluate,
    on_skip: Callable[[Candidate, str], Any] | None = None,
    weights: Sequence[int] | None = None,
    top_tiers: Sequence[int] | None = None,
    nonpreferred_budget: int = 0,
    used_keys: set[str] | None = None,
    used_answers: list[frozenset] | None = None,
    value_of: Callable[[Candidate, Any], Hashable | None] | None = None,
    value_caps: Sequence[int | None] | None = None,
) -> list[Accepted]:
    """Pick up to ``total`` candidates from per-variant pools of option groups.

    ``pools[i]`` is variant i's list of option groups (from ``shuffled_groups``).
    Each variant first fills its quota (``weights``, default equal), from its
    own pool and then its own set-aside candidates (see Preference); only
    then do the others make up a remaining shortfall in variant order. A key
    (entity) is used at most once across all variants. An answer that equals,
    contains or is contained in an accepted answer is skipped
    ("nested_answer"), so no two questions of a template have the same answer
    or one answer inside the other, whichever side of the split each lands on.

    Preference: a candidate whose tier is below its variant's top tier
    (``top_tiers[i]``) is set aside unless ``nonpreferred_budget`` still allows
    one. Set-aside candidates are used, best tier first, only when the pools
    cannot fill the quota with preferred candidates: a variant short of its
    quota takes its own set-aside candidates before a sibling covers the rest
    with preferred ones (design ruling in the design log, Rhea round 2: before it,
    a sibling's preferred candidates came first, and absence_negation's
    unused_child_classes got 0 of 14). A variant that fills its quota from
    preferred candidates, as every round-1 variant did, picks the same either
    way. Set-aside candidates of any variant come last.

    ``used_keys`` and ``used_answers`` carry keys and answers already taken by
    an earlier call (another tier of the same template); they are updated in
    place with this call's picks.

    Value diversity: when ``value_of(candidate, payload)`` gives a value and
    ``value_caps[i]`` is set, variant i accepts at most that many candidates
    with one value; others are skipped ("value_cap").
    """
    tops = list(top_tiers or [0] * len(pools))
    used_keys = set() if used_keys is None else used_keys
    used_answers = [] if used_answers is None else used_answers
    accepted: list[Accepted] = []
    cursors = [iter(p) for p in pools]
    exhausted = [False] * len(pools)
    deferred: list[list[tuple[int, int, Candidate, Any, frozenset]]] = [[] for _ in pools]
    budget = [nonpreferred_budget]
    counter = [0]
    caps = list(value_caps or [None] * len(pools))
    value_counts: list[dict[Hashable, int]] = [{} for _ in pools]

    def value(c: Candidate, payload: Any) -> Hashable | None:
        return value_of(c, payload) if value_of is not None else None

    def capped(i: int, c: Candidate, payload: Any) -> bool:
        v = value(c, payload)
        return v is not None and caps[i] is not None and value_counts[i].get(v, 0) >= caps[i]

    def skip(c: Candidate, why: str) -> None:
        if on_skip is not None:
            on_skip(c, why)

    def nested(answer: frozenset) -> bool:
        return any(answer <= other or other <= answer for other in used_answers)

    def take(c: Candidate, payload: Any, answer: frozenset, tier: int, i: int) -> None:
        used_keys.add(c.key)
        used_answers.append(answer)
        accepted.append(Accepted(c, payload, tier))
        v = value(c, payload)
        if v is not None:
            value_counts[i][v] = value_counts[i].get(v, 0) + 1

    def draw(i: int) -> bool:
        """Accept one candidate from variant i; False once the pool is exhausted."""
        for options in cursors[i]:
            if options[0].key in used_keys:
                skip(options[0], "entity_used")
                continue
            for c in options:
                result = evaluate(c)
                if result is None:
                    continue
                payload, answer = result[0], result[1]
                tier = result[2] if len(result) > 2 else 0
                if nested(answer):
                    skip(c, "nested_answer")
                    continue
                if capped(i, c, payload):
                    skip(c, "value_cap")
                    continue
                if tier < tops[i]:
                    if budget[0] > 0:
                        budget[0] -= 1
                    else:
                        counter[0] += 1
                        deferred[i].append((tier, counter[0], c, payload, answer))
                        skip(c, "not_preferred")
                        break  # the entity's other options stay unused; try the next entity
                take(c, payload, answer, tier, i)
                return True
        exhausted[i] = True
        return False

    def fall_back(i: int) -> bool:
        deferred[i].sort(key=lambda d: (-d[0], d[1]))
        while deferred[i]:
            tier, _, c, payload, answer = deferred[i].pop(0)
            if c.key in used_keys or nested(answer) or capped(i, c, payload):
                continue
            take(c, payload, answer, tier, i)
            return True
        return False

    quota = quotas(total, weights or [1] * len(pools))
    got = [0] * len(pools)
    for i, q in enumerate(quota):
        while got[i] < q and draw(i):
            got[i] += 1
    # A variant whose pool ran dry first takes its own set-aside candidates; only then do the siblings cover.
    for i, q in enumerate(quota):
        while got[i] < q and len(accepted) < total and fall_back(i):
            got[i] += 1
    while len(accepted) < total and not all(exhausted):
        for i in range(len(pools)):
            if len(accepted) >= total:
                break
            if not exhausted[i] and draw(i):
                got[i] += 1
    while len(accepted) < total and any(deferred):
        progressed = False
        for i in range(len(pools)):
            if len(accepted) < total and fall_back(i):
                got[i] += 1
                progressed = True
        if not progressed:
            break
    return accepted
