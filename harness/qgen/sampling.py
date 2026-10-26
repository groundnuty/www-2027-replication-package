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

from harness.qgen.base import Candidate, TemplateError

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
    used_answers: list[frozenset[Hashable]] | None = None,
    entities: Callable[[Candidate], frozenset[str]] | None = None,
    exclude: Callable[[Candidate], bool] | None = None,
    exclude_why: str = "excluded",
    value_of: Callable[[Candidate, Any], Hashable | None] | None = None,
    value_caps: Sequence[int | None] | None = None,
    anchors_of: Callable[[Candidate], frozenset[str]] | None = None,
    used_anchors: set[str] | None = None,
) -> list[Accepted]:
    """Pick up to ``total`` candidates from per-variant pools of option groups (see ``Selector``)."""
    return Selector(
        pools, total, evaluate, on_skip, weights, top_tiers, nonpreferred_budget, used_keys, used_answers, entities,
        exclude, exclude_why, value_of, value_caps, anchors_of, used_anchors,
    ).run()


class Selector:
    """Picks candidates from per-variant pools of option groups, and can later redraw one position.

    ``pools[i]`` is variant i's list of option groups (from ``shuffled_groups``).
    Each variant first fills its quota (``weights``, default equal); if one
    runs dry, the others make up the shortfall in variant order. A key
    (entity) is used at most once across all variants. An answer that equals,
    contains or is contained in an accepted answer is skipped
    ("nested_answer"), so no two questions of a template have the same answer
    or one answer inside the other, whichever side of the split each lands on.

    Preference: a candidate whose tier is below its variant's top tier
    (``top_tiers[i]``) is set aside unless ``nonpreferred_budget`` still allows
    one. Set-aside candidates are used, best tier first, only when the pools
    cannot fill the quota with preferred candidates.

    ``used_keys`` and ``used_answers``, when given, are shared with an earlier
    call (another tier of the same template) and are updated in place.

    ``entities`` (the bank-only extension): the entities a candidate names
    beyond its key (its anchors). When given, an option naming any entity
    already in ``used_keys`` is skipped ("anchor_used"), each option on its
    own, and taking a candidate adds all of them; without it only keys count.

    ``exclude`` (the bank-only extension): options it rejects are skipped
    before evaluation, counted under ``exclude_why``, wherever the selector
    draws (run, more, redraw, the fall-back).

    Value diversity (profiles with audit_rules; off by default): when
    ``value_of(candidate, payload)`` gives a value and ``value_caps[i]`` is set,
    variant i accepts at most that many candidates with one value; others are
    skipped ("value_cap").

    ``anchors_of`` (with ``used_anchors``, shared the same way): the entities a
    candidate names by label; a candidate naming one already named by an
    accepted candidate is skipped ("anchor_used"), so no anchor entity appears
    in two questions, whichever variant or tier named it first.

    ``run`` makes the selection. ``redraw`` replaces one accepted position
    afterwards (the split is assigned after selection, so a rule about test
    questions can only be applied then): the position's variant continues from
    where its pool stopped, so every other position keeps its candidate.
    """

    def __init__(
        self,
        pools: Sequence[Sequence[Sequence[Candidate]]],
        total: int,
        evaluate: Evaluate,
        on_skip: Callable[[Candidate, str], Any] | None = None,
        weights: Sequence[int] | None = None,
        top_tiers: Sequence[int] | None = None,
        nonpreferred_budget: int = 0,
        used_keys: set[str] | None = None,
        used_answers: list[frozenset[Hashable]] | None = None,
        entities: Callable[[Candidate], frozenset[str]] | None = None,
        exclude: Callable[[Candidate], bool] | None = None,
        exclude_why: str = "excluded",
        value_of: Callable[[Candidate, Any], Hashable | None] | None = None,
        value_caps: Sequence[int | None] | None = None,
        anchors_of: Callable[[Candidate], frozenset[str]] | None = None,
        used_anchors: set[str] | None = None,
    ) -> None:
        self.pools = pools
        self.total = total
        self.evaluate = evaluate
        self.on_skip = on_skip
        self.weights = weights
        self.tops = list(top_tiers or [0] * len(pools))
        self.budget = nonpreferred_budget
        self.used_keys = set() if used_keys is None else used_keys
        self.used_answers = [] if used_answers is None else used_answers
        self.entities = entities
        self.exclude = exclude
        self.exclude_why = exclude_why
        self.anchors_of = anchors_of
        self.used_anchors = set() if used_anchors is None else used_anchors
        self.accepted: list[Accepted] = []
        # The pool (variant index) each accepted position came from.
        self.origin: list[int] = []
        self.cursors = [iter(p) for p in pools]
        self.exhausted = [False] * len(pools)
        self.deferred: list[list[tuple[int, int, Candidate, Any, frozenset]]] = [[] for _ in pools]
        self.counter = 0
        self.value_of = value_of
        self.caps = list(value_caps or [None] * len(pools))
        self.value_counts: list[dict[Hashable, int]] = [{} for _ in pools]

    def _skip(self, c: Candidate, why: str) -> None:
        if self.on_skip is not None:
            self.on_skip(c, why)

    def _nested(self, answer: frozenset) -> bool:
        return any(answer <= other or other <= answer for other in self.used_answers)

    def _names_used(self, c: Candidate) -> bool:
        return self.entities is not None and not self.used_keys.isdisjoint(self.entities(c))

    def _value(self, c: Candidate, payload: Any) -> Hashable | None:
        return self.value_of(c, payload) if self.value_of is not None else None

    def _capped(self, i: int, c: Candidate, payload: Any) -> bool:
        v = self._value(c, payload)
        return v is not None and self.caps[i] is not None and self.value_counts[i].get(v, 0) >= self.caps[i]

    def _anchor_used(self, c: Candidate) -> bool:
        return self.anchors_of is not None and bool(self.anchors_of(c) & self.used_anchors)

    def _take(self, c: Candidate, payload: Any, answer: frozenset, tier: int, i: int) -> Accepted:
        self.used_keys.add(c.key)
        if self.entities is not None:
            self.used_keys.update(self.entities(c))
        self.used_answers.append(answer)
        v = self._value(c, payload)
        if v is not None:
            self.value_counts[i][v] = self.value_counts[i].get(v, 0) + 1
        if self.anchors_of is not None:
            self.used_anchors.update(self.anchors_of(c))
        return Accepted(c, payload, tier)

    def _draw(self, i: int, veto: Callable[[Candidate], bool] | None = None, why: str = "vetoed") -> Accepted | None:
        """One candidate from variant i (None once the pool is exhausted); ``veto`` rejects candidates before evaluation."""
        for options in self.cursors[i]:
            if options[0].key in self.used_keys:
                self._skip(options[0], "entity_used")
                continue
            for c in options:
                if self._names_used(c):
                    self._skip(c, "anchor_used")
                    continue
                if self.exclude is not None and self.exclude(c):
                    self._skip(c, self.exclude_why)
                    continue
                if veto is not None and veto(c):
                    self._skip(c, why)
                    continue
                if self._anchor_used(c):
                    self._skip(c, "anchor_used")
                    continue
                result = self.evaluate(c)
                if result is None:
                    continue
                payload, answer = result[0], result[1]
                tier = result[2] if len(result) > 2 else 0
                if self._nested(answer):
                    self._skip(c, "nested_answer")
                    continue
                if self._capped(i, c, payload):
                    self._skip(c, "value_cap")
                    continue
                if tier < self.tops[i]:
                    if self.budget > 0:
                        self.budget -= 1
                    else:
                        self.counter += 1
                        self.deferred[i].append((tier, self.counter, c, payload, answer))
                        self._skip(c, "not_preferred")
                        break  # the entity's other options stay unused; try the next entity
                return self._take(c, payload, answer, tier, i)
        self.exhausted[i] = True
        return None

    def _fall_back(self, i: int, veto: Callable[[Candidate], bool] | None = None) -> Accepted | None:
        self.deferred[i].sort(key=lambda d: (-d[0], d[1]))
        kept = []
        found = None
        while self.deferred[i]:
            entry = self.deferred[i].pop(0)
            tier, _, c, payload, answer = entry
            if (c.key in self.used_keys or self._nested(answer) or self._names_used(c) or self._anchor_used(c)
                    or self._capped(i, c, payload)):
                continue
            if self.exclude is not None and self.exclude(c):  # never set aside (skipped when drawn); kept for safety
                continue
            if veto is not None and veto(c):
                kept.append(entry)
                continue
            found = self._take(c, payload, answer, tier, i)
            break
        self.deferred[i][:0] = kept
        return found

    def _append(self, i: int, acc: Accepted | None) -> bool:
        if acc is None:
            return False
        self.accepted.append(acc)
        self.origin.append(i)
        return True

    def run(self) -> list[Accepted]:
        pools = self.pools
        quota = quotas(self.total, self.weights or [1] * len(pools))
        got = [0] * len(pools)
        for i, q in enumerate(quota):
            while got[i] < q and self._append(i, self._draw(i)):
                got[i] += 1
        while len(self.accepted) < self.total and not all(self.exhausted):
            for i in range(len(pools)):
                if len(self.accepted) >= self.total:
                    break
                if not self.exhausted[i] and self._append(i, self._draw(i)):
                    got[i] += 1
        for i, q in enumerate(quota):
            while got[i] < q and len(self.accepted) < self.total and self._append(i, self._fall_back(i)):
                got[i] += 1
        while len(self.accepted) < self.total and any(self.deferred):
            progressed = False
            for i in range(len(pools)):
                if len(self.accepted) < self.total and self._append(i, self._fall_back(i)):
                    got[i] += 1
                    progressed = True
            if not progressed:
                break
        return self.accepted

    def more(self, n: int) -> list[Accepted]:
        """Draw up to ``n`` positions beyond the total, after ``run`` (the bank-only extension's cross-tier make-up).

        The pools continue where they stopped, round-robin over the variants as in ``run``'s second phase, then
        the set-aside candidates, best tier first; every rule of ``_draw`` applies. Returns the new positions
        (fewer than ``n`` once everything is used); they can be redrawn like any other.
        """
        target = len(self.accepted) + n
        start = len(self.accepted)
        while len(self.accepted) < target and not all(self.exhausted):
            for i in range(len(self.pools)):
                if len(self.accepted) >= target:
                    break
                if not self.exhausted[i]:
                    self._append(i, self._draw(i))
        while len(self.accepted) < target and any(self.deferred):
            progressed = False
            for i in range(len(self.pools)):
                if len(self.accepted) < target and self._append(i, self._fall_back(i)):
                    progressed = True
            if not progressed:
                break
        return self.accepted[start:]

    def redraw(self, n: int, veto: Callable[[Candidate], bool], why: str = "vetoed") -> Accepted:
        """Replace accepted position ``n`` by the next acceptable candidate of its variant that ``veto`` does not reject.

        The variant's pool continues where it stopped (then its set-aside
        candidates, best tier first), under the same rules: unused entity, no
        nested answer, preference. The dropped candidate's entity and answer stay
        used, so nothing else in the set can move. Raises TemplateError when no
        candidate is left.
        """
        i = self.origin[n]
        acc = self._draw(i, veto, why) or self._fall_back(i, veto)
        if acc is None:
            raise TemplateError(
                f"no candidate of variant {self.accepted[n].candidate.variant} left to redraw position {n} "
                f"(dropped {self.accepted[n].candidate.key})"
            )
        self.accepted[n] = acc
        return acc
