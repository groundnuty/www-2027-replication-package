"""Development/test split, per template, grouped by base question.

Every question that shares a ``base_question_id`` (later: paraphrases and
opaque versions of one base question) lands on the same side, so no base
question is seen in development and scored in test.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping

from harness.qgen.sampling import rng_for, value_cap

DEV = "dev"
TEST = "test"
DEV_FRACTION = 0.3
MIN_DEV = 3


def dev_count(n_groups: int, dev_fraction: float = DEV_FRACTION, min_dev: int = MIN_DEV) -> int:
    """Base questions that go to dev: 30% rounded half up, at least ``min_dev``, at most all."""
    if n_groups <= 0:
        return 0
    return min(n_groups, max(min_dev, math.floor(n_groups * dev_fraction + 0.5)))


def allocate(total: int, sizes: Mapping[str, int]) -> dict[str, int]:
    """Share ``total`` dev slots over strata in proportion to their sizes (largest remainder).

    Ties in the remainder go to the larger stratum, then to the stratum name in
    sorted order, so the allocation is deterministic.
    """
    n = sum(sizes.values())
    if n == 0:
        return {s: 0 for s in sizes}
    exact = {s: total * size / n for s, size in sizes.items()}
    out = {s: math.floor(x) for s, x in exact.items()}
    order = sorted(sizes, key=lambda s: (-(exact[s] - out[s]), -sizes[s], s))
    for s in order[: total - sum(out.values())]:
        out[s] += 1
    return out


def assign_splits(
    questions: Iterable[Mapping[str, str]],
    seed: int,
    dev_fraction: float = DEV_FRACTION,
    min_dev: int = MIN_DEV,
) -> dict[str, str]:
    """Map each base_question_id to "dev" or "test".

    ``questions`` are mappings with ``template``, ``base_question_id``, an
    optional ``stratum`` (the template variant) and an optional ``tier``.
    Within each template (and tier, when given), ``dev_count`` base questions
    go to dev, shared over the strata in proportion to their sizes, so every
    variant shows up in dev, not only in test. Within a stratum the base ids
    are sorted, shuffled with a generator seeded from (seed, template,
    stratum), and the first ones go to dev.

    A question may carry an answer ``value`` (single-value variants) and
    ``two_valued``. Then the stratum's equal values are spread over dev and
    test so that neither side has more than ``value_cap`` of one value
    (design ruling in the design log): values that would exceed the test cap send
    their first members (in the shuffled order) to dev, and the remaining dev
    slots are filled in shuffled order without exceeding the dev cap. Strata
    without values are split exactly as before. Raises ValueError when the
    caps cannot be met. (Ported from feat/qgen, keeping this branch's tier key.)
    """
    strata: OrderedDict[tuple[str, str], OrderedDict[str, set[str]]] = OrderedDict()
    owner: dict[str, tuple[str, str, str]] = {}
    values: dict[str, str] = {}
    two_valued: dict[tuple[str, str, str], bool] = {}
    for q in questions:
        template, base, stratum, tier = q["template"], q["base_question_id"], q.get("stratum", ""), q.get("tier", "")
        if owner.setdefault(base, (template, stratum, tier)) != (template, stratum, tier):
            raise ValueError(f"base question {base} appears under two templates, variants or tiers")
        strata.setdefault((template, tier), OrderedDict()).setdefault(stratum, set()).add(base)
        if q.get("value") is not None:
            values[base] = q["value"]
            two_valued[(template, tier, stratum)] = bool(q.get("two_valued"))
    out: dict[str, str] = {}
    for template, tier in sorted(strata):
        groups = strata[(template, tier)]
        k = dev_count(sum(len(b) for b in groups.values()), dev_fraction, min_dev)
        shares = allocate(k, {s: len(b) for s, b in groups.items()})
        for stratum in sorted(groups):
            bases = sorted(groups[stratum])
            rng_for(seed, "split", template, stratum).shuffle(bases)
            if any(b in values for b in bases):
                dev = _value_spread_dev(bases, values, shares[stratum], two_valued.get((template, tier, stratum), False))
            else:
                dev = set(bases[: shares[stratum]])
            for base in bases:
                out[base] = DEV if base in dev else TEST
    return out


def _value_spread_dev(order: list[str], values: Mapping[str, str], n_dev: int, two_valued: bool) -> set[str]:
    """The dev members of a stratum, chosen in ``order`` so that no value exceeds its cap on either side."""
    n_test = len(order) - n_dev
    cap_dev, cap_test = value_cap(n_dev, two_valued), value_cap(n_test, two_valued)
    members: OrderedDict[str, list[str]] = OrderedDict()
    for base in order:
        members.setdefault(values.get(base, base), []).append(base)  # a base without a value is its own value
    dev: list[str] = []
    counts: dict[str, int] = {}
    for value, bases in members.items():  # values that would overflow test send their first members to dev
        need = max(0, len(bases) - cap_test)
        if need > cap_dev:
            raise ValueError(f"value {value!r} occurs {len(bases)} times; dev and test caps {cap_dev}+{cap_test}")
        dev.extend(bases[:need])
        counts[value] = need
    if len(dev) > n_dev:
        raise ValueError(f"{len(dev)} questions must go to dev to keep values within the caps, but dev has {n_dev}")
    chosen = set(dev)
    for base in order:  # fill the rest in shuffled order
        if len(chosen) >= n_dev:
            break
        value = values.get(base, base)
        if base not in chosen and counts.get(value, 0) < cap_dev:
            chosen.add(base)
            counts[value] = counts.get(value, 0) + 1
    if len(chosen) < n_dev:
        raise ValueError(f"cannot fill {n_dev} dev slots without exceeding the dev cap {cap_dev}")
    return chosen
