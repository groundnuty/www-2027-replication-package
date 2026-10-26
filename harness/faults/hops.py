"""How many swap candidates each distance offers, measured offline on the census graph alone.

For every predicate and class slot of every gold query and each candidate distance, this counts the
slots that have candidates and the candidates. Constants need the store, so they are left out. It
reads no check outcome.
"""

from __future__ import annotations

from collections import defaultdict

from harness.faults.operators import Context
from harness.faults.query import parse


def hop_profile(questions: list[dict], ctx: Context, distances: tuple[int, ...] = (2, 4, 6)) -> dict:
    out: dict = {}
    for d in distances:
        stats: dict = defaultdict(lambda: {"slots": 0, "with_candidates": 0, "candidates": 0})
        for question in questions:
            q = parse(question["gold_sparql"], ctx.is_class)
            for slot in q.slots:
                if slot.role not in ("predicate", "class"):
                    continue
                s = stats[slot.role]
                s["slots"] += 1
                cands = ctx.graph.candidates(slot.iri, slot.role, d)
                if cands:
                    s["with_candidates"] += 1
                    s["candidates"] += len(cands)
        out[str(d)] = {role: dict(s) for role, s in sorted(stats.items())}
    return out
