#!/usr/bin/env python3
"""Capacity of the frozen UniProt round-2 templates (experiments#12 feasibility; helper, not committed).

Reuses the generator's own pieces (candidate queries and expansion,
shuffled_groups, select, the evaluate checks of _sample_template and the
per-record checks of generate()) against the frozen set: the set's gold
answers seed the set-wide answer uniqueness, its keys seed each template's
used keys, its texts the duplicate-text check. Nothing under data/ is
written; output goes to --out (a scratch dir).

A failure that would abort a real generator run (a strict time-out, a cold
gold at 10 s or more, a warm gold at 1 s or more, an anchor that is not
unique at record time, nested GRAPH, a reading that fails or lacks its
clause, same-family columns without an order clause, a comparison whose
reorder changes the gold or breaks the step check) is caught per candidate,
tallied as "abort:<reason>" and the candidate is skipped, so the counts are
of candidates that pass every check.

Modes:
  joint     generate()'s selection for an extension of --per-template questions per template
  isolated  one variant at a time, quota removed (all entities, or the first --sample of the seeded order)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from harness.qgen import generate as G
from harness.qgen.base import DISCRIMINATIONS, TemplateError, resolve_templates
from harness.qgen.profiles import (
    PROFILES,
    TEMPLATE_SETS,
    expected_counts,
    store_mismatches,
    with_knobs,
)
from harness.qgen.readings import ReadingError, check_readings, enforce, nests_graph
from harness.qgen.sampling import quotas, rng_for, select, shuffled_groups
from harness.qgen.split import assign_splits
from harness.qgen.text_guard import database_names
from harness.sparql import SparqlClient, SparqlError
from harness.terms import row_from_json

REPO = Path(__file__).resolve().parent.parent
GRAPH = "uniprot-sp-2026_03"
FROZEN = REPO / "data" / "questions" / GRAPH / "questions.jsonl"
FROZEN_SHA = "0f84b9761825d436cfa1dceae8ffe2a3b1929fbf2bd97cdc83c598a575011cc9"
POOL_TIMEOUT = 280.0  # SparqlClient adds 10 s for the HTTP backstop: 290 s


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, file=sys.stderr, flush=True)


class Meter(SparqlClient):
    """A client that caches results by (time-out, query) and meters store time per phase.

    ``requested`` sums the elapsed time of every query asked for (cached or not): the
    store time a cold run of the same work would need.
    """

    phase = "other"

    def __init__(self, endpoint: str, timeout_s: float, cache: dict, meter: dict):
        super().__init__(endpoint, timeout_s=timeout_s)
        self.cache, self.meter = cache, meter

    def query(self, query: str):
        key = (self.timeout_s, query)
        m = self.meter[Meter.phase]
        hit = self.cache.get(key)
        if hit is None:
            try:
                r = super().query(query)
            except SparqlError as exc:
                m["errors"] += 1
                self.cache[key] = ("error", str(exc), exc.kind, exc.status)
                self.meter["_new"] += 1
                raise
            r = replace(r, raw={})
            self.cache[key] = r
            self.meter["_new"] += 1
            m["calls"] += 1
            m["store_s"] += r.elapsed_s
            hit = r
        else:
            m["hits"] += 1
        if isinstance(hit, tuple):
            m["requested_errors"] += 1
            raise SparqlError(hit[1], hit[2], hit[3])
        m["requested"] += 1
        m["requested_s"] += hit.elapsed_s
        return hit

    def fresh(self, query: str) -> float:
        """Run uncached (warm-time runs); returns the elapsed time."""
        r = SparqlClient.query(self, query)
        m = self.meter["warm"]
        m["calls"] += 1
        m["store_s"] += r.elapsed_s
        m["requested"] += 1
        m["requested_s"] += r.elapsed_s
        return r.elapsed_s


def new_meter() -> dict:
    d: dict = defaultdict(lambda: Counter())
    d["_new"] = 0
    return d


class Capacity:
    def __init__(self, args):
        self.args = args
        self.out = args.out
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache_path = args.cache
        self.cache: dict = pickle.loads(self.cache_path.read_bytes()) if self.cache_path.exists() else {}
        self.meter = new_meter()
        self.gold = Meter(args.endpoint, G.MAX_SECONDS, self.cache, self.meter)
        self.pool = Meter(args.endpoint, POOL_TIMEOUT, self.cache, self.meter)
        profile = PROFILES[GRAPH]
        self.profile = with_knobs(profile, dict(profile.calibration_rounds[2]))
        self.templates = resolve_templates(TEMPLATE_SETS[GRAPH], self.profile)
        self.knobs = dict(self.profile.knobs)
        self.seed = args.seed
        self.frozen = [json.loads(line) for line in FROZEN.read_text().splitlines()]
        self.frozen_answers = [G.row_set(r["gold_rows"]) for r in self.frozen]
        self.frozen_keys: dict[str, set[str]] = defaultdict(set)
        for r in self.frozen:
            t = next(t for t in self.templates if t.id == r["template"])
            v = t.variant(r["params"]["variant"])
            self.frozen_keys[t.id].add(str(r["params"][v.key_param]))
        self.frozen_texts = {r["question"] for r in self.frozen}
        self.trace = (self.out / "trace.jsonl").open("a")
        self.saved_at = time.monotonic()

    # -- cache -------------------------------------------------------------
    def save(self, force: bool = False) -> None:
        if not force and self.meter["_new"] < 300 and time.monotonic() - self.saved_at < 300:
            return
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(self.cache, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(self.cache_path)
        self.meter["_new"] = 0
        self.saved_at = time.monotonic()

    # -- pools ---------------------------------------------------------------
    def groups(self, template, variant):
        Meter.phase = "pool"
        result = self.pool.query(variant.candidate_sparql(self.knobs))
        candidates = variant.candidates(G.pool_rows(result), self.knobs)
        for c in candidates:
            c.params.update({k: self.knobs[k] for k in variant.knob_params})
        groups = shuffled_groups(candidates, rng_for(self.seed, "sample", template.id, variant.name))
        return candidates, groups

    # -- evaluate: _sample_template's checks, then generate()'s per-record checks ----------
    def evaluator(self, template, vstats: dict, used_answers: list, tag: str, post_limit: int | None = None):
        gold_client = self.gold
        posted = Counter()

        def reject(c, why: str) -> None:
            vstats[c.variant]["rejected"][why] += 1

        def trace(c, outcome: str, **extra) -> None:
            self.trace.write(json.dumps({"run": tag, "template": template.id, "variant": c.variant, "key": c.key,
                                         "outcome": outcome, **extra}) + "\n")

        def evaluate(c):
            variant = template.variant(c.variant)
            s = vstats[c.variant]
            s["evaluated"] += 1
            Meter.phase = "evaluate"
            text = variant.question(c.params)
            try:
                if database_names(text):
                    reject(c, "database_name")
                    trace(c, "database_name")
                    return None
                for anchor in variant.anchors:
                    if G.anchor_count(gold_client, variant, anchor, c.params) != 1:
                        reject(c, "ambiguous_anchor")
                        trace(c, "ambiguous_anchor")
                        return None
                sparql = variant.gold_sparql(c.params)
                result = gold_client.query(sparql)
                if result.total_rows < G.MIN_ROWS:
                    reject(c, "empty")
                    trace(c, "empty")
                    return None
                if result.total_rows > G.MAX_ROWS:
                    reject(c, "too_many_rows")
                    trace(c, "too_many_rows", rows=result.total_rows)
                    return None
                G.check_row_kinds(variant.answer_columns, result.rows)
                rows = G.canonical_rows(result.rows)
                if len({json.dumps(r, sort_keys=True) for r in rows}) != len(rows):
                    raise TemplateError("gold has duplicate rows")
                step = None
                if variant.step_check is not None:
                    step = G.step_outcome(gold_client, variant, c.params, rows)
                    if not step["matters"]:
                        reject(c, "step_not_load_bearing")
                        trace(c, "step_not_load_bearing")
                        return None
                gold = G.Gold(sparql, rows, result.elapsed_s, step=step)
                for name in variant.prefer:
                    gold.discriminates[name] = G.discrimination(
                        gold_client, name, sparql, variant.mode(template), rows, variant.alternative_sparql(c.params),
                        strict_timeouts=True,
                    )
                tier = sum(1 for name in variant.prefer if gold.discriminates[name])
                answer = G.row_set(rows)
                value = G.answer_value(rows) if len(rows) == 1 else None
                if any(answer <= o or o <= answer for o in used_answers):
                    trace(c, "nested_answer", rows=len(rows), value=value)
                    return gold, answer, tier  # select() skips it as nested_answer
                if text in self.frozen_texts:
                    raise TemplateError("duplicate question text (frozen set)")
                if post_limit is not None and posted[c.variant] >= post_limit:
                    s["post_unchecked"] += 1
                    trace(c, "pass_post_unchecked", rows=len(rows), value=value, cold=round(result.elapsed_s, 4))
                    return gold, answer, tier
                posted[c.variant] += 1
                post = self.post_checks(template, variant, c.params, gold, text)
            except (SparqlError, TemplateError, ReadingError, ValueError) as exc:
                kind = getattr(exc, "kind", None) or type(exc).__name__
                reason = f"abort:{kind}"
                reject(c, reason)
                trace(c, reason, error=str(exc)[:300])
                return None
            if post is not None:
                reject(c, f"abort:{post[0]}")
                trace(c, f"abort:{post[0]}", error=post[1][:300])
                return None
            trace(c, "pass", rows=len(rows), value=value, cold=round(result.elapsed_s, 4),
                  warm=round(self.last_warm, 4), tier=tier)
            return gold, answer, tier

        return evaluate, reject

    def post_checks(self, template, variant, params, gold, text):
        """generate()'s record checks and _balance_slots' reorder checks; (reason, detail) on a failure."""
        client = self.gold
        Meter.phase = "post"
        if gold.seconds >= G.MAX_SECONDS:
            return "cold_over_cap", f"{gold.seconds:.2f} s"
        if nests_graph(gold.sparql):
            return "nested_graph", ""
        Meter.phase = "warm"
        warm = min([gold.seconds, *(client.fresh(gold.sparql) for _ in range(G.WARM_RUNS))])
        self.last_warm = warm
        if warm >= G.WARM_GOLD_SECONDS:
            return "warm_over_1s", f"{warm:.2f} s"
        Meter.phase = "post"
        for anchor in variant.anchors:
            n = G.anchor_count(client, variant, anchor, params)
            if n != 1:
                return "anchor_not_unique", str(n)
        same_family = G.same_family_columns(variant.answer_columns, [row_from_json(r) for r in gold.rows])
        if same_family and not G.COLUMN_ORDER_MARKER.search(text):
            return "same_family_no_order", text
        mode = variant.mode(template)
        for name in DISCRIMINATIONS:
            if name not in gold.discriminates:
                G.discrimination(client, name, gold.sparql, mode, gold.rows, variant.alternative_sparql(params),
                                 strict_timeouts=True)
        readings = check_readings(client, variant, params, gold.sparql, gold.rows, text)
        enforce("candidate", readings)
        if variant.slots:
            Meter.phase = "slots"
            winner = variant.winner(params, gold.rows, gold.step)
            for target in range(len(variant.slots)):
                new = G.place_winner(params, variant.slots, winner, target, rng_for(self.seed, "slot-order", str(params)))
                sparql = variant.gold_sparql(new)
                if G.canonical_rows(client.query(sparql).rows) != gold.rows:
                    return "reorder_changes_gold", f"target {target}"
                if variant.step_check is not None and not G.step_outcome(client, variant, new, gold.rows)["matters"]:
                    return "reorder_breaks_step", f"target {target}"
                new_text = variant.question(new)
                if new_text in self.frozen_texts:
                    return "duplicate_text", new_text
        return None

    # -- runs ------------------------------------------------------------------
    def variant_stats(self, variants) -> dict:
        return OrderedDict((v.name, {"evaluated": 0, "rejected": Counter(), "post_unchecked": 0}) for v in variants)

    def jointmax(self, sample: int | None, sampled: list[str]) -> dict:
        """generate()'s selection over every template with the quota removed (bank-only, query-level checks:
        no record-level checks run). Variants in ``sampled`` use the first ``sample`` entities of their order."""
        tag = "jointmax"
        start = time.monotonic()
        used_answers = list(self.frozen_answers)
        report: dict[str, Any] = OrderedDict()
        for template in self.templates:
            used_keys = set(self.frozen_keys[template.id])
            report[template.id] = OrderedDict()
            for tier in ("lookup", "compositional"):
                variants = [v for v in template.variants if v.tier == tier]
                vstats = self.variant_stats(variants)
                pools = []
                for v in variants:
                    _, groups = self.groups(template, v)
                    vstats[v.name]["entities"] = len(groups)
                    vstats[v.name]["sampled"] = v.name in sampled and bool(sample)
                    pools.append(groups[:sample] if v.name in sampled and sample else groups)
                    vstats[v.name]["entities_examined"] = len(pools[-1])
                evaluate, reject = self.evaluator(template, vstats, used_answers, tag, post_limit=0)
                accepted = select(
                    pools, 10**9, evaluate, on_skip=reject, top_tiers=[len(v.prefer) for v in variants],
                    nonpreferred_budget=0, used_keys=used_keys, used_answers=used_answers,
                )
                for v in variants:
                    s = vstats[v.name]
                    s["accepted"] = sum(1 for a in accepted if a.candidate.variant == v.name)
                    s["rejected"] = dict(sorted(s["rejected"].items()))
                    report[template.id][v.name] = {"tier": tier, **s}
                    log(f"{tag} {template.id}/{v.name}: {s['accepted']} (examined {s['entities_examined']}/{s['entities']}, "
                        f"rejected {s['rejected']})")
                self.trace.flush()
                self.save()
        out = {"run": tag, "seed": self.seed, "sample": sample, "sampled": sampled, "by_variant": report,
               "wall_s": round(time.monotonic() - start, 1)}
        self.save(force=True)
        (self.out / f"{tag}.json").write_text(json.dumps(out, indent=1, default=list))
        return out

    def joint(self, per_template: int) -> dict:
        tag = f"joint-{per_template}"
        start = time.monotonic()
        meter0 = {k: Counter(v) for k, v in self.meter.items() if k != "_new"}
        used_answers = list(self.frozen_answers)
        picked = []
        report: dict[str, Any] = OrderedDict()
        for template in self.templates:
            used_keys = set(self.frozen_keys[template.id])
            report[template.id] = OrderedDict()
            for tier, count in self.profile.tier_counts(per_template).items():
                variants = [v for v in template.variants if v.tier == tier]
                weights = [int(self.knobs.get("variant_weights", {}).get(v.name, v.weight)) for v in variants]
                vstats = self.variant_stats(variants)
                pools = []
                for v in variants:
                    candidates, groups = self.groups(template, v)
                    vstats[v.name]["candidates"] = len(candidates)
                    vstats[v.name]["entities"] = len(groups)
                    pools.append(groups)
                evaluate, reject = self.evaluator(template, vstats, used_answers, tag)
                accepted = select(
                    pools, count, evaluate, on_skip=reject, weights=weights,
                    top_tiers=[len(v.prefer) for v in variants], nonpreferred_budget=0,
                    used_keys=used_keys, used_answers=used_answers,
                )
                for v, q in zip(variants, quotas(count, weights)):
                    vstats[v.name]["quota"] = q
                    vstats[v.name]["accepted"] = sum(1 for a in accepted if a.candidate.variant == v.name)
                for a in accepted:
                    picked.append((template, a))
                for name, s in vstats.items():
                    s["rejected"] = dict(sorted(s["rejected"].items()))
                    report[template.id][name] = {"tier": tier, **s}
                    log(f"{tag} {template.id}/{name}: {s['accepted']}/{s['quota']} (evaluated {s['evaluated']}, "
                        f"rejected {s['rejected']})")
                self.trace.flush()
                self.save()
        # Splits and the set-level checks, as generate() makes them (ids continue the frozen numbering).
        counters = Counter({t.id: 43 for t in self.templates})
        recs = []
        for template, acc in picked:
            counters[template.id] += 1
            qid = f"{self.profile.id_prefix}-{template.id}-{counters[template.id]:03d}"
            v = template.variant(acc.candidate.variant)
            recs.append({"template": template.id, "base_question_id": qid, "stratum": v.name, "tier": v.tier,
                         "rows": acc.payload.rows, "text": v.question(acc.candidate.params)})
        splits = assign_splits(recs, self.seed)
        split_counts: dict[str, Counter] = defaultdict(Counter)
        for r in recs:
            r["split"] = splits[r["base_question_id"]]
            split_counts[r["template"]][r["split"]] += 1
            split_counts[f"{r['template']}/{r['tier']}"][r["split"]] += 1
        texts = Counter(r["text"] for r in recs)
        diversity = self.value_diversity(recs)
        meter1 = {k: Counter(v) for k, v in self.meter.items() if k != "_new"}
        out = {
            "run": tag,
            "seed": self.seed,
            "per_template": per_template,
            "tier_counts": self.profile.tier_counts(per_template),
            "accepted": len(picked),
            "by_variant": report,
            "splits": {k: dict(v) for k, v in split_counts.items()},
            "duplicate_texts": sum(n - 1 for n in texts.values() if n > 1),
            "value_diversity": diversity,
            "wall_s": round(time.monotonic() - start, 1),
            "meter": {k: {f: round(x, 3) for f, x in (meter1[k] - meter0.get(k, Counter())).items()} for k in meter1},
            "meter_requested": {k: {"requested": meter1[k]["requested"], "requested_s": round(meter1[k]["requested_s"], 2)}
                                for k in meter1},
        }
        self.save(force=True)
        (self.out / f"{tag}.json").write_text(json.dumps(out, indent=1, default=list))
        return out

    def value_diversity(self, recs) -> dict:
        """Per (variant, split) of one-row answers: the most common stripped value against the cap,
        for the extension alone and merged with the frozen set (split as recorded)."""
        def strata(rows):
            out = {}
            groups: dict[tuple, Counter] = defaultdict(Counter)
            for variant, split, gold in rows:
                if len(gold) == 1:
                    groups[(variant, split)][G.answer_value(gold)] += 1
            for (variant, split), c in groups.items():
                n = sum(c.values())
                out[f"{variant}|{split}"] = {"one_row": n, "cap": G.value_cap(n), "max_share": max(c.values())}
            broken = {k: v for k, v in out.items() if v["max_share"] > v["cap"]}
            return {"strata": len(out), "broken": broken}
        ext = [(f"{r['template']}/{r['stratum']}", r["split"], r["rows"]) for r in recs]
        frozen = [(f"{r['template']}/{r['params']['variant']}", r["split"], r["gold_rows"]) for r in self.frozen]
        return {"extension": strata(ext), "merged": strata(ext + frozen)}

    def isolated(self, names: list[str], sample: int | None, post_limit: int | None) -> dict:
        results = {}
        for template in self.templates:
            for v in template.variants:
                if names and v.name not in names:
                    continue
                tag = f"isolated-{v.name}"
                start = time.monotonic()
                meter0 = {k: Counter(x) for k, x in self.meter.items() if k != "_new"}
                candidates, groups = self.groups(template, v)
                pool = groups[:sample] if sample else groups
                vstats = self.variant_stats([v])
                used_answers = list(self.frozen_answers)
                used_keys = set(self.frozen_keys[template.id])
                evaluate, reject = self.evaluator(template, vstats, used_answers, tag, post_limit=post_limit)
                accepted = select(
                    [pool], 10**9, evaluate, on_skip=reject, top_tiers=[len(v.prefer)], nonpreferred_budget=0,
                    used_keys=used_keys, used_answers=used_answers,
                )
                s = vstats[v.name]
                meter1 = {k: Counter(x) for k, x in self.meter.items() if k != "_new"}
                values = Counter(G.answer_value(a.payload.rows) for a in accepted if len(a.payload.rows) == 1)
                res = {
                    "template": template.id,
                    "tier": v.tier,
                    "candidates": len(candidates),
                    "entities": len(groups),
                    "entities_examined": len(pool),
                    "frozen_keys_in_pool": sum(1 for g in groups if g[0].key in self.frozen_keys[template.id]),
                    "full": sample is None or sample >= len(groups),
                    "evaluated_options": s["evaluated"],
                    "accepted": len(accepted),
                    "accepted_preferred": sum(1 for a in accepted if a.tier >= len(v.prefer)),
                    "accepted_post_unchecked": s["post_unchecked"],
                    "one_row_answers": sum(values.values()),
                    "rejected": dict(sorted(s["rejected"].items())),
                    "wall_s": round(time.monotonic() - start, 1),
                    "meter": {k: {f: round(x, 3) for f, x in (meter1[k] - meter0.get(k, Counter())).items()} for k in meter1},
                }
                results[v.name] = res
                log(f"{tag}: accepted {res['accepted']} of {res['entities_examined']}/{res['entities']} entities "
                    f"({res['evaluated_options']} options), rejected {res['rejected']}, {res['wall_s']} s")
                (self.out / f"{tag}.json").write_text(json.dumps(res, indent=1))
                self.trace.flush()
                self.save(force=True)
        return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["joint", "isolated", "jointmax", "check"])
    ap.add_argument("--endpoint", default="http://localhost:7020/")
    ap.add_argument("--seed", type=int, default=20271018)
    ap.add_argument("--per-template", type=int, nargs="*", default=[43])
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--sample", type=int, default=None, help="isolated: the first N entities of the seeded order")
    ap.add_argument("--sampled", nargs="*", default=[], help="jointmax: variants limited to --sample entities")
    ap.add_argument("--post-limit", type=int, default=None, help="isolated: full record checks for the first N passes only")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    args = ap.parse_args(argv)
    import hashlib
    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != FROZEN_SHA:
        print("frozen questions.jsonl is not 0f84b976…", file=sys.stderr)
        return 2
    cap = Capacity(args)
    Meter.phase = "pool"
    mism = store_mismatches(G.store_triples(SparqlClient(args.endpoint, timeout_s=POOL_TIMEOUT)), expected_counts(GRAPH))
    if mism:
        print("store mismatch:\n  " + "\n  ".join(mism), file=sys.stderr)
        return 2
    log("store matches the pinned triple counts")
    if args.mode == "check":
        return 0
    if args.mode == "joint":
        for n in args.per_template:
            out = cap.joint(n)
            log(f"joint {n}: accepted {out['accepted']}, wall {out['wall_s']} s")
    elif args.mode == "jointmax":
        cap.jointmax(args.sample, args.sampled)
    else:
        cap.isolated(args.variants, args.sample, args.post_limit)
    cap.save(force=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
