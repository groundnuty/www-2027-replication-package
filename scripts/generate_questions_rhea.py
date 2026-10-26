#!/usr/bin/env python3
"""Generate the template questions and gold answers for a pinned store.

Usage:
  uv run python scripts/generate_questions_rhea.py --endpoint http://localhost:7019/ \
      --seed 20261018 --per-template 43 --out data/questions/rhea-142/

Each template gets one third lookup questions (14 of 43) and the rest
compositional. --calibration-round picks the compositional tier's knobs of that
round (1: the frozen round-1 set 00e84760..., 2: every compositional variant one
structural step up; default 2); --knobs takes a JSON object (or a path to one)
overriding them, e.g. '{"path_hops": [6], "min_rows": 3}'. --previous-round
takes the previous round's questions.jsonl, which the manifest compares with
this one (changed texts, the lookup tier, stepped against redrawn items).

The store is checked first: its per-graph triple counts must equal the
expected counts (stores/rhea-142/expected_counts.json by default), so the
questions are never generated against the wrong store.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from harness.qgen_rhea.base import Knobs
from harness.qgen_rhea.generate import CALIBRATION_ROUNDS, MAX_SECONDS, generate, store_triples, write_outputs
from harness.qgen_rhea.templates import TEMPLATES, TEMPLATES_BY_ID
from harness.sparql import SparqlClient

REPO = Path(__file__).resolve().parent.parent
DEFAULT_EXPECTED = REPO / "stores" / "rhea-142" / "expected_counts.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--per-template", type=int, default=43)
    ap.add_argument("--lookup-per-template", type=int, default=None,
                    help="lookup questions per template (default: one third of --per-template)")
    ap.add_argument("--knobs", default=None, help="JSON object (or path to one) overriding the compositional knobs")
    ap.add_argument("--calibration-round", type=int, default=2, choices=sorted(CALIBRATION_ROUNDS))
    ap.add_argument("--previous-round", type=Path, default=None, help="the previous round's questions.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--templates", nargs="*", choices=sorted(TEMPLATES_BY_ID), help="subset of templates (default: all)")
    ap.add_argument("--min-preferred-share", type=float, default=1.0,
                    help="share of each template's questions that must have the variants' preferred properties "
                         "(discriminating direct vs transitive, or graph scope) when the pool allows; default all")
    ap.add_argument("--expected-counts", default=str(DEFAULT_EXPECTED),
                    help="JSON of expected triples per named graph; an empty string skips the check")
    args = ap.parse_args(argv)

    if args.per_template < 10:
        print("note: fewer than 10 questions per template is below the experiment's minimum", file=sys.stderr)
    templates = [TEMPLATES_BY_ID[t] for t in args.templates] if args.templates else list(TEMPLATES)
    knobs = CALIBRATION_ROUNDS[args.calibration_round]
    if args.knobs:
        raw = Path(args.knobs).read_text() if Path(args.knobs).is_file() else args.knobs
        overrides = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(raw).items()}
        unknown = set(overrides) - set(Knobs.__dataclass_fields__)
        if unknown:
            print(f"unknown knobs: {sorted(unknown)}", file=sys.stderr)
            return 2
        knobs = replace(knobs, **overrides)
    gold_client = SparqlClient(args.endpoint, timeout_s=MAX_SECONDS)
    pool_client = SparqlClient(args.endpoint, timeout_s=285)  # the store's limit is 290 s per query

    if args.expected_counts:
        expected = json.loads(Path(args.expected_counts).read_text())
        actual = store_triples(pool_client)
        if actual != expected:
            print(f"store mismatch: expected {expected}, endpoint has {actual}", file=sys.stderr)
            return 2

    previous = None
    if args.previous_round is not None:
        previous = [json.loads(line) for line in args.previous_round.read_text(encoding="utf-8").splitlines() if line]
    elif knobs.calibration_round() >= 2:
        # Its dev items' anchors are kept out of the test items, so the round cannot be drawn without it.
        print(f"round {knobs.calibration_round()} needs --previous-round (the previous round's questions.jsonl)", file=sys.stderr)
        return 2
    gen = generate(
        gold_client, args.seed, args.per_template, templates, pool_client=pool_client,
        min_preferred_share=args.min_preferred_share, knobs=knobs, lookup_per_template=args.lookup_per_template,
        previous_round=previous,
    )
    qpath, mpath = write_outputs(
        args.out, gen, seed=args.seed, per_template=args.per_template, endpoint=args.endpoint, templates=templates,
        calibration_round=knobs.calibration_round(), previous=previous,
    )
    print(f"knobs (round {args.calibration_round}): {json.dumps(knobs.to_json())}")
    for t in templates:
        rows = [r for r in gen.records if r["template"] == t.id]
        dev = sum(r["split"] == "dev" for r in rows)
        tiers = ", ".join(
            f"{tier} {sum(r['tier'] == tier for r in rows)} ({sum(r['tier'] == tier and r['split'] == 'dev' for r in rows)} dev)"
            for tier in ("lookup", "compositional")
        )
        print(f"{t.id:20} {len(rows):3} questions ({dev} dev, {len(rows) - dev} test): {tiers}")
        if len(rows) < args.per_template:
            print(f"  warning: only {len(rows)} of {args.per_template} requested", file=sys.stderr)
    print(f"label anchors: {gen.anchor_check['label_anchors_checked']} checked, each unique within its kind; "
          f"{gen.anchor_check['shared_with_the_other_kind']} share their label with the other kind (noted in the text)")
    overlaps = gen.overlaps
    print(f"split overlap: {len(overlaps['nested'])} nested dev/test answer pairs (refused if any); "
          f"{len(overlaps['equal_single_value'])} pairs of equal one-row numeric answers (listed); "
          f"{len(overlaps['intersecting'])} dev/test pairs share some rows")
    by_template: dict[str, int] = {}
    for pair in overlaps["intersecting"]:
        by_template[pair["template"]] = by_template.get(pair["template"], 0) + 1
    for template, n in by_template.items():
        print(f"  {template:20} {n} intersecting pairs")
    sensitive = [r for r in gen.records if r["engine_sensitive"]]
    print(f"engine-sensitive questions: {len(sensitive)}")
    same = sorted({f"{r['template']}/{r['params']['variant']}" for r in gen.records if r["same_family_columns"]})
    print(f"variants with same-family answer columns (order stated): {', '.join(same) or 'none'}")
    for t in templates:
        rows = [r for r in gen.records if r["template"] == t.id]
        dvt = [r["discriminates_direct_vs_transitive"] for r in rows if r["discriminates_direct_vs_transitive"] is not None]
        scope = [r["discriminates_graph_scope"] for r in rows]
        print(f"  {t.id:20} direct-vs-transitive {sum(dvt)}/{len(dvt)}  graph-scope {sum(scope)}/{len(scope)}")
    print(f"wrote {qpath} and {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
