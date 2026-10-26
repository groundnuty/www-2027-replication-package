#!/usr/bin/env python3
"""The size calibration of every confirmatory claim (prereg §4.2; harness/analysis/calibration.py).

Subcommands:
- ``structure``: writes ``OUT/structure.json`` once, before submission (the structure every task verifies).
- ``simulate --step S --config I --chunk J``: runs datasets [J*size, (J+1)*size) of configuration I for step S
  (S = 1: N1 = 20,000; S = 2: N2 = 100,000, a fresh seed) and writes ``OUT/step{S}/c{I:02d}-{J:04d}.npz``
  (p-values: datasets x claims x tests). Slurm array tasks call this.
- ``alpha``: step 1, alpha'(a) from every step-1 chunk, and each family's tests and common step-1 factor (change-log
  entry 46: from step 1 only, so final); writes ``OUT/step1.json``.
- ``table``: both candidate families (design log), each through step 1 (tests and factor, on the step-1 chunks), then
  step 2's verification and the FWER (on the step-2 chunks), with ``OUT/step1.json``; writes ``OUT/calibration-table.json``
  (calibration-table/2, which prereg.Calibration reads), ``OUT/step2.json`` and ``OUT/calibration.md``.
- ``benchmark``: seconds per dataset on this machine, and the projected core-hours.
- ``plan``: the number of array tasks per step for a chunk size.
- ``simulate3 --phase selection|validation --task T``: one chunk of calibration step 3 (entry 106): C2 two-sided,
  the selection run on C2's two-sided null configurations (N3_SELECTION each, the selection seed) or the validation run
  on every configuration of step 3's list (N3 each, the validation seed), into ``OUT/step3-<phase>/``, refused before
  simulating unless ``--config`` binds this code's configurations and its fingerprint; ``plan3``: its task counts and
  the fingerprint to bind. The aggregation is scripts/weighted_holm_tables.py ``step3-select``, ``step3`` and
  ``step3-verify`` (scripts/calibrate-step3-aggregate.sbatch).
- ``choose-c1a --power-pooled P --power-two-family Q --entry81-dropped [MODEL ...]``: records the C1a family in Holm
  in the config (design log ruling 2: pooled if P >= Q; two-family when entry 81 took any model of P out,
  prereg.C1A_ELIGIBILITY), once, before the freeze.

The bank structure is the provisional UniProt bank's test split (source-free counts; --bank) with k_q = min(available,
--k) rewrites per question, the rewrite_redundant_join operator dropped (change-log entry 41); the analysis's questions
are the UniProt test split (210 base questions).

    uv run python scripts/calibrate.py simulate --step 1 --config 0 --chunk 0 --out OUT --bank BANK
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.analysis import calibration as cal
from harness.analysis.bank_bounds import BankDesign, load_bank_items
from harness.analysis.bank_power import rewrite_counts
from harness.analysis.digest import code_digest
from harness.analysis.inputs import load_questions
from harness.analysis.prereg import (
    CALIBRATION_RULES,
    STEP2_FWER_FAILURE,
    index_from_questions,
    record_c1a_choice,
    resamples,
)
from harness.analysis.simulate import BankStructure, bank_index, structure_from_bank

QUESTIONS = ROOT / "data" / "questions" / "uniprot-sp-2026_03" / "questions.jsonl"
CONFIG = ROOT / "data" / "analysis" / "prereg-config.json"
STEP2_SECONDS = 0.504  # step 2's measured cost per dataset on SiteA (arrays 17388-17390: 658.4 core-hours / 4.7e6)
JOIN = "rewrite_redundant_join"


def setup(args):
    cfg = cal.simulation_config()
    index = index_from_questions(load_questions(args.questions), cfg.split)
    items = load_bank_items(Path(args.bank) / "items.jsonl")
    d, _ = BankDesign.from_items(items, "test", gold_only=True)
    order = sorted({i["question_id"] for i in items if i["split"] == "test"})
    stratum = {}
    for i in items:
        if i["split"] == "test" and i.get("kind") == "control":
            stratum[i["question_id"]] = f"{i.get('template')}|{i.get('tier')}"
    k = rewrite_counts(items, "test", args.k, {JOIN})  # entry 41: min(available, 3), the join operator dropped
    strata = [stratum[q] for q in order]
    if args.per_template is None and len(order) == len(index):
        structure = structure_from_bank(strata, d.wrong, d.type_wrong, d.neutral, index, k)
        _record(args, "the bank's test split mapped onto the 210 agent questions", list(index.strata), structure)
        return cfg, index, cal.configurations(structure), None
    if args.per_template is None:  # the bank holds its own questions (a bank-only extension): its structure as it is
        structure = BankStructure(tuple(int(x) for x in d.type_wrong), tuple(int(x) for x in d.wrong - d.type_wrong),
                                  tuple(int(x) for x in d.neutral), tuple(int(x) for x in k))
        _record(args, "the bank's own test-split questions (a real bank-only extension)", strata, structure)
        return cfg, index, cal.configurations(structure), resamples(bank_index(len(order)), cfg)
    # a bank-only extension (entry 41): the per-question structure resampled within template x tier strata to
    # per_template questions per template (until the real extension's structure replaces it)
    rng = np.random.default_rng([cal.SEEDS[1], args.per_template])
    idx = []
    for st in sorted(set(strata)):
        members = [i for i, t in enumerate(strata) if t == st]
        idx.extend(rng.choice(members, size=len(members) * args.per_template // 30, replace=True).tolist())
    idx = np.array(idx)
    structure = BankStructure(tuple(int(x) for x in d.type_wrong[idx]), tuple(int(x) for x in (d.wrong - d.type_wrong)[idx]),
                              tuple(int(x) for x in d.neutral[idx]), tuple(int(x) for x in k[idx]))
    _record(args, f"resampled from the bank's 210 test questions within template x tier strata to {args.per_template} "
                  f"per template (seed {[cal.SEEDS[1], args.per_template]}); not a real extension", [strata[i] for i in idx],
            structure)
    return cfg, index, cal.configurations(structure), resamples(bank_index(len(idx)), cfg)


def _record(args, source: str, strata: list[str], st: BankStructure) -> None:
    """The structure the calibration runs on, per question in the form harness/analysis/structure_check.py compares
    with a real extension (W, C = gold + answer-neutral, T, k). The ``structure`` subcommand writes it once, before
    submission; every other subcommand verifies OUT/structure.json against it (``_verify_structure``) and the
    fingerprint binds its content."""
    rows = [{"template": s.split("|")[0], "tier": s.split("|")[1], "W": tw + ow, "C": 1 + ne, "T": tw, "k": kk}
            for s, tw, ow, ne, kk in zip(strata, st.type_wrong, st.other_wrong, st.neutral, st.rewrites)]
    args.structure_text = json.dumps({"source": source, "n": len(rows), "questions": rows}, indent=1) + "\n"


def fingerprint(args, confs) -> str:
    """Everything a chunk's p-values depend on (code, inputs, settings): every chunk records it, and the aggregation
    refuses any chunk with another (a calibration bound to its simulation settings)."""
    import hashlib

    cfg = cal.simulation_config()
    parts = {"code": code_digest(), "bank_items": hashlib.sha256((Path(args.bank) / "items.jsonl").read_bytes()).hexdigest(),
             "questions": hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(), "k": args.k,
             "per_template": args.per_template, "N1": cal.N1, "N2": cal.N2, "seeds": cal.SEEDS, "n_boot": cfg.n_boot,
             "seed": cfg.seed, "configurations": [c.name for c in confs], "scenarios": [repr(c.scenario) for c in confs],
             "structure": hashlib.sha256(args.structure_text.encode()).hexdigest(),
             # design log: the rule versions (grid, both families, the C1a choice, the joint step 1, the fallback, the
             # step-2 failure rule) and their grids
             "rules": CALIBRATION_RULES, "grid_steps": cal.GRID_STEPS, "capped_steps": cal.CAPPED_STEPS,
             "factor_steps": cal.FACTOR_STEPS, "families": cal.FAMILY, "seed_scheme": cal.SEED_SCHEME}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _verify_structure(args) -> None:
    path = Path(args.out) / "structure.json"
    if not path.is_file() or path.read_text(encoding="utf-8") != args.structure_text:
        raise SystemExit(f"{path}: missing or not this run's structure; run the structure subcommand once first")


def cmd_structure(args) -> int:
    setup(args)
    path = Path(args.out) / "structure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != args.structure_text:
        raise SystemExit(f"{path} holds another structure; use a fresh output directory")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(args.structure_text, encoding="utf-8")
    tmp.replace(path)  # atomic
    print(f"{path}: {json.loads(args.structure_text)['source']}")
    return 0


def cmd_simulate(args) -> int:
    cfg, index, confs, K_bank = setup(args)
    _verify_structure(args)
    K = resamples(index, cfg)
    n_total = cal.N1 if args.step == 1 else cal.N2
    lo, hi = args.chunk * args.size, min((args.chunk + 1) * args.size, n_total)
    conf = confs[args.config]
    p = np.stack([cal.dataset_p(conf.scenario, index, K, cfg, cal.SEEDS[args.step], args.config, i, K_bank)
                  for i in range(lo, hi)])
    out = Path(args.out) / f"step{args.step}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"c{args.config:02d}-{args.chunk:04d}.npz", p=p.astype(np.float64), lo=lo, hi=hi,
                        digest=fingerprint(args, confs), configuration=conf.name, config_index=args.config,
                        step=args.step, seed=cal.SEEDS[args.step], seed_scheme=cal.SEED_SCHEME)
    print(f"step {args.step} configuration {args.config} datasets {lo}..{hi}")
    return 0


def _load(out: Path, step: int, confs, digest: str) -> list[np.ndarray]:
    """Every configuration's p-values for one step: each chunk from this run's fingerprint and configuration, generated
    for this step (its recorded step, seed and seed scheme: a step-2 chunk among step 1's is refused), the chunks'
    [lo, hi) covering [0, N) exactly, without gaps or overlaps."""
    arrays = []
    want = cal.N1 if step == 1 else cal.N2
    for i, conf in enumerate(confs):
        chunks = []
        for f in sorted((out / f"step{step}").glob(f"c{i:02d}-*.npz")):
            z = np.load(f)
            if str(z["digest"]) != digest:
                raise SystemExit(f"{f}: produced under fingerprint {z['digest']}, not this run's {digest}")
            if str(z["configuration"]) != conf.name or int(z["config_index"]) != i:
                raise SystemExit(f"{f}: configuration {z['configuration']!s}, not {conf.name}")
            if not {"step", "seed", "seed_scheme"} <= set(z.files):
                raise SystemExit(f"{f}: records no step, seed and seed scheme")
            if int(z["step"]) != step:
                raise SystemExit(f"{f}: generated for step {int(z['step'])}, found among step {step}'s chunks")
            if int(z["seed"]) != cal.SEEDS[step]:
                raise SystemExit(f"{f}: generated with seed {int(z['seed'])}, not step {step}'s {cal.SEEDS[step]}")
            if str(z["seed_scheme"]) != cal.SEED_SCHEME:
                raise SystemExit(f"{f}: seed scheme {z['seed_scheme']!s}, not {cal.SEED_SCHEME}")
            lo, hi, p = int(z["lo"]), int(z["hi"]), z["p"]
            if len(p) != hi - lo:
                raise SystemExit(f"{f}: {len(p)} rows for datasets {lo}..{hi}")
            chunks.append((lo, hi, p))
        chunks.sort(key=lambda c: c[0])
        edge = 0
        for lo, hi, _ in chunks:
            if lo != edge:
                raise SystemExit(f"step {step} configuration {i}: datasets {edge}..{lo} missing or overlapping")
            edge = hi
        if edge != want:
            raise SystemExit(f"step {step} configuration {i}: datasets cover 0..{edge}, not 0..{want}")
        arrays.append(np.concatenate([c[2] for c in chunks]))
    return arrays


def cmd_alpha(args) -> int:
    cfg, index, confs, _ = setup(args)
    _verify_structure(args)
    digest = fingerprint(args, confs)
    p = _load(Path(args.out), 1, confs, digest)
    nulls = [cal.nulls(c.scenario, cfg, len(index)) for c in confs]
    alphas = cal.step_one(p, nulls)
    res = {"digest": digest, "N1": cal.N1, "seed": cal.SEEDS[1], "configurations": [c.name for c in confs],
           "alpha": alphas, "families": cal.step_one_families(p, nulls, alphas)}
    (Path(args.out) / "step1.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for fam, j in res["families"].items():  # entry 46: final, step 2 cannot change these
        tests = ", ".join(f"{c} {t or 'none'}" for c, t in j["tests"].items())
        print(f"{fam}: tests {tests}; step-1 FWER {max(j['fwer_unscaled']):.4f} unscaled, common factor {j['factor']}")
    print(f"{args.out}/step1.json")
    return 0


def _fmt(levels: dict | None, key: str) -> str:
    return "–" if levels is None else f"{levels[key]:.6g}"


def cmd_table(args) -> int:
    cfg, index, confs, _ = setup(args)
    _verify_structure(args)
    digest = fingerprint(args, confs)
    s1 = json.loads((Path(args.out) / "step1.json").read_text(encoding="utf-8"))
    if s1["digest"] != digest:
        raise SystemExit("step 1 was computed under another fingerprint (code, inputs or settings)")
    # step 1's tests and common factor (entry 46, ruling 3) are recomputed from the step-1 chunks; step 2 only verifies
    p1 = _load(Path(args.out), 1, confs, digest)
    p2 = _load(Path(args.out), 2, confs, digest)
    nulls = [cal.nulls(c.scenario, cfg, len(index)) for c in confs]
    res = cal.calibrate_families(p1, p2, nulls, [c.name for c in confs], s1["alpha"])
    meta = {"fingerprint": digest, "code_digest": code_digest(), "N1": cal.N1, "N2": cal.N2, "seeds": cal.SEEDS,
            "grid_steps": cal.GRID_STEPS, "configurations": [c.name for c in confs], "bank": str(args.bank), "k": args.k,
            "per_template": args.per_template, "structure": json.loads(args.structure_text)["source"]}
    tab = cal.table(res, meta)
    out = Path(args.out)
    (out / "calibration-table.json").write_text(json.dumps(tab, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "step2.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Size calibration (prereg §4.2, design log rulings)", "",
             (f"Fingerprint `{digest}`; N1 {cal.N1}, N2 {cal.N2}, seeds {cal.SEEDS}. The C1a family in Holm is chosen by "
              f"prereg.choose_c1a and recorded in the config; a chosen family failing step 2's FWER check gives "
              f"{STEP2_FWER_FAILURE}."), ""]
    for fam, f in res["families"].items():
        s1f, s2f = f["step1"], f["step2"]
        worst1 = max(s1f["fwer"]) if s1f["fwer"] else None
        worst2 = max((r["fwer"] for r in s2f["configurations"]), default=None)
        lines += [f"## {fam}", "",
                  (f"Common step-1 factor {s1f['factor']} (step-1 FWER {max(s1f['fwer_unscaled']):.4f} unscaled, "
                   f"{'–' if worst1 is None else f'{worst1:.4f}'} scaled; limit 0.05); step-2 FWER "
                   f"{'–' if worst2 is None else f'{worst2:.4f}'} at worst, passes: {f['fwer_pass']}."), "",
                  ("| Claim | Test in Holm | Step-1 test (step-1 size ok: primary, fallback) | Step 2 verifies | "
                   "alpha'(0.05/6) | alpha'(0.05) | unscaled alpha'(0.05/6) | capped at a: alpha'(0.05/6) |"),
                  "|---|---|---|---|---|---|---|---|"]
        for c, v in f["claims"].items():
            t1, ok = v["step1_test"], v["step1_size_ok"]
            capped = s1["alpha"][c][t1]["capped_at_a_diagnostic"] if t1 else None  # ruling 1's diagnostic
            verified = "–" if v["step2"] is None else ("yes" if v["step2"]["pass"] else "no (p = 1)")
            lines.append(f"| {c} | {v['test'] or 'none (p = 1)'} | {t1 or 'none'} ({ok['boot_t']}, {ok['cr2']}) | "
                         f"{verified} | {_fmt(v['alpha_prime'], '6')} | {_fmt(v['alpha_prime'], '1')} | "
                         f"{_fmt(v['alpha_prime_unscaled'], '6')} | {_fmt(capped, '6')} |")
        lines.append("")
    (out / "calibration.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{out}/calibration-table.json")
    return 0


# --- step 3 (prereg §4.2, change-log entry 106) ----------------------------------------------------------------------


def setup3(args):
    """Step 3's setup: the same bank structure as steps 1 and 2 (OUT/structure.json must be theirs), step 3's
    configuration list (calibration.step3_configurations) and each configuration's null claims with C2 two-sided,
    refused unless they are the config's bound step3_configurations (``bound3``)."""
    cfg, index, confs, K_bank = setup(args)
    confs3 = cal.step3_configurations(confs[0].scenario.bank.structure)
    nulls3 = [cal.nulls(c.scenario, cfg, len(index), two_sided_c2=True) for c in confs3]
    bound3(confs3, nulls3, getattr(args, "config", None) or CONFIG)
    return cfg, index, confs3, nulls3, K_bank


def bound3(confs3, nulls3, config: Path = CONFIG) -> None:
    """Every step-3 task and producer runs on the configurations and null claims bound in the config before the runs
    (step3_configurations by sha256; the review of 1e31f1f): this code's, in order, must be exactly them."""
    from harness.analysis import multiplicity as mu
    from harness.analysis.prereg import PLACEHOLDER, PreregConfig, PreregError

    cfg = PreregConfig.load(config)
    if PLACEHOLDER in (cfg.step3_configurations, cfg.step3_configurations_sha256):
        raise SystemExit("step 3's configurations are not bound (step3_configurations [FILL]): bound before it runs")
    try:
        bound = mu.read_step3_configurations(ROOT / cfg.step3_configurations, cfg.step3_configurations_sha256)
    except PreregError as exc:
        raise SystemExit(str(exc)) from None
    mine = [{"name": c.name, "null": [k for k in cal.CLAIMS if nl[k]]} for c, nl in zip(confs3, nulls3, strict=True)]
    if mine != bound:
        raise SystemExit(f"this code's {len(mine)} step-3 configurations and null claims are not the bound {len(bound)} "
                         f"({cfg.step3_configurations})")


def step3_phase(phase: str, nulls3) -> tuple[int, int, list[int]]:
    """(datasets per configuration, seed, configuration indices) of step 3's selection run (C2's two-sided null
    configurations) or validation run (every configuration)."""
    if phase == "selection":
        return cal.N3_SELECTION, cal.STEP3_SEEDS["selection"], [i for i, nl in enumerate(nulls3) if nl["C2"]]
    return cal.N3, cal.STEP3_SEEDS["validation"], list(range(len(nulls3)))


def fingerprint3(args, confs3) -> str:
    """Everything a step-3 chunk's p-values depend on: this code, the inputs and structure (as steps 1 and 2's),
    step 3's sizes, seeds, configurations and rules; every chunk records it and the producers refuse any other."""
    import hashlib

    from harness.analysis import multiplicity as mu

    sim = cal.simulation_config()
    parts = {"code": code_digest(), "bank_items": hashlib.sha256((Path(args.bank) / "items.jsonl").read_bytes()).hexdigest(),
             "questions": hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(), "k": args.k,
             "per_template": args.per_template, "N3_selection": cal.N3_SELECTION, "N3": cal.N3,
             "seeds": cal.STEP3_SEEDS, "n_boot": sim.n_boot, "seed": sim.seed,
             "configurations": [c.name for c in confs3], "scenarios": [repr(c.scenario) for c in confs3],
             "structure": hashlib.sha256(args.structure_text.encode()).hexdigest(), "rules": CALIBRATION_RULES,
             "step3_rules": mu.STEP3_RULES, "grid_steps": cal.GRID_STEPS, "seed_scheme": cal.SEED_SCHEME}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def bound_fingerprint3(args, confs3) -> str:
    """This task's fingerprint, refused before any simulation unless it is the step3_fingerprint ``args.config`` binds
    (plan3 prints it before submission; the review of 3b149eb: a task of another code tree would otherwise spend its
    compute on chunks every producer then refuses)."""
    from harness.analysis.prereg import PLACEHOLDER, PreregConfig

    fp, bound = fingerprint3(args, confs3), PreregConfig.load(args.config).step3_fingerprint
    if bound == PLACEHOLDER:
        raise SystemExit(f"{args.config}: step3_fingerprint is not bound: bind the fingerprint plan3 prints before "
                         f"submission")
    if fp != bound:
        raise SystemExit(f"this task's fingerprint {fp} is not the bound step3_fingerprint {bound} ({args.config})")
    return fp


def cmd_simulate3(args) -> int:
    cfg, index, confs3, nulls3, K_bank = setup3(args)
    _verify_structure(args)
    fp = bound_fingerprint3(args, confs3)
    n, seed, indices = step3_phase(args.phase, nulls3)
    per = -(-n // args.size)
    pos, chunk = divmod(args.task, per)
    if pos >= len(indices):
        raise SystemExit(f"task {args.task}: step 3's {args.phase} run has {len(indices) * per} tasks")
    i = indices[pos]
    K = resamples(index, cfg)
    lo, hi = chunk * args.size, min((chunk + 1) * args.size, n)
    conf = confs3[i]
    p = np.stack([cal.dataset_p(conf.scenario, index, K, cfg, seed, i, d, K_bank) for d in range(lo, hi)])
    out = Path(args.out) / f"step3-{args.phase}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"c{i:02d}-{chunk:04d}.npz"
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, p=p.astype(np.float64), lo=lo, hi=hi, digest=fp,
                        configuration=conf.name, config_index=i, step=3, phase=args.phase, seed=seed,
                        seed_scheme=cal.SEED_SCHEME)
    tmp.replace(path)  # atomic: a killed task leaves no partial chunk under the chunk's name
    print(f"step 3 {args.phase}: configuration {i} datasets {lo}..{hi}")
    return 0


def cmd_plan3(args) -> int:
    _, _, confs3, nulls3, _ = setup3(args)
    _verify_structure(args)
    total = 0
    for phase in ("selection", "validation"):
        n, seed, indices = step3_phase(phase, nulls3)
        per = -(-n // args.size)
        total += len(indices) * n
        print(f"step 3 {phase} (seed {seed}): {len(indices)} configurations x {per} chunks of {args.size} = "
              f"{len(indices) * per} tasks")
    print(f"{total} datasets, about {total * STEP2_SECONDS / 3600:.0f} core-hours at step 2's measured "
          f"{STEP2_SECONDS} s per dataset")
    print(f"fingerprint {fingerprint3(args, confs3)} (bind it in the config as step3_fingerprint before submission)")
    return 0


def cmd_choose_c1a(args) -> int:
    """Records the C1a family (design log ruling 2) from the two calibrated powers at the 3-point margin."""
    family = record_c1a_choice(args.config, args.power_pooled, args.power_two_family, dropped=args.entry81_dropped)
    print(f"{args.config}: c1a_family = {family}")
    return 0


def cmd_benchmark(args) -> int:
    cfg, index, confs, K_bank = setup(args)
    K = resamples(index, cfg)
    t = time.time()
    for i in range(args.datasets):
        conf = confs[i % len(confs)]
        cal.dataset_p(conf.scenario, index, K, cfg, 99, i % len(confs), i, K_bank)
    per = (time.time() - t) / args.datasets
    total = len(confs) * (cal.N1 + cal.N2) * per / 3600
    print(json.dumps({"seconds_per_dataset": per, "configurations": len(confs), "datasets": len(confs) * (cal.N1 + cal.N2),
                      "core_hours": total}))
    return 0


def cmd_plan(args) -> int:
    _, _, confs, _ = setup(args)
    for step, n in ((1, cal.N1), (2, cal.N2)):
        chunks = -(-n // args.size)
        print(f"step {step}: {len(confs)} configurations x {chunks} chunks of {args.size} = {len(confs) * chunks} tasks")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    choose = sub.add_parser("choose-c1a")
    choose.add_argument("--config", type=Path, default=CONFIG)
    choose.add_argument("--power-pooled", type=float, required=True,
                        help="the pooled C1a's calibrated power at the 3-point margin from the dev grid's rates")
    choose.add_argument("--power-two-family", type=float, required=True, help="the two-family C1a's, likewise")
    choose.add_argument("--entry81-dropped", nargs="*", required=True, metavar="MODEL",
                        help="the models of P entry 81 took out (none: the flag alone); any makes the two-family C1a "
                             "enter (prereg.C1A_ELIGIBILITY)")
    for name in ("structure", "simulate", "alpha", "table", "benchmark", "plan", "simulate3", "plan3"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=Path("calibration-out"))
        p.add_argument("--bank", type=Path, required=True)
        p.add_argument("--k", type=int, default=3, help="the cap on rewrites per question (entry 41: 3)")
        p.add_argument("--questions", type=Path, default=QUESTIONS)
        p.add_argument("--size", type=int, default=1000)
        p.add_argument("--per-template", type=int, default=None,
                       help="resample the bank's structure to this many questions per template (a stand-in for a bank-only "
                            "extension that is not built yet); by default the bank's own questions are used")
        if name == "simulate":
            p.add_argument("--step", type=int, choices=(1, 2), required=True)
            p.add_argument("--config", type=int, required=True)
            p.add_argument("--chunk", type=int, required=True)
        if name == "benchmark":
            p.add_argument("--datasets", type=int, default=20)
        if name in ("simulate3", "plan3"):
            p.add_argument("--config", type=Path, default=CONFIG,
                           help="the config binding step 3's configurations (and, for simulate3, its fingerprint)")
        if name == "simulate3":
            p.add_argument("--phase", choices=("selection", "validation"), required=True)
            p.add_argument("--task", type=int, required=True, help="the array task: configuration and chunk")
    args = ap.parse_args(argv)
    return {"structure": cmd_structure, "simulate": cmd_simulate, "alpha": cmd_alpha, "table": cmd_table, "benchmark": cmd_benchmark,
            "plan": cmd_plan, "choose-c1a": cmd_choose_c1a, "simulate3": cmd_simulate3, "plan3": cmd_plan3}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
