#!/usr/bin/env python3
"""The weighted Holm's frozen inputs (design log, change-log entries 74, 76 and 91), written by the registered producers
in harness/analysis/multiplicity.py, each file with its sha256 printed for the config.

Subcommands:
- ``alpha``: step 1's alpha' at every reachable level, per family and claim, with each claim's test and the family's
  common factor (``multiplicity.step_one_table``), from step 1's stored p-values (calibrate.py's ``OUT/step1`` chunks):
  the weighted-holm-alpha/1 table the config's ``multiplicity_table`` names. Before computing, it binds the chunks to
  step 1: first by their content (``_step1_content``: ``multiplicity.outputs_digest`` of ``OUT/step1`` must be the
  config's ``step1_outputs``, before any chunk is read for its p-values: step 1 recorded only the fingerprint, which
  every chunk carries whatever its p-values, so p-values moved between datasets within a configuration would keep
  every marginal alpha' and the metadata and change the cross-claim dependence); ``OUT/step1.json`` must have prereg
  §5's sha256; the fingerprint recomputed from the bank, the questions, the
  structure and the calibrated commit's code (``--calibrated-tree``, a checkout of 87c7126, whose code digest is prereg
  §5's) must be step 1's, and every chunk must carry it; the configurations' null claims must give step1.json's counts;
  and the same grid rule at the plain-Holm levels 0.05/k must give step1.json's alpha' exactly.
- ``step2``: step 2's verification of that table on step 2's chunks (``OUT/step2``; ``multiplicity.step_two``), one
  configuration at a time, as the dated step-2 addendum (entry 91; ``multiplicity.addendum_document``) the config's
  ``step2_addendum`` names. It refuses unless the current multiplicity code is the config's frozen digest (the
  procedure is bound at the freeze).

    uv run python scripts/weighted_holm_tables.py alpha --calibration-out OUT --bank BANK --calibrated-tree TREE \\
        --out data/analysis/weighted-holm-alpha.json
    uv run python scripts/weighted_holm_tables.py step2 --calibration-out OUT --out data/analysis/weighted-holm-step2.json
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import datetime
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.analysis import calibration as cal
from harness.analysis import multiplicity as mu
from harness.analysis.digest import code_digest
from harness.analysis.inference import multiplicity_digest
from harness.analysis.prereg import (
    PLACEHOLDER,
    TESTS,
    PreregConfig,
    PreregError,
)

CONFIG = ROOT / "data" / "analysis" / "prereg-config.json"
STEP1_SHA256 = "0e9232e9f3d0f7766ac54e631bbee5ad80d82228b38f994355e951e4f6eb89c7"  # prereg §5 "Calibration of tests"
CALIBRATED_CODE_DIGEST = "81708e646509b230ac6de30d5655a82f7c4a88ebc3e2bd8ed2470b18b0d39a68"  # prereg §5 (87c7126)
OTHERS = ("x2", "x3", "x4", "x5", "x6")  # with a claim, a six-claim family of equal weights: s = 1..6, 0.05/k


def _calibrate():
    spec = importlib.util.spec_from_file_location("calibrate", ROOT / "scripts" / "calibrate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, doc: dict) -> str:
    raw = mu.document_bytes(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _step1_content(out: Path, frozen: dict) -> dict:
    """Step 1's chunks by their content (``multiplicity.outputs_digest`` of ``out/step1``), refused unless they are the
    config's frozen ``step1_outputs`` (the review of 9d9c88c): never recorded as found."""
    found = mu.outputs_digest(out, 1)
    if not frozen or found != frozen:
        raise SystemExit(f"{out / 'step1'}: the chunks' content {found} is not step 1's frozen content "
                         f"{frozen or '(step1_outputs is not in the config)'}: refused before any chunk's p-values are used")
    return found


def cmd_alpha(args) -> int:
    cfg = PreregConfig.load(args.config)
    out = args.calibration_out
    content = _step1_content(out, cfg.step1_outputs)
    raw = (out / "step1.json").read_bytes()
    s1_sha = hashlib.sha256(raw).hexdigest()
    if s1_sha != STEP1_SHA256:
        raise SystemExit(f"{out / 'step1.json'}: sha256 {s1_sha}, not prereg §5's step 1 {STEP1_SHA256}")
    s1 = json.loads(raw)
    calibrated = code_digest(args.calibrated_tree)
    if calibrated != CALIBRATED_CODE_DIGEST:
        raise SystemExit(f"{args.calibrated_tree}: code digest {calibrated}, not the calibrated commit's "
                         f"{CALIBRATED_CODE_DIGEST}")
    C = _calibrate()
    setup = argparse.Namespace(out=out, bank=args.bank, k=args.k, questions=args.questions, size=1000,
                               per_template=args.per_template)
    sim, index, confs, _ = C.setup(setup)
    C._verify_structure(setup)
    C.code_digest = lambda: calibrated  # the fingerprint binds the code the chunks were simulated on, 87c7126's
    fingerprint = C.fingerprint(setup, confs)
    if fingerprint != s1["digest"] or [c.name for c in confs] != s1["configurations"]:
        raise SystemExit(f"fingerprint {fingerprint} and the configurations are not step 1's ({s1['digest']})")
    nulls = [cal.nulls(c.scenario, sim, len(index)) for c in confs]
    counts = {c: sum(nl[c] for nl in nulls) for c in cal.CLAIMS}
    if counts != {c: s1["alpha"][c]["null_configurations"] for c in cal.CLAIMS}:
        raise SystemExit(f"null configurations per claim {counts} are not step 1's")
    p1 = [mu.configuration_p(out, 1, i, conf.name, fingerprint, cal.N1, cal.SEEDS[1], cal.SEED_SCHEME)
          for i, conf in enumerate(confs)]
    for ci, c in enumerate(cal.CLAIMS):  # the grid rule as step 1 ran it, at the plain-Holm levels 0.05 / k
        for ti, t in enumerate(TESTS):
            null_p = [p[:, ci, ti] for p, nl in zip(p1, nulls) if nl[c]]
            members = (c, *OTHERS)
            got = mu.alpha_primes(null_p, c, members, dict.fromkeys(members, 1), cfg.alpha)
            want = {int(k): v for k, v in s1["alpha"][c][t]["alpha_prime"].items()}
            if got != want:
                raise SystemExit(f"the grid rule does not give step 1's alpha' for {c} {t}: {got} != {want}")
    families, record = mu.step_one_table(p1, nulls, cfg.holm_weights, cfg.alpha)
    inputs = {"fingerprint": fingerprint, "step1": {"file": "results/calibration-step1-20261019/step1.json",
                                                    "sha256": s1_sha},
              "calibrated_code_digest": calibrated, "n": cal.N1, "seed": cal.SEEDS[1], "seed_scheme": cal.SEED_SCHEME,
              "configurations": [{"name": conf.name, "null": [c for c in cal.CLAIMS if nl[c]]}
                                 for conf, nl in zip(confs, nulls)],
              "outputs": content,
              "producer": "harness.analysis.multiplicity.step_one_table (scripts/weighted_holm_tables.py alpha)"}
    sha = _write(args.out, mu.table_document(families, cfg.alpha, cfg.holm_weights, inputs, step1=record))
    for fam, r in record.items():
        tests = {c: e["test"] for c, e in families[fam].items()}
        print(f"{fam}: tests {tests}, factor {r['factor']}, step-1 FWER max {r['fwer_max']}")
    print(f"{args.out}\nsha256 {sha}")
    return 0


def cmd_step2(args) -> int:
    cfg = PreregConfig.load(args.config)
    digest = multiplicity_digest()
    if digest != cfg.step2_procedure_digest:  # step 2 ran as that code; the code has grown step 3 since (entry 106)
        raise SystemExit(f"the multiplicity code's digest {digest} is not step 2's frozen procedure "
                         f"{cfg.step2_procedure_digest}: step 2's procedure is bound at the freeze (entry 91)")
    raw = args.table.read_bytes()
    table_sha = hashlib.sha256(raw).hexdigest()
    if table_sha != cfg.multiplicity_table_sha256:
        raise SystemExit(f"{args.table}: sha256 {table_sha}, not the config's multiplicity_table_sha256")
    table = json.loads(raw)
    confs, fingerprint = table["inputs"]["configurations"], table["inputs"]["fingerprint"]
    out = args.calibration_out.resolve()
    result = mu.step_two(lambda i: mu.configuration_p(out, 2, i, confs[i]["name"], fingerprint, cal.N2, cal.SEEDS[2],
                                                      cal.SEED_SCHEME),
                         confs, table, cfg.holm_weights, cfg.alpha, cal.N2)
    inputs = {"n": cal.N2, "seed": cal.SEEDS[2], "seed_scheme": cal.SEED_SCHEME, "fingerprint": fingerprint,
              "configurations": [c["name"] for c in confs], "procedure": digest,
              "outputs": {"directory": str(out), **mu.outputs_digest(out, 2)},
              "producer": "harness.analysis.multiplicity.step_two (scripts/weighted_holm_tables.py step2)"}
    sha = _write(args.out, mu.addendum_document(table_sha, result, inputs, args.date, cfg.alpha, cfg.holm_weights))
    for fam, r in result.items():
        dropped = {c: len(levels) - len(r["usable"][c]) for c, levels in r["levels"].items() if levels}
        print(f"{fam}: FWER check {'passed' if r['fwer_pass'] else 'FAILED (every claim p = 1)'}; unusable levels "
              f"{dropped}")
    print(f"{args.out}\nsha256 {sha}")
    return 0


# --- step 3 (prereg §4.2, change-log entry 106) ----------------------------------------------------------------------

STEP2_ADDENDUM_SHA256 = "87dd9abaa9fe04181b95e968bf088a273309cd6c6fc332db2083c8ba14a9bc22"  # dated 2026-10-20


def _step3_frozen(args) -> tuple[PreregConfig, dict, dict, str, str]:
    """The config, the alpha' table and the step-2 addendum, each by its sha256, with this code the frozen procedure:
    step 3's procedure is bound before it runs (entry 106, as entry 91 bound step 2's)."""
    from harness.analysis.inference import inference_digest

    cfg = PreregConfig.load(args.config)
    if multiplicity_digest() != cfg.multiplicity_digest:
        raise SystemExit(f"the multiplicity code's digest {multiplicity_digest()} is not the config's frozen "
                         f"{cfg.multiplicity_digest}: step 3's procedure is bound before it runs")
    if inference_digest() != cfg.calibrated_inference_digest:
        raise SystemExit(f"the inference code's digest {inference_digest()} is not the config's calibrated "
                         f"{cfg.calibrated_inference_digest}")
    table_raw, usable_raw = args.table.read_bytes(), args.step2_addendum.read_bytes()
    if hashlib.sha256(table_raw).hexdigest() != cfg.multiplicity_table_sha256:
        raise SystemExit(f"{args.table}: not the config's multiplicity_table_sha256")
    usable_sha = hashlib.sha256(usable_raw).hexdigest()
    if usable_sha != args.step2_addendum_sha256:
        raise SystemExit(f"{args.step2_addendum}: sha256 {usable_sha}, not step 2's dated addendum {args.step2_addendum_sha256}")
    return cfg, json.loads(table_raw), json.loads(usable_raw), cfg.multiplicity_table_sha256, usable_sha


def bound_configurations(cfg: PreregConfig, computed: list[dict]) -> list[dict]:
    """The config's bound step-3 configurations (multiplicity.read_step3_configurations), refused unless this code's
    configurations and null claims are exactly them: a run is bound to them before it starts (the review of 1e31f1f)."""
    if PLACEHOLDER in (cfg.step3_configurations, cfg.step3_configurations_sha256):
        raise SystemExit("step 3's configurations are not bound (step3_configurations [FILL]): bound before it runs")
    try:
        bound = mu.read_step3_configurations(ROOT / cfg.step3_configurations, cfg.step3_configurations_sha256)
    except PreregError as exc:
        raise SystemExit(str(exc)) from None
    if computed != bound:
        raise SystemExit(f"this code's {len(computed)} step-3 configurations and null claims are not the bound "
                         f"{len(bound)} ({cfg.step3_configurations})")
    return bound


def _bound_fingerprint(cfg: PreregConfig, fp3: str) -> None:
    """This run's fingerprint against the config's step3_fingerprint once it is bound (plan3 prints it before
    submission); [FILL] until then, when the loader refuses the outputs instead."""
    if cfg.step3_fingerprint != PLACEHOLDER and fp3 != cfg.step3_fingerprint:
        raise SystemExit(f"this run's fingerprint {fp3} is not the bound step3_fingerprint {cfg.step3_fingerprint}")


def _setup3(args, cfg: PreregConfig):
    C = _calibrate()
    ns = argparse.Namespace(out=args.step3_out, bank=args.bank, k=args.k, questions=args.questions, size=1000,
                            per_template=args.per_template)
    _, _, confs3, nulls3, _ = C.setup3(ns)
    C._verify_structure(ns)
    confs = bound_configurations(cfg, _configurations3(confs3, nulls3))
    fp3 = C.fingerprint3(ns, confs3)
    _bound_fingerprint(cfg, fp3)
    return confs, fp3


def _configurations3(confs3, nulls3) -> list[dict]:
    return [{"name": c.name, "null": [k for k in cal.CLAIMS if nl[k]]} for c, nl in zip(confs3, nulls3)]


def _steps12_outputs(cfg: PreregConfig, usable: dict) -> dict:
    """Steps 1 and 2's frozen chunk content: the config's step1_outputs and the step-2 addendum's outputs."""
    return {"step1": cfg.step1_outputs, "step2": usable["inputs"]["outputs"]}


def cmd_step3_select(args) -> int:
    from harness.analysis.inference import inference_digest

    cfg, table, usable, table_sha, usable_sha = _step3_frozen(args)
    confs, fp3 = _setup3(args, cfg)
    out3, cal_out = args.step3_out.resolve(), args.calibration_out.resolve()
    frozen12 = _steps12_outputs(cfg, usable)
    found12 = {"step1": mu.outputs_digest(cal_out, 1), "step2": mu.outputs_digest(cal_out, 2)}
    if {k: {kk: v[kk] for kk in ("chunks", "sha256")} for k, v in frozen12.items()} != found12:
        raise SystemExit(f"{cal_out}: steps 1 and 2's chunks are {found12}, not their frozen content: refused before "
                         f"any chunk's p-values are used")
    n_sel, seed = cal.N3_SELECTION, cal.STEP3_SEEDS["selection"]
    try:
        c2_null_p, repair_null_p, pooled_from = mu.step3_selection_arrays(table, usable, confs, out3, cal_out, fp3, n_sel,
                                                                          seed, cal.SEED_SCHEME)
    except PreregError as exc:
        raise SystemExit(str(exc)) from None
    selection = mu.step3_select(c2_null_p, repair_null_p, table, usable, cfg.alpha)
    inputs = {"table_sha256": table_sha, "step2_addendum_sha256": usable_sha, "fingerprint": fp3,
              "steps12_fingerprint": table["inputs"]["fingerprint"], "n_selection": n_sel, "seed": seed,
              "seed_scheme": cal.SEED_SCHEME, "configurations": confs,
              "selection_configurations": [c["name"] for c in confs if "C2" in c["null"]], "repair_data": pooled_from,
              "outputs": {"directory": str(out3), **mu.outputs_digest(out3, "3-selection")},
              "steps12_directory": str(cal_out), "steps12_outputs": found12,
              "procedure": multiplicity_digest(), "inference": inference_digest(),
              "producer": "harness.analysis.multiplicity.step3_select (scripts/weighted_holm_tables.py step3-select)"}
    sha = _write(args.out, mu.step3_alpha_document(selection, inputs, cfg.alpha, cfg.holm_weights))
    for fam, f in selection.items():
        print(f"{fam}: C2 test {f['C2']['test']}, alpha' {f['C2']['alpha_prime']}; repairs "
              f"{ {c: {s: (r['alpha_prime'], r['usable']) for s, r in v['levels'].items()} for c, v in f['repairs'].items()} }")
    print(f"{args.out}\nsha256 {sha}")
    return 0


def _checked_selection(args, cfg: PreregConfig, table: dict, usable: dict, table_sha: str, usable_sha: str,
                       confs: list[dict], fp3: str, require_recomputed: bool) -> tuple[dict, str, dict]:
    """The selection file, refused unless the frozen procedure produced it from its frozen inputs
    (multiplicity.check_selection; the review of 1e31f1f: its schema and fingerprint alone bind no threshold)."""
    from harness.analysis.inference import inference_digest

    sel_raw = args.selection.read_bytes()
    selection, sel_sha = json.loads(sel_raw), hashlib.sha256(sel_raw).hexdigest()
    try:
        report = mu.check_selection(selection, table, usable, configurations=confs, fingerprint=fp3,
                                    table_sha256=table_sha, usable_sha256=usable_sha, n=cal.N3_SELECTION,
                                    seed=cal.STEP3_SEEDS["selection"], seed_scheme=cal.SEED_SCHEME,
                                    procedure=multiplicity_digest(), inference=inference_digest(), alpha=cfg.alpha,
                                    weights=cfg.holm_weights, steps12_outputs=_steps12_outputs(cfg, usable),
                                    selection_sha256=sel_sha, require_recomputed=require_recomputed)
    except PreregError as exc:
        raise SystemExit(f"{args.selection}: {exc}") from None
    return selection, sel_sha, report


def cmd_step3(args) -> int:
    from harness.analysis.inference import inference_digest

    cfg, table, usable, table_sha, usable_sha = _step3_frozen(args)
    confs, fp3 = _setup3(args, cfg)
    selection, sel_sha, checked = _checked_selection(args, cfg, table, usable, table_sha, usable_sha, confs, fp3, True)
    n3, seed = cal.N3, cal.STEP3_SEEDS["validation"]
    out3 = args.step3_out.resolve()
    result = mu.step_three(lambda i: mu.configuration_p(out3, 3, i, confs[i]["name"], fp3, n3, seed, cal.SEED_SCHEME,
                                                        phase="validation"),
                           confs, table, usable, selection, cfg.alpha, n3)
    inputs = {"selection_sha256": sel_sha, "table_sha256": table_sha,
              "step2_addendum_sha256": usable_sha, "n": n3, "seed": seed, "seed_scheme": cal.SEED_SCHEME,
              "fingerprint": fp3, "configurations": confs, "procedure": multiplicity_digest(),
              "inference": inference_digest(), "outputs": {"directory": str(out3), **mu.outputs_digest(out3, "3-validation")},
              "producer": "harness.analysis.multiplicity.step_three (scripts/weighted_holm_tables.py step3)"}
    sha = _write(args.out, mu.step3_document(result, inputs, args.date, cfg.alpha, cfg.holm_weights))
    print(f"the selection: {checked['checked']}")
    for fam, r in result.items():
        print(f"{fam}: FWER check {'passed' if r['fwer_pass'] else 'FAILED (every claim p = 1)'}; C2 usable "
              f"{r['usable']['C2']}; verified { {c: {s: v['usable'] for s, v in lv.items()} for c, lv in r['levels'].items()} }")
    print(f"{args.out}\nselection sha256 {sel_sha}\nsha256 {sha}")
    return 0


def cmd_step3_verify(args) -> int:
    """Where the chunks are (SiteA): the selection and the addendum recomputed from them (check_selection and check_step3
    with require_recomputed), then the freeze registry's two records of those recomputations printed as JSON
    (step3_selection_outputs and step3_outputs), for a host without the chunks."""
    from harness.analysis.inference import inference_digest

    cfg, table, usable, table_sha, usable_sha = _step3_frozen(args)
    confs, fp3 = _setup3(args, cfg)
    selection, sel_sha, checked = _checked_selection(args, cfg, table, usable, table_sha, usable_sha, confs, fp3, True)
    raw = args.addendum.read_bytes()
    step3, step3_sha = json.loads(raw), hashlib.sha256(raw).hexdigest()
    try:
        report = mu.check_step3(step3, table, usable, selection, configurations=confs, fingerprint=fp3, n=cal.N3,
                                seed=cal.STEP3_SEEDS["validation"], seed_scheme=cal.SEED_SCHEME,
                                procedure=multiplicity_digest(), inference=inference_digest(), alpha=cfg.alpha,
                                weights=cfg.holm_weights, step3_sha256=step3_sha, selection_sha256=sel_sha)
    except PreregError as exc:
        raise SystemExit(f"{args.addendum}: {exc}") from None
    if not (checked["recomputed"] and report["recomputed"]):
        raise SystemExit("the validation chunks are not on this host: nothing recomputed, nothing to record")
    rec_sel, rec3 = selection["inputs"], step3["inputs"]["outputs"]
    records = {"step3_selection_outputs": [{
        "selection_sha256": sel_sha, "procedure": multiplicity_digest(),
        "outputs": {"selection": {k: rec_sel["outputs"][k] for k in ("chunks", "sha256")},
                    **{k: {kk: rec_sel["steps12_outputs"][k][kk] for kk in ("chunks", "sha256")}
                       for k in ("step1", "step2")}}}],
        "step3_outputs": [{"addendum_sha256": step3_sha, "procedure": multiplicity_digest(),
                           "outputs": {k: rec3[k] for k in ("chunks", "sha256")}}]}
    print(f"the selection: {checked['checked']}\nthe addendum: {report['checked']}")
    print(json.dumps(records, indent=1, sort_keys=True))
    return 0


def cmd_step2_verify(args) -> int:
    """Where step 2's chunks are (SiteA): the step-2 addendum recomputed from them (multiplicity.check_addendum under step
    2's own procedure, prereg.STEP2_PROCEDURE_DIGEST), then the freeze registry's step2_outputs record of that
    recomputation printed as JSON, for a host without the chunks."""
    cfg = PreregConfig.load(args.config)
    table_raw, raw = args.table.read_bytes(), args.step2_addendum.read_bytes()
    if hashlib.sha256(table_raw).hexdigest() != cfg.multiplicity_table_sha256:
        raise SystemExit(f"{args.table}: not the config's multiplicity_table_sha256")
    sha = hashlib.sha256(raw).hexdigest()
    if sha != args.step2_addendum_sha256:
        raise SystemExit(f"{args.step2_addendum}: sha256 {sha}, not step 2's dated addendum {args.step2_addendum_sha256}")
    addendum = json.loads(raw)
    try:
        report = mu.check_addendum(addendum, json.loads(table_raw), n=cal.N2, seed=cal.SEEDS[2],
                                   seed_scheme=cal.SEED_SCHEME, procedure=cfg.step2_procedure_digest, alpha=cfg.alpha,
                                   weights=cfg.holm_weights, addendum_sha256=sha)
    except PreregError as exc:
        raise SystemExit(f"{args.step2_addendum}: {exc}") from None
    if not report["recomputed"]:
        raise SystemExit("step 2's chunks are not on this host: nothing recomputed, nothing to record")
    out = addendum["inputs"]["outputs"]
    print(f"the step-2 addendum: {report['checked']}")
    print(json.dumps({"step2_outputs": [{"addendum_sha256": sha, "procedure": cfg.step2_procedure_digest,
                                         "outputs": {k: out[k] for k in ("chunks", "sha256")}}]}, indent=1,
                     sort_keys=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("alpha", "step2", "step3-select", "step3", "step3-verify", "step2-verify"):
        p = sub.add_parser(name)
        verify = name in ("step3-verify", "step2-verify")
        p.add_argument("--calibration-out", type=Path, required=name != "step2-verify",
                       help="calibrate.py's OUT (step1/, step2/ chunks)")
        p.add_argument("--config", type=Path, default=CONFIG)
        p.add_argument("--out", type=Path, required=not verify)
        if name == "step2-verify":
            p.add_argument("--table", type=Path, default=ROOT / "data" / "analysis" / "weighted-holm-alpha.json")
            p.add_argument("--step2-addendum", type=Path, default=ROOT / "data" / "analysis" / "weighted-holm-step2.json")
            p.add_argument("--step2-addendum-sha256", default=STEP2_ADDENDUM_SHA256)
        elif name == "alpha":
            p.add_argument("--bank", type=Path, required=True, help="the calibration's bank (the fingerprint binds it)")
            p.add_argument("--calibrated-tree", type=Path, required=True, help="a checkout of the calibrated commit 87c7126")
            p.add_argument("--questions", type=Path, default=ROOT / "data" / "questions" / "uniprot-sp-2026_03" /
                           "questions.jsonl")
            p.add_argument("--k", type=int, default=3)
            p.add_argument("--per-template", type=int, default=180)
        elif name == "step2":
            p.add_argument("--table", type=Path, default=ROOT / "data" / "analysis" / "weighted-holm-alpha.json")
            p.add_argument("--date", default=datetime.datetime.now(datetime.UTC).date().isoformat())
        else:
            p.add_argument("--step3-out", type=Path, required=True, help="calibrate.py's step-3 OUT (step3-<phase>/)")
            p.add_argument("--bank", type=Path, required=True)
            p.add_argument("--questions", type=Path, default=ROOT / "data" / "questions" / "uniprot-sp-2026_03" /
                           "questions.jsonl")
            p.add_argument("--k", type=int, default=3)
            p.add_argument("--per-template", type=int, default=180)
            p.add_argument("--table", type=Path, default=ROOT / "data" / "analysis" / "weighted-holm-alpha.json")
            p.add_argument("--step2-addendum", type=Path, default=ROOT / "data" / "analysis" / "weighted-holm-step2.json")
            p.add_argument("--step2-addendum-sha256", default=STEP2_ADDENDUM_SHA256)
            if name in ("step3", "step3-verify"):
                p.add_argument("--selection", type=Path,
                               default=ROOT / "data" / "analysis" / "weighted-holm-step3-alpha.json")
            if name == "step3":
                p.add_argument("--date", default=datetime.datetime.now(datetime.UTC).date().isoformat())
            if name == "step3-verify":
                p.add_argument("--addendum", type=Path, default=ROOT / "data" / "analysis" / "weighted-holm-step3.json")
    args = ap.parse_args(argv)
    return {"alpha": cmd_alpha, "step2": cmd_step2, "step3-select": cmd_step3_select, "step3": cmd_step3,
            "step3-verify": cmd_step3_verify, "step2-verify": cmd_step2_verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
