#!/usr/bin/env python3
"""Power under the registered weighted Holm (prereg §4.4, design log) on the final bank structure, with the final
calibration of both candidate families (multiplicity.load_final: step 1's alpha' table, step 2's addendum and step 3's
selection and addendum, C2 two-sided; entry 106) (harness/analysis/weighted_power.py).

Subcommands (OUT is a fixed results directory):
- ``dev-rates --grid-out G --out FILE``: C1's dev rates (per model of P and prompted arm) from a dev-grid OUT, read by
  the analysis's own loaders (inputs.load_batch with the grid's cells), under the frozen scorer (scorer-rules-v6 since
  change-log entry 127; v5: entries 120 and 121): the dev OUT runs on v4 to the end (science, design log, 04:12), so each
  v4- or v5-scored record of the C1 arms is re-scored under the frozen rules in memory first (``rescore_frozen``), and
  the file records it.
- ``plan --out OUT [--dev-rates FILE] [--c5-inputs FILE]``: the cells and their scenarios, written to OUT/plan.json
  with the run's fingerprint (code, bank items, structure, settings). The bank cells are always planned; the C1 cells
  (margin and zero) and the C5 cell only with their inputs, else listed as pending. A plan never changes a cell some
  chunk was simulated for (a new plan with other scenarios needs a new OUT).
- ``simulate --out OUT --cell I --chunk J``: datasets [J*size, (J+1)*size) of cell I by calibration.dataset_p.
- ``run --out OUT --jobs K [--reuse DIR] [--no-aggregate]``: every missing chunk of the plan, K processes; a chunk of
  DIR simulated for the same fingerprint, cell, scenario, seed and datasets is copied instead of recomputed; then the
  aggregation unless ``--no-aggregate``.
- ``aggregate --out OUT [--config C]``: OUT/power.json and OUT/power.md on the registered calibration only
  (``registered_calibrations``: the final one once step 3's addendum is bound, else step 3's selection levels with the
  validation pending, labelled so, change-log entry 131; never the one-sided one, the review of 1e31f1f): per cell and
  family the claims' power as simulated and bracketed over the stand-in claims; the per-claim table; and, when the C1
  cell is there, the two C1a powers ``choose-c1a`` takes (the weighted-Holm rejection powers at the 3-point margin, the
  first-level powers beside them). The config may be a later one than the plan's that binds step 3
  (``calibration_config``). While no config binds step 3's selection it writes nothing and exits 3 (STEP3_PENDING).
- ``replay --draws DIR --out FILE``: the earlier power job's stored draws (results/power-holm-20261019, job 17559:
  the provisional structure, 2,000 datasets per cell, simulated with the one-sided C2) on the historical one-sided
  calibration (step 1's table and step 2's addendum, ``historical_calibrations``), labelled so; no C1a choice.

Heavy (about 0.4 s per dataset): run it on run-node under nice -n 19 with at most 4 processes, never on the agents
VM.

    nice -n 19 .venv/bin/python scripts/power_weighted.py plan --out OUT
    nice -n 19 .venv/bin/python scripts/power_weighted.py run --out OUT --jobs 4
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import dataclasses
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.analysis import calibration as cal
from harness.analysis import multiplicity as mu
from harness.analysis import prereg as pr
from harness.analysis import weighted_power as wp
from harness.analysis.digest import code_digest
from harness.analysis.simulate import truths

SCHEMA = "power-weighted/1"
CONFIG = ROOT / "data" / "analysis" / "prereg-config.json"
BANK = ROOT / "data" / "banks" / "uniprot-sp-2026_03"  # the frozen UniProt bank (manifest bd17577e)
ADDENDUM = ROOT / "data" / "analysis" / "weighted-holm-step2.json"
ADDENDUM_SHA256 = "87dd9abaa9fe04181b95e968bf088a273309cd6c6fc332db2083c8ba14a9bc22"  # the dated step-2 addendum
SEED = 20261120  # not a registered seed (20261026, 20261101, 20261102) nor an earlier power study's (20261119)
SEED_SCHEME = "numpy default_rng([SEED, cell index, dataset index]) via calibration.dataset_p"
N = 1000  # datasets per cell (Monte Carlo SE at most 1.6 points)
SIZE = 100  # datasets per chunk
C1_MARGIN = 0.03


def _calibrate():
    spec = importlib.util.spec_from_file_location("calibrate", ROOT / "scripts" / "calibrate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def setup(out: Path):
    """calibrate.setup on the frozen bank (its own 1,260 test questions: the real-structure path, no resampling), the
    items written to OUT/bank/items.jsonl and checked against the bank manifest's sha256 first."""
    manifest = json.loads((BANK / "manifest.json").read_text(encoding="utf-8"))
    raw = gzip.decompress((BANK / "items.jsonl.gz").read_bytes())
    if _sha(raw) != manifest["files"]["items.jsonl"]:
        raise SystemExit(f"{BANK}: items.jsonl.gz is not the manifest's items ({manifest['files']['items.jsonl']})")
    (out / "bank").mkdir(parents=True, exist_ok=True)
    items = out / "bank" / "items.jsonl"
    if not items.is_file() or _sha(items.read_bytes()) != _sha(raw):
        items.write_bytes(raw)
    C = _calibrate()
    args = argparse.Namespace(out=out, bank=out / "bank", k=3, questions=C.QUESTIONS, size=1000, per_template=None)
    cfg, index, confs, K_bank = C.setup(args)
    structure = confs[0].scenario.bank.structure
    from harness.analysis.inference import inference_digest

    # the inference code the draws are simulated with (C2 two-sided since entry 106): aggregate refuses draws of
    # another one as registered powers (the review of 3b149eb)
    meta = {"code": code_digest(), "inference": inference_digest(), "bank_manifest": _sha((BANK / "manifest.json").read_bytes()),
            "bank_items": _sha(raw), "questions": _sha(Path(C.QUESTIONS).read_bytes()),
            "structure": _sha(args.structure_text.encode()), "structure_source": json.loads(args.structure_text)["source"],
            "n_bank_questions": len(structure.type_wrong), "n_boot": cfg.n_boot, "resample_seed": cfg.seed,
            "seed": SEED, "seed_scheme": SEED_SCHEME}
    return cfg, index, K_bank, structure, meta


def cells(structure, dev_rates: dict | None, c5_inputs: dict | None, panel: list[str],
          agent_only: bool = False) -> list[wp.Cell]:
    """Every cell in a fixed order (its index seeds its datasets): the bank grid, then C1 at the margin, C1 at zero and
    C5, each simulated only with its inputs, then the two dev bank cells (weighted_power.dev_bank_cells). An agent cell
    studies its own claims; the others are stand-ins there. ``agent_only`` leaves the bank grid and the dev bank cells
    unsimulated (they are another OUT's), keeping every index."""
    bank = wp.reference_bank(structure)
    out = list(wp.bank_cells(structure))
    if agent_only:
        out = [wp.Cell(c.name, None, c.studied, "not simulated here (agent cells only): see the bank OUT")
               for c in out]
    waiting = "pending: the dev grid's prompted phase (C1's dev rates)"
    out.append(wp.Cell("C1 dev rates, margin 0.03", wp.c1_scenario(dev_rates, bank, C1_MARGIN) if dev_rates else None,
                       ("C1a", "C1a pooled"), "C1a at the 3-point margin" if dev_rates else waiting))
    out.append(wp.Cell("C1 dev rates, difference 0", wp.c1_scenario(dev_rates, bank, 0.0) if dev_rates else None,
                       ("C1b",), "C1b with every curated - census difference 0" if dev_rates else waiting))
    out.append(wp.Cell("C5 entry-77 dev counts", wp.c5_scenario(c5_inputs, panel, bank) if c5_inputs else None, ("C5",),
                       "C5 at 10 points per flagged output" if c5_inputs else "pending: the entry-77 replay's counts"))
    # the dev bank's rates (science, design log, 21 Oct 10:54), after every earlier cell so each keeps its index
    for c in wp.dev_bank_cells(structure):
        out.append(wp.Cell(c.name, None, c.studied, "not simulated here (agent cells only): see the bank OUT")
                   if agent_only else c)
    return out


def _plan_doc(out: Path, args) -> dict[str, Any]:
    cfg_file = pr.PreregConfig.load(args.config)
    sim_cfg, _, _, structure, meta = setup(out)
    dev = json.loads(Path(args.dev_rates).read_text(encoding="utf-8")) if args.dev_rates else None
    if dev is not None and (dev.get("schema") != wp.DEV_RATES_SCHEMA or dev.get("split") != "dev"
                            or set(dev.get("models") or {}) != set(cfg_file.panel)):
        raise SystemExit(f"{args.dev_rates}: not {wp.DEV_RATES_SCHEMA} dev rates for P {list(cfg_file.panel)}")
    c5 = wp.read_c5_inputs(Path(args.c5_inputs), list(cfg_file.panel), cfg_file.graph) if args.c5_inputs else None
    rows = []
    for i, cell in enumerate(cells(structure, dev, c5, list(cfg_file.panel), args.agent_only)):
        row = {"index": i, "name": cell.name, "studied": list(cell.studied), "note": cell.note,
               "simulated": cell.scenario is not None}
        if cell.scenario is not None:
            n = len(structure.type_wrong)
            t = truths(cell.scenario, sim_cfg, n)
            # C2's truth by its registered two-sided claim (entry 106), as the draws' statistic is
            row.update(scenario=repr(cell.scenario), nulls=cal.nulls(cell.scenario, sim_cfg, n, two_sided_c2=True),
                       truths={k: v for k, v in t.items() if k != "C5 per model"})
        rows.append(row)
    inputs = {"config": {"file": str(Path(args.config).resolve()), "sha256": _sha(Path(args.config).read_bytes())},
              "dev_rates": None if dev is None else {"file": str(args.dev_rates),
                                                      "sha256": _sha(Path(args.dev_rates).read_bytes())},
              "c5_inputs": None if c5 is None else {"file": str(args.c5_inputs),
                                                     "sha256": _sha(Path(args.c5_inputs).read_bytes())}}
    return {"schema": SCHEMA, "fingerprint": meta, "n": args.n, "size": args.size, "inputs": inputs,
            "agent_only": bool(args.agent_only), "cells": rows}


def cmd_plan(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = _plan_doc(out, args)
    path = out / "plan.json"
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        changed = [r["name"] for r, o in zip(doc["cells"], old["cells"])
                   if o.get("simulated") and (r.get("scenario") != o.get("scenario"))]
        if old["fingerprint"] != doc["fingerprint"] or old["n"] != doc["n"] or old["size"] != doc["size"] or (
                changed and any((out / "chunks").glob("cell*.npz"))):
            raise SystemExit(f"{path}: another plan was simulated here ({changed or 'fingerprint or n'}); use a new OUT")
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    pending = [c["name"] for c in doc["cells"] if not c["simulated"]]
    print(f"{path}: {sum(c['simulated'] for c in doc['cells'])} cells to simulate, pending {pending}")
    return 0


def _load_plan(out: Path) -> dict[str, Any]:
    return json.loads((out / "plan.json").read_text(encoding="utf-8"))


def _plan_config(plan: dict, given: Path | None = None) -> Path:
    """The config the plan was made with (its P is the simulated panel), refused when it changed since, or when another
    config is given; a plan from before the config was recorded uses the given one, else the repository's."""
    rec = (plan.get("inputs") or {}).get("config")
    if rec is None:
        return Path(given) if given else CONFIG
    if _sha(Path(rec["file"]).read_bytes()) != rec["sha256"]:
        raise SystemExit(f"{rec['file']}: changed since the plan (sha256 {rec['sha256']})")
    if given is not None and _sha(Path(given).read_bytes()) != rec["sha256"]:
        raise SystemExit(f"{given}: not the config the plan was made with ({rec['file']})")
    return Path(rec["file"])


def _chunk_path(out: Path, cell: int, chunk: int) -> Path:
    return out / "chunks" / f"cell{cell:02d}-{chunk:04d}.npz"


def simulate_chunk(out: Path, cell: int, chunk: int) -> str:
    """One chunk of one cell; the plan is recomputed and must be the stored one (code, inputs and scenario)."""
    plan = _load_plan(out)
    config = _plan_config(plan)
    args = argparse.Namespace(config=config, n=plan["n"], size=plan["size"],
                              dev_rates=(plan["inputs"]["dev_rates"] or {}).get("file"),
                              c5_inputs=(plan["inputs"]["c5_inputs"] or {}).get("file"))
    agent_only = bool(plan.get("agent_only"))
    sim_cfg, index, K_bank, structure, meta = setup(out)
    if meta != plan["fingerprint"]:
        raise SystemExit(f"{out}/plan.json was planned under another fingerprint")
    pcfg = pr.PreregConfig.load(config)
    dev = json.loads(Path(args.dev_rates).read_text(encoding="utf-8")) if args.dev_rates else None
    c5 = wp.read_c5_inputs(Path(args.c5_inputs), list(pcfg.panel), pcfg.graph) if args.c5_inputs else None
    sc = cells(structure, dev, c5, list(pcfg.panel), agent_only)[cell].scenario
    if sc is None or repr(sc) != plan["cells"][cell].get("scenario"):
        raise SystemExit(f"cell {cell}: not simulated in this plan, or its scenario changed since the plan")
    K = pr.resamples(index, sim_cfg)
    lo, hi = chunk * plan["size"], min((chunk + 1) * plan["size"], plan["n"])
    p = np.stack([cal.dataset_p(sc, index, K, sim_cfg, SEED, cell, i, K_bank) for i in range(lo, hi)])
    path = _chunk_path(out, cell, chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, p=p.astype(np.float64), lo=lo, hi=hi, cell=cell, scenario=repr(sc),
                        fingerprint=json.dumps(meta, sort_keys=True), seed=SEED, seed_scheme=SEED_SCHEME)
    tmp.replace(path)
    return f"cell {cell} datasets {lo}..{hi}"


def cmd_simulate(args) -> int:
    print(simulate_chunk(Path(args.out), args.cell, args.chunk))
    return 0


def _same_chunk(a: Path, plan: dict, cell: int, chunk: int) -> bool:
    try:
        z = np.load(a)
    except (OSError, ValueError):
        return False
    lo, hi = chunk * plan["size"], min((chunk + 1) * plan["size"], plan["n"])
    return (str(z["fingerprint"]) == json.dumps(plan["fingerprint"], sort_keys=True) and int(z["cell"]) == cell
            and str(z["scenario"]) == plan["cells"][cell]["scenario"] and int(z["seed"]) == SEED
            and str(z["seed_scheme"]) == SEED_SCHEME and int(z["lo"]) == lo and int(z["hi"]) == hi)


def missing_chunks(out: Path, plan: dict) -> list[tuple[int, int]]:
    per = -(-plan["n"] // plan["size"])
    return [(c["index"], j) for c in plan["cells"] if c["simulated"] for j in range(per)
            if not _chunk_path(out, c["index"], j).is_file()]


def cmd_run(args) -> int:
    out = Path(args.out)
    plan = _load_plan(out)
    todo = missing_chunks(out, plan)
    if args.reuse:
        for cell, chunk in list(todo):
            src = _chunk_path(Path(args.reuse), cell, chunk)
            if src.is_file() and _same_chunk(src, plan, cell, chunk):
                _chunk_path(out, cell, chunk).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, _chunk_path(out, cell, chunk))
                todo.remove((cell, chunk))
    print(f"{len(todo)} chunks to simulate with {args.jobs} processes", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for msg in pool.map(simulate_chunk, [out] * len(todo), [c for c, _ in todo], [j for _, j in todo]):
            print(msg, flush=True)
    return 0 if args.no_aggregate else cmd_aggregate(args)


STEP3_PENDING = 3  # aggregate's exit code while the config does not bind step 3: the registered powers wait for it
# the config settings that bind the calibration (and its code), not the simulation: a later config that binds step 3
# may differ from the plan's in these only
CALIBRATION_BINDINGS = ("note", "step2_addendum", "step2_addendum_sha256", "freeze_registry", "step3_configurations",
                        "step3_configurations_sha256", "step3_fingerprint", "step3_selection", "step3_selection_sha256",
                        "step3_addendum", "step3_addendum_sha256", "calibrated_inference_digest", "grouping_digest",
                        "multiplicity_digest", "calibrated_kat_sha256", "grouping_kat_sha256", "multiplicity_kat_sha256",
                        "kat_hosts", "kat_hosts_sha256")


def final_calibrations(cfg: pr.PreregConfig) -> dict[str, mu.WeightedCalibration]:
    """Both candidate families' final calibrations (multiplicity.load_final: step 2's addendum by its own procedure,
    step 3's bound configurations and fingerprint, its selection re-derived and its addendum checked): the calibration
    the registered powers and the C1a choice are computed with once step 3's addendum is bound (entry 106; the review of
    1e31f1f: C2 is two-sided, its levels and the repaired ones are step 3's); before it, step 3's selection levels
    (selection_calibrations, entry 131; registered_calibrations picks). PreregError while the config does not bind
    step 3."""
    finals = mu.load_final(cfg, ROOT)
    if set(finals) != set(pr.FAMILY_CLAIMS):
        raise pr.PreregError(f"the final calibration holds {sorted(finals)}, not both candidate families")
    return finals


SELECTION_STAGE = "calibration: step 3 selection levels; validation pending"  # change-log entry 131


def _read_bound(path: Path, sha256: str, what: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if _sha(raw) != sha256:
        raise pr.PreregError(f"{path}: sha256 {_sha(raw)} is not the bound {what}'s {sha256}")
    return json.loads(raw)


def selection_calibrations(cfg: pr.PreregConfig) -> dict[str, mu.WeightedCalibration]:
    """Change-log entry 131: before step 3's addendum is bound, the C1a choice and the pre-freeze powers take step 3's
    selection levels (C2 two-sided at its selected levels with its selected test, the usable repaired levels) and step
    2's for every other level (multiplicity.step3_candidates); validation can only remove levels. Verified as a
    confirmatory run verifies them, short of the addendum: step 2's addendum by its own procedure (step2_report), the
    bound configurations and fingerprint, and the selection by its sha256 re-derived from its inputs (check_selection:
    from its chunks, or the freeze record of that recomputation). Labelled SELECTION_STAGE; only this script's
    aggregation (the registered powers, the C1a choice) reads it, and a confirmatory analysis still refuses until the
    addendum is bound (multiplicity.load). The multiplicity code is unchanged (its digest binds step 3's records)."""
    step2 = mu.step2_report(cfg, ROOT)
    if any(v in (pr.PLACEHOLDER, "", None) for v in (cfg.step3_configurations, cfg.step3_configurations_sha256,
                                                      cfg.step3_fingerprint, cfg.step3_selection,
                                                      cfg.step3_selection_sha256)):
        raise pr.PreregError("step 3's configurations, fingerprint or selection are not bound (step3_selection and its "
                             "sha256 [FILL]): the registered powers and the C1a choice wait for the selection (entry 131)")
    table_path, usable_path = ROOT / cfg.multiplicity_table, ROOT / cfg.step2_addendum
    table = _read_bound(table_path, cfg.multiplicity_table_sha256, "alpha' table")
    usable = _read_bound(usable_path, cfg.step2_addendum_sha256, "step-2 addendum")
    sel_path = ROOT / cfg.step3_selection
    selection = _read_bound(sel_path, cfg.step3_selection_sha256, "step-3 selection")
    configurations = mu.read_step3_configurations(ROOT / cfg.step3_configurations, cfg.step3_configurations_sha256)
    checked = mu.check_selection(selection, table, usable, configurations=configurations,
                                 fingerprint=cfg.step3_fingerprint, table_sha256=cfg.multiplicity_table_sha256,
                                 usable_sha256=cfg.step2_addendum_sha256, n=cal.N3_SELECTION,
                                 seed=pr.STEP3_SEEDS["selection"], seed_scheme=cal.SEED_SCHEME,
                                 procedure=cfg.multiplicity_digest, inference=cfg.calibrated_inference_digest,
                                 alpha=cfg.alpha, weights=cfg.holm_weights,
                                 steps12_outputs={"step1": cfg.step1_outputs, "step2": usable["inputs"]["outputs"]},
                                 verified=mu.freeze_records(ROOT / cfg.freeze_registry, "step3_selection_outputs"),
                                 selection_sha256=cfg.step3_selection_sha256)
    out = {}
    for fam in pr.FAMILY_CLAIMS:
        base = mu.load_files(table_path, cfg.multiplicity_table_sha256, usable_path, cfg.step2_addendum_sha256, fam,
                             cfg.alpha, cfg.holm_weights)
        tests, alphas, _ = mu.step3_candidates(table, usable, selection, fam)
        claims = {c: {"test": tests[c], "alpha_prime": {str(s): a for s, a in sorted(alphas[c].items())}}
                  for c in pr.FAMILY_CLAIMS[fam]}
        out[fam] = mu.WeightedCalibration(
            fam, claims, dict(cfg.holm_weights), cfg.alpha, base.fwer_pass,
            f"{SELECTION_STAGE} (entry 131): step 3's selection {sel_path} (sha256 {cfg.step3_selection_sha256}) over "
            f"{base.source}", step2=step2,
            step3={"stage": SELECTION_STAGE, "selection": f"{sel_path} (sha256 {cfg.step3_selection_sha256})",
                   "selection_checked": checked})
    return out


def registered_calibrations(cfg: pr.PreregConfig) -> tuple[dict[str, mu.WeightedCalibration], dict[str, Any]]:
    """The calibration the registered powers and the C1a choice take, with its record: the final one once step 3's
    addendum is bound (final_calibrations, never a fallback when it is refused), else step 3's selection levels
    (selection_calibrations, entry 131). PreregError while neither is bound."""
    if pr.PLACEHOLDER in (cfg.step3_addendum, cfg.step3_addendum_sha256):
        return selection_calibrations(cfg), calibration_record(cfg, "selection")
    return final_calibrations(cfg), calibration_record(cfg, True)


def historical_calibrations(cfg: pr.PreregConfig, addendum: Path, addendum_sha256: str) -> dict[str, mu.WeightedCalibration]:
    """Step 1's alpha' table and the step-2 addendum, each by its sha256: the one-sided C2's calibration, for replaying
    draws simulated with the one-sided statistic (job 17559's) only, never for a registered power or the C1a choice."""
    return {fam: mu.load_files(ROOT / cfg.multiplicity_table, cfg.multiplicity_table_sha256, addendum, addendum_sha256,
                               fam, cfg.alpha, cfg.holm_weights) for fam in pr.FAMILY_CLAIMS}


def calibration_config(plan_config: Path, given: Path) -> pr.PreregConfig:
    """The config whose final calibration the aggregation reads: ``given``, refused unless it equals the plan's config
    (whose P is the simulated panel) in every setting but the calibration's bindings (CALIBRATION_BINDINGS), so a config
    that binds step 3 after the simulation aggregates it."""
    planned, cfg = pr.PreregConfig.load(plan_config), pr.PreregConfig.load(given)
    a, b = planned.to_json(), cfg.to_json()
    differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and k not in CALIBRATION_BINDINGS)
    if differ:
        raise SystemExit(f"{given}: differs from the plan's config {plan_config} in {differ}, not only in the "
                         f"calibration's bindings")
    return cfg


def calibration_record(cfg: pr.PreregConfig, final: bool | str, addendum_sha256: str | None = None) -> dict[str, Any]:
    """Which calibration a power result was aggregated with: the final one (step 3 by its bound files; ``final`` True),
    step 3's selection levels with the validation pending (``final`` "selection", entry 131), or the historical
    one-sided one (step 1's table and step 2's addendum)."""
    if final == "selection":
        return {"final": False, "stage": SELECTION_STAGE, "statistic": "C2 two-sided (entry 106)",
                "alpha_table_sha256": cfg.multiplicity_table_sha256, "step2_addendum_sha256": cfg.step2_addendum_sha256,
                "step3_selection_sha256": cfg.step3_selection_sha256, "step3_addendum_sha256": None,
                "step3_fingerprint": cfg.step3_fingerprint}
    if final is True:
        return {"final": True, "stage": "final calibration", "statistic": "C2 two-sided (entry 106)",
                "alpha_table_sha256": cfg.multiplicity_table_sha256,
                "step2_addendum_sha256": cfg.step2_addendum_sha256, "step3_selection_sha256": cfg.step3_selection_sha256,
                "step3_addendum_sha256": cfg.step3_addendum_sha256, "step3_fingerprint": cfg.step3_fingerprint}
    return {"final": False, "statistic": "C2 one-sided (before entry 106): the statistic the replayed draws were "
                                         "simulated with", "alpha_table_sha256": cfg.multiplicity_table_sha256,
            "step2_addendum_sha256": addendum_sha256}


def _load_cell(out: Path, plan: dict, cell: int) -> np.ndarray:
    per = -(-plan["n"] // plan["size"])
    parts = []
    for j in range(per):
        path = _chunk_path(out, cell, j)
        if not _same_chunk(path, plan, cell, j):
            raise SystemExit(f"{path}: missing, or not this plan's chunk")
        parts.append(np.load(path)["p"])
    return np.concatenate(parts)


def summarise(cell_rows: list[dict]) -> dict[str, Any]:
    """Per family and claim: over the cells that study it (C4: the B cells, where it is at its SESOI; never the dev bank
    cells, a column of their own in the per-cell table), the range of its power as simulated, with the stand-ins
    rejected and not, and of its first-level power."""
    out: dict[str, Any] = {}
    for fam, members in pr.FAMILY_CLAIMS.items():
        out[fam] = {}
        for c in members:
            rows = [r for r in cell_rows if r.get("families") and c in r["studied"]
                    and (c != "C4" or r["name"].startswith("B ")) and r["name"] not in wp.DEV_BANK_CELLS]
            if not rows:
                out[fam][c] = None
                continue
            fams = [r["families"][fam] for r in rows]

            def span(key: str, what: str, fams=fams, c=c) -> list[float] | None:
                vals = [f[key][c][what]["rate"] for f in fams if key in f and what in f[key][c]]
                return [min(vals), max(vals)] if vals else None

            out[fam][c] = {"cells": [r["name"] for r in rows],
                           "as simulated": span("as simulated", "power"),
                           "others rejected": span("others rejected", "power"),
                           "only C5 of the others rejected": span("only C5 of the others rejected", "power"),
                           "others not rejected": span("others not rejected", "power"),
                           "first level": span("as simulated", "first_level")}
    return out


def aggregate_arrays(arrays: dict[int, np.ndarray], plan_cells: list[dict], ws: dict) -> list[dict]:
    rows = []
    for c in plan_cells:
        row = {k: c.get(k) for k in ("index", "name", "note", "studied", "nulls", "truths")}
        if c["index"] in arrays:
            p = arrays[c["index"]]
            row["n"] = len(p)
            row["families"] = {fam: wp.bracketed(p, wc, members := pr.FAMILY_CLAIMS[fam],
                                                 [m for m in members if m not in c["studied"]])
                               for fam, wc in ws.items()}
        rows.append(row)
    return rows


def _pct(x: list[float] | None) -> str:
    if x is None:
        return "–"
    return f"{100 * x[0]:.1f}" if abs(x[0] - x[1]) < 5e-4 else f"{100 * x[0]:.1f}–{100 * x[1]:.1f}"


def markdown(res: dict) -> str:
    lines = [f"# Power under the weighted Holm ({res['source']})", "",
             (f"Weighted Holm (entry 74) at the frozen alpha' table `{res['alpha_table_sha256'][:12]}` restricted to the "
              + (f"final calibration: step 3's addendum `{res['calibration']['step3_addendum_sha256'][:12]}` (C2 "
                 f"two-sided at its validated levels, the repaired levels; entry 106) over the step-2 addendum "
                 f"`{res['addendum_sha256'][:12]}`" if (res.get("calibration") or {}).get("final") else
                 f"{SELECTION_STAGE}: step 3's selection `{res['calibration']['step3_selection_sha256'][:12]}` (C2 "
                 f"two-sided at its selected levels, the repaired levels; entry 131) over the step-2 addendum "
                 f"`{res['addendum_sha256'][:12]}`" if (res.get("calibration") or {}).get("stage") == SELECTION_STAGE else
                 f"step-2 addendum's usable levels `{res['addendum_sha256'][:12]}` (the historical one-sided C2)")
              + f" (entry 76; FWER check passed: {res['fwer_pass']}). {res['n_note']}"), "",
             ("Power is the claim's rejection rate by the step-down, in the cells that study it. The family's other "
              "claims are stand-ins there (their inputs measured in other cells or still pending), so they are "
              "bracketed: all forced rejected, all forced not rejected, and (where C5 is a stand-in) C5 alone of them "
              "rejected. First level: p* <= 0.05 w / 16 (weighted Bonferroni, whatever the other claims do)."), ""]
    for fam, claims in res["summary"].items():
        lines += [f"## {fam} family (%)", "",
                  ("| Claim | Cells | As simulated | Others rejected | Only C5 of the others rejected | "
                   "Others not rejected | First level |"),
                  "|---|---|---|---|---|---|---|"]
        for c, s in claims.items():
            if s is None:
                lines.append(f"| {c} | pending | – | – | – | – | – |")
            else:
                cells_txt = s["cells"][0] if len(s["cells"]) == 1 else f"{len(s['cells'])} cells"
                lines.append(f"| {c} | {cells_txt} | {_pct(s['as simulated'])} | {_pct(s['others rejected'])} | "
                             f"{_pct(s.get('only C5 of the others rejected'))} | {_pct(s['others not rejected'])} | "
                             f"{_pct(s['first level'])} |")
        lines.append("")
    lines += ["## Per cell (power % of each studied claim: others rejected / only C5 of them / none)", ""]
    for fam, members in pr.FAMILY_CLAIMS.items():
        lines += [f"### {fam}", "", "| Cell | " + " | ".join(members) + " |", "|---" * (len(members) + 1) + "|"]
        for r in res["cells"]:
            if not r.get("families"):
                lines.append(f"| {r['name']} | " + " | ".join(["–"] * len(members)) + f" |  {r.get('note') or ''}")
                continue
            f = r["families"][fam]
            vals = []
            for c in members:
                hi, lo = f["others rejected"][c], f["others not rejected"][c]
                mid = (f.get("only C5 of the others rejected") or {}).get(c)
                vals.append("·" if "forced" in hi else " / ".join(
                    f"{100 * x['power']['rate']:.1f}" for x in (hi, mid, lo) if x is not None))
            lines.append(f"| {r['name']} | " + " | ".join(vals) + " |")
        lines.append("")
    if res.get("c1a_comparison"):
        c, pw = res["c1a_comparison"], res["c1a_powers"]
        fd, gd = c["first_level_difference"], c["full_family_difference"]
        lines += ["## C1a choice inputs (design log ruling 2)", "",
                  (f"At the 3-point margin, both forms on the same {c['n']} datasets (common random numbers). "
                   f"**Weighted-Holm rejection power as simulated** (choose_c1a's input, change-log entry 104): pooled "
                   f"{c['full_family']['pooled C1a']:.4f} (MC SE {c['full_family_mc_se']['pooled C1a']:.4f}), "
                   f"two-family {c['full_family']['two-family']:.4f} (MC SE "
                   f"{c['full_family_mc_se']['two-family']:.4f}); pooled - two-family {gd['difference']:+.4f} (paired "
                   f"MC SE {gd['mc_se']:.4f}), choice {c['choice_full_family']} (pooled on a tie). **First-level "
                   f"power** (p* <= 0.05/16, recorded beside it): pooled {c['first_level']['pooled C1a']:.4f} (MC SE "
                   f"{c['first_level_mc_se']['pooled C1a']:.4f}), two-family {c['first_level']['two-family']:.4f} "
                   f"(MC SE {c['first_level_mc_se']['two-family']:.4f}); difference {fd['difference']:+.4f} (paired "
                   f"MC SE {fd['mc_se']:.4f}), choice {c['choice_first_level']}."), "",
                  (f"The two definitions agree: {c['definitions_agree']}. A close call (weighted-Holm difference "
                   f"within 2 SE): {c['close_call']}. The other claims are as simulated in the C1 cell (the reference "
                   f"bank cell, C5 at the calibration's alternative), so the weighted-Holm power depends on them. Every model is simulated at the equal-weight mean over P of the census "
                   f"arm's dev accuracy (simulate.Scenario has one accuracy), which understates the two-family C1a's "
                   f"power when Sonnet's accuracy is far from the mean."), "",
                  (f"Entry 81: {pw['needs_ruling'] or ('took ' + ', '.join(pw['entry81_dropped']) + ' out: the pooled C1a is not eligible, the two-family C1a enters' if pw['entry81_dropped'] else 'kept every model of P')}. "
                   f"Choice: {pw['choice']}."), ""]
    return "\n".join(lines)


def _result(cfg, ws, rows, source: str, n_note: str, calibration: dict[str, Any], arrays: dict | None = None,
            dev_rates: dict | None = None) -> dict[str, Any]:
    res = {"schema": SCHEMA, "source": source, "n_note": n_note, "calibration": calibration,
           "alpha_table_sha256": cfg.multiplicity_table_sha256, "addendum_sha256": calibration["step2_addendum_sha256"],
           "fwer_pass": {fam: wc.fwer_pass for fam, wc in ws.items()},
           "usable": {fam: {c: sorted((wc.final(c)[1] or {}).keys()) for c in pr.FAMILY_CLAIMS[fam]}
                      for fam, wc in ws.items()},
           "cells": rows, "summary": summarise(rows)}
    margin = next((r for r in rows if r["name"] == "C1 dev rates, margin 0.03" and r.get("families")), None)
    registered = calibration.get("final") is True or calibration.get("stage") == SELECTION_STAGE  # entry 131
    if margin and arrays is not None and registered:  # never on the one-sided calibration
        cmp = wp.c1a_comparison(arrays[margin["index"]], ws)
        leave = ((dev_rates or {}).get("entry81") or {}).get("leave")
        res["c1a_comparison"] = cmp
        res["entry81"] = (dev_rates or {}).get("entry81")
        # choose_c1a's inputs (design log ruling 2, "calibrated power at the 3-point margin from the dev grid's rates";
        # prereg 4.4 and entry 104: rejection power under the full weighted Holm): each form's rejection power as
        # simulated, the first-level powers beside them. The pooled C1a is eligible only if entry 81 keeps all seven
        # models of P (prereg.C1A_ELIGIBILITY, design log 18:50): when it takes any out, the two-family C1a enters
        # (prereg.c1a_choice). A ruling is needed only when entry 81's rates are missing or c1a_choice refuses.
        dropped, why = leave, None
        unmeasurable = ((dev_rates or {}).get("entry81") or {}).get("unmeasurable")
        if dropped is None:
            why = (f"entry 81 cannot be measured for {unmeasurable} (a fallback model with no completed record in an "
                   f"arm): the pooled C1a's eligibility needs a ruling" if unmeasurable else
                   "entry 81's capped:context rates are not in the dev rates")
            choice = None
        else:
            try:
                choice = pr.c1a_choice(cmp["full_family"][pr.POOLED], cmp["full_family"][pr.TWO_FAMILY], dropped)[0]
            except pr.PreregError as exc:
                choice, why = None, str(exc)
        res["c1a_powers"] = {**cmp["full_family"], "choice": choice,
                             "definition": "weighted Holm rejection power as simulated (C1 dev-rates cell)",
                             "mc_se": cmp["full_family_mc_se"], "difference": cmp["full_family_difference"],
                             "first_level": cmp["first_level"], "first_level_mc_se": cmp["first_level_mc_se"],
                             "first_level_difference": cmp["first_level_difference"],
                             "choice_first_level": cmp["choice_first_level"], "entry81_dropped": dropped,
                             "eligibility_rule": pr.C1A_ELIGIBILITY, "needs_ruling": why}
    return res


def cmd_aggregate(args) -> int:
    """The registered powers (and the C1a powers choose-c1a takes) on the registered calibration of both families
    (registered_calibrations: the final calibration once step 3's addendum is bound, else step 3's selection levels
    with the validation pending, entry 131; never the one-sided one); exit STEP3_PENDING, nothing written, while the
    config does not bind step 3's selection."""
    out = Path(args.out)
    plan = _load_plan(out)
    cfg = calibration_config(_plan_config(plan), Path(args.config))
    try:
        ws, record = registered_calibrations(cfg)
    except pr.PreregError as exc:
        print(f"the registered powers and the C1a choice wait for calibration step 3's selection (entry 131) or final "
              f"calibration (entry 106): {exc}", file=sys.stderr)
        return STEP3_PENDING
    simulated_by = (plan.get("fingerprint") or {}).get("inference")
    if simulated_by != cfg.calibrated_inference_digest:  # e.g. draws of the one-sided C2, simulated before 1e31f1f
        raise SystemExit(f"{out}/plan.json: simulated by inference code {simulated_by}, not the calibrated "
                         f"{cfg.calibrated_inference_digest} the final calibration is for: replay its draws as "
                         f"historical, or simulate them again")
    arrays = {c["index"]: _load_cell(out, plan, c["index"]) for c in plan["cells"] if c["simulated"]}
    rows = aggregate_arrays(arrays, plan["cells"], ws)
    dev_file = (plan["inputs"]["dev_rates"] or {}).get("file")
    dev = json.loads(Path(dev_file).read_text(encoding="utf-8")) if dev_file else None
    res = _result(cfg, ws, rows, f"final bank structure, {plan['fingerprint']['structure_source']}",
                  f"{plan['n']} datasets per cell (seed {SEED}); fingerprint code {plan['fingerprint']['code'][:12]}, "
                  f"bank items {plan['fingerprint']['bank_items'][:12]}.", record, arrays, dev)
    res.update(fingerprint=plan["fingerprint"], inputs=plan["inputs"], n=plan["n"])
    (out / "power.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "power.md").write_text(markdown(res) + "\n", encoding="utf-8")
    if "c1a_powers" in res:
        print(f"C1a powers: {res['c1a_powers']}")
    print(f"{out}/power.json\n{out}/power.md")
    return 0


def cmd_replay(args) -> int:
    """Job 17559's stored draws (results/power-holm-20261019, the provisional structure) by cell name: power_holm's grid
    is bank_cells' grid in the same order (A, R, B over r and f), studying C2, C3 and C4."""
    cfg = pr.PreregConfig.load(args.config)
    ws = historical_calibrations(cfg, Path(args.addendum), args.addendum_sha256)  # the draws' one-sided statistic
    meta = json.loads((Path(args.draws) / "cells.json").read_text(encoding="utf-8"))
    arrays, plan_cells = {}, []
    for c in meta["cells"]:
        row = {"index": c["index"], "name": c["name"], "note": c.get("note"), "studied": list(wp.BANK_STUDIED),
               "nulls": c.get("nulls"), "truths": c.get("truths")}
        plan_cells.append(row)
        if not c.get("feasible"):
            continue
        parts = sorted(Path(args.draws).glob(f"out/cell{c['index']:02d}-*.npz"))
        chunks = sorted(((int(np.load(f)["lo"]), np.load(f)["p"]) for f in parts), key=lambda x: x[0])
        p = np.concatenate([x[1] for x in chunks])
        if len(p) != args.n:
            raise SystemExit(f"cell {c['index']}: {len(p)} datasets, not {args.n}")
        arrays[c["index"]] = p
    rows = aggregate_arrays(arrays, plan_cells, ws)
    res = _result(cfg, ws, rows, "replay of job 17559's stored draws, provisional structure, on the historical "
                                 "one-sided calibration",
                  f"{args.n} datasets per cell, fingerprint {meta.get('fingerprint')} (step 1's).",
                  calibration_record(cfg, False, args.addendum_sha256))
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".md").write_text(markdown(res) + "\n", encoding="utf-8")
    print(f"{args.out}")
    return 0


def cmd_c5_adapt(args) -> int:
    """The entry-77 replay's planning counts as this run's C5 input file (weighted_power.c5_inputs_from_planning)."""
    cfg = pr.PreregConfig.load(args.config)
    doc = wp.c5_inputs_from_planning(json.loads(Path(args.planning).read_text(encoding="utf-8")), list(cfg.panel),
                                     cfg.graph)
    doc["planning_sha256"] = _sha(Path(args.planning).read_bytes())
    Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    wp.read_c5_inputs(Path(args.out), list(cfg.panel), cfg.graph)  # the file as written is accepted
    print(f"{args.out}")
    return 0


C1_ARMS = ("curated", "census")  # the prompted arms C1's dev rates read (weighted_power.c1_dev_rates)


def rescore_frozen(records: list[dict], cohorts: dict[str, str], questions: dict[str, dict],
                   arms: tuple[str, ...] = C1_ARMS) -> dict[str, int]:
    """Each scored record of ``arms`` whose score was made under an earlier rules version (scorer-rules-v4 or v5)
    re-scored under the frozen scorer's rules as they now read (v6 since change-log entry 127), through the runner's own
    path (loop.score_outcome against harness.score.question_gold under the question's rules), the record changed in
    place, in memory only. First the same path under the record's own rules version (the question's rules with every v5
    rule off for v4; with the v6 kind rule off for v5) must reproduce the stored score, which shows the record, the
    question and its gold belong together, and the record's question must be the question file's. A score under the
    frozen rules is kept; any other rules version is refused. Returns the counts."""
    from dataclasses import replace

    from harness.agent.loop import score_outcome
    from harness.score import ScoringRules, question_gold
    from harness.score.rules import RULES_VERSION, RULES_VERSION_V4, RULES_VERSION_V5

    counts = dict.fromkeys(("rescored", "changed", "kept_current", "unscored", "other_arms", "from_v4", "from_v5"), 0)
    for r in records:
        if cohorts.get(r["run_id"]) not in arms:
            counts["other_arms"] += 1
            continue
        score = r.get("score")
        if score is None:
            counts["unscored"] += 1
            continue
        version = (score.get("rules") or {}).get("version")
        if version == RULES_VERSION:
            counts["kept_current"] += 1
            continue
        if version not in (RULES_VERSION_V4, RULES_VERSION_V5):
            raise wp.PowerInputError(f"{r['run_id']}: its score's rules version {version!r} is none of "
                                     f"{RULES_VERSION_V4!r}, {RULES_VERSION_V5!r} and {RULES_VERSION!r}")
        q = questions.get(r["question_id"])
        if q is None:
            raise wp.PowerInputError(f"{r['run_id']}: its question {r['question_id']} is not in the question file")
        if (r.get("transcript") or [{}])[0].get("text") != q.get("question"):
            raise wp.PowerInputError(f"{r['run_id']}: its first message is not the question file's question text")
        base, gold = ScoringRules(**q["scoring_rules"]), question_gold(q)
        own = base.as_v4() if version == RULES_VERSION_V4 else replace(base, extra_columns_by_kind=False)
        if score_outcome(r["outcome"], r.get("final_rows"), gold, own.to_json()) != score:
            raise wp.PowerInputError(f"{r['run_id']}: the {version} scoring path does not reproduce its stored score")
        new = score_outcome(r["outcome"], r.get("final_rows"), gold, base.to_json())
        counts["rescored"] += 1
        counts["from_v4" if version == RULES_VERSION_V4 else "from_v5"] += 1
        counts["changed"] += new["exact"] != score["exact"]
        r["score"] = new
    return counts


def parse_fallback(items: Sequence[str]) -> dict[str, str | None]:
    """The fallback rule's explicit list (weighted_power.FALLBACK_RULE): each item MODEL (every completed prompted record)
    or MODEL=ISO-TIME (the records that ended before it; GLM: 05:45Z). Refused: an empty model, a repeated one, a cutoff
    that is not an ISO time."""
    out: dict[str, str | None] = {}
    for item in items:
        model, sep, cutoff = item.partition("=")
        if not model or model in out:
            raise SystemExit(f"--fallback {item!r}: an empty or repeated model")
        if sep:
            try:
                wp.utc_instant(cutoff, "the cutoff")
            except (ValueError, wp.PowerInputError) as exc:
                raise SystemExit(f"--fallback {item!r}: its cutoff is not an ISO time with a zone ({exc})") from None
        out[model] = cutoff if sep else None
    return out


def cmd_dev_rates(args) -> int:
    from harness.agent.loop import FROZEN_SCORER_DIGEST
    from harness.analysis import gate, inputs
    from harness.score.rules import RULES_VERSION

    cfg = pr.PreregConfig.load(args.config)
    if cfg.scorer_version != RULES_VERSION or cfg.scorer_digest != FROZEN_SCORER_DIGEST:
        raise SystemExit(f"{args.config}: its scorer ({cfg.scorer_version}, {cfg.scorer_digest[:12]}) is not this "
                         f"code's frozen one ({RULES_VERSION}, {FROZEN_SCORER_DIGEST[:12]})")
    fallback = parse_fallback(getattr(args, "fallback", None) or [])
    stray = [m for m in fallback if m not in cfg.panel]
    if stray:
        raise SystemExit(f"--fallback {stray}: not a model of P {list(cfg.panel)}")
    batch = inputs.load_batch(Path(args.grid_out), gate.grid_cells(Path(args.grid_out)))
    questions = inputs.load_questions(Path(args.questions))
    left_out: dict[str, dict[str, int]] = {}
    if fallback:  # the fallback rule's records: a fallback model's prompted records after its cutoff left out
        kept, left_out = wp.fallback_records(batch.records, batch.cohorts, fallback, C1_ARMS)
        batch = dataclasses.replace(batch, records=kept)
    arms = (*C1_ARMS, cfg.natural_cohort) if fallback else C1_ARMS
    rescored = rescore_frozen(batch.records, batch.cohorts, questions, arms=arms)
    attempts = inputs.attempts_from_batch(batch, questions)
    rates = wp.c1_dev_rates(attempts, list(cfg.panel), cfg.n_reps, fallback=list(fallback),
                            natural_cohort=cfg.natural_cohort)
    for m, cutoff in fallback.items():
        rates["fallback"]["models"][m].update(
            cutoff=cutoff, left_out=left_out[m],
            prompted_records={a: sum(r["model"] == m and batch.cohorts.get(r["run_id"]) == a for r in batch.records)
                              for a in C1_ARMS})
    rates["scorer"] = {"version": RULES_VERSION, "digest": FROZEN_SCORER_DIGEST, "rescored": rescored,
                       "how": f"each v4- or v5-scored record of {list(arms)} re-scored under the frozen rules in memory "
                              f"(rescore_frozen)"}
    rates["entry81"] = wp.entry81([(r["model"], batch.cohorts[r["run_id"]], (r.get("outcome") or {}).get("cap")
                                    == "context") for r in batch.records], list(cfg.panel),
                                  unmeasurable_ok=list(fallback))
    rates["grid_out"] = str(args.grid_out)
    rates["records"] = len(batch.records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rates, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "simulate", "run", "aggregate", "replay", "dev-rates", "c5-adapt"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=CONFIG)
        p.add_argument("--addendum", type=Path, default=ADDENDUM)
        p.add_argument("--addendum-sha256", default=ADDENDUM_SHA256)
        if name in ("plan", "simulate", "run", "aggregate"):
            p.add_argument("--out", type=Path, required=True)
        if name == "plan":
            p.add_argument("--dev-rates", type=Path, default=None)
            p.add_argument("--c5-inputs", type=Path, default=None)
            p.add_argument("--n", type=int, default=N)
            p.add_argument("--size", type=int, default=SIZE)
            p.add_argument("--agent-only", action="store_true",
                           help="simulate only the C1 and C5 cells (the bank cells are another OUT's)")
        if name == "simulate":
            p.add_argument("--cell", type=int, required=True)
            p.add_argument("--chunk", type=int, required=True)
        if name == "run":
            p.add_argument("--jobs", type=int, default=4)
            p.add_argument("--reuse", type=Path, default=None)
            p.add_argument("--no-aggregate", action="store_true", help="simulate only (aggregate waits for step 3)")
        if name == "replay":
            p.add_argument("--draws", type=Path, required=True, help="power-20261019 (cells.json, out/cell*.npz)")
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--n", type=int, default=2000)
        if name == "c5-adapt":
            p.add_argument("--planning", type=Path, required=True, help="the entry-77 replay's c5-inputs.json")
            p.add_argument("--out", type=Path, required=True)
        if name == "dev-rates":
            p.add_argument("--fallback", nargs="*", default=[], metavar="MODEL[=ISO-CUTOFF]",
                           help="models whose incomplete prompted phase takes the fallback rule (given, never inferred)")
            p.add_argument("--grid-out", type=Path, required=True)
            p.add_argument("--questions", type=Path,
                           default=ROOT / "data" / "questions" / "uniprot-sp-2026_03" / "questions.jsonl")
            p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run" and args.jobs > 4:
        ap.error("--jobs: at most 4 processes on the shared spoke")
    return {"plan": cmd_plan, "simulate": cmd_simulate, "run": cmd_run, "aggregate": cmd_aggregate,
            "replay": cmd_replay, "dev-rates": cmd_dev_rates, "c5-adapt": cmd_c5_adapt}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
