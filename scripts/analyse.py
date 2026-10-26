#!/usr/bin/env python3
"""Run records to the paper's numbers (design log, EA.1; paper/preregistration.md draft 3).

Every batch (a runner OUT directory) must have a passing, current replay check
(scripts/verify_batch.py) and is joined with the questions file it ran on (by sha256 against
its manifest). The design is every question of --split with the frozen config's repetitions per
cohort (--n-reps must agree with it); a registered attempt with no record is missing (prereg §4).
With --prereg-config, the fault bank's ledger rows must equal the frozen bank manifest. Checker-loop outputs
(``checker-ledger/1``) are optional; with --prereg-config the registered confirmatory
analysis is added (--test-run: a test run of it, never on the test split, its macros written only as test values
under OUT/test-run/); it runs on the pinned
confirmatory host, single-threaded (change-log entry 73: the preflight
refuses unless OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 are set when the process starts), and refuses before it
reads any record when a digest or a known answer differs. Writes, to --out:

- ``results.json``: every computed number (arms, strata, the confirmatory analysis) and the settings;
- ``numbers.tex``: one generated \\newcommand per number (never edit it; regenerate);
- ``numbers.json``: each macro's results file, JSON pointer, raw value and definition;
- with --test-run, instead of those two, ``test-run/numbers-test-run.tex`` and ``test-run/numbers-test-run.json``: the
  test run's macros, marked as test values and named \\testRun..., never the paper's numbers (design ruling, design log,
  20 Oct: the dress rehearsal reaches the macros).

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/analyse.py --batch OUT [--batch OUT ...] --questions QUESTIONS.jsonl \\
        --split test [--n-reps 3] [--ledger LEDGER.jsonl ...] \\
        [--prereg-config data/analysis/prereg-config.json --bank-manifest BANK/manifest.json] \\
        [--tier-if-missing lookup] [--test-run [--bank-questions IDS]] --out DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.analysis.gate import BatchRefused
from harness.analysis.inputs import InputError, read_question_ids
from harness.analysis.macros import TEST_LABEL, generate
from harness.analysis.pipeline import DEFAULT_N_BOOT, DEFAULT_SEED, analyse_batches
from harness.analysis.prereg import PreregConfig, PreregError


def checked_call_fn():
    """Contract v2 rule 22 as scripts/pilot_summary.py implements it (one implementation)."""
    spec = importlib.util.spec_from_file_location("pilot_summary", ROOT / "scripts" / "pilot_summary.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pilot_summary"] = module
    spec.loader.exec_module(module)
    return module.checked_call


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", type=Path, action="append", required=True, help="a runner OUT directory")
    ap.add_argument("--questions", type=Path, action="append", required=True,
                    help="questions files; each batch uses the one its manifest names by sha256")
    ap.add_argument("--split", required=True, help="the registered split (prereg §1: test)")
    ap.add_argument("--n-reps", type=int,
                    help="repetitions per question (prereg §4 'Fixed stopping'); with --prereg-config it must equal "
                         "the frozen value (taken from the config when omitted); otherwise 3 when omitted")
    ap.add_argument("--ledger", type=Path, action="append", default=[], help="checker-ledger/1 files")
    ap.add_argument("--prereg-config", type=Path, help="add the registered confirmatory analysis")
    ap.add_argument("--bank-manifest", type=Path,
                    help="the frozen fault bank's manifest.json (items.jsonl beside it); required with --prereg-config, "
                         "checked against the config's bank_manifest_sha256")
    ap.add_argument("--bank-questions", type=Path,
                    help="a test run only: a declared subset of the split's bank questions (a JSON array of ids, or one "
                         "per line); the bank rows must be exactly those questions' items, and the subset is recorded "
                         "in results.json (settings.bank.question_subset); a confirmatory run refuses it")
    ap.add_argument("--tier-if-missing",
                    help="tier for questions files written before the two tiers (change log entry 6); recorded")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--test-run", action="store_true",
                    help="a test run of the registered analysis (design log ruling): never on the registered split (test), "
                         "labelled not confirmatory, unfrozen tables giving p* = 1; its macros are written as test "
                         "values to OUT/test-run/, never as the paper's numbers")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.test_run and not args.prereg_config:
        ap.error("--test-run runs the registered analysis: it needs --prereg-config")
    if args.bank_questions and not args.test_run:
        ap.error("--bank-questions is accepted in a test run only (--test-run): a confirmatory run checks every bank "
                 "item of the split")
    try:
        bank_questions = read_question_ids(args.bank_questions) if args.bank_questions else None
        prereg = PreregConfig.load(args.prereg_config) if args.prereg_config else None
        results = analyse_batches(args.batch, args.questions, split=args.split, n_reps=args.n_reps,
                                  ledgers=args.ledger, prereg=prereg, bank_manifest=args.bank_manifest,
                                  checked_call=checked_call_fn(),
                                  tier_if_missing=args.tier_if_missing, seed=args.seed, n_boot=args.n_boot,
                                  bank_questions=bank_questions,
                                  **({"testing": True} if args.test_run else {}))
    except (BatchRefused, PreregError, InputError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.json"
    results_path.write_text(json.dumps(results, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                            encoding="utf-8")
    if args.test_run:  # never the paper's numbers: test values, in a test-labelled place
        tex, side = generate(results_path, args.out / TEST_LABEL / f"numbers-{TEST_LABEL}.tex",
                             args.out / TEST_LABEL / f"numbers-{TEST_LABEL}.json", test_run=True)
        n = len(json.loads(side.read_text(encoding="utf-8"))["macros"])
        print(f"{results_path}: {len(results['arms'])} arms, a test run (not confirmatory); {tex}: {n} test-value macros")
        return 0
    tex, side = generate(results_path, args.out / "numbers.tex", args.out / "numbers.json")
    n = len(json.loads(side.read_text(encoding="utf-8"))["macros"])
    conf = "with" if results["confirmatory"] else "without"
    print(f"{results_path}: {len(results['arms'])} arms, {conf} the confirmatory analysis; {tex}: {n} macros")
    return 0


if __name__ == "__main__":
    sys.exit(main())
