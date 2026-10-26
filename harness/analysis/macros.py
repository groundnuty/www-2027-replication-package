"""LaTeX macros generated from a results file: every number in the paper, never typed by hand.

``generate`` reads one results file (``analysis-results/1``) and writes:
- a ``.tex`` file with one ``\\newcommand`` per reported number, sorted by name;
- a side file (JSON) recording, for each macro, the results file (path and sha256), the JSON
  pointer of the value inside it (the query that produced the macro), the raw value, its
  formatted text and, for metrics, the metric's definition.

Names are TeX control words (letters only): ``res`` followed by the CamelCase parts of the
value's position (graph, model, cohort, use, source, metric, statistic), with digits spelled
out (``H1`` -> ``HOne``) and short aliases for the graphs and models in ``ALIASES``. Two
numbers that would get the same name are refused rather than silently overwritten. Output is
deterministic: regenerating from the same results file gives the same bytes.

A results file whose registered analysis is not stamped confirmatory (prereg.preflight: a test
run, with a cluster map, resample matrix or calibration table passed in, or a config built in
code) is refused outright: no macros are written from it, under any name.

A test run's macros are written only when asked (``test_run``; design ruling, design log, 20 Oct: the dress rehearsal
reaches the macros, so the generation path is exercised end to end), and only as test values: from a test run's
results alone (a confirmatory run's numbers are the paper's), every name under TEST_PREFIX instead of PREFIX (so that
no \\res macro of the paper can be defined from them), both files named with TEST_LABEL (never where a confirmatory
run's go), the .tex file's first line saying so and the side file recording ``test_run``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from harness.analysis.metrics import METRICS
from harness.analysis.prereg import UNBOUNDED

PREFIX = "res"
TEST_PREFIX = "testRun"  # a test run's macros (never the paper's names)
TEST_LABEL = "test-run"  # in the name of every file a test run's macros are written to
MACROS_SCHEMA = "analysis-macros/1"
NOT_ESTIMABLE = "n.e."
UNBOUNDED_TEXT = "unbounded"  # change-log entry 73: a bound beyond +-1 (prereg.UNBOUNDED), never a number
ALIASES = {
    "rhea-142": "Rhea",
    "rhea-142-opaque": "RheaOpaque",
    "uniprot-sp-2026_03": "Uniprot",
    "claude-sonnet-5-5": "Sonnet",
    "Qwen/Qwen3.8-27B": "Qwen",
    "-": "",
}
_DIGITS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")
_TOKEN = re.compile(r"[A-Za-z]+|[0-9]")
_CONTROL_WORD = re.compile(r"^[A-Za-z]+$")
_COST_KINDS = {"agent_usd": "usd", "retry_usd": "usd", "wall_s": "seconds", "check_s": "seconds",
               "input_tokens": "count", "output_tokens": "count", "evidence_triples": "count", "tool_calls": "number"}


class MacroError(ValueError):
    """A macro cannot be named or written."""


def name_part(value: str) -> str:
    if value in ALIASES:
        return ALIASES[value]
    return "".join(_DIGITS[int(t)] if t.isdigit() else t[0].upper() + t[1:] for t in _TOKEN.findall(str(value)))


def macro_name(*parts: str, prefix: str = PREFIX) -> str:
    body = "".join(name_part(p) for p in parts)
    name = prefix + body
    if not body or not _CONTROL_WORD.match(name):
        raise MacroError(f"no TeX control word for {parts!r}")
    return name


def _signed(text: str, negative: bool) -> str:
    return r"\ensuremath{-}" + text if negative and text.strip("0.") else text


def format_value(value: Any, kind: str) -> str:
    if value is None:
        return NOT_ESTIMABLE
    if isinstance(value, str) and value == UNBOUNDED:
        return UNBOUNDED_TEXT
    if kind == "text":
        return str(value)
    if kind == "count":
        return str(round(value))
    if kind == "rate":
        return f"{100 * value:.1f}"
    if kind == "change":
        return _signed(f"{abs(100 * value):.1f}", value < 0)
    if kind == "score":
        return f"{value:.3f}"
    if kind == "signed_score":
        return _signed(f"{abs(value):.3f}", value < 0)
    if kind == "p":
        return r"\ensuremath{<}0.001" if value < 0.001 else f"{value:.3f}"
    if kind == "usd":
        return f"{value:.4f}"
    if kind in ("seconds", "number"):
        return f"{value:.1f}"
    if kind == "level":
        return f"{100 * value:g}"
    if kind == "alpha":
        return f"{value:g}"
    raise MacroError(f"unknown value kind {kind!r}")


def _pointer(*tokens: Any) -> str:
    return "".join("/" + str(t).replace("~", "~0").replace("/", "~1") for t in tokens)


class _Collector:
    def __init__(self, prefix: str = PREFIX) -> None:
        self.macros: dict[str, dict[str, Any]] = {}
        self.prefix = prefix

    def add(self, parts: list[str], path: list[Any], value: Any, kind: str, **extra: Any) -> None:
        name = macro_name(*parts, prefix=self.prefix)
        pointer = _pointer(*path)
        if name in self.macros and self.macros[name]["pointer"] != pointer:
            raise MacroError(f"{self.macros[name]['pointer']} and {pointer} get the same name \\{name}")
        self.macros[name] = {"pointer": pointer, "value": value, "kind": kind, "text": format_value(value, kind),
                             **extra}

    def interval(self, parts: list[str], path: list[Any], node: dict[str, Any], field: str, kind: str,
                 suffixes: tuple[str, str], **extra: Any) -> None:
        bounds = node.get(field)
        for i, suffix in enumerate(suffixes):
            if bounds is None:
                self.add(parts + [suffix], path + [field], None, kind, **extra)
            else:
                self.add(parts + [suffix], path + [field, i], bounds[i], kind, **extra)

    def metric(self, parts: list[str], path: list[Any], name: str, node: dict[str, Any]) -> None:
        kind = METRICS[name].kind if name in METRICS else "rate"
        extra = {"definition": METRICS[name].definition} if name in METRICS else {}
        self.add(parts + [name], path + ["estimate"], node["estimate"], kind, **extra)
        self.interval(parts + [name], path, node, "ci", kind, ("Lo", "Hi"), **extra)
        if node.get("question_event"):
            qe, qpath = node["question_event"], path + ["question_event"]
            self.add(parts + [name, "QuestionsWithEvent"], qpath + ["k"], qe["k"], "count")
            self.add(parts + [name, "QuestionsAtRisk"], qpath + ["n"], qe["n"], "count")
            self.interval(parts + [name, "Questions"], qpath, qe, "exact", "rate", ("ExactLo", "ExactHi"),
                          definition=qe["event"])

    def bounds(self, parts: list[str], path: list[Any], iv: dict[str, Any], kind: str, prefix: str) -> None:
        """A prereg interval node {lo, hi, sided}: the finite bounds only."""
        for side, suffix in (("lo", "Lo"), ("hi", "Hi")):
            if iv.get("sided") == "greater" and side == "hi" or iv.get("sided") == "less" and side == "lo":
                continue
            self.add(parts + [prefix + suffix], path + [side], iv[side], kind)


def _component_parts(label: str) -> list[str]:
    return label.split(": ", 1) if ": " in label else [label]


def _confirmatory(out: _Collector, conf: dict[str, Any]) -> None:
    """Macros for the registered analysis (prereg §2, §4.4) under the weighted Holm (change-log entries 74, 76): per
    contrast its test p (primary and fallback), calibrated p*, weight, the step and level the step-down rejected it at
    (n.e. where it did not), claim, the table's test, and whether a compared arm is compromised; per component its
    estimate, unadjusted 95% interval and weighted-Bonferroni claim-level interval; CR2 beside the primary as a
    descriptive cross-check (nominal weighted Holm: its level and claim, and its intervals); C2's "at least 5 points";
    C3's equivalence at 5 points (entry 85: "equivalent within 5 points", or not, or "not certified");
    the bank rates C3 is reported beside; C5's per-model effects. No Holm-adjusted p: the weighted procedure decides
    by its thresholds, and an adjusted p can differ from them by an ulp on a level."""
    cfg = conf["config"]
    out.add(["Analysis", "FamilyAlpha"], ["confirmatory", "config", "alpha"], cfg["alpha"], "alpha")
    out.add(["Analysis", "FamilySize"], ["confirmatory", "n_contrasts"], conf["n_contrasts"], "count")
    for unit, name in (("agent", "Clusters"), ("bank", "BankClusters")):  # entry 52: the resampled units
        for field, suffix in (("clusters", ""), ("questions", "Questions"), ("largest", "Largest")):
            out.add(["Analysis", name + suffix], ["confirmatory", "clusters", unit, field], conf["clusters"][unit][field],
                    "count")
    for cid, c in conf["contrasts"].items():
        parts, path = ["Conf", cid], ["confirmatory", "contrasts", cid]
        fam, fpath = c["family"], path + ["family"]
        out.add(parts + ["P"], fpath + ["p_star"], fam["p_star"], "p")
        for test, name in (("boot_t", "PBootstrap"), ("cr2", "PCrTwo")):
            out.add(parts + [name], path + ["p_test", test], c["p_test"][test], "p")
        out.add(parts + ["Weight"], fpath + ["weight"], fam["weight"], "count")
        out.add(parts + ["HolmStep"], fpath + ["step"], fam["step"], "count")
        out.add(parts + ["HolmLevel"], fpath + ["level"], fam["level"], "p")
        out.add(parts + ["Claim"], fpath + ["claim"], "yes" if fam["claim"] else "no", "text")
        out.add(parts + ["Test"], fpath + ["test"], fam["test"], "text")
        cross, xpath = c["cross_check_cr2"], path + ["cross_check_cr2"]  # CR2 beside the primary, descriptive
        out.add(parts + ["CrossCrTwoLevel"], xpath + ["level"], cross["level"], "p")
        out.add(parts + ["CrossCrTwoClaim"], xpath + ["rejected_nominal"], "yes" if cross["rejected_nominal"] else "no",
                "text")
        # change-log entry 73: the resamples whose se* is floating-point residue of no spread, and, when there are
        # any, the claim under the sensitivity analysis treating them as undefined
        spread, zpath = conf["zero_spread"], ["confirmatory", "zero_spread"]
        out.add(parts + ["NearZeroSpread"], zpath + ["counts", cid, "resamples"], spread["counts"][cid]["resamples"],
                "count")
        if spread["sensitivity"] is not None:
            sens, spath = spread["sensitivity"][cid], zpath + ["sensitivity", cid]
            out.add(parts + ["SensitivityP"], spath + ["p_star"], sens["p_star"], "p")
            out.add(parts + ["SensitivityHolmLevel"], spath + ["level"], sens["level"], "p")
            out.add(parts + ["SensitivityClaim"], spath + ["claim"], "yes" if sens["claim"] else "no", "text")
        concl = conf["conclusions"][cid]
        cpath = ["confirmatory", "conclusions", cid]
        out.add(parts + ["Compromised"], cpath + ["compromised"], "yes" if concl["compromised"] else "no", "text")
        out.add(parts + ["MaxMissing"], cpath + ["max_missing_share"], concl["max_missing_share"], "rate")
        for i, comp in enumerate(c["components"]):
            cparts, ppath = parts + _component_parts(comp["label"]), path + ["components", i]
            out.add(cparts + ["Est"], ppath + ["estimate"], comp["estimate"], "change",
                    definition=f"{cid} component {comp['label']}")
            out.bounds(cparts, ppath + ["ci95"], comp["ci95"], "change", "")
            # design log ruling: a nominal claim-level interval (its first level uncalibrated) is never named or shown as
            # a calibrated one: "Uncertified" bounds, and its coverage text the paper prints beside them
            wb = comp["ci_weighted_bonferroni"]
            certified = wb.get("calibrated") is True
            out.bounds(cparts, ppath + ["ci_weighted_bonferroni"], wb, "change",
                       "WeightedBonferroni" if certified else "WeightedBonferroniUncertified")
            out.add(cparts + ["WeightedBonferroniCoverage"], ppath + ["ci_weighted_bonferroni", "coverage"],
                    wb.get("coverage") if certified else "not certified", "text")
            ccross = comp["cross_check_cr2"]
            out.bounds(cparts, ppath + ["cross_check_cr2", "ci95"], ccross["ci95"], "change", "CrTwo")
            out.bounds(cparts, ppath + ["cross_check_cr2", "ci_weighted_bonferroni"], ccross["ci_weighted_bonferroni"],
                       "change", "CrTwoWeightedBonferroni")
            # design log ruling: CR2's are nominal by design and descriptive, labelled so beside its bounds
            out.add(cparts + ["CrTwoWeightedBonferroniCoverage"],
                    ppath + ["cross_check_cr2", "ci_weighted_bonferroni", "coverage"],
                    ccross["ci_weighted_bonferroni"]["coverage"], "text")
        if cid == "C2":
            for label, ok in fam["at_least_min_effect"].items():
                out.add(parts + _component_parts(label) + ["AtLeastFivePoints"], fpath + ["at_least_min_effect", label],
                        "not certified" if ok is None else "yes" if ok else "no", "text")
        if cid == "C3":  # entry 85: the equivalence reading, never from an uncertified interval (n.e. when not estimable)
            out.add(parts + ["Equivalence"], fpath + ["equivalence", "reading"], fam["equivalence"]["reading"], "text")
        if cid in ("C2", "C3"):
            for source, rates in c["rates"].items():
                for field in ("f", "f_gold", "f_rewrite", "recall", "recall_type_wrong", "precision"):
                    out.add(parts + [source, field], path + ["rates", source, field], rates[field], "rate")
        if cid == "C5":
            for model, node in c["descriptive"]["per_model"].items():
                mpath = path + ["descriptive", "per_model", model]
                out.add(parts + [model, "Est"], mpath + ["estimate"], node["estimate"], "change")
                out.add(parts + [model, "Unconditional"], mpath + ["unconditional"], node["unconditional"], "change")


def _ledger(out: _Collector, ledger: dict[str, Any]) -> None:
    """Macros for the ledger on real outputs beside the bank (prereg §7, change-log entries 85 and 86; descriptive):
    per row (graph x cohort x use x source, "Ledger") and per model (the appendix, "LedgerModel"), its counts, its
    catches, false flags, fixes and corruptions per attempt, its unknown rate and, where estimable, its non-empty
    column; per row its bank rates, on all items and on the non-empty ones."""
    for table, prefix, fields in (("rows", "Ledger", ("graph", "cohort", "use", "source")),
                                  ("by_model", "LedgerModel", ("graph", "model", "cohort", "use", "source"))):
        for i, row in enumerate(ledger[table]):
            parts, path = [prefix, *(row[f] for f in fields)], ["real_output_ledger", table, i]
            for name, value in row["counts"].items():
                out.add(parts + ["Count", name], path + ["counts", name], value, "count")
            for name, node in row["per_attempt"].items():
                out.metric(parts + ["PerAttempt"], path + ["per_attempt", name], name, node)
            out.metric(parts, path + ["unknown_rate"], "unknown_rate", row["unknown_rate"])
            if row["non_empty"]["estimable"]:
                for name in ("catches", "false_flags"):
                    out.metric(parts + ["NonEmptyPerAttempt"], path + ["non_empty", name], name, row["non_empty"][name])
            bank = row.get("bank")
            if bank is None:
                continue
            for name in ("recall", "false_flag_rate", "unknown_rate"):
                out.metric(parts + ["Bank"], path + ["bank", name], name, bank[name])
            if bank["non_empty"]["estimable"]:
                for name in ("recall", "false_flag_rate"):
                    out.metric(parts + ["BankNonEmpty"], path + ["bank", "non_empty", name], name,
                               bank["non_empty"][name])


def collect(results: dict[str, Any], *, test_run: bool = False) -> dict[str, dict[str, Any]]:
    """Every macro of a results file; ``test_run``: a test run's, as test values (TEST_PREFIX), from a test run's results
    only."""
    if test_run:
        conf = results.get("confirmatory")
        if not conf:
            raise MacroError("a test run's macros come from a test run of the registered analysis: these results have no "
                             "registered analysis")
        if conf.get("confirmatory") is not False:
            raise MacroError("a confirmatory run's numbers are the paper's: they are not written as a test run's test "
                             "values")
    out = _Collector(TEST_PREFIX if test_run else PREFIX)
    settings = results["settings"]
    out.add(["Analysis", "BootstrapResamples"], ["settings", "n_boot"], settings["n_boot"], "count")
    out.add(["Analysis", "Seed"], ["settings", "seed"], settings["seed"], "count")
    out.add(["Analysis", "CiLevel"], ["settings", "level"], settings["level"], "level")
    out.add(["Analysis", "Questions"], ["settings", "n_questions"], settings["n_questions"], "count")
    for label, arm in sorted(results["arms"].items()):
        key = arm["key"]
        parts = [key["graph"], key["model"], key["cohort"], key["use"], key["source"]]
        base = ["arms", label]
        for name, value in sorted(arm["counts"].items()):
            out.add(parts + [name], base + ["counts", name], value, "count")
        for name, node in sorted(arm["metrics"].items()):
            out.metric(parts, base + ["metrics", name], name, node)
        for t, node in sorted((arm.get("by_check_type") or {}).items()):
            out.add(parts + ["Type", t, "W_F"], base + ["by_check_type", t, "W_F"], node["W_F"], "count")
            out.add(parts + ["Type", t, "C_F"], base + ["by_check_type", t, "C_F"], node["C_F"], "count")
            out.metric(parts + ["Type", t], base + ["by_check_type", t, "flag_precision"], "flag_precision",
                       node["flag_precision"])
        for pi, node in sorted((arm.get("precision_at_prevalence") or {}).items()):
            pparts = parts + ["PrecisionAtPrevalence", str(round(float(pi) * 100))]
            ppath = base + ["precision_at_prevalence", pi]
            out.add(pparts, ppath + ["estimate"], node["estimate"], "rate")
            out.interval(pparts, ppath, node, "ci", "rate", ("Lo", "Hi"))
        for field, dist in sorted((arm.get("costs") or {}).items()):
            for stat in ("n", "mean", "median", "total"):
                kind = "count" if stat == "n" else _COST_KINDS.get(field, "number")
                out.add(parts + ["Cost", field, stat], base + ["costs", field, stat], dist[stat], kind)
        for dim, levels in sorted((arm.get("strata") or {}).items()):
            for level, node in sorted(levels.items()):
                sparts, spath = parts + ["By", dim, level], base + ["strata", dim, level]
                for name, value in sorted(node["counts"].items()):
                    out.add(sparts + [name], spath + ["counts", name], value, "count")
                for name, mnode in sorted(node["metrics"].items()):
                    out.metric(sparts, spath + ["metrics", name], name, mnode)
    if results.get("real_output_ledger"):
        _ledger(out, results["real_output_ledger"])
    if results.get("confirmatory"):
        if not test_run and results["confirmatory"].get("confirmatory") is not True:  # prereg.preflight's stamp
            raise MacroError("the results' registered analysis is not a confirmatory run (a test run: a cluster map, "
                             "resample matrix or calibration table passed in, or a config built in code): no macros "
                             "are written from it")
        _confirmatory(out, results["confirmatory"])
    return out.macros


def generate(results_path: Path, tex_path: Path, side_path: Path, *, test_run: bool = False) -> tuple[Path, Path]:
    """Write the macros and their side file from one results file; ``test_run``: a test run's macros as test values, to
    files named with TEST_LABEL (the module docstring)."""
    if test_run and not all(TEST_LABEL in Path(p).name for p in (tex_path, side_path)):
        raise MacroError(f"a test run's macros are written only to a test-labelled place: both file names must hold "
                         f"{TEST_LABEL!r} (never where a confirmatory run's macros go)")
    raw = Path(results_path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    results = json.loads(raw)
    macros = collect(results, test_run=test_run)
    gated = bool(results["settings"].get("gated"))
    conf = results.get("confirmatory")
    status = ("a test run of the registered analysis" if test_run else
              "confirmatory analysis (prereg draft 3)" if conf else "no confirmatory analysis")
    header = [
        *([("% TEST RUN (not confirmatory): test values, never the paper's numbers; every name begins "
            f"\\{TEST_PREFIX}.")] if test_run else []),
        f"% Generated by harness/analysis/macros.py from {results_path} (sha256 {sha}); do not edit, regenerate.",
        (f"% gated: {'yes' if gated else 'no'} (every input batch passed its replay check); {status}; "
         f"seed {results['settings']['seed']}, {results['settings']['n_boot']} bootstrap resamples."),
        f"% {len(macros)} macros; the side file {Path(side_path).name} gives each one's JSON pointer in the results file.",
    ]
    lines = header + [f"\\newcommand{{\\{name}}}{{{m['text']}}}" for name, m in sorted(macros.items())]
    Path(tex_path).parent.mkdir(parents=True, exist_ok=True)
    Path(side_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tex_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    side = {"schema": MACROS_SCHEMA, "results_file": str(results_path), "results_sha256": sha, "gated": gated,
            "confirmatory": bool(conf) and conf.get("confirmatory") is True, "test_run": test_run,
            "tex_file": str(tex_path), "macros": macros}
    Path(side_path).write_text(json.dumps(side, sort_keys=True, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return Path(tex_path), Path(side_path)
