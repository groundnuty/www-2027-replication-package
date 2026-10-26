"""The inference code bound by two digests (design log): what the calibrated alpha' depend on, and the cluster grouping
added after the calibration (change-log entry 52).

The calibration table cannot record the code it was computed on beyond its own fingerprint, and the confirmatory
analysis loads the table by its frozen sha256 and rule versions. So the inference code itself is bound here, in two
registries, each a list of functions, classes and class members (by qualified name) and a list of module constants:

- The calibrated registry (``INFERENCE_FUNCTIONS``, ``INFERENCE_CONSTANTS``): everything the calibrated alpha' depend on.
  That is the bootstrap-t, CR2-Satterthwaite, the resampling, the studentisation, the t distribution, Holm, the
  calibrated-p application, the table's loader (load_calibration, Calibration.load, the C1a family choice) and alpha'
  lookup, the per-question estimators of the six contrasts, and the config's model sets and arm keys they read; and,
  reached from confirmatory, the descriptive retry contrast and the arms each conclusion's missing shares qualify. Every
  object in it is byte-identical at the calibrated commit 87c7126, so its digest is frozen as the value computed there
  (``calibrated_inference_digest`` in the config).
- The grouping registry (``GROUPING_FUNCTIONS``, ``GROUPING_CONSTANTS``): the code added after the calibration between
  the data and the calibrated functions. That is the cluster dispatcher prereg.studentise, Clusters, the cluster map
  (its frozen-file loader and self-check, connected_clusters, load_clusters) and the grouping before run_contrasts
  (clustered_contrasts, which takes its config, map, matrix and table only from the preflight); the preflight a
  confirmatory run starts with before it reads any data (prereg.preflight: the C1a choice, the guard
  check_inference_digest, the frozen inputs, the config reloaded from its file and compared) together with the digest
  computation the guard runs (this module's inference_digest, grouping_digest and their helpers, and the registry
  lists) and the known-answer test it recomputes (kat.py, prereg.check_known_answers); the config class whole (its construction: __init__, __post_init__, load); and confirmatory, the
  orchestration from the preflight to the report. None of it exists unchanged at 87c7126, so its digest is frozen at
  the freeze commit (``grouping_digest`` in the config, [FILL] until then).

- The multiplicity registry (``MULTIPLICITY_FUNCTIONS``, ``MULTIPLICITY_CONSTANTS``; design log, change-log entries 74 and
  76): the weighted Holm on calibrated p's (multiplicity.py) and its known-answer part, new code with its own digest
  (``multiplicity_digest`` in the config, frozen after its review); with the producers of its frozen inputs (step 1's
  alpha' table from step 1's stored p-values, step 2's verification, the dated step-2 addendum and its check; entry
  91: step 2's procedure is bound at the freeze, its output by the addendum) and step 1's grid rule they reuse. The plain Holm stays in the calibrated registry,
  as the calibration ran it; a confirmatory run no longer reaches it (the contrasts run uncalibrated).

``inference_digest()``, ``grouping_digest()`` and ``multiplicity_digest()`` hash each registry's sources (inspect.getsource, line endings
normalised) and constant values, and what every module-level name a registered object reads is bound to
(``bound_names``: a callable's defining module and qualified name, a module's name; for a class registered whole, each
member's identity). So an import alias (``cr2_p as bootstrap_t_p``), a module-level assignment or a class member replaced
after its class body changes a digest although no registered source does. The calibrated digest binds the calibrated
objects' names as they were at 87c7126, with the one approved rebinding (``REBOUND``: the contrasts' studentise, now the
entry-52 dispatcher) recorded as its calibrated target; the grouping digest binds every registered object's names as
they are. prereg.check_inference_digest refuses a confirmatory run unless both digests equal the frozen values.

The registries are closed (``unclosed``; tests/test_analysis_inference.py). ``scan`` walks each registered object's code
by Python's scoping rules and finds what it reaches:
- every in-repo function and class it reads by name (a global, a local import, or an in-repo module's attribute);
- every module constant;
- every member of an in-repo class named in the code (``ClusterMap.load``), and the class a classmethod's ``cls`` is;
- the name of every other attribute it reads (``cfg.c5_models``, ``calibration.entry``, ``K.sum``).

The receiver's class of such an attribute is not inferred, because annotations do not hold (the contrasts' ``K:
np.ndarray`` receives Clusters). The rule is by name: every method, property or class constant of that name on any
class under harness/ must be registered (or its class whole), unless ``ALLOWED`` names that caller and attribute with
why the receiver there is never an in-repo object. Names looked up at run time (getattr, eval) are reported. Classes
whose instances the inference compares, hashes, measures or constructs are registered whole, since implicit dunder calls
(``len(index)``, ``bank_index == index``, a dict keyed by ArmKey) are not attribute reads.

The digests describe what a name is bound to by its module and qualified name, so what runs under a name can still
change without any registered source changing (a wrapper keeping the wrapped function's name, a generated method, a
module attribute swapped, a builtin replaced): the known answers (kat.py) bind what the inference computes, on fixed
inputs, in the run's own process, and the preflight refuses when they change.

Registering the guard in the digest it checks is not circular (the frozen value is stored in the config, not in the
code), and it catches an edit to the guard or to the digest computation that leaves the comparison running. An edit that
removes the comparison itself cannot be caught by the run that no longer compares; the test suite compares the config's
frozen grouping digest with the current code's on its own (tests/test_analysis_inference.py).

Not covered, by design: the data preparation and input refusals confirmatory runs after the preflight, which read the
attempt units (``BOUNDARY``: vectors_from_units, the arm, scorer and repetition checks, the missing shares). They get the
caller's config; the inference gets the preflight's own copy. The calibration simulated question vectors directly.
tests/test_analysis_inference.py also checks that confirmatory's first statement is the preflight, before it reads
any data, and that it reaches the inference only through the preflight and clustered_contrasts.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import importlib.util
import inspect
import json
import textwrap
import types
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

_S, _P, _M = "harness.analysis.stats.", "harness.analysis.prereg.", "harness.analysis.metrics."
INFERENCE_FUNCTIONS = (
    # the resampling, the studentised statistic, the bootstrap-t, the t distribution and CR2-Satterthwaite, Holm
    *(_S + n for n in ("resample_matrix", "Term", "Studentised", "_theta", "studentise", "_t_star", "bootstrap_t_p",
                       "bootstrap_t_bound", "_betacf", "_betacf_scalar", "betainc", "t_sf", "cr2_p", "holm", "holm_levels")),
    # the calibrated p, the per-question estimators and the six contrasts, their intervals, the family
    *(_P + n for n in ("PreregError", "calibrated_p", "resamples", "_share", "_test_ps", "_max_ps", "_two_sided",
                       "_accuracy_terms", "_c1_stat", "c1a", "c1b", "c5", "_bank_design", "_f_q", "_f_terms", "_f_diff_terms",
                       "_rate", "source_rates", "c2", "c3", "c4", "_t_quantile", "_bound", "_interval", "run_contrasts")),
    # C2 two-sided since change-log entry 106 (calibrated afresh by step 3): its "at least 5 points" in its direction
    _P + "c2_at_least",
    # reached by attribute or by implicit dunder calls (blocker 2 of the reviews of bff8c89): the question index (its
    # len and strata drive the resampling), the config's model sets and arm keys, the calibration table (its loader by
    # the frozen hash and the alpha' lookup) and the arm key the contrasts look their vectors up by
    _S + "QuestionIndex",
    *(_P + n for n in ("PreregConfig.c1a_pooled", "PreregConfig.c5_models", "PreregConfig.c1a_models",
                       "PreregConfig.families", "PreregConfig.prompt", "PreregConfig.bank", "Calibration")),
    _M + "ArmKey",
    # reached from the preflight: which family's alpha' are loaded, and from which table
    *(_P + n for n in ("load_calibration", "PreregConfig.require_c1a_choice", "choose_c1a")),
    # reached from confirmatory: the arms each conclusion's missing shares qualify (§4.3), the descriptive retry contrast
    *(_P + n for n in ("compared_arms", "c4_retry", "PreregConfig.checker")),
)
INFERENCE_CONSTANTS = (
    *(_S + n for n in ("SCHEMES", "PAIRWISE_MAX", "QUANTILE_METHOD")),
    *(_P + n for n in ("CONTRASTS", "TESTS", "HOLM_DIVISORS", "F_WEIGHTS", "TWO_FAMILY", "POOLED", "UNCALIBRATED")),
    *(_P + n for n in ("BANK", "CALIBRATION_RULES", "FAMILY_CLAIMS", "STEP2_FWER_FAILURE", "PLACEHOLDER")),
    _M + "NO_SOURCE",
    _P + "C1A_FAMILIES",
)
_I, _K, _L = "harness.analysis.inference.", "harness.analysis.kat.", "harness.analysis.pipeline."
GROUPING_FUNCTIONS = (
    # the cluster grouping (entry 52): the dispatcher, the clusters, the self-checking map and its loader, the grouping
    # before run_contrasts
    *(_P + n for n in ("studentise", "Clusters", "ClusterMap", "connected_clusters", "load_clusters",
                       "clustered_contrasts")),
    # the preflight a confirmatory run starts with, before reading data: the guard comparing both digests, the frozen
    # inputs (no map, matrix or table passed in; the config reloaded from its file and compared), and the digest
    # computation the guard runs
    *(_P + n for n in ("Preflight", "preflight", "check_inference_digest", "check_multiplicity_digest",
                       "check_known_answers", "check_test_run_split")),
    # the zero-spread report and the bounds of the confirmatory report (change-log entry 73)
    *(_P + n for n in ("near_zero_spread", "spread_undefined", "_claims", "zero_spread", "_bound_shown", "bounded")),
    *(_I + n for n in ("inference_digest", "grouping_digest", "multiplicity_digest", "_digest", "source_digest",
                       "resolve", "_source", "_code")),
    # the known-answer test the preflight recomputes (kat.py): its inputs, the paths it runs, its serialisation
    *(_K + n for n in ("_uniforms", "_hits", "_spread", "kat_index", "kat_config", "kat_vectors", "_empty",
                       "kat_matrix", "_attempt", "_statistic", "_primitives", "_statistics", "_table_bytes", "_tables",
                       "_contrasts", "_family", "calibrated_answers", "_map_bytes", "grouping_answers",
                       "_config_answers", "_ledger_names", "_key", "_canon", "canonical", "known_answers", "_openblas", "_cpu",
                       "environment")),
    # the bindings in both digests and the walk that finds them
    *(_I + n for n in ("bound_names", "dataclass_generated", "references", "_module_name", "scan", "_Walk", "Reach",
                       "_describe", "_Owner", "_Imported", "_import_bindings", "_bindings", "_arguments")),
    # the config whole (its construction: __init__, __post_init__, load), and the orchestration from the preflight to
    # the report (the review of 049e4b3)
    *(_P + n for n in ("PreregConfig", "confirmatory", "repo_root")),
    # the pooled C1a's eligibility on top of ruling 2 (design log, 18:50): the choice under it, its record's check and the
    # known answer that runs them
    *(_P + n for n in ("c1a_choice", "_entry81_dropped")), _K + "_eligibility_answers",
    # the ledger's source names -> the config's arms (the review of 9d9c88c: swapping the mapping swaps C2's, C4's and
    # C5's arms, so it is bound with the grouping), and the check that licenses gen:any's generic rows as C5's generic arm
    # (design log entry 91, spec trace B10: its records decided on curated sources alone)
    _L + "ledger_source_names", _L + "curated_generic_rows",
)
GROUPING_CONSTANTS = (_P + "CLUSTER_MAP_SCHEMA", _P + "HOLM_WEIGHTS", _P + "REGISTERED_SPLIT", _P + "NEAR_ZERO_SPREAD", _P + "NEAR_ZERO_RELATIVE", _P + "UNBOUNDED",
                      _L + "LEDGER_SOURCES", _L + "GENERIC_ON_CURATED", _L + "CURATED_GENERIC_VERSIONS", _L + "CURATED_KIND",
                      _P + "REFUSAL_RULE", _P + "C1A_ELIGIBILITY",
                      *(_K + n for n in ("KAT_SCHEMA", "KAT_SEED", "KAT_N_BOOT", "KAT_REPS", "KAT_MODELS", "KAT_PANEL",
                                         "KAT_PANEL_FAMILIES", "KAT_AGENT_QUESTIONS", "KAT_BANK_QUESTIONS",
                                         "KAT_ACCURACY", "KAT_RETRY", "KAT_BANK_RATES", "KAT_C5", "KAT_MASKED", "KAT_MISSING",
                                         "KAT_TABLE", "KAT_STEP1", "KAT_STEP1_P", "KAT_ANCHORS", "KAT_BETA", "KAT_X",
                                         "KAT_T", "KAT_DF", "KAT_Q")),
                      *(_I + n for n in ("INFERENCE_FUNCTIONS", "INFERENCE_CONSTANTS", "GROUPING_FUNCTIONS",
                                         "GROUPING_CONSTANTS", "MULTIPLICITY_FUNCTIONS", "MULTIPLICITY_CONSTANTS",
                                         "ALL_FUNCTIONS", "REBOUND", "REPO_PACKAGE", "_DYNAMIC")))
_U = "harness.analysis.multiplicity."
# The multiplicity registry (design log, change-log entries 74 and 76): the weighted Holm on calibrated p's, new code
# with its own digest (``multiplicity_digest`` in the config), apart from the calibrated one (frozen from 87c7126) and
# the grouping one; frozen after its review. The plain Holm stays in the calibrated registry, as the calibration ran it.
_C = "harness.analysis.calibration."
MULTIPLICITY_FUNCTIONS = (*(_U + n for n in (
    "level", "reachable_sums", "reachable_levels", "p_star", "step_down", "WeightedCalibration", "_read", "_number",
    "load_files", "load", "table_document", "usable_document", "document_bytes", "_claim_name", "decisions",
    "c3_equivalence", "weighted_family")),
    # the producers of the frozen inputs (entries 74, 76, 91): step 1's alpha' table from its stored p-values, step 2's
    # verification on its own, the dated addendum and its check; with step 1's grid rule they reuse (calibration.py,
    # unchanged since 87c7126)
    *(_U + n for n in ("configuration_p", "outputs_digest", "alpha_primes", "sizes_within", "p_star_array",
                       "step_down_rejections", "calibrated_matrix", "family_fwer", "joint_factor", "step_one_table",
                       "_se", "_table_levels", "step_two", "addendum_document", "freeze_records", "_refuse_unless",
                       "check_addendum")),
    # step 3 (entry 106): the Clopper-Pearson margin, the selection and its repairs, the validation, the documents, the
    # loader and the addendum's check
    *(_U + n for n in ("binom_cdf", "cp_max_count", "cp_grid_alpha", "cp_alpha_primes", "cp_repair", "step3_scope",
                       "step3_select", "step3_candidates", "step_three", "step3_alpha_document", "step3_document",
                       "_refuse3", "_monotone", "load_step3", "check_step3", "load_step2", "_read3", "step2_report",
                       "load_final", "step3_configurations_document", "read_step3_configurations",
                       "step3_selection_arrays", "check_selection", "_check_step3_records", "repair_records",
                       "_refuse_selection", "_check_selection_records")),
    *(_C + n for n in ("_grid_alpha", "_snap")),
    *(_K + n for n in ("_configurations", "_weighted", "_producers", "_weighted_primary", "_doctored",
                       "_freeze_registries", "multiplicity_answers", "_step3_pass", "_step3_producers",
                       "_step3_selection_pass")))
MULTIPLICITY_CONSTANTS = (*(_U + n for n in ("REGISTERED_WEIGHTS", "PROCEDURE", "NOMINAL", "ALPHA_SCHEMA",
                                             "USABLE_SCHEMA", "FREEZE_SCHEMA", "ADDENDUM", "RULES", "_PLAIN_HOLM",
                                             "NO_TABLE", "CP_CONFIDENCE", "STEP3_ALPHA_SCHEMA", "STEP3_SCHEMA",
                                             "STEP3_RULES", "STEP3_CONFIGURATIONS_SCHEMA")),
                          *(_C + n for n in ("CLAIMS", "GRID_STEPS", "FACTOR_STEPS", "N1", "N2", "SEEDS", "SEED_SCHEME",
                                             "N3", "N3_SELECTION")),
                          _P + "STEP3_SEEDS", _P + "STEP2_PROCEDURE_DIGEST",
                          *(_K + n for n in ("KAT_FINGERPRINT", "KAT_CONFIGURATIONS", "KAT_PROCEDURE", "KAT_NO_OUTPUTS", "KAT_DATE",
                                             "KAT_PRODUCER_N", "KAT_PRODUCER_CHUNK", "KAT_INFERENCE",
                                             "KAT_STEP3_FINGERPRINT", "KAT_NO_STEP3_OUTPUTS", "KAT_STEP3_DATE")),
                          *(_K + n for n in ("KAT_WEIGHTED_STEP1", "KAT_WEIGHTED_P", "KAT_WEIGHTED_TABLE",
                                             "KAT_DOCTORED", "KAT_HOLM_WEIGHTS", "KAT_HOLM_ROWS",
                                             "KAT_C3_EQUIVALENCE")))
ALL_FUNCTIONS = (*INFERENCE_FUNCTIONS, *GROUPING_FUNCTIONS, *MULTIPLICITY_FUNCTIONS)
# The rebinding approved after the calibration (change-log entry 52): in prereg the contrasts' name studentise, bound to
# stats.studentise at 87c7126, is now the cluster dispatcher, which on a question-resample matrix is stats.studentise
# itself (tests). The calibrated digest records that one target as the calibrated function; the grouping digest records
# every binding as it is, so the dispatcher's own identity and source are bound there.
REBOUND = {_P + "studentise": _S + "studentise"}
ALL_CONSTANTS = (*INFERENCE_CONSTANTS, *GROUPING_CONSTANTS, *MULTIPLICITY_CONSTANTS)
# (caller, attribute): why that attribute read is never an in-repo class member there, although a class under harness/
# has a member of that name (checks.outcomes.OutputReport.get, checks.closure.Requirement.any, ...); or (caller, a
# run-time lookup): why it is safe there
ALLOWED: dict[tuple[str, str], str] = {
    (_S + "betainc", "any"): "inner.any(): inner is a boolean ndarray (x > 0) & (x < 1)",
    (_S + "QuestionIndex", "get"): "self.__dict__.get: the instance dict",
    (_P + "c5", "get"): "fam.get: fam is the dict cfg.families returns (PreregConfig.families, registered)",
    (_P + "source_rates", "mean"): "_f_q(...).mean(): _f_q returns an ndarray",
    (_P + "Calibration", "get"): "data, fam and e are dicts parsed from the table's JSON, self.claims a dict of them",
    (_P + "Clusters", "group"): "self.group: Clusters' own field, an ndarray of cluster positions",
    (_P + "ClusterMap", "add"): "templates.setdefault(cid, set()).add: a set of template names",
    (_P + "ClusterMap", "get"): "data.get: data is the dict parsed from the map's JSON",
    (_P + "PreregConfig", "get"): "self.__dict__.get and self.questions_sha256.get: the instance dict, a dict setting",
    (_I + "resolve", "getattr"): "it resolves the registry names, the digested constants INFERENCE_FUNCTIONS, ...",
    (_P + "preflight", "to_json"): "cfg.to_json and fresh.to_json: PreregConfigs (PreregConfig is registered whole)",
    (_P + "confirmatory", "to_json"): "frozen.to_json: the preflight's PreregConfig (registered whole)",
    # the walk behind the bindings: introspection of the registered objects is its job
    (_I + "bound_names", "add"): "pairs.add: a set",
    (_I + "bound_names", "get"): "rebound.get: a dict",
    (_I + "bound_names", "vars"): "vars(obj): the registered class's own members",
    (_I + "dataclass_generated", "vars"): "vars(cls) and vars(probe): the class's own members and the probe's",
    (_L + "curated_generic_rows", "get"): "r.get, settings.get, s.get, rep.get: dicts parsed from the run records' JSON",
    (_L + "curated_generic_rows", "add"): "verified.add: a set",
    (_I + "dataclass_generated", "getattr"): "getattr(params, n): the class's dataclass parameters, by the names of the "
                                             "decorator's own parameters",
    (_I + "_Walk", "add"): "the Reach's sets",
    (_I + "_Walk", "get"): "vars(self.module).get: the module's globals",
    (_I + "_Walk", "id"): "ast.Name.id",
    (_I + "_Walk", "scopes"): "self.scopes: the walk's own list",
    (_I + "_Walk", "getattr"): "the module attribute or class member the walked code names, and a value's __module__",
    (_I + "_Walk", "vars"): "vars(self.module) and vars(c): the walked module's globals, a class's own members",
    (_I + "_describe", "getattr"): "a bound object's __qualname__, __name__ and __module__",
    (_I + "_import_bindings", "getattr"): "the object an import inside the walked code binds",
    (_I + "_bindings", "id"): "ast.Name.id",
    # the weighted Holm's tables (entries 74, 76)
    (_U + "WeightedCalibration", "get"): "self.claims.get and entry.get: the dicts the tables were read into",
    (_U + "_read", "get"): "data.get: the dict parsed from the file's JSON",
    (_U + "load_files", "get"): "the dicts parsed from the two files' JSON",
    (_U + "table_document", "get"): "(step1 or {}).get(fam): a dict of each family's step-1 record",
    (_U + "calibrated_matrix", "get"): "tests.get(c): a dict of each claim's test",
    (_U + "step_two", "get"): "usable.get(c): a dict of each claim's usable alpha'",
    (_U + "check_addendum", "get"): "the dicts parsed from the addendum's and the table's JSON, and the freeze registry's "
                                    "records (dicts parsed from its JSON)",
    (_U + "freeze_records", "get"): "data.get: the dict parsed from the freeze registry's JSON",
    (_K + "_weighted", "get"): "(usable or {}).get(c, ...): a dict of hand usable levels",
    # the known-answer test
    (_P + "bounded", "get"): "{...}.get(x['sided'], ()): the sides each sidedness computes, a dict literal",
    (_P + "check_known_answers", "get"): "now.get and then.get: kat.environment()'s dict and the config's kat_environment",
    (_K + "_uniforms", "digest"): "hashlib.sha256(...).digest()",
    (_K + "_statistics", "to_json"): "key.to_json(): the ArmKeys of the KAT's arms (ArmKey is registered whole)",
    (_K + "_config_answers", "to_json"): "two.to_json and loaded.to_json: PreregConfigs (registered whole)",
    (_K + "_map_bytes", "get"): "anchors.get: a dict of the declared anchors",
    (_K + "grouping_answers", "get"): "KAT_ANCHORS.get: a dict constant (registered)",
    (_K + "environment", "get"): "the dicts numpy.show_config(mode='dicts') returns",
    (_K + "_openblas", "getattr"): "OpenBLAS's functions in the loaded library, by their exported names, with a default",
}
# Name collisions that appeared when the integration merged other branches (feat/fault-bank's _Reader.values and
# .resolve, feat/baselines' and feat/fault-bank's .label, feat/checker-loop's Arms.names, feat/a15's Vocab.names and
# .use): none of these reads is one of those members, and the digests are unchanged by the merge. Kept apart so a check
# against the calibrated commit, where these members do not exist, can tell them from its own allowances.
INTEGRATION_ALLOWED: dict[tuple[str, str], str] = {
    (_S + "resample_matrix", "values"): "strata.values(): a dict of index arrays",
    (_P + "c5", "values"): "fam.values(): the dict cfg.families returns",
    (_P + "run_contrasts", "values"): "c['at_least_min_effect'].values(): a dict of booleans",
    (_P + "check_known_answers", "values"): "variables.values(): a dict of environment variables",
    (_P + "zero_spread", "values"): "counts.values(): a dict of resample counts",
    (_P + "PreregConfig", "values"): "self.holm_weights.values(): a dict setting",
    (_P + "PreregConfig", "resolve"): "Path(path).resolve(): pathlib",
    (_P + "confirmatory", "values"): "arms.values(): a dict of unit lists",
    (_P + "confirmatory", "label"): "k.label and frozen.checker(...).label: ArmKeys (ArmKey is registered whole)",
    (_P + "repo_root", "resolve"): "Path(__file__).resolve(): pathlib",
    (_K + "_statistics", "label"): "key.label: the ArmKeys of the KAT's arms (ArmKey is registered whole)",
    (_K + "calibrated_answers", "resolve"): "Path(tmp).resolve(): pathlib",
    (_K + "grouping_answers", "resolve"): "Path(tmp).resolve(): pathlib",
    (_K + "multiplicity_answers", "resolve"): "Path(tmp).resolve(): pathlib",
    (_U + "load_files", "values"): "ap.values(): a dict parsed from JSON",
    (_U + "weighted_family", "values"): "fam['at_least_min_effect'].values(): a dict of booleans",
    ("harness.analysis.metrics.ArmKey", "use"): "self.use: ArmKey's own field",
    (_I + "_bindings", "names"): "node.names: an ast Global or Nonlocal node's names",
    (_I + "_import_bindings", "names"): "node.names: an ast Import or ImportFrom node's aliases",
    # feat/fault-bank's _Reader.index and .values, met again by feat/analysis's step-1 and step-2 code (f95834e)
    (_U + "calibrated_matrix", "index"): "cal.CLAIMS.index and prereg.TESTS.index: tuple.index",
    (_U + "step_one_table", "index"): "cal.CLAIMS.index: tuple.index",
    (_U + "step_two", "index"): "cal.CLAIMS.index and prereg.TESTS.index: tuple.index",
    (_U + "step_two", "values"): "out.values(): a dict of families",
    # step 3 (entry 106): the dicts its documents parse to, tuple.index, a dict of families
    (_U + "step3_candidates", "get"): "the selection's and the table's dicts (parsed from their JSON)",
    (_U + "step_three", "index"): "cal.CLAIMS.index and prereg.TESTS.index: tuple.index",
    (_U + "step_three", "values"): "out.values(): a dict of families",
    (_U + "load_step3", "get"): "the step-3 addendum's dicts (parsed from its JSON)",
    (_U + "check_step3", "get"): "the step-3 addendum's dicts and the freeze registry's records (parsed from JSON)",
    (_U + "_read3", "get"): "data.get: the dict parsed from the file's JSON",
    (_K + "_step3_producers", "index"): "calibration.CLAIMS.index and prereg.TESTS.index: tuple.index",
    # the review of 1e31f1f: the bound configurations, the selection's check and the records' re-derivation
    (_U + "read_step3_configurations", "get"): "data.get: the dict parsed from the file's JSON",
    (_U + "step3_selection_arrays", "index"): "cal.CLAIMS.index and prereg.TESTS.index: tuple.index",
    (_U + "check_selection", "get"): "the selection's dicts, the frozen outputs and the registry's records (JSON)",
    (_U + "check_selection", "values"): "frozen12.values(): a dict of the steps' outputs",
    (_U + "_check_step3_records", "get"): "the step-3 addendum's and the selection's dicts (parsed from their JSON)",
    (_U + "_check_selection_records", "get"): "the selection's dicts (parsed from its JSON)",
    (_U + "_check_selection_records", "values"): "ok.values() and levels.values(): the selection's dicts",
    (_K + "_step3_selection_pass", "get"): "the table's alpha' dict (parsed from its JSON)",
    (_K + "_step3_producers", "values"): "smallest[\"families\"].values(): a dict of families",
    # the one-graph census's Queries.templates and the query cache's CachingClient.key (867d073)
    (_P + "ClusterMap", "templates"): "self.templates: ClusterMap's own field, the question templates of the map's file",
    (_K + "grouping_answers", "templates"): "cmap.templates: a ClusterMap's field (ClusterMap is registered whole)",
    (_I + "_Walk", "key"): "node.key: an ast DictComp's key",
    # feat/query-gate-slots' QueryGate.level (change-log entry 133), met by the walk's own import resolution
    (_I + "_import_bindings", "level"): "node.level: an ast ImportFrom node's relative-import depth",
}
ALLOWED.update(INTEGRATION_ALLOWED)
# (caller, callee): in-repo functions a registered function calls that are not digested, by design: the data preparation
# and input refusals that read the attempt units, which confirmatory runs after the preflight with the caller's config
# (the inference uses the preflight's own copy). The calibration simulated question vectors directly. And the
# known-answer test's input container, whose contents the known answers bind.
BOUNDARY: dict[tuple[str, str], str] = {
    **{(_P + "confirmatory", _P + n): why for n, why in (
        ("_require_arms", "refuses inputs without the required arms"),
        ("_check_scorer", "refuses attempts scored by another scorer version"),
        ("_check_repetitions", "refuses arms without the fixed repetitions"),
        ("vectors_from_units", "the measurement: per-question vectors from the attempt units"),
        ("missing_shares", "each arm's missing share, from the attempt units (§4.3)"))},
    **{(_K + n, _P + "Vectors"): "the KAT's question vectors, built in kat_vectors: a dataclass of three dicts with no "
                                 "code of its own; the known answers bind what the inference computes from them"
       for n in ("kat_vectors", "_empty", "_statistics")},
    (_L + "curated_generic_rows", "harness.analysis.inputs.InputError"): (
        "the input refusal the check raises (a generic row whose record did not decide on curated sources alone): "
        "ValueError's subclass with no code of its own, as for ledger_source_names"),
    (_L + "ledger_source_names", "harness.analysis.inputs.InputError"): (
        "the input refusal the mapping raises (a ledger holding a name and the name it maps to): ValueError's subclass "
        "with no code of its own, unchanged since 87c7126, so in no registry of new code"),
    (_K + "_ledger_names", "harness.analysis.inputs.CheckedOutput"): (
        "the KAT's ledger rows: the mapping reads each row's source and replaces it; a row's own checks are the data "
        "preparation's, and the known answers bind the names the mapping gives"),
}
REPO_PACKAGE = "harness."
# a tuple, not a set: a registered constant's repr is digested, and a set's order varies between processes
_DYNAMIC = ("__import__", "compile", "delattr", "eval", "exec", "getattr", "globals", "hasattr", "locals", "setattr", "vars")


def resolve(qualname: str) -> Any:
    """The object a registry name names: the longest importable module prefix, then its attributes (a class member as
    the class defines it: the property or classmethod object, not its value)."""
    parts = qualname.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for name in parts[i:]:
            obj = inspect.getattr_static(obj, name) if isinstance(obj, type) else getattr(obj, name)
        return obj
    raise ImportError(f"no importable module in {qualname!r}")


def _code(obj: Any) -> list[Any]:
    """The functions holding a registered object's code: a property's accessors, a class or static method's function,
    else the object itself."""
    if isinstance(obj, property):
        return [f for f in (obj.fget, obj.fset, obj.fdel) if f is not None]
    if isinstance(obj, classmethod | staticmethod):
        return [obj.__func__]
    return [obj]


def _source(obj: Any) -> str:
    return "".join(inspect.getsource(f) for f in _code(obj)).replace("\r\n", "\n").replace("\r", "\n")


def source_digest(qualname: str) -> str:
    """The sha256 of one registered object's source."""
    return hashlib.sha256(_source(resolve(qualname)).encode("utf-8")).hexdigest()


def _digest(functions: tuple[str, ...], constants: tuple[str, ...], bound: tuple[str, ...],
            rebound: dict[str, str]) -> str:
    payload = {"functions": list(functions), "constants": list(constants),
               "sources": [source_digest(q) for q in functions], "values": [repr(resolve(q)) for q in constants],
               "bindings": [bound_names(q, rebound) for q in bound]}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def inference_digest() -> str:
    """sha256 of the calibrated registry (both lists, in order), each object's source, each constant's repr, and what
    each object's global names are bound to (``bound_names``, with the approved rebinding REBOUND as calibrated)."""
    return _digest(INFERENCE_FUNCTIONS, INFERENCE_CONSTANTS, INFERENCE_FUNCTIONS, REBOUND)


def grouping_digest() -> str:
    """The same over the grouping registry (the post-calibration code between the data and the calibrated functions),
    with the bindings of every registered object of both registries as they are."""
    return _digest(GROUPING_FUNCTIONS, GROUPING_CONSTANTS, ALL_FUNCTIONS, {})


def multiplicity_digest() -> str:
    """The same over the multiplicity registry (the weighted Holm, entries 74 and 76), with its objects' bindings."""
    return _digest(MULTIPLICITY_FUNCTIONS, MULTIPLICITY_CONSTANTS, MULTIPLICITY_FUNCTIONS, {})


def function_digests(functions: tuple[str, ...] = INFERENCE_FUNCTIONS) -> dict[str, str]:
    """Each registered object's source sha256 (to name what changed when a digest differs)."""
    return {q: source_digest(q) for q in functions}


def constant_values(constants: tuple[str, ...] = INFERENCE_CONSTANTS) -> dict[str, str]:
    """Each registered constant's repr, as hashed."""
    return {q: repr(resolve(q)) for q in constants}


# --- the closure walk ----------------------------------------------------------------------------


@dataclass
class Reach:
    """What a piece of code reaches: in-repo functions, classes and class members (by defining module and qualified
    name), module constants (by the module they are read from), the names of attributes read on other receivers, and
    run-time name lookups."""

    functions: set[str] = field(default_factory=set)
    constants: set[str] = field(default_factory=set)
    attributes: set[str] = field(default_factory=set)
    dynamic: set[str] = field(default_factory=set)
    bindings: set[tuple[str, str]] = field(default_factory=set)  # (a module-level name it reads, what it is bound to)


def _describe(value: Any) -> str:
    """What a name is bound to, deterministically: a module by its name, a callable by its defining module and qualified
    name, anything else by its type (a registered constant's value is digested as such)."""
    if isinstance(value, types.ModuleType):
        return f"module {value.__name__}"
    if callable(value):
        name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None) or type(value).__qualname__
        return f"{getattr(value, '__module__', None)}.{name}"
    return f"value of type {type(value).__module__}.{type(value).__qualname__}"


_LOCAL = object()  # a name bound in a function, comprehension or class body (a parameter, an assignment, a def)


@dataclass(frozen=True)
class _Owner:
    """``cls`` in a classmethod: the class it is defined on. Reading it constructs or names that class (a reference to
    the whole class); its attributes stay names, since at run time it may be a subclass."""

    cls: type


@dataclass(frozen=True)
class _Imported:
    """A name bound by an import inside the code: the object, and the module and name it was imported as."""

    obj: Any
    owner: str
    name: str


def _import_bindings(node: ast.Import | ast.ImportFrom, package: str | None) -> dict[str, _Imported]:
    if isinstance(node, ast.Import):
        out = {}
        for alias in node.names:
            name = alias.name if alias.asname else alias.name.split(".")[0]
            out[alias.asname or name] = _Imported(importlib.import_module(name), name, name)
        return out
    owner = importlib.import_module("." * node.level + (node.module or ""), package)
    out = {}
    for alias in node.names:
        try:
            obj = getattr(owner, alias.name)
        except AttributeError:
            obj = importlib.import_module(f"{owner.__name__}.{alias.name}")
        out[alias.asname or alias.name] = _Imported(obj, owner.__name__, alias.name)
    return out


def _bindings(body: list[ast.AST], package: str | None) -> dict[str, Any]:
    """The names a scope binds (assigned anywhere in it, outside nested scopes; global and nonlocal ones excluded),
    with what an import binds."""
    names: dict[str, Any] = {}
    declared: set[str] = set()
    todo = list(body)
    while todo:
        node = todo.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names[node.name] = _LOCAL
            todo += node.decorator_list  # the rest is evaluated here but binds nothing, or is the nested scope
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
            todo.append(node.generators[0].iter)  # the rest is the comprehension's own scope
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            names[node.id] = _LOCAL
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names.update(_import_bindings(node, package))
        elif isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar) and node.name:
            names[node.name] = _LOCAL
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names[node.rest] = _LOCAL
        elif isinstance(node, ast.Global | ast.Nonlocal):
            declared.update(node.names)
        todo += ast.iter_child_nodes(node)
    for name in declared:
        names.pop(name, None)
    return names


def _arguments(args: ast.arguments) -> dict[str, Any]:
    return {a.arg: _LOCAL for a in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
            if a is not None}


class _Walk(ast.NodeVisitor):
    """The walk behind ``scan``: Python's name resolution (function, comprehension and class scopes; a class body's
    names are invisible in its methods), annotations not read."""

    def __init__(self, module: types.ModuleType, owner: type | None = None) -> None:
        self.module = module
        self.reach = Reach()
        self.scopes: list[tuple[dict[str, Any], bool]] = []  # (bindings, is a class body), innermost last
        self.direct_class = owner  # the class whose body the walk is directly in (a member's class at the start)

    def _lookup(self, name: str) -> tuple[bool, Any]:
        """(True, binding) for a local or module-global name, (False, None) for a builtin or an unknown one."""
        for depth, (names, is_class) in enumerate(reversed(self.scopes)):
            if is_class and depth > 0:
                continue
            if name in names:
                return True, names[name]
        if name in vars(self.module):
            return True, vars(self.module)[name]
        return False, None

    def _static(self, node: ast.AST) -> Any:
        """The object a name or a chain of module attributes denotes, if known without running the code."""
        if isinstance(node, ast.Name):
            found, value = self._lookup(node.id)
            if not found or value is _LOCAL:
                return None
            if isinstance(value, _Owner):
                return None
            obj = value.obj if isinstance(value, _Imported) else value
            self.reach.bindings.add((node.id, _describe(obj)))
            return obj
        if isinstance(node, ast.Attribute):
            base = self._static(node.value)
            if isinstance(base, types.ModuleType):
                return getattr(base, node.attr, None)
        return None

    def _reference(self, value: Any, name: str, owner: str) -> None:
        """A name read: an in-repo function or class, or a constant (named by ``owner``, the module it is read from)."""
        if isinstance(value, types.ModuleType):
            return
        if callable(value):
            module = getattr(value, "__module__", None) or ""
            if module.startswith(REPO_PACKAGE):
                self.reach.functions.add(f"{module}.{getattr(value, '__qualname__', name)}")
        elif not name.startswith("__"):
            self.reach.constants.add(f"{owner}.{name}")

    def _member(self, cls: type, attr: str) -> None:
        """``Cls.attr`` for an in-repo class named in the code: the member where the class's MRO defines it."""
        for c in cls.__mro__:
            if attr in vars(c):
                if (c.__module__ or "").startswith(REPO_PACKAGE):
                    value, qual = vars(c)[attr], f"{c.__module__}.{c.__qualname__}.{attr}"
                    if callable(value) or isinstance(value, property | classmethod | staticmethod):
                        self.reach.functions.add(qual)
                    elif not attr.startswith("__"):
                        self.reach.constants.add(qual)
                return
        self.reach.attributes.add(attr)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return
        found, value = self._lookup(node.id)
        if not found:
            if node.id in _DYNAMIC:
                self.reach.dynamic.add(node.id)
        elif isinstance(value, _Imported):
            self.reach.bindings.add((node.id, _describe(value.obj)))
            self._reference(value.obj, value.name, value.owner)
        elif isinstance(value, _Owner):
            self._reference(value.cls, node.id, self.module.__name__)
        elif value is not _LOCAL:
            self.reach.bindings.add((node.id, _describe(value)))
            self._reference(value, node.id, self.module.__name__)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            base = self._static(node.value)
            if isinstance(base, types.ModuleType):
                if base.__name__.startswith(REPO_PACKAGE):
                    self._reference(getattr(base, node.attr, None), node.attr, base.__name__)
                return  # an external module's attribute (np.where)
            if isinstance(base, type) and (base.__module__ or "").startswith(REPO_PACKAGE):
                self._member(base, node.attr)
                return
            self.reach.attributes.add(node.attr)
        self.visit(node.value)

    def _body(self, names: dict[str, Any], body: list[ast.AST], is_class: bool = False) -> None:
        self.scopes.append((names, is_class))
        for stmt in body:
            self.visit(stmt)
        self.scopes.pop()

    def _defaults(self, args: ast.arguments) -> None:
        for d in [*args.defaults, *[d for d in args.kw_defaults if d is not None]]:
            self.visit(d)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for d in node.decorator_list:
            self.visit(d)
        self._defaults(node.args)  # annotations (arguments and return) are not read
        names = {**_arguments(node.args), **_bindings(node.body, self.module.__package__)}
        owner, first = self.direct_class, [*node.args.posonlyargs, *node.args.args][:1]
        if owner is not None and first and any(isinstance(d, ast.Name) and d.id == "classmethod"
                                               for d in node.decorator_list):
            names[first[0].arg] = _Owner(owner)
        self.direct_class = None  # a function's body is no class body
        self._body(names, node.body)
        self.direct_class = owner

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._defaults(node.args)
        owner, self.direct_class = self.direct_class, None
        self._body({**_arguments(node.args), **_bindings([node.body], self.module.__package__)}, [node.body])
        self.direct_class = owner

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for n in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(n)
        outer = self.direct_class
        if outer is not None:
            cls = getattr(outer, node.name, None)
        else:
            cls = vars(self.module).get(node.name) if not self.scopes else None  # a class defined in a function: unknown
        self.direct_class = cls if isinstance(cls, type) else None
        self._body(_bindings(node.body, self.module.__package__), node.body, is_class=True)
        self.direct_class = outer

    def _comprehension(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> None:
        first, *rest = node.generators
        self.visit(first.iter)  # evaluated in the enclosing scope
        names = {n.id: _LOCAL for g in node.generators for n in ast.walk(g.target) if isinstance(n, ast.Name)}
        parts = [*first.ifs, *[x for g in rest for x in (g.iter, *g.ifs)],
                 *((node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,))]
        self._body(names, parts)

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comprehension

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_arg(self, node: ast.arg) -> None:
        pass  # its annotation is not read


def scan(source: str, module: types.ModuleType, owner: type | None = None) -> Reach:
    """What ``source`` (a function or class definition) reaches, its names resolved in ``module``'s globals; ``owner``
    is the class a member's source is defined on (for ``cls`` in a classmethod)."""
    walk = _Walk(module, owner)
    walk.visit(ast.parse(textwrap.dedent(source)))
    return walk.reach


def _module_name(obj: Any) -> str:
    return _code(obj)[0].__module__


def references(qualname: str) -> Reach:
    """What a registered object's code reaches."""
    obj = resolve(qualname)
    try:
        owner = resolve(qualname.rpartition(".")[0])
    except (ImportError, AttributeError):
        owner = None
    return scan(_source(obj), importlib.import_module(_module_name(obj)), owner if isinstance(owner, type) else None)


def dataclass_generated(cls: type) -> frozenset[str]:
    """The members of ``cls`` the dataclass machinery generated, decided from the class's own dataclass fields and
    parameters, never from a list of names: a probe dataclass made with the same fields and parameters and no body of
    its own shows which members this Python's machinery adds (``__replace__`` since 3.13, a frozen class's
    ``__setattr__``, ...). A member of ``cls`` of one of those names is the machinery's when it is a function that is
    the very object the probe got (a shared helper such as dataclasses' ``_replace``) or whose code, behind its
    wrappers, was compiled from a string (the generated methods, the recursive-repr wrapper around ``__repr__``). A
    member the class body defines, or one replaced after the body, is neither, so it stays bound. Empty for a class the
    decorator did not run on (a subclass inherits the parameters, not the members)."""
    own = vars(cls)
    if "__dataclass_params__" not in own:
        return frozenset()
    params = own["__dataclass_params__"]
    takes = set(inspect.signature(dataclasses.make_dataclass).parameters)
    settings = {n: getattr(params, n) for n in inspect.signature(dataclasses.dataclass).parameters if n in takes}
    specs = [(f.name, f.type, dataclasses.field(default=f.default, default_factory=f.default_factory, init=f.init,
                                                 repr=f.repr, hash=f.hash, compare=f.compare, metadata=f.metadata,
                                                 kw_only=f.kw_only))
             for f in dataclasses.fields(cls)]
    probe = vars(dataclasses.make_dataclass(cls.__name__, specs, **settings))
    return frozenset(name for name, made in probe.items()
                     if name in own and isinstance(own[name], types.FunctionType)
                     and (own[name] is made or inspect.unwrap(own[name]).__code__.co_filename.startswith("<")))


def bound_names(qualname: str, rebound: dict[str, str] | None = None) -> list[list[str]]:
    """What a registered object's module-level names are bound to (``Reach.bindings``, each target mapped through
    ``rebound``), and for a class registered whole what each of its members is (a method's defining module and
    qualified name, another value's repr), so that an import alias, a module-level assignment or a class member replaced
    after its class body changes the digest although no registered source does. Members the dataclass machinery
    generates (``dataclass_generated``) follow from the class's source, fields and parameters, and which of them exist
    depends on the Python (3.13 added ``__replace__``), so they are left out and the digest is the same under every
    Python (design ruling, design log, 20 Oct); so are other methods compiled from a string."""
    rebound = rebound or {}
    pairs = {(name, rebound.get(target, target)) for name, target in references(qualname).bindings}
    obj = resolve(qualname)
    if isinstance(obj, type):
        generated = dataclass_generated(obj)
        for name, value in vars(obj).items():
            if name in generated:
                continue
            if isinstance(value, types.FunctionType | property | classmethod | staticmethod):
                pairs |= {(f".{name}", _describe(f)) for f in _code(value) if not f.__code__.co_filename.startswith("<")}
            elif callable(value):
                pairs.add((f".{name}", _describe(value)))
            elif not name.startswith("__"):
                text = repr(value)
                pairs.add((f".{name}", _describe(value) if " at 0x" in text else f"value {text}"))
    return sorted([name, target] for name, target in pairs)


@cache
def class_members() -> dict[str, frozenset[str]]:
    """Attribute name -> every in-repo class member of that name (module.Class.member): the methods, properties and
    class constants of every class defined at the top level of a module under harness/ (read from the source, nothing
    imported)."""
    spec = importlib.util.find_spec(REPO_PACKAGE.rstrip("."))
    base = Path(next(iter(spec.submodule_search_locations)))
    out: dict[str, set[str]] = {}

    def members(cls: ast.ClassDef, qual: str) -> None:
        for stmt in cls.body:
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                out.setdefault(stmt.name, set()).add(f"{qual}.{stmt.name}")
            elif isinstance(stmt, ast.Assign):
                for n in (n for t in stmt.targets for n in ast.walk(t) if isinstance(n, ast.Name)):
                    out.setdefault(n.id, set()).add(f"{qual}.{n.id}")
            elif isinstance(stmt, ast.ClassDef):
                members(stmt, f"{qual}.{stmt.name}")

    for path in sorted(base.rglob("*.py")):
        parts = path.relative_to(base.parent).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef):
                members(node, f"{module}.{node.name}")
    return {k: frozenset(v) for k, v in out.items()}


def _covered(name: str, registered: frozenset[str]) -> bool:
    return name in registered or any(name.startswith(r + ".") for r in registered)


def unclosed(functions: tuple[str, ...] | list[str] = ALL_FUNCTIONS,
             constants: tuple[str, ...] | list[str] = ALL_CONSTANTS) -> list[tuple[str, str, str]]:
    """What the registered ``functions`` reach that is not registered, as (caller, kind, target). Kinds: "call" (an
    in-repo function, class or class member read by name), "constant", "attribute" (an unregistered in-repo member with
    the name of an attribute read on a receiver of unknown class), "dynamic" (a run-time name lookup), and "allowance"
    (an ALLOWED entry for a registered caller that is no longer needed)."""
    registered, consts = frozenset(functions), frozenset(constants)
    members = class_members()
    out: list[tuple[str, str, str]] = []
    allowed_used = set()
    for q in functions:
        r = references(q)
        for f in sorted(r.functions):
            if (q, f) in BOUNDARY:
                allowed_used.add((q, f))
            elif not _covered(f, registered):
                out.append((q, "call", f))
        out += [(q, "constant", c) for c in sorted(r.constants) if c not in consts and not _covered(c, registered)]
        for a in sorted(r.attributes):
            loose = sorted(m for m in members.get(a, ()) if not _covered(m, registered))
            if loose and (q, a) in ALLOWED:
                allowed_used.add((q, a))
            else:
                out += [(q, "attribute", m) for m in loose]
        for d in sorted(r.dynamic):
            if (q, d) in ALLOWED:
                allowed_used.add((q, d))
            else:
                out.append((q, "dynamic", d))
    out += [(q, "allowance", a) for q, a in (*ALLOWED, *BOUNDARY) if q in registered and (q, a) not in allowed_used]
    return out
