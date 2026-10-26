"""What a check returns, and how an output's status follows from its checks
(S0.1 check contract v2, "Outcomes and output status").

Every check returns one of four outcomes with a machine-readable reason:

* pass            the constraint holds on what the check tested; a pass is
                  substantive unless the query itself already guarantees it
                  (rule 18: reasons "tautological", "vacuous");
* flag            the slice violates the constraint as written (closed world);
* unknown         the check applies, but the runtime evidence it needs is
                  missing (an untyped node, the bound, an unsupported query
                  construct, untyped census fillers);
* not_applicable  nothing to test: no numbers, or no axiom of that kind for the
                  terms used (including axioms the compiler cannot compile,
                  reason ``unsupported_axiom``).

Output status, over the ontology-derived checks (number traceability is its own
layer, rule 8, and never enters it):

* flagged          at least one check flags;
* substantive_pass no flag, and at least one substantive pass;
* unknown          no flag, no substantive pass, at least one unknown;
* vacuous_pass     none of the above.

Two switches can be replayed off (rule 23): the no-common-subclass rule (its own
check id) and branch-local flags (conflicts that only make an OPTIONAL group or
a negated part impossible, rules 7 and 17).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Outcome(str, Enum):
    PASS = "pass"
    FLAG = "flag"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


# Check ids. The check type groups them for reporting (rule 9 keeps counts per
# check type); the form says which layer produced them.
CHECKS: dict[str, tuple[str, str]] = {
    "vocabulary.static": ("vocabulary", "static"),
    "class.static": ("class", "static"),
    "domain_range.static": ("domain_range", "static"),
    "disjointness.static": ("disjointness", "static"),
    "no_common_subclass.static": ("no_common_subclass", "static"),
    "census_join.static": ("census", "static"),
    "census_pattern.static": ("census", "static"),
    "class.footprint": ("class", "footprint"),
    "domain_range.footprint": ("domain_range", "footprint"),
    "disjointness.footprint": ("disjointness", "footprint"),
    "number_traceability": ("number_traceability", "n/a"),
    "provenance": ("provenance", "footprint"),
    # OBQC as published (harness/baselines/obqc.py, design log): its eight rules, each its own check id. Rules 1-5
    # test declared domains and ranges; rule 6 vocabulary; rules 7-8 the SELECT clause.
    "obqc.domain": ("obqc_axiom", "static"),
    "obqc.range": ("obqc_axiom", "static"),
    "obqc.double_range": ("obqc_axiom", "static"),
    "obqc.double_domain": ("obqc_axiom", "static"),
    "obqc.domain_range": ("obqc_axiom", "static"),
    "obqc.incorrect_property": ("obqc_vocabulary", "static"),
    "obqc.iri_output": ("obqc_output", "static"),
    "obqc.subject_output": ("obqc_output", "static"),
    # Retry on empty's trigger (harness/baselines/empty.py), named in its bank rows and ledger rows: an
    # execution check, not ontology-derived; no compiler runs it.
    "empty_result": ("empty_result", "n/a"),
}
HEURISTIC_CHECK = "no_common_subclass.static"
# Ontology-independent layer (rule 8): reported on its own, never charged to a
# source, never a reason to retry.
INDEPENDENT_CHECKS = {"number_traceability"}

# Check types that count towards substantive coverage: ontology-derived, other
# than vocabulary existence.
SUBSTANTIVE_TYPES = {"class", "domain_range", "disjointness", "no_common_subclass", "census", "provenance",
                     "obqc_axiom"}
# Passes the query itself guarantees (rule 18): reported, never substantive.
NON_SUBSTANTIVE_REASONS = {"tautological", "vacuous", "branch_only_conflict_reported"}

# Rule 10's data-fault kinds (clarified 20 Oct, change-log entry 108): a witnessed data fault is a closed-world
# mismatch (a class membership the data does not assert, which no disjointness axiom excludes) or a logical clash (a
# witnessed disjointness or datatype clash). A reporting label in the flag's evidence (``data_fault_kind``), carried
# beside ``data_fault`` into the verdict's fired checks (on data faults only) and so the replay matrix; no verdict reads
# it.
CLOSED_WORLD_MISMATCH = "closed_world_mismatch"
LOGICAL_CLASH = "logical_clash"
MIXED_FAULT = "mixed"  # both kinds witnessed among one check's flagged nodes
FAULT_KINDS = (CLOSED_WORLD_MISMATCH, LOGICAL_CLASH, MIXED_FAULT)  # a curated or generated source's
# A census has no axioms, so rule 10 gives its data faults no kind (design ruling in the design log Oct 21:43): the label
# its witnessed data faults carry instead, whatever the flag
NOT_AXIOM = "not_axiom"


def fault_kind(kinds: Iterable[str]) -> str | None:
    """One label for the kinds witnessed: None for none, the kind when there is one, MIXED_FAULT for both. A single
    string, so a fired check stays hashable (the verdict dedupes them as tuples)."""
    found: set[str] = set()
    for k in kinds:
        if k not in FAULT_KINDS:
            raise ValueError(f"not a rule-10 data-fault kind: {k!r}")
        found |= {CLOSED_WORLD_MISMATCH, LOGICAL_CLASH} if k == MIXED_FAULT else {k}
    if not found:
        return None
    return MIXED_FAULT if len(found) > 1 else found.pop()


# Reason codes. Every result carries one (rule 12).
REASONS: dict[Outcome, set[str]] = {
    Outcome.PASS: {"holds", "no_terms", "tautological", "vacuous", "branch_only_conflict_reported"},
    Outcome.FLAG: {
        "not_in_source",
        "explicit_disjointness",
        "no_common_named_subclass",
        "outside_required_class",
        "wrong_datatype",
        "value_not_in_enumeration",
        "literal_where_class_required",
        "resource_where_literal_required",
        "literal_in_resource_position",
        "census_low_support",
        "census_empty_pattern",
        "disjoint_asserted_types",
        "number_not_in_rows",
        "no_provenance_path",
        "obqc_violation",  # an OBQC rule's published query returned this binding
    },
    Outcome.UNKNOWN: {
        "untyped_node",
        "blank_node",
        "types_outside_source",
        "bound",
        "unsupported_construct",
        "untyped_fillers",
        "optional_undetermined",
        "query_parse_error",
        "parser_mismatch",
        "query_failed",
        "no_store",
        "received_rows_unavailable",
    },
    Outcome.NOT_APPLICABLE: {
        "no_axiom",
        "vacuous_axiom",
        "unsupported_axiom",
        "no_constant",
        "no_type_pattern",
        "no_disjointness",
        "only_in_branch",
        "no_join",
        "not_census",
        "no_type_census",
        "graph_outside_census",
        "empty_result",
        "no_binding",
        "no_numbers",
        "no_provenance_model",
        "no_applicable_requirement",
    },
}


@dataclass
class CheckResult:
    check_id: str
    source: str
    outcome: Outcome
    reason: str
    key: str = "-"  # what the check is about: "?x", "subject-of <p>", "<c> subject-of <p>", ...
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    annotations: list[str] = field(default_factory=list)  # "open_world_accepts", "data_fault"
    branch_local: bool = False  # the conflict only makes an OPTIONAL group or a negated part impossible
    scope: str = "required"  # rule 28: "required", "optional" or "negated" part the static check read

    def __post_init__(self) -> None:
        if self.check_id not in CHECKS:
            raise ValueError(f"unknown check id {self.check_id!r}")
        self.outcome = Outcome(self.outcome)
        if self.reason not in REASONS[self.outcome]:
            raise ValueError(f"reason {self.reason!r} is not a {self.outcome.value} reason")

    @property
    def check_type(self) -> str:
        return CHECKS[self.check_id][0]

    @property
    def form(self) -> str:
        return CHECKS[self.check_id][1]

    @property
    def substantive(self) -> bool:
        return (self.check_type in SUBSTANTIVE_TYPES and self.outcome is Outcome.PASS
                and self.reason not in NON_SUBSTANTIVE_REASONS)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["outcome"] = self.outcome.value
        out["check_type"] = self.check_type
        out["form"] = self.form
        return out


_RANK = {Outcome.FLAG: 3, Outcome.UNKNOWN: 2, Outcome.PASS: 1, Outcome.NOT_APPLICABLE: 0}


def combine(outcomes: list[Outcome]) -> Outcome:
    """The strongest outcome: flag over unknown over pass over not applicable."""
    return max(outcomes, key=_RANK.__getitem__, default=Outcome.NOT_APPLICABLE)


@dataclass
class OutputReport:
    """All checks of one source on one output."""

    source: str
    role: str  # "agent" or "gold"
    results: list[CheckResult]
    negation: bool  # the query uses MINUS or NOT EXISTS: its own stratum
    unsupported_constructs: list[str]  # query constructs outside the frozen fragment (logged by the pilot)
    unsupported_coverage: dict[str, int]  # source constructs left out, plus rule 3 query-side counts
    evidence: dict[str, Any] = field(default_factory=dict)  # rerun query, fetch queries, evidence size
    checked_query: str = ""  # rule 22
    checked_query_status: str = "given"  # "aligned", "fallback", or "given" (not chosen from a trajectory)
    answer_rows_not_in_checked_result: int | None = None  # None when the checked query's rows are unknown
    tooling_limits: list[str] = field(default_factory=list)  # e.g. "parser_mismatch": the store ran it, rdflib cannot parse it
    # Change-log entry 72: the triple patterns a census source's checks could not judge because they are confined to a
    # graph neither the census nor the shared hierarchy describes (graph_outside_census); 0 for any other source.
    graph_outside_census: int = 0

    def enabled(self, heuristic: bool = True, branch_local: bool = True, branch_scopes: bool = True) -> list[CheckResult]:
        """The results under the switches: the no-common-subclass rule, branch-local
        flags (rule 23), and passes read in OPTIONAL-only or negated scopes (rule 28)."""
        return [
            r for r in self.results
            if (heuristic or r.check_id != HEURISTIC_CHECK) and (branch_local or not r.branch_local)
            and (branch_scopes or r.scope == "required" or r.outcome is not Outcome.PASS)
        ]

    def ontology(self, heuristic: bool = True, branch_local: bool = True, branch_scopes: bool = True) -> list[CheckResult]:
        """The source-attributed checks: everything but the independent layer."""
        return [r for r in self.enabled(heuristic, branch_local, branch_scopes) if r.check_id not in INDEPENDENT_CHECKS]

    def status(self, heuristic: bool = True, branch_local: bool = True, branch_scopes: bool = True) -> str:
        checks = self.ontology(heuristic, branch_local, branch_scopes)
        if any(r.outcome is Outcome.FLAG for r in checks):
            return "flagged"
        if any(r.substantive for r in checks):
            return "substantive_pass"
        if any(r.outcome is Outcome.UNKNOWN for r in checks):
            return "unknown"
        return "vacuous_pass"

    def any_unknown(self, heuristic: bool = True, branch_local: bool = True) -> bool:
        return any(r.outcome is Outcome.UNKNOWN for r in self.ontology(heuristic, branch_local))

    def retry(self, heuristic: bool = True, branch_local: bool = True) -> bool:
        """Rule 9: a retry only on a flag from an ontology-derived check."""
        return self.status(heuristic, branch_local) == "flagged"

    def flagged_checks(self, heuristic: bool = True, branch_local: bool = True) -> set[str]:
        """Every check id that flags, the independent layer included."""
        return {r.check_id for r in self.enabled(heuristic, branch_local) if r.outcome is Outcome.FLAG}

    def substantive(self, heuristic: bool = True, branch_local: bool = True, branch_scopes: bool = True) -> bool:
        return any(r.substantive for r in self.ontology(heuristic, branch_local, branch_scopes))

    def by_type(self, heuristic: bool = True, branch_local: bool = True) -> dict[str, dict[str, Any]]:
        """Per check type: the combined outcome and the per-check counts (rule 9)."""
        out: dict[str, dict[str, Any]] = {}
        for r in self.enabled(heuristic, branch_local):
            entry = out.setdefault(r.check_type, {"counts": {o.value: 0 for o in Outcome}, "outcomes": []})
            entry["counts"][r.outcome.value] += 1
            entry["outcomes"].append(r.outcome)
        for entry in out.values():
            entry["outcome"] = combine(entry.pop("outcomes")).value
        return out

    def get(self, check_id: str, key: str | None = None) -> list[CheckResult]:
        return [r for r in self.results if r.check_id == check_id and (key is None or r.key == key)]

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "role": self.role,
            "status": self.status(),
            "status_switches": {
                f"heuristic_{h}_branch_local_{b}": self.status(h, b) for h in (True, False) for b in (True, False)
            },
            "status_required_scope_passes_only": self.status(branch_scopes=False),
            "any_unknown": self.any_unknown(),
            "substantive": self.substantive(),
            "substantive_required_scope_only": self.substantive(branch_scopes=False),
            "negation": self.negation,
            "unsupported_constructs": self.unsupported_constructs,
            "tooling_limits": self.tooling_limits,
            "graph_outside_census": self.graph_outside_census,
            "checked_query": self.checked_query,
            "checked_query_status": self.checked_query_status,
            "answer_rows_not_in_checked_result": self.answer_rows_not_in_checked_result,
            "unsupported_coverage": self.unsupported_coverage,
            "evidence": self.evidence,
            "results": [r.to_json() for r in self.results],
        }
