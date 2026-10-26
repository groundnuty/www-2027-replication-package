"""An arm of the checker: the compiled checks of one source, or of several
together (the union arm, curated plus census: tests H3), and its verdict on one
output.

The verdict follows the S0.1 contract v2:
* the output status is four-way (flagged, substantive pass, unknown, vacuous
  pass) over the ontology-derived checks of every source in the arm, read with
  the detection switches (the no-common-subclass rule and branch-local flags,
  rule 23); number traceability is its own layer (rule 8);
* a retry follows only when an ontology-derived check flags (rule 9): never
  after unknown, NA or number traceability alone;
* the union's flags are deduplicated at output level: one feedback line per
  distinct fired check, whichever source raised it.

The union arm (rule 39, change-log entry 30) runs each of its sources exactly as
in that source's own arm, each with its own retrieval budget (rule 27's 10,000
triples per source), and deduplicates the flags, so its flags are exactly the
union of the single-source flags; its cost is reported as the sum of theirs.
This is the ruling on both reviews' first finding (the union's budget). One
store client serves every source of an output, so a query two sources both ask
is read once; that changes no result.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from harness.checkloop.feedback import (
    FEEDBACK_FORMAT_ID,
    fed_back,
    fired_lines,
    format_sha256,
    order_index,
)
from harness.checks.compiler import CheckCompiler, Output
from harness.checks.numbers import trace as trace_numbers
from harness.checks.outcomes import Outcome, OutputReport, combine
from harness.checks.query import analyse

# Rule 22 (change-log entry 30): an output with no executed query has no checked query; its
# static and footprint checks are unknown, it is never retried, and it is counted per arm.
NO_QUERY = "no_query"
_DIGESTS: dict[int, tuple[Any, str]] = {}


def _digest(obj: Any, make) -> str:
    """A content hash computed once per object (the curated source and the shared
    hierarchy are large); the object is kept alive with its hash."""
    hit = _DIGESTS.get(id(obj))
    if hit is None or hit[0] is not obj:
        hit = (obj, make())
        _DIGESTS[id(obj)] = hit
    return hit[1]


def hierarchy_sha256(hierarchy) -> str:
    def make() -> str:
        text = json.dumps(hierarchy.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return _digest(hierarchy, make)


@dataclass
class Arm:
    """``name`` is how the ledger names the arm ("curated", "census-t5", "union", ...);
    it is recorded, never shown to the agent."""

    name: str
    compilers: tuple[CheckCompiler, ...]

    def __post_init__(self) -> None:
        if not self.compilers:
            raise ValueError("an arm needs at least one source's compiled checks")
        if len({id(c.hierarchy) for c in self.compilers}) != 1:
            raise ValueError("every source of an arm resolves membership through the same shared hierarchy (rule 16)")
        if len({json.dumps(_configuration(c), sort_keys=True) for c in self.compilers}) != 1:
            raise ValueError("an arm's sources all run the same rule set (an OBQC arm is never combined with others)")

    def identity(self) -> dict[str, Any]:
        """What fixes the arm's checks: each source's content hash, the shared
        hierarchy's hash and the retrieval budget (rule 27); for a rule set other than the
        compiler's (the OBQC baseline, which reads no shared hierarchy), its configuration."""
        c0 = self.compilers[0]
        out = {
            "name": self.name,
            "sources": [{"name": c.source.name, "kind": c.source.kind,
                         "digest": _digest(c.source, c.source.digest)} for c in self.compilers],
            "hierarchy_sha256": hierarchy_sha256(c0.hierarchy) if c0.hierarchy is not None else None,
            "bound": c0.bound,
        }
        config = _configuration(c0)
        if config is not None:  # absent for the compiler's own arms, whose identities are unchanged
            out["configuration"] = config
        return out

    def feedback_format(self) -> dict[str, str]:
        """The id and hash of the feedback format the arm's flags are rendered in."""
        fmt = getattr(self.compilers[0], "feedback_format", None)
        return fmt() if fmt is not None else {"id": FEEDBACK_FORMAT_ID, "sha256": format_sha256()}

    def check(self, output: Output, client: Any) -> list[OutputReport]:
        """One report per source, the query parsed once; each compiler call has its own
        retrieval budget (rule 39)."""
        qa = analyse(output.query)
        return [c.check(output, client, qa=qa) for c in self.compilers]


def _configuration(compiler: Any) -> dict[str, Any] | None:
    config = getattr(compiler, "configuration", None)
    return config() if config is not None else None


@dataclass
class Verdict:
    status: str  # flagged | substantive_pass | unknown | vacuous_pass
    any_unknown: bool
    retry: bool
    fired: list[str]  # the feedback lines: distinct fired checks, fixed order, source-neutral
    fired_checks: list[dict[str, Any]] = field(default_factory=list)  # every flag, with its source (ledger only)
    per_source: dict[str, str] = field(default_factory=dict)
    traceability: str = "not_applicable"  # number traceability, its own layer (rule 8)
    reason: str | None = None  # why nothing was checked (NO_QUERY)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def combined_report(reports: Sequence[OutputReport], name: str = "arm") -> OutputReport:
    """The arm's checks as one report, so the four-way status comes from the
    contract's single implementation (OutputReport.status)."""
    return OutputReport(source=name, role=reports[0].role, results=[r for rep in reports for r in rep.results],
                        negation=any(rep.negation for rep in reports),
                        unsupported_constructs=sorted({c for rep in reports for c in rep.unsupported_constructs}),
                        unsupported_coverage={})


def verdict(reports: Sequence[OutputReport], *, heuristic: bool = True, branch_local: bool = True,
            no_query: bool = False, answer: Sequence | None = None) -> Verdict:
    """``answer``: the submitted rows, read only when there is no query (number traceability
    runs in every arm, rule 8: with no result received, a number cannot be traced)."""
    if no_query or not reports:
        return Verdict("unknown", True, False, [], reason=NO_QUERY,
                       traceability=trace_numbers(answer or [], []).outcome)
    arm = combined_report(reports)
    status = arm.status(heuristic, branch_local)
    flags = fed_back(arm.ontology(heuristic, branch_local))
    # rule 10's kind ("" for no data fault, so that the tuples sort): a data fault's evidence names it
    fired_checks = sorted({(order_index(r.check_id), r.check_id, r.source, r.key, r.reason, r.branch_local,
                            "data_fault" in r.annotations,
                            (r.evidence.get("data_fault_kind") or "") if "data_fault" in r.annotations else "")
                           for r in flags})
    trace = [r.outcome for rep in reports for r in rep.results if r.check_id == "number_traceability"]
    return Verdict(
        status=status,
        any_unknown=arm.any_unknown(heuristic, branch_local),
        retry=status == "flagged",
        fired=fired_lines(flags),
        # rule 10: a witnessed data fault (the ledger's, never fed back) and, on a data fault only, its kind (change-log
        # entry 108), so that a fired check without one reads as it did before the kind existed
        fired_checks=[{"check_id": c, "reason": reason, "key": key, "source": src, "branch_local": bl, "data_fault": df,
                       **({"data_fault_kind": kind or None} if df else {})}
                      for _, c, src, key, reason, bl, df, kind in fired_checks],
        per_source={rep.source: rep.status(heuristic, branch_local) for rep in reports},
        traceability=combine(trace).value if trace else Outcome.NOT_APPLICABLE.value,
    )
