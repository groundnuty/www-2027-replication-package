"""The check compiler (work package E0.4): turns an ontology source into checks
and runs them on an agent's output. The contract is proposal §7.2, refined by
the S0.1 check contract (version 2)."""

from harness.checks.compiler import CheckCompiler, Output, check_all, check_output
from harness.checks.outcomes import CHECKS, CheckResult, Outcome, OutputReport

__all__ = [
    "CHECKS",
    "CheckCompiler",
    "CheckResult",
    "Outcome",
    "Output",
    "OutputReport",
    "check_all",
    "check_output",
]
