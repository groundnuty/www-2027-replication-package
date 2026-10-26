"""A checker cell's online decision recomputed offline from its record (design log, AC8; dress rehearsal F3).

The replay of a record through the loop (harness/agent/replay.py) serves its checker decision back from the record
(``RecordedChecker``): it shows the loop is consistent with the decision, never that the decision was right. A
decision made by a broken checker (the rehearsal's thread race left every online verdict unknown) replays cleanly
that way. ``recompute_decision`` decides again from what the record holds, with no store and no model: a source arm
from its recorded store responses (``replay_decision``, which raises ReplayDivergence unless the checked query, the
verdict, the reports, the retry and its message are the recorded ones and every recorded store response was used and
none is missing), the OBQC arm from the query alone, retry on empty from the tool-call log, the judge from its
recorded reply. The replay check (harness/agent/replay_check.py) runs it on every record of a batch that holds a
checker decision; scripts/checker_replay.py runs it before its matrix.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from harness.checkloop.arms import Arm


class ArmUnavailable(Exception):
    """A record's checker arm cannot be built from the sources given (or its mode is unknown)."""


def recompute_decision(rec: dict, by_source: Mapping[str, Any], curated: Callable[[], Any]) -> bool:
    """Recompute a checker cell's decision from its record: True when it had one and it is the recorded one; False
    for a record with no checked initial submission. ReplayDivergence when the recomputed decision differs;
    ArmUnavailable for an arm the sources (``by_source``: compilers by source name; ``curated``: the curated
    ontology, for the OBQC arm) cannot build."""
    from harness.baselines.empty import MODE as EMPTY_MODE
    from harness.baselines.empty import replay_empty_decision
    from harness.baselines.judge import JUDGE_MODE, replay_judge_decision
    from harness.baselines.obqc import ObqcCompiler
    from harness.checkloop.online import replay_decision

    chk = rec.get("checker")
    if not chk or chk["initial"]["submission_index"] is None:
        return False
    settings = chk["settings"]
    arm = settings.get("arm") or {}
    if settings.get("mode") == EMPTY_MODE:
        replay_empty_decision(rec)
    elif settings.get("mode") == JUDGE_MODE:
        replay_judge_decision(rec)
    elif settings.get("mode") not in ("feedback", "generic"):
        raise ArmUnavailable(f"record {rec['run_id']}: unknown checker mode {settings.get('mode')!r}")
    elif (arm.get("configuration") or {}).get("rule_set") == "obqc":
        config = arm["configuration"]
        obqc = ObqcCompiler(curated(), rules=config["rules"], iri_output_graph=config["iri_output_graph"])
        replay_decision(rec, Arm(arm["name"], (obqc,)))
    else:
        names_used = [x["name"] for x in arm["sources"]]
        missing = [n for n in names_used if n not in by_source]
        if missing:
            raise ArmUnavailable(f"record {rec['run_id']}: its checker arm needs sources {names_used}; "
                                 f"{missing} not given")
        replay_decision(rec, Arm(arm["name"], tuple(by_source[n] for n in names_used)))
    return True


def arm_sources(rec: dict) -> list[str]:
    """The source names a record's checker arm was built from (none for a baseline's arm)."""
    arm = ((rec.get("checker") or {}).get("settings") or {}).get("arm") or {}
    return [x["name"] for x in arm.get("sources") or [] if isinstance(x, dict) and x.get("name")]


class DecisionRecompute:
    """``recompute_decision`` over compilers by source name: those given (the runner's arms), and, for a source no
    arm gave, one built on first use by ``load(name)`` (verify_batch.py: from a sources directory and the shared
    hierarchy). Called with a record; returns what ``recompute_decision`` returns."""

    def __init__(self, compilers: Mapping[str, Any] | None = None,
                 load: Callable[[str], Any] | None = None):
        self.by_source: dict[str, Any] = dict(compilers or {})
        self.load = load

    @classmethod
    def from_arms(cls, arms: Mapping[Any, Arm]) -> DecisionRecompute:
        return cls({c.source.name: c for arm in arms.values() for c in arm.compilers if hasattr(c, "source")})

    def _need(self, name: str) -> None:
        if name not in self.by_source and self.load is not None:
            self.by_source[name] = self.load(name)

    def curated(self) -> Any:
        self._need("curated")
        if "curated" not in self.by_source:
            raise ArmUnavailable("the OBQC arm reads the curated source, which was not given")
        return self.by_source["curated"].source

    def __call__(self, rec: dict) -> bool:
        for name in arm_sources(rec):
            self._need(name)
        return recompute_decision(rec, self.by_source, self.curated)
