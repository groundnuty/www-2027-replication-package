"""The probe record: one JSON file per closed-book probe try (schema ``closed-book-probe/1``).

A record of its own kind, not an agent-run version: a probe is one provider call
with no tools, and for probe (b) one query execution, so it shares the agent
record's parts where they mean the same thing (the transcript turn shapes, the
per-turn token counts and totals, the outcome fields, the store description, the
score) and holds what replay needs to rebuild everything else from the record
alone (replay.py):

    request         the keyword arguments of the provider call (no time-out), as sent;
                    for a Batch API call the request's ``params``
    call            how the call was made: ``mode`` "sync" or "batch", the batch id and
                    custom id, and the call's latency, wall time, retries and retry errors
    response        the provider's whole response, as JSON (None when the call failed)
    provider_failure  the failure that ended the call ({failure_type, message, details}), or None
    question_text   the user turn; ``gold`` the gold rows and the scoring rules, so the
                    score is recomputed from the record
    answer          probe (a): the answer's form, when unparseable the reasons, and ``declined``
                    (the ``declined-v1`` reporting stratum, or None)
    query           probe (b): the extracted text and its source, and the execution as the
                    agent's SPARQL tool would record it (the raw result capped, its meta,
                    store retries), with ``error_class`` when the query did not run cleanly

Writing follows the agent record: secret values found in the environment are
redacted in strings, ``indent=1``, and writing a loaded record gives back its
exact bytes.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.agent.record import redact_structure

SCHEMA_VERSION = "closed-book-probe/1"
READABLE_SCHEMAS = ("closed-book-probe/1",)


def is_probe_record(data: Any) -> bool:
    return isinstance(data, dict) and str(data.get("schema_version", "")).startswith("closed-book-probe/")


@dataclass
class ProbeRecord:
    run_id: str
    probe: str  # "answer" or "query"
    question_id: str
    base_question_id: str | None
    template: str | None
    tier: str | None
    split: str | None
    graph: str | None
    repetition: int
    model: str
    route: dict
    call: dict
    prompt: dict
    config: dict
    store: dict | None
    harness: dict
    started_at: str
    ended_at: str
    wall_s: float
    question_text: str
    request: dict
    response: dict | None
    provider_failure: dict | None
    transcript: list[dict]
    tokens: dict
    outcome: dict
    answer: dict | None
    query: dict | None
    final_rows: list | None
    gold: dict
    score: dict | None = None
    replay_of: str | None = None
    runner: dict | None = None
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        ordered = {"schema_version": data.pop("schema_version")}
        ordered.update(data)
        return ordered

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ProbeRecord":
        version = data.get("schema_version")
        if version not in READABLE_SCHEMAS:
            raise ValueError(f"unsupported probe record schema {version!r} (readable: {', '.join(READABLE_SCHEMAS)})")
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown probe record fields: {sorted(unknown)}")
        return cls(**data)

    def dumps(self) -> str:
        data, count = redact_structure(self.to_json())
        if count:
            warnings.warn(f"probe record {self.run_id}: redacted {count} secret value(s) before writing", stacklevel=2)
        return json.dumps(data, ensure_ascii=False, indent=1)

    def write(self, out_dir: str | os.PathLike) -> Path:
        """Write ``<out_dir>/<run_id>.json`` atomically and return its path."""
        return write_text_atomic(Path(out_dir), f"{self.run_id}.json", self.dumps())


def write_text_atomic(out: Path, name: str, text: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    target = out / name
    fd, tmp = tempfile.mkstemp(dir=out, prefix=f".{name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def load_probe_record(path: str | os.PathLike) -> ProbeRecord:
    return ProbeRecord.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
