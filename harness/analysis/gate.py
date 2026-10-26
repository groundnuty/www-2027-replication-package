"""The gate before analysis: a batch is analysed only with a passing, current replay check.

The replay check (scripts/verify_batch.py, E0.1) writes ``OUT/replay-check.json``:

    {"schema": "replay-check/1", "n_records", "n_ok", "failures", "records_sha256", ...}

``records_sha256`` is the sha256 of the compact JSON list of ``[path, sha256 of the file]``
pairs, sorted by path, over the ``*.json`` files in ``records/``, ``infrastructure/``,
``harness/`` and ``aborted/`` (set-aside sidecars excluded), with paths relative to OUT. This
module recomputes it from the files, so a record changed, added or removed after the check
refuses the batch; the count inside the file is also compared with the files on disk, since
``n_ok == n_records`` alone only shows the file agrees with itself.

The scheme is E0.1's (harness/agent/replay_check.py); it is reimplemented here, in one
place, so the gate can be reconciled with that module at merge.

The merged runner (integration/freeze ea7042b, harness/agent/replay_check.py) writes ``replay-check/2``, whose
``records_sha256`` covers every input of the analysis, not only the replayed records: the records as above, plus the
files that are only hashed (``hashed_only``: harness/'s set-aside sidecars and the runner's ``concurrency-events.jsonl``).
Its input rule (``input_kind``, ``batch_inputs``) and digest are vendored here verbatim, and compared byte for byte with
a pinned copy of that module (tests/test_analysis_gate.py). A /2 check is accepted only when it passed, counts exactly
the replayed records on disk, lists exactly the hashed-only inputs on disk, and its digest is recomputed from the
files; a /1 check (an earlier batch) as before; any other schema is refused. The runner's own gate
(require_current_check) also requires the check to have been made under the code in use, unchanged: that is left out
here on purpose, since it would refuse every batch verified before a later commit of the analysis; the check's code is
recorded with each batch's results instead (``replay_check_code``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"[0-9a-f]{64}")
CHECK_FILE = "replay-check.json"
CHECK_SCHEMA = "replay-check/1"  # E0.1's: the replayed records alone
CHECK_SCHEMA_2 = "replay-check/2"  # the merged runner's: every input of the analysis
BATCH_DIRS = ("records", "infrastructure", "harness", "aborted")
SETASIDE_SUFFIX = ".setaside.json"
HASHED_FILES = ("concurrency-events.jsonl",)


class BatchRefused(Exception):
    """A batch may not be analysed; the message says why and what to run."""


def batch_files(batch: Path) -> list[str]:
    """The files the replay check covers, as sorted paths relative to the batch."""
    found = []
    for sub in BATCH_DIRS:
        for path in (Path(batch) / sub).glob("*.json"):
            if path.is_file() and not path.name.endswith(SETASIDE_SUFFIX):
                found.append(f"{sub}/{path.name}")
    return sorted(found)


def _files_sha256(batch: Path, rels: list[str]) -> str:
    """The sha256 of the compact JSON list of [path, sha256 of the file] pairs, sorted by path."""
    batch = Path(batch)
    pairs = sorted([rel, hashlib.sha256((batch / rel).read_bytes()).hexdigest()] for rel in rels)
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("utf-8")).hexdigest()


def records_sha256(batch: Path) -> str:
    """A replay-check/1 digest: over the replayed records alone (batch_files)."""
    return _files_sha256(batch, batch_files(batch))


def input_kind(rel: str) -> str | None:
    """For a path relative to the batch, the runner's kind of it (replay_check.input_kind): "replay" (a run record),
    "hashed" (an analysis input that is not a record) or None (not an input of the analysis)."""
    if rel in HASHED_FILES:
        return "hashed"
    sub, _, name = rel.partition("/")
    if sub not in BATCH_DIRS or "/" in name or not name.endswith(".json"):
        return None
    if name.endswith(SETASIDE_SUFFIX):
        return "hashed" if sub == "harness" else None
    return "replay"


def batch_inputs(batch: Path) -> list[str]:
    """Every analysis input of a batch, as sorted paths relative to it (the runner's replay_check.batch_inputs)."""
    batch = Path(batch)
    found = [name for name in HASHED_FILES if (batch / name).is_file()]
    for sub in BATCH_DIRS:
        found += [f"{sub}/{p.name}" for p in (batch / sub).glob("*.json") if p.is_file()]
    return sorted(rel for rel in found if input_kind(rel))


def inputs_sha256(batch: Path) -> str:
    """A replay-check/2 digest: over every analysis input (batch_inputs; the runner's current_records_sha256)."""
    return _files_sha256(batch, batch_inputs(batch))


def require_replay_check(batch: Path) -> dict[str, Any]:
    """The batch's replay check, when it passed and still matches the files; else BatchRefused."""
    batch = Path(batch)
    rerun = f"run `uv run python scripts/verify_batch.py {batch}` first"
    path = batch / CHECK_FILE
    if not path.is_file():
        raise BatchRefused(f"{batch}: no {CHECK_FILE}, so its records were never shown to replay; {rerun}")
    try:
        check = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BatchRefused(f"{path}: not readable ({exc}); {rerun}") from None
    schema = check.get("schema") if isinstance(check, dict) else None
    if schema not in (CHECK_SCHEMA, CHECK_SCHEMA_2):
        raise BatchRefused(f"{path}: unknown schema {schema!r} (neither {CHECK_SCHEMA} nor {CHECK_SCHEMA_2})")
    if check.get("n_ok") != check.get("n_records") or check.get("failures"):
        raise BatchRefused(f"{batch}: the replay check failed ({check.get('n_ok')} of {check.get('n_records')} "
                           f"records passed, {len(check.get('failures') or [])} failures listed); {rerun}")
    if schema == CHECK_SCHEMA:
        replayed, digest = batch_files(batch), records_sha256
    else:
        inputs = batch_inputs(batch)
        replayed, digest = [rel for rel in inputs if input_kind(rel) == "replay"], inputs_sha256
        hashed = [rel for rel in inputs if input_kind(rel) == "hashed"]
        if check.get("hashed_only") != hashed:
            raise BatchRefused(f"{batch}: its inputs hashed only are {hashed}, but the check hashed only "
                               f"{check.get('hashed_only')}; {rerun}")
    if check.get("n_records") != len(replayed):
        raise BatchRefused(f"{batch}: the check covers {check.get('n_records')} records but the batch has "
                           f"{len(replayed)} files; {rerun}")
    if check.get("records_sha256") != digest(batch):
        raise BatchRefused(f"{batch}: records changed since the replay check (records_sha256 differs); {rerun}")
    return check


# The runner's two manifests (scripts/run_pilot.py), told apart by their schema alone. A run without the grid writes
# MANIFEST_SCHEMA, its settings (run_fingerprint: questions_sha256, scorer_digest, ...) in "fingerprint"; a dev-grid run
# (scripts/build_dev_grid.py, the grid launcher) writes GRID_OUT_SCHEMA, whose fingerprint is grid_fingerprint's: the
# settings every phase shares (the same questions_sha256 and scorer_digest among them) in "shared", beside "routes" and
# "phases". Each layout's settings are read only where its writer puts them: a manifest of another or no schema, or
# whose fingerprint is not its schema's shape, is refused, never read at the other layout's place.
PLAIN, GRID = "plain", "grid"
PLAIN_SCHEMA, GRID_SCHEMA = "pilot-manifest/1", "pilot-grid-manifest/1"  # MANIFEST_SCHEMA, GRID_OUT_SCHEMA
LAYOUTS = {PLAIN_SCHEMA: PLAIN, GRID_SCHEMA: GRID}
GRID_FINGERPRINT = ("shared", "routes", "phases")
SETTINGS_AT = {PLAIN: "fingerprint", GRID: "fingerprint.shared"}


def _fingerprint(batch: Path, what: str) -> tuple[str, dict[str, Any]]:
    """The batch's layout and its manifest's whole fingerprint, of that layout's shape (read_manifest)."""
    manifest_path = Path(batch) / "manifest.json"
    if not manifest_path.is_file():
        raise BatchRefused(f"{batch}: no manifest.json, so {what} unknown")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError:
        raise BatchRefused(f"{manifest_path}: not readable JSON, so {what} unknown") from None
    schema = manifest.get("schema") if isinstance(manifest, dict) else None
    layout = LAYOUTS.get(schema) if isinstance(schema, str) else None
    if layout is None:
        raise BatchRefused(f"{batch}: its manifest's schema {schema!r} is neither a plain run's ({PLAIN_SCHEMA}) nor a "
                           f"grid run's ({GRID_SCHEMA}), so {what} unknown")
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise BatchRefused(f"{batch}: its manifest's fingerprint is not an object, so {what} unknown")
    if layout == PLAIN:
        if "shared" in fingerprint:
            raise BatchRefused(f"{batch}: a plain run's fingerprint holds no 'shared' settings (a grid run's does, "
                               f"under its own schema), so {what} unknown")
        return layout, fingerprint
    if set(fingerprint) != set(GRID_FINGERPRINT) or not isinstance(fingerprint["shared"], dict):
        raise BatchRefused(f"{batch}: its manifest's fingerprint is not a grid run's fingerprint (exactly "
                           f"{', '.join(GRID_FINGERPRINT[:-1])} and {GRID_FINGERPRINT[-1]}, its settings in shared), so "
                           f"{what} unknown")
    return layout, fingerprint


def read_manifest(batch: Path, what: str) -> tuple[str, dict[str, Any]]:
    """The batch's layout (PLAIN or GRID) and the settings its manifest fingerprints (a plain run's fingerprint, a grid
    run's fingerprint.shared); else BatchRefused saying ``what`` stays unknown. The manifest holds no outcome."""
    layout, fingerprint = _fingerprint(batch, what)
    return layout, fingerprint if layout == PLAIN else fingerprint["shared"]


# A grid run's cells, as its manifest pins them per phase (scripts/run_pilot.py grid_fingerprint, phase_arm_fingerprints):
# an initial cell (phases natural and prompted) by the prompt it runs under, a branch cell (phase branch) by its parent,
# the initial cell whose outputs it checks. A record's cell is its load.arm (inputs.load_batch).
INITIAL_PHASES = ("natural", "prompted")
BRANCH_PHASE = "branch"
PROMPT_PINS = ("condition", "prompt_id", "template_sha256", "nudge_sha256", "ontology_block_sha256")


@dataclass(frozen=True)
class GridCells:
    initial: dict[str, dict[str, Any]]  # an initial cell -> its prompt's pins (PROMPT_PINS)
    branch: dict[str, str]  # a branch cell -> its parent, an initial cell


def grid_cells(batch: Path) -> GridCells | None:
    """A grid batch's cells from its manifest; None for a plain run's batch (its one arm is its condition). Refused: a
    phase the grid does not have, a cell pinned in two phases, an initial cell whose pins are not PROMPT_PINS, and a
    branch cell whose parent is not an initial cell."""
    what = "its cells are"
    layout, fingerprint = _fingerprint(batch, what)
    if layout == PLAIN:
        return None
    phases = fingerprint["phases"]
    if not isinstance(phases, dict) or set(phases) - {*INITIAL_PHASES, BRANCH_PHASE}:
        raise BatchRefused(f"{batch}: its manifest's phases are not the grid's ({', '.join(INITIAL_PHASES)} and "
                           f"{BRANCH_PHASE}), so {what} unknown")
    initial: dict[str, dict[str, Any]] = {}
    parents: dict[str, Any] = {}
    seen: list[str] = []
    for phase, pinned in phases.items():
        arms = pinned.get("arms") if isinstance(pinned, dict) else None
        if not isinstance(arms, dict):
            raise BatchRefused(f"{batch}: its manifest's phase {phase} pins no cells (arms), so {what} unknown")
        seen += list(arms)
        for cell, pins in arms.items():
            if phase == BRANCH_PHASE:
                parents[cell] = pins.get("parent") if isinstance(pins, dict) else None
            elif not isinstance(pins, dict) or set(pins) != set(PROMPT_PINS):
                raise BatchRefused(f"{batch}: its manifest pins the initial cell {cell!r} by other fields than its "
                                   f"prompt's ({', '.join(PROMPT_PINS)}), so {what} unknown")
            else:
                initial[cell] = dict(pins)
    if len(seen) != len(set(seen)):
        raise BatchRefused(f"{batch}: its manifest pins a cell in two phases, so {what} unknown")
    stray = sorted(cell for cell, parent in parents.items() if parent not in initial)
    if stray:
        raise BatchRefused(f"{batch}: its manifest's branch cells {stray} have no initial cell for a parent, so {what} "
                           f"unknown")
    return GridCells(initial, parents)


def manifest_layout(batch: Path) -> str:
    """The batch's layout, PLAIN or GRID (read_manifest), which decides the shape of its attempt keys."""
    return read_manifest(batch, "the shape of its attempt keys is")[0]


def _manifest_questions_sha(batch: Path) -> str | None:
    return read_manifest(batch, "the questions it ran on are")[1].get("questions_sha256")


def require_scorer(batch: Path, frozen: str) -> str:
    """The batch's scorer digest, when it is the frozen one (prereg §3: the sha256 of the scorer's closed module set,
    feat/scorer-digest's scorer_digest(), computed from the files, no git needed); else BatchRefused. The runner
    records it in the manifest's fingerprint (``scorer_digest``), which a resume must match exactly, so it binds every
    record of the batch; the manifest holds no outcome, so a batch is refused here before any record is hashed or
    read. The rules-version string in each record's score does not identify the scorer (spec trace C11: several
    branches carry the pre-entry-62 scorer as scorer-rules-v4), and no record carries the digest: a batch whose
    manifest records none is refused. A grid run records it with the settings every phase shares
    (``fingerprint.shared.scorer_digest``, read_manifest), and is held to it alike."""
    layout, settings = read_manifest(batch, "the scorer its records were scored by is")
    found = settings.get("scorer_digest")
    if found is None:
        raise BatchRefused(f"{batch}: its manifest records no scorer digest ({SETTINGS_AT[layout]}.scorer_digest), so "
                           f"the scorer its records were scored by is unknown (the rules-version string does not "
                           f"identify it)")
    if not isinstance(found, str) or SHA256.fullmatch(found) is None:
        raise BatchRefused(f"{batch}: its manifest's scorer digest is not a sha256 (64 lower-case hex digits)")
    if found != frozen:
        raise BatchRefused(f"{batch}: scored by the scorer {found}, not the frozen scorer {frozen} (prereg §3)")
    return found


def select_questions(batch: Path, candidates: list[Path] | tuple[Path, ...]) -> Path:
    """The candidate questions file the batch ran on; else BatchRefused."""
    expected = _manifest_questions_sha(batch)
    for path in candidates:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected:
            return Path(path)
    raise BatchRefused(f"{batch}: ran on questions {expected}; none of the given questions files has that sha256")


def require_questions_match(batch: Path, questions: Path) -> str:
    """The questions file's sha256, when it is the file the batch ran on (its manifest's
    ``fingerprint.questions_sha256``, a grid run's ``fingerprint.shared.questions_sha256``); else
    BatchRefused, since strata joined from another file would be silently wrong."""
    expected = _manifest_questions_sha(batch)
    actual = hashlib.sha256(Path(questions).read_bytes()).hexdigest()
    if actual != expected:
        raise BatchRefused(f"{batch}: ran on questions {expected}, but {questions} has sha256 {actual}")
    return actual
