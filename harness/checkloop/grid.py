"""A dev grid's OUT as the analysis reads it: its cells, each record's cohort, and a branch record's parent (design log).

The grid launcher (scripts/run_pilot.py --grid-manifest) writes one OUT per graph whose manifest
(``pilot-grid-manifest/1``) pins its cells per phase: an initial cell (phases natural and prompted)
by the prompt it runs under, a branch cell (phase branch) by its parent, the initial cell whose
outputs it checks. A record's cell is its load.arm. The analysis (feat/analysis fe2188a:
harness/analysis/gate.py grid_cells, inputs.py record_cohort and _check_branch) counts an initial
cell's records as the attempts of the cohort the cell names, and a branch cell's records as no
attempt: each checks its parent's output (checker.branched_from). The checker loop's matrix and
ledger key their rows the same way, so this module reads a grid OUT exactly as the analysis does
and refuses what it would refuse, naming the record and the fields, never their values.

A plain run's OUT has one arm per record, its condition; a record there whose cell is another
(a grid record given without its OUT's manifest, as in a records archive) is refused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GRID_MANIFEST = "pilot-grid-manifest/1"  # the grid launcher's OUT manifest (scripts/run_pilot.py GRID_OUT_SCHEMA)
INITIAL_PHASES = ("natural", "prompted")
BRANCH_PHASE = "branch"
PROMPT_PINS = ("condition", "prompt_id", "template_sha256", "nudge_sha256", "ontology_block_sha256")
ATTEMPT_FIELDS = ("question_id", "model", "repetition")  # a branch record's and its parent's


class GridError(ValueError):
    """A grid OUT, or a record of one, that the analysis would refuse."""


@dataclass(frozen=True)
class GridCells:
    initial: dict[str, dict[str, Any]]  # an initial cell -> its prompt's pins (PROMPT_PINS)
    branch: dict[str, str]  # a branch cell -> its parent, an initial cell


def read_grid_cells(out: Path) -> GridCells | None:
    """A grid OUT's cells from its manifest.json; None for a path holding no grid manifest (a plain
    run's OUT, a records archive). Refused, as the analysis refuses them: a phase the grid does not
    have, a phase pinning no cells, a cell pinned in two phases, an initial cell pinned by other
    fields than its prompt's, and a branch cell whose parent is not an initial cell."""
    path = Path(out) / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != GRID_MANIFEST:
        return None
    phases = (manifest.get("fingerprint") or {}).get("phases")
    if not isinstance(phases, dict) or set(phases) - {*INITIAL_PHASES, BRANCH_PHASE}:
        raise GridError(f"{path}: its phases are not the grid's ({', '.join(INITIAL_PHASES)} and {BRANCH_PHASE})")
    initial: dict[str, dict[str, Any]] = {}
    branch: dict[str, Any] = {}
    seen: list[str] = []
    for phase, pinned in phases.items():
        arms = pinned.get("arms") if isinstance(pinned, dict) else None
        if not isinstance(arms, dict):
            raise GridError(f"{path}: its phase {phase} pins no cells (arms)")
        seen += list(arms)
        for cell, pins in arms.items():
            if phase == BRANCH_PHASE:
                branch[cell] = pins.get("parent") if isinstance(pins, dict) else None
            elif not isinstance(pins, dict) or set(pins) != set(PROMPT_PINS):
                raise GridError(f"{path}: the initial cell {cell!r} is pinned by other fields than its prompt's")
            else:
                initial[cell] = dict(pins)
    if len(seen) != len(set(seen)):
        raise GridError(f"{path}: a cell is pinned in two phases")
    stray = sorted(cell for cell, parent in branch.items() if parent not in initial)
    if stray:
        raise GridError(f"{path}: the branch cells {stray} have no initial cell for a parent")
    return GridCells(initial, branch)


def _cell(rec: dict) -> Any:
    return (rec.get("load") or {}).get("arm")


def _prompt_differs(rec: dict, pins: dict[str, Any]) -> list[str]:
    """The pins a record's prompt does not have; a prompt naming another condition than its record is refused."""
    prompt = rec.get("prompt") or {}
    if prompt.get("condition") != rec.get("condition"):
        raise GridError(f"{rec.get('run_id')}: its prompt names another condition than its own")
    found = {"condition": rec.get("condition"), "prompt_id": prompt.get("id"),
             "template_sha256": prompt.get("template_sha256"), "nudge_sha256": prompt.get("nudge_sha256"),
             "ontology_block_sha256": (rec.get("agent_config") or {}).get("ontology_block_sha256")}
    return [pin for pin in PROMPT_PINS if found[pin] != pins[pin]]


def record_cohort(rec: dict, cells: GridCells | None) -> str | None:
    """The cohort a record is an attempt of, or None for a grid branch cell's record (no attempt). A
    plain run's record: its condition (a cell, when it records one, must be it). A grid record: its
    cell, which the manifest must pin as an initial cell, the record's prompt the cell's, field by
    field, and no checker on it."""
    run, cell = rec.get("run_id"), _cell(rec)
    if cells is None:
        if cell is not None and cell != rec.get("condition"):
            raise GridError(f"record {run}: its cell {cell!r} is not its condition, a plain run's one arm (a grid "
                            f"record needs its OUT's manifest.json, which pins its cells: pass the OUT directory)")
        return rec["condition"]
    if not isinstance(cell, str) or not cell:
        raise GridError(f"record {run}: a grid run's record names no cell (load.arm)")
    if cell in cells.branch:
        return None
    if cell not in cells.initial:
        raise GridError(f"record {run}: its cell {cell!r} is not one its OUT's manifest pins")
    differs = _prompt_differs(rec, cells.initial[cell])
    if differs:
        raise GridError(f"record {run}: its prompt ({differs}) is not the one the manifest pins for its cell {cell!r}")
    if rec.get("checker") is not None:
        raise GridError(f"record {run}: an initial cell's record carries a checker (its cell {cell!r} runs no check)")
    return cell


def branch_parent(rec: dict, cells: GridCells, by_run: dict[str, dict]) -> dict:
    """A branch cell's record's parent: a record of the same OUT (``by_run``) in the cell's parent
    cell with the record's question, route model, repetition and store; the branch record runs under
    that cell's prompt and its checker names the parent (checker.branched_from)."""
    run, cell = rec.get("run_id"), _cell(rec)
    parent_cell = cells.branch[cell]
    differs = _prompt_differs(rec, cells.initial[parent_cell])
    if differs:
        raise GridError(f"record {run}: its prompt ({differs}) is not the one the manifest pins for its cell's parent "
                        f"{parent_cell!r}")
    parent_id = (rec.get("checker") or {}).get("branched_from")
    if not parent_id:
        raise GridError(f"record {run}: a branch cell's record ({cell!r}) names no parent (checker.branched_from)")
    parent = by_run.get(parent_id)
    if parent is None:
        raise GridError(f"record {run} ({cell!r}): its parent {parent_id} is not a record of its grid OUT: pass the "
                        f"whole OUT, every phase")
    if _cell(parent) != parent_cell or parent.get("checker") is not None:
        raise GridError(f"record {run} ({cell!r}): its parent {parent_id} is not a record of the cell's parent cell "
                        f"{parent_cell!r}")
    if (any(parent.get(k) != rec.get(k) for k in ATTEMPT_FIELDS)
            or (parent.get("store") or {}).get("name") != (rec.get("store") or {}).get("name")):
        raise GridError(f"record {run} ({cell!r}): its parent {parent_id} is another attempt's (question, route model, "
                        f"repetition or store)")
    return parent


def batch_cohorts(records: list[dict], cells: GridCells | None) -> tuple[dict[str, str], dict[str, dict]]:
    """One OUT's records read as the analysis reads them: (each attempt's cohort by run id, each
    grid branch record's parent by the branch record's run id). Every record is checked, a branch
    record's parent included; anything the analysis would refuse raises GridError."""
    by_run = {r["run_id"]: r for r in records}
    cohorts: dict[str, str] = {}
    parents: dict[str, dict] = {}
    for rec in records:
        cohort = record_cohort(rec, cells)
        if cohort is None:
            parents[rec["run_id"]] = branch_parent(rec, cells, by_run)
        else:
            cohorts[rec["run_id"]] = cohort
    return cohorts, parents
