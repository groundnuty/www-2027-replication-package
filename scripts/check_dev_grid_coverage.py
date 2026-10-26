#!/usr/bin/env python3
"""Coverage of the development-split grid (E1.5a, the design log): does every cell hold
every development question's every repetition, as the manifest plans them? Lists what is missing.

    uv run python scripts/check_dev_grid_coverage.py --manifest OUT/manifest.json \\
        --run uniprot-sp-2026_03=runs/dev-grid-uniprot --run rhea-142=runs/dev-grid-rhea [--report R.json]

"The run completed" is not evidence (design log): the runner's exit status, ``runs.jsonl`` and the
replay check are never read. Coverage comes from the run records alone (``RUN/records/*.json``,
or ``RUN/records.tar.gz`` as committed), and a record counts only when its content says it is the
planned attempt:

* its model is a grid model (by route model), its question a planned development question, its
  split the grid's (``dev``), its repetition planned, and its cell one its model's role runs (the
  panel runs none, curated and census, and fb:curated and gen:any on none);
* its agent settings are the frozen ones (turns and wall clock, grid-wide), and its output-token and
  tool-result caps and its route (provider and settings: the transport) are its model's, as the
  manifest's ``model_settings`` records them from the runner's model table (a seeded retry's route
  adds its sampling seed), and so are its rules (absent but Gemma's report-key rule and Qwen3.6's reasoning-call
  rule); a run directory's recorded pin moves (its manifest's ``pin_moves``: Gemma's rule, GLM's turn and wall caps,
  Qwen3.6's rule) hold its records of the phases
  pinned before the move, and the branch records whose parents are of those phases, to the earlier values;
* its cohort follows from its prompt condition and ontology block hash (the spec's block for
  that graph and cohort; a block still FILL verifies nothing, so such a record does not count);
* its ``load`` names its cell (``arm``) and its phase's seed (``order_seed``): a record run in
  another order, or not by the grid runner, does not count;
* a branch record (with ``checker``) names its arm by the checker's mode and arm name; the arm was
  built from the cell's sources for its model's role (gen:any: every source of the role's feedback
  arms, so curated alone for the panel); it carries the frozen retry budget, switches and (when filled)
  feedback and generic hashes; its verdict holds the status of every source that decides the cell
  (``per_source`` against the manifest's ``verdict_sources``: under checker-loop/2 gen:any's curated
  sources alone, a feedback cell's every source), and it retried exactly when one of them flagged;
  and it was branched from the record of its parent
  cohort with the same model, question and repetition;
* it did not fail infrastructure (such a record is unscored and outside N_att: it needs a re-run).

The report gives the counts per cell, per role (must, panel, exploratory: never pooled) and per phase
(natural, prompted, branch, each with its seed, its cells and its counts per graph), and, for every
recorded attempt of the generic control (``gen:any``), the sources that flagged it, so the
analysis can restrict the control to one source's flagged set.

Offline replay (``--replay GRAPH=PATH``, a ``matrix.jsonl.gz`` from scripts/checker_replay.py or a
directory holding one; repeatable): every replay job of the manifest must have, for each counted
initial record of its model and cohort, a row per arm (every source of the graph, census-t2 and
census-t20 included, and the union) and per detection-switch variant, matched by the record's run
id. Missing rows are listed per record; rows of a run id that is no counted initial record, of an
arm or variant not planned, or repeated, are problems. Without ``--replay`` the replay is reported
as not checked (``replay.complete`` null), never as complete.

A planned branch attempt with no record is complete when its parent did not submit (there is
nothing to check, and the initial answer stands): reported as ``inherited``. When its parent is
missing or not counted, it is missing ("parent ..."). Records that are not planned attempts
(test split, unknown question, model, repetition, cohort or arm, duplicates, unreadable files) are
listed as problems; two records of one attempt are a problem and neither counts. Any problem or
missing attempt makes the grid incomplete.

Exit status: 0 complete (and the replay complete, when given); 1 attempts or replay rows missing,
or problems found; 2 refused (the manifest or its attempts file does not check out, or a run
directory or replay matrix does not exist).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tarfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "dev-grid-manifest/2"
PLACEHOLDER = "FILL"
NO_QUERY = "no_query"  # harness.checkloop.arms.NO_QUERY: no query before the submission, nothing checked
# Nothing grid-wide: each model's caps on each graph are in the manifest's model_settings (design log, entry 90), and its
# retry's turn and wall caps are its own (the retry gets the attempt's; the spec fixes only the nudges).
MODEL_CAPS = ("max_turns", "max_wall_s", "max_tokens", "max_result_chars")
RETRY_CAPS = ("max_turns", "max_wall_s")
# A model's rules, each in its records' agent_config only when its table entry sets it (harness.agent.loop.RULE_FIELDS):
# Gemma's report-key rule, Qwen3.6's reasoning-call rule
RULE_FIELDS = ("report_key_rule", "reasoning_call_rule")
ROUTE_SEED = "seed"  # a seeded retry's route settings add its sampling seed


class Refused(Exception):
    pass


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise Refused(f"{path}: schema {manifest.get('schema')!r}, expected {MANIFEST_SCHEMA!r}")
    raw = gzip.decompress((path.parent / manifest["attempts_file"]).read_bytes())
    if hashlib.sha256(raw).hexdigest() != manifest["attempts_sha256"]:
        raise Refused(f"{manifest['attempts_file']} does not match the manifest's attempts_sha256")
    return manifest, [json.loads(line) for line in raw.decode("utf-8").splitlines()]


def pin_moves_of(run_dir: Path) -> dict[str, dict]:
    """A grid OUT's recorded pin moves, per model (scripts/run_pilot.py check_grid_manifest; ``report_key_moves`` as a
    development commit named them); none without a manifest."""
    path = Path(run_dir) / "manifest.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {mv["model"]: mv for mv in (manifest.get("report_key_moves") or []) + (manifest.get("pin_moves") or [])}


def read_records(run_dir: Path) -> Iterator[tuple[str, dict | None]]:
    """(name, record or None when unreadable) for every record of a run directory."""
    records = run_dir / "records"
    if records.is_dir():
        for path in sorted(records.glob("*.json")):
            yield from _parse(f"{run_dir.name}/records/{path.name}", path.read_bytes())
    tar = run_dir / "records.tar.gz"
    if tar.is_file():
        with tarfile.open(tar) as tf:
            for member in sorted(tf.getmembers(), key=lambda m: m.name):
                if member.isfile() and member.name.endswith(".json") and "/records/" in f"/{member.name}":
                    yield from _parse(f"{run_dir.name}/{tar.name}:{member.name}", tf.extractfile(member).read())


def _parse(name: str, data: bytes) -> Iterator[tuple[str, dict | None]]:
    try:
        rec = json.loads(data)
    except ValueError:
        rec = None
    yield name, rec if isinstance(rec, dict) else None


class _Grid:
    """What the manifest plans for one graph, and how a record is told apart."""

    def __init__(self, manifest: dict, graph: str, attempts: list[dict]):
        spec = manifest["spec"]
        self.graph = graph
        self.spec = spec
        self.models = {m["route_model"]: m["name"] for m in spec["models"]}
        self.split = spec["split"]
        self.reps = {a["repetition"] for a in attempts}  # each model's own per phase: the planned attempts say which
        self.questions = {a["question_id"] for a in attempts}
        blocks = spec["graphs"][graph].get("prompt_blocks") or {}
        self.cohorts = {}  # (condition, block sha256 or None) -> cohort; a FILL block matches nothing
        self.unverifiable = set()
        for c, cs in spec["cohorts"].items():
            if cs.get("source") is None:
                self.cohorts[(cs["condition"], None)] = c
            elif (sha := (blocks.get(c) or {}).get("sha256")) not in (None, PLACEHOLDER):
                self.cohorts[(cs["condition"], sha)] = c
            else:
                self.unverifiable.add(cs["condition"])
        self.arms = {(b["parent"], b["mode"], b["arm"]): b["id"] for b in spec.get("branch_cells") or []}
        self.parent_of = {b["id"]: b["parent"] for b in spec.get("branch_cells") or []}
        roles = manifest["roles"]
        self.sources = {(m, c["id"]): c["sources"] for m, role in roles.items() for c in manifest["cells"][role]
                        if c["phase"] == "branch"}
        # the sources whose verdicts decide a branch cell: a generic cell's curated ones (checker-loop/2, M7)
        self.verdict_sources = {(m, c["id"]): c.get("verdict_sources", c["sources"]) for m, role in roles.items()
                                for c in manifest["cells"][role] if c["phase"] == "branch"}
        self.seed_of = {c: ph["seed"] for ph in manifest["order"]["phases"] for c in ph["cells"]}
        self.model_settings = manifest.get("model_settings") or {}

    def classify(self, rec: dict, moves: dict[str, dict] | None = None) -> tuple[tuple | None, str | None, bool]:
        """(key, reason it does not count or None, whether the reason is a problem beyond a re-run). ``moves``: the
        record's run directory's recorded pin moves (pin_moves_of)."""
        model = self.models.get(rec.get("model"))
        qid, rep, split = rec.get("question_id"), rec.get("repetition"), rec.get("split")
        if split != self.split:
            return None, (f"split {split!r}: no test-split question may run" if self.split == "dev" else
                          f"split {split!r}: this grid runs the {self.split} split only"), True
        if model is None:
            return None, f"model {rec.get('model')!r} is not a grid model", True
        if qid not in self.questions:
            return None, f"question {qid!r} is not a planned development question", True
        if rep not in self.reps:
            return None, f"repetition {rep!r} is not planned", True
        config = rec.get("agent_config") or {}
        condition = config.get("condition")
        if rec.get("condition", condition) != condition:
            return None, f"condition {rec.get('condition')!r} but agent_config says {condition!r}", True
        cohort = self.cohorts.get((condition, config.get("ontology_block_sha256")))
        if cohort is None:
            why = ("its cohort's block is still FILL in the spec" if condition in self.unverifiable else
                   f"condition {condition!r} with block {config.get('ontology_block_sha256')!r} is no cohort")
            return None, why, True
        ck = rec.get("checker")
        if ck is None:
            cell = cohort
        else:
            settings = ck.get("settings") or {}
            cell = self.arms.get((cohort, settings.get("mode"), (settings.get("arm") or {}).get("name")))
            if cell is None:
                return None, (f"checker arm {(settings.get('arm') or {}).get('name')!r} in mode "
                              f"{settings.get('mode')!r} on cohort {cohort!r} is no grid cell"), True
        key = (model, cell, qid, rep)
        wrong = []
        ms = self.model_settings.get(model)
        caps = (ms or {}).get("caps", {}).get(self.graph)
        if ms is None or caps is None:
            wrong.append(f"the manifest records no settings for {model} on {self.graph}")
        else:
            # the phase whose pins the record ran under: its cohort's (a branch record's config is its parent's), and a
            # model's recorded move keeps the phases pinned before it at the earlier values
            move = (moves or {}).get(model)
            earlier = move["from"] if move and self.spec["cohorts"][cohort]["phase"] in move["phases_before"] else {}
            caps = {**caps, **{k: v for k, v in earlier.items() if k in MODEL_CAPS}}
            wrong += [f"{k} {config.get(k)!r} != {caps[k]!r} ({model}'s cap)" for k in MODEL_CAPS
                      if config.get(k) != caps[k]]
            for field in RULE_FIELDS:
                rule = earlier[field] if field in earlier else ms.get(field)
                if config.get(field) != rule:
                    wrong.append(f"{field} {config.get(field)!r} != {rule!r} ({model}'s)")
            route = rec.get("route") or {}
            transport = {k: v for k, v in (route.get("settings") or {}).items() if k != ROUTE_SEED}
            if route.get("provider") != ms["provider"] or transport != ms["route_settings"]:
                diff = [f"{k} {transport.get(k)!r} != {ms['route_settings'].get(k)!r}"
                        for k in sorted(set(transport) | set(ms["route_settings"]))
                        if transport.get(k) != ms["route_settings"].get(k)]
                if route.get("provider") != ms["provider"]:
                    diff.insert(0, f"provider {route.get('provider')!r} != {ms['provider']!r}")
                wrong.append(f"route settings of {model} differ from the manifest's: " + "; ".join(diff))
        load = rec.get("load")
        if not isinstance(load, dict):
            wrong.append("no load (arm, order seed): not written by the grid runner")
        else:
            if load.get("arm") != cell:
                wrong.append(f"load.arm {load.get('arm')!r} != {cell!r}")
            if load.get("order_seed") != self.seed_of[cell]:
                wrong.append(f"load.order_seed {load.get('order_seed')!r} != its phase's seed {self.seed_of[cell]!r}")
        if ck is not None and (model, cell) in self.sources:
            wrong += self._arm_problems(ck, model, cell)
            budget = settings.get("retry_budget") or {}
            wanted = {**{k: (caps or {}).get(k) for k in RETRY_CAPS}, **self.spec["retry_budget"]}
            wrong += [f"retry_budget.{k} {budget.get(k)!r} != {v!r}" for k, v in wanted.items() if budget.get(k) != v]
            if settings.get("switches") != self.spec["switches"]:
                wrong.append(f"switches {settings.get('switches')!r} != {self.spec['switches']!r}")
            code = self.spec.get("code") or {}
            for name, got in (("feedback_format_sha256", (settings.get("feedback_format") or {}).get("sha256")),
                              ("generic_text_sha256", settings.get("generic_sha256"))):
                if code.get(name) not in (None, PLACEHOLDER) and got != code[name]:
                    wrong.append(f"{name} {got!r} != {code[name]!r}")
        if wrong:
            return key, "settings are not the frozen ones: " + "; ".join(wrong), True
        outcome = rec.get("outcome") or {}
        if outcome.get("failure_type") == "infrastructure":
            return key, "failed:infrastructure (unscored, needs a re-run)", False
        return key, None, False

    def _arm_problems(self, ck: dict, model: str, cell: str) -> list[str]:
        """A branch record's arm against its cell for its model's role: built from the cell's sources, a
        verdict per source that decides the cell (``verdict_sources``: gen:any's role's curated sources under
        checker-loop/2, every source of a feedback cell), and a retry exactly when one of them flagged."""
        wrong = []
        expected = sorted(self.sources[(model, cell)])
        built = sorted(s.get("name") for s in ((ck.get("settings") or {}).get("arm") or {}).get("sources") or [])
        if built != expected:
            wrong.append(f"arm sources {built} != {expected}")
        verdict = (ck.get("initial") or {}).get("verdict")
        if not isinstance(verdict, dict):
            return wrong + ["no verdict"]
        per_source = verdict.get("per_source") or {}
        deciding = sorted(self.verdict_sources[(model, cell)])
        if verdict.get("reason") != NO_QUERY and sorted(per_source) != deciding:
            wrong.append(f"per_source verdicts for {sorted(per_source)} != {deciding}")
        flagged = flagging_sources(ck)
        if bool(verdict.get("retry")) != bool(flagged) or (ck.get("retry") is not None) != bool(flagged):
            wrong.append(f"retry {'made' if ck.get('retry') is not None else 'not made'} with verdict retry "
                         f"{verdict.get('retry')!r}, but the sources that flagged are {flagged}")
        return wrong


def flagging_sources(ck: dict) -> list[str]:
    per_source = (((ck.get("initial") or {}).get("verdict") or {}).get("per_source")) or {}
    return sorted(s for s, status in per_source.items() if status == "flagged")


def read_matrix(path: Path) -> Iterator[tuple[str, dict | None]]:
    """(name, row or None when unreadable) for every row of an offline replay matrix."""
    file = path / "matrix.jsonl.gz" if path.is_dir() else path
    with gzip.open(file, "rt", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if line.strip():
                yield from _parse(f"{file.parent.name}/{file.name}:{n}", line.encode("utf-8"))


def check_replay(manifest: dict, graph: str, planned: list[dict], valid: dict[tuple, dict],
                 paths: list[Path]) -> tuple[list[dict], list[dict], list[dict]]:
    """(jobs with expected and found rows, missing rows per record, problems) for one graph."""
    jobs = [j for j in manifest["replay_jobs"] if j["graph"] == graph]
    cohorts = {j["cohort"] for j in jobs}
    initial = {rec["run_id"]: key for key, rec in valid.items() if key[1] in cohorts}
    plan = {(j["model"], j["cohort"]): j for j in jobs}
    seen: dict[tuple, int] = defaultdict(int)
    problems = []
    for path in paths:
        for name, row in read_matrix(path):
            if row is None:
                problems.append({"graph": graph, "row": name, "problem": "unreadable"})
                continue
            key = initial.get(row.get("run_id"))
            if key is None:
                problems.append({"graph": graph, "row": name,
                                 "problem": f"run id {row.get('run_id')!r} is no counted initial record"})
                continue
            job = plan[(key[0], key[1])]
            if row.get("arm") not in job["arms"] or row.get("variant") not in job["variants"]:
                problems.append({"graph": graph, "row": name, "problem": f"arm {row.get('arm')!r} with variant "
                                 f"{row.get('variant')!r} is not planned"})
                continue
            seen[(row["run_id"], row["arm"], row["variant"])] += 1
    problems += [{"graph": graph, "row": None, "problem": f"{n} rows for run id {k[0]!r}, arm {k[1]!r}, variant {k[2]!r}"}
                 for k, n in seen.items() if n > 1]
    summary, missing = [], []
    for job in jobs:
        found = 0
        for a in planned:
            if a["phase"] == "branch" or (a["model"], a["cell"]) != (job["model"], job["cohort"]):
                continue
            item = {"graph": graph, "model": a["model"], "cohort": a["cell"], "question_id": a["question_id"],
                    "repetition": a["repetition"]}
            rec = valid.get((a["model"], a["cell"], a["question_id"], a["repetition"]))
            if rec is None:
                missing.append({**item, "run_id": None, "missing": "all: the initial record is missing or does not count"})
                continue
            absent = [[arm, v] for arm in job["arms"] for v in job["variants"] if not seen[(rec["run_id"], arm, v)]]
            found += len(job["arms"]) * len(job["variants"]) - len(absent)
            if not paths:
                missing.append({**item, "run_id": rec["run_id"], "missing": "all: no replay matrix given for this graph"})
            elif absent:
                missing.append({**item, "run_id": rec["run_id"], "missing": absent})
        summary.append({"graph": graph, "model": job["model"], "cohort": job["cohort"],
                        "expected_rows": job["expected_rows"], "found_rows": found})
    return summary, missing, problems


def check_coverage(manifest: dict, attempts: list[dict], runs: dict[str, list[Path]],
                   replays: dict[str, list[Path]] | None = None) -> dict[str, Any]:
    missing, inherited, problems, generic_flags = [], [], [], []
    replay_jobs, replay_missing, replay_problems = [], [], []
    rows = []
    for graph in manifest["spec"]["graphs"]:
        planned = [a for a in attempts if a["graph"] == graph]
        grid = _Grid(manifest, graph, planned)
        found: dict[tuple, list[tuple[str, dict]]] = defaultdict(list)
        invalid: dict[tuple, str] = {}
        for run_dir in runs.get(graph, []):
            moves = pin_moves_of(run_dir)
            for name, rec in read_records(run_dir):
                if rec is None:
                    problems.append({"graph": graph, "record": name, "problem": "unreadable"})
                    continue
                key, why, is_problem = grid.classify(rec, moves)
                if why is None:
                    found[key].append((name, rec))
                    continue
                if is_problem:
                    problems.append({"graph": graph, "record": name, "problem": why})
                if key is not None:
                    invalid[key] = why
        # a counted record of a cell its model does not run, or of a repetition it does not run there, is no
        # planned attempt (each model's own cells and repetitions per phase: manifest model_cells, model_repetitions)
        planned_keys = {(a["model"], a["cell"], a["question_id"], a["repetition"]) for a in planned}
        planned_cells = {(a["model"], a["cell"]) for a in planned}
        for key in [k for k in found if k not in planned_keys]:
            role = manifest["roles"].get(key[0])
            why = (f"repetition {key[3]} of cell {key[1]!r} is not planned for {key[0]} (role {role})"
                   if (key[0], key[1]) in planned_cells else f"cell {key[1]!r} is not planned for {key[0]} (role {role})")
            problems += [{"graph": graph, "record": n, "problem": why} for n, _ in found.pop(key)]
        # two records of one attempt: which one is the attempt is unknown, so neither counts
        valid, names = {}, {}
        for key, recs in found.items():
            if len(recs) > 1:
                why = f"{len(recs)} records of one attempt {list(key)}: " + ", ".join(n for n, _ in recs)
                problems += [{"graph": graph, "record": n, "problem": why} for n, _ in recs[:1]]
                invalid[key] = why
            else:
                (names[key], valid[key]), = recs
                invalid.pop(key, None)
        # a branch record counts only when it was branched from its parent's record
        for key in [k for k in valid if k[1] in grid.parent_of]:
            model, cell, qid, rep = key
            parent = valid.get((model, grid.parent_of[cell], qid, rep))
            branched_from = (valid[key].get("checker") or {}).get("branched_from")
            if parent is None or parent.get("run_id") != branched_from:
                why = (f"branched from {branched_from!r}, but its parent {grid.parent_of[cell]} record is missing "
                       f"or does not count" if parent is None else
                       f"branched from {branched_from!r}, not from the parent record {parent.get('run_id')!r}")
                problems.append({"graph": graph, "record": names[key], "problem": why})
                invalid[key] = why
                del valid[key]
        per_cell: dict[tuple, dict[str, int]] = defaultdict(lambda: {"expected": 0, "covered": 0, "inherited": 0,
                                                                     "missing": 0})
        for a in planned:
            key = (a["model"], a["cell"], a["question_id"], a["repetition"])
            row = per_cell[(a["model"], a["cell"])]
            row["expected"] += 1
            item = {"graph": graph, "model": a["model"], "cell": a["cell"], "question_id": a["question_id"],
                    "repetition": a["repetition"]}
            if key in valid:
                row["covered"] += 1
                if a.get("mode") == "generic":
                    generic_flags.append({**item, "flagging_sources": flagging_sources(valid[key]["checker"])})
                continue
            if a["parent"] is not None:
                parent = valid.get(tuple(a["parent"]))
                status = ((parent or {}).get("outcome") or {}).get("status")
                if parent is not None and status != "submitted":
                    row["inherited"] += 1
                    inherited.append({**item, "parent_outcome": parent["outcome"].get("label", status)})
                    continue
                if parent is None and key not in invalid:
                    row["missing"] += 1
                    why = invalid.get(tuple(a["parent"]), "no record")
                    missing.append({**item, "reason": f"parent {a['parent'][1]}: {why}"})
                    continue
            row["missing"] += 1
            missing.append({**item, "reason": invalid.get(key, "no record" if runs.get(graph) else
                                                          "no run directory given for this graph")})
        rows += [{"graph": graph, "model": m, "role": manifest["roles"][m], "cell": c, **n}
                 for (m, c), n in per_cell.items()]
        if replays:
            j, m, pr = check_replay(manifest, graph, planned, valid, replays.get(graph, []))
            replay_jobs += j
            replay_missing += m
            replay_problems += pr
    counted = ("expected", "covered", "inherited", "missing")
    totals = {k: sum(r[k] for r in rows) for k in counted}
    totals["problems"] = len(problems)
    per_role = {role: {k: sum(r[k] for r in rows if manifest["roles"][r["model"]] == role) for k in counted}
                for role in dict.fromkeys(manifest["roles"].values())}
    phases = [{"phase": ph["phase"], "seed": ph["seed"], "starts": ph.get("starts"), "cells": ph["cells"],
               "graphs": {graph: {k: sum(r[k] for r in rows if r["graph"] == graph and r["cell"] in ph["cells"])
                                  for k in counted} for graph in manifest["spec"]["graphs"]}}
              for ph in manifest["order"]["phases"]]
    replay = {"checked": bool(replays), "complete": (not replay_missing and not replay_problems) if replays else None,
              "jobs": replay_jobs, "missing": replay_missing, "problems": replay_problems}
    return {"complete": not missing and not problems, "totals": totals, "per_role": per_role, "phases": phases,
            "cells": rows,
            "missing": missing, "inherited": inherited, "problems": problems, "generic_flags": generic_flags,
            "replay": replay}


def _parse_runs(items: list[str], graphs: list[str], flag: str = "--run") -> dict[str, list[Path]]:
    runs: dict[str, list[Path]] = defaultdict(list)
    for item in items:
        graph, sep, path = item.partition("=")
        if not sep or graph not in graphs:
            raise Refused(f"{flag} {item!r}: expected GRAPH=PATH with GRAPH one of {', '.join(graphs)}")
        exists = Path(path).is_dir() if flag == "--run" else (Path(path) / "matrix.jsonl.gz").is_file() or (
            Path(path).is_file())
        if not exists:
            raise Refused(f"{flag} {item!r}: {path} is not a run directory" if flag == "--run" else
                          f"{flag} {item!r}: no matrix.jsonl.gz at {path}")
        runs[graph].append(Path(path))
    return runs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest", type=Path, required=True, help="manifest.json from scripts/build_dev_grid.py")
    p.add_argument("--run", action="append", default=[], metavar="GRAPH=DIR",
                   help="a run output directory of one graph (repeatable)")
    p.add_argument("--replay", action="append", default=[], metavar="GRAPH=PATH",
                   help="an offline replay matrix of one graph: matrix.jsonl.gz or its directory (repeatable)")
    p.add_argument("--report", type=Path, default=None, help="write the full report here (JSON)")
    args = p.parse_args(argv)
    try:
        manifest, attempts = load_manifest(args.manifest)
        runs = _parse_runs(args.run, list(manifest["spec"]["graphs"]))
        replays = _parse_runs(args.replay, list(manifest["spec"]["graphs"]), "--replay")
    except (Refused, OSError, ValueError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    report = check_coverage(manifest, attempts, runs, replays)
    if args.report:
        args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t = report["totals"]
    print(f"{t['covered']} of {t['expected']} planned attempts recorded, {t['inherited']} inherited (parent did "
          f"not submit), {t['missing']} missing, {t['problems']} problems")
    for ph in report["phases"]:
        print(f"  phase {ph['phase']} (seed {ph['seed']}; {len(ph['cells'])} cells): "
              + "; ".join(f"{g} {n['covered'] + n['inherited']} of {n['expected']}" for g, n in ph["graphs"].items()))
    for row in report["cells"]:
        if row["missing"]:
            print(f"  {row['graph']} {row['model']} {row['cell']}: {row['missing']} of {row['expected']} missing")
    for item in report["missing"][:50]:
        print(f"  missing: {item['graph']} {item['model']} {item['cell']} {item['question_id']} "
              f"rep {item['repetition']}: {item['reason']}")
    for item in report["problems"][:50]:
        print(f"  problem: {item['graph']} {item['record']}: {item['problem']}")
    replay = report["replay"]
    if not replay["checked"]:
        print("replay: NOT CHECKED (no --replay given)")
    else:
        found = sum(j["found_rows"] for j in replay["jobs"])
        print(f"replay: {found} of {sum(j['expected_rows'] for j in replay['jobs'])} planned matrix rows, "
              f"{len(replay['missing'])} records with rows missing, {len(replay['problems'])} problems")
        for item in replay["missing"][:20]:
            print(f"  replay missing: {item['graph']} {item['model']} {item['cohort']} {item['question_id']} "
                  f"rep {item['repetition']}: {item['missing'] if isinstance(item['missing'], str) else len(item['missing'])}")
        for item in replay["problems"][:20]:
            print(f"  replay problem: {item['graph']} {item['row']}: {item['problem']}")
    ok = report["complete"] and replay["complete"] is not False
    print(("complete" if ok else "INCOMPLETE") + ("" if replay["checked"] else " (runs only)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
