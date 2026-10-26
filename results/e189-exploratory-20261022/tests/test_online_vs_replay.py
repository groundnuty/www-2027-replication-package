"""Tests of the online-against-replay disagreement counts (design ruling (B), 25 Oct 2026): results.json's ``online_vs_replay``.

The rule, written by hand (it is also ``exploratory.ONLINE_VS_REPLAY_RULE``): each online ledger row (use checker or both) has a
twin, the row of the same graph, model, cohort, run and source with use replay; the pair agrees when the two status letters are
equal, disagrees when they differ, and an online row with no twin is not comparable. Counts only, per online arm; the section
is empty when the ledger holds no replay row.

Hand-derived facts of the repository's synthetic grid-full fixture (tests/fixtures/analysis/grid-full), as its frozen exporter
writes the ledger of its branch records: for each of the two models, cohort "beta", online arms "curated" (use checker) and
"generic_retry/any" (use checker, named "generic" in the analysis), each with the model's three scored outputs, in the order
s01-pass (status P), s01-flag (F), s01-noquery (U); the fourth attempt, s01-abstain, has no output and so no row.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_exploratory import (E0, HAND_K, MODELS, QWEN, ROOT, _classes_file, _endpoint_table, _map, _run, ex,  # noqa: F401
                              world)  # (world is a fixture)

HERE = Path(__file__).resolve().parent
BASE_COMMIT = "351ad4a0"  # integration tip: all store and binding changes, before replay diagnostics
GEMMA = MODELS[1]


def _dump(result: dict) -> str:
    return json.dumps(result, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False)


def row(**kw):
    base = dict(graph="g", model="m", cohort="natural", run_id="r1", source="curated", use="checker", status="F")
    base.update(kw)
    return SimpleNamespace(**base)


def arm(model="m", cohort="natural", use="checker", source="curated", graph="g"):
    return f"{graph}|{model}|{cohort}|{use}|{source}"


def test_the_counts_of_a_hand_built_ledger():
    ledger = [
        # online curated rows of arm g|m|natural: r1 F/F agree, r2 F/P disagree, r3 U/U agree, r4 P/F disagree, r5 U/(none), r6 P/(none)
        row(run_id="r1", status="F"), row(run_id="r2", status="F"), row(run_id="r3", status="U"),
        row(run_id="r4", status="P"), row(run_id="r5", status="U"), row(run_id="r6", status="P"),
        row(run_id="r1", use="replay", status="F"), row(run_id="r2", use="replay", status="P"),
        row(run_id="r3", use="replay", status="U"), row(run_id="r4", use="replay", status="F"),
        # a replay twin of another cohort, model or graph is no twin; a replay row with no online row is not counted
        row(run_id="r5", use="replay", status="U", cohort="curated"), row(run_id="r6", use="replay", status="P", model="m2"),
        row(run_id="r7", use="replay", status="P"),
        # a "both" arm of another cohort: agrees on its one row
        row(run_id="r1", use="both", cohort="curated", status="P"), row(run_id="r1", use="replay", cohort="curated", status="P"),
        # a source with no replay arm at all (the generic control): not comparable
        row(run_id="r1", source="generic", status="F"), row(run_id="r2", source="generic", status="P"),
    ]
    got = ex.online_vs_replay(ledger)
    assert got["rule"] == ex.ONLINE_VS_REPLAY_RULE
    assert got["arms"] == {
        arm(): {"agree": 2, "disagree": 2, "not_comparable": 2},
        arm(cohort="curated", use="both"): {"agree": 1, "disagree": 0, "not_comparable": 0},
        arm(source="generic"): {"agree": 0, "disagree": 0, "not_comparable": 2},
    }


def test_the_section_is_empty_without_a_replay_row_and_holds_counts_only():
    ledger = [row(run_id="r1"), row(run_id="r2", status="P")]
    assert ex.online_vs_replay(ledger)["arms"] == {}
    assert ex.online_vs_replay([])["arms"] == {}
    only_replay = [row(run_id="r1", use="replay")]
    assert ex.online_vs_replay(only_replay)["arms"] == {}  # no online row: nothing to compare
    got = ex.online_vs_replay([row(run_id="r1"), row(run_id="r1", use="replay", status="P")])
    assert set(got) == {"rule", "arms"}
    assert all(set(v) == {"agree", "disagree", "not_comparable"} and all(type(n) is int for n in v.values())
               for v in got["arms"].values())
    assert got["arms"] == {arm(): {"agree": 0, "disagree": 1, "not_comparable": 0}}


def _script_at(commit: str, tmp: Path):
    """exploratory.py as it was at ``commit`` (git show), beside the other files of its directory, as a second module."""
    shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:results/e189-exploratory-20261022/exploratory.py"],
                           capture_output=True, check=False)
    assert shown.returncode == 0, f"{commit} is not in this repository: fetch it"
    where = tmp / "base-script" / HERE.parent.name
    where.mkdir(parents=True)
    for f in HERE.parent.iterdir():
        if f.name not in ("exploratory.py", "tests", "dry-run", "__pycache__"):
            (where / f.name).symlink_to(f)
    (where / "exploratory.py").write_bytes(shown.stdout)
    spec = importlib.util.spec_from_file_location("e189_exploratory_before_matrices", where / "exploratory.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e189_exploratory_before_matrices"] = module
    spec.loader.exec_module(module)
    return module, where


def test_a_call_without_matrices_gives_the_base_commits_results_byte_for_byte_apart_from_the_empty_section(world):
    base, where = _script_at(BASE_COMMIT, world["tmp"])
    Fb = base.load_frozen(ROOT)
    cm, rec = _map()
    old = base.run(Fb, base.Call((world["single"],), (world["questions"],), "s01", "dev", (world["voided"],), None,
                                 (world["ledger"],), 1, None), K_clusters=HAND_K, cluster_map=cm, map_record=rec)
    new = _run(world, [world["single"]], record=False, ledgers=(world["ledger"],), K=HAND_K)
    assert new["online_vs_replay"] == {"rule": ex.ONLINE_VS_REPLAY_RULE, "arms": {}}
    assert new["stores"]["replay_class_check"] == EMPTY_CLASS_CHECK
    # The integration tip already includes the other branch's store changes; only replay diagnostics are new.
    assert "online_vs_replay" not in old and "replay_class_check" not in old["stores"]
    del new["online_vs_replay"], new["stores"]["replay_class_check"]
    # as main() writes a result; the one other difference is the script's own directory, which the result records
    assert _dump(new) == _dump(old).replace(str(where), str(HERE.parent))


def _plain_records(batch: Path, cohorts=None):
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((batch / "records").glob("*.json"))]
    return [r for r in recs if r.get("checker") is None and (cohorts is None or r["load"]["arm"] in cohorts)]


def test_replay_rows_from_a_matrix_give_the_counts_by_hand(world, capsys):
    # The whole path through the FROZEN exporter: a replay matrix of every plain record under the arm "curated" (every row a
    # substantive pass, letter P), the fixture's own online rows, and the analysis. By hand, per model, the online "curated" arm of
    # cohort beta has s01-pass P (agrees with the replay P), s01-flag F and s01-noquery U (both disagree with P); the generic control
    # has no replay twin.
    from test_exploratory import FIXTURE, exporter
    batch = FIXTURE / "batch"
    matrix = _matrix_file(world["tmp"] / "all-cohorts.gz", _plain_records(batch))
    ledger = world["tmp"] / "with-replay.jsonl"
    assert exporter.main(["--split", "dev", "--matrix", str(matrix), "--records", str(batch), "--out", str(ledger)]) == 0
    capsys.readouterr()
    result = _run(world, [world["single"]], record=False, ledgers=(ledger,), K=HAND_K)
    assert result["online_vs_replay"] == {"rule": ex.ONLINE_VS_REPLAY_RULE, "arms": {
        f"s01|{GEMMA}|beta|checker|curated": {"agree": 1, "disagree": 2, "not_comparable": 0},
        f"s01|{GEMMA}|beta|checker|generic": {"agree": 0, "disagree": 0, "not_comparable": 3},
        f"s01|{QWEN}|beta|checker|curated": {"agree": 1, "disagree": 2, "not_comparable": 0},
        f"s01|{QWEN}|beta|checker|generic": {"agree": 0, "disagree": 0, "not_comparable": 3},
    }}
    # the replay arms exist beside the online ones, for every cohort that has outputs
    for model in (QWEN, GEMMA):
        for cohort in ("none", "alpha", "beta"):
            assert f"s01|{model}|{cohort}|replay|curated" in result["arms"]


def test_a_cohort_whose_rows_all_say_no_output_is_refused_by_the_analysis(world, capsys):
    # the exact failure an independent reviewer reproduced: every row of cohort none is consistent (output false, no label), so the
    # matrices pass every check of the matrices, the frozen exporter skips the cohort's rows, and only the analysis knows its
    # scored outputs
    from test_exploratory import FIXTURE, exporter
    batch = FIXTURE / "batch"
    matrix = _matrix_file(world["tmp"] / "none-no-output.gz", _plain_records(batch), no_output_cohorts={"none"})
    ledger = world["tmp"] / "none-no-output.jsonl"
    assert exporter.main(["--split", "dev", "--matrix", str(matrix), "--records", str(batch), "--out", str(ledger)]) == 0
    capsys.readouterr()
    with pytest.raises(ex.Refused, match=r"2 replay arms are missing for cohorts that have scored outputs"):
        _run(world, [world["single"]], record=False, ledgers=(ledger,), K=HAND_K)


def test_a_cohort_left_out_of_the_replay_matrices_is_refused_by_the_analysis(world, capsys):
    # Independent review: build_arms never creates, so never checks, a replay arm for a cohort with no replay row. Matrices of
    # cohort beta alone leave none and alpha, which have scored outputs, without any: the analysis refuses.
    from test_exploratory import FIXTURE, exporter
    batch = FIXTURE / "batch"
    matrix = _matrix_file(world["tmp"] / "beta-only.gz", _plain_records(batch, {"beta"}))
    ledger = world["tmp"] / "beta-only.jsonl"
    assert exporter.main(["--split", "dev", "--matrix", str(matrix), "--records", str(batch), "--out", str(ledger)]) == 0
    capsys.readouterr()
    with pytest.raises(ex.Refused, match=r"4 replay arms are missing for cohorts that have scored outputs"):
        _run(world, [world["single"]], record=False, ledgers=(ledger,), K=HAND_K)


# -- what the design relies on in the frozen exporter ------------------------------------------------------------------------


def _matrix_file(path: Path, records, arm: str = "curated", no_output_cohorts=()) -> Path:
    """A replay matrix (gzip JSON lines) of ``records``' initial outputs under one arm, the primary variant: the fields
    ``ledger.from_matrix_row`` reads, written by hand from each record (no checker run)."""
    import gzip
    rows = []
    for r in records:
        out = r["outcome"]["status"] == "submitted" and r["load"]["arm"] not in no_output_cohorts  # a cohort's rows all say output false
        rows.append({"version": "checker-replay-matrix/1", "run_id": r["run_id"], "graph": r["store"]["name"],
                     "question_id": r["question_id"], "base_question_id": r.get("base_question_id"),
                     "template": r.get("template"), "split": r["split"], "model": r["model"], "cohort": r["load"]["arm"],
                     "repetition": r.get("repetition"), "output": out,
                     "correct": (r.get("score") or {}).get("exact") if out else None, "arm": arm, "variant": "primary",
                     "status": "substantive_pass", "any_unknown": False, "fired_checks": [], "traceability": None,
                     "checked_query_status": "aligned", "result_rows": 1, "budget_used": 0})
    path.write_bytes(gzip.compress("".join(json.dumps(x) + "\n" for x in rows).encode(), mtime=0))
    return path


def test_the_frozen_exporter_takes_class_cell_matrices_as_they_are_and_refuses_a_row_twice(world, capsys):
    from test_exploratory import FIXTURE, exporter
    batch = FIXTURE / "batch"
    plain = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((batch / "records").glob("*.json"))]
    plain = [r for r in plain if r["load"]["arm"] == "none" and r.get("checker") is None]
    by_model = {m: [r for r in plain if r["model"] == m] for m in MODELS}
    assert [len(v) for v in by_model.values()] == [4, 4]  # the fixture: four natural attempts per model, three with an output
    tmp = world["tmp"]
    cell_a = _matrix_file(tmp / "cell-a.gz", by_model[QWEN])  # two "class cells": disjoint record sets, one OUT
    cell_b = _matrix_file(tmp / "cell-b.gz", by_model[GEMMA])
    out = tmp / "cells-ledger.jsonl"
    rc = exporter.main(["--split", "dev", "--matrix", str(cell_a), str(cell_b), "--records", str(batch), "--out", str(out)])
    assert rc == 0, capsys.readouterr().err
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()[1:]]
    replay = [r for r in rows if r["use"] == "replay"]
    assert len(replay) == 6 and {r["source"] for r in replay} == {"curated"} and {r["cohort"] for r in replay} == {"none"}
    assert len(rows) == 6 + 12  # the fixture's twelve online rows, unchanged
    # the same arm again (the OBQC pass's matrix holds the online pass's arms once more): a row twice refuses the export
    twice = tmp / "twice-ledger.jsonl"
    rc = exporter.main(["--split", "dev", "--matrix", str(cell_a), str(cell_b), str(cell_a), "--records", str(batch),
                        "--out", str(twice)])
    err = capsys.readouterr().err
    assert rc == 2 and "duplicate ledger rows" in err and not twice.exists()
    # a matrix whose records are not in the batch given: no provenance, refused
    rc = exporter.main(["--split", "dev", "--matrix", str(cell_a), "--out", str(tmp / "no-records.jsonl")])
    assert rc == 2 and "no record of a batch given with --records" in capsys.readouterr().err


# -- every replay row on a copy of its record's own class (science, 25 Oct 11:04Z) ---------------------------------------------

EMPTY_CLASS_CHECK = {"rule": ex.REPLAY_CLASS_RULE, "labels": dict(ex.REPLAY_CLASS_LABEL_TEMPLATES),
                     "routing": [], "arms": {}}
REAL = ("fast", "medium", "slow")


def _routing_doc(out, records, *, kind="main", classes=None, attested=(), pending=(), taken="20261025T123234Z") -> dict:
    """A schema-5 routing file by hand. ``records``: run id -> class (a real class, "unverifiable" or "not_recorded"); one shard
    per (class, basis): basis "attested" for the run ids in ``attested``, else "proven" (kind "cell": tree ``<out>-cell-<class>``)
    or "recorded" (kind "main": tree ``<out>``) for a real class, none otherwise. ``classes`` is ignored (an older helper's)."""
    groups: dict = {}
    for rid, cls in records.items():
        basis = None if cls not in REAL else "attested" if rid in attested else "proven" if kind == "cell" else "recorded"
        groups.setdefault((cls, basis), []).append(rid)
    shards, tfs, cover = {}, {}, {}
    for i, ((cls, basis), ids) in enumerate(sorted(groups.items(), key=str)):
        tree = out if kind == "main" else f"{out}-cell-{cls if cls in REAL else 'fast'}"
        key = f"{tree}/natural/s{i}"
        copy = {"endpoint": "192.0.2.101:7033", "copy": f"copy-{cls}"} if cls in REAL else None
        shards[key] = {"class": "unverifiable" if cls == "not_recorded" else cls, "basis": basis, "copy": copy, "runs": sorted(ids)}
        if cls == "not_recorded":
            shards[key]["subkind"] = "pending_attestation" if any(r in pending for r in ids) else "no_launch_record"
        if basis in ("proven", "recorded"):
            tfs[key] = "b" * 64
        if basis == "attested":
            cover[key] = {"tree": tree, "shard": f"s{i}", "class": cls, "copy": copy["copy"], "endpoint": copy["endpoint"]}
    atts = ([{"out": out, "phase": "natural", "shards": [cover[t] for t in sorted(cover)], "file_sha256": "c" * 64,
              "content_sha256": "d" * 64, "agent": "tier2-replay"}] if cover else [])
    return {"schema": "replay-routing/5", "out": out, "shards": shards,
            "inputs": {"cell_classes_sha256": "a" * 64, "time_files_sha256": tfs, "attestations": atts,
                       "timefiles_manifest": {"file_sha256": "e" * 64, "taken_at": taken,
                                              "manifests": [{"tree": tree, "phase": "natural", "file_sha256": "f" * 64}
                                                            for tree in sorted({key.split("/")[0] for key in shards})]}}}


def _replay_world(world, capsys, *, routing=None, named=True, attested=(), pending=(), doc_hook=None):
    """The fixture's records, all of cohorts none/alpha/beta replayed under the arm curated through the FROZEN exporter, and a
    routing file by hand: ``routing(run_id) -> class or None (left out)``. ``named``: the endpoint is a table-named copy of
    class fast (copy-a), so each record's own class is known; else the label has no copy class."""
    from test_exploratory import FIXTURE, exporter
    batch = FIXTURE / "batch"
    plain = _plain_records(batch)
    matrix = _matrix_file(world["tmp"] / "m.gz", plain)
    ledger = world["tmp"] / "l.jsonl"
    assert exporter.main(["--split", "dev", "--matrix", str(matrix), "--records", str(batch), "--out", str(ledger)]) == 0
    capsys.readouterr()
    kw = {"ledgers": (ledger,)}
    if named:
        kw.update(endpoint_copies=_endpoint_table(world["tmp"] / "ec.json", ("s01", E0, "copy-a")),
                  copy_classes=_classes_file(world["tmp"] / "cc.json"))
    if routing is not None:
        recs = {r["run_id"]: routing(r["run_id"]) for r in plain if routing(r["run_id"]) is not None}
        path = world["tmp"] / "routing.json"
        doc = _routing_doc("single", recs, attested=tuple(a for a in attested if a in recs), pending=pending)
        if doc_hook is not None:
            doc_hook(doc)
        path.write_text(json.dumps(doc))
        kw["replay_routing"] = (path,)
    return _run(world, [world["single"]], record=False, K=HAND_K, **kw)["stores"]["replay_class_check"], plain


def _arm_counts(check, model, cohort):
    return check["arms"][f"s01|{model}|{cohort}|replay|curated"]


def test_every_replay_row_on_a_copy_of_its_records_class_is_counted_as_such(world, capsys):
    check, plain = _replay_world(world, capsys, routing=lambda rid: "fast")
    assert check["rule"] == ex.REPLAY_CLASS_RULE and len(check["routing"]) == 1 and check["routing"][0]["records"] == len(plain)
    assert len(check["arms"]) == 6  # two models x cohorts none, alpha, beta
    for model in MODELS:
        for cohort in ("none", "alpha", "beta"):  # three scored outputs per model and cohort (the fourth attempt has none)
            assert _arm_counts(check, model, cohort) == {"rows": 3, "same_class_proven": 0, "same_class_recorded": 3, "same_class_attested": 0, "other_class": 0, "record_class_unknown": 0,
                                                         "no_launch_record": 0, "pending_attestation": 0, "replay_class_unverified": 0, "not_routed": 0,
                                                         "exceptions": 0}


def test_an_injected_cross_class_replay_row_is_counted(world, capsys):
    from test_exploratory import FIXTURE
    plain = _plain_records(FIXTURE / "batch")
    victim = next(r for r in plain if r["model"] == QWEN and r["load"]["arm"] == "alpha" and r["outcome"]["status"] == "submitted")
    check, _ = _replay_world(world, capsys, routing=lambda rid: "slow" if rid == victim["run_id"] else "fast")
    assert _arm_counts(check, QWEN, "alpha") == {"rows": 3, "same_class_proven": 0, "same_class_recorded": 2, "same_class_attested": 0, "other_class": 1, "record_class_unknown": 0,
                                                 "no_launch_record": 0, "pending_attestation": 0, "replay_class_unverified": 0, "not_routed": 0,
                                                 "exceptions": 1}
    for other in (("none", QWEN), ("beta", QWEN), ("alpha", MODELS[1])):  # the other arms are untouched
        assert _arm_counts(check, other[1], other[0])["exceptions"] == 0
    assert sum(a["exceptions"] for a in check["arms"].values()) == 1


def test_an_unrecorded_an_unverifiable_a_missing_and_an_unknown_class_are_exceptions_of_their_own_kinds(world, capsys):
    from test_exploratory import FIXTURE
    plain = _plain_records(FIXTURE / "batch")
    outs = [r for r in plain if r["model"] == QWEN and r["load"]["arm"] == "none" and r["outcome"]["status"] == "submitted"]
    unverified, missing, unrecorded = outs[0]["run_id"], outs[1]["run_id"], outs[2]["run_id"]
    check, _ = _replay_world(world, capsys, routing=lambda rid: "unverifiable" if rid == unverified else None if rid == missing
                             else "not_recorded" if rid == unrecorded else "fast")
    assert _arm_counts(check, QWEN, "none") == {"rows": 3, "same_class_proven": 0, "same_class_recorded": 0, "same_class_attested": 0, "other_class": 0, "record_class_unknown": 0,
                                                "no_launch_record": 1, "pending_attestation": 0, "replay_class_unverified": 2, "not_routed": 1,
                                                "exceptions": 3}
    # the record's own copy has no class (the endpoint is no table-named copy): the first kind that applies, per row
    check, _ = _replay_world(world, capsys, routing=lambda rid: "fast", named=False)
    for arm in check["arms"].values():
        assert arm["record_class_unknown"] == 3 and arm["same_class_recorded"] == 0 and arm["exceptions"] == 3
    # replay rows and no routing file at all: every row is unrouted
    check, _ = _replay_world(world, capsys, routing=None)
    assert check["routing"] == [] and all(a["not_routed"] == 3 and a["exceptions"] == 3 for a in check["arms"].values())
    # a copy that was not recorded is a count of its own, not folded into "unverified" (analysis lead, 11:24Z)
    check, _ = _replay_world(world, capsys, routing=lambda rid: "not_recorded")
    for arm in check["arms"].values():
        assert arm["no_launch_record"] == 3 and arm["replay_class_unverified"] == 3 and arm["exceptions"] == 3


def test_pending_attestation_is_a_labelled_unverified_subcount(world, capsys):
    from test_exploratory import FIXTURE
    plain = _plain_records(FIXTURE / "batch")
    victim = next(r["run_id"] for r in plain if r["model"] == QWEN and r["load"]["arm"] == "none"
                  and r["outcome"]["status"] == "submitted")
    check, _ = _replay_world(world, capsys, routing=lambda rid: "not_recorded" if rid == victim else "fast",
                             pending=(victim,))
    assert check["labels"]["pending_attestation"] == "pending attestation"
    arm = _arm_counts(check, QWEN, "none")
    assert arm["replay_class_unverified"] == arm["no_launch_record"] == arm["pending_attestation"] == 1
    assert arm["exceptions"] == 1


def test_a_call_without_replay_rows_has_an_empty_class_check(world):
    result = _run(world, [world["single"]], record=False, ledgers=(world["ledger"],), K=HAND_K)
    assert result["stores"]["replay_class_check"] == EMPTY_CLASS_CHECK


def _copy(d):
    return json.loads(json.dumps(d))


def test_the_replay_routing_files_are_read_strictly(tmp_path):
    good = tmp_path / "g.json"
    good.write_text(json.dumps(_routing_doc("o", {"r1": "fast", "r2": "unverifiable", "r4": "not_recorded"})))
    got = ex.read_replay_routing([good])
    assert dict(got.classes) == {"r1": "fast", "r2": "unverifiable", "r4": "unverifiable"} and dict(got.basis) == {"r1": "recorded"}
    assert got.files[0]["records"] == 3 and got.files[0]["out"] == "o" and got.files[0]["timefiles_manifest"]["taken_at"] == "20261025T123234Z"
    assert set(got.files[0]) == {"sha256", "out", "records", "timefiles_manifest", "attestations"}
    other = tmp_path / "o.json"
    other.write_text(json.dumps(_routing_doc("p", {"r3": "slow"}, kind="cell")))
    r = ex.read_replay_routing([good, other])
    assert dict(r.classes)["r3"] == "slow" and dict(r.basis)["r3"] == "proven"
    twin = tmp_path / "t.json"
    twin.write_text(json.dumps(_routing_doc("q", {"r1": "slow"}, kind="cell")))
    with pytest.raises(ex.Refused, match="1 run ids are in two replay routing files"):
        ex.read_replay_routing([good, twin])
    for text, why in (("[1]", "not a replay-routing/5 file"), ("not json", "is not JSON")):
        bad = tmp_path / "b.json"
        bad.write_text(text)
        with pytest.raises(ex.Refused, match=why):
            ex.read_replay_routing([bad])


def test_the_routing_reader_ties_every_basis_to_its_evidence(tmp_path):
    base = _routing_doc("o", {"r1": "fast", "r2": "fast", "r3": "slow"}, attested=("r2",))
    keys = sorted(base["shards"])
    rec = next(k for k, v in base["shards"].items() if v["basis"] == "recorded")
    att = next(k for k, v in base["shards"].items() if v["basis"] == "attested")
    cell = _routing_doc("o", {"r9": "fast"}, kind="cell")
    ckey = next(iter(cell["shards"]))

    def mut(f, doc=base):
        d = _copy(doc)
        f(d)
        return d

    cases = {
        "schema-4": (mut(lambda d: d.update(schema="replay-routing/4")), "not a replay-routing/5 file"),
        "extra-top-key": (mut(lambda d: d.update(records={})), "not a replay-routing/5 file"),
        "no-inputs": (mut(lambda d: d.pop("inputs")), "not a replay-routing/5 file"),
        "key-two-parts": (mut(lambda d: d["shards"].update({"o/s": {"class": "fast", "basis": "recorded", "runs": ["x"]}})), "not a replay-routing/5"),
        "shard-extra-key": (mut(lambda d: d["shards"][rec].update(x=1)), "not a replay-routing/5"),
        "bad-class": (mut(lambda d: d["shards"][rec].update(**{"class": "quick"})), "not a replay-routing/5"),
        "real-class-no-basis": (mut(lambda d: d["shards"][rec].update(basis=None)), "not a replay-routing/5"),
        "bad-basis": (mut(lambda d: d["shards"][rec].update(basis="inferred")), "not a replay-routing/5"),
        "unverifiable-with-basis": (mut(lambda d: d["shards"][rec].update(**{"class": "unverifiable"})), "not a replay-routing/5"),
        "no-runs": (mut(lambda d: d["shards"][rec].update(runs=[])), "not a replay-routing/5"),
        "dup-runs": (mut(lambda d: d["shards"][rec].update(runs=["r1", "r1"])), "not a replay-routing/5"),
        "run-in-two-shards": (mut(lambda d: d["shards"][rec].update(runs=["r1", "r2"])), "not a replay-routing/5"),
        "other-phase": (mut(lambda d: d["shards"].update({rec.replace("natural", "branch"): d["shards"].pop(rec)})), "not a replay-routing/5"),
        "tree-of-another-out": (mut(lambda d: d["shards"].update({"x/natural/s": d["shards"].pop(rec)})), "not a replay-routing/5"),
        "proven-in-main": (mut(lambda d: d["shards"][rec].update(basis="proven")), "'proven' without being a class cell"),
        "recorded-in-cell": (mut(lambda d: d["shards"][ckey].update(basis="recorded"), cell), "class cell's shard is 'recorded'"),
        "proven-wrong-class": (mut(lambda d: d["shards"][ckey].update(**{"class": "slow"}), cell), "'proven' without being a class cell of its own class"),
        "classed-without-time-sha": (mut(lambda d: d["inputs"]["time_files_sha256"].pop(rec)), "classed shard has no time file sha256"),
        "time-sha-for-no-shard": (mut(lambda d: d["inputs"]["time_files_sha256"].update({"o/natural/zz": "a" * 64})), "names no shard"),
        "attested-with-time-sha": (mut(lambda d: d["inputs"]["time_files_sha256"].update({att: "a" * 64})), "attested shard is not tied"),
        "attested-no-attestation": (mut(lambda d: d["inputs"].update(attestations=[])), "attested shard is not tied"),
        "attestation-other-out": (mut(lambda d: d["inputs"]["attestations"][0].update(out="p")), "an attestation names another OUT"),
        # external reviewer r18: an attested shard's class and copy must be the attestation's own for that tree
        "attested-class-changed": (mut(lambda d: d["shards"][att].update(**{"class": "slow"})), "attested shard is not tied"),
        "attested-copy-changed": (mut(lambda d: d["shards"][att]["copy"].update(copy="other")), "attested shard is not tied"),
        "attested-endpoint-changed": (mut(lambda d: d["shards"][att]["copy"].update(endpoint="1.2.3.4:5")), "attested shard is not tied"),
        "classed-without-copy": (mut(lambda d: d["shards"][rec].update(copy=None)), "not a replay-routing/5"),
        "copy-extra-key": (mut(lambda d: d["shards"][rec]["copy"].update(x="y")), "not a replay-routing/5"),
        "unverifiable-with-copy": (mut(lambda d: d["shards"][rec].update(**{"class": "unverifiable", "basis": None})), "not a replay-routing/5"),
        "attestation-tree-detail-class": (mut(lambda d: d["inputs"]["attestations"][0]["shards"][0].update(**{"class": "quick"})), "not a replay-routing/5"),
        "attestation-tree-as-string": (mut(lambda d: d["inputs"]["attestations"][0].update(shards=["o"])), "not a replay-routing/5"),
        "attestation-other-phase": (mut(lambda d: d["inputs"]["attestations"][0].update(phase="prompted")), "attested shard is not tied"),
        "attestation-other-tree": (mut(lambda d: d["inputs"]["attestations"][0].update(
            shards=[{"tree": "o-cell-fast", "shard": "s1", "class": "fast", "copy": "copy-fast", "endpoint": "192.0.2.101:7033"}])), "attested shard is not tied"),
        "attestation-extra-key": (mut(lambda d: d["inputs"]["attestations"][0].update(x=1)), "not a replay-routing/5"),
        "manifest-missing": (mut(lambda d: d["inputs"].pop("timefiles_manifest")), "not a replay-routing/5"),
        "manifest-bad-time": (mut(lambda d: d["inputs"]["timefiles_manifest"].update(taken_at="yesterday")), "not a replay-routing/5"),
        "manifest-bad-sha": (mut(lambda d: d["inputs"]["timefiles_manifest"].update(file_sha256="short")), "not a replay-routing/5"),
    }
    for name, (doc, why) in cases.items():
        path = tmp_path / "c.json"
        path.write_text(json.dumps(doc))
        with pytest.raises(ex.Refused, match=why):
            ex.read_replay_routing([path])
        _ = name
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(base))
    got = ex.read_replay_routing([ok])
    assert dict(got.basis) == {"r1": "recorded", "r2": "attested", "r3": "recorded"}
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps(base).replace('"out": "o"', '"out": "p", "out": "o"', 1))
    with pytest.raises(ex.Refused, match="repeats a key"):
        ex.read_replay_routing([dup])


def test_each_basis_reads_under_its_own_label_with_the_manifest_time(world, capsys):
    from test_exploratory import FIXTURE
    plain = _plain_records(FIXTURE / "batch")
    outs = [r for r in plain if r["model"] == QWEN and r["load"]["arm"] == "none" and r["outcome"]["status"] == "submitted"]
    attested = {outs[0]["run_id"]}
    check, _ = _replay_world(world, capsys, routing=lambda rid: "fast", attested=attested)
    assert check["labels"] == {"same_class_proven": "proven",
                               "same_class_recorded": "recorded (worker launch record, unsealed; sha manifest taken 2026-10-25T12:32Z)",
                               "same_class_attested": "operator attestation (from glue records)",
                               "pending_attestation": "pending attestation"}
    assert _arm_counts(check, QWEN, "none") == {"rows": 3, "same_class_proven": 0, "same_class_recorded": 2, "same_class_attested": 1,
                                                "other_class": 0, "record_class_unknown": 0, "no_launch_record": 0,
                                                "pending_attestation": 0,
                                                "replay_class_unverified": 0, "not_routed": 0, "exceptions": 0}  # a match is no exception
    assert _arm_counts(check, QWEN, "alpha")["same_class_recorded"] == 3
    assert check["routing"][0]["attestations"][0]["agent"] == "tier2-replay"
    assert check["routing"][0]["timefiles_manifest"]["taken_at"] == "20261025T123234Z"
    # an attested row whose class is not the record's own is an exception like any other
    check, _ = _replay_world(world, capsys, routing=lambda rid: "slow", attested=attested)
    arm = _arm_counts(check, QWEN, "none")
    assert arm["other_class"] == 3 and arm["same_class_attested"] == 0 and arm["exceptions"] == 3


TAG_A, TAKEN_A = "20261025T123234Z", "2026-10-25T12:32:38Z"  # the producer's first incremental time-file index
TAG_B, TAKEN_B = "20261025T153123Z", "2026-10-25T15:31:25Z"  # its second: the tree-phases taken later


def _consolidate(doc, *, moved=None):
    """``doc`` (a main-tree routing whose one recorded shard is natural) as the checker writes it from the consolidated index: the
    recorded shard's first ``moved`` runs (all, by default) go to a shard of the cell's other tree-phase, prompted. The natural
    tree-phase was covered by index A (taken 12:32:38Z), the prompted one by index B (taken 15:31:25Z); the union file's own tag
    (18:51:04Z) is a consolidation time and no row's."""
    key = next(k for k, v in doc["shards"].items() if v["basis"] == "recorded")
    info, tree = doc["shards"][key], key.split("/")[0]
    runs = info["runs"]
    moved = len(runs) if moved is None else moved
    new = f"{tree}/prompted/s9"
    doc["shards"][new] = {**info, "runs": runs[:moved]}
    doc["inputs"]["time_files_sha256"][new] = "b" * 64
    if moved == len(runs):
        del doc["shards"][key]
        del doc["inputs"]["time_files_sha256"][key]
    else:
        info["runs"] = runs[moved:]
    doc["inputs"]["timefiles_manifest"] = {
        "file_sha256": "e" * 64, "taken_at": "20261025T185104Z",
        "manifests": [{"tree": tree, "phase": "natural", "file_sha256": "f" * 64, "source_index": TAG_A, "taken_at": TAKEN_A},
                      {"tree": tree, "phase": "prompted", "file_sha256": "f" * 64, "source_index": TAG_B, "taken_at": TAKEN_B}],
        "sources": [{"tag": TAG_A, "file_sha256": "1" * 64, "verified": True}, {"tag": TAG_B, "file_sha256": "2" * 64, "verified": False}]}
    return doc


def test_the_recorded_label_names_the_time_of_the_manifest_that_covered_the_counted_shards(world, capsys):
    label = "recorded (worker launch record, unsealed; sha manifest taken {})"
    # a routing from the single index taken at 15:31:23Z (not the 12:32Z one): the label is not a constant
    check, plain = _replay_world(world, capsys, routing=lambda rid: "fast",
                                 doc_hook=lambda d: d["inputs"]["timefiles_manifest"].update(taken_at="20261025T153123Z"))
    assert check["labels"]["same_class_recorded"] == label.format("2026-10-25T15:31Z")
    # the consolidated index: the cell's natural shard was covered by the 12:32Z row, its prompted shard by the 15:31Z row
    half = len(plain) // 2
    check, _ = _replay_world(world, capsys, routing=lambda rid: "fast", doc_hook=lambda d: _consolidate(d, moved=half))
    assert check["labels"]["same_class_recorded"] == label.format("2026-10-25T12:32Z, 2026-10-25T15:31Z")
    # every counted row is of the prompted shard: the label has the 15:31Z time alone, though the file lists both manifests
    # and the union file's own tag is 18:51Z
    check, _ = _replay_world(world, capsys, routing=lambda rid: "fast", doc_hook=lambda d: _consolidate(d))
    assert check["labels"]["same_class_recorded"] == label.format("2026-10-25T15:31Z")
    assert len(check["routing"][0]["timefiles_manifest"]["manifests"]) == 2
    # rows replayed on another class are no recorded rows: the 12:32Z shard's rows are all of another class, so its manifest
    # time is not in the label; the one same-class row is of the prompted shard (the fast group moves there whole)
    first = sorted(r["run_id"] for r in plain)[0]
    check, _ = _replay_world(world, capsys, routing=lambda rid: "fast" if rid == first else "slow", doc_hook=lambda d: _consolidate(d))
    assert check["labels"]["same_class_recorded"] == label.format("2026-10-25T15:31Z")
    assert sum(a["same_class_recorded"] for a in check["arms"].values()) == 1
    # no row counted as recorded: the template, which names no time
    check, _ = _replay_world(world, capsys, routing=lambda rid: "not_recorded")
    assert check["labels"]["same_class_recorded"] == label.format("<the time of the sha manifest row that covered the shard's tree-phase>")


def test_the_routing_reader_gives_each_recorded_run_the_time_of_its_own_tree_phases_manifest(tmp_path):
    runs = {"r1": "fast", "r2": "fast", "r3": "fast"}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(_consolidate(_routing_doc("o", runs), moved=1)))  # r1 is in the prompted shard
    got = ex.read_replay_routing([path])
    assert dict(got.recorded_taken) == {"r1": "2026-10-25T15:31Z", "r2": "2026-10-25T12:32Z", "r3": "2026-10-25T12:32Z"}
    man = got.files[0]["timefiles_manifest"]
    assert man["taken_at"] == "20261025T185104Z" and [m["source_index"] for m in man["manifests"]] == [TAG_A, TAG_B]
    assert man["sources"] == [{"tag": TAG_A, "file_sha256": "1" * 64, "verified": True},
                              {"tag": TAG_B, "file_sha256": "2" * 64, "verified": False}]
    # a single index: the file's tag is the time of every tree-phase
    path.write_text(json.dumps(_routing_doc("o", runs, taken="20261025T162644Z")))
    assert dict(ex.read_replay_routing([path]).recorded_taken) == {r: "2026-10-25T16:26Z" for r in runs}
    # a class cell's shard is proven, not recorded: no manifest time
    path.write_text(json.dumps(_routing_doc("o", runs, kind="cell")))
    assert dict(ex.read_replay_routing([path]).recorded_taken) == {}


def test_the_routing_reader_takes_the_consolidated_form_whole_or_not_at_all(tmp_path):
    base = _consolidate(_routing_doc("o", {"r1": "fast", "r2": "fast"}), moved=1)

    def mut(f):
        d = _copy(base)
        f(d)
        return d
    man = lambda d: d["inputs"]["timefiles_manifest"]  # noqa: E731
    single = _routing_doc("o", {"r1": "fast"})
    cases = {
        "entry-without-taken-at": mut(lambda d: man(d)["manifests"][0].pop("taken_at")),
        "entry-without-source-index": mut(lambda d: man(d)["manifests"][1].pop("source_index")),
        "entry-of-an-unlisted-source": mut(lambda d: man(d)["manifests"][1].update(source_index="20261025T162644Z")),
        "entry-time-in-tag-form": mut(lambda d: man(d)["manifests"][0].update(taken_at=TAG_A)),
        "entry-time-without-seconds": mut(lambda d: man(d)["manifests"][0].update(taken_at="2026-10-25T12:32Z")),
        "entry-time-month-13": mut(lambda d: man(d)["manifests"][0].update(taken_at="2026-13-25T12:32:38Z")),
        "entry-time-not-text": mut(lambda d: man(d)["manifests"][0].update(taken_at=1234)),
        "two-times-for-one-source": mut(lambda d: man(d)["manifests"].append(
            {"tree": "o", "phase": "prompted", "file_sha256": "f" * 64, "source_index": TAG_A, "taken_at": "2026-10-25T12:33:38Z"})),
        "tree-phase-twice": mut(lambda d: man(d)["manifests"].append(dict(man(d)["manifests"][0]))),
        "no-sources": mut(lambda d: man(d).update(sources=[])),
        "sources-not-a-list": mut(lambda d: man(d).update(sources={})),
        "source-tag-repeated": mut(lambda d: man(d)["sources"].append(dict(man(d)["sources"][0]))),
        "source-tag-bad": mut(lambda d: man(d)["sources"][0].update(tag="20261025")),
        "source-sha-short": mut(lambda d: man(d)["sources"][0].update(file_sha256="abc")),
        "source-verified-not-a-bool": mut(lambda d: man(d)["sources"][0].update(verified="yes")),
        "source-extra-key": mut(lambda d: man(d)["sources"][0].update(note="x")),
        "file-tag-bad": mut(lambda d: man(d).update(taken_at="yesterday")),
        "extra-key-in-manifest": mut(lambda d: man(d).update(consolidated=True)),
    }
    for name, doc in cases.items():
        path = tmp_path / "c.json"
        path.write_text(json.dumps(doc))
        with pytest.raises(ex.Refused, match="not a replay-routing/5"):
            ex.read_replay_routing([path])
        _ = name
    # entries of a single-index file carry no time of their own (all or none)
    odd = _copy(single)
    man(odd)["manifests"][0].update(source_index=TAG_A, taken_at=TAKEN_A)
    path = tmp_path / "s.json"
    path.write_text(json.dumps(odd))
    with pytest.raises(ex.Refused, match="not a replay-routing/5"):
        ex.read_replay_routing([path])
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(base))
    assert dict(ex.read_replay_routing([ok]).recorded_taken) == {"r1": "2026-10-25T15:31Z", "r2": "2026-10-25T12:32Z"}


def test_class_cell_rows_read_as_proven(world, capsys):
    from test_exploratory import FIXTURE, exporter
    plain = _plain_records(FIXTURE / "batch")
    matrix = _matrix_file(world["tmp"] / "m2.gz", plain)
    ledger = world["tmp"] / "l2.jsonl"
    assert exporter.main(["--split", "dev", "--matrix", str(matrix), "--records", str(FIXTURE / "batch"), "--out", str(ledger)]) == 0
    capsys.readouterr()
    path = world["tmp"] / "cells.json"
    path.write_text(json.dumps(_routing_doc("single", {r["run_id"]: "fast" for r in plain}, kind="cell")))
    kw = dict(ledgers=(ledger,), endpoint_copies=_endpoint_table(world["tmp"] / "ec2.json", ("s01", E0, "copy-a")),
              copy_classes=_classes_file(world["tmp"] / "cc2.json"), replay_routing=(path,))
    check = _run(world, [world["single"]], record=False, K=HAND_K, **kw)["stores"]["replay_class_check"]
    for arm in check["arms"].values():
        assert arm["same_class_proven"] == 3 and arm["same_class_recorded"] == 0 and arm["exceptions"] == 0


def test_an_empty_replay_ledger_with_a_routing_file_leaves_the_section_empty_and_the_routing_unmentioned(world, tmp_path):
    # a valid /2 routing file given to a call whose ledger holds no replay row: the section stays empty, with no routing metadata
    routing = tmp_path / "routing-2.json"
    routing.write_text(json.dumps(_routing_doc("single", {"r1": "fast"})))
    result = _run(world, [world["single"]], record=False, ledgers=(world["ledger"],), K=HAND_K, replay_routing=(routing,))
    assert result["stores"]["replay_class_check"] == EMPTY_CLASS_CHECK




def test_the_routing_must_be_the_file_the_checker_wrote(tmp_path):
    good = tmp_path / "g.json"
    good.write_text(json.dumps(_routing_doc("o", {"r1": "fast"})))
    sha = ex.sha256_bytes(good.read_bytes())
    assert dict(ex.read_replay_routing([good], [sha]).classes) == {"r1": "fast"}
    good.write_text(good.read_text().replace('"class": "fast"', '"class": "slow"', 1))  # an edit after the checker wrote it
    with pytest.raises(ex.Refused, match="not the file the checker wrote"):
        ex.read_replay_routing([good], [sha])
    with pytest.raises(ex.Refused, match="differ in number"):
        ex.read_replay_routing([good], [sha, sha])
    # the command line cannot give a routing without its pin
    ap_err = subprocess.run([sys.executable, "-I", str(HERE.parent / "exploratory.py"), "analyse", "--replay-routing", str(good)],
                            capture_output=True, text=True)
    assert ap_err.returncode == 2 and "PATH:SHA256" in ap_err.stderr


def test_the_checkers_own_output_is_read_by_the_analysis_including_the_fallback_and_unlisted_cases(tmp_path, capsys):
    # end to end: whatever the checker writes, for every basis, the analysis reader accepts (external reviewer r18: it refused its own routing for an unlisted file)
    from test_replay_matrices import attested_world, make_attestation, rm
    import test_replay_matrices as trm
    w = attested_world(tmp_path)  # 02of02: no time file
    pin = w.make_manifest(drop=("opaque-sonnet/logs/m1-natural-01of02.time",))  # 01of02's file exists but is not listed
    att, _, _ = make_attestation(w)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin, "--attestation", att))
    assert rc == 0, err
    got = ex.read_replay_routing([w.stage / "replay-routing.json"], [ex.sha256_bytes((w.stage / "replay-routing.json").read_bytes())])
    assert dict(got.basis) == {"rid-n3": "attested", "rid-p1": "recorded", "rid-p2": "recorded"}
    # a split OUT: proven for the cells, recorded for the main tree, and a changed time file unverifiable
    import shutil
    w2 = trm.split_world(tmp_path / "second")
    w2.write_time("rhea-gemma-4-31b", "m1-natural-01of01", "natural", "192.0.2.105:7043", "rhea-c08", rm.utc_epoch(trm.SEALED_UTC) - 5)
    rc, out, err = w2.run("natural,prompted", capsys)
    assert rc == 0, err
    got = ex.read_replay_routing([w2.stage / "replay-routing.json"])
    assert dict(got.basis) == {"rid-n1": "recorded", "rid-n2": "recorded", "rid-p1": "proven", "rid-p2": "proven", "rid-p3": "proven"}
    shutil.rmtree(w2.stage)
    pin2 = w2.make_manifest(override={"rhea-gemma-4-31b-cell-slow/logs/m1-prompted-01of01.time": "0" * 64})
    rc, out, err = w2.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin2))
    assert rc == 0, err
    got = ex.read_replay_routing([w2.stage / "replay-routing.json"])
    assert got.classes["rid-p3"] == "unverifiable" and "rid-p3" not in got.basis


def test_the_command_line_routing_pins_reach_the_call_and_the_reader(tmp_path):
    # external reviewer r19: _call must unpack PATH:SHA256 into the Call's paths and pins (a pair put in the paths raised TypeError in the reader)
    import argparse
    good = tmp_path / "g.json"
    good.write_text(json.dumps(_routing_doc("o", {"r1": "fast"})))
    pin = ex.sha256_bytes(good.read_bytes())
    parsed = ex._routing_pin(f"{good}:{pin}")
    assert parsed == (good, pin)
    args = argparse.Namespace(batch=[tmp_path / "b"], questions=[tmp_path / "q"], graph="s01", split="dev", voided=[], partition_record=None,
                              ledger=[], n_reps=None, tier_if_missing=None, cluster_map=None, front_log=[], store_proof=[],
                              reference_pin=None, partitions=None, partition_binding=None, partition_plan=[],
                              private_copy_classes=None, private_copy_classes_sha256=None, private_endpoint_copies=None,
                              private_endpoint_copies_sha256=None, replay_routing=[parsed])
    call = ex._call(args)
    assert call.replay_routing == (good,) and call.replay_routing_pins == (pin,)
    assert dict(ex.read_replay_routing(call.replay_routing, call.replay_routing_pins).classes) == {"r1": "fast"}
    for bad in (str(good), f"{good}:short", f":{pin}"):
        with pytest.raises(argparse.ArgumentTypeError):
            ex._routing_pin(bad)
