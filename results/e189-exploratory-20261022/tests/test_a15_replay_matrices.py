"""Tests of driver/a15_replay_matrices.py (the audit-replay counterpart of replay_matrices.py: marker, folder and matrix bound by one
sha256, then the coverage check of the OUT's records against the matrix).

Synthetic fixtures only: an OUT view, a results folder, a marker and a matrix are built here by hand, under ``synthetic-*`` names (the
tool's OUT-phase list is monkeypatched; one test pins the real list). The fixture writer shares no code with the tool except the
reviewed matrix writer of test_replay_matrices.py (matrix_bytes), and every expected count below is written by hand.

Hand-derived facts (computed once with hashlib, independently of the tool; the plain claim digest is the first 40 hex of the sha256
of the JSON of [question, route model, repetition]):
    ["q1", "route-1", 1] -> 0d261dbb8eefab2a0e621d7785cbcc87e328713d   ["q2", "route-1", 1] -> 252127781707e23a111b3c96080aa23242dd47d4
    ["q3", "route-1", 1] -> d08ee8b470d641f57b6a0d7c0d74c4ef81f3b912   ["q4", "route-1", 1] -> 4414b7332962e234e0853af030412d39a22e9035
    ["q1", "route-2", 1] -> 5546a5f6ee96aaaa67e01d409435a2b6f9e6e3e6   (a grid key [arm, question, route, repetition] gives other digests)
A run id has 8 arms x 4 variants = 32 matrix rows; three records give 96.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from test_replay_matrices import DRIVER, matrix_bytes, sha

A15DRIVER = DRIVER.parent / "a15_replay_matrices.py"
_spec = importlib.util.spec_from_file_location("a15_replay_matrices", A15DRIVER)
a15 = importlib.util.module_from_spec(_spec)
sys.modules["a15_replay_matrices"] = a15
_spec.loader.exec_module(a15)

import replay_matrices as rm  # noqa: E402  (registered by test_replay_matrices; the tool must have reused that very module)

OUT = "synthetic-out"
OUT_P = "synthetic-out-p"
ARM_LIST = ["curated", "aligned-curated", "census-t5", "generated-1", "generated-2", "generated-3", "union", "union-aligned"]
SOURCE_LIST = ["curated", "aligned-curated", "census-t5", "generated-1", "generated-2", "generated-3"]
PLAIN = {
    ("q1", "route-1"): "0d261dbb8eefab2a0e621d7785cbcc87e328713d",
    ("q2", "route-1"): "252127781707e23a111b3c96080aa23242dd47d4",
    ("q3", "route-1"): "d08ee8b470d641f57b6a0d7c0d74c4ef81f3b912",
    ("q4", "route-1"): "4414b7332962e234e0853af030412d39a22e9035",
    ("q1", "route-2"): "5546a5f6ee96aaaa67e01d409435a2b6f9e6e3e6",
}
MARKER_TIME = "2026-10-25T12:00:00Z"
THREE = [("q1", "rid-1"), ("q2", "rid-2"), ("q3", "rid-3")]
REAL_NATURAL = ("a15-sonnet", "a15-glm", "a15-qwen38-e180", "a15-deepseek", "a15-coder", "a15-gemma-p0of3", "a15-gemma-p1of3", "a15-gemma-p2of3")
REAL_PROMPTED = ("a15-sonnet-census", "a15-sonnet-curated", "a15-sonnet-generated", "a15-qwen38-e180-census",
                 "a15-qwen38-e180-curated", "a15-qwen38-e180-generated-p0of2", "a15-qwen38-e180-generated-p1of2")
REAL_PAIRS = (a15.NATURAL_OUTS, a15.PROMPTED_OUTS, a15.COMBINED_NAMES)  # before any monkeypatch


@pytest.fixture(autouse=True)
def synthetic_outs(monkeypatch):
    monkeypatch.setattr(a15, "NATURAL_OUTS", (OUT,))
    monkeypatch.setattr(a15, "PROMPTED_OUTS", (OUT_P,))


def rec_name(q: str, rid: str, route: str = "route-1") -> str:
    return f"{PLAIN[(q, route)]}.{rid}.json"


def run_json(matrix_sha: str, *, sources=SOURCE_LIST, final=True, schema="a15-gated-replay/1") -> dict:
    return {"schema": schema, "final": final, "at": "2026-10-25T11:00:00Z",
            "build": {"records_in_matrix": 3, "rows": 96, "sha256": {"matrix.jsonl.gz": matrix_sha, "store-log.jsonl.gz": "0" * 64}},
            "pins": {"hierarchy": "1" * 64, "sources": {s: "2" * 64 for s in sources}}}


class AWorld:
    """One plain OUT's view, its results folder and its marker."""

    def __init__(self, tmp: Path, out: str = OUT, phase: str = "natural", items=THREE, *, view_name: str | None = None,
                 run_ids=None) -> None:
        self.out, self.phase, self.tmp = out, phase, tmp
        self.view = tmp / "views" / (view_name or f"{out}-r24-20261026T000000Z")
        self.audit_root = tmp / "audit-results"
        self.folder = self.audit_root / f"audit-{out}"
        self.done = tmp / "done"
        self.stage = tmp / "stage"
        for d in (self.view / "records", self.folder, self.done):
            d.mkdir(parents=True, exist_ok=True)
        (self.view / "served-models.json").write_text(json.dumps({"models": {"m1": {"route_model": "route-1"}}}))
        order = []
        for q, rid in items:
            (self.view / "records" / rec_name(q, rid)).write_text("{}")
            order.append(["m1", "condition-x", q, 1])
        (self.view / "order.json").write_text(json.dumps({"order": order}))
        self.set_matrix(matrix_bytes(run_ids if run_ids is not None else [rid for _, rid in items], arms=ARM_LIST))
        (self.folder / "rc").write_text("0 1a2b3c4d\n")

    def set_matrix(self, data: bytes, *, marker_sha: str | None = None, run_sha: str | None = None, marker_line: str | None = None,
                   **run_over) -> None:
        (self.folder / "matrix.jsonl.gz").write_bytes(data)
        self.matrix_sha = sha(data)
        (self.folder / "run.json").write_text(json.dumps(run_json(run_sha or self.matrix_sha, **run_over)))
        self.mark(marker_line if marker_line is not None else f"{marker_sha or self.matrix_sha} ok {MARKER_TIME}\n")

    def mark(self, text: str) -> None:
        (self.done / f"{self.out}-{self.phase}").write_text(text)

    def argv(self, **over) -> list:
        a = {"--audit-root": str(self.audit_root), "--done-dir": str(self.done), "--out-dir": str(self.view),
             "--out-name": self.out, "--phase": self.phase, "--stage": str(self.stage), **over}
        return ["check-a15"] + [x for kv in a.items() for x in kv]

    def check(self, capsys, **over):
        rc = a15.main(self.argv(**over))
        out, err = capsys.readouterr()
        # every output is counts, names and sha256s: never a run id or a claim digest
        assert "rid-" not in out + err
        assert not re.search(r"[0-9a-f]{40}\.", out + err)
        return rc, out, err


def refusal(capsys, w: AWorld, **over) -> str:
    rc, out, err = w.check(capsys, **over)
    assert rc == 2 and out == "", (rc, out, err)
    assert err.startswith("REFUSED ") and err.count("\n") == 1
    return err


# -- the pinned constants -----------------------------------------------------------------------------------------------------


def test_the_pinned_arms_and_variants_are_the_audit_replays_and_the_matrixs():
    # the launcher gives the wrapper six --source pins and two --union specs; the frozen Arms names are the sources, then the unions
    assert list(a15.SOURCES) == SOURCE_LIST
    assert a15.UNIONS == ("union", "union-aligned")
    assert list(a15.ARMS) == ARM_LIST and len(a15.ARMS) == 8
    assert not {"census-t2", "census-t20"} & set(a15.ARMS)  # Tier 2's arms do not apply
    assert list(a15.VARIANTS) == ["primary", "heuristic_off", "branch_local_off", "both_off"] == list(rm.VARIANTS)
    try:
        from harness.checkloop.matrix import VARIANTS as FROZEN
        from harness.checkloop.matrix import Arms
    except ImportError:  # pragma: no cover - the frozen code is on the path in the suite
        pytest.skip("harness not importable")
    assert list(FROZEN) == list(a15.VARIANTS)
    unions = {"union": ("curated", "census-t5"), "union-aligned": ("aligned-curated", "census-t5")}
    assert Arms({s: None for s in a15.SOURCES}, unions).names == ARM_LIST


def test_the_tool_reuses_the_reviewed_module_by_path():
    assert a15.rm is rm and a15.Refused is rm.Refused
    assert Path(rm.__file__).resolve() == DRIVER.resolve()
    text = A15DRIVER.read_text()
    assert not re.search(r"^\s*(import|from)\s+replay_matrices", text, re.M)  # -I leaves the script's directory off sys.path


def test_the_real_out_phase_list_is_exactly_the_fifteen_pairs_and_the_one_combined_name(monkeypatch):
    nat, pro, combined = REAL_PAIRS
    assert nat == REAL_NATURAL and pro == REAL_PROMPTED and combined == ("a15-gemma-copied",)
    monkeypatch.setattr(a15, "NATURAL_OUTS", nat)
    monkeypatch.setattr(a15, "PROMPTED_OUTS", pro)
    pairs = a15.out_phases()
    assert len(pairs) == 15 and ("a15-glm", "natural") in pairs and ("a15-sonnet-census", "prompted") in pairs
    assert ("a15-glm", "prompted") not in pairs and ("a15-sonnet-census", "natural") not in pairs
    assert {o for o, _ in pairs if "gemma" in o} == {"a15-gemma-p0of3", "a15-gemma-p1of3", "a15-gemma-p2of3"}
    assert all(ph == "natural" for o, ph in pairs if "gemma" in o)  # the partitions are natural-phase OUTs; the drained source OUT is no pair
    assert ("a15-gemma", "natural") not in pairs and not any(o == "a15-gemma-copied" for o, _ in pairs)


# -- the good cases -----------------------------------------------------------------------------------------------------------


def test_an_exact_match_is_accepted_and_staged(tmp_path, capsys):
    w = AWorld(tmp_path)
    w.set_matrix(matrix_bytes(["rid-1", "rid-2", "rid-3"], arms=ARM_LIST, no_output={"rid-3"}))
    rc, out, err = w.check(capsys)
    assert rc == 0 and err == ""
    marker_file_sha = sha(f"{w.matrix_sha} ok {MARKER_TIME}\n".encode())
    manifest = f"out\tphase\ttree\tshard\tsha256\trecords\n{OUT}\tnatural\taudit-{OUT}\taudit\t{w.matrix_sha}\t3\n"
    assert out.splitlines() == [
        f"LOCATED-A15 out={OUT} phase=natural marker_sha256={marker_file_sha} matrix_sha256={w.matrix_sha} disclosure=0",
        f"ROUTING out={OUT} kind=a15 unrouted=1 classes=none note=no-launch-records",
        f"COVERAGE out={OUT} order_keys_natural=3 records_natural=3 rows_natural=96 shards_natural=1 matrices=1 "
        f"manifest_sha256={sha(manifest.encode())} OK"]
    copy = w.stage / OUT / "natural" / f"audit-{OUT}" / "audit.matrix.jsonl.gz"
    assert copy.read_bytes() == (w.folder / "matrix.jsonl.gz").read_bytes() and sha(copy.read_bytes()) == w.matrix_sha
    assert (w.stage / "matrices.list").read_text() == f"{copy}\n"
    assert (w.stage / "replay-matrices-manifest.tsv").read_text() == manifest
    assert (w.stage / "replay-outputs.list").read_text() == "rid-1\nrid-2\n"  # rid-3 has no output
    assert sorted(p.name for p in w.stage.iterdir()) == ["matrices.list", "replay-matrices-manifest.tsv", "replay-outputs.list", OUT]
    assert not (w.stage / "replay-routing.json").exists()
    assert stat.S_IMODE(w.stage.stat().st_mode) == 0o700 and stat.S_IMODE(copy.stat().st_mode) == 0o600
    for name in ("matrices.list", "replay-matrices-manifest.tsv", "replay-outputs.list"):
        assert stat.S_IMODE((w.stage / name).stat().st_mode) == 0o600


def test_the_view_may_have_any_name_and_a_prompted_out_phase_is_served(tmp_path, capsys):
    w = AWorld(tmp_path, out=OUT_P, phase="prompted", view_name="views-0000-anything")
    rc, out, err = w.check(capsys)
    assert rc == 0, err
    assert f"COVERAGE out={OUT_P} order_keys_prompted=3 records_prompted=3 rows_prompted=96 shards_prompted=1 matrices=1 " in out
    assert (w.stage / OUT_P / "prompted" / f"audit-{OUT_P}" / "audit.matrix.jsonl.gz").is_file()


def test_a_disclosure_after_the_timestamp_is_accepted_and_flagged_not_interpreted(tmp_path, capsys):
    w = AWorld(tmp_path)
    w.mark(f"{w.matrix_sha} ok {MARKER_TIME} after-window taken late at 2026-10-25T13:00:00Z; some free text\n")
    rc, out, err = w.check(capsys)
    assert rc == 0, err
    assert out.splitlines()[0].endswith(" disclosure=1") and "free text" not in out


def test_the_plain_claim_digest_is_the_question_route_repetition_hash():
    for (q, route), digest in PLAIN.items():
        assert a15.plain_digest(q, route, 1) == digest
    assert a15.plain_digest("q1", "route-1", 1) != rm.claim_digest("natural", "q1", "route-1", 1)  # not the grid key


def test_setaside_files_are_ignored_and_two_runners_with_their_own_routes_are_served(tmp_path, capsys):
    w = AWorld(tmp_path)
    (w.view / "records" / f"{PLAIN[('q1', 'route-1')]}.rid-1.setaside.json").write_text("{}")
    rc, out, err = w.check(capsys)
    assert rc == 0, err
    w2 = AWorld(tmp_path / "two", items=[("q1", "rid-1")])
    (w2.view / "served-models.json").write_text(json.dumps({"models": {"m1": {"route_model": "route-1"}, "m2": {"route_model": "route-2"}}}))
    (w2.view / "order.json").write_text(json.dumps({"order": [["m1", "c", "q1", 1], ["m2", "c", "q1", 1]]}))
    (w2.view / "records" / rec_name("q1", "rid-9", "route-2")).write_text("{}")
    w2.set_matrix(matrix_bytes(["rid-1", "rid-9"], arms=ARM_LIST))
    rc, out, err = w2.check(capsys)
    assert rc == 0, err
    assert "records_natural=2 rows_natural=64 " in out


# -- the matrix against the OUT's records -------------------------------------------------------------------------------------


@pytest.mark.parametrize("matrix, expected", [
    # a run id of the OUT with no row at all: missing 1
    (dict(run_ids=["rid-1", "rid-2"]),
     "REFUSED matrix-differs-from-records out_records=3 matrix_records=2 missing=1 extra=0 rows=64 rows_expected=96\n"),
    # a run id that is no record of the OUT: extra 1
    (dict(run_ids=["rid-1", "rid-2", "rid-3", "rid-9"]),
     "REFUSED matrix-differs-from-records out_records=3 matrix_records=4 missing=0 extra=1 rows=128 rows_expected=96\n"),
    # as many rows as expected, but another run in place of one: both counts say so
    (dict(run_ids=["rid-1", "rid-2", "rid-9"]),
     "REFUSED matrix-differs-from-records out_records=3 matrix_records=3 missing=1 extra=1 rows=96 rows_expected=96\n"),
    # one row of one record missing
    (dict(run_ids=["rid-1", "rid-2", "rid-3"], drop={("rid-2", "union", "both_off")}),
     "REFUSED matrix-differs-from-records out_records=3 matrix_records=3 missing=0 extra=0 rows=95 rows_expected=96 missing_rows=1\n"),
    # one row repeated
    (dict(run_ids=["rid-1", "rid-2", "rid-3"], dup={("rid-1", "curated", "primary")}),
     "REFUSED matrix-differs-from-records out_records=3 matrix_records=3 missing=0 extra=0 rows=97 rows_expected=96 duplicate_rows=1\n"),
    # another arm (one extra row a run, and Tier 2's census-t2 arm)
    (dict(run_ids=["rid-1", "rid-2", "rid-3"], extra_arm="census-t2"),
     "REFUSED matrix-differs-from-records out_records=3 matrix_records=3 missing=0 extra=0 rows=99 rows_expected=96 extra_rows=3\n"),
    # a missing arm (the second union)
    (dict(run_ids=["rid-1", "rid-2", "rid-3"], arms=ARM_LIST[:-1]),
     "REFUSED matrix-differs-from-records out_records=3 matrix_records=3 missing=0 extra=0 rows=84 rows_expected=96 missing_rows=12\n"),
    # the Tier 2 arm set, which is not this one: its three other arms are extra, the two aligned arms missing
    (dict(run_ids=["rid-1", "rid-2", "rid-3"], arms=["curated", "census-t5", "census-t2", "census-t20", "generated-1", "generated-2",
                                                     "generated-3", "union"]),
     "REFUSED matrix-differs-from-records out_records=3 matrix_records=3 missing=0 extra=0 rows=96 rows_expected=96 extra_rows=24 "
     "missing_rows=24\n"),
    # an output flag that differs between the rows of one run: a flag, never a count
    (dict(run_ids=["rid-1", "rid-2", "rid-3"], flip_output={("rid-1", "curated", "primary")}),
     "REFUSED matrix-differs-from-records out_records=3 matrix_records=3 missing=0 extra=0 rows=96 rows_expected=96 output_disagrees=1\n"),
    # no output flag at all
    (dict(run_ids=["rid-1", "rid-2", "rid-3"], omit_output=True),
     "REFUSED matrix-differs-from-records out_records=3 matrix_records=3 missing=0 extra=0 rows=96 rows_expected=96 output_not_bool=1\n"),
    # a run with an output and no scorer label
    (dict(run_ids=["rid-1", "rid-2", "rid-3"], unlabelled={"rid-1", "rid-2"}),
     "REFUSED matrix-differs-from-records out_records=3 matrix_records=3 missing=0 extra=0 rows=96 rows_expected=96 output_without_label=1\n"),
    (dict(run_ids=["rid-1", "rid-2", "rid-3"], version="checker-replay-matrix/2"),
     "REFUSED matrix-differs-from-records out_records=3 matrix_records=3 missing=0 extra=0 rows=96 rows_expected=96 wrong_version_rows=96\n"),
    (dict(run_ids=["rid-1", "rid-2", "rid-3"], split="dev"),
     "REFUSED matrix-differs-from-records out_records=3 matrix_records=3 missing=0 extra=0 rows=96 rows_expected=96 wrong_split_rows=96\n"),
])
def test_a_matrix_that_is_not_exactly_the_outs_records_with_their_rows_is_refused(tmp_path, capsys, matrix, expected):
    w = AWorld(tmp_path)
    ids = matrix.pop("run_ids")
    matrix.setdefault("arms", ARM_LIST)
    w.set_matrix(matrix_bytes(ids, **matrix))
    assert refusal(capsys, w) == expected


def test_a_view_of_another_out_is_refused_by_the_run_id_counts(tmp_path, capsys):
    w = AWorld(tmp_path)
    other = AWorld(tmp_path / "other", items=[("q4", "rid-4"), ("q1", "rid-5"), ("q2", "rid-6")])
    assert refusal(capsys, w, **{"--out-dir": str(other.view)}) == (
        "REFUSED matrix-differs-from-records out_records=3 matrix_records=3 missing=3 extra=3 rows=96 rows_expected=96\n")


def test_a_truncated_matrix_is_an_io_refusal_with_the_class_name_only(tmp_path, capsys):
    w = AWorld(tmp_path)
    w.set_matrix(gzip.compress(b"x" * 100)[:-9])
    assert refusal(capsys, w).startswith("REFUSED io-error class=")


# -- the records of the OUT ---------------------------------------------------------------------------------------------------


def test_a_grid_out_a_missing_order_and_a_record_outside_the_order_are_refused(tmp_path, capsys):
    w = AWorld(tmp_path)
    (w.view / "order-natural.json").write_text(json.dumps({"order": []}))
    assert refusal(capsys, w) == "REFUSED order-not-plain\n"
    (w.view / "order-natural.json").unlink()
    (w.view / "order.json").rename(w.view / "order.bak")
    assert refusal(capsys, w) == "REFUSED no-order-file\n"
    (w.view / "order.bak").rename(w.view / "order.json")
    (w.view / "records" / rec_name("q4", "rid-4")).write_text("{}")
    assert refusal(capsys, w) == "REFUSED records-unclassified unclassified=1 odd_names=0 digest_twice=0 run_id_twice=0\n"
    (w.view / "records" / rec_name("q4", "rid-4")).unlink()
    (w.view / "records" / "notes.json").write_text("{}")
    assert refusal(capsys, w) == "REFUSED records-unclassified unclassified=0 odd_names=1 digest_twice=0 run_id_twice=0\n"
    assert not w.stage.exists()  # no copy was made for any of these


def test_a_run_id_or_a_digest_in_two_record_files_is_refused(tmp_path, capsys):
    w = AWorld(tmp_path)
    (w.view / "records" / rec_name("q1", "rid-8")).write_text("{}")  # two files of one attempt (q1): one digest twice
    assert refusal(capsys, w) == "REFUSED records-unclassified unclassified=0 odd_names=0 digest_twice=1 run_id_twice=0\n"
    (w.view / "records" / rec_name("q1", "rid-8")).unlink()
    (w.view / "records" / rec_name("q1", "rid-3")).write_text("{}")  # run id rid-3 in the files of q3 and of q1 (and q1's digest twice)
    assert refusal(capsys, w) == "REFUSED records-unclassified unclassified=0 odd_names=0 digest_twice=1 run_id_twice=1\n"
    (w.view / "records" / rec_name("q1", "rid-3")).unlink()
    (w.view / "records" / rec_name("q3", "rid-3")).unlink()
    (w.view / "records" / rec_name("q3", "rid-1")).write_text("{}")  # q3's digest once, but the run id rid-1 is in two files
    assert refusal(capsys, w) == "REFUSED records-unclassified unclassified=0 odd_names=0 digest_twice=0 run_id_twice=1\n"


def test_an_order_key_without_a_route_of_its_own_or_sharing_a_digest_is_refused(tmp_path, capsys):
    w = AWorld(tmp_path)
    (w.view / "order.json").write_text(json.dumps({"order": [["m1", "c", "q1", 1], ["m1", "c", "q2", 1], ["m1", "c", "q3", 1], ["mX", "c", "q3", 1]]}))
    assert refusal(capsys, w) == "REFUSED order-keys-ambiguous unrouted=1 shared=0\n"
    (w.view / "order.json").write_text(json.dumps({"order": [["m1", "c", "q1", 1], ["m1", "c", "q2", 1], ["m1", "d", "q2", 1], ["m1", "c", "q3", 1]]}))
    assert refusal(capsys, w) == "REFUSED order-keys-ambiguous unrouted=0 shared=2\n"
    (w.view / "served-models.json").write_text(json.dumps({"models": {"m1": {"route_model": "route-1"}, "m2": {"route_model": "route-1"}}}))
    (w.view / "order.json").write_text(json.dumps({"order": [["m1", "c", "q1", 1], ["m1", "c", "q2", 1], ["m1", "c", "q3", 1]]}))
    assert refusal(capsys, w) == "REFUSED order-keys-ambiguous unrouted=3 shared=0\n"  # two names for one route: neither has one
    (w.view / "order.json").write_text(json.dumps({"order": [["m1", "c", "q1"]]}))
    assert refusal(capsys, w) == "REFUSED order-malformed\n"


def test_an_out_with_no_record_is_refused(tmp_path, capsys):
    w = AWorld(tmp_path, items=[])
    assert refusal(capsys, w) == "REFUSED no-records\n"


# -- the marker, the folder and the names -------------------------------------------------------------------------------------


def test_names_and_phases_outside_the_pinned_pairs_are_refused_without_echo(tmp_path, capsys):
    w = AWorld(tmp_path)
    assert refusal(capsys, w, **{"--out-name": "a15-gemma-p0of3"}) == "REFUSED bad-name\n"
    assert refusal(capsys, w, **{"--out-name": "synthetic-out/../x"}) == "REFUSED bad-name\n"
    assert refusal(capsys, w, **{"--phase": "prompted"}) == "REFUSED bad-name\n"  # the pair (OUT, prompted) is not pinned
    assert refusal(capsys, w, **{"--out-name": OUT_P}) == "REFUSED bad-name\n"
    with pytest.raises(SystemExit) as exc:
        a15.main(w.argv(**{"--phase": "branch"}))
    assert exc.value.code == 2 and capsys.readouterr().err == "REFUSED bad-arguments\n"
    assert not w.stage.exists()


@pytest.mark.parametrize("text, expected", [
    (None, "REFUSED no-done-marker out=synthetic-out phase=natural\n"),
    ("", "REFUSED marker-format out=synthetic-out phase=natural\n"),
    ("{sha} ok 2026-10-25T12:00:00Z", "REFUSED marker-format out=synthetic-out phase=natural\n"),  # no newline
    ("{sha} ok 2026-10-25T12:00:00Z\n\n", "REFUSED marker-format out=synthetic-out phase=natural\n"),
    ("{sha} ok 2026-10-25T12:00:00Z\n{sha} ok 2026-10-25T12:00:00Z\n", "REFUSED marker-format out=synthetic-out phase=natural\n"),
    ("{sha}\n", "REFUSED marker-format out=synthetic-out phase=natural\n"),
    ("{sha} OK 2026-10-25T12:00:00Z\n", "REFUSED marker-format out=synthetic-out phase=natural\n"),
    ("{sha} ok yesterday\n", "REFUSED marker-format out=synthetic-out phase=natural\n"),
    ("{sha} ok 2026-10-25T12:00:00Z \n", "REFUSED marker-format out=synthetic-out phase=natural\n"),  # a disclosure needs text
    ("{sha_upper} ok 2026-10-25T12:00:00Z\n", "REFUSED marker-format out=synthetic-out phase=natural\n"),
    ("{sha_short} ok 2026-10-25T12:00:00Z\n", "REFUSED marker-format out=synthetic-out phase=natural\n"),
    ("OK shards 1 online 1 obqc 1 errors 0 2026-10-24T16:00:00Z\n", "REFUSED marker-format out=synthetic-out phase=natural\n"),
])
def test_a_marker_that_is_not_one_sha_ok_utc_line_is_refused(tmp_path, capsys, text, expected):
    w = AWorld(tmp_path)
    path = w.done / f"{OUT}-natural"
    if text is None:
        path.unlink()
    else:
        path.write_text(text.format(sha=w.matrix_sha, sha_upper=w.matrix_sha.upper(), sha_short=w.matrix_sha[:63]))
    assert refusal(capsys, w) == expected
    assert not w.stage.exists()  # a refusal before the copy leaves no stage


def test_a_marker_that_is_a_link_is_refused(tmp_path, capsys):
    w = AWorld(tmp_path)
    real = tmp_path / "real-marker"
    real.write_text(f"{w.matrix_sha} ok {MARKER_TIME}\n")
    (w.done / f"{OUT}-natural").unlink()
    os.symlink(real, w.done / f"{OUT}-natural")
    assert refusal(capsys, w) == "REFUSED unreadable file=synthetic-out-natural\n"


def test_the_three_sha256s_must_agree(tmp_path, capsys):
    w = AWorld(tmp_path)
    data = (w.folder / "matrix.jsonl.gz").read_bytes()
    w.set_matrix(data, marker_sha="e" * 64)  # the marker names another matrix; run.json names the real one
    assert refusal(capsys, w) == "REFUSED run-json-differs out=synthetic-out what=matrix-sha256\n"
    w.set_matrix(data, run_sha="e" * 64)  # run.json names another matrix
    assert refusal(capsys, w) == "REFUSED run-json-differs out=synthetic-out what=matrix-sha256\n"
    w.set_matrix(data, marker_sha="e" * 64, run_sha="e" * 64)  # marker and run.json agree, the file differs
    assert refusal(capsys, w) == "REFUSED matrix-sha-differs out=synthetic-out\n"
    assert not w.stage.exists()


def test_the_run_description_must_be_final_of_its_schema_and_pin_exactly_the_sources(tmp_path, capsys):
    cases = (({"final": False}, "not-final"), ({"final": "yes"}, "not-final"), ({"schema": "a15-gated-replay/2"}, "schema"),
             ({"sources": SOURCE_LIST[:-1]}, "sources"), ({"sources": SOURCE_LIST + ["census-t2"]}, "sources"), ({"sources": []}, "sources"))
    for i, (over, what) in enumerate(cases):
        w = AWorld(tmp_path / f"case-{i}")
        w.set_matrix((w.folder / "matrix.jsonl.gz").read_bytes(), **over)
        assert refusal(capsys, w) == f"REFUSED run-json-differs out=synthetic-out what={what}\n", over
    w = AWorld(tmp_path / "plain")
    (w.folder / "run.json").write_text("[]")
    assert refusal(capsys, w) == "REFUSED run-json-differs out=synthetic-out what=schema\n"
    (w.folder / "run.json").write_text("{not json")
    assert refusal(capsys, w) == "REFUSED malformed-json what=run.json\n"


def test_the_rc_must_be_zero_and_a_link_anywhere_is_refused(tmp_path, capsys):
    for i, text in enumerate(("1 1a2b3c4d\n", "0\n", "0 1a2b3c4d extra\n", "0 XYZ\n", "0 1a2b3c4d\n0 1a2b3c4d\n", "", "00 1a2b\n")):
        w = AWorld(tmp_path / f"rc-{i}")
        (w.folder / "rc").write_text(text)
        assert refusal(capsys, w) == "REFUSED rc-not-zero out=synthetic-out\n", text
    w = AWorld(tmp_path / "rc-missing")
    (w.folder / "rc").unlink()
    assert refusal(capsys, w) == "REFUSED unreadable file=rc\n"
    w = AWorld(tmp_path / "link-matrix")
    (w.folder / "matrix.jsonl.gz").rename(tmp_path / "moved.gz")
    os.symlink(tmp_path / "moved.gz", w.folder / "matrix.jsonl.gz")
    assert refusal(capsys, w) == "REFUSED matrix-missing-or-linked out=synthetic-out\n"
    (w.folder / "matrix.jsonl.gz").unlink()
    assert refusal(capsys, w) == "REFUSED matrix-missing-or-linked out=synthetic-out\n"
    w = AWorld(tmp_path / "link-run")
    (w.folder / "run.json").rename(tmp_path / "moved.json")
    os.symlink(tmp_path / "moved.json", w.folder / "run.json")
    assert refusal(capsys, w) == "REFUSED unreadable file=run.json\n"
    w = AWorld(tmp_path / "link-folder")
    (w.audit_root / f"audit-{OUT}").rename(tmp_path / "moved-folder")
    os.symlink(tmp_path / "moved-folder", w.audit_root / f"audit-{OUT}")
    assert refusal(capsys, w) == "REFUSED audit-folder-not-a-directory out=synthetic-out\n"
    (w.audit_root / f"audit-{OUT}").unlink()
    assert refusal(capsys, w) == "REFUSED audit-folder-missing out=synthetic-out\n"


def test_an_existing_stage_and_a_missing_view_are_refused(tmp_path, capsys):
    w = AWorld(tmp_path)
    w.stage.mkdir()
    assert refusal(capsys, w) == "REFUSED stage-exists\n"
    w.stage.rmdir()
    assert refusal(capsys, w, **{"--out-dir": str(tmp_path / "absent")}) == "REFUSED out-dir-missing\n"
    assert not w.stage.exists()


# -- a selection of the OUT's records, and a results folder and marker with other names ---------------------------------------


def names_file(tmp_path: Path, names, *, text=None, sha_of=None) -> str:
    """A pinned file of record names as FILE:SHA256 (``text`` overrides the body, ``sha_of`` the pinned bytes)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"names-{sha(repr((names, text)).encode())[:10]}.txt"
    body = text if text is not None else "".join(n + "\n" for n in names)
    path.write_text(body)
    return f"{path}:{sha((sha_of if sha_of is not None else body).encode())}"


def add_supplement(w: AWorld, name: str, run_ids, marker="supp-marker", **matrix) -> None:
    """A second results folder ``audit-<name>`` (the same structure) and its marker file ``marker`` in the done directory."""
    folder = w.audit_root / f"audit-{name}"
    folder.mkdir()
    data = matrix_bytes(run_ids, **{"arms": ARM_LIST, **matrix})
    (folder / "matrix.jsonl.gz").write_bytes(data)
    (folder / "run.json").write_text(json.dumps(run_json(sha(data))))
    (folder / "rc").write_text("0 1a2b3c4d\n")
    (w.done / marker).write_text(f"{sha(data)} ok {MARKER_TIME}\n")


def test_exclude_names_wants_the_records_but_the_listed_ones(tmp_path, capsys):
    w = AWorld(tmp_path)
    w.set_matrix(matrix_bytes(["rid-1", "rid-2"], arms=ARM_LIST))
    names = names_file(tmp_path, [rec_name("q3", "rid-3")])
    rc, out, err = w.check(capsys, **{"--exclude-names": names})
    assert (rc, err) == (0, "")
    manifest = f"out\tphase\ttree\tshard\tsha256\trecords\n{OUT}\tnatural\taudit-{OUT}\taudit\t{w.matrix_sha}\t2\n"
    assert out.splitlines()[-1] == (f"COVERAGE out={OUT} order_keys_natural=3 records_natural=2 rows_natural=64 shards_natural=1 "
                                    f"selection=exclude names_listed=1 matrices=1 manifest_sha256={sha(manifest.encode())} OK")
    assert (w.stage / "replay-matrices-manifest.tsv").read_text() == manifest
    assert (w.stage / "replay-outputs.list").read_text() == "rid-1\nrid-2\n"


def test_only_names_wants_exactly_the_listed_records(tmp_path, capsys):
    w = AWorld(tmp_path)
    w.set_matrix(matrix_bytes(["rid-3"], arms=ARM_LIST))
    names = names_file(tmp_path, [rec_name("q3", "rid-3")])
    rc, out, err = w.check(capsys, **{"--only-names": names})
    assert (rc, err) == (0, "")
    assert out.splitlines()[-1].startswith(f"COVERAGE out={OUT} order_keys_natural=3 records_natural=1 rows_natural=32 shards_natural=1 "
                                           "selection=only names_listed=1 matrices=1 manifest_sha256=")
    assert "selection=" not in a15_default_run_coverage(tmp_path / "plain", capsys)  # nothing is added without a selection


def a15_default_run_coverage(tmp_path: Path, capsys) -> str:
    w = AWorld(tmp_path)
    rc, out, err = w.check(capsys)
    assert rc == 0
    return out.splitlines()[-1]


@pytest.mark.parametrize("mode, matrix_ids, expected", [
    # exclude: the matrix is the whole OUT (the listed record is extra), or lacks a wanted record
    ("--exclude-names", ["rid-1", "rid-2", "rid-3"],
     "REFUSED matrix-differs-from-records out_records=2 matrix_records=3 missing=0 extra=1 rows=96 rows_expected=64\n"),
    ("--exclude-names", ["rid-1"],
     "REFUSED matrix-differs-from-records out_records=2 matrix_records=1 missing=1 extra=0 rows=32 rows_expected=64\n"),
    # only: the matrix has another record besides the listed one, or lacks it
    ("--only-names", ["rid-3", "rid-1"],
     "REFUSED matrix-differs-from-records out_records=1 matrix_records=2 missing=0 extra=1 rows=64 rows_expected=32\n"),
    ("--only-names", ["rid-1"],
     "REFUSED matrix-differs-from-records out_records=1 matrix_records=1 missing=1 extra=1 rows=32 rows_expected=32\n"),
])
def test_a_selection_does_not_relax_the_exact_match(tmp_path, capsys, mode, matrix_ids, expected):
    w = AWorld(tmp_path)
    w.set_matrix(matrix_bytes(matrix_ids, arms=ARM_LIST))
    assert refusal(capsys, w, **{mode: names_file(tmp_path, [rec_name("q3", "rid-3")])}) == expected


def test_the_names_file_is_pinned_before_it_is_parsed_and_must_be_well_formed(tmp_path, capsys):
    w = AWorld(tmp_path)
    good = rec_name("q3", "rid-3")
    # a wrong pin on a file that is also malformed: the pin is what is refused
    assert refusal(capsys, w, **{"--only-names": names_file(tmp_path, [], text="not a name\n", sha_of="x")}) == "REFUSED names-sha-differs\n"
    assert refusal(capsys, w, **{"--only-names": names_file(tmp_path, [good], sha_of="other")}) == "REFUSED names-sha-differs\n"
    for i, text in enumerate(("", "\n", good + "\n" + good + "\n", rec_name("q3", "rid-3") + "\n" + rec_name("q1", "rid-1") + "\n",
                              good + "\n\n", good + "\r\n", "plain.json\n", good.replace(".json", ".setaside.json") + "\n", good + " \n")):
        spec = names_file(tmp_path / f"n{i}", [], text=text)
        assert refusal(capsys, w, **{"--only-names": spec}) == "REFUSED names-malformed\n", repr(text)
    sha_good = sha((good + "\n").encode())
    for spec in (f"relative.txt:{sha_good}", f"/a/../b.txt:{sha_good}", f"/a b.txt:{sha_good}", "/a/b.txt", f"/a/b.txt:{sha_good.upper()}",
                 f"/a/b.txt:{sha_good[:-1]}", ""):
        assert refusal(capsys, w, **{"--exclude-names": spec}) == "REFUSED names-pin-malformed\n", spec
    link = tmp_path / "link.txt"
    real = tmp_path / "real.txt"
    real.write_text(good + "\n")
    os.symlink(real, link)
    assert refusal(capsys, w, **{"--only-names": f"{link}:{sha_good}"}).startswith("REFUSED unreadable")
    assert not w.stage.exists()  # every one of these leaves no stage


def test_a_listed_name_that_is_no_record_of_the_out_is_refused_by_count(tmp_path, capsys):
    w = AWorld(tmp_path)
    assert refusal(capsys, w, **{"--only-names": names_file(tmp_path, [rec_name("q4", "rid-9")])}) == "REFUSED names-not-records unknown=1\n"
    mixed = sorted([rec_name("q3", "rid-3"), rec_name("q4", "rid-9"), rec_name("q4", "rid-8")])
    assert refusal(capsys, w, **{"--exclude-names": names_file(tmp_path, mixed)}) == "REFUSED names-not-records unknown=2\n"
    assert not w.stage.exists()


def test_the_two_selections_are_exclusive_and_a_selection_that_leaves_nothing_is_refused(tmp_path, capsys):
    w = AWorld(tmp_path)
    names = names_file(tmp_path, [rec_name("q3", "rid-3")])
    with pytest.raises(SystemExit) as exc:
        a15.main(w.argv(**{"--exclude-names": names, "--only-names": names}))
    assert exc.value.code == 2 and capsys.readouterr().err == "REFUSED bad-arguments\n"
    everything = names_file(tmp_path, sorted(rec_name(q, rid) for q, rid in THREE))
    assert refusal(capsys, w, **{"--exclude-names": everything}) == "REFUSED no-records\n"


def test_two_checks_with_the_two_selections_cover_each_record_once_and_the_ledger_check_spans_them(tmp_path, capsys):
    w = AWorld(tmp_path)
    w.set_matrix(matrix_bytes(["rid-1", "rid-2"], arms=ARM_LIST, no_output={"rid-2"}))
    add_supplement(w, "supp-x", ["rid-3"])
    names = names_file(tmp_path, [rec_name("q3", "rid-3")])
    first = w.check(capsys, **{"--exclude-names": names})
    stage2 = tmp_path / "stage2"
    second = w.check(capsys, **{"--only-names": names, "--audit-name": "supp-x", "--marker-name": "supp-marker", "--stage": str(stage2)})
    assert first[0] == 0 and second[0] == 0, (first, second)
    # the staged run ids together are the OUT's records, each once (read straight from the copies)
    ids = []
    for copy in (w.stage / OUT / "natural" / f"audit-{OUT}" / "audit.matrix.jsonl.gz", stage2 / OUT / "natural" / "audit-supp-x" / "audit.matrix.jsonl.gz"):
        ids += sorted({json.loads(line)["run_id"] for line in gzip.open(copy, "rt")})
    assert ids == ["rid-1", "rid-2", "rid-3"]
    assert (stage2 / "replay-matrices-manifest.tsv").read_text() == (
        f"out\tphase\ttree\tshard\tsha256\trecords\n{OUT}\tnatural\taudit-supp-x\taudit\t{sha((tmp_path / 'audit-results' / 'audit-supp-x' / 'matrix.jsonl.gz').read_bytes())}\t1\n")
    rows = ledger(tmp_path / "l.jsonl", replay_rows(["rid-1", "rid-3"]))
    assert check_many(capsys, [w.stage, stage2], rows) == (0, "LEDGER-CHECK kind=a15 arms=8 OK\n", "")
    assert check_many(capsys, [w.stage, stage2], ledger(tmp_path / "l2.jsonl", replay_rows(["rid-1"]))) == (
        2, "", "REFUSED ledger-differs-from-matrices missing=1 extra=0 duplicate=0 other_sources=0\n")


def test_the_results_folder_and_the_marker_may_be_named_apart_from_the_out_and_must_still_pass_every_check(tmp_path, capsys):
    w = AWorld(tmp_path)
    add_supplement(w, "supp-x", ["rid-1", "rid-2", "rid-3"])
    rc, out, err = w.check(capsys, **{"--audit-name": "supp-x", "--marker-name": "supp-marker"})
    assert rc == 0, err
    # a folder or a marker that is not there; each is looked for under the name given, not the default
    fresh = iter(str(tmp_path / f"stage-{i}") for i in range(20))  # the first check made the default stage: every refusal gets its own
    assert refusal(capsys, w, **{"--audit-name": "absent", "--marker-name": "supp-marker", "--stage": next(fresh)}) == f"REFUSED audit-folder-missing out={OUT}\n"
    assert refusal(capsys, w, **{"--audit-name": "supp-x", "--marker-name": "absent", "--stage": next(fresh)}) == f"REFUSED no-done-marker out={OUT} phase=natural\n"
    (w.done / "supp-marker").write_text(f"{'e' * 64} ok {MARKER_TIME}\n")  # a marker that names another matrix
    assert refusal(capsys, w, **{"--audit-name": "supp-x", "--marker-name": "supp-marker", "--stage": next(fresh)}) == (
        f"REFUSED run-json-differs out={OUT} what=matrix-sha256\n")
    for over in ({"--audit-name": "../x"}, {"--audit-name": "A"}, {"--audit-name": ""}, {"--marker-name": "a/b"}, {"--marker-name": ".hidden"}, {"--marker-name": ""}):
        assert refusal(capsys, w, **over, **{"--stage": next(fresh)}) == "REFUSED bad-name\n", over


# -- a select of the OUT's rows, a matrix of several OUTs, and the join check ---------------------------------------------------------


GEM = ["synthetic-gem-p0of3", "synthetic-gem-p1of3", "synthetic-gem-p2of3"]
COPIED = "synthetic-copied"


class Gem:
    """Three plain partitions of three records each (q1, q2 copied; q3 new), each with a matrix over the whole partition (its middle
    record has no output), and one matrix over the six copied records of all three, in a folder and marker named apart."""

    def __init__(self, tmp: Path, monkeypatch) -> None:
        monkeypatch.setattr(a15, "NATURAL_OUTS", (OUT, *GEM))
        monkeypatch.setattr(a15, "COMBINED_NAMES", (COPIED,))
        self.tmp = tmp
        self.worlds = []
        for i, g in enumerate(GEM):
            items = [("q1", f"rid-{i}0"), ("q2", f"rid-{i}1"), ("q3", f"rid-{i}2")]
            w = AWorld(tmp, out=g, phase="natural", items=items, view_name=g)
            w.set_matrix(matrix_bytes([r for _, r in items], arms=ARM_LIST, no_output={f"rid-{i}1"}))
            self.worlds.append(w)
        self.copied_ids = [f"rid-{i}{j}" for i in range(3) for j in (0, 1)]
        add_supplement(self.worlds[0], COPIED, self.copied_ids, marker=f"{COPIED}-natural", no_output={f"rid-{i}1" for i in range(3)})
        self.new = [names_file(tmp / f"new{i}", [rec_name("q3", f"rid-{i}2")]) for i in range(3)]
        self.copied = [names_file(tmp / f"cop{i}", sorted([rec_name("q1", f"rid-{i}0"), rec_name("q2", f"rid-{i}1")])) for i in range(3)]
        self.views = [w.view for w in self.worlds]

    @property
    def base(self) -> list:
        w = self.worlds[0]
        return ["--audit-root", str(w.audit_root), "--done-dir", str(w.done)]

    def part(self, i: int, stage: Path, *extra) -> list:
        return ["check-a15", *self.base, "--out-dir", str(self.views[i]), "--out-name", GEM[i], "--phase", "natural",
                "--stage", str(stage), *extra]

    def copied_argv(self, stage: Path, *, lists=None, outs=None, name=COPIED, audit=COPIED, marker=f"{COPIED}-natural") -> list:
        argv = ["check-a15", *self.base]
        for v in (self.views if outs is None else outs):
            argv += ["--out-dir", str(v)]
        argv += ["--out-name", name, "--phase", "natural", "--stage", str(stage)]
        if audit:
            argv += ["--audit-name", audit]
        if marker:
            argv += ["--marker-name", marker]
        for spec in (self.copied if lists is None else lists):
            argv += ["--only-names", spec]
        return argv

    def join(self, stages, outs=None) -> list:
        argv = ["join-check-a15"]
        for v in (self.views if outs is None else outs):
            argv += ["--out-dir", str(v)]
        for st in stages:
            argv += ["--stage", str(st)]
        return argv

    def stages(self) -> list:
        return [self.tmp / f"stage-p{i}" for i in range(3)] + [self.tmp / "stage-copied"]

    def run_all(self, capsys) -> list:
        """The three partition checks (select the new record) and the copied check; their (rc, out, err)."""
        st = self.stages()
        results = [run_main(capsys, self.part(i, st[i], "--select-names", self.new[i])) for i in range(3)]
        return results + [run_main(capsys, self.copied_argv(st[3]))]


def run_main(capsys, argv):
    rc = a15.main(argv)
    out, err = capsys.readouterr()
    assert "rid-" not in out + err and not re.search(r"[0-9a-f]{40}\.", out + err)
    return rc, out, err


def staged_ids(stage: Path, phase_out: str) -> list:
    lines = (stage / "matrices.list").read_text().splitlines()
    assert len(lines) == 1
    return sorted({json.loads(line)["run_id"] for line in gzip.open(lines[0], "rt")})


def test_select_names_stages_a_copy_of_the_listed_records_rows_only_and_records_both_shas(tmp_path, capsys, monkeypatch):
    g = Gem(tmp_path, monkeypatch)
    stage = g.stages()[1]
    rc, out, err = run_main(capsys, g.part(1, stage, "--select-names", g.new[1]))
    assert (rc, err) == (0, "")
    w = g.worlds[1]
    original = stage / GEM[1] / "natural" / f"audit-{GEM[1]}" / "audit.matrix.jsonl.gz"
    chosen = original.with_name("audit.selected.matrix.jsonl.gz")
    assert sha(original.read_bytes()) == w.matrix_sha  # the original copy is the marker's matrix
    assert (stage / "matrices.list").read_text() == f"{chosen}\n" and staged_ids(stage, GEM[1]) == ["rid-12"]
    kept = [ln for ln in gzip.open(original, "rb") if json.loads(ln)["run_id"] == "rid-12"]  # byte for byte the original's rows of that run
    assert kept and list(gzip.open(chosen, "rb")) == kept and len(kept) == 32
    chosen_sha = sha(chosen.read_bytes())
    manifest = (f"out\tphase\ttree\tshard\tsha256\trecords\toriginal_sha256\n"
                f"{GEM[1]}\tnatural\taudit-{GEM[1]}\taudit\t{chosen_sha}\t1\t{w.matrix_sha}\n")
    assert (stage / "replay-matrices-manifest.tsv").read_text() == manifest
    assert (stage / "replay-outputs.list").read_text() == "rid-12\n"
    marker_file_sha = sha(f"{w.matrix_sha} ok {MARKER_TIME}\n".encode())
    assert out.splitlines()[0] == (f"LOCATED-A15 out={GEM[1]} phase=natural marker_sha256={marker_file_sha} matrix_sha256={w.matrix_sha} "
                                   f"disclosure=0 selected_sha256={chosen_sha}")
    assert out.splitlines()[-1] == (f"COVERAGE out={GEM[1]} order_keys_natural=3 records_natural=3 rows_natural=96 selected_natural=1 "
                                    f"selected_rows_natural=32 shards_natural=1 selection=select names_listed=1 matrices=1 "
                                    f"manifest_sha256={sha(manifest.encode())} OK")


@pytest.mark.parametrize("what", ["missing", "extra", "unknown-name", "other-partitions-name", "listed-twice"])
def test_select_names_still_needs_the_whole_partition_covered_and_only_its_records_listed(tmp_path, capsys, monkeypatch, what):
    g = Gem(tmp_path, monkeypatch)
    w, spec = g.worlds[0], g.new[0]
    if what == "missing":  # the partition matrix lacks a record that is not even a listed one
        w.set_matrix(matrix_bytes(["rid-02", "rid-01"], arms=ARM_LIST))
        expected = "REFUSED matrix-differs-from-records out_records=3 matrix_records=2 missing=1 extra=0 rows=64 rows_expected=96\n"
    elif what == "extra":
        w.set_matrix(matrix_bytes(["rid-00", "rid-01", "rid-02", "rid-77"], arms=ARM_LIST))
        expected = "REFUSED matrix-differs-from-records out_records=3 matrix_records=4 missing=0 extra=1 rows=128 rows_expected=96\n"
    elif what == "unknown-name":
        spec, expected = names_file(tmp_path / "u", [rec_name("q4", "rid-99")]), "REFUSED names-not-records unknown=1\n"
    elif what == "other-partitions-name":
        spec, expected = g.new[1], "REFUSED names-not-records unknown=1\n"
    else:
        spec = names_file(tmp_path / "t", [], text=rec_name("q3", "rid-02") + "\n" + rec_name("q3", "rid-02") + "\n")
        expected = "REFUSED names-malformed\n"
    rc, out, err = run_main(capsys, g.part(0, g.stages()[0], "--select-names", spec))
    assert (rc, out, err) == (2, "", expected)


def test_select_names_keeps_the_markers_sha_binding_of_the_whole_original(tmp_path, capsys, monkeypatch):
    g = Gem(tmp_path, monkeypatch)
    w = g.worlds[2]
    (w.folder / "matrix.jsonl.gz").write_bytes(matrix_bytes(["rid-20", "rid-21", "rid-22"], arms=ARM_LIST, no_output={"rid-20"}))  # not the marker's bytes
    stage = g.stages()[2]
    assert run_main(capsys, g.part(2, stage, "--select-names", g.new[2])) == (2, "", f"REFUSED matrix-sha-differs out={GEM[2]}\n")
    assert not stage.exists()


def test_a_matrix_of_several_outs_is_wanted_as_the_union_of_the_lists(tmp_path, capsys, monkeypatch):
    g = Gem(tmp_path, monkeypatch)
    stage = g.stages()[3]
    rc, out, err = run_main(capsys, g.copied_argv(stage))
    assert (rc, err) == (0, "")
    assert out.splitlines()[-1].startswith(f"COVERAGE out={COPIED} order_keys_natural=9 records_natural=6 rows_natural=192 shards_natural=1 "
                                           "selection=only names_listed=6 matrices=1 manifest_sha256=")
    assert staged_ids(stage, COPIED) == sorted(g.copied_ids)
    assert (stage / COPIED / "natural" / f"audit-{COPIED}" / "audit.matrix.jsonl.gz").is_file()


def test_the_matrix_of_several_outs_refuses_extra_or_missing_records_like_any_other(tmp_path, capsys, monkeypatch):
    g = Gem(tmp_path, monkeypatch)
    w = g.worlds[0]
    stages = iter(g.tmp / f"s{i}" for i in range(9))
    def swap(ids, **kw):
        data = matrix_bytes(ids, arms=ARM_LIST, **kw)
        (w.audit_root / f"audit-{COPIED}" / "matrix.jsonl.gz").write_bytes(data)
        (w.audit_root / f"audit-{COPIED}" / "run.json").write_text(json.dumps(run_json(sha(data))))
        (w.done / f"{COPIED}-natural").write_text(f"{sha(data)} ok {MARKER_TIME}\n")
    swap(g.copied_ids + ["rid-77"])
    assert run_main(capsys, g.copied_argv(next(stages))) == (
        2, "", "REFUSED matrix-differs-from-records out_records=6 matrix_records=7 missing=0 extra=1 rows=224 rows_expected=192\n")
    swap(g.copied_ids[:-1])
    assert run_main(capsys, g.copied_argv(next(stages))) == (
        2, "", "REFUSED matrix-differs-from-records out_records=6 matrix_records=5 missing=1 extra=0 rows=160 rows_expected=192\n")


def test_the_lists_of_a_matrix_of_several_outs_must_be_records_of_them_and_none_twice(tmp_path, capsys, monkeypatch):
    g = Gem(tmp_path, monkeypatch)
    stages = iter(g.tmp / f"s{i}" for i in range(9))
    twice = names_file(tmp_path / "x", [rec_name("q1", "rid-00")])  # also in the first list
    assert run_main(capsys, g.copied_argv(next(stages), lists=[*g.copied, twice])) == (2, "", "REFUSED names-twice twice=1\n")
    assert run_main(capsys, g.copied_argv(next(stages), lists=[*g.copied, g.copied[0]])) == (2, "", "REFUSED names-twice twice=2\n")
    nobody = names_file(tmp_path / "y", [rec_name("q4", "rid-99")])
    assert run_main(capsys, g.copied_argv(next(stages), lists=[*g.copied, nobody])) == (2, "", "REFUSED names-not-records unknown=1\n")
    # one record file name in two of the OUTs
    (g.views[1] / "records" / rec_name("q1", "rid-10")).unlink()
    (g.views[1] / "records" / rec_name("q1", "rid-00")).write_text("{}")  # the same file name, in two OUTs that are each well formed
    assert run_main(capsys, g.copied_argv(next(stages))) == (2, "", "REFUSED name-in-two-outs\n")


def test_the_combined_name_is_served_only_in_its_one_form(tmp_path, capsys, monkeypatch):
    g = Gem(tmp_path, monkeypatch)
    stages = iter(g.tmp / f"s{i}" for i in range(20))
    bad = "REFUSED bad-name\n"
    for kw in ({"audit": None}, {"marker": None}, {"audit": "other"}, {"marker": "other"}, {"marker": COPIED},  # the marker's file name has the phase
               {"marker": f"{COPIED}-prompted"}, {"outs": g.views[:1]}, {"name": "synthetic-other-copied"},
               {"name": GEM[0]}):  # the last: a partition name with several OUTs
        assert run_main(capsys, g.copied_argv(next(stages), **kw)) == (2, "", bad), kw
    base = g.copied_argv(next(stages))
    cut = base.index("--only-names")
    for tail in (["--exclude-names", g.copied[0]], ["--select-names", g.copied[0]], []):  # not --only-names, or no selection at all
        assert run_main(capsys, [*base[:cut], *tail]) == (2, "", bad), tail
    # a partition name with several OUT dirs
    assert run_main(capsys, g.part(0, next(stages)) + ["--out-dir", str(g.views[1])]) == (2, "", bad)
    with pytest.raises(SystemExit):
        a15.main(g.part(0, next(stages), "--select-names", g.new[0], "--only-names", g.new[0]))


def test_the_join_check_accepts_stages_that_are_disjoint_and_together_the_records(tmp_path, capsys, monkeypatch):
    g = Gem(tmp_path, monkeypatch)
    assert all(rc == 0 for rc, _, _ in g.run_all(capsys))
    st = g.stages()
    assert run_main(capsys, g.join(st)) == (0, "JOIN-CHECK kind=a15 stages=4 outs=3 records=9 OK\n", "")
    # the exporter's ledger across the four stages: each arm, the runs with an output (rid-i2 new, rid-i0 copied; the middle ones have none)
    ids = [f"rid-{i}{j}" for i in range(3) for j in (0, 2)]
    assert check_many(capsys, st, ledger(tmp_path / "l.jsonl", replay_rows(ids))) == (0, "LEDGER-CHECK kind=a15 arms=8 OK\n", "")
    assert check_many(capsys, st, ledger(tmp_path / "l2.jsonl", replay_rows(ids[:-1]))) == (
        2, "", "REFUSED ledger-differs-from-matrices missing=1 extra=0 duplicate=0 other_sources=0\n")


def test_the_join_check_refuses_a_record_in_neither_in_both_or_not_the_outs(tmp_path, capsys, monkeypatch):
    g = Gem(tmp_path, monkeypatch)
    st = g.stages()
    assert all(rc == 0 for rc, _, _ in g.run_all(capsys))
    # a record in neither: partition 1's new stage is replaced by a stage whose matrix holds nothing of that partition
    two = g.tmp / "second"
    two.mkdir()
    g2 = Gem(two, monkeypatch)
    cop = names_file(tmp_path / "c1", [rec_name("q1", "rid-10")])  # the copied list leaves rid-11 out
    cm = [matrix_bytes([f"rid-{i}{j}" for i in range(3) for j in (0, 1) if (i, j) != (1, 1)], arms=ARM_LIST)]
    w = g2.worlds[0]
    (w.audit_root / f"audit-{COPIED}" / "matrix.jsonl.gz").write_bytes(cm[0])
    (w.audit_root / f"audit-{COPIED}" / "run.json").write_text(json.dumps(run_json(sha(cm[0]))))
    (w.done / f"{COPIED}-natural").write_text(f"{sha(cm[0])} ok {MARKER_TIME}\n")
    s2 = [two / f"stage-p{i}" for i in range(3)] + [two / "stage-copied"]
    for i in range(3):
        assert run_main(capsys, g2.part(i, s2[i], "--select-names", g2.new[i]))[0] == 0
    lists = [g2.copied[0], cop, g2.copied[2]]
    assert run_main(capsys, g2.copied_argv(s2[3], lists=lists))[0] == 0
    assert run_main(capsys, g2.join(s2)) == (2, "", "REFUSED join-differs missing=1 extra=0 twice=0\n")  # rid-11 is in neither list
    # a record in both: the first partition's new record is also listed as copied
    three = g.tmp / "third"
    three.mkdir()
    g3 = Gem(three, monkeypatch)
    both = names_file(tmp_path / "b0", sorted([rec_name("q1", "rid-00"), rec_name("q2", "rid-01"), rec_name("q3", "rid-02")]))
    ids = [f"rid-0{j}" for j in range(3)] + [f"rid-{i}{j}" for i in (1, 2) for j in (0, 1)]
    w3 = g3.worlds[0]
    data = matrix_bytes(ids, arms=ARM_LIST)
    (w3.audit_root / f"audit-{COPIED}" / "matrix.jsonl.gz").write_bytes(data)
    (w3.audit_root / f"audit-{COPIED}" / "run.json").write_text(json.dumps(run_json(sha(data))))
    (w3.done / f"{COPIED}-natural").write_text(f"{sha(data)} ok {MARKER_TIME}\n")
    s3 = [three / f"stage-p{i}" for i in range(3)] + [three / "stage-copied"]
    for i in range(3):
        assert run_main(capsys, g3.part(i, s3[i], "--select-names", g3.new[i]))[0] == 0
    assert run_main(capsys, g3.copied_argv(s3[3], lists=[both, g3.copied[1], g3.copied[2]]))[0] == 0
    assert run_main(capsys, g3.join(s3)) == (2, "", "REFUSED join-differs missing=0 extra=0 twice=1\n")
    # the OUTs are not the stages' records
    assert run_main(capsys, g.join(st[:3])) == (2, "", "REFUSED join-differs missing=6 extra=0 twice=0\n")
    assert run_main(capsys, g.join(st, outs=g.views[:2])) == (2, "", "REFUSED join-differs missing=0 extra=3 twice=0\n")


def test_the_join_check_reads_the_staged_matrices_so_a_changed_stage_is_refused(tmp_path, capsys, monkeypatch):
    g = Gem(tmp_path, monkeypatch)
    st = g.stages()
    assert all(rc == 0 for rc, _, _ in g.run_all(capsys))
    chosen = st[0] / GEM[0] / "natural" / f"audit-{GEM[0]}" / "audit.selected.matrix.jsonl.gz"
    chosen.chmod(0o600)
    chosen.write_bytes(matrix_bytes(["rid-02", "rid-77"], arms=ARM_LIST))  # a filtered stage that is not the listed set
    assert run_main(capsys, g.join(st)) == (2, "", "REFUSED join-differs missing=0 extra=1 twice=0\n")
    chosen.write_bytes(matrix_bytes([], arms=ARM_LIST))
    assert run_main(capsys, g.join(st)) == (2, "", "REFUSED join-differs missing=1 extra=0 twice=0\n")
    (st[0] / "matrices.list").write_text(f"{tmp_path}/elsewhere.gz\n")  # a list that leaves the stage
    assert run_main(capsys, g.join(st)) == (2, "", "REFUSED stage-list\n")
    (st[0] / "matrices.list").write_text("a\nb\n")
    assert run_main(capsys, g.join(st)) == (2, "", "REFUSED stage-list\n")


# -- ledger-check-a15 ---------------------------------------------------------------------------------------------------------


def ledger(path: Path, rows: list, header=None) -> Path:
    head = header if header is not None else json.dumps({"schema": "checker-ledger/2", "split": "test"})
    path.write_text("\n".join([head] + [json.dumps(r) for r in rows]) + "\n")
    return path


def replay_rows(run_ids, arms=ARM_LIST):
    return [{"split": "test", "run_id": rid, "source": arm, "use": "replay", "status": "P"} for arm in arms for rid in run_ids]


def staged_world(tmp_path, capsys) -> AWorld:
    w = AWorld(tmp_path)
    w.set_matrix(matrix_bytes(["rid-1", "rid-2", "rid-3"], arms=ARM_LIST, no_output={"rid-3"}))
    assert w.check(capsys)[0] == 0
    return w


def check_ledger(capsys, w: AWorld, path: Path):
    rc = a15.main(["ledger-check-a15", "--stage", str(w.stage), "--ledger", str(path)])
    out, err = capsys.readouterr()
    return rc, out, err


def test_a_ledger_with_each_of_the_eight_arms_once_per_run_with_an_output_is_accepted(tmp_path, capsys):
    w = staged_world(tmp_path, capsys)
    rc, out, err = check_ledger(capsys, w, ledger(tmp_path / "l.jsonl", replay_rows(["rid-1", "rid-2"])))
    assert (rc, out, err) == (0, "LEDGER-CHECK kind=a15 arms=8 OK\n", "")


@pytest.mark.parametrize("rows, expected", [
    (lambda: replay_rows(["rid-1", "rid-2"])[:-1], "REFUSED ledger-differs-from-matrices missing=1 extra=0 duplicate=0 other_sources=0\n"),
    (lambda: replay_rows(["rid-1", "rid-2"], ARM_LIST[:-1]), "REFUSED ledger-differs-from-matrices missing=1 extra=0 duplicate=0 other_sources=0\n"),
    (lambda: replay_rows(["rid-1", "rid-2"]) + [replay_rows(["rid-1"])[0]],
     "REFUSED ledger-differs-from-matrices missing=0 extra=0 duplicate=1 other_sources=0\n"),
    (lambda: replay_rows(["rid-1", "rid-2", "rid-3"]), "REFUSED ledger-differs-from-matrices missing=0 extra=1 duplicate=0 other_sources=0\n"),
    (lambda: replay_rows(["rid-1", "rid-2"]) + replay_rows(["rid-1"], ["census-t2"]),
     "REFUSED ledger-differs-from-matrices missing=0 extra=0 duplicate=0 other_sources=1\n"),
    (lambda: [], "REFUSED ledger-differs-from-matrices missing=1 extra=0 duplicate=0 other_sources=0\n"),
])
def test_a_ledger_that_is_not_the_matrixs_outputs_per_arm_is_refused(tmp_path, capsys, rows, expected):
    w = staged_world(tmp_path, capsys)
    rc, out, err = check_ledger(capsys, w, ledger(tmp_path / "l.jsonl", rows()))
    assert (rc, out, err) == (2, "", expected)


def test_a_ledger_with_another_header_or_a_stage_with_no_outputs_is_refused(tmp_path, capsys):
    w = staged_world(tmp_path, capsys)
    bad = ledger(tmp_path / "l.jsonl", replay_rows(["rid-1", "rid-2"]), header=json.dumps({"schema": "checker-ledger/2", "split": "dev"}))
    assert check_ledger(capsys, w, bad)[1:] == ("", "REFUSED ledger-header\n")
    (w.stage / "replay-outputs.list").write_text("")
    good = ledger(tmp_path / "l2.jsonl", [])
    assert check_ledger(capsys, w, good)[1:] == ("", "REFUSED no-outputs-listed\n")


# -- ledger-check-a15 across the partitions of a union ------------------------------------------------------------------------


def two_stages(tmp_path, capsys):
    """Two partitions (their own views, folders, markers and run ids) checked into two stages; the pair of stage paths."""
    a = AWorld(tmp_path / "pa", items=THREE)
    a.set_matrix(matrix_bytes(["rid-1", "rid-2", "rid-3"], arms=ARM_LIST, no_output={"rid-3"}))
    b = AWorld(tmp_path / "pb", out=OUT_P, phase="prompted", items=[("q1", "rid-4"), ("q2", "rid-5"), ("q3", "rid-6")])
    b.set_matrix(matrix_bytes(["rid-4", "rid-5", "rid-6"], arms=ARM_LIST, no_output={"rid-4"}))
    assert a.check(capsys)[0] == 0 and b.check(capsys)[0] == 0
    return a, b


def check_many(capsys, stages, path: Path):
    argv = ["ledger-check-a15"] + [x for s in stages for x in ("--stage", str(s))] + ["--ledger", str(path)]
    rc = a15.main(argv)
    out, err = capsys.readouterr()
    return rc, out, err


def test_a_ledger_across_two_stages_holds_each_arm_once_for_the_runs_of_all_of_them(tmp_path, capsys):
    a, b = two_stages(tmp_path, capsys)  # outputs: rid-1, rid-2 and rid-5, rid-6
    good = ledger(tmp_path / "l.jsonl", replay_rows(["rid-1", "rid-2", "rid-5", "rid-6"]))
    before = sorted(p.name for p in a.stage.iterdir())
    assert check_many(capsys, [a.stage, b.stage], good) == (0, "LEDGER-CHECK kind=a15 arms=8 OK\n", "")
    assert sorted(p.name for p in a.stage.iterdir()) == before  # the combined list's directory is gone
    for rows, expected in (
            (replay_rows(["rid-1", "rid-2", "rid-5", "rid-6"])[:-1], "missing=1 extra=0 duplicate=0 other_sources=0"),  # a run of the second stage dropped
            (replay_rows(["rid-1", "rid-2", "rid-5"]), "missing=1 extra=0 duplicate=0 other_sources=0"),  # one stage's runs only
            (replay_rows(["rid-1", "rid-2", "rid-5", "rid-6", "rid-3"]), "missing=0 extra=1 duplicate=0 other_sources=0"),  # a run with no output
            (replay_rows(["rid-1", "rid-2", "rid-5", "rid-6"]) + replay_rows(["rid-5"], ["census-t2"]), "missing=0 extra=0 duplicate=0 other_sources=1")):
        bad = ledger(tmp_path / "bad.jsonl", rows)
        assert check_many(capsys, [a.stage, b.stage], bad) == (2, "", f"REFUSED ledger-differs-from-matrices {expected}\n")


def test_the_stages_of_one_ledger_check_must_not_share_a_run_id(tmp_path, capsys):
    a, b = two_stages(tmp_path, capsys)
    good = ledger(tmp_path / "l.jsonl", replay_rows(["rid-1", "rid-2", "rid-5", "rid-6"]))
    assert check_many(capsys, [a.stage, a.stage], good) == (2, "", "REFUSED stages-overlap\n")
    (b.stage / "replay-outputs.list").write_text("rid-2\nrid-6\n")
    assert check_many(capsys, [a.stage, b.stage], good) == (2, "", "REFUSED stages-overlap\n")
    (b.stage / "replay-outputs.list").write_text("")
    (a.stage / "replay-outputs.list").write_text("")
    assert check_many(capsys, [a.stage, b.stage], good) == (2, "", "REFUSED no-outputs-listed\n")


def test_one_stage_and_the_reviewed_check_agree(tmp_path, capsys):
    w = staged_world(tmp_path, capsys)
    for rows in (replay_rows(["rid-1", "rid-2"]), replay_rows(["rid-1", "rid-2"])[:-1], replay_rows(["rid-1"])):
        path = ledger(tmp_path / "l.jsonl", rows)
        try:
            rm.ledger_check(w.stage, path, a15.ARMS)
            reviewed = (0, "LEDGER-CHECK kind=a15 arms=8 OK\n", "")
        except rm.Refused as exc:
            reviewed = (2, "", f"REFUSED {exc}\n")
        assert check_ledger(capsys, w, path) == reviewed


# -- the command line under -I ------------------------------------------------------------------------------------------------


def test_the_tool_runs_under_isolated_mode_and_finds_the_reviewed_module_by_path(tmp_path):
    """python3 -I leaves the script's directory off sys.path: a plain ``import replay_matrices`` would fail. The wrapper only
    patches the tool's pair list (the real list names real OUTs); it is no copy of the tool."""
    w = AWorld(tmp_path)
    wrapper = tmp_path / "wrap.py"
    wrapper.write_text(
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('a15_tool', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['a15_tool'] = m\n"
        "spec.loader.exec_module(m)\n"
        f"m.NATURAL_OUTS = ({OUT!r},)\n"
        "sys.exit(m.main(sys.argv[2:]))\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-I", str(wrapper), str(A15DRIVER), *w.argv()], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert r.stdout.splitlines()[-1].startswith(f"COVERAGE out={OUT} order_keys_natural=3 records_natural=3 rows_natural=96 ")
    assert "ModuleNotFoundError" not in r.stderr
    bad = subprocess.run([sys.executable, "-I", str(A15DRIVER), *w.argv(**{"--stage": str(tmp_path / "other-stage")})],
                         capture_output=True, text=True, env=env, cwd=tmp_path)
    assert bad.returncode == 2 and bad.stderr == "REFUSED bad-name\n" and bad.stdout == ""  # the real list does not hold the synthetic OUT


def test_the_module_compiles_and_has_no_test_switch():
    text = A15DRIVER.read_text()
    compile(text, str(A15DRIVER), "exec")
    assert "TEST_MODE" not in text and "environ" not in text  # nothing in the environment changes what it accepts
