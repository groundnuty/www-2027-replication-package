"""Tests of driver/a15_class_table.py (the names-only table of the copies behind each OUT's outputs and behind its replay).

Synthetic tables only: copy names, endpoints and graphs here are made up (``synthetic-*``, ``copy-*``), the OUT-phase list of the tool
is monkeypatched, and every expected cell of the table is written by hand.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from test_replay_matrices import DRIVER

TOOL = DRIVER.parent / "a15_class_table.py"
_spec = importlib.util.spec_from_file_location("a15_class_table", TOOL)
ct = importlib.util.module_from_spec(_spec)
sys.modules["a15_class_table"] = ct
_spec.loader.exec_module(ct)

GRAPH = "synthetic-graph"
EP = {"a": "http://ep-a:1", "b": "http://ep-b:2", "c": "http://ep-c:3"}
REPLAY_SHA = {"copy-a": "a1" * 32, "copy-b": "b2" * 32}
HEADER = "out\tphase\toutput_copies\toutput_attempts\toutput_classes\treplay_copy\treplay_source\treplay_class\tsame_class\n"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(autouse=True)
def synthetic_outs(monkeypatch):
    monkeypatch.setattr(ct.a15, "NATURAL_OUTS", ("synthetic-out", "synthetic-out-2"))
    monkeypatch.setattr(ct.a15, "PROMPTED_OUTS", ("synthetic-out-p",))


def classes_doc(**classes) -> dict:
    return {"schema": "e189-copy-classes/1", "rule": "synthetic", "sources": {"src": {"path": "p", "sha256": "1" * 64}},
            "copies": {c: {"class": k, "label": k if k else ct.cc.NO_RULE_LABEL, "basis": "b", "source": "src"} for c, k in classes.items()}}


def endpoints_doc(**by_key) -> dict:
    return {"schema": "e189-endpoint-copies/1", "rule": "synthetic",
            "copies": [{"graph": GRAPH, "endpoint_sha256": sha(EP[k]), "replica": replica, "basis": "b",
                        "sources": [{"path": "p", "sha256": "2" * 64}]} for k, replica in by_key.items()]}


def change(src: str, dst: str, before: list) -> dict:
    return {"from": {"endpoint": EP[src]}, "to": {"endpoint": EP[dst]}, "pre_change_records": len(before), "pre_change_run_ids": before}


class Case:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        tmp.mkdir(parents=True, exist_ok=True)
        self.classes = tmp / "classes.json"
        self.endpoints = tmp / "endpoints.json"
        self.pins = tmp / "pins.tsv"
        self.rows = tmp / "rows.tsv"
        self.out = tmp / "table.tsv"
        self.write(self.classes, classes_doc(**{"copy-a": "fast", "copy-b": "slow", "copy-c": "fast", "copy-n": None}))
        self.write(self.endpoints, endpoints_doc(a="copy-a", b="copy-b", c="copy-c"))
        self.pins.write_text("# the launchers' copy table\n" + "".join(f"{c}\t{s}\n" for c, s in REPLAY_SHA.items()))
        self.lines: list[str] = []

    @staticmethod
    def write(path: Path, doc) -> None:
        path.write_text(json.dumps(doc))

    def add(self, out="synthetic-out", phase="natural", *, endpoint="a", changes=None, description=REPLAY_SHA["copy-a"], manifest=None, run=None,
            version=None):
        n = len(self.lines)
        m = self.tmp / f"manifest-{n}.json"
        r = self.tmp / f"run-{n}.json"
        self.write(m, manifest if manifest is not None else {
            "fingerprint": {"store": {"name": GRAPH, "endpoint": EP[endpoint]}}, **({"store_changes": changes} if changes is not None else {})})
        self.write(r, run if run is not None else {"schema": "x", "final": True, "store": {"description_sha256": description, "gate_slots": 1, **({"version": version} if version else {})}})
        self.lines.append(f"{out}\t{phase}\t{m}\t{r}\n")

    def argv(self, **over) -> list:
        self.rows.write_text("".join(self.lines))
        a = {"--rows": self.rows, "--classes": self.classes, "--classes-sha256": hashlib.sha256(self.classes.read_bytes()).hexdigest(),
             "--endpoints": self.endpoints, "--endpoints-sha256": hashlib.sha256(self.endpoints.read_bytes()).hexdigest(),
             "--description-pins": self.pins, "--description-pins-sha256": hashlib.sha256(self.pins.read_bytes()).hexdigest(),
             "--out": self.out, **over}
        return [x for kv in a.items() for x in (kv[0], str(kv[1]))]

    def run(self, capsys, **over):
        rc = ct.main(self.argv(**over))
        out, err = capsys.readouterr()
        return rc, out, err

    def table(self) -> list:
        return self.out.read_text().splitlines(keepends=True)


def refused(capsys, case: Case, expected: str, **over) -> None:
    rc, out, err = case.run(capsys, **over)
    assert (rc, out, err) == (2, "", f"REFUSED {expected}\n")
    assert not case.out.exists()


# -- the classes of the copies ------------------------------------------------------------------------------------------------


def test_one_copy_replayed_on_a_copy_of_its_class_is_yes(tmp_path, capsys):
    c = Case(tmp_path)
    c.add(description=REPLAY_SHA["copy-a"])
    rc, out, err = c.run(capsys)
    assert (rc, out, err) == (0, "ok rows=1\n", "")
    assert c.table() == [HEADER, "synthetic-out\tnatural\tcopy-a\t-\tfast\tcopy-a\tdescription-pin\tfast\tyes\n"]
    assert stat.S_IMODE(c.out.stat().st_mode) == 0o600


def test_another_class_is_no_and_a_mix_is_mixed_with_the_attempts_the_manifest_gives(tmp_path, capsys):
    c = Case(tmp_path)
    c.add(description=REPLAY_SHA["copy-b"])  # one fast copy, replayed on a slow one
    c.add("synthetic-out-p", "prompted", endpoint="a", changes=[change("a", "b", ["r1", "r2"])], description=REPLAY_SHA["copy-b"])  # fast then slow, replayed on slow
    c.add("synthetic-out-2", "natural", endpoint="a", changes=[change("a", "b", ["r1", "r2"]), change("b", "c", ["r1", "r2", "r3", "r4", "r5"])],
          description=REPLAY_SHA["copy-a"])  # fast, slow, fast: 2, 3 and no count; replayed on fast
    c.add("synthetic-out-p", "natural", endpoint="b", changes=[change("a", "b", ["r1"])], description=REPLAY_SHA["copy-b"])
    rc, out, err = c.run(capsys)
    assert (rc, out, err) == (2, "", "REFUSED rows-malformed\n")  # ("synthetic-out-p", "natural") is not a pinned pair
    c.lines.pop()
    rc, out, err = c.run(capsys)
    assert (rc, out, err) == (0, "ok rows=3\n", "")
    assert c.table() == [
        HEADER,
        "synthetic-out\tnatural\tcopy-a\t-\tfast\tcopy-b\tdescription-pin\tslow\tno\n",
        "synthetic-out-p\tprompted\tcopy-a,copy-b\t2,-\tfast,slow\tcopy-b\tdescription-pin\tslow\tmixed\n",
        "synthetic-out-2\tnatural\tcopy-a,copy-b,copy-c\t2,3,-\tfast,slow,fast\tcopy-a\tdescription-pin\tfast\tmixed\n"]


def test_a_chain_that_returns_to_a_copy_lists_the_copy_each_time(tmp_path, capsys):
    c = Case(tmp_path)
    c.add(endpoint="a", changes=[change("a", "b", ["r1"]), change("b", "a", ["r1", "r2"])], description=REPLAY_SHA["copy-a"])
    assert c.run(capsys)[0] == 0
    assert c.table()[1] == "synthetic-out\tnatural\tcopy-a,copy-b,copy-a\t1,1,-\tfast,slow,fast\tcopy-a\tdescription-pin\tfast\tmixed\n"
    # the manifest's endpoint is the chain's first or last: c is neither here, so that manifest is refused
    c2 = Case(tmp_path / "second")
    c2.add(endpoint="c", changes=[change("a", "b", ["r1"]), change("b", "a", ["r1", "r2"])])
    refused(capsys, c2, "manifest-endpoint-not-in-chain")


def test_a_copy_no_table_names_or_without_a_class_is_unknown_and_so_is_the_comparison(tmp_path, capsys):
    c = Case(tmp_path)
    c.write(c.endpoints, endpoints_doc(a="copy-a", b="copy-n"))  # b is a copy whose class is null
    c.add(endpoint="b", description=REPLAY_SHA["copy-a"])  # null class
    c.add("synthetic-out-p", "prompted", endpoint="c", description=REPLAY_SHA["copy-a"])  # an endpoint no table names
    c.add("synthetic-out-2", "natural", endpoint="a", description="e5" * 32)  # a description no pin lists
    assert c.run(capsys)[0] == 0
    assert c.table() == [
        HEADER,
        "synthetic-out\tnatural\tcopy-n\t-\tunknown\tcopy-a\tdescription-pin\tfast\tunknown\n",
        "synthetic-out-p\tprompted\tunknown\t-\tunknown\tcopy-a\tdescription-pin\tfast\tunknown\n",
        "synthetic-out-2\tnatural\tcopy-a\t-\tfast\tunknown\tunknown\tunknown\tunknown\n"]


def test_a_replica_the_class_table_lacks_has_the_class_unknown(tmp_path, capsys):
    c = Case(tmp_path)
    c.write(c.classes, classes_doc(**{"copy-a": "fast"}))
    c.add(endpoint="b", description=REPLAY_SHA["copy-b"])
    assert c.run(capsys)[0] == 0
    assert c.table()[1] == "synthetic-out\tnatural\tcopy-b\t-\tunknown\tcopy-b\tdescription-pin\tunknown\tunknown\n"


def test_the_graph_of_the_manifest_is_part_of_the_endpoint_key(tmp_path, capsys):
    c = Case(tmp_path)
    c.add(manifest={"fingerprint": {"store": {"name": "synthetic-other", "endpoint": EP["a"]}}})
    assert c.run(capsys)[0] == 0
    assert c.table()[1].split("\t")[2] == "unknown"


# -- the pins and the inputs --------------------------------------------------------------------------------------------------


def test_a_table_that_is_not_its_pin_is_refused_before_it_is_read(tmp_path, capsys):
    c = Case(tmp_path)
    c.add()
    refused(capsys, c, "pin-differs what=classes", **{"--classes-sha256": "0" * 64})
    refused(capsys, c, "pin-differs what=endpoints", **{"--endpoints-sha256": "0" * 64})
    refused(capsys, c, "pin-differs what=description-pins", **{"--description-pins-sha256": "0" * 64})
    refused(capsys, c, "pin-malformed what=classes", **{"--classes-sha256": "xyz"})


def test_malformed_tables_and_pins_are_refused_by_code_only(tmp_path, capsys):
    c = Case(tmp_path)
    c.add()
    c.write(c.classes, {"schema": "other"})
    refused(capsys, c, "table-malformed")
    c.write(c.classes, classes_doc(**{"copy-a": "fast"}))
    c.endpoints.write_text("[]")
    refused(capsys, c, "table-malformed")
    c.write(c.endpoints, endpoints_doc(a="copy-a"))
    for text in ("", "copy-a\n", f"copy-a\t{'a1' * 32}\ncopy-a\t{'b2' * 32}\n", f"copy-a\t{'a1' * 32}\ncopy-b\t{'a1' * 32}\n", f"copy a\t{'a1' * 32}\n",
                 f"copy-a\t{'A1' * 32}\n"):
        c.pins.write_text(text)
        refused(capsys, c, "pins-malformed")


def test_rows_must_be_pinned_distinct_pairs_with_clean_absolute_paths(tmp_path, capsys):
    c = Case(tmp_path)
    c.add()
    c.add()
    refused(capsys, c, "rows-malformed")  # the same pair twice
    c.lines = []
    c.add(out="other-out")
    refused(capsys, c, "rows-malformed")
    c.lines = ["synthetic-out\tnatural\tonly-three-fields\n"]
    refused(capsys, c, "rows-malformed")
    c.lines = []
    refused(capsys, c, "rows-malformed")
    for bad in ("relative/manifest.json", "/a/../manifest.json"):
        c.lines = [f"synthetic-out\tnatural\t{bad}\t{tmp_path}/run.json\n"]
        refused(capsys, c, "path-not-clean what=manifest")
    c.lines = [f"synthetic-out\tnatural\t{tmp_path}/manifest.json\trun.json\n"]
    refused(capsys, c, "path-not-clean what=run.json")


def test_a_manifest_or_run_json_that_does_not_say_what_is_needed_is_refused(tmp_path, capsys):
    cases = (({"fingerprint": {"store": {"name": GRAPH}}}, "manifest-store"), ({}, "manifest-store"),
                          ({"fingerprint": {"store": {"name": GRAPH, "endpoint": EP["a"]}}, "store_changes": {}}, "manifest-store-changes"),
                          ({"fingerprint": {"store": {"name": GRAPH, "endpoint": EP["a"]}}, "store_changes": ["x"]}, "manifest-store-changes"),
                          ({"fingerprint": {"store": {"name": GRAPH, "endpoint": EP["a"]}}, "store_changes": [change("a", "a", ["r1"])]},
                           "manifest-store-changes"),  # moves nothing
                          ({"fingerprint": {"store": {"name": GRAPH, "endpoint": EP["a"]}}, "store_changes": [change("a", "b", ["r1"]), change("c", "a", ["r1"])]},
                           "manifest-store-changes"),  # does not start where the last moved to
                          ({"fingerprint": {"store": {"name": GRAPH, "endpoint": EP["a"]}}, "store_changes": [change("a", "b", ["r1", "r2"]), change("b", "c", ["r1"])]},
                           "manifest-store-changes"),  # fewer records before the second change than before the first
                          ({"fingerprint": {"store": {"name": GRAPH, "endpoint": EP["a"]}}, "store_changes": [
                              {**change("a", "b", ["r1"]), "pre_change_records": 2}]}, "manifest-store-changes"),
                          ({"fingerprint": {"store": {"name": GRAPH, "endpoint": EP["a"]}}, "store_changes": [
                              {**change("a", "b", ["r1"]), "from": {"endpoint": EP["a"], "store_client": {"x": 1}}}]}, "store-change-side"),
                          ({"fingerprint": {"store": {"name": GRAPH, "endpoint": EP["a"]}}, "store_changes": [
                              {**change("a", "b", ["r1"]), "to": {"endpoint": EP["b"], "store": {"endpoint": EP["c"]}}}]}, "store-change-side"))
    for i, (manifest, why) in enumerate(cases):
        c = Case(tmp_path / f"manifest-{i}")
        c.add(manifest=manifest)
        refused(capsys, c, why)
    for i, run in enumerate(({}, {"store": {}}, {"store": {"description_sha256": "short"}}, {"store": []})):
        c = Case(tmp_path / f"run-{i}")
        c.add(run=run)
        refused(capsys, c, "run-json-description")


def test_the_replay_copy_is_the_runs_version_when_it_is_a_known_replica_else_the_description_pin(tmp_path, capsys):
    c = Case(tmp_path)
    c.add(version="copy-b", description="e5" * 32)  # the version names a replica; no pin lists the description
    c.add("synthetic-out-p", "prompted", version="copy-a", description=REPLAY_SHA["copy-a"])  # version and pin agree
    c.add("synthetic-out-2", "natural", version="a15-label-nobody-knows", description=REPLAY_SHA["copy-b"])  # an unknown version: the pin
    assert c.run(capsys)[0] == 0
    assert c.table() == [
        HEADER,
        "synthetic-out\tnatural\tcopy-a\t-\tfast\tcopy-b\tversion\tslow\tno\n",
        "synthetic-out-p\tprompted\tcopy-a\t-\tfast\tcopy-a\tversion\tfast\tyes\n",
        "synthetic-out-2\tnatural\tcopy-a\t-\tfast\tcopy-b\tdescription-pin\tslow\tno\n"]


def test_a_version_and_a_pin_that_name_different_copies_are_refused(tmp_path, capsys):
    c = Case(tmp_path)
    c.add(version="copy-a", description=REPLAY_SHA["copy-b"])
    refused(capsys, c, "replay-copy-conflict")


def test_only_version_and_the_description_sha256_of_run_json_are_used(tmp_path, capsys):
    c = Case(tmp_path)
    c.add(run={"store": {"description_sha256": REPLAY_SHA["copy-b"], "host": "copy-a", "gate_side": "copy-a"}, "copy": "copy-a", "version": "copy-a"})
    assert c.run(capsys)[0] == 0
    assert c.table()[1].split("\t")[5:7] == ["copy-b", "description-pin"]


def test_an_existing_table_is_never_overwritten_and_a_link_input_is_refused(tmp_path, capsys):
    c = Case(tmp_path)
    c.add()
    c.out.write_text("keep")
    rc, out, err = c.run(capsys)
    assert (rc, out, err) == (2, "", "REFUSED out-exists\n") and c.out.read_text() == "keep"
    c.out.unlink()
    moved = tmp_path / "moved.json"
    c.classes.rename(moved)
    os.symlink(moved, c.classes)
    rc, out, err = c.run(capsys)
    assert (rc, out, err) == (2, "", "REFUSED unreadable\n")


def test_nothing_private_reaches_the_output_lines(tmp_path, capsys):
    c = Case(tmp_path)
    c.add()
    rc, out, err = c.run(capsys)
    assert out == "ok rows=1\n" and err == ""
    for private in (GRAPH, "copy-a", EP["a"], REPLAY_SHA["copy-a"]):
        assert private not in out + err


def test_the_tool_runs_under_isolated_mode_and_loads_its_neighbours_by_path(tmp_path):
    c = Case(tmp_path)
    c.add()
    wrapper = tmp_path / "wrap.py"
    wrapper.write_text(
        "import importlib.util, sys\n"
        "def load(name, path):\n"
        "    spec = importlib.util.spec_from_file_location(name, path)\n"
        "    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m\n"
        "tool_dir = sys.argv[1].rsplit('/', 1)[0]\n"
        "a = load('a15_replay_matrices', tool_dir + '/a15_replay_matrices.py')\n"
        "a.NATURAL_OUTS = ('synthetic-out',)\n"
        "t = load('a15_class_table', sys.argv[1])\n"
        "sys.exit(t.main(sys.argv[2:]))\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-I", str(wrapper), str(TOOL), *c.argv()], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert (r.returncode, r.stdout, r.stderr) == (0, "ok rows=1\n", "")
    assert c.table()[1] == "synthetic-out\tnatural\tcopy-a\t-\tfast\tcopy-a\tdescription-pin\tfast\tyes\n"


def test_the_real_tool_serves_the_real_pairs_only(tmp_path):
    c = Case(tmp_path)
    c.add()  # synthetic-out is no real pair
    r = subprocess.run([sys.executable, "-I", str(TOOL), *c.argv()], capture_output=True, text=True,
                       env={"PYTHONDONTWRITEBYTECODE": "1"}, cwd=tmp_path)
    assert (r.returncode, r.stdout, r.stderr) == (2, "", "REFUSED rows-malformed\n")
