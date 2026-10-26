"""Tests of driver/labels_only_diff.py on small synthetic results objects (never a real results.json).

Hand-derived facts: the tool prints one line 'identical-outside-labels: true|false' and, for false, the key paths that
differ (a JSON array of keys per line, sorted), never a value; the four table labels in OLD (as keys of the store copies
and the share rows) are read as their replicas; the copy-classes / endpoint-copies file records and the class-labelled
rows of the class shares are not compared (their totals are); everything else, the branch-phase copy objects (labels and
class fields) included, must be equal."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent

_spec = importlib.util.spec_from_file_location("labels_only_diff_under_test", RESULTS / "driver" / "labels_only_diff.py")
lod = importlib.util.module_from_spec(_spec)
sys.modules["labels_only_diff_under_test"] = lod
_spec.loader.exec_module(lod)

L1, R1 = "endpoint sha256:0d4386b1394e", "rhea-c06"  # a table copy: its label in OLD, its replica
L2, R2 = "endpoint sha256:588836988a85", "corporate-c07"
OUT_A, OUT_B, OUT_C = "/runs/rhea-sonnet", "/runs/rhea-qwen", "/runs/rhea-haiku"
ARM = "s01|Qwen|none|prompt|-"
NONE = "no copy class (no entry in the copy classes)"
SECRET = 0.7311234  # a value that must never be printed
SECRET_TEXT = "a-secret-value"


def _share(n_of: dict[str, int], without: int = 0) -> dict:
    total = sum(n_of.values())
    return {"attempts": total, "without record": without,
            "rows": {k: {"n": n, "of": total, "share": n / total} for k, n in sorted(n_of.items())}}


def old_results() -> dict:
    """A look before the table: rhea-sonnet and rhea-haiku on the endpoint label L1, rhea-qwen proven on R1. Written by hand."""
    return {
        "arms": {ARM: {"estimate": SECRET, "text": SECRET_TEXT}},
        "exploratory": {"code": {"commit": "847cc275", "code_digest": "d1"}, "graph": "rhea-142"},
        "stores": {
            "config": {"path": "store-proofs.json", "sha256": "a" * 64, "entries": 3, "matched": 1,
                       "copy_classes": {"path": "copy-classes.json", "sha256": "1" * 64}},
            "copies": {
                R1: {"endpoints": [L1], "outs": [OUT_B], "listed_by": {OUT_B: ["its manifest store change 1 (from)"]},
                     "proofs": [{"replica": R1, "change_at": None}]},
                L1: {"endpoints": [L1], "outs": [OUT_A, OUT_C], "listed_by": {OUT_A: ["its deployment identity's store"],
                                                                              OUT_C: ["its deployment identity's store"]},
                     "proofs": []},
                L2: {"endpoints": [L2], "outs": ["/runs/corporate-sonnet"], "listed_by": {"/runs/corporate-sonnet": []},
                     "proofs": []}},
            "branch_phase_copies": [{"out": OUT_B, "branch_phase_start": "t", "attribution": "x",
                                     "parent_copies": [{"copy": R1, "class": "fast", "label": "fast", "basis": "b",
                                                        "source": "speed-class-v3"}],
                                     "branch_copy": {"copy": R1, "class": "fast", "label": "fast", "basis": "b",
                                                     "source": "speed-class-v3"}}],
            "shares": {ARM: _share({L1: 4, R1: 4})},
            "shares_by_out": {OUT_A: {ARM: _share({L1: 4})}, OUT_B: {ARM: _share({R1: 4})}},
            "class_shares": {ARM: _share({NONE: 4, "fast": 4})},
            "class_shares_by_out": {OUT_A: {ARM: _share({NONE: 4})}, OUT_B: {ARM: _share({"fast": 4})}},
            "attempt_shares": {ARM: _share({L1: 5, R1: 4})},
            "attempt_shares_by_out": {OUT_A: {ARM: _share({L1: 5})}, OUT_B: {ARM: _share({R1: 4})}},
            "store_changes": [{"out": OUT_B, "index": 1, "from": R1, "to": "copy-b", "shares_by_arm": {}}]}}


def new_results() -> dict:
    """The same look after the table: the unproven OUTs on R1 with its class. Written by hand, not by a label swap."""
    return {
        "arms": {ARM: {"estimate": SECRET, "text": SECRET_TEXT}},
        "exploratory": {"code": {"commit": "847cc275", "code_digest": "d1"}, "graph": "rhea-142"},
        "stores": {
            "config": {"path": "store-proofs.json", "sha256": "a" * 64, "entries": 3, "matched": 1,
                       "copy_classes": {"path": "copy-classes.json", "sha256": "2" * 64},
                       "endpoint_copies": {"path": "endpoint-copies.json", "sha256": "3" * 64}},
            "copies": {
                R1: {"endpoints": [L1], "outs": [OUT_C, OUT_B, OUT_A],  # (sorted: haiku, qwen, sonnet)
                     "listed_by": {OUT_A: ["its deployment identity's store"], OUT_B: ["its manifest store change 1 (from)"],
                                   OUT_C: ["its deployment identity's store"]},
                     "proofs": [{"replica": R1, "change_at": None}]},
                R2: {"endpoints": [L2], "outs": ["/runs/corporate-sonnet"], "listed_by": {"/runs/corporate-sonnet": []},
                     "proofs": []}},
            "branch_phase_copies": [{"out": OUT_B, "branch_phase_start": "t", "attribution": "x",
                                     "parent_copies": [{"copy": R1, "class": "fast", "label": "fast", "basis": "b",
                                                        "source": "speed-class-v3"}],
                                     "branch_copy": {"copy": R1, "class": "fast", "label": "fast", "basis": "b",
                                                     "source": "speed-class-v3"}}],
            "shares": {ARM: _share({R1: 8})},
            "shares_by_out": {OUT_A: {ARM: _share({R1: 4})}, OUT_B: {ARM: _share({R1: 4})}},
            "class_shares": {ARM: _share({"fast": 8})},
            "class_shares_by_out": {OUT_A: {ARM: _share({"fast": 4})}, OUT_B: {ARM: _share({"fast": 4})}},
            "attempt_shares": {ARM: _share({R1: 9})},
            "attempt_shares_by_out": {OUT_A: {ARM: _share({R1: 5})}, OUT_B: {ARM: _share({R1: 4})}},
            "store_changes": [{"out": OUT_B, "index": 1, "from": R1, "to": "copy-b", "shares_by_arm": {}}]}}


@pytest.fixture
def table(tmp_path) -> Path:
    doc = {"schema": "e189-endpoint-copies/1", "rule": "a test", "copies": [
        {"graph": "rhea-142", "endpoint_sha256": "0d4386b1394e" + "0" * 52, "replica": R1, "basis": "b",
         "sources": [{"path": "/i.json", "sha256": "ab" * 32}]},
        {"graph": "text2sparql-ck26", "endpoint_sha256": "588836988a85" + "0" * 52, "replica": R2, "basis": "b",
         "sources": [{"path": "/i.json", "sha256": "ab" * 32}]}]}
    path = tmp_path / "endpoint-copies.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def run(tmp_path, table, old, new, capsys):
    o, n = tmp_path / "old.json", tmp_path / "new.json"
    o.write_text(json.dumps(old), encoding="utf-8")
    n.write_text(json.dumps(new), encoding="utf-8")
    code = lod.main([str(o), str(n), "--endpoint-copies", str(table)])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def paths_of(out: str) -> list[list]:
    lines = out.splitlines()
    return [json.loads(line) for line in lines[1:]]


def assert_no_value(*texts: str) -> None:
    for text in texts:
        assert str(SECRET) not in text and SECRET_TEXT not in text


def test_two_looks_that_differ_only_in_labels_and_classes_are_identical_outside_labels(tmp_path, table, capsys):
    code, out, err = run(tmp_path, table, old_results(), new_results(), capsys)
    assert (code, out, err) == (0, "identical-outside-labels: true\n", "")


def test_the_same_file_twice_is_identical(tmp_path, table, capsys):
    code, out, _ = run(tmp_path, table, new_results(), new_results(), capsys)
    assert (code, out) == (0, "identical-outside-labels: true\n")


def _differs(tmp_path, table, capsys, edit, expected, where="new"):
    old, new = old_results(), new_results()
    edit(new if where == "new" else old)
    code, out, err = run(tmp_path, table, old, new, capsys)
    assert code == 0 and err == ""
    assert out.splitlines()[0] == "identical-outside-labels: false", out
    assert paths_of(out) == expected
    assert_no_value(out, err)


@pytest.mark.parametrize("name, edit, expected", [
    ("an arm's number", lambda d: d["arms"][ARM].update(estimate=0.5), [["arms", ARM, "estimate"]]),
    ("an arm's text", lambda d: d["arms"][ARM].update(text="other"), [["arms", ARM, "text"]]),
    ("a new top-level key (a driver's version field)", lambda d: d.update(e189_commit="abc"), [["e189_commit"]]),
    ("the frozen code's identity", lambda d: d["exploratory"]["code"].update(commit="x"),
     [["exploratory", "code", "commit"]]),
    ("a share's n", lambda d: d["stores"]["shares"][ARM]["rows"][R1].update(n=7),
     [["stores", "shares", ARM, "rows", R1, "n"]]),
    ("a share's attempts", lambda d: d["stores"]["shares"][ARM].update(attempts=9),
     [["stores", "shares", ARM, "attempts"]]),
    ("the attempts without record", lambda d: d["stores"]["shares_by_out"][OUT_A][ARM].update(**{"without record": 1}),
     [["stores", "shares_by_out", OUT_A, ARM, "without record"]]),
    ("an attempt share row", lambda d: d["stores"]["attempt_shares_by_out"][OUT_B][ARM]["rows"][R1].update(n=3),
     [["stores", "attempt_shares_by_out", OUT_B, ARM, "rows", R1, "n"]]),
    ("the attempts of a class share", lambda d: d["stores"]["class_shares"][ARM].update(attempts=9),
     [["stores", "class_shares", ARM, "attempts"]]),
    ("the rows total of a class share", lambda d: d["stores"]["class_shares_by_out"][OUT_B][ARM]["rows"]["fast"].update(n=3),
     [["stores", "class_shares_by_out", OUT_B, ARM, "rows_of"], ["stores", "class_shares_by_out", OUT_B, ARM, "rows_total_n"],
      ["stores", "class_shares_by_out", OUT_B, ARM, "rows_well_formed"]]),
    ("a store change row", lambda d: d["stores"]["store_changes"][0].update(to="copy-c"),
     [["stores", "store_changes", "[]", "to"]]),
    ("a store change added", lambda d: d["stores"]["store_changes"].append({}), [["stores", "store_changes"]]),
    ("the proofs of a copy", lambda d: d["stores"]["copies"][R1]["proofs"][0].update(change_at="t"),
     [["stores", "copies", R1, "proofs", "[]", "change_at"]]),
    ("a copy's OUTs", lambda d: d["stores"]["copies"][R1].update(outs=[OUT_A, OUT_B]),
     [["stores", "copies", R1, "outs"]]),
    ("a copy's endpoints given as the replica", lambda d: d["stores"]["copies"][R1].update(endpoints=[R1]),
     [["stores", "copies", R1, "endpoints", "[]"]]),
    ("the store proofs' record in the config", lambda d: d["stores"]["config"].update(sha256="b" * 64),
     [["stores", "config", "sha256"]]),
    ("the class of a proven copy in the branch phase",
     lambda d: d["stores"]["branch_phase_copies"][0]["branch_copy"].update(**{"class": "slow"}),
     [["stores", "branch_phase_copies", "[]", "branch_copy", "class"]]),
    ("the basis of a parent copy in the branch phase",
     lambda d: d["stores"]["branch_phase_copies"][0]["parent_copies"][0].update(basis="c"),
     [["stores", "branch_phase_copies", "[]", "parent_copies", "[]", "basis"]]),
    ("the denominator of a class row", lambda d: d["stores"]["class_shares"][ARM]["rows"]["fast"].update(of=9),
     [["stores", "class_shares", ARM, "rows_of"], ["stores", "class_shares", ARM, "rows_well_formed"],
      ["stores", "class_shares", ARM, "rows_total_n"]]),
    ("a class row whose share is not n over of", lambda d: d["stores"]["class_shares"][ARM]["rows"]["fast"].update(share=0.4),
     [["stores", "class_shares", ARM, "rows_of"], ["stores", "class_shares", ARM, "rows_well_formed"],
      ["stores", "class_shares", ARM, "rows_total_n"]]),
    ("a copy of the branch phase renamed", lambda d: d["stores"]["branch_phase_copies"][0]["branch_copy"].update(copy="copy-x"),
     [["stores", "branch_phase_copies", "[]", "branch_copy", "copy"]]),
    ("a branch-phase attribution sentence", lambda d: d["stores"]["branch_phase_copies"][0].update(attribution="y"),
     [["stores", "branch_phase_copies", "[]", "attribution"]]),
    ("a table copy left as its endpoint label (a table miss)",
     lambda d: d["stores"]["shares_by_out"][OUT_A][ARM]["rows"].update({L1: d["stores"]["shares_by_out"][OUT_A][ARM]["rows"].pop(R1)}),
     [["stores", "shares_by_out", OUT_A, ARM, "rows", L1], ["stores", "shares_by_out", OUT_A, ARM, "rows", R1]]),
])
def test_anything_but_a_label_is_reported_by_its_key_path_and_never_by_a_value(tmp_path, table, capsys, name, edit, expected):
    _differs(tmp_path, table, capsys, edit, sorted(expected, key=json.dumps))


def test_the_excluded_fields_may_differ(tmp_path, table, capsys):
    new = new_results()
    new["stores"]["config"]["copy_classes"] = {"path": "elsewhere.json", "sha256": "9" * 64}
    new["stores"]["config"]["endpoint_copies"]["sha256"] = "8" * 64
    new["stores"]["config"]["private_copy_classes"] = {"name": "private-classes.json", "sha256": "7" * 64}
    new["stores"]["config"]["private_endpoint_copies"] = {"name": "private-endpoints.json", "sha256": "6" * 64}
    new["stores"]["class_shares"][ARM]["rows"] = {"medium": {"n": 2, "of": 8, "share": 0.25}, "slow": {"n": 6, "of": 8, "share": 0.75}}
    code, out, _ = run(tmp_path, table, old_results(), new, capsys)
    assert (code, out) == (0, "identical-outside-labels: true\n" + lod.PRIVATE_NOTE + "\n")  # (private records: see the note)


def test_inputs_that_record_a_private_table_get_one_closing_note_and_others_get_none(tmp_path, table, capsys):
    note = lod.PRIVATE_NOTE
    assert note.startswith("private inputs:")
    for name in ("private_copy_classes", "private_endpoint_copies"):
        for side in ("old", "new"):
            old, new = old_results(), new_results()
            {"old": old, "new": new}[side]["stores"]["config"][name] = {"name": "t.json", "sha256": "5" * 64}
            code, out, _ = run(tmp_path, table, old, new, capsys)
            assert code == 0 and out.splitlines()[-1] == note and out.count(note) == 1, (name, side)
    # a differing run keeps its key paths, then the note last
    old, new = old_results(), new_results()
    new["stores"]["config"]["private_copy_classes"] = {"name": "t.json", "sha256": "5" * 64}
    new["stores"]["shares"][ARM]["rows"]["extra"] = {"n": 1, "of": 8, "share": 0.125}
    code, out, _ = run(tmp_path, table, old, new, capsys)
    lines = out.splitlines()
    assert code == 0 and lines[0] == "identical-outside-labels: false" and lines[-1] == note and len(lines) >= 3
    # no private record, no note
    code, out, _ = run(tmp_path, table, old_results(), new_results(), capsys)
    assert note not in out


def test_a_label_inside_a_longer_string_or_among_the_endpoints_is_not_mapped(tmp_path, table, capsys):
    old, new = old_results(), new_results()
    old["stores"]["shares"][ARM]["rows"] = {f"{L1} -> x": {"n": 8, "of": 8, "share": 1.0}}
    new["stores"]["shares"][ARM]["rows"] = {f"{R1} -> x": {"n": 8, "of": 8, "share": 1.0}}
    code, out, _ = run(tmp_path, table, old, new, capsys)
    assert code == 0 and out.splitlines()[0] == "identical-outside-labels: false"
    assert ["stores", "shares", ARM, "rows", f"{L1} -> x"] in paths_of(out)


def test_an_endpoint_label_the_table_does_not_name_is_never_mapped(tmp_path, table, capsys):
    old, new = old_results(), new_results()
    other = "endpoint sha256:ffffffffffff"
    old["stores"]["shares_by_out"][OUT_C] = {ARM: _share({other: 2})}
    new["stores"]["shares_by_out"][OUT_C] = {ARM: _share({other: 2})}
    assert run(tmp_path, table, old, new, capsys)[1] == "identical-outside-labels: true\n"
    new["stores"]["shares_by_out"][OUT_C] = {ARM: _share({R1: 2})}  # NEW named it, the table does not: a difference
    code, out, _ = run(tmp_path, table, old, new, capsys)
    assert out.splitlines()[0] == "identical-outside-labels: false"


def test_a_branch_phase_copy_given_as_a_table_label_in_old_is_a_difference_not_a_mapping(tmp_path, table, capsys):
    """external reviewer r2: an OUT with a branch phase has proof-named copies, so a table label there is never a legitimate change."""
    old = old_results()
    old["stores"]["branch_phase_copies"][0]["branch_copy"]["copy"] = L1  # OLD names the copy by the endpoint label
    code, out, err = run(tmp_path, table, old, new_results(), capsys)
    assert code == 0 and out.splitlines()[0] == "identical-outside-labels: false"
    assert paths_of(out) == [["stores", "branch_phase_copies", "[]", "branch_copy", "copy"]]
    assert_no_value(out, err)


def test_impossible_row_counts_are_not_merged_into_a_plausible_row(tmp_path, table, capsys):
    """external reviewer r2: n=-1 of 4 (share -0.25) and n=5 of 4 (share 1.25) merge to the 4 of 4 row NEW has; share_of cannot write either."""
    for bad_first, bad_second in (({"n": -1, "of": 4, "share": -0.25}, {"n": 5, "of": 4, "share": 1.25}),
                                  ({"n": 5, "of": 4, "share": 1.25}, {"n": -1, "of": 4, "share": -0.25})):
        old = old_results()
        old["stores"]["shares"][ARM]["rows"] = {L1: bad_first, R1: bad_second}
        code, out, err = run(tmp_path, table, old, new_results(), capsys)
        assert code == 0 and out.splitlines()[0] == "identical-outside-labels: false"
        assert ["stores", "shares", ARM, "rows", R1] in paths_of(out)
        assert_no_value(out, err)


def test_a_zero_count_row_is_not_merged_into_a_plausible_row(tmp_path, table, capsys):
    """external reviewer r3: share_of writes a row only for a label with an attempt, so an n=0 component is no row it could write."""
    old = old_results()
    old["stores"]["shares"][ARM]["rows"] = {L1: {"n": 0, "of": 8, "share": 0.0}, R1: {"n": 8, "of": 8, "share": 1.0}}
    code, out, err = run(tmp_path, table, old, new_results(), capsys)
    assert code == 0 and out.splitlines()[0] == "identical-outside-labels: false"
    assert ["stores", "shares", ARM, "rows", R1] in paths_of(out)
    assert_no_value(out, err)


def test_a_component_row_that_is_not_n_over_of_is_never_merged_away(tmp_path, table, capsys):
    """external reviewer r1: a collision recomputed the share from the summed n, so a changed OLD component share vanished."""
    for edit in (lambda r: r.update(share=0.4), lambda r: r.update(n=True), lambda r: r.update(n=4.0),
                 lambda r: r.update(extra=1), lambda r: r.pop("share")):
        old = old_results()
        edit(old["stores"]["shares"][ARM]["rows"][L1])  # one of the two components of the merged R1 row
        code, out, err = run(tmp_path, table, old, new_results(), capsys)
        assert code == 0 and out.splitlines()[0] == "identical-outside-labels: false", out
        assert ["stores", "shares", ARM, "rows", R1] in paths_of(out)
        assert_no_value(out, err)


def test_rows_of_one_label_merge_by_sum_with_the_same_total_and_not_otherwise(tmp_path, table, capsys):
    old, new = old_results(), new_results()  # OLD has rows L1: 4 and R1: 4 of 8, NEW R1: 8 of 8: the merge is identical
    assert run(tmp_path, table, old, new, capsys)[1] == "identical-outside-labels: true\n"
    old["stores"]["shares"][ARM]["rows"][L1]["of"] = 7  # rows of one share that disagree about their total
    code, out, _ = run(tmp_path, table, old, new, capsys)
    assert out.splitlines()[0] == "identical-outside-labels: false"
    assert ["stores", "shares", ARM, "rows", R1] in paths_of(out)


def test_copies_merge_by_the_union_of_their_outs_and_endpoints_and_a_shared_out_or_a_proof_is_a_difference(tmp_path, table, capsys):
    for edit, field in ((lambda d: d["stores"]["copies"][L1]["listed_by"].update({OUT_B: []}), "listed_by"),
                        (lambda d: d["stores"]["copies"][L1].update(proofs=[{"replica": R1}]), None),
                        (lambda d: d["stores"]["copies"][L1].update(periods=[1]), None)):
        old = old_results()
        edit(old)
        code, out, _ = run(tmp_path, table, old, new_results(), capsys)
        assert code == 0 and out.splitlines()[0] == "identical-outside-labels: false"
        assert ["stores", "copies", R1] in paths_of(out) or ["stores", "copies", R1, field] in paths_of(out)


def test_a_table_named_copy_with_no_proven_twin_is_renamed_in_place(tmp_path, table, capsys):
    old, new = old_results(), new_results()
    assert R2 not in old["stores"]["copies"] and L2 in old["stores"]["copies"] and R2 in new["stores"]["copies"]
    assert run(tmp_path, table, old, new, capsys)[1] == "identical-outside-labels: true\n"


def test_lists_differ_at_the_list_and_their_elements_are_the_key_brackets(tmp_path, table, capsys):
    _differs(tmp_path, table, capsys, lambda d: d["stores"]["copies"][R2]["listed_by"].update({"x": []}),
             [["stores", "copies", R2, "listed_by", "x"]])
    _differs(tmp_path, table, capsys, lambda d: d["stores"]["copies"][R2].update(outs=["/runs/corporate-sonnet", "/runs/x"]),
             [["stores", "copies", R2, "outs"]])


def test_bad_input_exits_2_and_prints_no_value(tmp_path, table, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(old_results()), encoding="utf-8")
    nostores = tmp_path / "nostores.json"
    nostores.write_text(json.dumps({"arms": {"a": SECRET_TEXT}}), encoding="utf-8")
    notjson = tmp_path / "notjson.json"
    notjson.write_text("not json " + SECRET_TEXT, encoding="utf-8")
    listed = tmp_path / "list.json"
    listed.write_text("[1]", encoding="utf-8")
    bad_table = tmp_path / "bad-table.json"
    bad_table.write_text(json.dumps({"schema": "x"}), encoding="utf-8")
    twice = json.loads(table.read_text(encoding="utf-8"))
    twice["copies"][1]["endpoint_sha256"] = "0d4386b1394e" + "1" * 52
    twice["copies"][1]["graph"] = "another"
    shared = tmp_path / "shared-prefix.json"
    shared.write_text(json.dumps(twice), encoding="utf-8")
    for argv in ([str(good), "--no-such-option", SECRET_TEXT], [str(good), str(good), SECRET_TEXT],
                 [str(good), str(nostores)], [str(nostores), str(good)], [str(good), str(notjson)],
                 [str(good), str(listed)], [str(good), str(tmp_path / "missing.json")],
                 [str(good), str(good), "--endpoint-copies", str(bad_table)],
                 [str(good), str(good), "--endpoint-copies", str(shared)],
                 [str(good), str(good), "--endpoint-copies", str(tmp_path / "missing.json")], [str(good)], []):
        argv = argv if "--endpoint-copies" in argv else [*argv, "--endpoint-copies", str(table)]
        assert lod.main(argv) == 2
        captured = capsys.readouterr()
        assert captured.out == "" and SECRET_TEXT not in captured.err and str(SECRET) not in captured.err


def test_the_command_runs_under_isolated_mode_with_the_committed_table_by_default(tmp_path):
    """python3 -I labels_only_diff.py OLD NEW: no flag, so the table beside the exploratory script, whose rhea-c06 label is
    'endpoint sha256:0d4386b1394e'."""
    old, new = old_results(), new_results()
    (tmp_path / "old.json").write_text(json.dumps(old), encoding="utf-8")
    (tmp_path / "new.json").write_text(json.dumps(new), encoding="utf-8")
    done = subprocess.run([sys.executable, "-I", str(RESULTS / "driver" / "labels_only_diff.py"), str(tmp_path / "old.json"),
                           str(tmp_path / "new.json")], capture_output=True, text=True, check=False)
    assert (done.returncode, done.stdout, done.stderr) == (0, "identical-outside-labels: true\n", "")
    new["stores"]["shares"][ARM]["attempts"] = 9
    (tmp_path / "new.json").write_text(json.dumps(new), encoding="utf-8")
    done = subprocess.run([sys.executable, "-I", str(RESULTS / "driver" / "labels_only_diff.py"), str(tmp_path / "old.json"),
                           str(tmp_path / "new.json")], capture_output=True, text=True, check=False)
    assert done.returncode == 0 and done.stdout.splitlines()[0] == "identical-outside-labels: false"
    assert json.loads(done.stdout.splitlines()[1]) == ["stores", "shares", ARM, "attempts"]
    assert str(SECRET) not in done.stdout + done.stderr
    assert subprocess.run([sys.executable, "-I", str(RESULTS / "driver" / "labels_only_diff.py"), "x"], capture_output=True,
                          check=False).returncode == 2


def test_the_inputs_are_not_changed(tmp_path, table, capsys):
    old, new = old_results(), new_results()
    before = copy.deepcopy((old, new))
    lod.compare(old, new, lod.label_mapping(table))
    assert (old, new) == before
