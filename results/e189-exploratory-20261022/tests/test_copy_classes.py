"""Tests of copy_classes.py (science's ruling of 25 Oct ~05:40-05:50Z): each precedence step of a copy's speed class on
small fixture sources, the build's determinism, the loader's strict validation, and the committed copy-classes.json
against the copies it covers and the outcomes the lead worked out by hand from the real sources.

Hand-derived facts: the first source that gives a class wins, in the order v3 FINAL speed-class.json, the per-copy
records, cell-classes.tsv, the host rule; "partial" is no class; a note with the word unmeasured is the table's host
rule, so its label says so; the host rule (an example table over example hosts) gives rhea-142 example-host-1 fast, example-host-2
medium, opaque and UniProt example-host-1 fast, example-host-2 slow, and corporate (text2sparql-ck26) none."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent

_spec = importlib.util.spec_from_file_location("e189_copy_classes_under_test", RESULTS / "copy_classes.py")
cc = importlib.util.module_from_spec(_spec)
sys.modules["e189_copy_classes_under_test"] = cc
_spec.loader.exec_module(cc)

SHA = "ab" * 32


def _write(path: Path, doc) -> Path:
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc, indent=1), encoding="utf-8")
    return path


def _hosts_entry(graph: str, host: str, basis: str = "a registry says so") -> dict:
    return {"graph": graph, "host": host, "basis": basis, "source": {"path": "/registry.json", "sha256": SHA}}


def _hosts_doc(**copies) -> dict:
    return {"schema": "e189-copy-hosts/1", "rule": "a test", "copies": copies}


def _endpoint_entry(replica: str, graph: str = "opaque-rhea-142", endpoint: str = "c" * 64, **more) -> dict:
    return {"graph": graph, "endpoint_sha256": endpoint, "replica": replica, "basis": "a test basis",
            "sources": [{"path": "/identity.json", "sha256": SHA}], **more}


def _endpoint_doc(*entries) -> dict:
    return {"schema": "e189-endpoint-copies/1", "rule": "a test", "copies": list(entries)}


def _extra_entry(replica: str, graph: str = "rhea-142", **more) -> dict:
    return {"replica": replica, "graph": graph, "basis": "a test basis", "sources": [{"path": "/registry.json", "sha256": SHA}],
            **more}


def _extra_doc(*entries) -> dict:
    return {"schema": "e189-extra-copies/1", "rule": "a test", "copies": list(entries)}


def _lines(*pairs) -> list:
    return [{"class": c, "note": n} for c, n in pairs]


def _info_entry(replica: str, table_rows=None, **more) -> dict:
    """An informational-row override; the default lines are the one the build tests give r15 in the table."""
    rows = table_rows if table_rows is not None else _lines(("medium", "informational: same class as a sibling"))
    return {"replica": replica, "table_rows": rows, "why": "a test why", "sources": [{"path": "/table.tsv", "sha256": SHA}], **more}


def _basis_entry(replica: str, basis: str = "a test basis text", table_rows=None, **more) -> dict:
    """A basis override; the default lines are r3's line in the fixture table."""
    rows = table_rows if table_rows is not None else _lines(("fast", "Host1, measured fast"))
    return {"replica": replica, "table_rows": rows, "basis": basis, "why": "a test why",
            "sources": [{"path": "/summary.txt", "sha256": SHA}], **more}


def _measured_entry(replica: str, cls: str = "medium", basis: str = "a test measured basis", **more) -> dict:
    return {"replica": replica, "class": cls, "basis": basis, "why": "a test why",
            "sources": [{"path": "/summary.txt", "sha256": SHA}], **more}


def _overrides_doc(info=(), bases=(), measured=()) -> dict:
    return {"schema": "e189-class-overrides/2", "rule": "a test", "informational_rows": list(info), "bases": list(bases),
            "measured": list(measured)}


# --- the precedence, step by step (the pure function on parsed sources) ------------------------------------------------


def test_v3_beats_the_records_and_the_table():
    v3 = {"c": ("fast", "v3 basis")}
    records = [("record:r.json", "c", "medium", "record basis")]
    cells = {"c": ("slow", "table note")}
    assert cc.classify("c", "rhea-142", v3, records, cells, {}) == {
        "class": "fast", "label": "fast", "basis": "v3 basis", "source": "speed-class-v3"}


def test_a_record_fills_a_gap_in_v3_and_beats_the_table():
    records = [("record:r.json", "c", "medium", "record basis, verbatim")]
    cells = {"c": ("slow", "table note")}
    assert cc.classify("c", "rhea-142", {}, records, cells, {}) == {
        "class": "medium", "label": "medium", "basis": "record basis, verbatim", "source": "record:r.json"}


def test_a_record_of_another_copy_is_no_source_of_this_one():
    records = [("record:r.json", "other", "medium", "record basis")]
    assert cc.classify("c", "rhea-142", {}, records, {"c": ("slow", "n")}, {})["source"] == "cell-classes"


def test_v3_partial_is_no_class_and_falls_through_to_the_table():
    v3 = cc.parse_v3({"copies": {"c": {"speed_class": "partial", "basis": "earlier"},
                                  "d": {"speed_class": "slow", "basis": "fresh"}}})
    assert v3 == {"d": ("slow", "fresh")}  # c has no class in v3
    cells = {"c": ("fast", "Host1, measured fast")}
    assert cc.classify("c", "rhea-142", v3, [], cells, {}) == {
        "class": "fast", "label": "fast", "basis": "Host1, measured fast", "source": "cell-classes"}


@pytest.mark.parametrize("value", [None, "", "unknown", "FAST", 3, True, ["fast"]])
def test_only_fast_medium_or_slow_is_a_class(value):
    assert cc.parse_v3({"copies": {"c": {"speed_class": value, "basis": "b"}}}) == {}
    assert cc.parse_record("r.json", {"copy": "c", "speed_class": value, "basis": "b"}) is None


def test_a_record_and_v3_that_give_different_classes_refuse_and_equal_ones_do_not():
    v3 = {"c": ("fast", "v3 basis")}
    with pytest.raises(cc.Refused, match=r"c: the measurements disagree"):
        cc.check_measurements(v3, [("record:r.json", "c", "slow", "b")])
    cc.check_measurements(v3, [("record:r.json", "c", "fast", "b")])  # the same class: v3's basis stands
    assert cc.classify("c", "rhea-142", v3, [("record:r.json", "c", "fast", "record basis")], {}, {})["basis"] == "v3 basis"
    with pytest.raises(cc.Refused, match=r"c: the measurements disagree"):  # two records
        cc.check_measurements({}, [("record:a.json", "c", "fast", "b"), ("record:b.json", "c", "slow", "b")])
    cc.check_measurements({"c": ("fast", "b")}, [("record:r.json", "d", "slow", "b")])  # another copy: no disagreement


def test_the_table_is_read_by_columns_and_a_copy_on_several_lines_has_one_class_and_its_first_note():
    text = ("# endpoint\tcopy id\tspeed class\tnote\n"
            "\n"
            "a:1\tc\tslow\tfirst note\n"
            "b:2\tc\tslow\tsecond note\n"
            "# a comment line\n"
            "x:3\td\tfast\tHost1, measured fast\n")
    assert cc.parse_cell_classes(text) == {"c": ("slow", "first note"), "d": ("fast", "Host1, measured fast")}
    with pytest.raises(cc.Refused, match=r"c is on several lines with different classes"):
        cc.parse_cell_classes("a:1\tc\tslow\tn\nb:2\tc\tfast\tn\n")
    with pytest.raises(cc.Refused, match=r"not one of fast, medium, slow"):
        cc.parse_cell_classes("a:1\tc\tpartial\tn\n")
    for bad in ("a:1\tc\tslow\n", "a:1\tc\tslow\tn\textra\n", "a:1\tc\t\tn\n", "a:1\tc slow n\n"):
        with pytest.raises(cc.Refused, match=r"not four filled tab-separated columns"):
            cc.parse_cell_classes(bad)


@pytest.mark.parametrize("note, label", [
    ("Host2, unmeasured = slow", "slow (host rule, unmeasured)"),
    ("Host1, unmeasured, dead since 23 Oct", "slow (host rule, unmeasured)"),
    ("Host2, measured slow", "slow"),
    ("Host3, represented by 9111", "slow"),
    ("unmeasuredish is not the word", "slow"),
])
def test_a_note_with_the_word_unmeasured_is_the_tables_host_rule_and_its_label_says_so(note, label):
    entry = cc.classify("c", "rhea-142", {}, [], {"c": ("slow", note)}, {})
    assert entry == {"class": "slow", "label": label, "basis": note, "source": "cell-classes"}


@pytest.mark.parametrize("graph, host, speed", [
    ("rhea-142", "example-host-1", "fast"), ("rhea-142", "example-host-2", "medium"),
    ("opaque-rhea-142", "example-host-1", "fast"), ("opaque-rhea-142", "example-host-2", "slow"),
    ("uniprot-sp-2026_03", "example-host-1", "fast"), ("uniprot-sp-2026_03", "example-host-2", "slow"),
])
def test_the_host_rule_by_graph_and_host(graph, host, speed):
    hosts = cc.parse_hosts(_hosts_doc(c=_hosts_entry(graph, host, "the entry's own basis")))
    assert cc.classify("c", graph, {}, [], {}, hosts) == {
        "class": speed, "label": f"{speed} (host rule, unmeasured)", "basis": "the entry's own basis",
        "source": "copy-hosts"}


def test_rhea_on_the_second_host_is_medium_where_opaque_on_it_is_slow():
    a = cc.parse_hosts(_hosts_doc(c=_hosts_entry("rhea-142", "example-host-2")))
    b = cc.parse_hosts(_hosts_doc(c=_hosts_entry("opaque-rhea-142", "example-host-2")))
    assert cc.classify("c", "rhea-142", {}, [], {}, a)["class"] == "medium"
    assert cc.classify("c", "opaque-rhea-142", {}, [], {}, b)["class"] == "slow"


@pytest.mark.parametrize("host", cc.HOSTS)
def test_corporate_has_no_host_rule_its_class_is_null_whatever_the_host(host):
    hosts = cc.parse_hosts(_hosts_doc(c=_hosts_entry("text2sparql-ck26", host, "the registry's words")))
    assert cc.classify("c", "text2sparql-ck26", {}, [], {}, hosts) == {
        "class": None, "label": "unmeasured (no host rule for corporate)", "basis": "the registry's words",
        "source": "copy-hosts"}


def test_an_unknown_graph_or_host_and_a_copy_found_nowhere_refuse():
    with pytest.raises(cc.Refused, match=r"no host rule for its graph"):
        cc.parse_hosts(_hosts_doc(c=_hosts_entry("some-other-graph", "example-host-1")))
    with pytest.raises(cc.Refused, match=r"its host is not one of"):
        cc.parse_hosts(_hosts_doc(c=_hosts_entry("rhea-142", "mars")))
    with pytest.raises(cc.Refused, match=r"c: found in no class source and not in the hosts file"):
        cc.classify("c", "rhea-142", {"d": ("fast", "b")}, [("r", "d", "fast", "b")], {"d": ("fast", "n")},
                    cc.parse_hosts(_hosts_doc(d=_hosts_entry("rhea-142", "example-host-1"))))
    hosts = cc.parse_hosts(_hosts_doc(c=_hosts_entry("rhea-142", "example-host-1")))
    with pytest.raises(cc.Refused, match=r"names another graph than its store proof does"):
        cc.classify("c", "opaque-rhea-142", {}, [], {}, hosts)


@pytest.mark.parametrize("edit", [
    lambda d: d.pop("rule"), lambda d: d.update(extra=1), lambda d: d.update(schema="e189-copy-hosts/2"),
    lambda d: d.update(copies=[]), lambda d: d["copies"]["c"].pop("basis"), lambda d: d["copies"]["c"].update(more=1),
    lambda d: d["copies"]["c"].update(basis=""), lambda d: d["copies"]["c"].pop("source"),
    lambda d: d["copies"]["c"]["source"].update(sha256="xyz"), lambda d: d["copies"]["c"]["source"].pop("path"),
    lambda d: d["copies"].update(c=None),
])
def test_a_malformed_hosts_file_refuses(edit):
    doc = _hosts_doc(c=_hosts_entry("rhea-142", "example-host-1"))
    edit(doc)
    with pytest.raises(cc.Refused):
        cc.parse_hosts(doc)


def test_the_store_proofs_give_the_copies_and_a_replica_of_two_graphs_refuses():
    proofs = {"schema": "e189-store-proofs/1", "proofs": [{"replica": "a", "graph": "rhea-142"},
                                                         {"replica": "a", "graph": "rhea-142"},
                                                         {"replica": "b", "graph": "opaque-rhea-142"}]}
    assert cc.proof_replicas(proofs) == {"a": "rhea-142", "b": "opaque-rhea-142"}
    with pytest.raises(cc.Refused, match=r"a is proven for two graphs"):
        cc.proof_replicas({**proofs, "proofs": [*proofs["proofs"], {"replica": "a", "graph": "opaque-rhea-142"}]})
    for bad in ({"schema": "other", "proofs": []}, {"schema": "e189-store-proofs/1"}, {"proofs": []},
                {"schema": "e189-store-proofs/1", "proofs": [{"replica": "a"}]}):
        with pytest.raises(cc.Refused):
            cc.proof_replicas(bad)


# --- the build on fixture files ------------------------------------------------------------------------------------


def _sources(tmp: Path) -> dict:
    """Nine copies, one per precedence case, as files (r9 only in the endpoint copies, with a table row; r12 only in the
    extra copies, with a table row): r1 in v3 (and the table, as slow: v3 wins), r2 only in a record,
    r3 v3 partial with a table row, r4 an unmeasured table row, r5 a copy of the host rule (example-host-2, Rhea: medium), r6 a
    corporate copy (no host rule), r7 an opaque copy on example-host-1."""
    proofs = [{"replica": r, "graph": g} for r, g in (
        ("r1", "rhea-142"), ("r2", "rhea-142"), ("r3", "rhea-142"), ("r4", "rhea-142"), ("r5", "rhea-142"),
        ("r6", "text2sparql-ck26"), ("r7", "opaque-rhea-142"))]
    v3 = {"copies": {"r1": {"speed_class": "fast", "basis": "fresh"}, "r3": {"speed_class": "partial", "basis": "earlier"}}}
    record = {"copy": "r2", "speed_class": "medium", "basis": "r2 itself was timed"}
    tsv = ("# comment\nh:1\tr1\tslow\tHost2\nh:2\tr3\tfast\tHost1, measured fast\nh:3\tr4\tslow\tHost2, unmeasured = slow\n"
           "h:9\tr9\tmedium\tHost1, timed by hand\n" "h:12\tr12\tmedium\tHost2, timed on siblings\n")
    hosts = _hosts_doc(r5=_hosts_entry("rhea-142", "example-host-2", "b5"), r6=_hosts_entry("text2sparql-ck26", "example-host-1", "b6"),
                       r7=_hosts_entry("opaque-rhea-142", "example-host-1", "b7"))
    return {"store_proofs": _write(tmp / "store-proofs.json", {"schema": "e189-store-proofs/1", "proofs": proofs}),
            "v3": _write(tmp / "speed-class.json", v3), "records": [_write(tmp / "speed-class-r2.json", record)],
            "cell_classes": _write(tmp / "cell-classes.tsv", tsv), "hosts": _write(tmp / "copy-hosts.json", hosts),
            "endpoint_copies": _write(tmp / "endpoint-copies.json", _endpoint_doc(_endpoint_entry("r9"))),
            "extra_copies": _write(tmp / "extra-copies.json", _extra_doc(_extra_entry("r12"))),
            "class_overrides": _write(tmp / "class-overrides.json", _overrides_doc())}


EXPECTED = {
    "r1": {"class": "fast", "label": "fast", "basis": "fresh", "source": "speed-class-v3"},
    "r2": {"class": "medium", "label": "medium", "basis": "r2 itself was timed", "source": "record:speed-class-r2.json"},
    "r3": {"class": "fast", "label": "fast", "basis": "Host1, measured fast", "source": "cell-classes"},
    "r4": {"class": "slow", "label": "slow (host rule, unmeasured)", "basis": "Host2, unmeasured = slow",
           "source": "cell-classes"},
    "r5": {"class": "medium", "label": "medium (host rule, unmeasured)", "basis": "b5", "source": "copy-hosts"},
    "r6": {"class": None, "label": "unmeasured (no host rule for corporate)", "basis": "b6", "source": "copy-hosts"},
    "r7": {"class": "fast", "label": "fast (host rule, unmeasured)", "basis": "b7", "source": "copy-hosts"},
    "r9": {"class": "medium", "label": "medium", "basis": "Host1, timed by hand", "source": "cell-classes"},
    "r12": {"class": "medium", "label": "medium", "basis": "Host2, timed on siblings", "source": "cell-classes"},
}


def test_the_build_gives_each_copy_its_first_source_and_names_every_source_by_path_and_sha256(tmp_path):
    s = _sources(tmp_path)
    doc = cc.build(**s)
    assert doc["copies"] == EXPECTED
    assert doc["schema"] == "e189-copy-classes/1"
    # the rule says, in words, what the brief requires of it (written by hand, not read back from the constant)
    for words in ("example-host-2 medium", "example-host-2 slow", "unmeasured (no host rule for corporate)",
                  "a copy found nowhere", "a partial measurement is none", "extra-copies.json",
                  "a copy in several of the three has one graph"):
        assert words in doc["rule"], words
    assert set(doc["sources"]) == {"store-proofs", "speed-class-v3", "cell-classes", "copy-hosts", "endpoint-copies",
                                   "extra-copies", "class-overrides", "record:speed-class-r2.json"}
    for name, path in (("store-proofs", s["store_proofs"]), ("speed-class-v3", s["v3"]), ("cell-classes", s["cell_classes"]),
                       ("copy-hosts", s["hosts"]), ("endpoint-copies", s["endpoint_copies"]),
                       ("extra-copies", s["extra_copies"]), ("class-overrides", s["class_overrides"]),
                       ("record:speed-class-r2.json", s["records"][0])):
        assert doc["sources"][name] == {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert cc.validate(doc) is doc


def test_the_build_is_deterministic_and_does_not_depend_on_the_order_of_the_records(tmp_path):
    s = _sources(tmp_path)
    extra = _write(tmp_path / "speed-class-r8.json", {"copy": "r8", "speed_class": "slow", "basis": "r8 timed"})
    first = cc.dumps(cc.build(s["store_proofs"], s["v3"], [*s["records"], extra], s["cell_classes"], s["hosts"], s["endpoint_copies"], s["extra_copies"], s["class_overrides"]))
    again = cc.dumps(cc.build(s["store_proofs"], s["v3"], [*s["records"], extra], s["cell_classes"], s["hosts"], s["endpoint_copies"], s["extra_copies"], s["class_overrides"]))
    swapped = cc.dumps(cc.build(s["store_proofs"], s["v3"], [extra, *s["records"]], s["cell_classes"], s["hosts"], s["endpoint_copies"], s["extra_copies"], s["class_overrides"]))
    assert first == again == swapped
    assert first.endswith("}\n") and json.loads(first)["copies"].keys() == EXPECTED.keys()
    assert list(json.loads(first)) == sorted(json.loads(first))  # sorted keys


def test_a_conflict_a_missing_copy_and_a_malformed_source_refuse_the_build(tmp_path):
    s = _sources(tmp_path)
    conflict = _write(tmp_path / "speed-class-r1.json", {"copy": "r1", "speed_class": "slow", "basis": "b"})
    with pytest.raises(cc.Refused, match=r"r1: the measurements disagree"):
        cc.build(s["store_proofs"], s["v3"], [*s["records"], conflict], s["cell_classes"], s["hosts"], s["endpoint_copies"], s["extra_copies"], s["class_overrides"])
    nowhere = _write(tmp_path / "proofs2.json", {"schema": "e189-store-proofs/1", "proofs": [
        {"replica": "zz", "graph": "rhea-142"}]})
    with pytest.raises(cc.Refused, match=r"zz: found in no class source"):
        cc.build(nowhere, s["v3"], s["records"], s["cell_classes"], s["hosts"], s["endpoint_copies"], s["extra_copies"], s["class_overrides"])
    twice = _write(tmp_path / "tsv2.tsv", "h:1\tr1\tslow\tn\nh:2\tr1\tfast\tn\n")
    with pytest.raises(cc.Refused, match=r"r1 is on several lines"):
        cc.build(s["store_proofs"], s["v3"], s["records"], twice, s["hosts"], s["endpoint_copies"], s["extra_copies"], s["class_overrides"])
    notjson = _write(tmp_path / "bad.json", "not json")
    for args in ((s["store_proofs"], notjson, s["records"], s["cell_classes"], s["hosts"], s["endpoint_copies"], s["extra_copies"], s["class_overrides"]),  # v3
                 (s["store_proofs"], s["v3"], s["records"], s["cell_classes"], notjson, s["endpoint_copies"], s["extra_copies"], s["class_overrides"]),  # hosts
                 (notjson, s["v3"], s["records"], s["cell_classes"], s["hosts"], s["endpoint_copies"], s["extra_copies"], s["class_overrides"]),  # store proofs
                 (s["store_proofs"], s["v3"], [notjson], s["cell_classes"], s["hosts"], s["endpoint_copies"], s["extra_copies"], s["class_overrides"]),  # record
                 (s["store_proofs"], s["v3"], s["records"], s["cell_classes"], s["hosts"], notjson, s["extra_copies"], s["class_overrides"]),  # endpoint copies
                 (s["store_proofs"], s["v3"], s["records"], s["cell_classes"], s["hosts"], s["endpoint_copies"], notjson, s["class_overrides"]),  # extra copies
                 (s["store_proofs"], s["v3"], s["records"], s["cell_classes"], s["hosts"], s["endpoint_copies"], s["extra_copies"], notjson)):  # class overrides
        with pytest.raises(cc.Refused, match=r"is not JSON"):
            cc.build(*args)
    nameless = _write(tmp_path / "speed-class-x.json", {"speed_class": "fast", "basis": "b"})
    with pytest.raises(cc.Refused, match=r"no copy named"):
        cc.build(s["store_proofs"], s["v3"], [nameless], s["cell_classes"], s["hosts"], s["endpoint_copies"], s["extra_copies"], s["class_overrides"])
    nobasis = _write(tmp_path / "speed-class-y.json", {"copy": "y", "speed_class": "fast"})
    with pytest.raises(cc.Refused, match=r"a speed class without a basis"):
        cc.build(s["store_proofs"], s["v3"], [nobasis], s["cell_classes"], s["hosts"], s["endpoint_copies"], s["extra_copies"], s["class_overrides"])
    with pytest.raises(cc.Refused, match=r"two records of one file name"):
        cc.build(s["store_proofs"], s["v3"], [s["records"][0], s["records"][0]], s["cell_classes"], s["hosts"], s["endpoint_copies"], s["extra_copies"], s["class_overrides"])


def test_the_command_writes_the_file_and_a_refusal_writes_nothing_and_exits_2(tmp_path, capsys):
    s = _sources(tmp_path)
    out = tmp_path / "copy-classes.json"
    argv = ["build", "--store-proofs", str(s["store_proofs"]), "--v3", str(s["v3"]), "--record", str(s["records"][0]),
            "--cell-classes", str(s["cell_classes"]), "--hosts", str(s["hosts"]),
            "--endpoint-copies", str(s["endpoint_copies"]), "--extra-copies", str(s["extra_copies"]),
            "--class-overrides", str(s["class_overrides"]), "--out", str(out)]
    assert cc.main(argv) == 0
    written = json.loads(out.read_text(encoding="utf-8"))  # by hand: the copies, sorted keys, a final newline, the hash
    assert written["copies"] == EXPECTED and list(written) == sorted(written) == ["copies", "rule", "schema", "sources"]
    assert out.read_text(encoding="utf-8").endswith("}\n") and not out.read_text(encoding="utf-8").endswith("\n\n")
    assert hashlib.sha256(out.read_bytes()).hexdigest() in capsys.readouterr().out
    out.unlink()
    bad = _write(tmp_path / "tsv-bad.tsv", "h:1\tr1\tpartial\tn\n")
    assert cc.main([*argv[:argv.index("--cell-classes") + 1], str(bad), *argv[argv.index("--cell-classes") + 2:]]) == 2
    assert not out.exists() and "refused" in capsys.readouterr().err


# --- the loader's strict validation ----------------------------------------------------------------------------------


@pytest.fixture
def good(tmp_path):
    return cc.build(**_sources(tmp_path))


def test_a_good_file_is_read_with_the_sha256_of_its_bytes(tmp_path, good):
    path = _write(tmp_path / "copy-classes.json", cc.dumps(good))
    doc, sha = cc.read(path)
    assert doc == good and sha == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name, edit", [
    ("a key missing", lambda d: d.pop("sources")), ("an extra key", lambda d: d.update(extra=1)),
    ("another schema", lambda d: d.update(schema="e189-copy-classes/2")), ("no rule", lambda d: d.update(rule=" ")),
    ("no sources", lambda d: d.update(sources={})), ("a source without a sha256", lambda d: d["sources"]["copy-hosts"].pop("sha256")),
    ("a source with a bad sha256", lambda d: d["sources"]["copy-hosts"].update(sha256="12")),
    ("a source without a path", lambda d: d["sources"]["copy-hosts"].update(path="")),
    ("no copies", lambda d: d.update(copies={})), ("a copy that is not an object", lambda d: d["copies"].update(r1="fast")),
    ("an entry key missing", lambda d: d["copies"]["r1"].pop("basis")),
    ("an entry extra key", lambda d: d["copies"]["r1"].update(note="x")),
    ("a class that is not one", lambda d: d["copies"]["r1"].update(**{"class": "partial"})),
    ("a class that is a number", lambda d: d["copies"]["r1"].update(**{"class": 3})),
    ("a label that does not match its class", lambda d: d["copies"]["r1"].update(label="slow")),
    ("a null class with a class label", lambda d: d["copies"]["r6"].update(label="fast")),
    ("a class with the no-rule label", lambda d: d["copies"]["r1"].update(label="unmeasured (no host rule for corporate)")),
    ("an empty basis", lambda d: d["copies"]["r1"].update(basis="")),
    ("a source the file does not list", lambda d: d["copies"]["r1"].update(source="nowhere")),
    ("a source that is no text", lambda d: d["copies"]["r1"].update(source=None)),
])
def test_a_malformed_file_refuses(tmp_path, good, name, edit):
    doc = copy.deepcopy(good)
    edit(doc)
    path = _write(tmp_path / "copy-classes.json", json.dumps(doc))
    with pytest.raises(cc.Refused):
        cc.read(path)


# Wrong at every position of the files (a text, an object with fixed keys, or a class): no container may be one of these
# either, since an object with other keys is no entry
WRONG_ANYWHERE = [None, True, False, 0, 3, 1.5, [], ["x"], [[]], {"a": 1}]
WRONG_LEAF = ["", " ", {}]  # and a text field may not be blank or an object


def _paths(doc, prefix=()):
    """Every key of the document at every depth, as the path to it."""
    if isinstance(doc, dict):
        for key, value in doc.items():
            yield (*prefix, key)
            yield from _paths(value, (*prefix, key))


def _with(doc, position, junk):
    """A copy of ``doc`` with the value at ``position`` set to ``junk`` (or removed, for "MISSING"), or None when that
    leaves it as it was."""
    doc = copy.deepcopy(doc)
    parent = doc
    for key in position[:-1]:
        parent = parent[key]
    if junk == "MISSING":
        del parent[position[-1]]
    elif parent[position[-1]] == junk and type(parent[position[-1]]) is type(junk):
        return None
    else:
        parent[position[-1]] = junk
    return doc


def _wrong_shapes(base):
    """Every (position, wrong value) of ``base``'s shape that must refuse: each key at each depth set to a value of
    WRONG_ANYWHERE, and each field that is a text (or the class, null for a copy with no rule) also blank, an empty
    object, or removed. Removing a whole copy or source, or setting a text field to other text, can be a valid file."""
    for position in _paths(base):
        leaf = base
        for key in position:
            leaf = leaf[key]
        for junk in WRONG_ANYWHERE:
            yield position, junk
        if not isinstance(leaf, dict):
            for junk in [*WRONG_LEAF, "MISSING"]:
                yield position, junk


def test_every_field_of_the_file_refuses_every_wrong_type_with_refused_and_never_another_exception(tmp_path, good):
    """External review of T4 (r1): a copy entry with "source": [] raised TypeError. Every key of the file's shape (each
    top-level key, each source's path and sha256, each copy and each of its four fields) is set to wrong values, and
    removed: the only acceptable outcome is Refused, from validate and from read."""
    shapes = [(p, j) for p, j in _wrong_shapes(good) if (d := _with(good, p, j)) is not None]
    assert len(shapes) > 400 and (("copies", "r1", "source"), []) in shapes
    assert (("sources", "copy-hosts", "sha256"), "MISSING") in shapes and (("copies", "r6", "class"), 3) in shapes
    for position, junk in shapes:
        doc = _with(good, position, junk)
        with pytest.raises(cc.Refused):
            cc.validate(doc)
        with pytest.raises(cc.Refused):
            cc.read(_write(tmp_path / "copy-classes.json", json.dumps(doc)))


def test_the_hosts_file_refuses_every_wrong_type_with_refused_and_never_another_exception():
    base = _hosts_doc(c=_hosts_entry("rhea-142", "example-host-1"))
    shapes = list(_wrong_shapes(base))
    assert len(shapes) > 100 and (("copies", "c", "graph"), []) in shapes
    for position, junk in shapes:
        doc = _with(base, position, junk)
        if doc is not None:
            with pytest.raises(cc.Refused):
                cc.parse_hosts(doc)


@pytest.mark.parametrize("text", ["", "not json", "[]", "null", '"x"'])
def test_a_file_that_is_not_a_json_object_refuses(tmp_path, text):
    with pytest.raises(cc.Refused, match=r"the copy classes is not JSON|the copy classes is not a JSON object"):
        cc.read(_write(tmp_path / "copy-classes.json", text))


# --- the endpoint copies (analysis lead 4, 25 Oct ~08:00Z) ---------------------------------------------------------------


def test_a_good_endpoint_copies_file_is_read_with_the_sha256_of_its_bytes(tmp_path):
    doc = _endpoint_doc(_endpoint_entry("a1", "g1", "1" * 64), _endpoint_entry("a2", "g1", "2" * 64),
                        _endpoint_entry("a3", "g2", "1" * 64))  # one endpoint hash may be another graph's
    path = _write(tmp_path / "endpoint-copies.json", doc)
    entries, sha = cc.read_endpoint_copies(path)
    assert entries == doc["copies"] and sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert cc.endpoint_replicas(entries) == {"a1": "g1", "a2": "g1", "a3": "g2"}


@pytest.mark.parametrize("name, edit", [
    ("another schema", lambda d: d.update(schema="e189-endpoint-copies/2")),
    ("an extra top-level key", lambda d: d.update(extra=1)), ("no rule", lambda d: d.pop("rule")),
    ("a blank rule", lambda d: d.update(rule=" ")), ("no copies", lambda d: d.update(copies=[])),
    ("copies as an object", lambda d: d.update(copies={"a": 1})),
    ("an entry that is no object", lambda d: d["copies"].append("x")),
    ("an entry with an extra key", lambda d: d["copies"][0].update(host="h")),
    ("an entry without its basis", lambda d: d["copies"][0].pop("basis")),
    ("an entry without its sources", lambda d: d["copies"][0].pop("sources")),
    ("a short endpoint sha256", lambda d: d["copies"][0].update(endpoint_sha256="ab" * 31)),
    ("an upper-case endpoint sha256", lambda d: d["copies"][0].update(endpoint_sha256="AB" * 32)),
    ("an endpoint given as text, not its sha256", lambda d: d["copies"][0].update(endpoint_sha256="not a sha256")),
    ("no sources", lambda d: d["copies"][0].update(sources=[])),
    ("a source without its sha256", lambda d: d["copies"][0]["sources"][0].pop("sha256")),
    ("a source with a short sha256", lambda d: d["copies"][0]["sources"][0].update(sha256="12")),
    ("a source with an extra key", lambda d: d["copies"][0]["sources"][0].update(note="x")),
    ("a source without a path", lambda d: d["copies"][0]["sources"][0].update(path=" ")),
    ("the same (graph, endpoint) twice", lambda d: d["copies"].append(_endpoint_entry("b", "g1", "1" * 64))),
    ("a replica for two endpoints of one graph", lambda d: d["copies"].append(_endpoint_entry("a1", "g1", "3" * 64))),
    ("a replica for two graphs", lambda d: d["copies"].append(_endpoint_entry("a1", "g2", "3" * 64))),
])
def test_the_endpoint_copies_reader_refuses(tmp_path, name, edit):
    doc = _endpoint_doc(_endpoint_entry("a1", "g1", "1" * 64), _endpoint_entry("a2", "g1", "2" * 64))
    edit(doc)
    with pytest.raises(cc.Refused):
        cc.parse_endpoint_copies(doc)
    with pytest.raises(cc.Refused):
        cc.read_endpoint_copies(_write(tmp_path / "endpoint-copies.json", json.dumps(doc)))


def test_every_field_of_the_endpoint_copies_refuses_every_wrong_type_with_refused_and_nothing_else(tmp_path):
    base = _endpoint_doc(_endpoint_entry("a1", "g1", "1" * 64))
    entry_fields = ("graph", "endpoint_sha256", "replica", "basis")
    shapes = []
    for key in ("schema", "rule", "copies"):
        shapes += [(("top", key), junk) for junk in WRONG_ANYWHERE]
    for key in entry_fields:
        shapes += [(("entry", key), junk) for junk in [*WRONG_ANYWHERE, *WRONG_LEAF]]
    shapes += [(("entry", "sources"), junk) for junk in WRONG_ANYWHERE]
    for key in ("path", "sha256"):
        shapes += [(("source", key), junk) for junk in [*WRONG_ANYWHERE, *WRONG_LEAF]]
    assert len(shapes) > 60
    for (where, key), junk in shapes:
        doc = copy.deepcopy(base)
        target = {"top": doc, "entry": doc["copies"][0], "source": doc["copies"][0]["sources"][0]}[where]
        if target[key] == junk and type(target[key]) is type(junk):
            continue
        target[key] = junk
        with pytest.raises(cc.Refused):
            cc.parse_endpoint_copies(doc)
    for key in ("path", "sha256"):  # a removed field
        doc = copy.deepcopy(base)
        del doc["copies"][0]["sources"][0][key]
        with pytest.raises(cc.Refused):
            cc.parse_endpoint_copies(doc)
    for not_json in ("", "not json", "[]", "null"):
        with pytest.raises(cc.Refused):
            cc.read_endpoint_copies(_write(tmp_path / "endpoint-copies.json", not_json))


def test_the_build_covers_the_endpoint_copies_and_leaves_every_other_entry_as_it_was(tmp_path):
    s = _sources(tmp_path)
    with_table = cc.build(**s)
    # the same sources with a table that names a second copy changes that copy's entry only
    more = _write(tmp_path / "endpoint-copies-2.json", _endpoint_doc(_endpoint_entry("r9"), _endpoint_entry(
        "r10", "rhea-142", "d" * 64)))
    tsv = Path(s["cell_classes"]).read_text(encoding="utf-8") + "h:10\tr10\tslow\tHost2, timed\n"
    again = cc.build(s["store_proofs"], s["v3"], s["records"], _write(tmp_path / "cells-2.tsv", tsv), s["hosts"], more, s["extra_copies"], s["class_overrides"])
    assert {k: v for k, v in again["copies"].items() if k != "r10"} == with_table["copies"] == EXPECTED
    assert again["copies"]["r10"] == {"class": "slow", "label": "slow", "basis": "Host2, timed", "source": "cell-classes"}


def test_a_table_copy_with_no_class_source_refuses_the_build_and_a_copy_in_both_files_has_one_graph(tmp_path):
    s = _sources(tmp_path)
    nowhere = _write(tmp_path / "endpoint-copies-3.json", _endpoint_doc(_endpoint_entry("r11")))
    with pytest.raises(cc.Refused, match=r"r11: found in no class source"):
        cc.build(s["store_proofs"], s["v3"], s["records"], s["cell_classes"], s["hosts"], nowhere, s["extra_copies"], s["class_overrides"])
    both = _write(tmp_path / "endpoint-copies-4.json", _endpoint_doc(_endpoint_entry("r1", "text2sparql-ck26")))
    with pytest.raises(cc.Refused, match=r"r1 is of one graph in the store proofs and of another"):
        cc.build(s["store_proofs"], s["v3"], s["records"], s["cell_classes"], s["hosts"], both, s["extra_copies"], s["class_overrides"])
    same = _write(tmp_path / "endpoint-copies-5.json", _endpoint_doc(_endpoint_entry("r1", "rhea-142")))
    assert cc.build(s["store_proofs"], s["v3"], s["records"], s["cell_classes"], s["hosts"], same, s["extra_copies"], s["class_overrides"])["copies"]["r1"] \
        == EXPECTED["r1"]  # a copy in both, of one graph, is covered once


def test_the_command_requires_the_endpoint_copies(tmp_path, capsys):
    s = _sources(tmp_path)
    argv = ["build", "--store-proofs", str(s["store_proofs"]), "--v3", str(s["v3"]), "--record", str(s["records"][0]),
            "--cell-classes", str(s["cell_classes"]), "--hosts", str(s["hosts"]), "--out", str(tmp_path / "x.json")]
    with pytest.raises(SystemExit) as exc:
        cc.main(argv)
    assert exc.value.code == 2 and not (tmp_path / "x.json").exists()
    assert "--endpoint-copies" in capsys.readouterr().err


# THE THREE SYNTHETIC ENDPOINT COPIES: the graph, the first 12 hex of the endpoint's sha256 (the label 'endpoint sha256:<12 hex>'
# results print), the replica
FOUR = [("uniprot-sp-2026_03", "2805a689b1da", "example-copy-4"), ("rhea-142", "f23c80fbc9a7", "example-copy-5"), ("uniprot-sp-2026_03", "04015f265acd", "example-copy-6")]


# the full endpoint sha256 of each
FOUR_FULL = {"example-copy-4": "2805a689b1da4baab370479dc86db2c3304a497c949bbb7aa4e6354ea0c4ea44", "example-copy-5": "f23c80fbc9a7011a2b7bc66aa675e797ef0a228346ff50b670496d97246199ef", "example-copy-6": "04015f265acde37906fd1adc8a6a3ab38b12e640f522794269fcc3f74eb5e813"}


def test_the_committed_endpoint_copies_carry_the_full_hashes_worked_out_from_the_evidence():
    entries, _ = cc.read_endpoint_copies(RESULTS / "endpoint-copies.json")
    assert {e["replica"]: e["endpoint_sha256"] for e in entries} == FOUR_FULL


def test_the_committed_endpoint_copies_are_the_four_the_lead_found():
    entries, _ = cc.read_endpoint_copies(RESULTS / "endpoint-copies.json")
    assert sorted((e["graph"], e["endpoint_sha256"][:12], e["replica"]) for e in entries) == sorted(FOUR)
    proofs = json.loads((RESULTS / "store-proofs.json").read_text(encoding="utf-8"))["proofs"]
    for e in entries:  # a copy the proofs know sits at the same full endpoint sha256 there
        theirs = {(p["graph"], p["endpoint_sha256"]) for p in proofs if p["replica"] == e["replica"]}
        assert theirs in (set(), {(e["graph"], e["endpoint_sha256"])})
    cited = {s["path"] for e in entries for s in e["sources"]}
    assert not any(token in cited_text for cited_text in cited for token in ("run-node", "http", "fwd"))


def test_the_committed_endpoint_copies_carry_no_host_name_port_or_url():
    text = (RESULTS / "endpoint-copies.json").read_text(encoding="utf-8")
    import re
    assert not re.search(r"https?:|run-node|fwd[0-9]|mab-n|pm-n|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|:[0-9]{3,5}\b",
                         text), "a host name, address, port or URL in the committed file"


# --- the extra copies (analysis lead 5, 25 Oct ~09:2xZ) ----------------------------------------------------------------------


def test_a_good_extra_copies_file_is_read_and_names_replica_and_graph(tmp_path):
    doc = _extra_doc(_extra_entry("e1", "rhea-142"), _extra_entry("e2", "uniprot-sp-2026_03"))
    path = _write(tmp_path / "extra-copies.json", doc)
    entries = cc.read_extra_copies(path)
    assert entries == doc["copies"] and cc.extra_replicas(entries) == {"e1": "rhea-142", "e2": "uniprot-sp-2026_03"}


@pytest.mark.parametrize("name, edit", [
    ("another schema", lambda d: d.update(schema="e189-extra-copies/2")),
    ("the endpoint copies' schema", lambda d: d.update(schema="e189-endpoint-copies/1")),
    ("an extra top-level key", lambda d: d.update(extra=1)), ("no rule", lambda d: d.pop("rule")),
    ("a blank rule", lambda d: d.update(rule=" ")), ("no copies", lambda d: d.update(copies=[])),
    ("copies as an object", lambda d: d.update(copies={"a": 1})),
    ("an entry that is no object", lambda d: d["copies"].append("x")),
    ("an entry with an extra key (an endpoint)", lambda d: d["copies"][0].update(endpoint_sha256="1" * 64)),
    ("an entry with an extra key (a host)", lambda d: d["copies"][0].update(host="h")),
    ("an entry without its graph", lambda d: d["copies"][0].pop("graph")),
    ("an entry without its basis", lambda d: d["copies"][0].pop("basis")),
    ("an entry without its sources", lambda d: d["copies"][0].pop("sources")),
    ("a graph no host rule names", lambda d: d["copies"][0].update(graph="another-store")),
    ("a graph given as a list", lambda d: d["copies"][0].update(graph=["rhea-142"])),
    ("a blank replica", lambda d: d["copies"][0].update(replica=" ")),
    ("no sources", lambda d: d["copies"][0].update(sources=[])),
    ("a source without its sha256", lambda d: d["copies"][0]["sources"][0].pop("sha256")),
    ("a source with a short sha256", lambda d: d["copies"][0]["sources"][0].update(sha256="12")),
    ("a source with an upper-case sha256", lambda d: d["copies"][0]["sources"][0].update(sha256="AB" * 32)),
    ("a source with an extra key", lambda d: d["copies"][0]["sources"][0].update(note="x")),
    ("a source without a path", lambda d: d["copies"][0]["sources"][0].update(path=" ")),
    ("a replica twice", lambda d: d["copies"].append(_extra_entry("e1", "rhea-142"))),
    ("a replica twice for two graphs", lambda d: d["copies"].append(_extra_entry("e1", "uniprot-sp-2026_03"))),
])
def test_the_extra_copies_reader_refuses(tmp_path, name, edit):
    doc = _extra_doc(_extra_entry("e1", "rhea-142"), _extra_entry("e2", "uniprot-sp-2026_03"))
    edit(doc)
    with pytest.raises(cc.Refused):
        cc.parse_extra_copies(doc)
    with pytest.raises(cc.Refused):
        cc.read_extra_copies(_write(tmp_path / "extra-copies.json", json.dumps(doc)))


def test_every_field_of_the_extra_copies_refuses_every_wrong_type_with_refused_and_nothing_else(tmp_path):
    base = _extra_doc(_extra_entry("e1", "rhea-142"))
    shapes = []
    for key in ("schema", "rule", "copies"):
        shapes += [(("top", key), junk) for junk in WRONG_ANYWHERE]
    for key in ("replica", "graph", "basis"):
        shapes += [(("entry", key), junk) for junk in [*WRONG_ANYWHERE, *WRONG_LEAF]]
    shapes += [(("entry", "sources"), junk) for junk in WRONG_ANYWHERE]
    for key in ("path", "sha256"):
        shapes += [(("source", key), junk) for junk in [*WRONG_ANYWHERE, *WRONG_LEAF]]
    assert len(shapes) > 50
    for (where, key), junk in shapes:
        doc = copy.deepcopy(base)
        target = {"top": doc, "entry": doc["copies"][0], "source": doc["copies"][0]["sources"][0]}[where]
        if target[key] == junk and type(target[key]) is type(junk):
            continue
        target[key] = junk
        with pytest.raises(cc.Refused):
            cc.parse_extra_copies(doc)
    for where, keys in (("entry", ("replica", "graph", "basis", "sources")), ("source", ("path", "sha256"))):
        for key in keys:  # a removed field
            doc = copy.deepcopy(base)
            target = doc["copies"][0] if where == "entry" else doc["copies"][0]["sources"][0]
            del target[key]
            with pytest.raises(cc.Refused):
                cc.parse_extra_copies(doc)
    for not_json in ("", "not json", "[]", "null"):
        with pytest.raises(cc.Refused):
            cc.read_extra_copies(_write(tmp_path / "extra-copies.json", not_json))
    with pytest.raises(cc.Refused, match=r"cannot be read"):  # a missing file, and a directory, are refused too
        cc.read_extra_copies(tmp_path / "no-such-extra-copies.json")
    with pytest.raises(cc.Refused, match=r"cannot be read"):
        cc.read_extra_copies(tmp_path)


def test_the_build_covers_the_extra_copies_and_leaves_every_other_entry_as_it_was(tmp_path):
    s = _sources(tmp_path)
    with_extra = cc.build(**s)
    assert with_extra["copies"] == EXPECTED and "r12" in with_extra["copies"]
    # the same sources with an extra file that names a second copy (with a table row) changes that copy's entry only
    more = _write(tmp_path / "extra-copies-2.json", _extra_doc(_extra_entry("r12"), _extra_entry("r13")))
    tsv = Path(s["cell_classes"]).read_text(encoding="utf-8") + "h:13\tr13\tslow\tHost2, timed\n"
    again = cc.build(s["store_proofs"], s["v3"], s["records"], _write(tmp_path / "cells-3.tsv", tsv), s["hosts"],
                     s["endpoint_copies"], more, s["class_overrides"])
    assert {k: v for k, v in again["copies"].items() if k != "r13"} == EXPECTED
    assert again["copies"]["r13"] == {"class": "slow", "label": "slow", "basis": "Host2, timed", "source": "cell-classes"}
    # and without its extra file's copy the copy is not covered (nothing else changes)
    one = _write(tmp_path / "extra-copies-3.json", _extra_doc(_extra_entry("r13")))
    without = cc.build(s["store_proofs"], s["v3"], s["records"], _write(tmp_path / "cells-4.tsv", tsv), s["hosts"],
                       s["endpoint_copies"], one, s["class_overrides"])
    assert set(without["copies"]) == (set(EXPECTED) - {"r12"}) | {"r13"}


def test_an_extra_copy_with_no_class_source_refuses_the_build_and_a_copy_in_several_files_has_one_graph(tmp_path):
    s = _sources(tmp_path)
    nowhere = _write(tmp_path / "extra-copies-4.json", _extra_doc(_extra_entry("r14")))
    with pytest.raises(cc.Refused, match=r"r14: found in no class source"):
        cc.build(s["store_proofs"], s["v3"], s["records"], s["cell_classes"], s["hosts"], s["endpoint_copies"], nowhere, s["class_overrides"])
    # (a class from the host rule is not a fallback for an extra copy: it must be in the hosts file to get one)
    hosts_doc = json.loads(Path(s["hosts"]).read_text(encoding="utf-8"))
    hosts_doc["copies"]["r14"] = _hosts_entry("rhea-142", "example-host-2", "b14")
    hosts_has = _write(tmp_path / "hosts-2.json", hosts_doc)
    again = cc.build(s["store_proofs"], s["v3"], s["records"], s["cell_classes"], hosts_has, s["endpoint_copies"], nowhere, s["class_overrides"])
    assert again["copies"]["r14"] == {"class": "medium", "label": "medium (host rule, unmeasured)", "basis": "b14",
                                      "source": "copy-hosts"}
    for name, graph in (("r1", "text2sparql-ck26"), ("r9", "rhea-142")):  # a proof's copy and an endpoint copy's
        other = _write(tmp_path / f"extra-copies-{name}.json", _extra_doc(_extra_entry(name, graph)))
        with pytest.raises(cc.Refused, match=rf"{name} is of one graph in the extra copies and of another"):
            cc.build(s["store_proofs"], s["v3"], s["records"], s["cell_classes"], s["hosts"], s["endpoint_copies"], other, s["class_overrides"])
    same = _write(tmp_path / "extra-copies-5.json", _extra_doc(_extra_entry("r1", "rhea-142"), _extra_entry("r9", "opaque-rhea-142")))
    covered = cc.build(s["store_proofs"], s["v3"], s["records"], s["cell_classes"], s["hosts"], s["endpoint_copies"], same, s["class_overrides"])
    assert {k: covered["copies"][k] for k in ("r1", "r9")} == {k: EXPECTED[k] for k in ("r1", "r9")}  # covered once


def test_the_command_requires_the_extra_copies(tmp_path, capsys):
    s = _sources(tmp_path)
    argv = ["build", "--store-proofs", str(s["store_proofs"]), "--v3", str(s["v3"]), "--record", str(s["records"][0]),
            "--cell-classes", str(s["cell_classes"]), "--hosts", str(s["hosts"]),
            "--endpoint-copies", str(s["endpoint_copies"]), "--class-overrides", str(s["class_overrides"]),
            "--out", str(tmp_path / "x.json")]
    with pytest.raises(SystemExit) as exc:
        cc.main(argv)
    assert exc.value.code == 2 and not (tmp_path / "x.json").exists()
    assert "--extra-copies" in capsys.readouterr().err
    bad = _write(tmp_path / "extra-bad.json", _extra_doc(_extra_entry("r12", "another-store")))
    assert cc.main([*argv, "--extra-copies", str(bad)]) == 2 and not (tmp_path / "x.json").exists()
    assert "refused" in capsys.readouterr().err
    assert cc.main([*argv, "--extra-copies", str(tmp_path / "missing.json")]) == 2  # a missing file: a refusal, no trace
    assert not (tmp_path / "x.json").exists() and "cannot be read" in capsys.readouterr().err


# --- the class overrides (analysis lead 5, 25 Oct ~12:5xZ) --------------------------------------------------------------


def test_a_good_overrides_file_is_read_as_two_maps(tmp_path):
    doc = _overrides_doc([_info_entry("o1"), _info_entry("o2")], [_basis_entry("b1")], [_measured_entry("m1", "slow")])
    info, bases, measured = cc.read_overrides(_write(tmp_path / "class-overrides.json", doc))
    assert sorted(measured) == ["m1"] and measured["m1"]["class"] == "slow"
    assert sorted(info) == ["o1", "o2"] and sorted(bases) == ["b1"] and bases["b1"]["basis"] == "a test basis text"
    assert cc.parse_overrides(_overrides_doc()) == ({}, {}, {})  # (empty lists are a valid file)


@pytest.mark.parametrize("name, edit", [
    ("another schema", lambda d: d.update(schema="e189-class-overrides/1")),
    ("no measured", lambda d: d.pop("measured")), ("measured as an object", lambda d: d.update(measured={})),
    ("a measured class without its class", lambda d: d["measured"][0].pop("class")),
    ("a measured class that is no class", lambda d: d["measured"][0].update({"class": "excellent"})),
    ("a measured class without its basis", lambda d: d["measured"][0].pop("basis")),
    ("a measured class with a blank basis", lambda d: d["measured"][0].update(basis=" ")),
    ("a measured class with table rows", lambda d: d["measured"][0].update(table_rows=_lines(("fast", "n")))),
    ("a measured class without sources", lambda d: d["measured"][0].update(sources=[])),
    ("a replica in a measured class and a basis", lambda d: d["measured"].append(_measured_entry("b1"))),
    ("a replica twice among the measured", lambda d: d["measured"].append(_measured_entry("m1"))),
    ("an extra top-level key", lambda d: d.update(extra=1)), ("no rule", lambda d: d.pop("rule")),
    ("a blank rule", lambda d: d.update(rule=" ")), ("no informational rows", lambda d: d.pop("informational_rows")),
    ("no bases", lambda d: d.pop("bases")), ("informational rows as an object", lambda d: d.update(informational_rows={})),
    ("bases as text", lambda d: d.update(bases="x")),
    ("an entry that is no object", lambda d: d["informational_rows"].append("x")),
    ("an informational row with a basis", lambda d: d["informational_rows"][0].update(basis="b")),
    ("an informational row without its why", lambda d: d["informational_rows"][0].pop("why")),
    ("an informational row without sources", lambda d: d["informational_rows"][0].pop("sources")),
    ("a basis without its basis text", lambda d: d["bases"][0].pop("basis")),
    ("a basis with a blank text", lambda d: d["bases"][0].update(basis=" ")),
    ("a basis without its why", lambda d: d["bases"][0].pop("why")),
    ("a basis with an extra key", lambda d: d["bases"][0].update(class_="slow")),
    ("a blank replica", lambda d: d["informational_rows"][0].update(replica=" ")),
    ("no table rows", lambda d: d["informational_rows"][0].pop("table_rows")),
    ("empty table rows", lambda d: d["bases"][0].update(table_rows=[])),
    ("table rows as an object", lambda d: d["bases"][0].update(table_rows={"class": "fast", "note": "n"})),
    ("a table row that is no object", lambda d: d["bases"][0]["table_rows"].append("x")),
    ("a table row with a class that is no class", lambda d: d["bases"][0]["table_rows"][0].update({"class": "excellent"})),
    ("a table row with a null class", lambda d: d["informational_rows"][0]["table_rows"][0].update({"class": None})),
    ("a table row without its note", lambda d: d["bases"][0]["table_rows"][0].pop("note")),
    ("a table row with a blank note", lambda d: d["informational_rows"][0]["table_rows"][0].update(note=" ")),
    ("a table row with an extra key", lambda d: d["bases"][0]["table_rows"][0].update(endpoint="e")),
    ("no sources", lambda d: d["bases"][0].update(sources=[])),
    ("a source with a short sha256", lambda d: d["bases"][0]["sources"][0].update(sha256="12")),
    ("a source with an extra key", lambda d: d["informational_rows"][0]["sources"][0].update(note="x")),
    ("a source without a path", lambda d: d["informational_rows"][0]["sources"][0].update(path=" ")),
    ("a replica twice among the informational rows", lambda d: d["informational_rows"].append(_info_entry("o1"))),
    ("a replica twice among the bases", lambda d: d["bases"].append(_basis_entry("b1"))),
    ("a replica in both lists", lambda d: d["bases"].append(_basis_entry("o1"))),
])
def test_the_overrides_reader_refuses(tmp_path, name, edit):
    doc = _overrides_doc([_info_entry("o1")], [_basis_entry("b1")], [_measured_entry("m1")])
    edit(doc)
    with pytest.raises(cc.Refused):
        cc.parse_overrides(doc)
    with pytest.raises(cc.Refused):
        cc.read_overrides(_write(tmp_path / "class-overrides.json", json.dumps(doc)))


def test_every_field_of_the_overrides_refuses_every_wrong_type_with_refused_and_nothing_else(tmp_path):
    base = _overrides_doc([_info_entry("o1")], [_basis_entry("b1")], [_measured_entry("m1")])
    shapes = [(("top",), key, junk) for key in ("schema", "rule", "informational_rows", "bases", "measured")
              for junk in WRONG_ANYWHERE]
    shapes += [(("measured",), key, junk) for key in ("replica", "class", "basis", "why") for junk in [*WRONG_ANYWHERE, *WRONG_LEAF]]
    shapes += [(("measured",), "sources", junk) for junk in WRONG_ANYWHERE]
    shapes += [(("measured", "sources"), key, junk) for key in ("path", "sha256") for junk in [*WRONG_ANYWHERE, *WRONG_LEAF]]
    for section, keys in (("informational_rows", ("replica", "why")), ("bases", ("replica", "basis", "why"))):
        shapes += [((section,), key, junk) for key in keys for junk in [*WRONG_ANYWHERE, *WRONG_LEAF]]
        shapes += [((section,), "sources", junk) for junk in WRONG_ANYWHERE]
        shapes += [((section, "sources"), key, junk) for key in ("path", "sha256") for junk in [*WRONG_ANYWHERE, *WRONG_LEAF]]
        shapes += [((section,), "table_rows", junk) for junk in WRONG_ANYWHERE]
        shapes += [((section, "table_rows"), key, junk) for key in ("class", "note") for junk in [*WRONG_ANYWHERE, *WRONG_LEAF]]
    assert len(shapes) > 100
    for where, key, junk in shapes:
        doc = copy.deepcopy(base)
        target = doc if where[0] == "top" else doc[where[0]][0]
        if len(where) == 2:
            target = target[where[1]][0]
        if target[key] == junk and type(target[key]) is type(junk):
            continue
        if where == ("top",) and key in ("informational_rows", "bases", "measured") and junk == []:
            continue  # (an empty list is a valid file: a union of overrides may be empty)
        target[key] = junk
        with pytest.raises(cc.Refused):
            cc.parse_overrides(doc)
    for section, keys in (("informational_rows", ("replica", "table_rows", "why", "sources")),
                          ("bases", ("replica", "table_rows", "basis", "why", "sources")),
                          ("measured", ("replica", "class", "basis", "why", "sources"))):
        for key in keys:  # a removed field
            doc = copy.deepcopy(base)
            del doc[section][0][key]
            with pytest.raises(cc.Refused):
                cc.parse_overrides(doc)
    for not_json in ("", "not json", "[]", "null"):
        with pytest.raises(cc.Refused):
            cc.read_overrides(_write(tmp_path / "class-overrides.json", not_json))
    with pytest.raises(cc.Refused, match=r"cannot be read"):
        cc.read_overrides(tmp_path / "no-such-overrides.json")
    with pytest.raises(cc.Refused, match=r"cannot be read"):
        cc.read_overrides(tmp_path)


def _with_r15(tmp_path, s):
    """The fixture sources plus r15: an extra copy (rhea-142) with an informational table row (medium) and a host-rule entry
    (example-host-2 on Rhea: medium); returns the keyword sources with those files swapped in."""
    extra = _write(tmp_path / "extra-r15.json", _extra_doc(_extra_entry("r12"), _extra_entry("r15")))
    tsv = Path(s["cell_classes"]).read_text(encoding="utf-8") + "h:15\tr15\tmedium\tinformational: same class as a sibling\n"
    hosts_doc = json.loads(Path(s["hosts"]).read_text(encoding="utf-8"))
    hosts_doc["copies"]["r15"] = _hosts_entry("rhea-142", "example-host-2", "b15")
    return {**s, "extra_copies": extra, "cell_classes": _write(tmp_path / "cells-r15.tsv", tsv),
            "hosts": _write(tmp_path / "hosts-r15.json", hosts_doc)}


def test_an_informational_row_is_ignored_for_the_copies_the_overrides_name_and_nothing_else_changes(tmp_path):
    s = _with_r15(tmp_path, _sources(tmp_path))
    plain = cc.build(**s)  # without an override the table's row decides (the order of the precedence)
    assert plain["copies"]["r15"] == {"class": "medium", "label": "medium", "basis": "informational: same class as a sibling",
                                      "source": "cell-classes"}
    overrides = _write(tmp_path / "ov.json", _overrides_doc([_info_entry("r15")]))
    done = cc.build(**{**s, "class_overrides": overrides})
    assert done["copies"]["r15"] == {"class": "medium", "label": "medium (host rule, unmeasured)", "basis": "b15",
                                     "source": "copy-hosts"}
    assert {k: v for k, v in done["copies"].items() if k != "r15"} == {k: v for k, v in plain["copies"].items() if k != "r15"}
    assert done["sources"]["class-overrides"] == {"path": str(overrides), "sha256": hashlib.sha256(overrides.read_bytes()).hexdigest()}
    assert cc.validate(done) is done


def test_a_basis_override_changes_the_basis_of_a_table_entry_only(tmp_path):
    s = _sources(tmp_path)
    overrides = _write(tmp_path / "ov.json", _overrides_doc(bases=[_basis_entry("r3", "the basis the override gives")]))
    done = cc.build(**{**s, "class_overrides": overrides})
    assert done["copies"]["r3"] == {**EXPECTED["r3"], "basis": "the basis the override gives"}  # class, label, source as they were
    assert {k: v for k, v in done["copies"].items() if k != "r3"} == {k: v for k, v in EXPECTED.items() if k != "r3"}


def test_an_override_with_nothing_to_override_refuses_the_build(tmp_path):
    s = _with_r15(tmp_path, _sources(tmp_path))

    def refused(match, info=(), bases=(), **swap):
        overrides = _write(tmp_path / "ov-refused.json", _overrides_doc(info, bases))
        with pytest.raises(cc.Refused, match=match):
            cc.build(**{**s, **swap, "class_overrides": overrides})

    refused(r"zz: an informational-row override for a copy the build does not cover", info=[_info_entry("zz")])
    refused(r"r6: its informational-row override has no row in the cell table", info=[_info_entry("r6")])  # (a stale override)
    refused(r"r1: its informational-row override is for a copy with a measurement",
            info=[_info_entry("r1", _lines(("slow", "Host2")))])  # v3 (r1's table line is slow, "Host2")
    with_row = _write(tmp_path / "cells-r2.tsv", Path(s["cell_classes"]).read_text(encoding="utf-8") + "h:2\tr2\tfast\tn\n")
    refused(r"r2: its informational-row override is for a copy with a measurement",
            info=[_info_entry("r2", _lines(("fast", "n")))], cell_classes=with_row)  # a record
    refused(r"zz: a basis override for a copy the build does not cover", bases=[_basis_entry("zz")])
    refused(r"r1: a basis override is for an entry that comes from the cell table, not from speed-class-v3", bases=[_basis_entry("r1")])
    refused(r"r5: a basis override is for an entry that comes from the cell table, not from copy-hosts", bases=[_basis_entry("r5")])
    # the table's lines are no longer the ones the override was reviewed against: a re-pinned table that changed them
    gone = r"the cell table's lines are no longer the ones its (informational-row|basis) override was reviewed against"
    refused(rf"r15: {gone}", info=[_info_entry("r15", _lines(("slow", "informational: same class as a sibling")))])  # class
    refused(rf"r15: {gone}", info=[_info_entry("r15", _lines(("medium", "informational: same class as a sibling, but measured")))])
    refused(rf"r3: {gone}", bases=[_basis_entry("r3", table_rows=_lines(("medium", "Host1, measured fast")))])
    refused(rf"r3: {gone}", bases=[_basis_entry("r3", table_rows=_lines(("fast", "Host1, measured fast, again")))])
    refused(rf"r15: {gone}", info=[_info_entry("r15", _lines(("medium", "informational: same class as a sibling"),
                                                             ("medium", "a second line the override never saw")))])  # fewer lines
    # the case that matters: a LINE ADDED after the reviewed one (same class: the parser keeps only the first note) says the copy was
    # measured; the override, which holds every line, no longer matches and the build refuses instead of dropping the measurement
    added = _write(tmp_path / "cells-added.tsv", Path(s["cell_classes"]).read_text(encoding="utf-8")
                   + "h:16\tr15\tmedium\tmeasured on this copy itself\n")
    refused(rf"r15: {gone}", info=[_info_entry("r15")], cell_classes=added)
    added3 = _write(tmp_path / "cells-added3.tsv", Path(s["cell_classes"]).read_text(encoding="utf-8")
                    + "h:16\tr3\tfast\tmeasured again\n")
    refused(rf"r3: {gone}", bases=[_basis_entry("r3")], cell_classes=added3)
    # an informational row of a copy the hosts file does not give: the host rule has nothing, so the copy has no class source
    hosts_doc = json.loads(Path(s["hosts"]).read_text(encoding="utf-8"))
    del hosts_doc["copies"]["r15"]
    refused(r"r15: found in no class source", info=[_info_entry("r15")], hosts=_write(tmp_path / "hosts-no15.json", hosts_doc))


def test_a_measured_class_gives_a_copy_with_no_other_class_source_its_class_and_the_override_as_its_source(tmp_path):
    s = _sources(tmp_path)
    extra = _write(tmp_path / "extra-r20.json", _extra_doc(_extra_entry("r12"), _extra_entry("r20")))
    plain = {**s, "extra_copies": extra}
    with pytest.raises(cc.Refused, match=r"r20: found in no class source"):  # nothing gives r20 a class without the override
        cc.build(**plain)
    overrides = _write(tmp_path / "ov-m.json", _overrides_doc(measured=[_measured_entry("r20", "medium", "measured, test")]))
    done = cc.build(**{**plain, "class_overrides": overrides})
    assert done["copies"]["r20"] == {"class": "medium", "label": "medium", "basis": "measured, test", "source": "class-overrides"}
    assert {k: v for k, v in done["copies"].items() if k != "r20"} == EXPECTED
    assert done["sources"]["class-overrides"]["sha256"] == hashlib.sha256(overrides.read_bytes()).hexdigest()
    assert cc.validate(done) is done  # (its source, class-overrides, is one the document lists)


def test_a_measured_class_refuses_when_another_source_has_a_class_or_a_line_for_the_copy(tmp_path):
    s = _with_r15(tmp_path, _sources(tmp_path))

    def refused(match, replica, **swap):
        overrides = _write(tmp_path / "ov-mr.json", _overrides_doc(measured=[_measured_entry(replica)]))
        with pytest.raises(cc.Refused, match=match):
            cc.build(**{**s, **swap, "class_overrides": overrides})

    now = r"its measured-class override is for a copy that now has a v3 row, a per-copy record or a cell-table row"
    refused(rf"r1: {now}", "r1")  # a v3 row (and a table line)
    refused(rf"r2: {now}", "r2")  # a per-copy record
    refused(rf"r3: {now}", "r3")  # a table line only (r3's v3 partial falls through to it)
    refused(rf"r15: {now}", "r15")  # an informational-looking table line: needs review, not silently preferred
    refused(r"zz: a measured-class override for a copy the build does not cover", "zz")


def test_the_command_requires_the_class_overrides_and_refuses_a_bad_file(tmp_path, capsys):
    s = _sources(tmp_path)
    argv = ["build", "--store-proofs", str(s["store_proofs"]), "--v3", str(s["v3"]), "--record", str(s["records"][0]),
            "--cell-classes", str(s["cell_classes"]), "--hosts", str(s["hosts"]), "--endpoint-copies", str(s["endpoint_copies"]),
            "--extra-copies", str(s["extra_copies"]), "--out", str(tmp_path / "x.json")]
    with pytest.raises(SystemExit) as exc:
        cc.main(argv)
    assert exc.value.code == 2 and not (tmp_path / "x.json").exists()
    assert "--class-overrides" in capsys.readouterr().err
    bad = _write(tmp_path / "ov-bad.json", _overrides_doc([_info_entry("zz")]))
    assert cc.main([*argv, "--class-overrides", str(bad)]) == 2 and not (tmp_path / "x.json").exists()
    assert "refused" in capsys.readouterr().err
    assert cc.main([*argv, "--class-overrides", str(tmp_path / "missing.json")]) == 2
    assert not (tmp_path / "x.json").exists() and "cannot be read" in capsys.readouterr().err


# --- the committed file -----------------------------------------------------------------------------------------------


def _committed() -> dict:
    doc, _ = cc.read(RESULTS / "copy-classes.json")
    return doc


def test_the_committed_file_covers_exactly_the_replicas_of_store_proofs_json_endpoint_copies_json_and_extra_copies_json():
    proofs = json.loads((RESULTS / "store-proofs.json").read_text(encoding="utf-8"))
    table, _ = cc.read_endpoint_copies(RESULTS / "endpoint-copies.json")
    extra = cc.read_extra_copies(RESULTS / "extra-copies.json")
    assert set(_committed()["copies"]) == {p["replica"] for p in proofs["proofs"]} | {e["replica"] for e in table} \
        | {e["replica"] for e in extra}
    assert {e["replica"] for e in table} == {"example-copy-4", "example-copy-5", "example-copy-6"}
    assert {e["replica"]: e["graph"] for e in extra} == {"example-copy-7": "opaque-rhea-142", "example-copy-8": "text2sparql-ck26"}
    assert set(e["replica"] for e in extra) <= set(_committed()["copies"])
    assert not {"example-copy-1", "example-copy-2", "example-copy-3"} & {e["replica"] for e in extra}
    assert len(_committed()["copies"]) == 8


def test_the_committed_file_names_the_committed_hosts_copy_and_the_store_proofs_by_their_sha256():
    sources = _committed()["sources"]
    for name, file in (("copy-hosts", "copy-hosts.json"), ("store-proofs", "store-proofs.json"),
                       ("endpoint-copies", "endpoint-copies.json"), ("extra-copies", "extra-copies.json"),
                       ("class-overrides", "class-overrides.json")):
        assert sources[name] == {"path": file, "sha256": hashlib.sha256((RESULTS / file).read_bytes()).hexdigest()}
    assert sorted(n for n in sources if n.startswith("record:")) == []


V3, CELLS, HOSTS = "speed-class-v3", "cell-classes", "copy-hosts"
SLOW_RULE, FAST_RULE = "slow (host rule, unmeasured)", "fast (host rule, unmeasured)"
# The expected entry (class, label, source) of each copy of the synthetic tables
LEAD_EXPECTS = {
    "example-copy-1": ("fast", "fast", V3), "example-copy-2": ("slow", SLOW_RULE, HOSTS),
    "example-copy-3": ("fast", "fast", V3), "example-copy-4": ("fast", "fast", CELLS),
    "example-copy-5": ("fast", FAST_RULE, HOSTS), "example-copy-6": ("fast", FAST_RULE, HOSTS),
    "example-copy-7": ("medium", "medium", "class-overrides"),
    "example-copy-8": (None, "unmeasured (no host rule for corporate)", HOSTS),
}


@pytest.mark.parametrize("replica", sorted(LEAD_EXPECTS))
def test_the_committed_classes_of_the_branch_phase_copies_are_the_ones_the_lead_worked_out_by_hand(replica):
    entry = _committed()["copies"][replica]
    assert (entry["class"], entry["label"], entry["source"]) == LEAD_EXPECTS[replica]


def test_the_committed_tables_note_and_the_hosts_rule_copies_carry_the_registrys_basis():
    pytest.skip("needs the authors' store tables")


def test_the_committed_extra_copies_classes_carry_the_sources_own_words():
    pytest.skip("needs the authors' store tables")


def test_the_committed_extra_copies_carry_no_host_name_port_or_url():
    text = (RESULTS / "extra-copies.json").read_text(encoding="utf-8")
    import re
    assert not re.search(r"https?:|run-node|fwd[0-9]|mab-n|pm-n|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|:[0-9]{3,5}\b",
                         text), "a host name, address, port or URL in the committed file"
    for entry in json.loads(text)["copies"]:
        assert set(entry) == {"replica", "graph", "basis", "sources"}  # no endpoint hash either


def _pinned(path: str, sha256: str) -> Path:
    """A class source the committed file cites, or a skip when this machine does not hold it; a file that is there must be
    the cited bytes."""
    file = Path(path)
    if not file.is_file():
        pytest.skip(f"{file.name} is not on this machine")
    assert hashlib.sha256(file.read_bytes()).hexdigest() == sha256, f"{file.name} is not the file the copy classes cite"
    return file


def test_the_two_extra_entries_read_back_from_their_named_sources_by_code_that_shares_nothing_with_the_build():
    pytest.skip("needs the authors' store tables")


def test_the_four_dsh_copies_read_back_from_the_v3_file_by_code_that_shares_nothing_with_the_build():
    pytest.skip("needs the authors' store tables")


def test_the_four_tier2_replay_copies_read_back_from_their_named_sources_by_code_that_shares_nothing_with_the_build():
    pytest.skip("needs the authors' store tables")


def test_the_committed_class_overrides_are_the_three_the_rulings_name_and_cite_files_that_say_so():
    pytest.skip("needs the authors' store tables")


# --- a private table beside the committed one: the loader's bytes entry and the merge rule ------------------------------

PC_FAST = {"class": "fast", "label": "fast", "basis": "basis fast", "source": "src"}
PC_SLOW = {"class": "slow", "label": "slow (host rule, unmeasured)", "basis": "basis slow", "source": "src"}
PC_MED = {"class": "medium", "label": "medium", "basis": "basis medium", "source": "src"}


def _classes_doc(**copies) -> dict:
    return {"schema": "e189-copy-classes/1", "rule": "a test", "sources": {"src": {"path": "/src.json", "sha256": SHA}},
            "copies": copies}


def test_loads_is_the_parse_step_of_read_and_refuses_what_read_refuses(tmp_path):
    doc = _classes_doc(c1=PC_FAST, c2=PC_SLOW)
    raw = json.dumps(doc, indent=1).encode()
    assert cc.loads(raw) == doc
    assert cc.read(_write(tmp_path / "t.json", raw.decode())) == (doc, hashlib.sha256(raw).hexdigest())
    wrong_label = _classes_doc(c1={**PC_FAST, "label": "slow"})
    cases = ((b"not json", "the copy classes is not JSON"), (b"[]", "the copy classes is not a JSON object"),
             (json.dumps({**doc, "extra": 1}).encode(), "the copy classes are not a document of schema, rule, sources "
                                                        "and copies"),
             (json.dumps(wrong_label).encode(), "the label of c1 does not match its class"),
             (json.dumps(_classes_doc()).encode(), "the copy classes have no copies"))
    for n, (bad, message) in enumerate(cases):  # the expected texts are written by hand
        with pytest.raises(cc.Refused) as via_loads:
            cc.loads(bad)
        with pytest.raises(cc.Refused) as via_read:
            cc.read(_write(tmp_path / f"bad-{n}.json", bad.decode()))
        assert str(via_loads.value) == str(via_read.value) == message


def test_merge_classes_takes_every_copy_of_either_and_a_copy_in_both_only_when_its_entries_are_identical():
    committed = {"c1": dict(PC_FAST), "c2": dict(PC_SLOW)}
    private = {"c3": dict(PC_MED), "c2": dict(PC_SLOW)}
    before = json.dumps([committed, private], sort_keys=True)
    merged = cc.merge_classes(committed, private)
    assert merged == {"c1": PC_FAST, "c2": PC_SLOW, "c3": PC_MED}
    assert cc.merge_classes(private, committed) == merged  # the same table whichever side is the private one
    assert json.dumps([committed, private], sort_keys=True) == before  # neither input is changed
    merged["c1"]["class"] = "slow"
    assert committed["c1"] == PC_FAST  # nor shares an entry with the merged table
    assert cc.merge_classes(committed, {}) == committed and cc.merge_classes({}, private) == private
    for field, value in (("class", "medium"), ("label", "medium"), ("basis", "another basis"), ("source", "another")):
        with pytest.raises(cc.Refused, match=r"^a copy is in the committed copy classes and in the private table with "
                                             r"different entries$") as refused:
            cc.merge_classes(committed, {"c2": {**PC_SLOW, field: value}})
        assert "c2" not in str(refused.value)  # the private table's copy is not named


def test_the_private_table_path_validates_and_merges_an_entry_of_the_measured_override_kind():
    """A copy class from the measured kind of class-overrides.json (source "class-overrides", label the class) goes through the
    private table's two functions as any entry does: loads validates it (its source is one of the document's), and merge_classes
    keeps an identical entry in both tables and refuses a different one. The committed entry is the one tested."""
    committed = _committed()
    nine = committed["copies"]["example-copy-7"]
    assert nine["source"] == "class-overrides" and nine["label"] == nine["class"] == "medium"
    private_doc = {**committed, "copies": {"private-1": dict(nine), "example-copy-7": dict(nine)}}
    assert cc.loads(json.dumps(private_doc).encode())["copies"]["private-1"] == nine  # validated like the committed file
    merged = cc.merge_classes(committed["copies"], private_doc["copies"])
    assert merged["example-copy-7"] == nine and merged["private-1"] == nine and len(merged) == len(committed["copies"]) + 1
    for field, value in (("class", "slow"), ("label", "slow"), ("basis", "another"), ("source", "cell-classes")):
        with pytest.raises(cc.Refused, match=r"^a copy is in the committed copy classes and in the private table with "
                                             r"different entries$"):
            cc.merge_classes(committed["copies"], {"example-copy-7": {**nine, field: value}})
    unlisted = {**private_doc, "copies": {"private-1": {**nine, "source": "not-a-source-of-the-document"}}}
    with pytest.raises(cc.Refused, match=r"lacks its basis or names a source the file does not list"):
        cc.loads(json.dumps(unlisted).encode())


def test_loads_endpoint_copies_is_the_parse_step_of_read_endpoint_copies_and_refuses_what_it_refuses(tmp_path):
    doc = _endpoint_doc(_endpoint_entry("c1", endpoint="1" * 64), _endpoint_entry("c2", endpoint="2" * 64))
    raw = json.dumps(doc, indent=1).encode()
    assert cc.loads_endpoint_copies(raw) == doc["copies"]
    assert cc.read_endpoint_copies(_write(tmp_path / "t.json", raw.decode())) == (doc["copies"],
                                                                                 hashlib.sha256(raw).hexdigest())
    bad_entry = _endpoint_doc({"graph": "g"})
    cases = ((b"not json", "the endpoint copies is not JSON"), (b"[]", "the endpoint copies is not a JSON object"),
             (json.dumps(bad_entry).encode(), "an endpoint copy is not graph, endpoint_sha256, replica, basis and sources"),
             (json.dumps(_endpoint_doc()).encode(), "the endpoint copies are not a e189-endpoint-copies/1 file (schema, "
                                                    "rule and a non-empty copies list, nothing else)"))
    for n, (bad, message) in enumerate(cases):  # the expected texts are written by hand
        with pytest.raises(cc.Refused) as via_loads:
            cc.loads_endpoint_copies(bad)
        with pytest.raises(cc.Refused) as via_read:
            cc.read_endpoint_copies(_write(tmp_path / f"bad-{n}.json", bad.decode()))
        assert str(via_loads.value) == str(via_read.value) == message


def test_merge_endpoint_copies_is_one_validated_list_and_an_endpoint_or_a_replica_twice_refuses():
    committed = [_endpoint_entry("c1", graph="g", endpoint="1" * 64)]
    private = [_endpoint_entry("c2", graph="g", endpoint="2" * 64), _endpoint_entry("c3", graph="h", endpoint="2" * 64)]
    before = json.dumps([committed, private], sort_keys=True)
    merged = cc.merge_endpoint_copies(committed, private)
    assert merged == [*committed, *private] and cc.merge_endpoint_copies(committed, []) == committed
    assert json.dumps([committed, private], sort_keys=True) == before  # neither input is changed
    merged[0]["replica"] = "other"
    assert committed[0]["replica"] == "c1"  # nor shares an entry with the merged list
    fixed = r"^the private endpoint copies and the committed ones name an endpoint twice, or a replica for two " \
            r"endpoints or graphs$"
    for clash in ([dict(committed[0])],  # an endpoint in both tables: identical too
                  [_endpoint_entry("c1", graph="g", endpoint="3" * 64)],  # a replica at two endpoints
                  [_endpoint_entry("c1", graph="h", endpoint="1" * 64)]):  # a replica of two graphs
        with pytest.raises(cc.Refused, match=fixed) as refused:
            cc.merge_endpoint_copies(committed, clash)
        assert not any(name in str(refused.value) for name in ("c1", "g twice", "1" * 12))  # no copy, graph or endpoint

