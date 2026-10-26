"""Synthetic checks for a privately pinned graph's plain-run partition union."""

import copy
import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "exploratory.py"
SPEC = importlib.util.spec_from_file_location("plain_partition_exploratory", SCRIPT)
ex = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ex
SPEC.loader.exec_module(ex)


# Literal expectations (design ruling (b), 25 Oct): the identity view's store note, and the sentence a plain union's results carry
NOTE = "not recorded for this OUT (identity view): no store check is claimed"
DISCLOSURE = "a15 copies: identity by manifest and pinned description; no store-stats check (not QLever)"
QLEVER_CHECK = "?cmd=stats equal to synthetic in 3 fields at synthetic (store check sha256 " + "a" * 64 + ")"


def sha(body):
    return hashlib.sha256(body).hexdigest()


def fixture(tmp_path):
    ids = [f"synthetic-{i}" for i in range(12)]
    raw = b"".join(json.dumps({"id": q, "base_question_id": q, "split": "test"}).encode() + b"\n" for q in ids)
    qpath = tmp_path / "registered.jsonl"
    qpath.write_bytes(raw)
    groups = [[q for q in ids if int(sha(b"20261022|" + q.encode()), 16) % 2 == i] for i in range(2)]
    assert all(groups)
    lines = {json.loads(line)["id"]: line for line in raw.splitlines()}
    parts = {}
    for i, mine in enumerate(groups):
        name = f"questions-p{i}of2.jsonl"
        body = b"".join(lines[q] + b"\n" for q in ids if q in mine)
        (tmp_path / name).write_bytes(body)
        parts[str(i)] = {"copy": f"copy-{i}", "questions": len(mine), "attempts": 3 * len(mine),
                         "ids_sha256": sha("".join(f"{q}\n" for q in sorted(mine)).encode()),
                         "file": name, "file_sha256": sha(body)}
    record = {"rule": ex.PLAIN_RULE, "salt": ex.SALT, "k": 2, "copies": ["copy-0", "copy-1"],
              "reps": 3, "questions_sha256": sha(raw), "test_questions": len(ids), "partitions": parts}
    rpath = tmp_path / "record.json"
    rpath.write_text(json.dumps(record))
    return ids, qpath, rpath, record


def outs(record, check=NOTE):
    return [ex.Out(Path(f"/synthetic/union-p{i}of2"), {"out": f"/synthetic/union-p{i}of2",
                   "store": {"replica": f"copy-{i}", "identity_check": check}},
                   {"schema": "pilot-manifest/1", "fingerprint": {
                       "questions_sha256": record["partitions"][str(i)]["file_sha256"]}}) for i in range(2)]


def accepted(tmp_path, check=NOTE, identity_note=NOTE):
    ids, qpath, rpath, source = fixture(tmp_path)
    record, pin = ex.read_plain_partition_record(rpath, sha(rpath.read_bytes()))
    paths, normalized = ex.plain_partition_files(record, rpath, qpath, "test")
    assert len(paths) == 2
    graph = next(iter(ex.PRIVATE_REFERENCE_PINS))
    ex.require_plain_private_graph(graph)
    normalized["graph"] = graph
    adapted = ex.plain_union_outs(outs(source, check), normalized, source["partitions"], pin, "/synthetic/union")
    batch = SimpleNamespace(records=[], branch=[])
    frozen = SimpleNamespace(inputs=SimpleNamespace(load_batch=lambda *a: batch),
                             gate=SimpleNamespace(grid_cells=lambda *a: None))
    union = ex.check_union(frozen, adapted, normalized, pin, ids, graph, "test", source["questions_sha256"],
                           identity_note=identity_note)
    assert union.k == len(adapted) == 2
    assert set(union.index_of.values()) == {0, 1}
    return ids, qpath, rpath, source, normalized, pin, adapted, frozen, graph


def test_accepted_plain_union(tmp_path):
    accepted(tmp_path)


def test_wrong_record_sha_and_exact_fields(tmp_path):
    _, _, path, record = fixture(tmp_path)
    with pytest.raises(ex.Refused, match="not the command-line pinned file"):
        ex.read_plain_partition_record(path, "0" * 64)
    for mutation in (lambda d: d.update(extra=True), lambda d: d.pop("reps"),
                     lambda d: d["partitions"]["0"].update(extra=True),
                     lambda d: d["partitions"]["0"].pop("file_sha256")):
        candidate = copy.deepcopy(record)
        mutation(candidate)
        path.write_text(json.dumps(candidate))
        with pytest.raises(ex.Refused, match="extra or missing field|exact partitions"):
            ex.read_plain_partition_record(path, sha(path.read_bytes()))


def test_file_pin_partition_coverage_and_k(tmp_path):
    ids, qpath, path, source, normalized, pin, adapted, frozen, graph = accepted(tmp_path)
    bad = outs(source)
    bad[0].manifest["fingerprint"]["questions_sha256"] = "0" * 64
    with pytest.raises(ex.Refused, match="does not pin a partition questions file"):
        ex.plain_union_outs(bad, normalized, source["partitions"], pin, "/synthetic/union")
    overlap = outs(source)
    overlap[1].manifest["fingerprint"]["questions_sha256"] = overlap[0].manifest["fingerprint"]["questions_sha256"]
    overlap[1].identity["store"]["replica"] = "copy-0"
    with pytest.raises(ex.Refused, match="OUT name does not match partition 0"):
        ex.plain_union_outs(overlap, normalized, source["partitions"], pin, "/synthetic/union")
    duplicate = [outs(source)[0], outs(source)[0]]
    with pytest.raises(ex.Refused, match="not each of 0..1 once"):
        ex.check_union(frozen, ex.plain_union_outs(duplicate, normalized, source["partitions"], pin, "/synthetic/union"),
                       normalized, pin, ids, graph, "test", source["questions_sha256"], identity_note=NOTE)
    # A partition that omits a registered row cannot reproduce its count and digest.
    missing = copy.deepcopy(source)
    missing["partitions"]["0"]["questions"] -= 1
    with pytest.raises(ex.Refused, match="not the producer's rule"):
        ex.plain_partition_files(missing, path, qpath, "test")
    with pytest.raises(ex.Refused, match="one OUT per distinct partition"):
        ex.plain_union_outs(outs(source)[:1], normalized, source["partitions"], pin, "/synthetic/union")
    with pytest.raises(ex.Refused, match="one OUT per distinct partition"):
        ex.plain_union_outs(outs(source), {**normalized, "k": 3}, source["partitions"], pin, "/synthetic/union")


def test_out_names_are_bound_to_the_fingerprint_and_prefix(tmp_path):
    _, _, _, source, normalized, pin, *_ = accepted(tmp_path)
    for names in (("union-p0of2", "other-p1of2"), ("union-p0of2", "union-p0of3"),
                  ("union-p0of2", "union-p0of2"), ("union-p1of2", "union-p0of2")):
        changed = outs(source)
        changed = [dataclasses.replace(o, path=o.path.with_name(name)) for o, name in zip(changed, names)]
        with pytest.raises(ex.Refused, match="OUT name does not match|OUT names must"):
            ex.plain_union_outs(changed, normalized, source["partitions"], pin, "/synthetic/union")
    changed = outs(source)
    changed[0].identity["out"] = "/synthetic/another-out"
    with pytest.raises(ex.Refused, match="deployment identity does not name this partition OUT"):
        ex.plain_union_outs(changed, normalized, source["partitions"], pin, "/synthetic/union")


def test_plain_union_out_spec_refuses_prefix_order_and_k(tmp_path):
    pins = ("a" * 64, "b" * 64)
    batches = [Path(f"/synthetic/union-p{i}of2") for i in range(2)]
    spec = "/synthetic/union:" + ",".join(pins)
    assert ex.parse_plain_union_outs(spec, batches) == ("/synthetic/union", pins)
    for wrong_spec, wrong_batches in (
        ("/synthetic/other:" + ",".join(pins), batches),
        (spec, batches[::-1]),
        (spec, batches[:1]),
        ("/synthetic/union:" + ",".join(pins[:1]), batches),
        ("/synthetic/union:" + ",".join(pins + ("c" * 64,)), batches),
        ("/synthetic/union:" + pins[0] + ",", batches),
    ):
        with pytest.raises(ex.Refused):
            ex.parse_plain_union_outs(wrong_spec, wrong_batches)


def test_manifest_sha_is_checked_from_the_once_read_bytes_before_json(tmp_path):
    out = tmp_path / "union-p0of1"
    out.mkdir()
    manifest = out / "manifest.json"
    manifest.write_bytes(b'{"schema":"pilot-manifest/1"}\n')
    pin = sha(manifest.read_bytes())
    assert ex.Out.read(out, pin).manifest["schema"] == "pilot-manifest/1"
    with pytest.raises(ex.Refused, match="manifest.json sha256 differs"):
        ex.Out.read(out, "0" * 64)
    manifest.write_bytes(b"not JSON")
    with pytest.raises(ex.Refused, match="manifest.json sha256 differs"):
        ex.Out.read(out, pin)
    manifest.unlink()
    with pytest.raises(ex.Refused, match="cannot read pinned manifest.json"):
        ex.Out.read(out, pin)


def test_wrong_copy_and_non_private_graph(tmp_path):
    ids, _, _, source, normalized, pin, adapted, frozen, graph = accepted(tmp_path)
    altered = copy.deepcopy(adapted)
    altered[0].identity["store"]["replica"] = "wrong-copy"
    with pytest.raises(ex.Refused, match="is not partition 0's copy in the record"):
        ex.check_union(frozen, altered, normalized, pin, ids, graph, "test", source["questions_sha256"],
                       identity_note=NOTE)
    with pytest.raises(ex.Refused, match="only for a privately pinned graph"):
        ex.require_plain_private_graph("synthetic-public")


def base_module(tmp_path):
    # (merged with the replay-matrix work: the base is the commit before the plain-partition union was merged, 070b85cf, which has the
    # replay diagnostics sections every results.json now carries; 351ad4a0 predates them)
    base = subprocess.run(["git", "show", "070b85cf:results/e189-exploratory-20261022/exploratory.py"],
                          check=True, capture_output=True).stdout
    old_path = tmp_path / "base.py"
    old_path.write_bytes(base)
    spec = importlib.util.spec_from_file_location("base_plain_comparison", old_path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    return old


def test_without_new_option_matches_base_record_reader_byte_for_byte(tmp_path):
    old = base_module(tmp_path)
    assert ex.RECORD_SCHEMAS == old.RECORD_SCHEMAS
    assert ex.read_record.__code__.co_code == old.read_record.__code__.co_code
    assert ex.check_record.__code__.co_code == old.check_record.__code__.co_code
    # (check_union gained the keyword identity_note, so its bytecode is no longer the base's: without it the union below, and the
    # refusal of the view note in the next test, are the base's byte for byte; test_exploratory.py compares whole analyse calls)
    ids, _, _, source, normalized, pin, adapted, frozen, graph = accepted(tmp_path, check=QLEVER_CHECK, identity_note=None)
    ordinary = dict(normalized, schema="entry180-partition/1")
    rpath = tmp_path / "ordinary.json"
    rpath.write_text(json.dumps(ordinary))
    assert json.dumps(ex.read_record(rpath), sort_keys=True).encode() == json.dumps(
        old.read_record(rpath), sort_keys=True).encode()
    former_outs = [old.Out(o.path, o.identity, o.manifest) for o in adapted]
    current = ex.check_union(frozen, adapted, ordinary, pin, ids, graph, "test", source["questions_sha256"])
    former = old.check_union(frozen, former_outs, ordinary, pin, ids, graph, "test", source["questions_sha256"])
    assert json.dumps(dataclasses.asdict(current), sort_keys=True).encode() == json.dumps(
        dataclasses.asdict(former), sort_keys=True).encode()


def test_the_constants_are_the_rulings_words():
    assert ex.VIEW_STORE_CHECK_NOTE == NOTE
    assert ex.PLAIN_UNION_COPY_IDENTITY == DISCLOSURE
    assert ex.PLAIN_UNION_ENDPOINT_PROOFS == (
        "a15 copies: no store-proofs.json entry is required for any (OUT, endpoint); each endpoint is named by its OUT's "
        "identity and is the pinned endpoint copies' endpoint of that OUT's copy")
    assert ex.IDENTITY_CHECK.fullmatch(QLEVER_CHECK) and not ex.IDENTITY_CHECK.fullmatch(NOTE)


def test_a_plain_union_whose_outs_carry_the_view_note_is_accepted(tmp_path):
    ids, _, _, source, normalized, pin, adapted, frozen, graph = accepted(tmp_path)  # (the fixture's OUTs carry NOTE)
    assert [o.identity["store"]["identity_check"] for o in adapted] == [NOTE, NOTE]
    union = ex.check_union(frozen, adapted, normalized, pin, ids, graph, "test", source["questions_sha256"],
                           identity_note=NOTE)
    assert union.index_of == {"/synthetic/union-p0of2": 0, "/synthetic/union-p1of2": 1}


def test_a_plain_union_refuses_what_is_not_exactly_the_view_note(tmp_path):
    ids, _, _, source, normalized, pin, adapted, frozen, graph = accepted(tmp_path)
    near_misses = [NOTE + " ", " " + NOTE, NOTE.upper(), NOTE[:-1], NOTE[1:], "x" + NOTE, NOTE + "x", "", None, 3,
                   [NOTE], {"note": NOTE}, "not a check at all", QLEVER_CHECK]  # (a valid QLever check is refused too)
    for bad in near_misses:
        altered = copy.deepcopy(adapted)
        altered[1].identity["store"]["identity_check"] = bad
        with pytest.raises(ex.Refused, match=r"/synthetic/union-p1of2: its store copy does not carry the identity view's "
                                              r"store note exactly"):
            ex.check_union(frozen, altered, normalized, pin, ids, graph, "test", source["questions_sha256"],
                           identity_note=NOTE)
    missing = copy.deepcopy(adapted)
    del missing[0].identity["store"]["identity_check"]
    with pytest.raises(ex.Refused, match="its store copy does not carry the identity view's store note exactly"):
        ex.check_union(frozen, missing, normalized, pin, ids, graph, "test", source["questions_sha256"], identity_note=NOTE)
    # every block of a chain, and a parents' copy, are held to the note
    moved = copy.deepcopy(adapted)
    here = moved[0].identity["store"]
    elsewhere = dict(here, identity_check=QLEVER_CHECK)
    moved[0].identity["store_changes"] = [{"from": here, "to": elsewhere}]
    moved[0].identity["store"] = elsewhere
    with pytest.raises(ex.Refused, match="its store copy after its identity's store change 1 does not carry the identity"):
        ex.check_union(frozen, moved, normalized, pin, ids, graph, "test", source["questions_sha256"], identity_note=NOTE)
    parents = copy.deepcopy(adapted)
    parents[0].identity["parents_store"] = {"replica": "copy-0", "identity_check": QLEVER_CHECK}
    with pytest.raises(ex.Refused, match="its parents' store copy does not carry the identity"):
        ex.check_union(frozen, parents, normalized, pin, ids, graph, "test", source["questions_sha256"], identity_note=NOTE)


def test_a_union_without_the_plain_note_still_refuses_the_view_note_with_the_base_message(tmp_path):
    old = base_module(tmp_path)
    ids, _, _, source, normalized, pin, adapted, frozen, graph = accepted(tmp_path)  # OUTs carry NOTE
    ordinary = dict(normalized, schema="entry180-partition/1")
    one = ("{}: its store copy carries no passed identity check as the partition builders record it "
           "(?cmd=stats equal to ... (store check sha256 ...))")
    literal = " | ".join(one.format(f"/synthetic/union-p{i}of2") for i in range(2))
    with pytest.raises(ex.Refused) as current:
        ex.check_union(frozen, adapted, ordinary, pin, ids, graph, "test", source["questions_sha256"])
    assert str(current.value) == literal
    with pytest.raises(old.Refused) as former:
        old.check_union(frozen, [old.Out(o.path, o.identity, o.manifest) for o in adapted], ordinary, pin, ids, graph,
                        "test", source["questions_sha256"])
    assert str(former.value) == literal


def pinned_tables(record, graph, **drop):
    """Synthetic stand-ins for the two pinned private tables (their reading is read_private_table's, tested elsewhere)."""
    copies = [p["copy"] for p in record["partitions"]]
    classes = ex.PrivateTable("classes.json", "1" * 64, {"copies": {c: {"class": "medium"} for c in copies}})
    endpoints = ex.PrivateTable("endpoints.json", "2" * 64, [{"graph": graph, "replica": c} for c in copies])
    return classes, endpoints


def test_a_plain_union_needs_its_copies_in_the_pinned_description(tmp_path):
    ids, _, _, source, normalized, pin, adapted, frozen, graph = accepted(tmp_path)
    reference = {"graph": graph, "reference_sha256": "3" * 64}
    classes, endpoints = pinned_tables(normalized, graph)
    assert ex.require_plain_pinned_copies(normalized, graph, reference, classes, endpoints) is None
    needs = "needs the privately pinned graph's reference pin, private copy classes and private endpoint copies"
    for given in ((None, classes, endpoints), (reference, None, endpoints), (reference, classes, None)):
        with pytest.raises(ex.Refused, match=needs):
            ex.require_plain_pinned_copies(normalized, graph, *given)
    unlisted = "must be listed in the pinned copy classes and endpoint copies: not listed for partitions "
    missing_class = ex.PrivateTable("classes.json", "1" * 64, {"copies": {"copy-0": {"class": "medium"}}})
    with pytest.raises(ex.Refused, match=unlisted + r"\[1\]$"):
        ex.require_plain_pinned_copies(normalized, graph, reference, missing_class, endpoints)
    missing_endpoint = ex.PrivateTable("endpoints.json", "2" * 64, [{"graph": graph, "replica": "copy-1"}])
    with pytest.raises(ex.Refused, match=unlisted + r"\[0\]$"):
        ex.require_plain_pinned_copies(normalized, graph, reference, classes, missing_endpoint)
    other_graph = ex.PrivateTable("endpoints.json", "2" * 64, [{"graph": "another", "replica": c} for c in ("copy-0", "copy-1")])
    with pytest.raises(ex.Refused, match=unlisted + r"\[0, 1\]$"):
        ex.require_plain_pinned_copies(normalized, graph, reference, classes, other_graph)
    with pytest.raises(ex.Refused) as refusal:  # no copy name, in any refusal
        ex.require_plain_pinned_copies(normalized, graph, reference, missing_class, endpoints)
    assert "copy-" not in str(refusal.value)
