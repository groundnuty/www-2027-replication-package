"""Synthetic checks for a WEIGHTED plain partition record (the plain record plus ``weights`` and ``weights_reduced``, and a
rule string of its own): its reader, the one partition rule every membership decision takes, and that every mod-K input is
what it was. The reference below is written here from the producer's rule (sha256 of the salt and the id; the slot is the hash
mod the sum of the weights divided by their gcd; the partition is the weight interval containing the slot), never from the
script's own functions. Synthetic ids and fixtures only."""

import ast
import copy
import dataclasses
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

import test_exploratory as TE
from test_exploratory import _private_a15_root, _store_refusals_carry_no_number  # noqa: F401  (autouse fixtures of that module)

ex = TE.ex
F = TE.F

WEIGHTED_RULE = ("slot = int(sha256('20261022|' + question id).hexdigest(), 16) mod sum(w), w = the record's weights divided by "
                 "their gcd; partition p is the weight interval containing slot; partition p runs on copy p of the record's list")
MODK_RULE = "int(sha256('20261022|' + question id).hexdigest(), 16) mod K; partition p runs on copy p of the record's list"
NOTE = "not recorded for this OUT (identity view): no store check is claimed"
WEIGHT_VECTORS = ([2, 3, 5], [1, 1, 2], [7, 3, 5], [6, 3], [4, 6, 10], [1, 1], [2, 2, 2], [9, 1], [1, 2, 3, 4], [5, 5, 5, 5])


def sha(body):
    return hashlib.sha256(body).hexdigest()


def ref_reduce(weights):
    g = math.gcd(*weights)
    return [w // g for w in weights]


def ref_partition(qid, k, weights=None):
    """The producer's partition_of, restated: no weights is the hash mod K; with weights the interval containing the slot."""
    h = int(sha(("20261022|" + qid).encode()), 16)
    if weights is None:
        return h % k
    w = ref_reduce(weights)
    slot = h % sum(w)
    total = 0
    for i, x in enumerate(w):
        total += x
        if slot < total:
            return i
    raise AssertionError("unreachable")


def make(tmp_path, weights, *, n=90, files="weighted", file_weights=None):
    """A weighted record over ``n`` synthetic ids: the registered questions file, one partition questions file per partition
    (by the weighted rule, or by the mod-K rule when ``files`` is "modk", or by ``file_weights``) and the record file."""
    k = len(weights)
    ids = [f"synthetic-{i}" for i in range(n)]
    raw = b"".join(json.dumps({"id": q, "base_question_id": q, "split": "test"}).encode() + b"\n" for q in ids)
    qpath = tmp_path / "registered.jsonl"
    qpath.write_bytes(raw)
    lines = {json.loads(line)["id"]: line for line in raw.splitlines()}
    if files == "modk":
        place = lambda q: ref_partition(q, k)  # noqa: E731
    else:
        place = lambda q: ref_partition(q, k, file_weights or weights)  # noqa: E731
    groups = [[q for q in ids if place(q) == i] for i in range(k)]
    assert all(groups), "the fixture must put a question in every partition"
    parts = {}
    for i, mine in enumerate(groups):
        name = f"questions-p{i}of{k}.jsonl"
        body = b"".join(lines[q] + b"\n" for q in ids if q in mine)
        (tmp_path / name).write_bytes(body)
        parts[str(i)] = {"copy": f"copy-{i}", "questions": len(mine), "attempts": 3 * len(mine),
                         "ids_sha256": sha("".join(f"{q}\n" for q in sorted(mine)).encode()),
                         "file": name, "file_sha256": sha(body)}
    record = {"rule": WEIGHTED_RULE, "salt": "20261022|", "k": k, "copies": [f"copy-{i}" for i in range(k)], "reps": 3,
              "questions_sha256": sha(raw), "test_questions": n, "partitions": parts, "weights": list(weights),
              "weights_reduced": ref_reduce(weights)}
    rpath = tmp_path / "record.json"
    rpath.write_text(json.dumps(record))
    return ids, qpath, rpath, record


def outs(record, check=NOTE):
    k = record["k"]
    return [ex.Out(Path(f"/synthetic/union-p{i}of{k}"), {"out": f"/synthetic/union-p{i}of{k}",
                   "store": {"replica": f"copy-{i}", "identity_check": check}},
                   {"schema": "pilot-manifest/1", "fingerprint": {
                       "questions_sha256": record["partitions"][str(i)]["file_sha256"]}}) for i in range(k)]


def batches(ids, k, place):
    """One batch per OUT path: one record per id, run id "run-<id>", in the partition ``place`` gives the id."""
    return {f"/synthetic/union-p{i}of{k}": SimpleNamespace(
        records=[{"question_id": q, "run_id": f"run-{q}"} for q in ids if place(q) == i], branch=[]) for i in range(k)}


def stub(by_path):
    return SimpleNamespace(inputs=SimpleNamespace(load_batch=lambda path, *a: by_path[str(path)]),
                           gate=SimpleNamespace(grid_cells=lambda *a: None))


def accepted(tmp_path, weights, **kw):
    ids, qpath, rpath, source = make(tmp_path, weights, **kw)
    k = len(weights)
    record, pin = ex.read_plain_partition_record(rpath, sha(rpath.read_bytes()))
    paths, normalized = ex.plain_partition_files(record, rpath, qpath, "test")
    assert len(paths) == k
    graph = next(iter(ex.PRIVATE_REFERENCE_PINS))
    normalized["graph"] = graph
    adapted = ex.plain_union_outs(outs(source), normalized, source["partitions"], pin, "/synthetic/union")
    by_path = batches(ids, k, lambda q: ref_partition(q, k, weights))
    union = ex.check_union(stub(by_path), adapted, normalized, pin, ids, graph, "test", source["questions_sha256"],
                           identity_note=NOTE)
    return SimpleNamespace(ids=ids, qpath=qpath, rpath=rpath, source=source, record=record, pin=pin, normalized=normalized,
                           adapted=adapted, graph=graph, union=union, k=k, weights=weights)


def check_with(a, place, adapted=None, normalized=None):
    return ex.check_union(stub(batches(a.ids, a.k, place)), adapted or a.adapted, normalized or a.normalized, a.pin, a.ids,
                          a.graph, "test", a.source["questions_sha256"], identity_note=NOTE)


def rewrite(path, record):
    path.write_text(json.dumps(record))
    return sha(path.read_bytes())


# --- the rule's words and the one partition rule ---------------------------------------------------------------------------

def test_the_rule_constants_are_the_producers_words():
    assert ex.WEIGHTED_PLAIN_RULE == WEIGHTED_RULE
    assert ex.PLAIN_RULE == MODK_RULE
    assert ex.WEIGHTED_PLAIN_RECORD_FIELDS == ex.PLAIN_RECORD_FIELDS | {"weights", "weights_reduced"}
    assert ex.PLAIN_RECORD_FIELDS == {"rule", "salt", "k", "copies", "reps", "questions_sha256", "test_questions", "partitions"}


def test_partition_of_equals_the_reference_on_many_ids_and_weight_vectors():
    rng = random.Random(20261026)
    ids = [f"synthetic-{rng.getrandbits(48):x}-{i}" for i in range(5200)]
    for weights in WEIGHT_VECTORS:
        k = len(weights)
        got = [ex.partition_of(q, k, weights) for q in ids]
        assert got == [ref_partition(q, k, weights) for q in ids], weights
        assert set(got) == set(range(k)), weights  # every interval is used by these ids
        counts = [got.count(i) for i in range(k)]
        reduced = ref_reduce(weights)
        for i in range(k):  # and in proportion to its width (a loose bound: 5200 ids)
            assert abs(counts[i] / len(ids) - reduced[i] / sum(reduced)) < 0.05, (weights, counts)
    # no weights: the hash mod K, for every K
    for k in (2, 3, 5, 7):
        assert [ex.partition_of(q, k) for q in ids[:600]] == [ref_partition(q, k) for q in ids[:600]]
        assert [ex.partition_of(q, k, None) for q in ids[:600]] == [ref_partition(q, k) for q in ids[:600]]


def test_the_interval_rule_is_hand_checked_on_a_slot():
    # slot = h mod 10 for [2, 3, 5]: partition 0 is slots 0-1, 1 is 2-4, 2 is 5-9 (every boundary, in the right order)
    h = {q: int(sha(("20261022|" + q).encode()), 16) for q in (f"hand-{i}" for i in range(400))}
    seen = {}
    for q, value in h.items():
        seen.setdefault(value % 10, q)
    assert set(seen) == set(range(10))
    for slot, q in seen.items():
        assert ex.partition_of(q, 3, [2, 3, 5]) == (0 if slot < 2 else 1 if slot < 5 else 2)
        assert ex.partition_of(q, 3, [4, 6, 10]) == (0 if slot < 2 else 1 if slot < 5 else 2)  # the gcd (2) is divided out
        assert ex.partition_of(q, 3, [5, 3, 2]) == (0 if slot < 5 else 1 if slot < 8 else 2)  # the order is the record's


def test_reduced_equal_weights_coincide_with_mod_k_and_unequal_ones_do_not():
    ids = [f"synthetic-{i}" for i in range(3000)]
    for weights in ([1, 1, 1], [4, 4, 4], [3, 3, 3]):
        assert [ex.partition_of(q, 3, weights) for q in ids] == [ex.partition_of(q, 3) for q in ids]
    assert [ex.partition_of(q, 3, [2, 3, 5]) for q in ids] != [ex.partition_of(q, 3) for q in ids]
    assert [ex.partition_of(q, 2, [1, 2]) for q in ids] != [ex.partition_of(q, 2) for q in ids]


@pytest.mark.parametrize("weights, k", [([2, 3], 3), ([2, 3, 5], 2), ([2], 1), ([], 0), ([1, 1], 3)])
def test_partition_of_needs_one_weight_per_partition(weights, k):
    with pytest.raises(ex.Refused, match="weights"):
        ex.partition_of("synthetic-1", k, weights)


@pytest.mark.parametrize("weights", [[0, 1], [1, -1], [True, 2], [2.0, 3], [2, 3.5], ["2", 3], [None, 1], 5, "23", {"a": 1},
                                     [1], []])
def test_reduce_weights_refuses_what_is_not_two_or_more_positive_integers(weights):
    with pytest.raises(ex.Refused, match="weights"):
        ex.reduce_weights(weights)
    assert ex.reduce_weights([6, 3, 9]) == [2, 1, 3] and ex.reduce_weights((4, 6)) == [2, 3]


# --- the reader ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("weights", WEIGHT_VECTORS)
def test_the_reader_accepts_a_weighted_record(tmp_path, weights):
    _, _, rpath, source = make(tmp_path, weights)
    record, pin = ex.read_plain_partition_record(rpath, sha(rpath.read_bytes()))
    assert record == source and pin == sha(rpath.read_bytes())
    assert record["weights_reduced"] == ref_reduce(weights)


READER_REFUSALS = [
    # (name, mutation, the message the reader gives)
    ("weights not a list: int", lambda d: d.update(weights=5), "weights are not one positive integer per partition"),
    ("weights not a list: str", lambda d: d.update(weights="235"), "weights are not one positive integer per partition"),
    ("weights not a list: dict", lambda d: d.update(weights={"0": 2}), "weights are not one positive integer per partition"),
    ("weights not a list: null", lambda d: d.update(weights=None), "weights are not one positive integer per partition"),
    ("weights one short", lambda d: d.update(weights=d["weights"][:-1]), "weights are not one positive integer per partition"),
    ("weights one long", lambda d: d.update(weights=d["weights"] + [1]), "weights are not one positive integer per partition"),
    ("weights zero", lambda d: d["weights"].__setitem__(0, 0), "weights are not one positive integer per partition"),
    ("weights negative", lambda d: d["weights"].__setitem__(1, -3), "weights are not one positive integer per partition"),
    ("weights bool", lambda d: d["weights"].__setitem__(0, True), "weights are not one positive integer per partition"),
    ("weights float", lambda d: d["weights"].__setitem__(2, 5.0), "weights are not one positive integer per partition"),
    ("weights string entry", lambda d: d["weights"].__setitem__(0, "2"), "weights are not one positive integer per partition"),
    ("reduced not reduced", lambda d: d.update(weights_reduced=list(d["weights"])), "weights_reduced are not"),
    ("reduced other numbers", lambda d: d.update(weights_reduced=[1, 1, 1]), "weights_reduced are not"),
    ("reduced reordered", lambda d: d.update(weights_reduced=d["weights_reduced"][::-1]), "weights_reduced are not"),
    ("reduced one short", lambda d: d.update(weights_reduced=d["weights_reduced"][:-1]), "weights_reduced are not"),
    ("reduced not a list", lambda d: d.update(weights_reduced=7), "weights_reduced are not"),
    ("reduced null", lambda d: d.update(weights_reduced=None), "weights_reduced are not"),
    ("reduced float entries", lambda d: d.update(weights_reduced=[float(w) for w in d["weights_reduced"]]),
     "weights_reduced are not"),
    ("reduced bool entry", lambda d: d.update(weights_reduced=[True, 3, 5]), "weights_reduced are not"),
    ("mod-K rule with weights", lambda d: d.update(rule=MODK_RULE), "extra or missing field"),
    ("weighted rule without weights", lambda d: (d.pop("weights"), d.pop("weights_reduced")),
     "rule or registered question pin is invalid"),
    ("weighted rule with only weights", lambda d: d.pop("weights_reduced"), "extra or missing field"),
    ("weighted rule with only reduced", lambda d: d.pop("weights"), "extra or missing field"),
    ("extra unknown field", lambda d: d.update(extra=True), "extra or missing field"),
    ("unknown rule with weights", lambda d: d.update(rule="mod something else"), "extra or missing field"),
    ("rule with a trailing space", lambda d: d.update(rule=WEIGHTED_RULE + " "), "extra or missing field"),
    ("rule one letter changed", lambda d: d.update(rule=WEIGHTED_RULE.replace("interval", "intervals")), "extra or missing field"),
    ("rule with another salt", lambda d: d.update(rule=WEIGHTED_RULE.replace("20261022", "20261023")), "extra or missing field"),
    ("another salt", lambda d: d.update(salt="20261023|"), "rule or registered question pin is invalid"),
    ("reps not 3", lambda d: d.update(reps=2), "rule or registered question pin is invalid"),
    ("k = 1", lambda d: d.update(k=1), "exact partitions and copies"),
]


@pytest.mark.parametrize("name, mutation, message", READER_REFUSALS, ids=[r[0] for r in READER_REFUSALS])
def test_the_reader_refuses(tmp_path, name, mutation, message):
    _, _, rpath, source = make(tmp_path, [4, 6, 10])  # gcd 2: the reduced weights [2, 3, 5] are not the weights
    candidate = copy.deepcopy(source)
    mutation(candidate)
    with pytest.raises(ex.Refused, match=message):
        ex.read_plain_partition_record(rpath, rewrite(rpath, candidate))


def test_a_mod_k_record_keeps_its_exact_field_set(tmp_path):
    # the mod-K record of test_plain_partition_union.py's builder: accepted as before, refused with either weights field
    # added (the base's message), and a stray weights key is never read by plain_partition_files
    ids = [f"synthetic-{i}" for i in range(40)]
    raw = b"".join(json.dumps({"id": q, "base_question_id": q, "split": "test"}).encode() + b"\n" for q in ids)
    (tmp_path / "registered.jsonl").write_bytes(raw)
    lines = {json.loads(line)["id"]: line for line in raw.splitlines()}
    parts = {}
    for i in range(2):
        mine = [q for q in ids if ref_partition(q, 2) == i]
        body = b"".join(lines[q] + b"\n" for q in ids if q in mine)
        (tmp_path / f"questions-p{i}of2.jsonl").write_bytes(body)
        parts[str(i)] = {"copy": f"copy-{i}", "questions": len(mine), "attempts": 3 * len(mine),
                         "ids_sha256": sha("".join(f"{q}\n" for q in sorted(mine)).encode()),
                         "file": f"questions-p{i}of2.jsonl", "file_sha256": sha(body)}
    record = {"rule": ex.PLAIN_RULE, "salt": ex.SALT, "k": 2, "copies": ["copy-0", "copy-1"], "reps": 3,
              "questions_sha256": sha(raw), "test_questions": len(ids), "partitions": parts}
    rpath = tmp_path / "record.json"
    got, pin = ex.read_plain_partition_record(rpath, rewrite(rpath, record))
    assert got == record
    paths, normalized = ex.plain_partition_files(got, rpath, tmp_path / "registered.jsonl", "test")
    assert "weights" not in normalized and ex.union_weights(normalized) is None
    stray = dict(record, weights=[1, 3], weights_reduced=[1, 3])
    for candidate in (stray, dict(record, weights=[1, 3]), dict(record, weights_reduced=[1, 3])):
        with pytest.raises(ex.Refused, match="the plain partition record has an extra or missing field"):
            ex.read_plain_partition_record(rpath, rewrite(rpath, candidate))
    # a mod-K record dict that carries a weights key (never read from a file): the files are still placed mod K
    assert ex.plain_partition_files(stray, rpath, tmp_path / "registered.jsonl", "test")[1] == normalized


# --- the union: files, membership, ledger -------------------------------------------------------------------------------------

@pytest.mark.parametrize("weights", WEIGHT_VECTORS)
def test_a_weighted_union_is_accepted_and_every_question_is_in_the_right_partition(tmp_path, weights):
    a = accepted(tmp_path, weights)
    assert isinstance(a.union, ex.WeightedUnion) and a.union.weights == tuple(weights) and a.union.k == len(weights)
    assert a.normalized["weights"] == list(weights)
    for i in range(a.k):
        mine = [q for q in a.ids if ref_partition(q, a.k, weights) == i]
        assert a.normalized["partitions"][i]["n_questions"] == len(mine)
        assert a.normalized["partitions"][i]["question_id_sha256"] == sorted(sha(q.encode()) for q in mine)
        assert a.record["partitions"][str(i)]["ids_sha256"] == ex.ids_digest(mine)
    assert set(a.union.index_of.values()) == set(range(a.k))
    assert {run: p for run, p in a.union.run_partition.items()} == {f"run-{q}": ref_partition(q, a.k, weights) for q in a.ids}


def test_the_weighted_files_are_not_the_mod_k_files(tmp_path):
    ids, qpath, rpath, source = make(tmp_path, [2, 3, 5])
    modk = {i: [q for q in ids if ref_partition(q, 3) == i] for i in range(3)}
    weighted = {i: [q for q in ids if ref_partition(q, 3, [2, 3, 5]) == i] for i in range(3)}
    assert modk != weighted
    assert [source["partitions"][str(i)]["questions"] for i in range(3)] == [len(weighted[i]) for i in range(3)]


def test_a_weighted_record_whose_files_were_built_by_the_mod_k_rule_is_refused(tmp_path):
    ids, qpath, rpath, source = make(tmp_path, [2, 3, 5], files="modk")
    record, _ = ex.read_plain_partition_record(rpath, sha(rpath.read_bytes()))  # the record itself is well formed
    with pytest.raises(ex.Refused, match="partition [0-9] is not the producer's rule"):
        ex.plain_partition_files(record, rpath, qpath, "test")


def test_files_built_with_other_weights_than_the_records_are_refused(tmp_path):
    ids, qpath, rpath, source = make(tmp_path, [2, 3, 5], file_weights=[5, 3, 2])
    record, _ = ex.read_plain_partition_record(rpath, sha(rpath.read_bytes()))
    with pytest.raises(ex.Refused, match="is not the producer's rule"):
        ex.plain_partition_files(record, rpath, qpath, "test")
    # the same files are the record's once the record's weights are theirs
    ids, qpath, rpath, source = make(tmp_path, [5, 3, 2])
    record, _ = ex.read_plain_partition_record(rpath, sha(rpath.read_bytes()))
    assert len(ex.plain_partition_files(record, rpath, qpath, "test")[0]) == 3


def test_an_equal_weights_record_places_questions_as_mod_k_does(tmp_path):
    a = accepted(tmp_path, [2, 2, 2])
    assert all(ref_partition(q, 3, [2, 2, 2]) == ref_partition(q, 3) for q in a.ids)
    assert a.normalized["weights"] == [2, 2, 2] and isinstance(a.union, ex.WeightedUnion)


def test_a_record_in_the_wrong_partition_is_refused_at_the_membership_check(tmp_path):
    a = accepted(tmp_path, [2, 3, 5])
    place = lambda q: ref_partition(q, 3, [2, 3, 5])  # noqa: E731
    check_with(a, place)
    victim = next(q for q in a.ids if place(q) == 1)
    wrong = lambda q: 2 if q == victim else place(q)  # noqa: E731
    with pytest.raises(ex.Refused, match=rf"partition 2\): record run-{victim}'s question is partition 1's"):
        check_with(a, wrong)
    # the mod-K placement is a different placement for the same ids: refused at those ids, naming the weighted partition
    assert any(ref_partition(q, 3) != place(q) for q in a.ids)
    with pytest.raises(ex.Refused, match=r"partition problems"):
        check_with(a, lambda q: ref_partition(q, 3))
    # a record that is in two partitions is still refused (the weights change nothing there)
    twice = batches(a.ids, 3, place)
    twice["/synthetic/union-p0of3"].records.append(dict(twice["/synthetic/union-p1of3"].records[0]))
    with pytest.raises(ex.Refused, match="partition problems"):
        ex.check_union(stub(twice), a.adapted, a.normalized, a.pin, a.ids, a.graph, "test", a.source["questions_sha256"],
                       identity_note=NOTE)


def test_the_record_check_takes_the_weights_and_keeps_the_base_checks(tmp_path):
    a = accepted(tmp_path, [2, 3, 5])
    place = lambda q: ref_partition(q, 3, [2, 3, 5])  # noqa: E731
    # a normalized record whose weights are not the ones its partitions were made by: refused by the rule's own check
    for weights in ([5, 3, 2], [3, 5, 2], [1, 1, 1], [2, 3, 6]):
        other = dict(a.normalized, weights=list(weights))
        with pytest.raises(ex.Refused, match=r"partition 0 is not the rule's \(sha256\('20261022\|' \+ id\) mod the sum of the "
                                              r"weights"):
            check_with(a, place, normalized=other)
    # the base checks of check_record, with the base's messages, are still made for a weighted record
    for change, message in ((dict(salt="x"), "the record is of"), (dict(split="dev"), "the record is of"),
                            (dict(all_ids_sha256="0" * 64), "question set is not the analysed split's"),
                            (dict(n_questions=1), "question set is not the analysed split's"),
                            (dict(k=True), "the record's K is True")):
        with pytest.raises(ex.Refused, match=message):
            ex.check_record_weighted(dict(a.normalized, **change), a.ids, a.graph, "test", a.source["questions_sha256"],
                                     (2, 3, 5))
        base = ex.check_record
        with pytest.raises(ex.Refused) as former:
            base(dict(a.normalized, **change), a.ids, a.graph, "test", a.source["questions_sha256"])
        with pytest.raises(ex.Refused) as latter:
            ex.check_record_weighted(dict(a.normalized, **change), a.ids, a.graph, "test", a.source["questions_sha256"],
                                     (2, 3, 5))
        assert str(former.value) == str(latter.value)
    assert ex.check_record_weighted(a.normalized, a.ids, a.graph, "test", a.source["questions_sha256"], (2, 3, 5)) == 3


def test_a_ledger_row_is_checked_by_the_weighted_rule(tmp_path):
    a = accepted(tmp_path, [2, 3, 5])
    rows = [SimpleNamespace(run_id=f"run-{q}", source="s", use="u", question_id=q) for q in a.ids]
    assert ex.check_union_ledger(rows, a.union) is None
    victim = next(q for q in a.ids if ref_partition(q, 3, [2, 3, 5]) != ref_partition(q, 3))
    bad = [SimpleNamespace(run_id=f"run-{victim}", source="s", use="u", question_id=victim)]
    assert a.union.run_partition[f"run-{victim}"] == ref_partition(victim, 3, [2, 3, 5])
    # the row's record is in the weighted partition, so a row naming another question of another partition is refused
    other = next(q for q in a.ids if ref_partition(q, 3, [2, 3, 5]) != ref_partition(victim, 3, [2, 3, 5]))
    bad = [SimpleNamespace(run_id=f"run-{victim}", source="s", use="u", question_id=other)]
    with pytest.raises(ex.Refused, match=rf"its question is partition {ref_partition(other, 3, [2, 3, 5])}'s, its record partition "
                                          rf"{ref_partition(victim, 3, [2, 3, 5])}'s"):
        ex.check_union_ledger(bad, a.union)
    # a row whose question is in the same weighted partition as its record, but a different mod-K one, is accepted
    twin = next(q for q in a.ids if q != victim and ref_partition(q, 3, [2, 3, 5]) == ref_partition(victim, 3, [2, 3, 5])
                and ref_partition(q, 3) != ref_partition(victim, 3))
    assert ex.check_union_ledger([SimpleNamespace(run_id=f"run-{victim}", source="s", use="u", question_id=twin)], a.union) is None


def test_the_subset_functions_take_the_weights(tmp_path):
    """check_bases_whole and subset_block are reached by a --partitions subset, which a plain union never takes (prepare refuses
    it); both are given the weights all the same, so a union object that carries weights is never read by the mod-K rule."""
    a = accepted(tmp_path, [2, 3, 5])
    union = dataclasses.replace(a.union, subset=(0, 2))
    weighted = lambda q: ref_partition(q, 3, [2, 3, 5]) in (0, 2)  # noqa: E731
    modk = lambda q: ref_partition(q, 3) in (0, 2)  # noqa: E731
    pool = [f"pool-{i}" for i in range(400)]
    # a base question of two ids that the weighted rule keeps together but the mod-K rule splits (accepted), and one the other way
    together = next((x, y) for x in pool for y in pool if x < y and weighted(x) == weighted(y) and modk(x) != modk(y))
    split = next((x, y) for x in pool for y in pool if x < y and weighted(x) != weighted(y) and modk(x) == modk(y))

    def questions(pair):
        return {q: {"split": "test", "base_question_id": "base"} for q in pair}

    assert ex.check_bases_whole(questions(together), "test", union) is None
    with pytest.raises(ex.Refused, match="base questions have questions both inside and outside the subset"):
        ex.check_bases_whole(questions(split), "test", union)
    # subset_block's cluster cut: two questions in one cluster, in and out of the subset by the weighted rule
    for pair, cut in ((together, 0), (split, 1)):
        loaded = SimpleNamespace(questions={q: {"split": "test"} for q in pair}, design=list(pair))
        cluster_map = SimpleNamespace(clusters={q: "c1" for q in pair})
        union_n = dataclasses.replace(union, record=dict(union.record, n_questions=2))
        assert ex.subset_block(union_n, loaded, cluster_map, "test")["clusters_cut"] == cut


def test_every_partition_of_call_site_takes_the_weights_but_check_record():
    """A wiring check on the whole script: each call of partition_of passes a third argument (the weights), except the one in
    check_record, which serves mod-K and every other union kind unchanged (the weighted records go through
    check_record_weighted). The call sites of 2e860be9 were check_record, plain_partition_files, check_union (twice),
    check_union_ledger (twice), check_bases_whole, subset_block and the design filter ``keep`` in prepare."""
    tree = ast.parse(Path(ex.__file__).read_text(encoding="utf-8"))
    sites = {}

    def walk(node, owner):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call) and getattr(child.func, "id", None) == "partition_of":
                sites.setdefault(owner, []).append(len(child.args) + len(child.keywords))
            walk(child, child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else owner)

    walk(tree, "<module>")
    assert sorted(sites) == ["check_bases_whole", "check_record", "check_record_weighted", "check_union", "check_union_ledger",
                             "keep", "plain_partition_files", "subset_block"]
    assert sites["check_record"] == [2]
    assert {name: calls for name, calls in sites.items() if name != "check_record"} == {
        "check_bases_whole": [3], "check_record_weighted": [3], "check_union": [3, 3], "check_union_ledger": [3, 3],
        "keep": [3], "plain_partition_files": [3], "subset_block": [3]}


def test_a_mod_k_union_is_what_it_was(tmp_path):
    """Nothing of a mod-K union changes shape: no weights field in the Union, no weights key in the normalized record."""
    assert "weights" not in {f.name for f in dataclasses.fields(ex.Union)}
    assert ex.Union.weights is None
    assert {f.name for f in dataclasses.fields(ex.WeightedUnion)} == {f.name for f in dataclasses.fields(ex.Union)} | {"weights"}
    assert ex.union_weights({"schema": "entry180-partition/1", "weights": [1, 2]}) is None  # never read from another schema
    assert ex.union_weights({"schema": ex.PLAIN_PARTITION_SCHEMA}) is None
    assert ex.union_weights({"schema": ex.PLAIN_PARTITION_SCHEMA, "weights": [2, 3]}) == (2, 3)
    assert ex.partition_of("synthetic-1", 3) == ref_partition("synthetic-1", 3)


# --- the whole call (prepare and run), on test_exploratory's synthetic plain union with a weighted record ------------------

WEIGHTS_2 = [6, 3]  # gcd 3: reduced [2, 1]; the fixture's three questions fall (t02-q3) | (t01-q1, t01-q2), not as mod 2 does


def weighted_inputs(tmp_path, monkeypatch, weights=WEIGHTS_2):
    source = TE.ROOT / "tests" / "fixtures" / "analysis" / "plain-run"
    rows = [json.loads(line) for line in (source / "questions.jsonl").read_text().splitlines()]
    monkeypatch.setattr(ex, "REGISTERED_SPLIT", "dev")
    registered = tmp_path / "registered.jsonl"
    registered.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    groups = tuple(tuple(q for q in ("t01-q1", "t01-q2", "t02-q3") if ref_partition(q, 2, weights) == i) for i in range(2))
    assert groups == (("t02-q3",), ("t01-q1", "t01-q2"))
    assert groups != TE.PLAIN_GROUPS  # the weighted rule places them otherwise than mod 2
    record = {"rule": WEIGHTED_RULE, "salt": ex.SALT, "k": 2, "copies": ["copy-0", "copy-1"], "reps": 3,
              "questions_sha256": sha(registered.read_bytes()), "test_questions": 3, "partitions": {},
              "weights": list(weights), "weights_reduced": ref_reduce(weights)}
    bodies = []
    for i, group in enumerate(groups):
        body = b"".join(json.dumps(row).encode() + b"\n" for row in rows if row["id"] in group)
        name = f"questions-p{i}of2.jsonl"
        (tmp_path / name).write_bytes(body)
        bodies.append(body)
        record["partitions"][str(i)] = {"copy": f"copy-{i}", "questions": len(group), "attempts": 3 * len(group),
                                         "ids_sha256": TE._digest(group), "file": name, "file_sha256": sha(body)}
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record))
    map_record = {"graph": "rhea-test", "store": "rhea-test", "path": "synthetic", "sha256": "synthetic",
                  "questions_sha256": record["questions_sha256"]}
    monkeypatch.setattr(ex, "require_plain_private_graph", lambda graph: None if graph == "rhea-test" else
                        (_ for _ in ()).throw(ex.Refused("wrong synthetic graph")))
    private = TE._plain_union_private(tmp_path, monkeypatch, ("copy-0", "copy-1"))
    monkeypatch.setattr(TE, "PLAIN_GROUPS", groups)  # the helpers below build the OUTs of these groups
    return {"source": source, "rows": rows, "registered": registered, "record_path": record_path, "bodies": bodies,
            "voided": TE._voided(tmp_path / "voided.jsonl"), "map_record": map_record, "groups": groups, **private}


def test_prepare_and_run_accept_a_weighted_plain_union_end_to_end(tmp_path, monkeypatch):
    inputs = weighted_inputs(tmp_path, monkeypatch)
    outs_ = TE._plain_union_outs(tmp_path, inputs, "synthetic-union")
    pins = TE._plain_union_pins(outs_)
    prefix = str(tmp_path / "synthetic-union")
    _, union, context = ex.prepare(F, TE._plain_union_call(inputs, outs_, prefix, pins), map_record=inputs["map_record"])
    assert isinstance(union, ex.WeightedUnion) and union.weights == tuple(WEIGHTS_2) and union.k == 2
    assert [o["partition"] for o in context["outs"]] == [0, 1]
    # each question's partition is the weighted rule's, which the OUTs' records carry (one OUT per partition of the record)
    assert {q: union.run_partition[run] for run, q in _runs_by_question(outs_).items()} == {
        "t02-q3": 0, "t01-q1": 1, "t01-q2": 1}
    result = ex.run(F, TE._plain_union_call(inputs, outs_, prefix, pins), map_record=inputs["map_record"])
    assert result["exploratory"]["partition_union"]["k"] == 2 and result["arms"]


def _runs_by_question(outs_):
    found = {}
    for out in outs_:
        for path in (out / "records").glob("*.json"):
            record = json.loads(path.read_text())
            found[record["run_id"]] = record["question_id"]
    return found


def test_prepare_refuses_a_weighted_union_whose_outs_are_the_mod_k_ones(tmp_path, monkeypatch):
    """The OUTs here are those of the mod-2 groups, run against a record whose partitions are the weighted ones: the OUT of
    partition 0 holds questions the weighted rule puts in partition 1 (and the other way), so the union is refused."""
    inputs = weighted_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(TE, "PLAIN_GROUPS", (("t01-q1",), ("t01-q2", "t02-q3")))  # the OUT builder takes mod 2's groups
    outs_ = TE._plain_union_outs(tmp_path, inputs, "synthetic-union")
    for i, body in enumerate(inputs["bodies"]):  # the manifests pin the record's partition files, as they do in a real union
        manifest = json.loads((outs_[i] / "manifest.json").read_text())
        manifest["fingerprint"]["questions_sha256"] = sha(body)
        (outs_[i] / "manifest.json").write_text(json.dumps(manifest))
    pins = TE._plain_union_pins(outs_)
    with pytest.raises(ex.Refused) as refusal:
        ex.prepare(F, TE._plain_union_call(inputs, outs_, str(tmp_path / "synthetic-union"), pins),
                   map_record=inputs["map_record"])
    text = str(refusal.value)  # the records of each OUT are in the other partition's weighted interval
    assert "partition problems" in text and "(partition 0): record" in text and "(partition 1): record" in text
    assert "question is partition 1's" in text and "question is partition 0's" in text
