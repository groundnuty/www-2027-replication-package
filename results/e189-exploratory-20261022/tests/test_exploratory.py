"""Tests of the exploratory interval script (change-log entry 189 items 5 and 7), on the repository's grid-full fixture
(tests/fixtures/analysis/grid-full: two models, three prompt cells and two branch cells, one repetition of four dev
questions, store "s01") split into partition OUTs by entry 179's rule, and on toys built here.

Hand-derived facts the tests rely on (never the script's output):
- entry 179's rule, sha256("20261022|" + id) mod 3, puts the fixture's dev questions s01-pass in partition 0, s01-flag
  and s01-abstain in 1, s01-noquery in 2 (computed once by hand with hashlib, written below);
- in the fixture, Qwen's natural ("none") attempts are right on s01-noquery and s01-pass and wrong on s01-flag (not an
  exact match) and s01-abstain (abstained): per question 0, 0, 1, 1 in the index's order abstain, flag, noquery, pass;
- with s01-pass and s01-noquery in one cluster (they are in different partitions) the clusters in index order are
  abstain, flag and {noquery, pass}; the resample rows (1, 0, 2), (2, 1, 0), (0, 2, 1) over them weight the questions
  (1, 0, 2, 2), (2, 1, 0, 0), (0, 2, 1, 1), so accuracy's draws are 4/5, 0/3 and 2/4, and its 95% percentile interval
  over three draws (inverted-CDF quantiles at 0.025 and 0.975) is [0.0, 0.8] around the point estimate 2/4;
- a toy of two templates, every question of one wrong and of the other right: a stratified draw always takes two of
  each, so its interval is exactly [0.5, 0.5]; an unstratified draw of four singletons is all-wrong (or all-right)
  with probability 1/16 > 0.025, so its interval is [0.0, 1.0].
"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "analysis" / "grid-full"

_spec = importlib.util.spec_from_file_location("e189_exploratory", HERE.parent / "exploratory.py")
ex = importlib.util.module_from_spec(_spec)
sys.modules["e189_exploratory"] = ex
_spec.loader.exec_module(ex)
F = ex.load_frozen(ROOT)
_espec = importlib.util.spec_from_file_location("export_checker_ledger", ROOT / "scripts" / "export_checker_ledger.py")
exporter = importlib.util.module_from_spec(_espec)
sys.modules["export_checker_ledger"] = exporter
_espec.loader.exec_module(exporter)

K3 = {"s01-pass": 0, "s01-flag": 1, "s01-abstain": 1, "s01-noquery": 2}  # by hand, entry 179's rule mod 3
DEV = sorted(K3)
QWEN = "Qwen/Qwen3.8-27B"
MODELS = (QWEN, "google/gemma-4-31B")
ARM = f"s01|{QWEN}|none|prompt|-"

# The store section's branch-phase rule sentence as science's 25 Oct ~04:55Z acceptance words it, written by hand
BRANCH_RULE_TEXT = ("branch-phase rows (started at or after the manifest's phase-branch instant) carry the recorded "
                    "endpoint of their parent attempt, because harness/checkloop/online.py:542 (branch_attempt) copies a "
                    "branch record's store from its parent record; they are outside the order check, and each one's live "
                    "retry is attributed to the copy current at its start, the chain's last copy (every change_at "
                    "precedes the branch-phase instant); per OUT the report lists the parent copies' classes and the "
                    "branch copy's class (science, 25 Oct ~04:55Z)")


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(ids) -> str:
    return _h("".join(f"{q}\n" for q in sorted(ids)))


def _recheck(out: Path) -> None:
    """A replay-check/2 file over the OUT's inputs as they now are (the split moved records; the dry run re-verifies
    real partitions with verify_batch.py instead)."""
    check = json.loads((FIXTURE / "batch" / "replay-check.json").read_text(encoding="utf-8"))
    inputs = F.gate.batch_inputs(out)
    replay = [r for r in inputs if F.gate.input_kind(r) == "replay"]
    check.update(n_records=len(replay), n_ok=len(replay), failures=[], records_sha256=F.gate.inputs_sha256(out),
                 hashed_only=[r for r in inputs if F.gate.input_kind(r) == "hashed"])
    (out / "replay-check.json").write_text(json.dumps(check, indent=1) + "\n", encoding="utf-8")


def _record(k: int = 3, schema: str = "entry180-partition/1") -> dict:
    parts = []
    for p in range(k):
        mine = [q for q in DEV if K3[q] == p]
        parts.append({"index": p, "copy": f"copy-{p}", "n_questions": len(mine), "ids_sha256": _digest(mine),
                      "question_id_sha256": sorted(_h(q) for q in mine)})
    qsha = hashlib.sha256((FIXTURE / "questions.jsonl").read_bytes()).hexdigest()
    return {"schema": schema, "salt": "20261022|", "k": k, "graph": "s01", "split": "dev",
            "questions": {"path": "questions.jsonl", "sha256": qsha}, "n_questions": len(DEV),
            "all_ids_sha256": _digest(DEV), "copies": [f"copy-{p}" for p in range(k)], "partitions": parts}


CHECK = f"?cmd=stats equal to ref-s01-stats.json in 41 fields at 2026-10-22T19:45:01Z (store check sha256 {'a' * 64})"


def _identity(out: Path, *, stack: str = "stack-a", partition: dict | None = None) -> dict:
    ident = {"out": f"/host/{out.name}", "stacks_by_route_model": {m: stack for m in MODELS},
             "store": {"out": f"/host/{out.name}"}}
    if partition is not None:
        ident["partition"] = partition
        ident["store"].update(replica=partition["copy"], identity_check=CHECK)
    return ident


def _voided(path: Path, *outs: str) -> Path:
    """A voided-run record: the five voided OUTs the script knows, and ``outs``."""
    path.write_text("".join(json.dumps({"out": o}) + "\n" for o in (*ex.KNOWN_VOIDED, *outs)), encoding="utf-8")
    return path


def _write_identity(out: Path, ident: dict) -> None:
    (out / "deployment-identity.json").write_text(json.dumps(ident, indent=1) + "\n", encoding="utf-8")


@pytest.fixture
def world(tmp_path):
    """The fixture whole (``single``, with an identity and no partition block) and split into three partition OUTs
    (``parts``) bound to ``record``; a voided-run record naming another OUT; the fixture's questions and ledger."""
    batch = FIXTURE / "batch"
    single = tmp_path / "single"
    shutil.copytree(batch, single, ignore=shutil.ignore_patterns("claims"))
    _write_identity(single, _identity(single))
    record = _record()
    rec_path = tmp_path / "record.json"
    rec_path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    rec_sha = hashlib.sha256(rec_path.read_bytes()).hexdigest()
    parts = []
    for p in range(3):
        out = tmp_path / f"part-{p}"
        shutil.copytree(batch, out, ignore=shutil.ignore_patterns("claims", "records"))
        (out / "records").mkdir()
        for f in (batch / "records").glob("*.json"):
            if K3[json.loads(f.read_text(encoding="utf-8"))["question_id"]] == p:
                shutil.copy2(f, out / "records" / f.name)
        _recheck(out)
        _write_identity(out, _identity(out, partition={"k": 3, "index": p, "copy": f"copy-{p}", "record_sha256": rec_sha,
                                                       "ids_sha256": record["partitions"][p]["ids_sha256"]}))
        parts.append(out)
    voided = _voided(tmp_path / "voided.jsonl", "/host/some-voided-out")
    admit_b = _admissions(tmp_path / "admit-b.json", ADMIT_B)  # FP_B admitted for Qwen on stack-a, by (c)
    ledger = tmp_path / "ledger.jsonl"  # the frozen exporter's ledger of the fixture's branch records (keyed by parent)
    assert exporter.main(["--split", "dev", "--records", str(batch), "--out", str(ledger)]) == 0
    return {"tmp": tmp_path, "single": single, "parts": parts, "record": rec_path, "record_sha": rec_sha,
            "voided": voided, "questions": FIXTURE / "questions.jsonl", "ledger": ledger, "admit_b": admit_b}


SPAN = {"s01-pass": "c1", "s01-noquery": "c1", "s01-flag": "s01-flag", "s01-abstain": "s01-abstain",
        "s01-held": "s01-held"}  # s01-pass (partition 0) and s01-noquery (partition 2) share a cluster


def _map(clusters=None):
    qsha = hashlib.sha256((FIXTURE / "questions.jsonl").read_bytes()).hexdigest()
    return (F.prereg.ClusterMap(dict(clusters or SPAN), "test map"),
            {"graph": "s01", "store": "s01", "path": "test", "sha256": "test", "questions_sha256": qsha})


def _call(w, batches, *, record=True, ledgers=(), voided=None, admissions=None, **stores) -> object:
    return ex.Call(tuple(batches), (w["questions"],), "s01", "dev", (voided or w["voided"],),
                   w["record"] if record else None, tuple(ledgers), 1, None, admissions=admissions, **stores)


def _run(w, batches, **kw):
    cm, rec = _map(kw.pop("clusters", None))
    K = kw.pop("K", None)
    return ex.run(F, _call(w, batches, **kw), K_clusters=K, cluster_map=cm, map_record=rec)


def _none_record(out: Path) -> Path:
    """A natural ("none") cell's record: no branch cell names it as its parent, so moving it breaks nothing else."""
    return next(f for f in sorted((out / "records").glob("*.json"))
                if json.loads(f.read_text(encoding="utf-8"))["load"]["arm"] == "none")


def _refused(w, batches, match, **kw):
    with pytest.raises(F.refusals, match=match):
        _run(w, batches, **kw)


# --- the cluster bootstrap ------------------------------------------------------------------------------------------


HAND_K = np.array([[1, 0, 2], [2, 1, 0], [0, 2, 1]], dtype=float)


def test_a_union_whose_clusters_span_partitions_resamples_them_together(world):
    result = _run(world, world["parts"], K=HAND_K, ledgers=(world["ledger"],))
    acc = result["arms"][ARM]["metrics"]["accuracy"]
    assert acc["estimate"] == 0.5 and acc["ci"] == [0.0, 0.8]
    ex_ = result["exploratory"]
    assert ex_["clusters"] == {"questions": 4, "clusters": 3, "sizes": {"1": 2, "2": 1}, "largest": 2}
    assert ex_["partition_union"]["k"] == 3 and sorted(o["partition"] for o in ex_["outs"]) == [0, 1, 2]
    assert set(ex_["partition_union"]) == {"record", "record_sha256", "schema", "k", "graph", "stack_check"}


def test_the_per_question_weights_are_the_clusters_draws(world):
    cm, _ = _map()
    index = F.prereg.QuestionIndex(tuple(DEV), ("t|unrecorded",) * 4)  # abstain, flag, noquery, pass
    K, clusters = ex.cluster_counts(F, cm, index, None, HAND_K)
    assert clusters.ids == ("s01-abstain", "s01-flag", "c1")
    assert K.tolist() == [[1, 0, 2, 2], [2, 1, 0, 0], [0, 2, 1, 1]]


def test_the_pooled_union_equals_the_unpartitioned_call(world):
    union = _run(world, world["parts"], ledgers=(world["ledger"],))
    single = _run(world, [world["single"]], record=False, ledgers=(world["ledger"],))
    same = ex.compare_sections(union, single, skip_settings=("inputs", "code_digest"))
    assert all(same[s] for s in ex.SECTIONS) and same["settings_differing"] == []
    assert union["settings"]["resampling"] == ex.RESAMPLING and union["settings"]["n_boot"] == 10_000


def _analyse_with_base_and_current(world, tmp_path, monkeypatch, batches, *extra):
    """Call analyse in the base commit's script and in this one on the same synthetic inputs: the two results.json bytes."""
    # (merged with the replay-matrix work: the base is the commit before the plain-partition union was merged, 070b85cf, which has the
    # replay diagnostics sections every results.json now carries; 351ad4a0 predates them)
    source = subprocess.run(["git", "show", "070b85cf:results/e189-exploratory-20261022/exploratory.py"],
                            check=True, capture_output=True).stdout
    old_path = tmp_path / "base_exploratory.py"
    old_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("e189_exploratory_base_full_call", old_path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    old.HERE = ex.HERE  # companion analysis modules are the same frozen files for both calls
    cm, map_record = _map()
    for module in (ex, old):
        monkeypatch.setattr(module, "load_frozen", lambda root: F)
        monkeypatch.setattr(module, "cluster_map_for", lambda *args: (cm, map_record))
    argv = ["analyse", "--code-root", str(ROOT), "--graph", "s01", "--split", "dev",
            *itertools.chain.from_iterable(("--batch", str(b)) for b in batches),
            "--questions", str(world["questions"]), "--voided", str(world["voided"]), "--ledger", str(world["ledger"]),
            "--n-reps", "1", *extra]
    current_dir, base_dir = tmp_path / "current", tmp_path / "base"
    assert ex.main([*argv, "--out", str(current_dir)]) == 0
    assert old.main([*argv, "--out", str(base_dir)]) == 0
    return (current_dir / "results.json").read_bytes(), (base_dir / "results.json").read_bytes()


def test_no_new_option_analyse_matches_base_commit_byte_for_byte(world, tmp_path, monkeypatch):
    """Call analyse in both versions on one synthetic OUT and compare the written bytes."""
    current, base = _analyse_with_base_and_current(world, tmp_path, monkeypatch, [world["single"]])
    assert current == base


def test_no_new_option_union_analyse_matches_base_commit_byte_for_byte(world, tmp_path, monkeypatch):
    """The same over an existing-schema partition union (entry180-partition/1, three partition OUTs): the plain union's
    changes leave this call's results untouched, a union without the plain options keeps IDENTITY_CHECK."""
    current, base = _analyse_with_base_and_current(world, tmp_path, monkeypatch, world["parts"],
                                                   "--partition-record", str(world["record"]))
    assert current == base
    block = json.loads(current)["exploratory"]["partition_union"]
    assert block["k"] == 3 and "each store copy identity-checked" in block["stack_check"]
    assert set(block) == {"record", "record_sha256", "schema", "k", "graph", "stack_check"}  # no plain-union key


def test_a_union_without_the_plain_options_still_refuses_the_view_note_with_the_base_message(world):
    for out in world["parts"]:  # each partition OUT's store block carries the identity view's note instead of a check
        identity = json.loads((out / "deployment-identity.json").read_text(encoding="utf-8"))
        identity["store"]["identity_check"] = PLAIN_NOTE
        _write_identity(out, identity)
    with pytest.raises(ex.Refused) as refusal:
        _run(world, world["parts"], ledgers=(world["ledger"],))
    assert str(refusal.value) == " | ".join(
        f"{out}: its store copy carries no passed identity check as the partition builders record it "
        f"(?cmd=stats equal to ... (store check sha256 ...))" for out in world["parts"])


PLAIN_GROUPS = (("t01-q1",), ("t01-q2", "t02-q3"))  # independent partition expectations, K=2
PLAIN_LATER = "2026-10-21T00:00:00.000+00:00"  # a manifest's created_at in an alternative OUT of the same name


def _plain_union_inputs(tmp_path, monkeypatch):
    """The fixed inputs of a synthetic plain partition union: the registered questions, the producer's record with its two
    partition questions files, a voided-run record and the graph's map record. No OUT is built here."""
    source = ROOT / "tests" / "fixtures" / "analysis" / "plain-run"
    rows = [json.loads(line) for line in (source / "questions.jsonl").read_text().splitlines()]
    monkeypatch.setattr(ex, "REGISTERED_SPLIT", "dev")  # this public fixture's completed split
    registered = tmp_path / "registered.jsonl"
    registered.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    assert tuple(tuple(q for q in group if ex.partition_of(q, 2) == i) for i, group in enumerate(PLAIN_GROUPS)) == PLAIN_GROUPS
    record = {"rule": ex.PLAIN_RULE, "salt": ex.SALT, "k": 2, "copies": ["copy-0", "copy-1"],
              "reps": 3, "questions_sha256": hashlib.sha256(registered.read_bytes()).hexdigest(),
              "test_questions": 3, "partitions": {}}
    bodies = []
    for i, group in enumerate(PLAIN_GROUPS):
        body = b"".join(json.dumps(row).encode() + b"\n" for row in rows if row["id"] in group)
        name = f"questions-p{i}of2.jsonl"
        (tmp_path / name).write_bytes(body)
        bodies.append(body)
        record["partitions"][str(i)] = {"copy": f"copy-{i}", "questions": len(group),
                                         "attempts": 3 * len(group), "ids_sha256": _digest(group),
                                         "file": name, "file_sha256": hashlib.sha256(body).hexdigest()}
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record))
    map_record = {"graph": "rhea-test", "store": "rhea-test", "path": "synthetic", "sha256": "synthetic",
                  "questions_sha256": record["questions_sha256"]}
    monkeypatch.setattr(ex, "require_plain_private_graph", lambda graph: None if graph == "rhea-test" else
                        (_ for _ in ()).throw(ex.Refused("wrong synthetic graph")))
    private = _plain_union_private(tmp_path, monkeypatch, ("copy-0", "copy-1"))
    return {"source": source, "rows": rows, "registered": registered, "record_path": record_path, "bodies": bodies,
            "voided": _voided(tmp_path / "voided.jsonl"), "map_record": map_record, **private}


PLAIN_NOTE = "not recorded for this OUT (identity view): no store check is claimed"  # the view tool's, as science ruled (b)
PLAIN_DISCLOSURE = "a15 copies: identity by manifest and pinned description; no store-stats check (not QLever)"
PLAIN_ENDPOINT_PROOFS = ("a15 copies: no store-proofs.json entry is required for any (OUT, endpoint); each endpoint is named "
                         "by its OUT's identity and is the pinned endpoint copies' endpoint of that OUT's copy")
PLAIN_ENDPOINTS = ("http://plain-copy-0.invalid:7000/", "http://plain-copy-1.invalid:7000/")  # the private table's defaults


def _plain_union_private(tmp_path, monkeypatch, copies, *, name="plain", endpoints=None):
    """The reference pin and the two pinned private tables of the synthetic plain union's graph, listing ``copies``
    (private files under tmp_path, which the autouse fixture makes the private a15 tree); ``endpoints``: each copy's
    endpoint, where the call's OUTs record one."""
    pin_dir = tmp_path / f"{name}-pin-private"
    pin_dir.mkdir()
    pin = pin_dir / "pin.json"
    pin.write_text(json.dumps({"schema": ex.REFERENCE_PIN_SCHEMA, "graph": "rhea-test", "reference_sha256": "5" * 64}))
    pin.chmod(0o600)
    monkeypatch.setitem(ex.PRIVATE_REFERENCE_PINS, "rhea-test", _sha(pin))
    classes = _private_table(tmp_path, name=f"{name}-classes", copies={c: CLASS_C for c in copies})
    endpoints = _private_endpoints(tmp_path, *(("rhea-test", (endpoints or {}).get(c, f"http://plain-{c}.invalid:7000/"), c)
                                               for c in copies), name=f"{name}-endpoints")
    return {"pin": pin, "classes": classes, "endpoints": endpoints}


def _plain_union_outs(tmp_path, inputs, prefix_name, *, later=(), check=PLAIN_NOTE, endpoints=None, recorded=None):
    """A complete plain OUT per partition at ``tmp_path/<prefix_name>-p<i>of2``, valid on its own: that partition's records
    only, the partition's questions file in the manifest's fingerprint, an identity naming the OUT, and a replay check. The
    manifests of the partitions in ``later`` carry another created_at, which makes them other bytes. ``endpoints`` (one per
    partition): the store endpoint each identity names and, unless ``recorded`` gives others, each partition's records record."""
    source = inputs["source"]
    claim_by_prefix = {}
    for claim in (source / "batch" / "claims").glob("*.claim"):
        claim_by_prefix[claim.name.split(".")[0]] = json.loads(claim.read_text())["key"][0]
    outs = []
    for i, group in enumerate(PLAIN_GROUPS):
        out = tmp_path / f"{prefix_name}-p{i}of2"
        shutil.copytree(source / "batch", out, ignore=shutil.ignore_patterns("records", "claims", "runs.jsonl"))
        (out / "records").mkdir()
        for src in (source / "batch" / "records").glob("*.json"):
            if claim_by_prefix[src.name.split(".")[0]] in group:
                shutil.copy2(src, out / "records" / src.name)
                if endpoints is not None:
                    record = json.loads(src.read_text())
                    record["store"]["endpoint"] = (recorded or endpoints)[i]
                    (out / "records" / src.name).write_text(json.dumps(record))
        manifest = json.loads((out / "manifest.json").read_text())
        manifest["fingerprint"]["questions_sha256"] = hashlib.sha256(inputs["bodies"][i]).hexdigest()
        if i in later:
            manifest["created_at"] = PLAIN_LATER
        (out / "manifest.json").write_text(json.dumps(manifest))
        identity = _identity(out, partition={"copy": f"copy-{i}"})
        identity.pop("partition")
        identity["store"]["identity_check"] = check  # what the identity view writes, not a QLever check
        if endpoints is not None:
            identity["store"]["endpoint"] = endpoints[i]
        identity["stacks_by_route_model"] = {"oracle-model": "synthetic-stack"}
        _write_identity(out, identity)
        _recheck(out)
        outs.append(out)
    return outs


def _plain_union_pins(outs):
    return tuple(hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest() for out in outs)


def _plain_union_call(inputs, outs, prefix, pins, **given):
    """The call that submits ``outs`` against the intended OUT prefix and per-OUT manifest sha256 pins."""
    record_sha = hashlib.sha256(inputs["record_path"].read_bytes()).hexdigest()
    return ex.Call(tuple(outs), (inputs["registered"],), "rhea-test", "dev", (inputs["voided"],), inputs["record_path"],
                   (), 2, None, plain_partition_record_sha256=record_sha,
                   plain_union_outs=prefix + ":" + ",".join(pins), **_plain_union_private_options(inputs, **given))


def _plain_union_private_options(inputs, **given):
    """The call's reference pin and private tables: those of ``inputs``, unless ``given`` replaces or drops (None) one."""
    options = {"reference_pin": inputs["pin"], "private_copy_classes": inputs["classes"][0],
               "private_copy_classes_sha256": inputs["classes"][1], "private_endpoint_copies": inputs["endpoints"][0],
               "private_endpoint_copies_sha256": inputs["endpoints"][1]}
    options.update(given)
    return {k: v for k, v in options.items() if v is not None}


def _plain_union_accepts(inputs, outs, prefix, pins):
    _, union, context = ex.prepare(F, _plain_union_call(inputs, outs, prefix, pins), map_record=inputs["map_record"])
    assert union.k == 2 and [o["partition"] for o in context["outs"]] == [0, 1]
    assert [o["path"] for o in context["outs"]] == [str(o) for o in outs]
    return context


def _plain_union_refuses(inputs, outs, prefix, pins, text, **given):
    with pytest.raises(ex.Refused) as refusal:
        ex.prepare(F, _plain_union_call(inputs, outs, prefix, pins, **given), map_record=inputs["map_record"])
    assert str(refusal.value) == text


def test_plain_partition_union_runs_a_complete_synthetic_analysis(tmp_path, monkeypatch):
    """A plain OUT for each partition goes through prepare, loading, and analysis as one call."""
    inputs = _plain_union_inputs(tmp_path, monkeypatch)
    outs = _plain_union_outs(tmp_path, inputs, "synthetic-union")
    call = _plain_union_call(inputs, outs, str(tmp_path / "synthetic-union"), _plain_union_pins(outs))
    result = ex.run(F, call, map_record=inputs["map_record"])
    block = result["exploratory"]["partition_union"]
    assert block["k"] == 2
    assert [o["partition"] for o in result["exploratory"]["outs"]] == [0, 1]
    assert result["arms"]
    # the results say how each copy was identified, and show which manifests the OUTs were pinned to (hashed by hand here)
    assert block["copy_identity"] == PLAIN_DISCLOSURE
    assert block["out_manifest_sha256"] == [hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest() for out in outs]
    assert block["stack_check"] == ("one stack per (graph, model) from the partitions' deployment identities (in place of "
                                    "check_one_stack's one-URL/one-endpoint refusals; entries 150, 184)")


def test_a_complete_alternative_plain_out_set_at_another_prefix_is_refused_against_the_pins(tmp_path, monkeypatch):
    """Two complete OUT sets, A and B, each accepted under its own pins; the manifests are the same bytes, so only the
    directory names tell them apart. Pinned to A, set B is refused, and A is refused under B's pins."""
    inputs = _plain_union_inputs(tmp_path, monkeypatch)
    set_a = _plain_union_outs(tmp_path, inputs, "pinned-union")
    set_b = _plain_union_outs(tmp_path, inputs, "other-union")
    prefix_a, prefix_b = str(tmp_path / "pinned-union"), str(tmp_path / "other-union")
    pins_a, pins_b = _plain_union_pins(set_a), _plain_union_pins(set_b)
    assert pins_a == pins_b and set_a != set_b
    _plain_union_accepts(inputs, set_a, prefix_a, pins_a)
    _plain_union_accepts(inputs, set_b, prefix_b, pins_b)  # B is valid on its own
    order = "--plain-union-outs requires OUTs in exact partition index order"
    _plain_union_refuses(inputs, set_b, prefix_a, pins_a, order)
    _plain_union_refuses(inputs, set_a, prefix_b, pins_b, order)
    _plain_union_refuses(inputs, set_b, prefix_a, pins_b, order)  # B's own pins do not make A's prefix its


@pytest.mark.parametrize("later", [(0, 1), (1,)])
def test_a_complete_alternative_plain_out_set_with_other_manifests_is_refused_against_the_pins(tmp_path, monkeypatch, later):
    """Set B replaces set A at the same prefix with manifests that differ in some or all partitions. Each is accepted under
    its own pins; pinned to A, B is refused by the first differing manifest, and the same B passes once re-pinned."""
    inputs = _plain_union_inputs(tmp_path, monkeypatch)
    prefix = str(tmp_path / "union")
    set_a = _plain_union_outs(tmp_path, inputs, "union")
    pins_a = _plain_union_pins(set_a)
    _plain_union_accepts(inputs, set_a, prefix, pins_a)
    for out in set_a:
        shutil.rmtree(out)  # the same names again: the alternative set is another run's OUTs, not the pinned ones
    set_b = _plain_union_outs(tmp_path, inputs, "union", later=later)
    pins_b = _plain_union_pins(set_b)
    assert set_b == set_a
    assert [a != b for a, b in zip(pins_a, pins_b)] == [i in later for i in range(2)]
    _plain_union_accepts(inputs, set_b, prefix, pins_b)  # B is valid on its own
    _plain_union_refuses(inputs, set_b, prefix, pins_a,
                         f"{set_b[min(later)]}: manifest.json sha256 differs from the OUT pin")


NOT_THE_NOTE = ("{}: its store copy does not carry the identity view's store note exactly (a plain partition union's "
                "per-copy identity is its manifest identity and the pinned store description)")
NEEDS_PIN_AND_TABLES = ("a plain partition union needs the privately pinned graph's reference pin, private copy classes and "
                        "private endpoint copies (--reference-pin, --private-copy-classes, --private-endpoint-copies)")


def test_a_plain_union_is_refused_unless_each_store_block_carries_exactly_the_view_note(tmp_path, monkeypatch):
    inputs = _plain_union_inputs(tmp_path, monkeypatch)
    prefix = str(tmp_path / "union")
    outs = _plain_union_outs(tmp_path, inputs, "union")
    context = _plain_union_accepts(inputs, outs, prefix, _plain_union_pins(outs))
    assert context["partition"]["copy_identity"] == PLAIN_DISCLOSURE
    for shutil_out in outs:
        shutil.rmtree(shutil_out)
    for check in (CHECK, PLAIN_NOTE + ".", PLAIN_NOTE.capitalize(), "not recorded"):  # a QLever check is refused here too
        bad = _plain_union_outs(tmp_path, inputs, "union", check=check)
        _plain_union_refuses(inputs, bad, prefix, _plain_union_pins(bad),
                             " | ".join(NOT_THE_NOTE.format(out) for out in bad))
        for out in bad:
            shutil.rmtree(out)


def test_a_plain_union_is_refused_without_the_pin_and_both_tables_and_for_a_copy_they_do_not_list(tmp_path, monkeypatch):
    inputs = _plain_union_inputs(tmp_path, monkeypatch)
    prefix = str(tmp_path / "union")
    outs = _plain_union_outs(tmp_path, inputs, "union")
    pins = _plain_union_pins(outs)
    _plain_union_accepts(inputs, outs, prefix, pins)
    for dropped in ({"reference_pin": None},
                    {"private_copy_classes": None, "private_copy_classes_sha256": None},
                    {"private_endpoint_copies": None, "private_endpoint_copies_sha256": None}):
        _plain_union_refuses(inputs, outs, prefix, pins, NEEDS_PIN_AND_TABLES, **dropped)
    for listed, bad in ((("copy-0",), "[1]"), (("copy-1",), "[0]"), (("copy-9",), "[0, 1]")):
        partial = _plain_union_private(tmp_path, monkeypatch, listed, name=f"only-{'-'.join(listed)}")
        _plain_union_refuses({**inputs, **partial}, outs, prefix, pins,
                             f"a plain partition union's copies must be listed in the pinned copy classes and endpoint copies: "
                             f"not listed for partitions {bad}")
    classes_only = _private_table(tmp_path, name="classes-only", copies={"copy-0": CLASS_C, "copy-1": CLASS_C})
    other = _private_endpoints(tmp_path, ("rhea-test", "http://plain-copy-0.invalid:7000/", "copy-0"), name="one-endpoint")
    _plain_union_refuses({**inputs, "classes": classes_only, "endpoints": other}, outs, prefix, pins,
                         "a plain partition union's copies must be listed in the pinned copy classes and endpoint copies: "
                         "not listed for partitions [1]")


UNPINNED = "its endpoint is not the pinned endpoint copies' endpoint of the copy its identity names"
NOT_LISTED = ("an endpoint the OUT does not list (its deployment identity's store blocks or its manifest's store_changes)")


def test_a_plain_union_with_two_endpoints_and_no_store_proofs_is_accepted_when_both_are_pinned(tmp_path, monkeypatch):
    inputs = _plain_union_inputs(tmp_path, monkeypatch)
    outs = _plain_union_outs(tmp_path, inputs, "union", endpoints=PLAIN_ENDPOINTS)
    context = _plain_union_accepts(inputs, outs, str(tmp_path / "union"), _plain_union_pins(outs))
    block, stores = context["partition"], context["stores"]
    assert block["copy_identity"] == PLAIN_DISCLOSURE and block["endpoint_proofs"] == PLAIN_ENDPOINT_PROOFS
    assert stores.several == [{"graph": "rhea-test", "model": "oracle-model", "endpoints": 2, "outs": 2}]
    assert sorted(stores.copies) == ["copy-0", "copy-1"] and all(c["proofs"] == [] for c in stores.copies.values())
    assert stores.config["matched"] == 0  # not one store-proofs entry used (and none was given)
    assert stores.labels == {(str(outs[0]), PLAIN_ENDPOINTS[0]): "copy-0", (str(outs[1]), PLAIN_ENDPOINTS[1]): "copy-1"}


def test_a_plain_union_is_refused_for_an_endpoint_the_pinned_description_does_not_list_for_its_copy(tmp_path, monkeypatch):
    inputs = _plain_union_inputs(tmp_path, monkeypatch)
    prefix = str(tmp_path / "union")
    outs = _plain_union_outs(tmp_path, inputs, "union", endpoints=PLAIN_ENDPOINTS)
    pins = _plain_union_pins(outs)
    refusal = ("store copies not listed or not proven (several store endpoints per graph and model are analysed only listed "
               "in their OUT and identity-proven): ")

    def where(i, endpoint=None):
        return f"{outs[i]} endpoint sha256:{_h(endpoint or PLAIN_ENDPOINTS[i])[:12]}: "

    # the second copy is pinned at another endpoint
    other = _plain_union_private(tmp_path, monkeypatch, ("copy-0", "copy-1"), name="other",
                                 endpoints={"copy-1": "http://elsewhere.invalid:7000/"})
    _plain_union_refuses({**inputs, **other}, outs, prefix, pins, refusal + where(1) + UNPINNED)
    # each endpoint is pinned for the other partition's copy (check_stores' older identity-against-table check refuses first,
    # then this rule says the same of both endpoints)
    crossed = _plain_union_private(tmp_path, monkeypatch, ("copy-0", "copy-1"), name="crossed",
                                   endpoints={"copy-0": PLAIN_ENDPOINTS[1], "copy-1": PLAIN_ENDPOINTS[0]})
    with pytest.raises(ex.Refused) as crossed_refusal:
        ex.prepare(F, _plain_union_call({**inputs, **crossed}, outs, prefix, pins), map_record=inputs["map_record"])
    head, _, rest = str(crossed_refusal.value).partition(refusal)  # (the order of the four problems is not the test's concern)
    assert head == "" and sorted(rest.split(" | ")) == sorted([
            f"{outs[0]} {ex.endpoint_label(PLAIN_ENDPOINTS[0])}: its deployment identity names copy-0, the endpoint copies "
            f"a replica of the private endpoint copies",
            f"{outs[1]} {ex.endpoint_label(PLAIN_ENDPOINTS[1])}: its deployment identity names copy-1, the endpoint copies "
            f"a replica of the private endpoint copies",
            where(0) + UNPINNED, where(1) + UNPINNED])
    # the second partition's records name an endpoint its own identity does not (the pinned table cannot make it so)
    unnamed = _plain_union_outs(tmp_path, inputs, "unnamed", endpoints=PLAIN_ENDPOINTS,
                                recorded=(PLAIN_ENDPOINTS[0], "http://other.invalid:7000/"))
    _plain_union_refuses(inputs, unnamed, str(tmp_path / "unnamed"), _plain_union_pins(unnamed),
                         refusal + f"{unnamed[1]} endpoint sha256:{_h('http://other.invalid:7000/')[:12]}: " + NOT_LISTED)


def test_a_plain_union_on_one_endpoint_or_none_has_the_store_section_it_had_without_the_pinned_rule(tmp_path, monkeypatch):
    inputs = _plain_union_inputs(tmp_path, monkeypatch)

    def table(flag, given, path):
        classes = ex.read_private_table("rhea-test", "--private-copy-classes", given["classes"][0], given["classes"][1],
                                        lambda module, raw: module.loads(raw), lambda doc: doc["copies"])
        endpoints = ex.read_private_table("rhea-test", "--private-endpoint-copies", given["endpoints"][0],
                                          given["endpoints"][1], lambda module, raw: module.loads_endpoint_copies(raw),
                                          lambda doc: (e["replica"] for e in doc))
        return ex.check_stores(ex.load_attribution(), [ex.Out.read(o) for o in path], "rhea-test", (), None,
                               ex.reference_pin("rhea-test", given["pin"]), None, None, classes, endpoints, plain_union=flag)

    for name, endpoints in (("none", None), ("one", ("http://shared.invalid:7000/",) * 2)):
        outs = _plain_union_outs(tmp_path, inputs, name, endpoints=endpoints)  # no endpoint recorded, or the same in both
        with_flag, without_flag = table(True, inputs, outs), table(False, inputs, outs)
        assert with_flag == without_flag and with_flag.several == []


def test_a_union_without_the_plain_options_with_two_endpoints_and_no_entries_is_refused_as_ever(world):
    for out, endpoint in zip(world["parts"], (E0, E1, E0)):
        _set_endpoints(out, lambda r, e=endpoint: e)
    with pytest.raises(ex.Refused) as refusal:
        _run(world, world["parts"], ledgers=(world["ledger"],), copy_classes=_classes_file(world["tmp"] / "classes.json"))
    assert str(refusal.value) == STORE_REFUSAL + " | ".join(
        f"{out} {ex.endpoint_label(endpoint)}: {what}" for out, endpoint in zip(world["parts"], (E0, E1, E0))
        for what in (NOT_LISTED, "no store-proofs.json entry proves its copy"))


def test_check_stores_holds_a_plain_union_to_the_private_endpoint_copies_or_refuses(tmp_path, monkeypatch):
    inputs = _plain_union_inputs(tmp_path, monkeypatch)
    outs = _plain_union_outs(tmp_path, inputs, "union", endpoints=PLAIN_ENDPOINTS)
    with pytest.raises(ex.Refused, match="^a plain partition union's endpoints are held to the private endpoint copies, which "
                                          "are not given$"):
        ex.check_stores(ex.load_attribution(), [ex.Out.read(o) for o in outs], "rhea-test", (), None, None, None, None, None,
                        None, plain_union=True)


# --- end to end through the real identity-view tool (a15_identity_view.py lives outside git) -----------------------------

VIEW_TOOL = os.environ.get("A15_IDENTITY_VIEW_TOOL")
STARTED = "2026-10-20T02:24:16.250+00:00"  # a plain OUT's manifest start; its launcher line is two seconds earlier
LAUNCHED = "2026-10-20T02:24:14Z"


def _view_tool(tmp_path, monkeypatch):
    """A copy of the tool (mode 700, outside git) loaded as a module, its private tree made tmp_path's own."""
    for ancestor in Path(os.path.realpath(tmp_path)).parents:
        if os.path.lexists(ancestor / ".git"):
            pytest.skip("the tool refuses a VIEW under a git work tree: give pytest a --basetemp that has no .git above it")
    copy = Path(tmp_path) / "tool" / "a15_identity_view.py"
    copy.parent.mkdir()
    shutil.copy2(VIEW_TOOL, copy)
    copy.chmod(0o700)
    spec = importlib.util.spec_from_file_location("a15_identity_view_under_test", copy)
    tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)
    root = Path(os.path.realpath(tmp_path)) / "private" / "a15"
    (root / "runs" / "test-synthetic").mkdir(parents=True)
    monkeypatch.setattr(tool, "PRIVATE_ROOT", root)
    monkeypatch.setattr(tool, "TEST_RUNS_ROOT", root / "runs" / "test-synthetic")
    monkeypatch.setattr(tool, "RUNS_TREE", root / "runs")
    return tool, root


def _raw_plain_out(inputs, root, i, endpoint, *, records_name_endpoint):
    """Partition ``i``'s plain OUT as a runner leaves it, with no deployment identity: the plain-run fixture's records of that
    partition, each recording its store endpoint, the manifest naming the endpoint and a start, and the launcher's start record
    beside it. Modes as the tool requires them."""
    source = inputs["source"]
    claim_by_prefix = {}
    for claim in (source / "batch" / "claims").glob("*.claim"):
        claim_by_prefix[claim.name.split(".")[0]] = json.loads(claim.read_text())["key"][0]
    out = root / "runs" / "test-synthetic" / f"union-p{i}of2"
    shutil.copytree(source / "batch", out, ignore=shutil.ignore_patterns("records", "claims", "runs.jsonl"))
    for sub in ("records", "infrastructure", "harness", "aborted", "claims"):
        (out / sub).mkdir(exist_ok=True)
    for src in (source / "batch" / "records").glob("*.json"):
        if claim_by_prefix[src.name.split(".")[0]] in PLAIN_GROUPS[i]:
            record = json.loads(src.read_text())
            if records_name_endpoint:
                record["store"]["endpoint"] = endpoint  # the endpoint the record's queries went to
            (out / "records" / src.name).write_text(json.dumps(record))
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["fingerprint"]["questions_sha256"] = hashlib.sha256(inputs["bodies"][i]).hexdigest()
    manifest["fingerprint"]["store"]["endpoint"] = endpoint
    manifest["starts"] = [{"at": STARTED}]
    (out / "manifest.json").write_text(json.dumps(manifest))
    _recheck(out)
    for path in out.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    out.chmod(0o700)
    host_port = endpoint.removeprefix("http://")
    starts = out.with_name(out.name + ".starts")
    starts.write_text(f"{LAUNCHED} live models fake workers 3 stack synthetic-stack base-url-file /synthetic/base-url "
                      f"copy copy-{i} (synthetic forward {host_port})\n")
    starts.chmod(0o600)
    return out, starts


needs_view_tool = pytest.mark.skipif(not VIEW_TOOL or not Path(VIEW_TOOL or "").is_file(),
                                     reason="A15_IDENTITY_VIEW_TOOL is unset or names no file")


def _plain_union_views(tmp_path, monkeypatch, capsys, *, records_name_endpoint):
    """Two synthetic plain OUTs (K = 2, one copy each) without a deployment identity, each with its VIEW made by the real
    tool: (the call's inputs, the views, the views' directory, the views' manifest sha256 pins)."""
    tool, root = _view_tool(tmp_path, monkeypatch)
    assert tool.STORE_CHECK_NOTE == PLAIN_NOTE == ex.VIEW_STORE_CHECK_NOTE  # the tool's words are the constant's
    endpoints = {"copy-0": "http://127.0.0.1:9000", "copy-1": "http://127.0.0.1:9001"}
    inputs = _plain_union_inputs(tmp_path, monkeypatch)
    inputs.update(_plain_union_private(tmp_path, monkeypatch, ("copy-0", "copy-1"), name="view", endpoints=endpoints))
    views_dir = root / "looks" / "views"
    views_dir.mkdir(parents=True)
    views = []
    for i in range(2):
        out, starts = _raw_plain_out(inputs, root, i, endpoints[f"copy-{i}"], records_name_endpoint=records_name_endpoint)
        assert not (out / "deployment-identity.json").exists()  # a plain OUT never had one
        view = views_dir / out.name  # the identity's out names the OUT: the view directory carries the OUT's own name
        pins = [f"starts={starts}", f"starts.sha256={_sha(starts)}", f"manifest.sha256={_sha(out / 'manifest.json')}",
                f"served_models.sha256={_sha(out / 'served-models.json')}"]
        assert tool.main([str(out), str(view), *itertools.chain.from_iterable(("--source", item) for item in pins)]) == 0
        assert capsys.readouterr().out == "ok\n"
        identity = json.loads((view / "deployment-identity.json").read_text())
        assert identity["store"]["identity_check"] == PLAIN_NOTE and identity["store"]["replica"] == f"copy-{i}"
        assert identity["store"]["endpoint"] == endpoints[f"copy-{i}"]
        views.append(view)
    return inputs, views, views_dir, tuple(_sha(view / "manifest.json") for view in views)


@needs_view_tool
def test_a_plain_union_is_accepted_on_views_made_by_the_real_identity_view_tool(tmp_path, monkeypatch, capsys):
    """The seam that broke: no test fed the tool's own identities through prepare. The plain union's prepare, on the real
    views, passes the union's identity check (the note) and the pinned description. The records here name no store endpoint
    (see the next test for records that do)."""
    inputs, views, views_dir, call_pins = _plain_union_views(tmp_path, monkeypatch, capsys, records_name_endpoint=False)
    context = _plain_union_accepts(inputs, views, str(views_dir / "union"), call_pins)
    assert context["partition"]["copy_identity"] == PLAIN_DISCLOSURE
    assert context["partition"]["out_manifest_sha256"] == list(call_pins)


@needs_view_tool
def test_a_synthetic_plain_union_whose_records_name_different_store_endpoints_is_accepted_on_the_pinned_description(
        tmp_path, monkeypatch, capsys):
    """Design ruling (b), 25 Oct, for every (OUT, endpoint): on the real views, with a different store endpoint recorded in
    each partition's records, no store-proofs.json entry is needed; each endpoint is named by its view's identity and pinned by
    the private endpoint copies for that view's copy."""
    inputs, views, views_dir, call_pins = _plain_union_views(tmp_path, monkeypatch, capsys, records_name_endpoint=True)
    context = _plain_union_accepts(inputs, views, str(views_dir / "union"), call_pins)
    assert context["partition"]["endpoint_proofs"] == PLAIN_ENDPOINT_PROOFS
    assert context["stores"].several == [{"graph": "rhea-test", "model": "oracle-model", "endpoints": 2, "outs": 2}]
    assert sorted(context["stores"].copies) == ["copy-0", "copy-1"]


def test_unstratified_clusters_against_the_frozen_stratified_matrix_on_a_toy():
    index = F.prereg.QuestionIndex(("a1", "a2", "b1", "b2"), ("A|x", "A|x", "B|x", "B|x"))
    num, den = np.array([0.0, 0.0, 1.0, 1.0]), np.ones(4)
    cfg, _ = ex.frozen_config(F)
    K_mine, _ = ex.cluster_counts(F, F.prereg.ClusterMap.singletons(index.questions), index, cfg)
    K_frozen = F.stats.resample_matrix(index, 10_000, seed=20261026)  # the frozen per-arm path's (stratified)
    mine = F.stats.interval(F.stats.ratio_draws(K_mine, num, den), 0.95)
    frozen = F.stats.interval(F.stats.ratio_draws(K_frozen, num, den), 0.95)
    assert (frozen["lo"], frozen["hi"]) == (0.5, 0.5)
    assert (mine["lo"], mine["hi"]) == (0.0, 1.0)
    assert K_mine.shape == (10_000, 4)


def test_the_script_reproduces_the_frozen_per_arm_path_on_its_own_matrix(world):
    """Only the matrix differs: with the frozen stratified matrix, this script's loading and assembly give the frozen
    path's results section for section (for one OUT, and for the union, where the fixture's one URL lets the frozen
    path read it too)."""
    for batches, record in (([world["single"]], False), (world["parts"], True)):
        cm, rec = _map()
        same = ex.parity(F, _call(world, batches, record=record, ledgers=(world["ledger"],)), cluster_map=cm,
                         map_record=rec)
        assert all(same[s] for s in ex.SECTIONS) and same["settings_differing"] == [], same


# --- refusals, each beside its passing twin -------------------------------------------------------------------------


def test_the_twin_union_passes(world):
    assert _run(world, world["parts"], ledgers=(world["ledger"],))["arms"]


def test_a_question_in_two_partitions_is_refused(world):
    src = _none_record(world["parts"][0])
    shutil.copy2(src, world["parts"][1] / "records" / src.name)
    _recheck(world["parts"][1])
    _refused(world, world["parts"], r"question s01-pass in two partitions")


def test_a_run_id_in_two_partitions_is_refused(world):
    run0 = json.loads(_none_record(world["parts"][0]).read_text(encoding="utf-8"))["run_id"]
    f = _none_record(world["parts"][2])
    r = json.loads(f.read_text(encoding="utf-8"))
    r["run_id"] = run0
    f.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    _recheck(world["parts"][2])
    with pytest.raises(F.refusals, match=rf"run id {run0} in two partitions") as exc:
        _run(world, world["parts"])
    assert "question" not in str(exc.value).split("run id")[0].split("partition problems:")[1]


def test_a_question_outside_its_outs_partition_is_refused(world):
    f = _none_record(world["parts"][0])
    shutil.move(f, world["parts"][2] / "records" / f.name)
    for p in (0, 2):
        _recheck(world["parts"][p])
    _refused(world, world["parts"], r"part-2 \(partition 2\): record .* question is partition 0's")


def test_an_out_without_a_partition_block_is_refused(world):
    """GLM's source natural OUT: its identity has no partition block."""
    _refused(world, [*world["parts"], world["single"]], r"single: no partition block")


def test_an_out_another_partition_names_as_its_source_is_refused(world):
    ident = json.loads((world["parts"][0] / "deployment-identity.json").read_text(encoding="utf-8"))
    ident["derived_from"] = {"path": "/host/part-2", "by": "glm_partition.py build"}
    _write_identity(world["parts"][0], ident)
    _refused(world, world["parts"], r"names as its source .*part-2")


def test_an_out_a_manifest_names_as_its_source_is_refused(world):
    m = json.loads((world["parts"][1] / "manifest.json").read_text(encoding="utf-8"))
    m["entry184"] = {"source_out": str(world["parts"][0])}
    (world["parts"][1] / "manifest.json").write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    _refused(world, world["parts"], r"names as its source .*part-0")


def test_a_voided_out_is_refused_by_its_on_host_path(world):
    voided = _voided(world["tmp"] / "voided-part.jsonl", "/host/part-1")
    _refused(world, world["parts"], r"voided OUTs are never analysed .*part-1", voided=voided)


def test_a_voided_out_is_refused_by_its_path_without_a_partition_too(world):
    voided = _voided(world["tmp"] / "voided-single.jsonl", str(world["single"]))
    _refused(world, [world["single"]], r"voided OUTs", record=False, voided=voided)
    assert _run(world, [world["single"]], record=False)["arms"]  # twin: the other voided record


def test_an_empty_voided_record_is_refused(world):
    empty = world["tmp"] / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    _refused(world, world["parts"], r"no entries", voided=empty)


def test_partitions_on_different_stacks_are_refused(world):
    ident = json.loads((world["parts"][2] / "deployment-identity.json").read_text(encoding="utf-8"))
    ident["stacks_by_route_model"] = {m: "stack-b" for m in MODELS}
    _write_identity(world["parts"][2], ident)
    _refused(world, world["parts"], r"2 stacks")


def _route(out: Path, url: str) -> None:
    for f in (out / "records").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        r["route"]["base_url"] = url
        f.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    _recheck(out)


def test_partitions_through_different_routers_pass_where_check_one_stack_refuses(world):
    for p, out in enumerate(world["parts"]):
        _route(out, f"http://router-{p}:8000/v1")
    assert _run(world, world["parts"])["arms"]  # the union's stack check: one stack named, URLs free
    with pytest.raises(F.refusals, match=r"route base URLs"):  # without a record: the frozen check_one_stack
        for out in world["parts"]:
            ident = json.loads((out / "deployment-identity.json").read_text(encoding="utf-8"))
            del ident["partition"]
            _write_identity(out, ident)
        _run(world, world["parts"], record=False)


def _edit_identity(out: Path, edit) -> None:
    ident = json.loads((out / "deployment-identity.json").read_text(encoding="utf-8"))
    edit(ident)
    _write_identity(out, ident)


def test_a_store_copy_without_its_identity_check_is_refused(world):
    _edit_identity(world["parts"][1], lambda i: i["store"].pop("identity_check"))
    _refused(world, world["parts"], r"part-1: its store copy carries no passed identity check")


def test_a_store_copy_with_another_kind_of_check_is_refused(world):
    _edit_identity(world["parts"][1], lambda i: i["store"].update(identity_check="passed"))
    _refused(world, world["parts"], r"part-1: its store copy carries no passed identity check")


def test_a_parents_copy_without_its_identity_check_is_refused(world):
    _edit_identity(world["parts"][1], lambda i: i.update(parents_store={"replica": "copy-x"}))
    _refused(world, world["parts"], r"part-1: its parents' store copy carries no passed identity check")
    _edit_identity(world["parts"][1], lambda i: i["parents_store"].update(identity_check=CHECK))
    assert _run(world, world["parts"])["arms"]  # twin: the parents' copy checked


def test_a_partition_on_another_copy_than_its_records_is_refused(world):
    _edit_identity(world["parts"][2], lambda i: i["store"].update(replica="copy-0"))
    _refused(world, world["parts"], r"part-2: its store copy \(copy-2, replica copy-0\) is not partition 2's copy")


def test_a_voided_record_without_the_known_voided_outs_is_refused(world):
    partial = world["tmp"] / "partial.jsonl"
    partial.write_text(json.dumps({"out": ex.KNOWN_VOIDED[0]}) + "\n", encoding="utf-8")
    _refused(world, world["parts"], r"lack 4 of the 5 voided OUTs", voided=partial)


def test_an_out_of_the_registered_split_without_its_identity_is_refused(world):
    (world["single"] / "deployment-identity.json").unlink()
    call = ex.Call((world["single"],), (world["questions"],), "s01", "test", (world["voided"],), None, (), 1, None)
    cm, rec = _map()
    with pytest.raises(ex.Refused, match=r"OUTs of the test split without a deployment identity"):
        ex.run(F, call, cluster_map=cm, map_record=rec)
    assert _run(world, [world["single"]], record=False)["arms"]  # twin: a dev call reads it without one


def test_the_question_level_exact_bounds_stay_as_registered(world):
    """Science, design log, 23 Oct 00:57Z: the Clopper-Pearson bounds on question-level events stay beside k and n (§3)."""
    result = _run(world, world["parts"], ledgers=(world["ledger"],))
    events = []

    def walk(x):
        if isinstance(x, dict):
            if "question_event" in x:
                events.append(x["question_event"])
            for v in x.values():
                walk(v)

    walk(result["arms"])
    assert events and all({"k", "n", "exact"} <= set(e) for e in events)
    assert any(e["exact"] is not None for e in events)
    assert result["settings"]["exact_bounds"] == "Clopper-Pearson for binary question-level events only (prereg §3)"


def test_a_deploy_of_other_code_is_refused(tmp_path):
    root = tmp_path / "deploy"
    for sub in ("harness", "scripts", "data/analysis", "data/freeze"):
        shutil.copytree(ROOT / sub, root / sub, ignore=shutil.ignore_patterns("__pycache__"))
    with open(root / "harness" / "analysis" / "stats.py", "a", encoding="utf-8") as fh:
        fh.write("# changed\n")
    out = subprocess.run([sys.executable, "-I", str(HERE.parent / "exploratory.py"), "clusters", "--code-root",
                          str(root), "--out-dir", str(tmp_path / "maps")], capture_output=True, text=True, check=False)
    assert out.returncode == 2 and "code digest" in out.stderr and "not the frozen analysis's" in out.stderr


def test_a_missing_partition_is_refused(world):
    _refused(world, world["parts"][:2], r"holds partitions \[0, 1\], not each of 0..2 once")


def test_a_partition_given_twice_is_refused(world):
    twin = world["tmp"] / "part-1b"
    shutil.copytree(world["parts"][1], twin)
    _refused(world, [*world["parts"], twin], r"holds partitions \[0, 1, 1, 2\]")


def test_a_partition_of_another_record_is_refused(world):
    ident = json.loads((world["parts"][0] / "deployment-identity.json").read_text(encoding="utf-8"))
    ident["partition"]["record_sha256"] = "0" * 64
    _write_identity(world["parts"][0], ident)
    _refused(world, world["parts"], r"part-0: its partition block .* is not a partition of the record")


def test_a_record_of_an_unknown_schema_is_refused(world):
    rec = json.loads(world["record"].read_text(encoding="utf-8"))
    rec["schema"] = "gemma-partition/1"
    world["record"].write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    _refused(world, world["parts"], r"schema 'gemma-partition/1', not one of")


@pytest.mark.parametrize("schema", ["entry184-partition/1", "entry190-partition/1"])
def test_a_record_of_entry_184s_or_190s_schema_is_read(world, schema):
    rec = _record(schema=schema)
    world["record"].write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    sha = hashlib.sha256(world["record"].read_bytes()).hexdigest()
    for out in world["parts"]:
        ident = json.loads((out / "deployment-identity.json").read_text(encoding="utf-8"))
        ident["partition"]["record_sha256"] = sha
        _write_identity(out, ident)
    assert _run(world, world["parts"])["exploratory"]["partition_union"]["schema"] == schema


def test_a_record_not_following_the_rule_is_refused(world):
    rec = json.loads(world["record"].read_text(encoding="utf-8"))
    rec["partitions"][1]["question_id_sha256"] = sorted([*rec["partitions"][1]["question_id_sha256"], _h("s01-pass")])
    world["record"].write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    sha = hashlib.sha256(world["record"].read_bytes()).hexdigest()
    for out in world["parts"]:
        ident = json.loads((out / "deployment-identity.json").read_text(encoding="utf-8"))
        ident["partition"]["record_sha256"] = sha
        _write_identity(out, ident)
    _refused(world, world["parts"], r"partition 1 is not the rule's")


def test_a_partition_out_alone_is_refused(world):
    _refused(world, world["parts"][:1], r"analysed only in their union", record=False)


def test_a_ledger_row_of_no_union_record_is_refused(world):
    rows = world["ledger"].read_text(encoding="utf-8").splitlines()
    i = next(i for i, line in enumerate(rows[1:], 1) if json.loads(line)["source"] == "curated")
    row = json.loads(rows[i])
    row["run_id"] = "not-a-run"
    bad = world["tmp"] / "ledger-bad.jsonl"
    bad.write_text("\n".join([*rows[:i], json.dumps(row), *rows[i + 1:]]) + "\n", encoding="utf-8")
    _refused(world, world["parts"], r"ledger row not-a-run .*no record of the union", ledgers=(bad,))


def test_one_ledger_per_staged_out_is_the_whole_ledger_and_the_union_reads_them(world):
    """driver/partition_union.sbatch exports one ledger per staged OUT, from the staged copy: manifest.json and records/ only (the stager
    leaves out runs.jsonl and logs; the exporter reads nothing else of a directory batch). Here the copy has even less: no replay check,
    identity or served-models file. The K exports hold exactly the rows of one export over the whole fixture, and the union analysis given
    the K files (the job's --ledger arguments, in partition order) equals the one given the whole ledger."""
    tmp, ledgers = world["tmp"], []
    for p, out in enumerate(world["parts"]):
        staged = tmp / f"staged-{p}"
        staged.mkdir()
        shutil.copy2(out / "manifest.json", staged / "manifest.json")
        shutil.copytree(out / "records", staged / "records")
        assert sorted(f.name for f in staged.iterdir()) == ["manifest.json", "records"]
        ledgers.append(tmp / f"ledger-{p}.jsonl")
        assert exporter.main(["--split", "dev", "--records", str(staged), "--out", str(ledgers[-1])]) == 0
    whole = world["ledger"].read_text(encoding="utf-8").splitlines()
    parts = [f.read_text(encoding="utf-8").splitlines() for f in ledgers]
    assert all(rows[0] == whole[0] for rows in parts), "every export has the whole ledger's header"
    assert all(len(rows) > 1 for rows in parts)  # the fixture has branch records in every partition
    assert sorted(r for rows in parts for r in rows[1:]) == sorted(whole[1:])
    one = _run(world, world["parts"], K=HAND_K, ledgers=(world["ledger"],))
    many = _run(world, world["parts"], K=HAND_K, ledgers=tuple(ledgers))
    assert many["arms"] == one["arms"] and many["arms"]
    assert many["real_output_ledger"] == one["real_output_ledger"]
    bare = _run(world, world["parts"], K=HAND_K)  # the union job before this change: no --ledger, so no checker arms
    assert len(many["arms"]) > len(bare["arms"]) and many["real_output_ledger"] != bare["real_output_ledger"]


def test_a_ledger_given_twice_is_refused(world):
    _refused(world, world["parts"], r"in two ledger files", ledgers=(world["ledger"], world["ledger"]))


def test_a_graph_without_a_map_is_refused():
    cfg, _ = ex.frozen_config(F)
    with pytest.raises(ex.Refused, match=r"other-graph: no cluster map"):
        ex.cluster_map_for(F, cfg, "other-graph")


def test_the_pinned_maps_load_by_their_hash_and_the_registered_one_is_the_configs():
    cfg, _ = ex.frozen_config(F)
    sizes = {"rhea-142": 301, "opaque-rhea-142": 301, "text2sparql-ck26": 300}
    for graph, (rel, sha) in ex.CLUSTER_MAPS.items():
        cm, rec = ex.cluster_map_for(F, cfg, graph)
        assert rec["sha256"] == sha == hashlib.sha256((HERE.parent / rel).read_bytes()).hexdigest()
        assert len(cm.clusters) == sizes[graph] and rec["store"] == graph
    _, rec = ex.cluster_map_for(F, cfg, cfg.graph)
    assert rec["sha256"] == cfg.cluster_map_sha256


def test_batches_on_another_question_set_than_the_maps_are_refused(world):
    cm, rec = _map()
    with pytest.raises(F.refusals, match=r"not s01's question set"):
        ex.run(F, _call(world, world["parts"]), cluster_map=cm, map_record={**rec, "questions_sha256": "0" * 64})


# --- science, design log, 23 Oct 00:57Z: corporate, a15 (private) and SIB (each question its own cluster) ----------------


def test_sib_questions_are_each_their_own_cluster_and_nothing_is_read_for_them(world):
    cfg, _ = ex.frozen_config(F)
    cm, rec = ex.cluster_map_for(F, cfg, ex.SIB)
    assert cm is None and rec["sha256"] is None and rec["store"] == "uniprot-sp-2026_03"
    _, fixture = _map()
    result = ex.run(F, _call(world, world["parts"]), cluster_map=None,
                    map_record={**fixture, "rule": "singletons"})  # the rule over the fixture's four dev questions
    assert result["exploratory"]["clusters"] == {"questions": 4, "clusters": 4, "sizes": {"1": 4}, "largest": 1}
    with pytest.raises(ex.Refused, match=r"--cluster-map is given only for a private map"):
        ex.cluster_map_for(F, cfg, ex.SIB, Path("/tmp/x.json"))


def _private_graph(monkeypatch, tmp_path, mode=0o600, inside_git=False):
    """A private map of the fixture's questions (s01-pass and s01-noquery in one cluster), written outside or inside a
    git work tree, and a graph entry pinning it."""
    d = tmp_path / ("repo" if inside_git else "private")
    d.mkdir()
    if inside_git:
        (d / ".git").mkdir()
        (d / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    path = d / "map.json"
    anchors = {"s01-pass": ["e:1"], "s01-noquery": ["e:1"], "s01-flag": [], "s01-abstain": [], "s01-held": []}
    F.prereg.write_cluster_map(path, anchors, {q: "t" for q in anchors})
    path.chmod(mode)
    qsha = hashlib.sha256((FIXTURE / "questions.jsonl").read_bytes()).hexdigest()
    monkeypatch.setitem(ex.GRAPHS, "private-test", {"store": "s01", "questions_sha256": qsha, "map": "private",
                                                     "map_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return path


def test_a_private_map_is_read_by_its_pinned_hash_outside_git(world, monkeypatch):
    path = _private_graph(monkeypatch, world["tmp"])
    call = ex.Call(tuple(world["parts"]), (world["questions"],), "private-test", "dev", (world["voided"],),
                   world["record"], (), 1, None, path)
    result = ex.run(F, call)
    assert result["exploratory"]["clusters"] == {"questions": 4, "clusters": 3, "sizes": {"1": 2, "2": 1}, "largest": 2}
    assert result["exploratory"]["cluster_map"]["path"] == "private"


def test_a_private_map_readable_by_others_is_refused(world, monkeypatch):
    path = _private_graph(monkeypatch, world["tmp"], mode=0o644)
    cfg, _ = ex.frozen_config(F)
    with pytest.raises(ex.Refused, match=r"readable by its owner alone"):
        ex.cluster_map_for(F, cfg, "private-test", path)


def test_a_private_map_inside_a_git_work_tree_is_refused(world, monkeypatch):
    path = _private_graph(monkeypatch, world["tmp"], inside_git=True)
    cfg, _ = ex.frozen_config(F)
    with pytest.raises(ex.Refused, match=r"inside the git work tree"):
        ex.cluster_map_for(F, cfg, "private-test", path)


def test_a_private_map_is_required_and_given_only_for_a_private_graph(world, monkeypatch):
    path = _private_graph(monkeypatch, world["tmp"])
    cfg, _ = ex.frozen_config(F)
    with pytest.raises(ex.Refused, match=r"--cluster-map is required"):
        ex.cluster_map_for(F, cfg, "private-test")
    with pytest.raises(ex.Refused, match=r"--cluster-map is given only for a private map"):
        ex.cluster_map_for(F, cfg, "rhea-142", path)
    with pytest.raises(ex.Refused, match=r"--cluster-map is required"):
        ex.cluster_map_for(F, cfg, ex.A15)


def test_a_private_map_of_another_hash_is_refused(world, monkeypatch):
    path = _private_graph(monkeypatch, world["tmp"])
    monkeypatch.setitem(ex.GRAPHS["private-test"], "map_sha256", "0" * 64)
    cfg, _ = ex.frozen_config(F)
    with pytest.raises(F.refusals, match=r"not the frozen cluster map's"):
        ex.cluster_map_for(F, cfg, "private-test", path)


# check_manifest on a toy, by hand: q1 and q2 share the IRI anchor I, q3 names the key anchor k:x, all in template t

I = "http://x.org/I"
TOY = ({"q1": [I], "q2": [I], "q3": ["k:x"]}, {"q1": "t", "q2": "t", "q3": "t"}, {"q1": "q1", "q2": "q2", "q3": "q3"})


def _toy_manifest():
    return {"anchor_reuse": {"by_template": {"t": {"questions": 3, "anchor_entities": 2, "reused": {I: ["q1", "q2"]},
                                                   "test_naming_a_dev_anchor": 0}}},
            "entity_reuse": {"entities": 4, "shared": {I: [{"id": "q1"}, {"id": "q2"}, {"id": "q9"}]}}}


def test_the_manifest_checks_pass_on_the_toy_and_refuse_each_defect():
    assert ex.check_manifest(_toy_manifest(), *TOY) == [
        "anchor_reuse: equal", "entity_reuse: every shared IRI anchor recorded with its questions"]
    m = _toy_manifest()
    m["anchor_reuse"]["by_template"]["t"]["reused"] = {}
    with pytest.raises(ex.Refused, match=r"anchor_reuse"):
        ex.check_manifest(m, *TOY)
    m = _toy_manifest()
    m["entity_reuse"]["shared"][I] = [{"id": "q1"}]
    with pytest.raises(ex.Refused, match=r"entity_reuse"):
        ex.check_manifest(m, *TOY)
    m = _toy_manifest()
    del m["anchor_reuse"]  # a profile without the unique-anchor rule (a15): entity_reuse alone
    assert ex.check_manifest(m, *TOY) == ["entity_reuse: every shared IRI anchor recorded with its questions"]


def test_the_corporate_map_rebuilds_byte_for_byte_from_the_generators_anchors(tmp_path):
    qs = ROOT / "data" / "questions-frozen" / "text2sparql-ck26"
    rec = ex.build_template_map(F, ex.CORPORATE, qs / "questions.jsonl", qs / "manifest.json", tmp_path / "m.json",
                                tmp_path / "r.json", private=False)
    assert rec["map_sha256"] == ex.CLUSTER_MAPS[ex.CORPORATE][1]
    assert rec["manifest_checks_passed"][0] == "anchor_reuse: equal"


def test_the_template_builder_refuses_other_graphs_and_the_wrong_privacy(tmp_path):
    qs = ROOT / "data" / "questions-frozen" / "text2sparql-ck26"
    args = (qs / "questions.jsonl", qs / "manifest.json", tmp_path / "m.json", tmp_path / "r.json")
    with pytest.raises(ex.Refused, match=r"built here from templates only"):
        ex.build_template_map(F, "rhea-142", *args, private=False)
    with pytest.raises(ex.Refused, match=r"privately exactly when its map is private"):
        ex.build_template_map(F, ex.CORPORATE, *args, private=True)
    with pytest.raises(ex.Refused, match=r"privately exactly when its map is private"):
        ex.build_template_map(F, ex.A15, *args, private=False)
    rhea = ROOT / "data" / "questions-frozen" / "rhea-142" / "questions.jsonl"
    with pytest.raises(ex.Refused, match=r"not text2sparql-ck26's question set"):
        ex.build_template_map(F, ex.CORPORATE, rhea, *args[1:], private=False)


def test_a_label_anchor_is_the_iri_its_other_kind_reading_pairs_it_with():
    """By hand: a15's label_anchor declares no entity parameter; the frozen other_kind_anchor reading pairs its label
    parameter with the entity parameter whose IRI the gold names, so the anchor is that IRI. An unpaired label anchor,
    a pairing to a value that is no IRI, and a label the text does not name are refused, never keyed by the label."""
    base = importlib.import_module("harness.qgen.base")
    generate = importlib.import_module("harness.qgen.generate")
    a15 = importlib.import_module("harness.qgen.a15")
    label = a15.label_anchor("thing", "C", "lab")
    entity = base.Anchor("other", ("name",), "?e rdfs:label $name_lit .", param="ent")
    params = {"lab": "Foo", "lab_entity": "http://x.org/L", "name": "Bar", "ent": "http://x.org/e"}

    class V:  # the fields anchor_ids reads
        anchors = (label, entity)
        readings = (a15.other_kind_anchor("lab_entity", "lab", "thing"),)

    assert ex.anchor_ids(generate, V, params, "Is «Foo» related to «Bar»?") == ["http://x.org/L", "http://x.org/e"]
    with pytest.raises(ex.Refused, match=r"label is not named in its question's text"):
        ex.anchor_ids(generate, V, params, "Is it related?")
    with pytest.raises(ex.Refused, match=r"not one IRI parameter paired"):
        ex.anchor_ids(generate, V, {**params, "lab_entity": "Foo"}, "Is «Foo» related?")

    class Unpaired:
        anchors = (label,)
        readings = ()

    with pytest.raises(ex.Refused, match=r"not one IRI parameter paired"):
        ex.anchor_ids(generate, Unpaired, params, "Is «Foo» here?")

    class W:
        anchors = (entity,)

    assert ex.anchor_ids(generate, W, params, "") == ["http://x.org/e"]


def test_a_private_file_reached_through_a_symlink_or_inside_git_is_refused(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    inside = repo / "map.json"
    inside.write_text("{}", encoding="utf-8")
    link = tmp_path / "outside-link.json"
    link.symlink_to(inside)
    with pytest.raises(ex.Refused, match=r"a symlink"):
        ex.outside_git(link)
    with pytest.raises(ex.Refused, match=r"inside the git work tree"):
        ex.outside_git(repo / "record.json")
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(repo)
    with pytest.raises(ex.Refused, match=r"inside the git work tree"):
        ex.outside_git(linked_dir / "record.json")
    assert ex.outside_git(tmp_path / "private" / "m.json") == (tmp_path / "private" / "m.json").resolve()


def test_corporates_manifest_must_hold_its_anchor_reuse_record():
    m = _toy_manifest()
    del m["anchor_reuse"]
    with pytest.raises(ex.Refused, match=r"no anchor_reuse record"):
        ex.check_manifest(m, *TOY, require_anchor_reuse=True)
    assert ex.CORPORATE in ex.ANCHOR_REUSE_GRAPHS and ex.A15 not in ex.ANCHOR_REUSE_GRAPHS


def test_the_builder_refuses_an_existing_temporary_file_or_symlink(tmp_path):
    qs = ROOT / "data" / "questions-frozen" / "text2sparql-ck26"
    target = tmp_path / "elsewhere.json"
    (tmp_path / "m.json.tmp").symlink_to(target)  # dangling: lexists, not exists
    with pytest.raises(ex.Refused, match=r"its temporary m.json.tmp"):
        ex.build_template_map(F, ex.CORPORATE, qs / "questions.jsonl", qs / "manifest.json", tmp_path / "m.json",
                              tmp_path / "r.json", private=False)
    assert not target.exists() and not (tmp_path / "m.json").exists()


# --- deployment shares and the conditional sensitivity (science, design log, ~03:35Z; entry 189 item 4) -----------------
# By hand: one route URL (the frozen check_one_stack refuses two in one OUT), and the served fingerprint tells the
# deployments apart (attribution.py: an unfronted route's deployment is its URL and fingerprint). Qwen's natural
# ("none") attempts on s01-pass and s01-noquery are served as fp-A, on s01-flag and s01-abstain as fp-B; every other
# record (Qwen's alpha and beta cells, its retries, all of Gemma's) as fp-A. So Qwen's none
# arm is half A, half B, and every other arm all A: the four pairs of none with alpha, beta and beta's two checker arms
# differ by 1/2 and are triggered, the only deployment they share is A, and none restricted to A holds s01-pass and
# s01-noquery, both right (accuracy 1); of 2 x C(5, 2) = 20 pairs, those four are the only ones triggered.

URL, FP_A, FP_B = "http://a.example:8000/v1", "vllm-0.32.0-aaaa0001", "vllm-0.32.0-bbbb0002"
CHECKPOINT, VERDICT = "c0ffee" * 6 + "c0ff", "ab" * 32
ADMIT_B = {"route_model": QWEN, "stack": "stack-a", "system_fingerprint": FP_B, "vllm_version": "0.32.0",
           "checkpoint": CHECKPOINT, "entry170_verdict_sha256": VERDICT, "admitted_at": "2026-10-23T06:00:00Z",
           "change_log": "entry 999 (a test)"}


def _admissions(path: Path, *entries: dict) -> Path:
    path.write_text(json.dumps({"schema": ex.ADMISSIONS_SCHEMA, "admitted": list(entries)}), encoding="utf-8")
    return path
NONE_ON_B = {"s01-flag", "s01-abstain"}
QWEN_ARMS = {f"s01|{QWEN}|{c}" for c in ("alpha|prompt|-", "beta|prompt|-", "beta|checker|curated",
                                          "beta|checker|generic")}


def _serve(out: Path, fp_of) -> None:
    """Every record of the OUT on one route URL, each turn's served fingerprint ``fp_of(record)``; every model's
    baseline (served-models.json's first fingerprint) FP_A."""
    served = json.loads((out / "served-models.json").read_text(encoding="utf-8"))
    for entry in served["models"].values():
        entry["system_fingerprint"] = FP_A
    (out / "served-models.json").write_text(json.dumps(served, indent=1) + "\n", encoding="utf-8")
    for f in (out / "records").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        r["route"]["base_url"] = URL
        for t in r["tokens"]["per_turn"]:
            t["system_fingerprint"] = fp_of(r)
        f.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    _recheck(out)


def _hand_fps(r: dict) -> str:
    natural_b = r["model"] == QWEN and r["load"]["arm"] == "none" and r["question_id"] in NONE_ON_B
    return FP_B if natural_b else FP_A


def _deployments(world, batches, record=True):
    for out in batches:
        _serve(out, _hand_fps)
    return _run(world, batches, record=record, ledgers=(world["ledger"],), admissions=world["admit_b"])["deployments"]


def test_share_comparison_at_exactly_ten_points_and_just_over():
    a10 = ex.share_of(["A"] * 5 + ["B"] * 5)
    assert a10["rows"]["A"] == {"n": 5, "of": 10, "share": 0.5}
    nd = ("mixed", "unattributed")
    exactly = ex.compare_shares(a10, ex.share_of(["A"] * 4 + ["B"] * 6), nd)  # 5/10 - 4/10 = 1/10 exactly
    assert exactly["comparable"] and not exactly["triggered"]
    assert exactly["differences"]["A"] == {"first": "1/2", "second": "2/5", "difference": "1/10"}
    over = ex.compare_shares(ex.share_of(["A"] * 50 + ["B"] * 50), ex.share_of(["A"] * 39 + ["B"] * 61), nd)
    assert over["triggered"]  # 11/100 > 1/10
    at_100 = ex.compare_shares(ex.share_of(["A"] * 50 + ["B"] * 50), ex.share_of(["A"] * 40 + ["B"] * 60), nd)
    assert not at_100["triggered"]  # 10/100, as a float 0.5 - 0.4 = 0.09999999999999998: exact fractions decide
    shared = ex.compare_shares(ex.share_of(["A", "mixed", None]), ex.share_of(["B", "B", "mixed"]), nd)
    assert shared["triggered"] and shared["shared_deployments"] == []  # mixed is no deployment
    assert ex.share_of(["A", None])["without record"] == 1
    assert ex.compare_shares(ex.share_of([None]), a10, nd) == {"comparable": False, "triggered": False}


def test_two_arms_on_different_deployments_are_also_reported_on_the_one_they_share(world):
    dep = _deployments(world, world["parts"])
    none = dep["shares"][ARM]
    assert none["attempts"] == 4 and sorted(r["n"] for r in none["rows"].values()) == [2, 2]
    alpha = dep["shares"][f"s01|{QWEN}|alpha|prompt|-"]
    (a_row,) = alpha["rows"]
    assert alpha["rows"][a_row]["n"] == 4 and a_row in none["rows"]
    assert dep["pairs_compared"] == 20
    assert sorted(p["arms"][0] if p["arms"][1] == ARM else p["arms"][1] for p in dep["triggered_pairs"]) == sorted(
        QWEN_ARMS) and all(ARM in p["arms"] and p["shared_deployments"] == [a_row] for p in dep["triggered_pairs"])
    assert set(dep["restricted"]) == {a_row} and set(dep["restricted"][a_row]) == QWEN_ARMS | {ARM}
    restricted = dep["restricted"][a_row][ARM]
    assert restricted["counts"]["N_att"] == 2 and restricted["metrics"]["accuracy"]["estimate"] == 1.0
    assert "infra_tries" not in restricted["counts"]


def test_an_arm_served_wholly_by_the_shared_deployment_is_unchanged_when_restricted(world):
    for out in world["parts"]:
        _serve(out, _hand_fps)
    result = _run(world, world["parts"], ledgers=(world["ledger"],), admissions=world["admit_b"])
    (a_row,) = result["deployments"]["restricted"]
    label = f"s01|{QWEN}|alpha|prompt|-"
    assert result["deployments"]["restricted"][a_row][label]["metrics"] == result["arms"][label]["metrics"]


def test_the_shares_of_a_union_are_its_one_outs(world):
    union = _deployments(world, world["parts"])
    single = _deployments(world, [world["single"]], record=False)
    cu, cs = union.pop("fingerprint_changes"), single.pop("fingerprint_changes")
    assert union == single
    # the one admitted change (FP_B, Qwen's none on s01-flag and s01-abstain, partition 1) is per OUT: part-1 and single
    assert [c["out"].rsplit("/", 1)[1] for c in cu] == ["part-1"] and [c["out"].rsplit("/", 1)[1] for c in cs] == ["single"]
    assert {(c["system_fingerprint"], c["first_turn"]["run_id"]) for c in cu} == \
        {(c["system_fingerprint"], c["first_turn"]["run_id"]) for c in cs}


def test_a_retry_served_elsewhere_makes_its_output_mixed(world):
    for out in world["parts"]:
        _serve(out, lambda r: FP_B if r["load"]["arm"] == "fb:curated" and r["question_id"] == "s01-flag"
               and r["model"] == QWEN else FP_A)
    dep = _run(world, world["parts"], ledgers=(world["ledger"],), admissions=world["admit_b"])["deployments"]
    curated = dep["shares"][f"s01|{QWEN}|beta|checker|curated"]["rows"]
    assert curated["mixed"]["n"] == 1 and sorted(r["n"] for r in curated.values()) == [1, 3]
    generic = dep["shares"][f"s01|{QWEN}|beta|checker|generic"]["rows"]  # its own retry stayed on A
    assert next(iter(generic.values()))["n"] == 4 and len(generic) == 1


def test_a_retry_without_record_is_its_own_row(world):
    for out in world["parts"]:
        _serve(out, lambda r: FP_A)
    p = K3["s01-flag"]
    retry = next(f for f in (world["parts"][p] / "records").glob("*.json")
                 if json.loads(f.read_text(encoding="utf-8"))["load"]["arm"] == "fb:curated"
                 and json.loads(f.read_text(encoding="utf-8"))["question_id"] == "s01-flag"
                 and json.loads(f.read_text(encoding="utf-8"))["model"] == QWEN)
    retry.unlink()
    _recheck(world["parts"][p])
    dep = _run(world, world["parts"], ledgers=(world["ledger"],))["deployments"]
    curated = dep["shares"][f"s01|{QWEN}|beta|checker|curated"]["rows"]
    assert curated[ex.RETRY_WITHOUT_RECORD]["n"] == 1


def test_a_fronted_record_without_the_fronts_log_is_refused(world):
    f = next((world["single"] / "records").glob("*.json"))
    r = json.loads(f.read_text(encoding="utf-8"))
    r["route"]["front_url"] = "http://front.example:18500/v1"
    f.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    _recheck(world["single"])
    _refused(world, [world["single"]], r"deployment attribution: .*no provenance log", record=False)


def test_the_vendored_attribution_is_the_pinned_one():
    assert hashlib.sha256((HERE.parent / "attribution.py").read_bytes()).hexdigest() == ex.ATTRIBUTION_SHA256


def test_a_branch_cell_of_another_mode_has_no_checker_source(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    arms = {"fb:curated": {"parent": "beta", "by_role": {"must": {"mode": "feedback", "arm": {"name": "curated"}}}},
            "gen:any": {"parent": "beta", "by_role": {"must": {"mode": "generic", "arm": {"name": "any"}}}},
            "fb:census-t5": {"parent": "beta", "by_role": {"must": {"mode": "feedback", "arm": {"name": "census-t5"}}}},
            "dep:curated": {"parent": "beta", "by_role": {"must": {"mode": "deploy", "arm": {"name": "curated"}}}}}
    (out / "manifest.json").write_text(json.dumps({"fingerprint": {"phases": {"branch": {"arms": arms}}}}),
                                       encoding="utf-8")
    assert ex.branch_sources(F, out) == {"fb:curated": "curated", "gen:any": "generic", "fb:census-t5": "census"}
    arms["gen:union"] = {"parent": "beta", "by_role": {"must": {"mode": "generic", "arm": {"name": "union"}}}}
    (out / "manifest.json").write_text(json.dumps({"fingerprint": {"phases": {"branch": {"arms": arms}}}}),
                                       encoding="utf-8")
    assert ex.branch_sources(F, out)["gen:union"] == "generic_retry/union"  # unmapped: the ledger's own name
    arms["fb:curated"]["by_role"]["panel"] = {"mode": "feedback", "arm": {"name": "union"}}
    (out / "manifest.json").write_text(json.dumps({"fingerprint": {"phases": {"branch": {"arms": arms}}}}),
                                       encoding="utf-8")
    with pytest.raises(ex.Refused, match=r"names 2 sources"):
        ex.branch_sources(F, out)


def test_a_retried_output_of_a_plain_run_is_its_own_records_row():
    """By hand: a plain run's checker record holds its retry as its own later turns, so the output's row (A) covers it;
    a grid's retried output with no retry record in the arm's cells is RETRY_WITHOUT_RECORD."""
    from types import SimpleNamespace as NS

    at = ex.load_attribution()
    key = NS(use="checker", source="curated")
    unit = NS(attempt=NS(run_id="r1"), checked=NS(retried=True))
    plain = ex.Deployments({"r1": "A"}, {}, {}, Counter(), 0, {"r1"})
    assert ex.unit_row(at, unit, key, plain, "prompt") == "A"
    grid = ex.Deployments({"r1": "A"}, {}, {"curated": {"fb:curated"}}, Counter(), 0, set())
    assert ex.unit_row(at, unit, key, grid, "prompt") == ex.RETRY_WITHOUT_RECORD
    grid.retries["r1"] = {"fb:curated": ["r2"]}
    grid.served["r2"] = "B"
    assert ex.unit_row(at, unit, key, grid, "prompt") == "mixed"
    grid.served["r2"] = "A"
    assert ex.unit_row(at, unit, key, grid, "prompt") == "A"



# --- served-fingerprint changes (science, design log, ~05:55Z): (a) baseline, (b) the OUT's admission, (c) the config ------
# By hand: one OUT (the fixture whole), Qwen's baseline FP_A (its stack identity: vLLM 0.32.0, snapshot CHECKPOINT); only
# Qwen's beta record on s01-flag (run 6f03924f...) is served with another fingerprint. Its first turn is that record's.

FLAG_RUN = "6f03924f"


def _fp_world(world, fp_b: str, *, runner: dict | None = None) -> Path:
    out = world["single"]
    _serve(out, lambda r: fp_b if r["run_id"].startswith(FLAG_RUN) and r["load"]["arm"] == "beta" else FP_A)
    ident = json.loads((out / "deployment-identity.json").read_text(encoding="utf-8"))
    ident["sitea"] = {"model": "qwen3.8-27b", "system_fingerprint": FP_A,
                     "identity": {"vllm_version": "0.32.0", "snapshot": CHECKPOINT}}
    _write_identity(out, ident)
    if runner is not None:
        (out / "admitted-fingerprints.json").write_text(json.dumps({"schema": "admitted-fingerprints/1", "models": {
            "qwen3.8-27b": runner}}), encoding="utf-8")
    return out


def _runner(fp: str, revision: str = CHECKPOINT) -> dict:
    """admit.py's per-model entry: the baseline and one joined backend with its admission id and verdict."""
    return {"served_model": QWEN, "vllm_version": "0.32.0", "revision": revision, "admitted": [
        {"system_fingerprint": FP_A, "backend": "baseline", "admitted_at": "2026-10-23T06:00:00Z",
         "evidence": {"admission_id": "adm-1", "is_baseline": True}},
        {"system_fingerprint": fp, "backend": "gemma-siteb-2", "admitted_at": "2026-10-23T06:00:00Z",
         "evidence": {"admission_id": "adm-1", "behaviour_verdict": {"file_sha256": VERDICT, "verdict": "pass"}}}]}


def _single(world, admissions: Path):
    return _run(world, [world["single"]], record=False, admissions=admissions)["deployments"]


def _qwen_starts(out: Path) -> tuple[str, dict]:
    """By hand from the records: the changed record's start, and each Qwen initial cell's attempts before and after it."""
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in (out / "records").glob("*.json")]
    cut = next(r["started_at"] for r in recs if r["run_id"].startswith(FLAG_RUN))
    counts = {}
    for cell in ("none", "alpha", "beta"):
        starts = [r["started_at"] for r in recs if r["model"] == QWEN and r["load"]["arm"] == cell]
        counts[cell] = (sum(s < cut for s in starts), sum(s >= cut for s in starts))
    return cut, counts


def test_a_fingerprint_admitted_by_the_outs_own_admission_is_accepted_and_reported(world, tmp_path):
    out = _fp_world(world, FP_B, runner=_runner(FP_B))
    dep = _single(world, _admissions(tmp_path / "empty.json"))
    (change,) = dep["fingerprint_changes"]
    assert (change["route_model"], change["baseline"], change["system_fingerprint"]) == (QWEN, FP_A, FP_B)
    assert change["sources"] == [{"source": "the OUT's admitted-fingerprints.json (admit.py)", "admission_id": "adm-1",
                                  "backend": "gemma-siteb-2", "admitted_at": "2026-10-23T06:00:00Z",
                                  "behaviour_verdict_sha256": VERDICT}]
    cut, counts = _qwen_starts(out)
    assert change["first_turn"]["run_id"].startswith(FLAG_RUN) and change["first_turn"]["started_at"] == cut
    for cell, (before, after) in counts.items():
        arm = change["shares_by_arm"][f"s01|{QWEN}|{cell}|prompt|-"]
        assert (arm["before"]["attempts"], arm["after"]["attempts"]) == (before, after)


def test_a_fingerprint_listed_in_the_config_is_accepted_and_reported(world):
    _fp_world(world, FP_B)
    (change,) = _single(world, world["admit_b"])["fingerprint_changes"]
    assert change["sources"] == [{"source": f"results/{HERE.parent.name}/{ex.ADMISSIONS}", **ADMIT_B}]


def test_an_unlisted_fingerprint_is_refused(world, tmp_path):
    _fp_world(world, FP_B)
    _refused(world, [world["single"]], r"admitted neither by the OUT's admitted-fingerprints.json nor by",
             record=False, admissions=_admissions(tmp_path / "empty.json"))


def test_another_vllm_version_is_refused_even_admitted_and_listed(world, tmp_path):
    other = "vllm-0.33.0-bbbb0002"
    _fp_world(world, other, runner=_runner(other))
    listed = _admissions(tmp_path / "listed.json", {**ADMIT_B, "system_fingerprint": other, "vllm_version": "0.33.0"})
    _refused(world, [world["single"]], r"its vLLM version is not the baseline's", record=False, admissions=listed)


def test_another_checkpoint_is_refused_even_admitted_or_listed(world, tmp_path):
    _fp_world(world, FP_B, runner=_runner(FP_B, revision="dead" * 10))
    _refused(world, [world["single"]], r"admitted by the OUT for checkpoint dead", record=False,
             admissions=_admissions(tmp_path / "empty.json"))
    (world["single"] / "admitted-fingerprints.json").unlink()
    listed = _admissions(tmp_path / "listed.json", {**ADMIT_B, "checkpoint": "dead" * 10})
    _refused(world, [world["single"]], r"listed in admitted-fingerprints.json with vLLM 0.32.0 and checkpoint dead",
             record=False, admissions=listed)


def test_a_config_entry_must_cite_its_ruling(tmp_path):
    bare = {k: v for k, v in ADMIT_B.items() if k != "change_log"}
    with pytest.raises(ex.Refused, match=r"cites no ruling or change-log entry"):
        ex.read_admissions(_admissions(tmp_path / "bare.json", bare))
    with pytest.raises(ex.Refused, match=r"lacks one of"):
        ex.read_admissions(_admissions(tmp_path / "short.json", {**ADMIT_B, "checkpoint": ""}))
    assert ex.read_admissions(_admissions(tmp_path / "ok.json", {**bare, "ruling": "#3 comment (a test)"}))[0]


def test_the_committed_config_is_an_empty_list():
    entries, _ = ex.read_admissions(HERE.parent / ex.ADMISSIONS)
    assert entries == []


def test_the_confirmatory_files_are_the_frozen_ones():
    """Science's condition (design log, ~06:xxZ): the rule lives in the exploratory script only; the analysis-untouched
    list (harness/analysis, the scorer modules and digest.py, the frozen config e46fd504 and its pin) is 847cc275's."""
    digest, n = ex.untouched_list(ROOT)
    assert (digest, n) == (ex.UNTOUCHED_LIST_SHA256, 33)
    assert hashlib.sha256((ROOT / ex.CONFIG).read_bytes()).hexdigest() == ex.FROZEN_CONFIG_SHA256
    if (ROOT / ".git").exists() and shutil.which("git"):  # in a checkout: byte for byte against 847cc275's objects
        for line in subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", ex.FROZEN_COMMIT, "--", "harness",
                                    "scripts", "data"], capture_output=True, text=True, check=True).stdout.splitlines():
            raise AssertionError(f"differs from {ex.FROZEN_COMMIT[:12]}: {line}")



def test_a_mixed_fingerprint_is_refused_even_when_both_parts_are_admitted(world, tmp_path):
    mixed = f"{FP_A} | {FP_B}"
    _fp_world(world, mixed, runner=_runner(FP_B))
    _refused(world, [world["single"]], r"a mixed fingerprint", record=False, admissions=world["admit_b"])


def test_an_unadmitted_fingerprint_in_a_set_aside_try_is_refused(world, tmp_path):
    out = _fp_world(world, FP_A)  # every attempt on the baseline
    src = next(f for f in (out / "records").glob("*.json") if f.name.split(".")[1].startswith(FLAG_RUN))
    r = json.loads(src.read_text(encoding="utf-8"))
    r["run_id"] = "aaaaaaaa-0000-0000-0000-000000000001"  # an earlier infrastructure try of the same attempt
    for t in r["tokens"]["per_turn"]:
        t["system_fingerprint"] = FP_B
    (out / "infrastructure").mkdir(exist_ok=True)
    (out / "infrastructure" / f"{src.name.split('.')[0]}.{r['run_id']}.json").write_text(json.dumps(r, indent=1) + "\n",
                                                                                          encoding="utf-8")
    _recheck(out)
    empty = _admissions(tmp_path / "empty.json")
    _refused(world, [out], r"admitted neither", record=False, admissions=empty)
    assert _single(world, world["admit_b"])["fingerprint_changes"][0]["first_turn"]["run_id"] == r["run_id"]



def test_a_mixed_fingerprint_is_refused_even_as_the_baseline(world):
    mixed = f"{FP_A} | {FP_B}"
    out = _fp_world(world, mixed)
    ident = json.loads((out / "deployment-identity.json").read_text(encoding="utf-8"))
    ident["sitea"]["system_fingerprint"] = mixed
    _write_identity(out, ident)
    _refused(world, [out], r"a mixed fingerprint", record=False, admissions=world["admit_b"])


# --- store copies (science, ~09:50Z: entry 150's in-OUT store changes, the analysis side; the lead's binding,
# ~09:55Z) ----------------------------------------------------------------------------------------------------------------
# By hand: the fixture's 36 records name one endpoint, E0, and every initial cell (none, alpha, beta) of each model has one
# record per question. In the in-OUT change, the records on s01-pass and s01-flag (prompt and branch, both models) stay
# on E0 (copy-a) and are the change's records made before it; those on s01-noquery and s01-abstain name E1 (copy-b). So
# each initial cell's arm is 2 of 4 attempts on copy-a and 2 on copy-b: before the change 2 of 2 on copy-a, after it 2 of
# 2 on copy-b. In the union, partition p's records name its copy's endpoint: each initial cell's arm is 1 of 4 on copy-0
# (s01-pass), 2 on copy-1 (s01-flag, s01-abstain) and 1 on copy-2 (s01-noquery), and in each partition OUT wholly its copy.
# A retry is on its parent's question, so on its parent's copy: no checker arm's row is mixed.

E0 = "http://store.invalid:7039/"
E1 = "http://copy-b.invalid:7040/"
EX = "http://copy-x.invalid:7050/"
REF_SHA, NAME_INDEX = "fe" * 32, "s01@" + "cd" * 32
BEFORE_Q = {"s01-pass", "s01-flag"}
CELLS = ("none", "alpha", "beta")
CHANGE_AT, CHECKED_AT = "2026-10-23T09:18:45Z", "2026-10-23T09:00:00Z"


# The store check's refusals reach an operator whose stderr is private (the check-stores command prints them), so they
# carry no number: no count of records, attempts, entries or watchdogs and no run id (entry 191; entry 199). Every
# refusal that check_stores, read_store_proofs, read_proof_files, reference_pin, read_store_rows or read_private_table
# raises in ANY test of this module is audited, so the cases the fixtures reach are all covered. A digit may stay only inside what is listed:
ALLOWED_IN_A_STORE_REFUSAL = re.compile("|".join([
    r"[0-9a-f]{12,64}",  # a sha256, or its first twelve hex digits (endpoint labels, "(sha256 xxxxxxxxxxxx)")
    r"\d{4}-\d\d-\d\dT[0-9:.]+(?:Z|[+-]\d\d:\d\d)?",  # an ISO 8601 instant
    r"/[^\s:|()\[\]',]+",  # a file system path of an OUT or of a proof file (pytest's tmp dirs hold digits)
    r"(?:records|infrastructure|aborted|harness)/[^\s'\],]+",  # a record NAME (records/<key>.<run id>.json), permitted
    r"copy-[\w-]+|parents",  # the fixtures' replica names
    r"\bchange \d+",  # a manifest store change by its position ("its manifest store change 2 ... where change 1 ...")
    r"\bs0\d\b|\ba15\b",  # the fixtures' graph name, and a15 (a privately pinned graph: its name is a graph's)
    r"sha256s?|load10(?:\.py)?|REFERENCE_STATS_SHA256",  # the names of a hash, a known-answer tool and a constant
    *map(re.escape, ["rhea-142", "opaque-rhea-142", "text2sparql-ck26", "uniprot-sp-2026_03",
                     "entry180-store-check/2", "copy-assignments/1", "e189-reference-pin/1", "e189-store-proofs/1"]),
]))


def _no_number_in(message: str) -> None:
    rest = ALLOWED_IN_A_STORE_REFUSAL.sub(" ", message)
    assert not re.search(r"\d", rest), f"a number in a store refusal: {message!r} (left: {rest!r})"


@pytest.fixture(autouse=True)
def _store_refusals_carry_no_number(monkeypatch):
    def audited(function):
        @functools.wraps(function)
        def inner(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except ex.Refused as exc:
                _no_number_in(str(exc))
                raise
        return inner

    for name in ("check_stores", "read_store_proofs", "read_proof_files", "reference_pin", "read_store_rows",
                 "read_private_table"):
        monkeypatch.setattr(ex, name, audited(getattr(ex, name)))


@pytest.fixture
def ref(monkeypatch):
    """The test graph's recorded reference stats and index, pinned as the real graphs' are."""
    monkeypatch.setitem(ex.REFERENCE_STATS_SHA256, "s01", REF_SHA)
    monkeypatch.setitem(ex.REFERENCE_NAME_INDEX, "s01", NAME_INDEX)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _storecheck(path: Path, ep: str, copy: str, **over) -> Path:
    """partition_run.py storecheck's file of a copy identical to the reference."""
    doc = {"schema": "entry180-store-check/2", "graph": "s01", "replica": copy, "endpoint": ep,
           "url": ep.rstrip("/") + "/?cmd=stats", "checked_utc": CHECKED_AT, "stats_sha256": "0" * 64,
           "reference": "http://ref.invalid:7039", "reference_sha256": REF_SHA,
           "fields": ["name-index", "num-triples-normal"], "no_text_index_fields": [], "differ": [],
           "identical_to_reference": True, **over}
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return path


def _load10(path: Path, ep: str, **over) -> Path:
    """load10.py's output: ten known answers, all passed, at the endpoint (load10 names it without its slash)."""
    doc = {"endpoint": ep.rstrip("/"), "started_utc": "2026-10-23T09:00:05Z", "concurrent": 10,
           "wall_seconds": 0.1, "passed": 10, "results": [{"query": i} for i in range(10)], **over}
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return path


def _copies(path: Path, ep: str, replica: str, **over) -> Path:
    """The store agent's check-copies file with one passed entry of the replica at the endpoint (index and known
    answers), beside another copy's."""
    entry = {"replica": replica, "store": "s01", "endpoint": ep, "identical_stats": True,
             "name_index": NAME_INDEX, "index_list_sha256": NAME_INDEX.split("@")[1], "index_identity_ok": True,
             "load": {"passed": 10, "of": 10}, "ok": True, **over}
    other = {**entry, "replica": "another-copy", "endpoint": "http://another.invalid:28780/"}
    path.write_text(json.dumps({"checked_utc": CHECKED_AT, "copies": [other, entry]}, indent=1) + "\n",
                    encoding="utf-8")
    return path


def _copy_proofs(tmp: Path, name: str, endpoint: str, replica: str) -> tuple[Path, Path]:
    return (_storecheck(tmp / f"sc-{name}.json", endpoint, replica), _load10(tmp / f"ka-{name}.json", endpoint))


def _entry(out: str, endpoint: str, replica: str, files: tuple[Path, Path], **over) -> dict:
    return {"graph": "s01", "out": f"/host/{out}", "endpoint_sha256": _h(endpoint), "replica": replica,
            "storecheck_sha256": _sha(files[0]), "known_answers_sha256": _sha(files[1]),
            "change_log": "entry 150 (a test)", **over}


def _store_config(path: Path, *entries: dict) -> Path:
    path.write_text(json.dumps({"schema": ex.STORE_PROOFS_SCHEMA, "proofs": list(entries)}), encoding="utf-8")
    return path


def _set_endpoints(out: Path, endpoint_of) -> list[str]:
    """Every record of the OUT (records/ and set-aside) on ``endpoint_of(record)``; the run ids of those left on E0."""
    kept = []
    for f in sorted(p for d in ("records", "infrastructure") if (out / d).is_dir() for p in (out / d).glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        r["store"]["endpoint"] = endpoint_of(r)
        if r["store"]["endpoint"] == E0 and f.parent.name == "records":
            kept.append(r["run_id"])
        f.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    _recheck(out)
    return sorted(kept)


def _side(out: Path, endpoint: str) -> dict:
    shared = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["fingerprint"]["shared"]
    return {"endpoint": endpoint, "store": {**shared["store"], "endpoint": endpoint}, "store_client": None}


SRC_FROM = datetime(2026, 10, 23, 9, 10, 0, tzinfo=timezone.utc)  # the old copy's records start here, a second apart
DST_FROM = datetime(2026, 10, 23, 9, 18, 45, tzinfo=timezone.utc)  # the new copy's first record: CHANGE_AT, by hand
MANIFEST_AT = "2026-10-23T09:18:39.962+00:00"  # the fixture's store change, written six seconds before DST_FROM


def _iso(t: datetime) -> str:
    """A time in the runner's form (milliseconds, +00:00)."""
    return t.isoformat(timespec="milliseconds")


def _record_files(out: Path) -> list[Path]:
    return sorted(p for d in ("records", "infrastructure") if (out / d).is_dir() for p in (out / d).glob("*.json"))


def _retime(out: Path, start_of) -> None:
    """Set started_at and ended_at of each record file ``start_of(record, file)`` gives a time: a start (a datetime or
    the text to write; the end is half a second later) or a (start, end) pair; the others are left as they are."""
    for f in _record_files(out):
        r = json.loads(f.read_text(encoding="utf-8"))
        t = start_of(r, f)
        if t is None:
            continue
        start, end = t if isinstance(t, tuple) else (t, None)
        r["started_at"] = start if isinstance(start, str) else _iso(start)
        if end is None:
            end = datetime.fromisoformat(r["started_at"]) + timedelta(milliseconds=500)
        r["ended_at"] = end if isinstance(end, str) else _iso(end)
        f.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    _recheck(out)


def _stamp_by_endpoint(out: Path, src: str, dst: str, src_from: datetime = SRC_FROM,
                       dst_from: datetime = DST_FROM) -> None:
    """Time the OUT's records by their endpoint: those on ``src`` from ``src_from``, those on ``dst`` from ``dst_from``,
    a second apart in file-name order (other endpoints keep their times)."""
    seen: Counter = Counter()

    def start_of(r: dict, f: Path):
        endpoint = r["store"]["endpoint"]
        if endpoint not in (src, dst):
            return None
        seen[endpoint] += 1
        return (src_from if endpoint == src else dst_from) + timedelta(seconds=seen[endpoint] - 1)

    _retime(out, start_of)


def _store_change(out: Path, src: str, dst: str, before: list[str], at: str = MANIFEST_AT, *,
                  stamp: bool = True) -> None:
    """One store_changes entry in run_pilot.check_manifest's form, and (unless ``stamp`` is false) the OUT's records
    timed by their endpoint (the old copy's from SRC_FROM, the new copy's from DST_FROM = CHANGE_AT), as a clean move
    leaves them. The fixture manifest's own "phase branch" additions entry (dated 20 Oct, before every instant stamped
    here) is removed, so a store-change fixture has NO branch phase; _set_branch_phase gives it one (call it after the
    store changes are written)."""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["additions"] = [a for a in manifest.get("additions", []) if "phase branch" not in a.get("added", [])]
    manifest.setdefault("store_changes", []).append(
        {"at": at, "note": "entry 150 (a test)", "changed": ["endpoint", "store.endpoint"],
         "from": _side(out, src), "to": _side(out, dst), "pre_change_records": len(before),
         "pre_change_run_ids": before})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    if stamp:
        _stamp_by_endpoint(out, src, dst)
    else:
        _recheck(out)


def _set_branch_phase(out: Path, at) -> None:
    """The manifest's "phase branch" additions entry as a manifest writes it: ``at`` the instant (text, or any value for
    a malformed case), or None for none; any other entry is kept."""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["additions"] = [a for a in manifest.get("additions", []) if "phase branch" not in a.get("added", [])]
    if at is not None:
        manifest["additions"].append({"added": ["phase branch"], "at": at, "routes": []})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    _recheck(out)


def _moved_world(world) -> dict:
    """The fixture whole, moved from copy-a (E0) to copy-b (E1) by one recorded store change, and both copies' proofs."""
    out = world["single"]
    before = _set_endpoints(out, lambda r: E0 if r["question_id"] in BEFORE_Q else E1)
    _store_change(out, E0, E1, before)
    tmp = world["tmp"]
    a, b = _copy_proofs(tmp, "a", E0, "copy-a"), _copy_proofs(tmp, "b", E1, "copy-b")
    return {"out": out, "before": before, "a": a, "b": b, "proofs": (*a, *b),
            "entries": [_entry("single", E0, "copy-a", a), _entry("single", E1, "copy-b", b, change_at=CHANGE_AT)]}


CLASS_A = {"class": "fast", "label": "fast", "basis": "basis of copy-a", "source": "src"}
CLASS_B = {"class": "slow", "label": "slow (host rule, unmeasured)", "basis": "basis of copy-b", "source": "src"}
CLASS_C = {"class": "medium", "label": "medium", "basis": "basis of copy-c", "source": "src"}


def _classes_file(path: Path, **copies: dict) -> Path:
    """A copy-classes file written by hand (not by copy_classes.build): the fixtures' replicas with classes of their own
    (copy-a fast, copy-b slow by the host rule, copy-c medium) unless ``copies`` give others."""
    doc = {"schema": "e189-copy-classes/1", "rule": "a test", "sources": {"src": {"path": "/src.json", "sha256": "ab" * 32}},
           "copies": {"copy-a": CLASS_A, "copy-b": CLASS_B, "copy-c": CLASS_C, **copies}}
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _stores(world, batches, files, entries, *, record=False, name="proofs.json", **kw):
    config = _store_config(world["tmp"] / name, *entries)
    kw.setdefault("copy_classes", _classes_file(world["tmp"] / "classes.json"))
    return _run(world, batches, record=record, ledgers=(world["ledger"],), store_proofs=tuple(files),
                store_config=config, **kw)["stores"]


def _refused_stores(world, batches, files, entries, match, *, record=False, **kw):
    config = _store_config(world["tmp"] / "refused.json", *entries)
    kw.setdefault("copy_classes", _classes_file(world["tmp"] / "classes.json"))
    _refused(world, batches, match, record=record, ledgers=(world["ledger"],), store_proofs=tuple(files),
             store_config=config, **kw)


def _rows(n_of: dict[str, int]) -> dict:
    total = sum(n_of.values())
    return {"attempts": total, "without record": 0,
            "rows": {d: {"n": n, "of": total, "share": n / total} for d, n in sorted(n_of.items())}}


def test_an_in_out_store_change_with_its_proofs_is_accepted_and_its_shares_reported_before_and_after(world, ref):
    m = _moved_world(world)
    st = _stores(world, [m["out"]], m["proofs"], m["entries"])
    out = str(m["out"])
    assert {(s["model"], s["endpoints"]) for s in st["several_endpoints"]} == {(QWEN, 2), ("google/gemma-4-31B", 2)}
    for model in MODELS:
        for cell in CELLS:
            arm = f"s01|{model}|{cell}|prompt|-"
            assert st["shares"][arm] == _rows({"copy-a": 2, "copy-b": 2})
            assert st["shares_by_out"][out][arm] == st["shares"][arm]
    for arm in QWEN_ARMS:  # the checker arms too: a retry on its parent's copy, never mixed
        assert set(st["shares"][arm]["rows"]) <= {"copy-a", "copy-b"}
    (change,) = st["store_changes"]
    assert (change["out"], change["index"], change["from"], change["to"]) == (out, 1, "copy-a", "copy-b")
    assert change["manifest_pre_change_records"] == len(m["before"]) == 20  # s01-pass's and s01-flag's: 5 cells x 2 models each
    for cell in CELLS:
        split = change["shares_by_arm"][f"s01|{QWEN}|{cell}|prompt|-"]
        assert split == {"before": _rows({"copy-a": 2}), "after": _rows({"copy-b": 2})}
    assert st["copies"]["copy-a"]["listed_by"] == {out: ["its manifest store change 1 (from)"]}
    assert st["copies"]["copy-b"]["listed_by"] == {out: ["its manifest store change 1 (to)"]}
    assert [p["change_at"] for p in st["copies"]["copy-b"]["proofs"]] == [CHANGE_AT]
    assert st["config"]["matched"] == 2 and E1 not in json.dumps(st) and E0 not in json.dumps(st)  # no host named


def test_two_endpoints_without_a_store_change_are_refused_although_check_one_stack_passes_them(world, ref):
    m = _moved_world(world)
    manifest = json.loads((m["out"] / "manifest.json").read_text(encoding="utf-8"))
    del manifest["store_changes"]
    (m["out"] / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    _recheck(m["out"])
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"], r"an endpoint the OUT does not list")


def test_an_unlisted_endpoint_in_a_set_aside_try_is_refused(world, ref):
    m = _moved_world(world)
    src = min((m["out"] / "records").glob("*.json"))
    r = json.loads(src.read_text(encoding="utf-8"))
    r["run_id"], r["store"]["endpoint"] = "aaaaaaaa-0000-0000-0000-000000000002", EX  # an earlier try, on a third copy
    (m["out"] / "infrastructure").mkdir(exist_ok=True)
    (m["out"] / "infrastructure" / f"{src.name.split('.')[0]}.{r['run_id']}.json").write_text(json.dumps(r),
                                                                                             encoding="utf-8")
    _recheck(m["out"])
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"],
                    rf"single {ex.endpoint_label(EX)}: an endpoint the OUT does not list")


def test_a_listed_endpoint_without_its_proof_is_refused(world, ref):
    m = _moved_world(world)
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"][:1], r"no store-proofs.json entry proves its copy")
    _refused_stores(world, [m["out"]], m["a"], m["entries"], r"\(copy-b\): its storecheck file .* was not given")


@pytest.mark.parametrize(("which", "defect", "match"), [
    ("sc", {"identical_to_reference": False}, r"not identical to the reference stats"),
    ("sc", {"differ": ["num-triples-normal"]}, r"not identical to the reference stats"),
    ("sc", {"reference_sha256": "11" * 32}, r"compares with another reference than s01's recorded stats"),
    ("sc", {"fields": ["num-triples-normal"]}, r"does not compare the index \(name-index\)"),
    ("sc", {"endpoint": EX, "url": EX + "?cmd=stats"}, r"storecheck is of another endpoint than the OUT's"),
    ("sc", {"replica": "copy-z"}, r"storecheck is of another graph or replica than s01's copy-b"),
    ("sc", {"checked_utc": "2026-10-23T09:30:00Z"}, r"storecheck was made after the copy took the OUT's attempts"),
    ("sc", {"schema": "entry180-store-check/1"}, r"not a entry180-store-check/2 file"),
    ("ka", {"passed": 9}, r"known answers did not all pass"),
    ("ka", {"results": [{"query": 0}]}, r"known answers did not all pass"),
    ("ka", {"endpoint": EX.rstrip("/")}, r"known answers were run at another endpoint"),
    ("ka", {"started_utc": "2026-10-23T09:30:00Z"}, r"known answers were run after the copy took"),
])
def test_a_listed_endpoint_with_a_failing_proof_is_refused(world, ref, which, defect, match):
    m = _moved_world(world)
    tmp = world["tmp"]
    sc = _storecheck(tmp / "bad-sc.json", E1, "copy-b", **(defect if which == "sc" else {}))
    ka = _load10(tmp / "bad-ka.json", E1, **(defect if which == "ka" else {}))
    bad = _entry("single", E1, "copy-b", (sc, ka), change_at=CHANGE_AT)
    _refused_stores(world, [m["out"]], (*m["a"], sc, ka), [m["entries"][0], bad], rf"\(copy-b\): .*{match}")


def test_a_copy_a_store_change_moved_to_needs_its_change_at(world, ref):
    m = _moved_world(world)
    late = {k: v for k, v in m["entries"][1].items() if k != "change_at"}
    _refused_stores(world, [m["out"]], m["proofs"], [m["entries"][0], late], r"without its change_at")


def test_a_record_on_the_old_copy_after_the_first_on_the_new_one_is_a_mix_and_refused(world, ref):
    """The old placement test, now the timeline's: an A record that starts after the first B record is the copies
    interleaving (A B A), not a late manifest timestamp."""
    m = _moved_world(world)
    f = next(p for p in (m["out"] / "records").glob("*.json") if p.name.split(".")[1] == m["before"][0])
    r = json.loads(f.read_text(encoding="utf-8"))
    assert r["store"]["endpoint"] == E0  # one of the old copy's records ...
    r["started_at"], r["ended_at"] = "2026-10-23T09:20:00.000+00:00", "2026-10-23T09:20:00.500+00:00"
    f.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")  # ... made after the first B record (09:18:45)
    _recheck(m["out"])
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"],
                    r"single: its records' own endpoints and times do not follow its manifest store changes "
                    r"\(interleaved\): .*a mix of copies, not a late timestamp")


def test_known_answers_from_a_check_copies_file_are_accepted_and_another_index_refused(world, ref):
    m = _moved_world(world)
    tmp = world["tmp"]
    good = _copies(tmp / "copies.json", E1, "copy-b")
    entry = {**m["entries"][1], "known_answers_sha256": _sha(good)}
    st = _stores(world, [m["out"]], (*m["proofs"], good), [m["entries"][0], entry])
    assert st["shares"][ARM] == _rows({"copy-a": 2, "copy-b": 2})
    for name, defect, match in (("index", {"index_list_sha256": "ab" * 32}, r"index is not s01's reference index"),
                                ("endpoint", {"endpoint": EX}, r"entry is of another endpoint than the OUT's"),
                                ("load", {"load": {"passed": 10, "of": 11}}, r"did not all pass")):
        bad = _copies(tmp / f"copies-{name}.json", E1, "copy-b", **defect)
        _refused_stores(world, [m["out"]], (*m["proofs"], bad),
                        [m["entries"][0], {**entry, "known_answers_sha256": _sha(bad)}], rf"\(copy-b\): .*{match}")


def test_a_graph_without_a_recorded_reference_has_no_copies_to_prove(world):
    m = _moved_world(world)  # no ``ref``: s01 is not among the pinned graphs
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"], r"s01 has no recorded reference stats here")


def test_one_endpoint_runs_as_before_and_its_share_is_reported(world, ref):
    st = _stores(world, [world["single"]], (), ())  # no proof needed: the frozen fixture names one endpoint
    label = f"endpoint sha256:{_h(E0)[:12]}"
    assert st["several_endpoints"] == [] and st["store_changes"] == []
    for cell in CELLS:
        assert st["shares"][f"s01|{QWEN}|{cell}|prompt|-"] == _rows({label: 4})
    sc, ka = _copy_proofs(world["tmp"], "a", E0, "copy-z")  # an entry given is checked whatever the count
    _refused_stores(world, [world["single"]], (sc, ka), [_entry("single", E0, "copy-a", (sc, ka))],
                    r"storecheck is of another graph or replica than s01's copy-a")


def _union_copies(world) -> tuple[list[Path], list[dict]]:
    """Partition p's records on its copy's endpoint, its identity's store block naming that endpoint with its
    storecheck's sha256 (partition_run.py identity's form), and the copies' proofs and entries."""
    files, entries = [], []
    for p, out in enumerate(world["parts"]):
        endpoint = f"http://copy-{p}.invalid:70{p}0/"
        _set_endpoints(out, lambda r, e=endpoint: e)
        sc, ka = _copy_proofs(world["tmp"], f"p{p}", endpoint, f"copy-{p}")
        check = f"?cmd=stats equal to ref in 2 fields at {CHECKED_AT} (store check sha256 {_sha(sc)})"
        _edit_identity(out, lambda i, e=endpoint, c=check: i["store"].update(endpoint=e, identity_check=c))
        files += [sc, ka]
        entries.append(_entry(out.name, endpoint, f"copy-{p}", (sc, ka)))
    return files, entries


def test_per_partition_copies_with_their_proofs_are_accepted_and_their_shares_reported(world, ref):
    files, entries = _union_copies(world)
    st = _stores(world, world["parts"], files, entries, record=True)
    for model in MODELS:
        for cell in CELLS:
            arm = f"s01|{model}|{cell}|prompt|-"
            assert st["shares"][arm] == _rows({"copy-0": 1, "copy-1": 2, "copy-2": 1})
            for p, out in enumerate(world["parts"]):
                assert st["shares_by_out"][str(out)][arm] == _rows({f"copy-{p}": 2 if p == 1 else 1})
    assert st["copies"]["copy-1"]["listed_by"] == {str(world["parts"][1]): ["its deployment identity's store"]}
    assert st["store_changes"] == []


def test_a_partition_copy_proven_by_another_storecheck_than_its_identity_records_is_refused(world, ref):
    files, entries = _union_copies(world)
    other = _storecheck(world["tmp"] / "sc-p1-later.json", "http://copy-1.invalid:7010/", "copy-1",
                        checked_utc="2026-10-23T09:05:00Z")
    entries[1] = {**entries[1], "storecheck_sha256": _sha(other)}
    _refused_stores(world, world["parts"], [*files, other], entries,
                    r"\(copy-1\): its storecheck is not the one the OUT's identity records", record=True)


def test_an_unlisted_endpoint_in_a_partition_is_refused(world, ref):
    files, entries = _union_copies(world)
    _set_endpoints(world["parts"][2], lambda r: EX if r["model"] == QWEN else "http://copy-2.invalid:7020/")
    _refused_stores(world, world["parts"], files, entries,
                    rf"part-2 {ex.endpoint_label(EX)}: an endpoint the OUT does not list", record=True)


def test_a_partition_moved_within_its_out_is_its_records_copy_and_a_broken_chain_is_refused(world, ref):
    files, entries = _union_copies(world)
    out = world["parts"][1]
    e1, e1b = "http://copy-1.invalid:7010/", "http://copy-1b.invalid:7011/"
    before = _set_endpoints(out, lambda r: E0 if r["question_id"] == "s01-flag" else e1b)  # s01-flag before the move
    _set_endpoints(out, lambda r: e1 if r["question_id"] == "s01-flag" else e1b)
    _store_change(out, e1, e1b, before)
    sc, ka = _copy_proofs(world["tmp"], "p1b", e1b, "copy-1b")
    old = json.loads((out / "deployment-identity.json").read_text(encoding="utf-8"))["store"]
    new = {**old, "replica": "copy-1b", "endpoint": e1b,
           "identity_check": f"?cmd=stats equal to ref in 2 fields at {CHECKED_AT} (store check sha256 {_sha(sc)})"}
    _edit_identity(out, lambda i: i.update(store=new, store_changes=[{"at": CHANGE_AT, "note": "a test", "from": old,
                                                                      "to": new}]))
    entries.append(_entry(out.name, e1b, "copy-1b", (sc, ka), change_at=CHANGE_AT))
    st = _stores(world, world["parts"], [*files, sc, ka], entries, record=True)
    arm = f"s01|{QWEN}|none|prompt|-"
    assert st["shares_by_out"][str(out)][arm] == _rows({"copy-1": 1, "copy-1b": 1})  # s01-flag before, s01-abstain after
    assert st["copies"]["copy-1b"]["listed_by"][str(out)] == ["its deployment identity's store change 1",
                                                              "its manifest store change 1 (to)"]
    (change,) = st["store_changes"]
    assert change["shares_by_arm"][arm] == {"before": _rows({"copy-1": 1}), "after": _rows({"copy-1b": 1})}
    _edit_identity(out, lambda i: i["store"].update(replica="copy-1c"))  # the store block no longer the chain's end
    _refused_stores(world, world["parts"], [*files, sc, ka], entries,
                    r"part-1: its identity's store block is not its last identity store change's copy", record=True)


def test_a_partitions_parents_copy_and_its_own_both_proven_are_accepted(world, ref):
    """GLM's partitions (entry 184): the natural parents copied in ran on the parents' copy, named by the identity's
    parents_store and listed by the build's manifest store change to the partition's copy (its records made before)."""
    files, entries = _union_copies(world)
    out, ep = world["parts"][0], "http://parents.invalid:7045/"
    e0 = "http://copy-0.invalid:7000/"
    before = _set_endpoints(out, lambda r: E0 if r["load"]["arm"] == "none" else e0)  # the parents: s01-pass's none
    _set_endpoints(out, lambda r: ep if r["load"]["arm"] == "none" else e0)
    _store_change(out, ep, e0, before)
    sp, kp = _copy_proofs(world["tmp"], "parents", ep, "parents")
    _edit_identity(out, lambda i: i.update(parents_store={
        "replica": "parents", "endpoint": ep,
        "identity_check": f"?cmd=stats equal to ref in 2 fields at {CHECKED_AT} (store check sha256 {_sha(sp)})"}))
    entries[0] = {**entries[0], "change_at": CHANGE_AT}  # the build's store change moved the OUT to its copy
    parents = _entry(out.name, ep, "parents", (sp, kp))
    st = _stores(world, world["parts"], [*files, sp, kp], [*entries, parents], record=True)
    assert st["shares"][f"s01|{QWEN}|none|prompt|-"] == _rows({"parents": 1, "copy-1": 2, "copy-2": 1})
    assert st["shares"][f"s01|{QWEN}|alpha|prompt|-"] == _rows({"copy-0": 1, "copy-1": 2, "copy-2": 1})
    assert st["copies"]["parents"]["listed_by"] == {str(out): ["its manifest store change 1 (from)"]}
    other = _storecheck(world["tmp"] / "sc-parents-later.json", ep, "parents", checked_utc="2026-10-23T09:05:00Z")
    _refused_stores(world, world["parts"], [*files, sp, kp, other],
                    [*entries, {**parents, "storecheck_sha256": _sha(other)}],
                    r"\(parents\): its storecheck is not the one the OUT's identity records", record=True)


PARENTS, LATER = "http://parents.invalid:7045/", "http://natural-later.invalid:48787/"


def _glm_top_up(world, top_up_endpoint: str) -> tuple[list[Path], list[dict]]:
    """A GLM-like partition (entry 184): its natural parents on the parents' copy, listed by the build's store change
    to its own copy, all proven, but for one parent added after the build (a top-up) on ``top_up_endpoint``, which the
    build's pre_change_run_ids do not hold; and the natural OUT's later copy proven for the natural OUT itself."""
    files, entries = _union_copies(world)
    out, e0 = world["parts"][0], "http://copy-0.invalid:7000/"
    parents = _set_endpoints(out, lambda r: E0 if r["load"]["arm"] == "none" else e0)  # the natural parents
    top_up = parents[-1]
    _set_endpoints(out, lambda r: top_up_endpoint if r["run_id"] == top_up
                   else PARENTS if r["load"]["arm"] == "none" else e0)
    _store_change(out, PARENTS, e0, parents[:-1])
    _retime(out, lambda r, f: "2026-10-23T09:30:00.000+00:00" if r["run_id"] == top_up else None)  # after the build
    sp, kp = _copy_proofs(world["tmp"], "parents", PARENTS, "parents")
    _edit_identity(out, lambda i: i.update(parents_store={
        "replica": "parents", "endpoint": PARENTS,
        "identity_check": f"?cmd=stats equal to ref in 2 fields at {CHECKED_AT} (store check sha256 {_sha(sp)})"}))
    sl, kl = _copy_proofs(world["tmp"], "later", LATER, "later")
    entries[0] = {**entries[0], "change_at": CHANGE_AT}
    return ([*files, sp, kp, sl, kl],
            [*entries, _entry(out.name, PARENTS, "parents", (sp, kp)),
             _entry("natural", LATER, "later", (sl, kl), change_at=CHANGE_AT)])  # the natural OUT's, not the partition's


def test_a_glm_top_up_on_the_parents_copy_after_the_build_is_a_mix_and_refused(world, ref):
    files, entries = _glm_top_up(world, PARENTS)  # the top-up starts at 09:30, long after the first record on e0
    _refused_stores(world, world["parts"], files, entries,
                    r"part-0: its records' own endpoints and times do not follow its manifest store changes "
                    r"\(interleaved\): .*a mix of copies, not a late timestamp", record=True)


def test_a_glm_top_up_on_the_natural_outs_later_copy_is_refused_though_that_copy_is_proven(world, ref):
    files, entries = _glm_top_up(world, LATER)
    _refused_stores(world, world["parts"], files, entries,
                    rf"part-0 {ex.endpoint_label(LATER)}: an endpoint the OUT does not list", record=True)


def test_a_copy_behind_an_endpoint_is_accepted_with_its_proof_and_reported_unsplittable(world, ref):
    files, entries = _union_copies(world)
    e2 = "http://copy-2.invalid:7020/"
    sc, ka = _copy_proofs(world["tmp"], "p2b", e2, "copy-2b")
    behind = _entry("part-2", e2, "copy-2b", (sc, ka), behind_endpoint=True, change_at=CHANGE_AT)
    st = _stores(world, world["parts"], [*files, sc, ka], [*entries, behind], record=True)
    label = "copy-2 then copy-2b (one endpoint, unsplittable)"
    assert st["shares"][ARM] == _rows({"copy-0": 1, "copy-1": 2, label: 1})
    assert st["copies"][label]["split"] == ex.UNSPLITTABLE
    assert st["copies"][label]["switch_instants"] == [CHANGE_AT]
    assert [p["replica"] for p in st["copies"][label]["proofs"]] == ["copy-2", "copy-2b"]
    _refused_stores(world, world["parts"], files, [*entries, behind],
                    r"\(copy-2b\): its storecheck file .* was not given", record=True)
    bad = _storecheck(world["tmp"] / "sc-p2b-bad.json", e2, "copy-2b", identical_to_reference=False)
    _refused_stores(world, world["parts"], [*files, bad, ka], [*entries, {**behind, "storecheck_sha256": _sha(bad)}],
                    r"\(copy-2b\): its storecheck is not identical", record=True)


def test_a_copy_behind_an_endpoint_the_out_does_not_list_is_refused(world, ref):
    sc, ka = _copy_proofs(world["tmp"], "a", E0, "copy-a")  # the fixture whole: its identity names no endpoint
    sb, kb = _copy_proofs(world["tmp"], "b", E0, "copy-b")
    entries = [_entry("single", E0, "copy-a", (sc, ka)),
               _entry("single", E0, "copy-b", (sb, kb), behind_endpoint=True, change_at=CHANGE_AT)]
    _refused_stores(world, [world["single"]], (sc, ka, sb, kb), entries, r"an endpoint the OUT does not list")


# --- the two reviewed allowances (science, ~11:30Z): a forwarded copy's direct-endpoint known answers (GLM e184
# p0/p1/p2's own copies), and the old ident.py identity files (entry 150's UniProt copies 7028 and 7024) -------------

FWD, DIRECT = "http://127.0.0.1:48782/", "http://192.0.2.7:28782/"  # a spoke forward, and its copy's direct endpoint
# spoke_forward_watchdog.sh's lines for one forward, by hand in its format: an ssh started, ssh's own message, its exit
# and a restart, both ssh to the same target; up at 00:52:24Z, well before CHANGE_AT, never ended
FORWARD_LINES = ("2026-10-23T00:52:24Z e184-copy-b ssh pid 101 127.0.0.1:48782 -> 192.0.2.7:28782",
                 "Timeout, server login.invalid not responding.",
                 "2026-10-23T02:10:54Z e184-copy-b ssh pid 101 exited rc=255 after 4710 s",
                 "2026-10-23T02:10:59Z e184-copy-b ssh pid 102 127.0.0.1:48782 -> 192.0.2.7:28782")


def _forward_log(path: Path, lines: tuple[str, ...]) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _forward_world(world, lines: tuple[str, ...] = FORWARD_LINES) -> dict:
    """The fixture whole moved from copy-a (E0) to copy-b behind the spoke forward FWD by one store change: copy-b's
    storecheck through the forward, its known answers in a check-copies run at its direct endpoint, its entry naming
    the forward -> direct mapping and the forward's watchdog log."""
    out, tmp = world["single"], world["tmp"]
    before = _set_endpoints(out, lambda r: E0 if r["question_id"] in BEFORE_Q else FWD)
    _store_change(out, E0, FWD, before)
    a = _copy_proofs(tmp, "a", E0, "copy-a")
    sc, ka = _storecheck(tmp / "sc-fwd.json", FWD, "copy-b"), _copies(tmp / "copies-direct.json", DIRECT, "copy-b")
    log = _forward_log(tmp / "fwd-e184-copy-b.log", lines)
    b = _entry("single", FWD, "copy-b", (sc, ka), change_at=CHANGE_AT,
               forward={"direct_endpoint_sha256": _h(DIRECT), "record_sha256": _sha(log)})
    return {"out": out, "sc": sc, "log": log, "proofs": [*a, sc, ka, log],
            "entries": [_entry("single", E0, "copy-a", a), b]}


def test_a_forwarded_copys_known_answers_at_its_direct_endpoint_are_accepted_with_its_forward_record(world, ref):
    f = _forward_world(world)
    st = _stores(world, [f["out"]], f["proofs"], f["entries"])
    assert st["shares"][ARM] == _rows({"copy-a": 2, "copy-b": 2})  # as the in-OUT store change's: 2 of 4 each
    (proof,) = st["copies"]["copy-b"]["proofs"]
    assert proof["forward"] == {"direct_endpoint": f"endpoint sha256:{_h(DIRECT)[:12]}",
                                "record_sha256": _sha(f["log"])}
    assert "127.0.0.1" not in json.dumps(st) and "192.0.2.7" not in json.dumps(st)  # endpoints by sha256 only
    assert "forward" not in st["copies"]["copy-a"]["proofs"][0] and "not_checked" not in st["copies"]["copy-b"]


@pytest.mark.parametrize(("case", "match"), [
    ("no forward", r"its check-copies entry is of another endpoint than the OUT's"),
    ("another direct endpoint", r"forwards to another endpoint than the entry's direct endpoint"),
    ("record not given", r"its forward record \(sha256 [0-9a-f]{12}\) was not given"),
    ("record not a log", r"its forward record is not a forward's watchdog log"),
    ("two targets", r"its forward record names more than one mapping"),
    ("another listen port", r"listens on another address than the OUT's endpoint"),
    ("up after change_at", r"its forward came up after the copy took the OUT's attempts"),
    ("ended before change_at", r"its forward ended before the copy took the OUT's attempts"),
    ("its one ssh exited before change_at", r"no ssh of its forward was running when the copy took the OUT's attempts"),
    ("lines out of time order", r"lines out of time order"),
    ("restarted after its end", r"starts an ssh after its watchdog ended"),
    ("two watchdogs", r"its forward record is of more than one watchdog"),
    ("load10 at the direct endpoint", r"its known answers were run at another endpoint than the OUT's"),
    ("storecheck at the direct endpoint", r"its storecheck is of another endpoint than the OUT's"),
])
def test_a_forwarded_copy_without_a_record_proving_its_forward_is_refused(world, ref, case, match):
    start = "{} e184-copy-b ssh pid 103 127.0.0.1:{} -> 192.0.2.{}:28782"
    end = "{} e184-copy-b STOP file: watchdog ends"
    lines = {"two targets": (*FORWARD_LINES, start.format("2026-10-23T03:00:00Z", 48782, 8)),
             "another listen port": (start.format("2026-10-23T00:52:24Z", 48783, 7),),
             "up after change_at": (start.format("2026-10-23T09:30:00Z", 48782, 7),),
             "ended before change_at": (*FORWARD_LINES, end.format("2026-10-23T09:00:00Z")),
             "its one ssh exited before change_at": FORWARD_LINES[:3],  # no restart: a stale forward
             "lines out of time order": (FORWARD_LINES[3], FORWARD_LINES[0]),
             "restarted after its end": (FORWARD_LINES[0], end.format("2026-10-23T01:00:00Z"), FORWARD_LINES[3]),
             "two watchdogs": (*FORWARD_LINES, FORWARD_LINES[3].replace("e184-copy-b", "e184-copy-c"))}
    f = _forward_world(world, lines.get(case, FORWARD_LINES))
    tmp, proofs, (a, b) = world["tmp"], f["proofs"], f["entries"]
    if case == "no forward":
        b = {k: v for k, v in b.items() if k != "forward"}
    elif case == "another direct endpoint":
        b = {**b, "forward": {**b["forward"], "direct_endpoint_sha256": _h("http://192.0.2.8:28782/")}}
    elif case == "record not given":
        proofs.remove(f["log"])
    elif case == "record not a log":
        b = {**b, "forward": {**b["forward"], "record_sha256": _sha(f["sc"])}}
    elif case == "load10 at the direct endpoint":  # only a check-copies entry, naming the replica, counts there
        proofs.append(_load10(tmp / "ka-direct.json", DIRECT))
        b = {**b, "known_answers_sha256": _sha(proofs[-1])}
    elif case == "storecheck at the direct endpoint":  # the storecheck stays the forward's
        proofs.append(_storecheck(tmp / "sc-direct.json", DIRECT, "copy-b"))
        b = {**b, "storecheck_sha256": _sha(proofs[-1])}
    _refused_stores(world, [f["out"]], proofs, [a, b], rf"single {ex.endpoint_label(FWD)} \(copy-b\): .*{match}")


def test_a_proof_file_neither_json_nor_a_forwards_log_is_refused(tmp_path):
    (tmp_path / "notes.txt").write_text("forward up, all fine\n", encoding="utf-8")
    with pytest.raises(ex.Refused, match=r"neither JSON nor a forward's watchdog log"):
        ex.read_proof_files([tmp_path / "notes.txt"])


# By hand: what ident.py (its FIELDS: the reference's 13 ?cmd=stats fields, per copy URL) lacks against
# entry180-store-check/2 (every ?cmd=stats field, the replica) and check-copies.sh (index_list_sha256 and
# index_identity_ok, the image its job holds open, settings and settings_ok, all in its ok); its 13 fields hold the
# stats, the name-index and the triple counts, and the known answers are load10.py's, so neither is missing
NOT_CHECKED = ["replica", "index-files", "server-image", "server-command-line", "stats-beyond-reference-fields"]
STATS = {"git-hash-index": "9bebc60", "git-hash-server": "9bebc60", "version-server": "9bebc60",
         "name-index": NAME_INDEX, "num-permutations": 6, "num-predicates-normal": 282, "num-predicates-internal": 3,
         "num-subjects-normal": 720, "num-subjects-internal": 0, "num-objects-normal": 919, "num-objects-internal": 0,
         "num-triples-normal": 4042, "num-triples-internal": 720}
OLD = {"identity_format": "ident.py", "not_checked": NOT_CHECKED}


def _ident(path: Path, copies: dict[str, dict], **over) -> Path:
    """ident.py's output: the reference's 13 stats fields, and per copy, by its URL without its slash, its stats in
    those fields, identical."""
    doc = {"checked_utc": CHECKED_AT, "reference": "http://ref.invalid:7022", "reference_stats": dict(STATS),
           "reference_missing_fields": [],
           "copies": {url.rstrip("/"): {"reachable": True, "stats_seconds": 0.01, "stats": stats, "mismatches": {},
                                        "identical": True} for url, stats in copies.items()}, **over}
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def old_world(world, monkeypatch) -> dict:
    """The fixture whole moved from copy-a (E0) to copy-b (E1), both identity-checked by ident.py only: copy-a in the
    pinned reference file's own copies block (as UniProt's 7024 in ref-7022), copy-b in its own file against a live read
    of the reference (as 7028); load10.py's known answers for each."""
    out, tmp = world["single"], world["tmp"]
    before = _set_endpoints(out, lambda r: E0 if r["question_id"] in BEFORE_Q else E1)
    _store_change(out, E0, E1, before)
    reference = _ident(tmp / "ref-stats.json", {E0: dict(STATS)}, checked_utc="2026-10-23T08:00:00Z")
    monkeypatch.setitem(ex.REFERENCE_STATS_SHA256, "s01", _sha(reference))
    monkeypatch.setitem(ex.REFERENCE_NAME_INDEX, "s01", NAME_INDEX)
    ident_b, ka_a, ka_b = _ident(tmp / "ident-b.json", {E1: dict(STATS)}), _load10(tmp / "ka-a.json", E0), _load10(
        tmp / "ka-b.json", E1)
    return {"out": out, "reference": reference, "ka_b": ka_b, "proofs": [reference, ka_a, ident_b, ka_b],
            "entries": [_entry("single", E0, "copy-a", (reference, ka_a), **OLD),
                        _entry("single", E1, "copy-b", (ident_b, ka_b), change_at=CHANGE_AT, **OLD)]}


def test_old_format_identity_files_are_accepted_with_what_they_do_not_check_reported(world, old_world):
    st = _stores(world, [old_world["out"]], old_world["proofs"], old_world["entries"])
    assert list(ex.IDENT_PY_NOT_CHECKED) == NOT_CHECKED
    assert st["shares"][ARM] == _rows({"copy-a": 2, "copy-b": 2})
    for copy in ("copy-a", "copy-b"):
        assert st["copies"][copy]["not_checked"] == sorted(NOT_CHECKED)
        (proof,) = st["copies"][copy]["proofs"]
        assert proof["identity_format"] == "ident.py" and list(proof["not_checked"]) == NOT_CHECKED
    assert list(st["not_checked"]) == ["copy-a", "copy-b"] and list(st["not_checked"]["copy-b"]) == sorted(NOT_CHECKED)


@pytest.mark.parametrize(("case", "match"), [
    ("no identity_format", r"its storecheck is not a entry180-store-check/2 file"),
    ("a copy's stats differ", (r"its identity check's copy differs from s01's recorded reference in "
                               r"\['num-triples-normal'\]")),
    ("a copy's stats of another type", (r"its identity check's copy differs from s01's recorded reference in "
                                        r"\['num-permutations'\]")),
    ("another reference", r"its identity check compares with other stats than s01's recorded reference"),
    ("reference not given", r"its reference file \(sha256 [0-9a-f]{12}\) was not given"),
    ("endpoint not among its copies", r"its identity check does not have exactly one entry of the OUT's endpoint"),
    ("not identical by its own verdict", r"its identity check's copy is not identical to the reference stats"),
    ("made after change_at", r"its identity check was made after the copy took the OUT's attempts"),
    ("another reference index", r"its identity check's copy does not serve s01's reference index"),
])
def test_a_mismatching_old_format_identity_file_is_refused(world, old_world, monkeypatch, case, match):
    tmp, proofs, (a, b) = world["tmp"], old_world["proofs"], old_world["entries"]
    bad = {"a copy's stats differ": {"copies": {E1: {**STATS, "num-triples-normal": 4043}}},
           "a copy's stats of another type": {"copies": {E1: {**STATS, "num-permutations": 6.0}}},
           "another reference": {"reference_stats": {**STATS, "num-triples-normal": 4043}},
           "endpoint not among its copies": {"copies": {EX: dict(STATS)}},
           "made after change_at": {"checked_utc": "2026-10-23T09:30:00Z"}}
    if case == "no identity_format":
        b = {k: v for k, v in b.items() if k not in OLD}
    elif case == "reference not given":
        proofs.remove(old_world["reference"])
    elif case == "another reference index":
        monkeypatch.setitem(ex.REFERENCE_NAME_INDEX, "s01", "s01@" + "ab" * 32)
    elif case == "not identical by its own verdict":
        doc = json.loads(_ident(tmp / "ident-b-bad.json", {E1: dict(STATS)}).read_text(encoding="utf-8"))
        doc["copies"][E1.rstrip("/")].update(identical=False, mismatches={"num-triples-normal": [4042, 4042]})
        (tmp / "ident-b-bad.json").write_text(json.dumps(doc), encoding="utf-8")
        proofs.append(tmp / "ident-b-bad.json")
        b = {**b, "storecheck_sha256": _sha(proofs[-1])}
    else:
        over = bad[case]
        proofs.append(_ident(tmp / "ident-b-bad.json", over.pop("copies", {E1: dict(STATS)}), **over))
        b = {**b, "storecheck_sha256": _sha(proofs[-1])}
    _refused_stores(world, [old_world["out"]], proofs, [a, b], rf"single {ex.endpoint_label(E1)} \(copy-b\): .*{match}")


# --- an alias (science ~18:1xZ 23 Oct: rhea-c06, named by its OUTs by host name and by its recorded checks by address,
# the same host by the copy table): only the pair pinned in STORE_ALIASES, for ident.py and load10.py files ----------
EA = "http://192.0.2.39:7039/"  # copy-a's address, as its recorded checks name it; E0 is its host name


def _table(path: Path, host_port: str = "store.invalid:7039", addr_port: str = "192.0.2.39:7039", **over) -> Path:
    """The store-copy tooling's copy table (copy-assignments/1): both host:port forms of copy-a, and its endpoints."""
    doc = {"schema": "copy-assignments/1",
           "aliases": {host_port: "copy-a", addr_port: "copy-a", "copy-b.invalid:7040": "copy-b"},
           "copies": {"copy-a": {"copy": "copy-a", "endpoints": [addr_port], "node": "store"},
                      "copy-b": {"copy": "copy-b", "endpoints": ["copy-b.invalid:7040"]}}, **over}
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return path


def _pin_table(monkeypatch, table: Path, alias: str = EA, endpoint: str = E0) -> dict:
    monkeypatch.setitem(ex.STORE_ALIASES, ("s01", _h(endpoint)), (_h(alias), _sha(table)))
    return {"endpoint_sha256": _h(alias), "record_sha256": _sha(table)}


def _host_port(url: str) -> str:
    return url.split("//", 1)[1].rstrip("/")


def _alias_kit(world, monkeypatch, host: str, addr: str) -> dict:
    """What a copy named by ``host`` and checked at ``addr`` is proven by, once for any number of its periods: the
    pinned reference file read from it (at ``addr``), the copy table naming both, the pin of (s01, host) and the entry's
    ``alias`` block and ``further`` points (the reference read from it with its known answers, and a later re-proof)."""
    tmp = world["tmp"]
    reference = _ident(tmp / "ref-stats.json", {}, checked_utc="2026-10-21T19:45:26Z", reference=addr.rstrip("/"))
    monkeypatch.setitem(ex.REFERENCE_STATS_SHA256, "s01", _sha(reference))
    monkeypatch.setitem(ex.REFERENCE_NAME_INDEX, "s01", NAME_INDEX)
    table = _table(tmp / "copy-table.json", _host_port(host), _host_port(addr))
    ka_ref = _load10(tmp / "ka-ref.json", addr, started_utc="2026-10-21T19:47:27Z")
    ident_r = _ident(tmp / "ident-r.json", {addr: dict(STATS)}, checked_utc="2026-10-23T18:16:13Z")
    ka_r = _load10(tmp / "ka-r.json", addr, started_utc="2026-10-23T18:16:13Z")
    further = [{"point": "1: the reference read from it", "storecheck_sha256": _sha(reference),
                "known_answers_sha256": _sha(ka_ref)},
               {"point": "3: its re-proof", "storecheck_sha256": _sha(ident_r), "known_answers_sha256": _sha(ka_r)}]
    return {"reference": reference, "table": table, "ka_ref": ka_ref, "ident_r": ident_r, "ka_r": ka_r,
            "further": further, "alias": _pin_table(monkeypatch, table, addr, host)}


def _alias_world(world, monkeypatch, host: str, addr: str) -> dict:
    """The fixture whole moved from copy-a (``host``, its host name) to copy-b (E1): copy-a proven by an ident.py check
    and load10.py output at its address ``addr``, by the alias pinned for ``host`` and the copy table naming both, with
    its further points: the pinned reference file read from it (at ``addr``) with its known answers, and a later
    re-proof there."""
    out, tmp = world["single"], world["tmp"]
    _set_endpoints(out, lambda r: host if r["question_id"] in BEFORE_Q else E1)
    _store_change(out, host, E1, _ids(out, *BEFORE_Q))
    k = _alias_kit(world, monkeypatch, host, addr)
    ident_a, ka_a = _ident(tmp / "ident-a.json", {addr: dict(STATS)}), _load10(tmp / "ka-a.json", addr)
    b = (_storecheck(tmp / "sc-b.json", E1, "copy-b", reference_sha256=_sha(k["reference"])),
         _load10(tmp / "ka-b.json", E1))
    a = _entry("single", host, "copy-a", (ident_a, ka_a), **OLD, alias=k["alias"], further=k["further"])
    return {"out": out, "table": k["table"], "ka_r": k["ka_r"],
            "proofs": [k["reference"], ident_a, ka_a, k["table"], k["ka_ref"], k["ident_r"], k["ka_r"], *b],
            "entries": [a, _entry("single", E1, "copy-b", b, change_at=CHANGE_AT)]}


@pytest.fixture
def alias_world(world, monkeypatch) -> dict:
    """The first pinned pair's world (rhea-c06's shape): E0 by host name, EA its address."""
    return _alias_world(world, monkeypatch, E0, EA)


def test_a_copy_named_by_host_name_is_proven_by_its_checks_at_its_pinned_alias(world, alias_world):
    st = _stores(world, [alias_world["out"]], alias_world["proofs"], alias_world["entries"])
    assert st["shares"][ARM] == _rows({"copy-a": 2, "copy-b": 2})
    (proof,) = st["copies"]["copy-a"]["proofs"]
    assert proof["alias"] == {"endpoint": f"endpoint sha256:{_h(EA)[:12]}", "record_sha256": _sha(alias_world["table"])}
    assert proof["identity_format"] == "ident.py" and "alias" not in st["copies"]["copy-b"]["proofs"][0]
    assert proof["further"] == alias_world["entries"][0]["further"]
    assert "192.0.2" not in json.dumps(st)  # the alias by its sha256 only


@pytest.mark.parametrize(("case", "match"), [
    ("no alias", r"its identity check does not have exactly one entry of the OUT's endpoint"),
    ("alias not pinned", r"its alias is not the one pinned for this graph's endpoint"),
    ("another alias endpoint", r"its alias is not the one pinned for this graph's endpoint"),
    ("another copy table", r"its alias is not the one pinned for this graph's endpoint"),
    ("record not given", r"its alias record \(sha256 [0-9a-f]{12}\) was not given"),
    ("record not a copy table", r"its alias record is not a copy-assignments/1 copy table"),
    ("host name mapped to another copy", r"its alias record does not map the OUT's endpoint and the alias to copy-a"),
    ("alias not among the copy's endpoints", r"its alias record's copy-a does not list the alias among its endpoints"),
    ("the alias not among its keys", r"its alias record does not hold exactly one key of the entry's alias endpoint"),
    ("load10 at another address", r"its known answers were run at another endpoint than the OUT's"),
    ("storecheck at the alias", r"its storecheck is of another endpoint than the OUT's"),
    ("the reference not read from it", r"its further point \(storecheck sha256 [0-9a-f]{12}\): the graph's pinned reference was not read"),
    ("a further point at another address", r"its further point \(storecheck sha256 [0-9a-f]{12}\): its identity check does not have exactly one entry"),
    ("a further point's files not given", r"its further point \(storecheck sha256 [0-9a-f]{12}\): its files were not given"),
    ("a further point's known answers failed", r"its further point \(storecheck sha256 [0-9a-f]{12}\): its known answers did not all pass"),
])
def test_a_copy_named_by_host_name_without_its_pinned_alias_and_table_is_refused(world, alias_world, monkeypatch, case,
                                                                                 match):
    tmp, proofs, (a, b) = world["tmp"], alias_world["proofs"], alias_world["entries"]
    table = alias_world["table"]
    retable = {"record not a copy table": {"schema": "copy-assignments/0"},
               "host name mapped to another copy": {"aliases": {"store.invalid:7039": "copy-z",
                                                                "192.0.2.39:7039": "copy-a"}},
               "alias not among the copy's endpoints": {"copies": {"copy-a": {"endpoints": ["store.invalid:7039"]}}},
               "the alias not among its keys": {"aliases": {"store.invalid:7039": "copy-a"}}}
    if case == "no alias":
        a = {k: v for k, v in a.items() if k != "alias"}
    elif case == "alias not pinned":
        monkeypatch.delitem(ex.STORE_ALIASES, ("s01", _h(E0)))
    elif case == "another alias endpoint":
        a = {**a, "alias": {**a["alias"], "endpoint_sha256": _h("http://192.0.2.40:7039/")}}
    elif case == "another copy table":
        a = {**a, "alias": {**a["alias"], "record_sha256": "ab" * 32}}
    elif case == "record not given":
        proofs.remove(table)
    elif case in retable:
        proofs.remove(table)
        proofs.append(_table(tmp / "copy-table-bad.json", **retable[case]))
        a = {**a, "alias": _pin_table(monkeypatch, proofs[-1])}
    elif case == "load10 at another address":
        proofs.append(_load10(tmp / "ka-x.json", EX))
        a = {**a, "known_answers_sha256": _sha(proofs[-1])}
    elif case == "the reference not read from it":  # the pinned reference read from another server
        other = _ident(tmp / "ref-other.json", {}, reference="http://ref.invalid:7022")
        monkeypatch.setitem(ex.REFERENCE_STATS_SHA256, "s01", _sha(other))
        proofs += [other, _storecheck(tmp / "sc-b2.json", E1, "copy-b", reference_sha256=_sha(other))]
        b = {**b, "storecheck_sha256": _sha(proofs[-1])}
        a = {**a, "further": [{**a["further"][0], "storecheck_sha256": _sha(other)}, a["further"][1]]}
    elif case == "a further point at another address":
        proofs.append(_ident(tmp / "ident-x.json", {EX: dict(STATS)}))
        a = {**a, "further": [a["further"][0], {**a["further"][1], "storecheck_sha256": _sha(proofs[-1])}]}
    elif case == "a further point's files not given":
        proofs.remove(alias_world["ka_r"])
    elif case == "a further point's known answers failed":
        proofs.append(_load10(tmp / "ka-r-bad.json", EA, passed=9))
        a = {**a, "further": [a["further"][0], {**a["further"][1], "known_answers_sha256": _sha(proofs[-1])}]}
    elif case == "storecheck at the alias":  # the alias counts for ident.py's copy URL and load10.py's endpoint alone
        proofs.append(_storecheck(tmp / "sc-alias.json", EA, "copy-a"))
        a = {**{k: v for k, v in a.items() if k not in OLD}, "storecheck_sha256": _sha(proofs[-1])}
    _refused_stores(world, [alias_world["out"]], proofs, [a, b],
                    rf"single {ex.endpoint_label(E0)} \(copy-a\): .*{match}")


def test_the_aliases_are_the_two_pinned_by_hand():
    # By hand: for each pair the sha256 of the endpoint string the OUTs name (rhea-c06's; Gemma's corporate OUT's
    # "http://" + its node's host name + ":7041/"), of its address as "http://HOST:PORT/" (the copy table's keys), and
    # the sha256sum of the copy table's snapshot (rhea-c06's copy-assignments.json of 18:14:19Z; the corporate pair's
    # copy-assignments-20261025T002051Z.json); the repository names no address
    assert ex.STORE_ALIASES == {
        ("rhea-142", "0d4386b1394edd1aa67f23d0b51d90aa6a66345526be51a164116ce9f7fcd037"): (
            "10d664f0e09a2afd038a0976aba2beb91451353265f9e2d5eb09e038d16f7d6c",
            "2bac52722532af6d4da30d4bd7b92581c55c74673acebf4919c40a3f5b4c3b63"),
        ("text2sparql-ck26", "588836988a85adcc45c488e8cadd0e91d1f5e72e62072b8254eea01f8dd6e7f0"): (
            "39e7e948895a527bbd9da773024e528a52092c5aebc2d17b051725e2fc321616",
            "b5d234e05051a1f29df565e3623814cf120259893a6c818cc025270ae1491fed")}


# A pin is keyed by (graph, the OUT's endpoint sha256) and by nothing else: a second pair, with fixtures of its own
# invention (a node named node-2.test and an address on a private network), behaves as the first does
HOST2, ADDR2 = "http://node-2.test:7041/", "http://10.0.0.1:7041/"


@pytest.fixture
def alias_world_2(world, monkeypatch) -> dict:
    return _alias_world(world, monkeypatch, HOST2, ADDR2)


def test_a_second_pinned_pair_is_accepted_with_a_table_that_maps_both_names_to_the_replica(world, alias_world_2,
                                                                                         monkeypatch):
    w = alias_world_2
    assert ex.STORE_ALIASES[("s01", _h(HOST2))] == (_h(ADDR2), _sha(w["table"]))  # the monkeypatched pin, by hand
    st = _stores(world, [w["out"]], w["proofs"], w["entries"])
    assert st["shares"][ARM] == _rows({"copy-a": 2, "copy-b": 2})
    (proof,) = st["copies"]["copy-a"]["proofs"]
    assert proof["alias"] == {"endpoint": f"endpoint sha256:{_h(ADDR2)[:12]}", "record_sha256": _sha(w["table"])}
    assert "10.0.0.1" not in json.dumps(st) and "node-2" not in json.dumps(st)  # sha256 only
    assert ("text2sparql-ck26", "588836988a85adcc45c488e8cadd0e91d1f5e72e62072b8254eea01f8dd6e7f0") in ex.STORE_ALIASES


@pytest.mark.parametrize("case", ["host name", "address", "both"])
def test_a_second_pinned_pair_is_refused_when_the_table_does_not_map_both_names_to_the_replica(world, alias_world_2,
                                                                                               monkeypatch, case):
    tmp, proofs, (a, b) = world["tmp"], alias_world_2["proofs"], alias_world_2["entries"]
    names = {"host name": {"node-2.test:7041": "copy-z", "10.0.0.1:7041": "copy-a"},
             "address": {"node-2.test:7041": "copy-a", "10.0.0.1:7041": "copy-z"},
             "both": {"node-2.test:7041": "copy-z", "10.0.0.1:7041": "copy-z"}}[case]
    proofs.remove(alias_world_2["table"])
    proofs.append(_table(tmp / "copy-table-bad.json", "node-2.test:7041", "10.0.0.1:7041", aliases=names))
    a = {**a, "alias": _pin_table(monkeypatch, proofs[-1], ADDR2, HOST2)}
    _refused_stores(world, [alias_world_2["out"]], proofs, [a, b],
                    rf"single {ex.endpoint_label(HOST2)} \(copy-a\): its alias record does not map the OUT's endpoint "
                    r"and the alias to copy-a")


def test_a_pin_belongs_to_its_graph_and_its_endpoint_alone(monkeypatch):
    monkeypatch.setitem(ex.STORE_ALIASES, ("s01", _h(HOST2)), (_h(ADDR2), "ab" * 32))
    entry = {"replica": "copy-a", "storecheck_sha256": "cd" * 32, "known_answers_sha256": "ef" * 32,
             "alias": {"endpoint_sha256": _h(ADDR2), "record_sha256": "ab" * 32}}
    ref = ex.Reference("fe" * 32, NAME_INDEX, "a test")
    pinned = "its alias is not the one pinned for this graph's endpoint"

    def says(graph: str, endpoint: str, text: str) -> bool:
        found = ex.prove({**entry, "graph": graph}, graph, endpoint, {}, None, None, False, ref)
        return any(text in problem for problem in found)

    assert not says("s01", HOST2, pinned)  # the pin applies: only its table is missing
    assert says("s01", HOST2, "its alias record (sha256 abababababab) was not given (--store-proof)")
    assert says("s02", HOST2, pinned)  # another graph's call at that endpoint: not the pin
    assert says("s01", E0, pinned)  # another endpoint of that graph
    assert says("s01", ADDR2, pinned)  # the alias is not itself a pinned endpoint


def test_the_store_proofs_config_refuses_malformed_entries(tmp_path, monkeypatch):
    monkeypatch.setitem(ex.REFERENCE_STATS_SHA256, "s01", REF_SHA)
    files = (_storecheck(tmp_path / "sc.json", E1, "copy-b"), _load10(tmp_path / "ka.json", E1))
    good = _entry("single", E1, "copy-b", files)
    bare = {k: v for k, v in good.items() if k != "change_log"}
    fwd = {"direct_endpoint_sha256": "ab" * 32, "record_sha256": "cd" * 32}
    ali = {"endpoint_sha256": "ab" * 32, "record_sha256": "cd" * 32}
    fur = [{"point": "1", "storecheck_sha256": REF_SHA, "known_answers_sha256": "cd" * 32},
           {"point": "3", "storecheck_sha256": "ab" * 32, "known_answers_sha256": "ce" * 32}]
    cases = [(bare, r"cites no ruling"), ({**good, "replica": ""}, r"lacks one of"),
             ({**good, "endpoint_sha256": "x"}, r"endpoint_sha256 is not a sha256"),
             ({**good, "behind_endpoint": True}, r"needs its switch instant"),
             ({**good, "second_period": "yes"}, r"second_period is not true or false"),
             ({**good, "second_period": 1}, r"second_period is not true or false"),
             ({**good, "second_period": True}, r"a later period of a copy \(second_period\) needs its change_at"),
             ({**good, "second_period": True, "behind_endpoint": True, "change_at": CHANGE_AT},
              r"is not also a copy behind an endpoint \(behind_endpoint\)"),
             ({**good, "change_at": "09:18"}, r"change_at is not a UTC time"),
             ({**good, "change_at": "2026-10-23T09:18:45"}, r"change_at is not a UTC time"),
             ({**good, "forward": {"direct_endpoint_sha256": "ab" * 32}}, r"forward is not the sha256s"),
             ({**good, "forward": {**fwd, "record_sha256": "x"}}, r"forward is not the sha256s"),
             ({**good, "forward": {**fwd, "note": "extra"}}, r"forward is not the sha256s"),
             ({**good, "identity_format": "ident.py"}, r"not_checked exactly"),
             ({**good, "not_checked": NOT_CHECKED}, r"not_checked exactly"),
             ({**good, "identity_format": "ident.py", "not_checked": NOT_CHECKED[:-1]}, r"not_checked exactly"),
             ({**good, "identity_format": "entry180-store-check/2", "not_checked": NOT_CHECKED},
              r"not_checked exactly"),
             ({**good, "alias": {"endpoint_sha256": "ab" * 32}}, r"alias is not the sha256s"),
             ({**good, "alias": {**ali, "record_sha256": "x"}}, r"alias is not the sha256s"),
             ({**good, "alias": ali, "forward": fwd}, r"a forward or an alias, not both"),
             ({**good, "alias": ali, **OLD}, r"an alias entry needs its further points"),
             ({**good, "alias": ali, "further": [fur[0]], **OLD}, r"an alias entry needs its further points"),
             ({**good, "alias": ali, "further": [fur[1], {**fur[1], "storecheck_sha256": "aa" * 32,
                                                         "known_answers_sha256": "bb" * 32}], **OLD},
              r"an alias entry needs its further points"),
             ({**good, "alias": ali, "further": [fur[0], {**fur[0], "known_answers_sha256": "bb" * 32}], **OLD},
              r"further points name files distinct"),  # the reference twice
             ({**good, "further": []}, r"further is not a list of points"),
             ({**good, "further": [{**fur[0], "point": " "}]}, r"further is not a list of points"),
             ({**good, "further": [{**fur[0], "note": "extra"}]}, r"further is not a list of points"),
             ({**good, "further": [{**fur[0], "known_answers_sha256": "x"}]}, r"further is not a list of points"),
             ({**good, "alias": ali, "further": [fur[0], {**fur[1], "storecheck_sha256": good["storecheck_sha256"]}],
               **OLD}, r"further points name files distinct from its pair"),
             ({**good, "alias": ali,
               "further": [fur[0], {**fur[1], "known_answers_sha256": good["known_answers_sha256"]}], **OLD},
              r"further points name files distinct from its pair"),
             ({**good, "alias": ali, "further": [fur[0], {**fur[1], "known_answers_sha256": "ef" * 32},
                                                 {**fur[1], "point": "4"}], **OLD},
              r"further points name files distinct from its pair and from one another")]
    for entry, match in cases:
        with pytest.raises(ex.Refused, match=match):
            ex.read_store_proofs(_store_config(tmp_path / "c.json", entry))
    with pytest.raises(ex.Refused, match=r"a second first copy"):
        ex.read_store_proofs(_store_config(tmp_path / "c.json", good, {**good, "replica": "copy-c"}))
    with pytest.raises(ex.Refused, match=r"twice"):
        ex.read_store_proofs(_store_config(tmp_path / "c.json", good, {**good, "behind_endpoint": True,
                                                                        "change_at": CHANGE_AT}))
    later = {**good, "second_period": True, "change_at": CHANGE_AT}
    with pytest.raises(ex.Refused, match=r"twice for one later period \(the same change_at\)"):
        ex.read_store_proofs(_store_config(tmp_path / "c.json", good, later, {**later, "change_at": "2026-10-23T09:18:45.0Z"}))
    # a later period is not a second first copy and not the same replica twice; several of them are the count's matter
    assert len(ex.read_store_proofs(_store_config(tmp_path / "c.json", good, later))[0]) == 2
    assert len(ex.read_store_proofs(_store_config(tmp_path / "c.json", good, later,
                                                  {**later, "change_at": "2026-10-23T09:50:00Z"}, {**good, "second_period": False,
                                                                                                  "replica": "copy-b2",
                                                                                                  "endpoint_sha256": _h(EX)}))[0]) == 4
    assert ex.read_store_proofs(_store_config(tmp_path / "c.json", {**bare, "ruling": "#3 (a test)"}))[0]
    assert ex.read_store_proofs(_store_config(tmp_path / "c.json", {**good, "forward": fwd, **OLD}))[0]
    assert ex.read_store_proofs(_store_config(tmp_path / "c.json", {**good, "alias": ali, "further": fur, **OLD}))[0]


def test_the_committed_store_proofs_config_cites_the_recorded_checks_of_each_entry():
    """Filled from recorded checks only (science, ~10:35Z): each entry's files are cited where they were recorded, with
    the sha256 it names, each change_at by its STATE event (or a build's manifest store change), and each forward by
    its watchdog log; no endpoint is named but by its sha256 (no URL, no IP address); no privately pinned graph has an
    entry; the ~11:30Z allowances only for the OUTs science ruled on. GLM's partition top-up (natural parents made after the build, on the natural OUT's later copies) waits for
    its own store change: no partition entry names a copy the natural OUT moved to, so the top-up refuses as an
    unlisted, unproven endpoint (test_an_unlisted_endpoint_in_a_partition_is_refused) until that change is built."""
    raw = (HERE.parent / ex.STORE_PROOFS).read_text(encoding="utf-8")
    entries, _ = ex.read_store_proofs(HERE.parent / ex.STORE_PROOFS)
    assert entries and "http" not in raw and not re.search(r"\b[0-9]{1,3}(\.[0-9]{1,3}){3}\b", raw)
    for e in entries:
        assert e["graph"] in ex.REFERENCE_STATS_SHA256 and e["graph"] not in ex.PRIVATE_REFERENCE_PINS
        src = e["sources"]
        for kind in ("storecheck", "known_answers"):
            assert src[kind]["sha256"] == e[f"{kind}_sha256"] and src[kind]["recorded"] and src[kind]["hub"]
        assert ("change_at" in e) == ("change_at" in src) and ("forward" in e) == ("forward" in src)
        assert ("alias" in e) == ("alias" in src)
        if "alias" in e:  # the pinned pair, its copy table cited by its sha256
            assert ex.STORE_ALIASES[(e["graph"], e["endpoint_sha256"])] == (e["alias"]["endpoint_sha256"],
                                                                           e["alias"]["record_sha256"])
            assert src["alias"]["sha256"] == e["alias"]["record_sha256"] and src["alias"]["recorded"]
            (reference,) = [f for f in e["further"] if f["storecheck_sha256"] == ex.REFERENCE_STATS_SHA256[e["graph"]]]
            cited = {v["sha256"] for pt in src["points"].values() for v in pt.values() if isinstance(v, dict)}
            assert len(e["further"]) == 2 and {f[k] for f in e["further"] for k in ex.FURTHER_FIELDS[1:]} <= cited
        if "change_at" in e and e.get("second_period"):  # the return's first record (the timeline); the STATE line corroborates
            assert src["change_at"]["timeline"] and src["change_at"]["state"]
        elif "change_at" in e:  # a STATE event by its line's sha256, or a build's manifest store change by its sha256, kept as provenance
            assert (len(src["change_at"].get("line_sha256", "")) == 64
                    or (len(src["change_at"].get("manifest_sha256", "")) == 64 and src["change_at"]["store_change"]))
            if "timeline" in src["change_at"]:  # science 25 Oct 00:10Z (point iv): change_at is the first record on the copy
                assert src["change_at"]["state_instant_before_timeline"] == src["change_at"]["at"]
                assert ex.parse_utc(e["change_at"]) is not None
            else:  # not yet re-derived (its timeline is STOP, or was not run): the STATE-based instant, which refuses
                assert src["change_at"]["at"] == e["change_at"]
        if "forward" in e:
            assert src["forward"]["sha256"] == e["forward"]["record_sha256"] and src["forward"]["recorded"]
    # science 25 Oct 00:10Z (point iv), by hand: the record-derived change_at of every OUT whose store timeline came out CLEAN
    # (the first record on the copy, as the reviewed store-timeline command printed it from the OUTs' records), and the STATE-based
    # instants of the OUTs whose timeline was not run (entry 150's copied UniProt OUTs, Gemma's opaque OUTs), which keep refusing
    rederived = {
        ("opaque-glm-5.3-flash", "opaque-rhea-siteb-17"): "2026-10-23T09:19:22.624+00:00",
        ("opaque-glm-5.3-flash-e184-p0of3", "opaque-rhea-siteb-12"): "2026-10-23T03:37:55.853+00:00",
        ("opaque-glm-5.3-flash-e184-p1of3", "opaque-rhea-siteb-13"): "2026-10-23T03:38:29.618+00:00",
        ("opaque-glm-5.3-flash-e184-p2of3", "opaque-rhea-siteb-14"): "2026-10-23T03:39:05.063+00:00",
        ("rhea-gemma-4-31b", "rhea-siteb-4"): "2026-10-23T18:07:10.343+00:00",
        ("opaque-qwen3.8-27b-e180-p0of6", "opaque-rhea-sitec-c10"): "2026-10-23T21:11:07.494+00:00",
        # the OUTs whose branch phase had started (science's branch-phase ruling, 25 Oct): the first record on the copy
        ("rhea-qwen3-coder-30b-a3b", "rhea-siteb-3"): "2026-10-23T16:14:46.908+00:00",
        ("corporate-gemma-4-31b", "corporate-siteb-2"): "2026-10-23T18:03:02.031+00:00",
        ("opaque-qwen3.8-27b-e180-p2of6", "opaque-rhea-sitec-c11"): "2026-10-23T21:44:04.556+00:00",
        ("opaque-qwen3.8-27b-e180-p3of6", "opaque-rhea-siteb-16"): "2026-10-23T12:26:22.537+00:00",
        ("opaque-qwen3.8-27b-e180-p4of6", "opaque-rhea-sitec-c12"): "2026-10-23T22:16:46.390+00:00",
        ("opaque-qwen3.8-27b-e180-p5of6", "opaque-rhea-siteb-17"): "2026-10-24T14:41:13.696+00:00"}
    second_period_at = {("rhea-gemma-4-31b", "rhea-c06"): "2026-10-24T03:00:07.440+00:00"}
    state_based = {
        ("uniprot-e150-claude-sonnet-5-5", "uniprot-a-c02"): "2026-10-22T18:30:02Z",
        ("uniprot-e150-qwen3.8-27b", "uniprot-a-c02"): "2026-10-23T04:01:39Z",
        ("opaque-gemma-4-31b-e190-p2of3", "opaque-rhea-sitec-c12"): "2026-10-24T01:52:21Z",
        ("opaque-gemma-4-31b-e190-p0of3", "opaque-rhea-sitec-c10"): "2026-10-24T09:56:52Z"}
    got = {(e["out"].rsplit("/", 1)[1], e["replica"]): e["change_at"] for e in entries
           if "change_at" in e and not e.get("behind_endpoint") and not e.get("second_period")}
    assert got == {**rederived, **state_based}
    # the route switch behind an unchanged forward (no manifest store change, so no timeline): its switch instant by hand
    assert {(e["out"].rsplit("/", 1)[1], e["replica"]): e["change_at"] for e in entries if e.get("behind_endpoint")} == {
        ("opaque-glm-5.3-flash-e184-p2of3", "opaque-rhea-siteb-15"): "2026-10-23T07:55:27Z"}
    assert {(e["out"].rsplit("/", 1)[1], e["replica"]): e["change_at"] for e in entries if e.get("second_period")} == second_period_at
    for e in entries:
        key = (e["out"].rsplit("/", 1)[1], e["replica"])
        if key in rederived:
            assert e["sources"]["change_at"]["state_instant_before_timeline"] == e["sources"]["change_at"]["at"]
        if key in state_based:
            assert "timeline" not in e["sources"]["change_at"] and e["sources"]["change_at"]["at"] == e["change_at"]
    natural_moves = {e["replica"] for e in entries if "change_at" in e and not e.get("behind_endpoint")
                     and "-e184-" not in e["out"]}
    assert natural_moves and not [e for e in entries if "-e184-" in e["out"] and e["replica"] in natural_moves]
    # science ~11:30Z, by hand: the forward allowance for GLM's partitions' own copies only, ident.py for entry 150's
    # UniProt OUTs' two copies only
    assert {(e["out"].rsplit("/", 1)[1], e["replica"]) for e in entries if "forward" in e} == {
        ("opaque-glm-5.3-flash-e184-p0of3", "opaque-rhea-siteb-12"),
        ("opaque-glm-5.3-flash-e184-p1of3", "opaque-rhea-siteb-13"),
        ("opaque-glm-5.3-flash-e184-p2of3", "opaque-rhea-siteb-14")}
    # science ~18:1xZ, by hand: the alias, with ident.py, for rhea-c06 as the two Rhea OUTs' first copy only
    rhea_7039 = {(out, "rhea-c06") for out in ("rhea-qwen3-coder-30b-a3b", "rhea-gemma-4-31b")}
    # and (science 25 Oct 00:10Z, point ii) for corporate-c07 as the first copy of Gemma's corporate OUT only
    corporate_7041 = {("corporate-gemma-4-31b", "corporate-c07")}
    assert {(e["out"].rsplit("/", 1)[1], e["replica"]) for e in entries if "identity_format" in e} == {
        (out, copy) for out in ("uniprot-e150-claude-sonnet-5-5", "uniprot-e150-qwen3.8-27b")
        for copy in ("uniprot-c-c04", "uniprot-a-c02")} | rhea_7039 | corporate_7041
    assert {(e["out"].rsplit("/", 1)[1], e["replica"]) for e in entries if "alias" in e} == rhea_7039 | corporate_7041
    # science 25 Oct 00:10Z (point iii): the second period of rhea-c06 in Gemma's Rhea OUT, and no other second period
    assert [(e["out"].rsplit("/", 1)[1], e["replica"]) for e in entries if e.get("second_period")] == [
        ("rhea-gemma-4-31b", "rhea-c06")]
    assert all(e["not_checked"] == NOT_CHECKED for e in entries if "identity_format" in e)
    # science ~15:1xZ 23 Oct (entry 202), by hand: corporate-siteb-2, the arm64 copy ruled IDENTICAL (build variant), is
    # named only by Gemma's corporate OUT's in-OUT move, with the ruling cited; no entry names another corporate copy,
    # and its change_at is before the prompted runner's start (the first attempt on it)
    corporate = [e for e in entries if e["graph"] == "text2sparql-ck26"]
    assert [(e["out"].rsplit("/", 1)[1], e["replica"]) for e in corporate] == [
        ("corporate-gemma-4-31b", "corporate-c07"), ("corporate-gemma-4-31b", "corporate-siteb-2")]
    (siteb_2,) = [e for e in corporate if e["replica"] == "corporate-siteb-2"]
    # the first record on the copy (the timeline's instant) is the prompted runner's first attempt, a moment after the runner's
    # start (18:03:01.826Z, the manifest's phase-prompted addition), not before it as the STATE-based instant was
    assert (ex.parse_utc("2026-10-23T18:03:01.826+00:00") <= ex.parse_utc(siteb_2["change_at"])
            < ex.parse_utc("2026-10-23T18:03:03+00:00") and len(siteb_2["sources"]["arm64_ruling"]["sha256"]) == 64)
    # the opaque partitions' in-OUT moves onto the SiteC copies and siteb-17 (23-24 Oct): each by the cutover storecheck
    # its identity records (no ident.py, forward or alias), change_at the first record on the copy (the STATE restart that
    # resumed the OUT on it is kept in sources as provenance), with the move tool's own lines (runner stopped, manifest change, confirmed restart, prep) cited beside it
    moved = {(e["out"].rsplit("/", 1)[1], e["replica"]) for e in entries if "sitec" in e["replica"]
             or (e["replica"] == "opaque-rhea-siteb-17" and "e180-p5of6" in e["out"])}
    assert moved == {
        ("opaque-qwen3.8-27b-e180-p0of6", "opaque-rhea-sitec-c10"),
        ("opaque-qwen3.8-27b-e180-p2of6", "opaque-rhea-sitec-c11"),
        ("opaque-qwen3.8-27b-e180-p4of6", "opaque-rhea-sitec-c12"),
        ("opaque-qwen3.8-27b-e180-p5of6", "opaque-rhea-siteb-17"),
        ("opaque-gemma-4-31b-e190-p0of3", "opaque-rhea-sitec-c10"),
        ("opaque-gemma-4-31b-e190-p2of3", "opaque-rhea-sitec-c12")}
    for e in entries:
        if (e["out"].rsplit("/", 1)[1], e["replica"]) in moved:
            # science 25 Oct 00:10Z (point i): p5's siteb-17 carries one further point, the fresh check at its own endpoint
            fresh = ("e180-p5of6" in e["out"] and e["replica"] == "opaque-rhea-siteb-17")
            assert not {"forward", "alias", "identity_format", "behind_endpoint"} & set(e)
            assert ("further" in e) == fresh and (len(e.get("further", ())) == 1) == fresh
            at = e["sources"]["change_at"]
            assert at["event"] == "restart" and at["state"].endswith("starts.jsonl")
            assert at["at"] == e["change_at"] or at["state_instant_before_timeline"] == at["at"]
            assert {"runner_stopped", "manifest_store_change", "restart_confirmed", "known_answers_named_by"} <= set(at)


# --- the pinned references (science, ~10:35Z): UniProt (so SIB) and corporate committed, a15's in the private tree -


# By hand, from the recorded reference files themselves (sha256sum of the store agent's
# store-copies-20261021/siteb/ref-7022-stats-20261021T175545Z.json and ref-text2sparql-ck26-stats.json, and the
# name-index each one's reference_stats holds), never from the script
PINNED = {"uniprot-sp-2026_03": ("0513d3f79dfdd75bade876d713e29eb23cfa46962d499e82a36cf3ccf811fee8",
                                 "uniprot-sp-2026_03@9484989e47e8903a78937402a0b3c69b43a12499721ea71d3c940ec9e62a62b1"),
          "text2sparql-ck26": ("61d538bb8e110a01213d0b9c084667a631981f2d2fcc299bc1e0413c7833d5c9",
                               "text2sparql-ck26@c51a9638d28fd0c27b262336aa04df3d9b7faa371d216678c1e8d77e4cbc9691")}
EP = "http://copy.invalid:28741/"


def _real_check(graph: str, **over) -> dict:
    """partition_run.py storecheck's file of a real graph's copy, identical to that graph's pinned reference."""
    return {"schema": ex.STORECHECK_SCHEMA, "graph": graph, "replica": "copy-1", "endpoint": EP,
            "url": EP.rstrip("/") + "/?cmd=stats", "checked_utc": CHECKED_AT,
            "reference_sha256": PINNED.get(graph, ("ab" * 32,))[0], "fields": ["name-index", "num-triples-normal"],
            "differ": [], "identical_to_reference": True, **over}


@pytest.mark.parametrize("graph", sorted(PINNED))
def test_uniprot_and_corporate_references_are_pinned_and_another_reference_refuses(graph):
    assert (ex.REFERENCE_STATS_SHA256[graph], ex.REFERENCE_NAME_INDEX[graph]) == PINNED[graph]
    assert ex.check_storecheck(_real_check(graph), graph, EP, "copy-1", None) == []
    assert ex.check_storecheck(_real_check(graph, reference_sha256="ab" * 32), graph, EP, "copy-1", None) == [
        f"its storecheck compares with another reference than {graph}'s recorded stats"]
    entry = {"replica": "copy-1", "store": graph, "endpoint": EP, "identical_stats": True, "index_identity_ok": True,
             "ok": True, "name_index": PINNED[graph][1], "index_list_sha256": PINNED[graph][1].split("@")[1],
             "load": {"passed": 10, "of": 10}}
    assert ex.check_known_answers({"checked_utc": CHECKED_AT, "copies": [entry]}, graph, EP, "copy-1", None) == []
    other = {**entry, "name_index": f"{graph}@{'ab' * 32}", "index_list_sha256": "ab" * 32}
    assert ex.check_known_answers({"checked_utc": CHECKED_AT, "copies": [other]}, graph, EP, "copy-1", None) == [
        f"its check-copies entry's index is not {graph}'s reference index"]


def test_sib_calls_are_proven_against_the_uniprot_pin():
    assert ex.GRAPHS[ex.SIB]["store"] == "uniprot-sp-2026_03"  # check_stores takes the call's store name
    assert ex.reference_of(ex.GRAPHS[ex.SIB]["store"]).stats_sha256 == PINNED["uniprot-sp-2026_03"][0]


def test_a15s_reference_is_pinned_in_the_private_tree_and_without_its_pin_several_endpoints_refuse():
    """Only the pin file's sha256 is here; a15 is in neither committed table, so without --reference-pin its copies
    have no reference and any several endpoints of it refuse."""
    assert set(ex.PRIVATE_REFERENCE_PINS) == {ex.A15} and len(ex.PRIVATE_REFERENCE_PINS[ex.A15]) == 64
    assert ex.A15 not in ex.REFERENCE_STATS_SHA256 and ex.A15 not in ex.REFERENCE_NAME_INDEX
    assert ex.reference_of(ex.A15).stats_sha256 is None
    assert ex.check_storecheck(_real_check(ex.A15), ex.A15, EP, "copy-1", None) == [
        "a15 has no recorded reference stats here (REFERENCE_STATS_SHA256, or its private pin)"]


def _pin(tmp: Path, monkeypatch, *, reference: str = REF_SHA, graph: str = "s01", mode: int = 0o600,
         inside_git: bool = False, name: str = "pin") -> Path:
    """A private reference pin of the fixture's graph (s01 made privately pinned, as a15 is), pinned by its sha256."""
    d = tmp / (f"{name}-repo" if inside_git else f"{name}-private")
    d.mkdir()
    if inside_git:
        (d / ".git").mkdir()
        (d / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    path = d / "pin.json"
    path.write_text(json.dumps({"schema": ex.REFERENCE_PIN_SCHEMA, "graph": graph, "reference_sha256": reference}),
                    encoding="utf-8")
    path.chmod(mode)
    monkeypatch.setitem(ex.PRIVATE_REFERENCE_PINS, "s01", _sha(path))
    monkeypatch.setitem(ex.REFERENCE_NAME_INDEX, "s01", NAME_INDEX)
    return path


def test_a_privately_pinned_graph_is_proven_against_its_pin_and_its_reference_stays_out_of_the_results(world,
                                                                                                         monkeypatch):
    m = _moved_world(world)  # its storechecks compare with REF_SHA
    pin = _pin(world["tmp"], monkeypatch)
    st = _stores(world, [m["out"]], m["proofs"], m["entries"], reference_pin=pin)
    assert st["config"]["reference"] == {"graph": "s01", "source": f"a private pin (sha256 {_sha(pin)})"}
    assert REF_SHA not in json.dumps(st)
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"],
                    r"s01 has no recorded reference stats here \(REFERENCE_STATS_SHA256, or its private pin\)")
    other = _pin(world["tmp"], monkeypatch, reference="ab" * 32, name="other")
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"],
                    r"its storecheck compares with another reference than s01's recorded stats", reference_pin=other)


def test_a_reference_pin_is_refused_unless_pinned_private_outside_git_and_of_its_graph(world, monkeypatch):
    tmp = world["tmp"]
    pin = _pin(tmp, monkeypatch)
    assert ex.reference_pin("s01", pin) == {"graph": "s01", "reference_sha256": REF_SHA}
    assert ex.reference_pin("s01", None) is None
    with pytest.raises(ex.Refused, match=r"--reference-pin is given only for a privately pinned reference"):
        ex.reference_pin("rhea-142", pin)
    monkeypatch.setitem(ex.PRIVATE_REFERENCE_PINS, "s01", "0" * 64)
    with pytest.raises(ex.Refused, match=r"not its pinned one"):
        ex.reference_pin("s01", pin)
    cases = [({"mode": 0o644, "name": "open"}, r"by its owner alone \(owner-only mode\)"),
             ({"mode": 0o700, "name": "exec"}, r"by its owner alone \(owner-only mode\)"),
             ({"inside_git": True, "name": "git"}, r"inside the git work tree"),
             ({"graph": "other", "name": "graph"}, r"not a e189-reference-pin/1 file of s01"),
             ({"reference": "x", "name": "sha"}, r"naming a sha256")]
    for over, match in cases:
        bad = _pin(tmp, monkeypatch, **over)
        with pytest.raises(ex.Refused, match=match):
            ex.reference_pin("s01", bad)
    monkeypatch.delitem(ex.PRIVATE_REFERENCE_PINS, "s01")  # the call's graph pinned publicly, or not at all
    _refused(world, [world["single"]], r"--reference-pin is given only", reference_pin=pin, record=False)


def test_the_store_reader_reads_only_its_fields():
    """The store reader's fields are STORE_FIELDS alone (the e189 attribution tests' tracking record)."""
    data = json.loads(next((FIXTURE / "batch" / "records").glob("*.json")).read_text(encoding="utf-8"))
    Tracked.reads = []
    row = ex.store_row(_track(data, ""), "records/x.json")
    assert set(Tracked.reads) <= {"run_id", "model", "store", "store.name", "store.endpoint", "started_at",
                                  "ended_at"}, Tracked.reads
    assert (row.store, row.endpoint, row.directory) == ("s01", E0, "records")
    assert (row.started, row.ended) == (data["started_at"], data["ended_at"])  # as written, never parsed or a mtime
    assert list(ex.STORE_FIELDS) == ["run_id", "model", "store.name", "store.endpoint", "started_at", "ended_at"]


class Tracked(dict):
    """A record that logs the path of every key read (get or [])."""

    reads: ClassVar[list[str]] = []

    def __init__(self, data, path=""):
        super().__init__({k: _track(v, f"{path}.{k}" if path else k) for k, v in data.items()})
        self._path = path

    def _note(self, key):
        Tracked.reads.append(f"{self._path}.{key}" if self._path else key)

    def get(self, key, default=None):
        self._note(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self._note(key)
        return super().__getitem__(key)

    def __iter__(self):
        raise AssertionError(f"iterated over {self._path or 'the record'}'s keys")

    def items(self):
        raise AssertionError(f"read every field of {self._path or 'the record'}")

    keys = values = items


def _track(v, path):
    if isinstance(v, dict):
        return Tracked(v, path)
    if isinstance(v, list):
        return [_track(x, f"{path}[*]") for x in v]
    return v


# --- --partitions: the union of a strict subset of the record's partitions -------------------------------------------
#
# Hand-derived (never the script's output), on the three-partition world (K3 above, SPAN's clusters: s01-pass in
# partition 0 and s01-noquery in partition 2 share c1):
# - subset {0, 1}: the questions are s01-abstain, s01-flag, s01-pass; c1 keeps s01-pass alone, so 3 clusters of one, and
#   c1 is cut by the subset (s01-noquery, in partition 2, is outside). With HAND_K over (abstain, flag, c1) the weights
#   of the questions are (1, 0, 2), (2, 1, 0), (0, 2, 1); Qwen's natural attempts are right only on s01-pass, so the
#   draws of accuracy are 2/3, 0 and 1/3, the estimate is 1/3 and the interval is [0, 2/3];
# - subset {0, 2}: s01-noquery and s01-pass, both in c1: one cluster of two, nothing cut.

BASE_COMMIT = "0a81008845ad792dba240aafd22c147781bff382"  # feat/e189-exploratory as it was before --partitions


def _dump(result: dict) -> str:
    """A result as main() writes it."""
    return json.dumps(result, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False)


def _cut(w, ps):
    return [w["parts"][p] for p in ps]


def _block(result: dict) -> dict:
    return result["exploratory"]["partition_union"]


def test_partitions_parse_sorted_and_refuse_anything_else():
    assert ex.parse_partitions("1,3,5") == (1, 3, 5) and ex.parse_partitions("5,1,3") == (1, 3, 5)
    assert ex.parse_partitions("0") == (0,) and ex.parse_partitions("10,2") == (2, 10)
    for bad in ("", ",", "1,,3", "1,", " 1", "1 ,3", "-1", "1.0", "a", "01", "1,1", "1;3"):
        with pytest.raises(argparse.ArgumentTypeError):
            ex.parse_partitions(bad)


def test_the_option_reaches_the_call_and_a_bad_one_is_refused_by_the_parser(world, monkeypatch):
    """A main() that dropped the option would silently analyse the full union."""

    class Stop(Exception):
        def __init__(self, call):
            super().__init__("stop")
            self.call = call

    def fake(_F, call, **_kw):
        raise Stop(call)

    monkeypatch.setattr(ex, "run", fake)
    monkeypatch.setattr(ex, "parity", fake)
    argv = ["--code-root", str(ROOT), "--graph", "s01", "--split", "dev", "--questions", str(world["questions"]),
            "--voided", str(world["voided"]), "--partition-record", str(world["record"]),
            *[a for b in world["parts"] for a in ("--batch", str(b))]]
    for cmd, tail in (("analyse", ["--out", str(world["tmp"] / "o")]), ("parity", [])):
        with pytest.raises(Stop) as seen:
            ex.main([cmd, *argv, "--partitions", "2,0", *tail])
        assert seen.value.call.partitions == (0, 2)
        with pytest.raises(Stop) as seen:
            ex.main([cmd, *argv, *tail])
        assert seen.value.call.partitions is None
        with pytest.raises(SystemExit):
            ex.main([cmd, *argv, "--partitions", "1,1", *tail])


def test_naming_every_partition_is_the_call_without_the_option(world):
    for K in (HAND_K, None):  # a given matrix, and the registered draws
        bare = _run(world, world["parts"], ledgers=(world["ledger"],), K=K)
        for named in ((0, 1, 2), (2, 0, 1)):
            whole = _run(world, world["parts"], ledgers=(world["ledger"],), K=K, partitions=named)
            assert _dump(whole) == _dump(bare)
    assert set(_block(bare)) == {"record", "record_sha256", "schema", "k", "graph", "stack_check"}


def _base_script(tmp: Path):
    """feat/e189-exploratory's exploratory.py before --partitions (git show of BASE_COMMIT), beside the other files of
    its directory, as a second module."""
    blob = f"{BASE_COMMIT}:results/e189-exploratory-20261022/exploratory.py"
    shown = subprocess.run(["git", "-C", str(ROOT), "show", blob], capture_output=True, check=False)
    assert shown.returncode == 0, f"the base commit {BASE_COMMIT} is not in this repository: fetch it"
    where = tmp / "base-script" / HERE.parent.name  # the same directory name: results record it
    where.mkdir(parents=True)
    for f in HERE.parent.iterdir():
        if f.name not in ("exploratory.py", "store-proofs.json", "tests", "dry-run", "__pycache__"):
            (where / f.name).symlink_to(f)
    (where / "exploratory.py").write_bytes(shown.stdout)
    # the base script reads the config of its own commit: the committed one has entries (second_period, further points) that
    # its validation does not know
    config = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:results/e189-exploratory-20261022/store-proofs.json"],
                            capture_output=True, check=False)
    assert config.returncode == 0, f"the base commit {BASE_COMMIT} has no store-proofs.json"
    (where / "store-proofs.json").write_bytes(config.stdout)
    spec = importlib.util.spec_from_file_location("e189_exploratory_base", where / "exploratory.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e189_exploratory_base"] = module
    spec.loader.exec_module(module)
    return module, where


def test_a_call_without_the_option_gives_the_base_commits_results_byte_for_byte(world):
    base, where = _base_script(world["tmp"])
    Fb = base.load_frozen(ROOT)
    for batches, record in ((world["parts"], True), ([world["single"]], False)):
        for K in (HAND_K, None):
            cm, rec = _map()
            old = base.run(Fb, base.Call(tuple(batches), (world["questions"],), "s01", "dev", (world["voided"],),
                                         world["record"] if record else None, (world["ledger"],), 1, None),
                           K_clusters=K, cluster_map=cm, map_record=rec)
            new = _run(world, batches, record=record, ledgers=(world["ledger"],), K=K)
            # science's ruling (B), 25 Oct 2026: the one section a call without replay rows gains, empty (tests/test_online_vs_replay.py)
            assert new.pop("online_vs_replay") == {"rule": ex.ONLINE_VS_REPLAY_RULE, "arms": {}}
            assert new["stores"].pop("replay_class_check") == {"rule": ex.REPLAY_CLASS_RULE,
                                                                  "labels": {k: v.format(taken="-") for k, v in ex.REPLAY_CLASS_LABEL_TEMPLATES.items()},
                                                                  "routing": [], "arms": {}}
            # The two things science's 25 Oct 00:10Z ruling changed in a call without store changes, written by hand:
            # the reader's fields (started_at and ended_at) and the store section's rule sentence
            old["stores"]["reader"] = {"fields": ["run_id", "model", "store.name", "store.endpoint", "started_at",
                                                  "ended_at"]}
            old["stores"]["rule"] += ("; science, 25 Oct 00:10Z: the records' own recorded endpoint is the ground truth; "
                                      "a store change's change_at is the start of the first record on the new "
                                      "endpoint; records of the two copies interleaving is a mix and refuses"
                                      ". " + BRANCH_RULE_TEXT)
            # the two calls read different store-proofs.json files (each commit's own): only the file's digest and entry count
            # differ in the stores section, by hand
            old["stores"]["config"]["sha256"] = new["stores"]["config"]["sha256"]
            old["stores"]["config"]["entries"] = new["stores"]["config"]["entries"]
            # what the branch-phase ruling (25 Oct ~04:55Z) and the copy classes added to the store section, by hand: the
            # committed copy-classes.json's path and sha256 in the config, and no OUT with store changes and a branch phase
            classes = {"path": str(HERE.parent / "copy-classes.json"), "sha256": _sha(HERE.parent / "copy-classes.json")}
            assert new["stores"]["config"]["copy_classes"] == classes
            old["stores"]["config"]["copy_classes"] = classes
            # what the endpoint copies added (analysis lead 4, 25 Oct ~08:00Z): the committed endpoint-copies.json's path and
            # sha256 in the config, and nothing else: this call's graph has no entry in the table, so no label changes
            table_record = {"path": str(HERE.parent / "endpoint-copies.json"),
                            "sha256": _sha(HERE.parent / "endpoint-copies.json")}
            assert new["stores"]["config"]["endpoint_copies"] == table_record
            old["stores"]["config"]["endpoint_copies"] = table_record
            old["stores"]["branch_phase_copies"] = []
            # what addendum 1 (25 Oct ~06:2xZ) added, by hand: the retried-unit rule sentence, and the class and attempt shares.
            # This call's copies are endpoint labels (no proof entry, no replica), so every unit's class label is "no copy
            # class"; a prompt arm has one attempt per unit, a checker arm one more per retry, and the fixture's one retried
            # output per checker arm (s01-flag, retried on its own copy; both models) makes 5 attempts of its 4 units.
            assert new["stores"]["retried_unit_rule"] == ex.STORE_UNIT_RULE
            old["stores"]["retried_unit_rule"] = ex.STORE_UNIT_RULE
            none = "no copy class (no entry in the copy classes)"

            def one_class(n: int) -> dict:
                return {"attempts": n, "without record": 0, "rows": {none: {"n": n, "of": n, "share": 1.0}}}

            def per_copy(n_of: dict) -> dict:
                total = sum(n_of.values())
                return {"attempts": total, "without record": 0,
                        "rows": {k: {"n": n, "of": total, "share": n / total} for k, n in sorted(n_of.items())}}

            shares = old["stores"]["shares"]
            checker = {arm for arm in shares if "|checker|" in arm}
            assert len(checker) == 4 and len(shares) == 10  # two models x (none, alpha, beta, curated, generic)
            old["stores"]["class_shares"] = {arm: one_class(4) for arm in shares}
            old["stores"]["attempt_shares"] = {
                arm: (per_copy({k: (5 if arm in checker else 4) for k in shares[arm]["rows"]}) if arm in checker
                      else shares[arm]) for arm in shares}
            by_out = old["stores"]["shares_by_out"]
            old["stores"]["class_shares_by_out"] = {out: {arm: one_class(sh["attempts"]) for arm, sh in arms.items()}
                                                    for out, arms in by_out.items()}
            # (s01-flag is in partition 1 and in the whole fixture: only those OUTs hold the retried output)
            old["stores"]["attempt_shares_by_out"] = {
                out: {arm: (per_copy({k: v["n"] + (1 if out.endswith(("part-1", "single")) else 0)
                                      for k, v in sh["rows"].items()}) if arm in checker else sh)
                      for arm, sh in arms.items()} for out, arms in by_out.items()}
            assert _dump(new) == _dump(old).replace(str(where), str(HERE.parent))  # the script's own directory


def test_an_out_of_a_partition_the_option_does_not_name_is_refused(world):
    _refused(world, world["parts"], r"part-2: partition 2, which --partitions \[0, 1\] does not name",
             partitions=(0, 1))
    assert _run(world, _cut(world, (0, 1)), partitions=(0, 1))["arms"]  # the twin
    _refused(world, _cut(world, (0, 1, 2)), r"part-0: partition 0, which --partitions \[1, 2\] does not name",
             partitions=(1, 2))
    assert _run(world, _cut(world, (1, 2)), partitions=(1, 2))["arms"]


def test_a_named_partition_that_is_missing_is_refused(world):
    _refused(world, _cut(world, (0,)), r"not each of the named \[0, 1\] once", partitions=(0, 1))
    _refused(world, _cut(world, (2,)), r"not each of the named \[0, 2\] once", partitions=(0, 2))
    _refused(world, _cut(world, (0, 1)), r"not each of 0\.\.2 once", partitions=(0, 1, 2))
    _refused(world, _cut(world, (0, 1)), r"not each of 0\.\.2 once")  # the call without the option, as before


def test_a_bad_option_is_refused(world):
    _refused(world, world["parts"], r"needs --partition-record", record=False, partitions=(0, 1))
    _refused(world, _cut(world, (0, 1)), r"not all of 0\.\.2, the record's partitions", partitions=(0, 3))
    _refused(world, _cut(world, (0, 1)), r"not partition indices, each once", partitions=(1, 1))
    _refused(world, _cut(world, (1,)), r"union of at least two", partitions=(1,))
    _refused(world, _cut(world, (0, 1)), r"not partition indices, each once", partitions=(0, True))


def test_the_record_is_checked_in_full_beside_a_subset(world):
    record = json.loads(world["record"].read_text(encoding="utf-8"))
    record["partitions"][2]["n_questions"] += 1  # a partition the subset does not name
    world["record"].write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    _refused(world, _cut(world, (0, 1)), r"the record's partition 2 is not the rule's", partitions=(0, 1))


def _reduced(w, keep):
    """The fixture built only from the partitions ``keep``: one OUT with just their records, and a questions file with
    just their dev questions (its sha256 put into the OUT's manifest)."""
    cache = w.setdefault("reduced", {})
    if keep in cache:
        return cache[keep]
    where = w["tmp"] / f"reduced-{''.join(map(str, keep))}"
    where.mkdir()
    src = FIXTURE / "questions.jsonl"
    rows = [line for line in src.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["split"] != "dev" or K3[json.loads(line)["id"]] in keep]
    questions = where / "questions.jsonl"
    questions.write_text("\n".join(rows) + "\n", encoding="utf-8")
    old, new = (hashlib.sha256(f.read_bytes()).hexdigest() for f in (src, questions))
    out = where / "out"
    shutil.copytree(w["single"], out, ignore=shutil.ignore_patterns("records"))
    (out / "records").mkdir()
    for f in (w["single"] / "records").glob("*.json"):
        if K3[json.loads(f.read_text(encoding="utf-8"))["question_id"]] in keep:
            shutil.copy2(f, out / "records" / f.name)
    manifest = out / "manifest.json"
    text = manifest.read_text(encoding="utf-8")
    assert old in text
    manifest.write_text(text.replace(old, new), encoding="utf-8")
    _write_identity(out, _identity(out))
    _recheck(out)
    ledger = where / "ledger.jsonl"
    assert exporter.main(["--split", "dev", "--records", str(out), "--out", str(ledger)]) == 0
    cache[keep] = out, questions, new, ledger
    return cache[keep]


def _twin(w, keep, K=None, **kw):
    """The same analysis on the fixture built only from the partitions ``keep`` (no partition record)."""
    out, questions, qsha, ledger = _reduced(w, keep)
    cm, rec = _map()
    call = ex.Call((out,), (questions,), "s01", "dev", (w["voided"],), None, (ledger,), 1, None)
    return ex.run(F, call, K_clusters=K, cluster_map=cm, map_record={**rec, "questions_sha256": qsha}), ledger


@pytest.mark.parametrize("keep", [(0, 1), (1, 2), (0, 2)])
def test_a_subset_equals_the_same_analysis_on_fixtures_built_only_from_those_partitions(world, keep):
    matrix = {(0, 1): HAND_K, (1, 2): HAND_K, (0, 2): np.array([[1.0], [2.0], [0.0]])}  # 3, 3 and 1 clusters
    for K in (matrix[keep], None):
        twin, ledger = _twin(world, keep, K)
        sub = _run(world, _cut(world, keep), partitions=keep, ledgers=(ledger,), K=K)
        same = ex.compare_sections(sub, twin, skip_settings=("inputs", "code_digest"))
        assert all(same[s] for s in ex.SECTIONS) and same["settings_differing"] == [], same
        assert sub["exploratory"]["clusters"] == twin["exploratory"]["clusters"]
        assert sub["deployments"]["shares"] == twin["deployments"]["shares"]
        assert sub["settings"]["n_questions"] == twin["settings"]["n_questions"]
        assert not any(a.get("missing") for a in sub["arms"].values())  # no unnamed partition's attempt is "missing"


def test_a_subsets_hand_derived_numbers_and_what_it_records(world):
    result = _run(world, _cut(world, (0, 1)), partitions=(0, 1), K=HAND_K)
    acc = result["arms"][ARM]["metrics"]["accuracy"]
    assert acc["estimate"] == pytest.approx(1 / 3) and acc["ci"] == pytest.approx([0.0, 2 / 3])
    assert result["exploratory"]["clusters"] == {"questions": 3, "clusters": 3, "sizes": {"1": 3}, "largest": 1}
    block = _block(result)
    assert block["k"] == 3 and sorted(o["partition"] for o in result["exploratory"]["outs"]) == [0, 1]
    sub = block["subset"]
    assert {k: v for k, v in sub.items() if k != "note"} == {
        "partitions": [0, 1], "of_k": 3, "record_questions": 4, "subset_questions": 3, "clusters_cut": 1,
        "outside_questions_not_in_map": 0}
    assert "cluster with questions in a partition outside the subset" in sub["note"]
    nothing = _run(world, _cut(world, (0, 2)), partitions=(0, 2), K=np.array([[1.0], [0.0], [2.0]]))["exploratory"]
    assert nothing["clusters"] == {"questions": 2, "clusters": 1, "sizes": {"2": 1}, "largest": 2}
    assert nothing["partition_union"]["subset"]["clusters_cut"] == 0


def test_a_ledger_row_of_a_partition_the_subset_does_not_name_is_refused(world):
    _refused(world, _cut(world, (0, 1)), r"no record of (its generic cell|the union)", partitions=(0, 1),
             ledgers=(world["ledger"],))
    _, ledger = _twin(world, (0, 1))
    assert _run(world, _cut(world, (0, 1)), partitions=(0, 1), ledgers=(ledger,))["arms"]  # the twin


def test_a_subset_must_not_split_a_base_question():
    union = ex.Union({}, "", 3, {}, {}, (0, 1))
    pass0 = next(i for i in ("a", "b", "c", "d", "e", "f") if ex.partition_of(i, 3) == 0)
    pass2 = next(i for i in ("a", "b", "c", "d", "e", "f") if ex.partition_of(i, 3) == 2)
    rows = {pass0: {"split": "dev", "base_question_id": "b1"}, pass2: {"split": "dev", "base_question_id": "b2"}}
    ex.check_bases_whole(rows, "dev", union)  # the twin: one base each
    rows[pass2]["base_question_id"] = "b1"  # one base question, in partitions 0 and 2
    with pytest.raises(ex.Refused, match=r"1 base questions have questions both inside and outside"):
        ex.check_bases_whole(rows, "dev", union)
    rows[pass2]["split"] = "test"  # another split's question is not the design's
    ex.check_bases_whole(rows, "dev", union)


def test_parity_has_no_twin_for_a_strict_subset_but_runs_for_every_partition(world):
    cm, rec = _map()
    with pytest.raises(F.refusals, match=r"no twin for a strict subset"):
        ex.parity(F, _call(world, _cut(world, (0, 1)), partitions=(0, 1)), cluster_map=cm, map_record=rec)
    same = ex.parity(F, _call(world, world["parts"], partitions=(2, 1, 0)), cluster_map=cm, map_record=rec)
    assert all(same[s] for s in ex.SECTIONS) and same["settings_differing"] == [], same


# --- a partition binding cited with --partitions (science, 25 Oct ~11:5xZ): a PROPER subset only ---------------------------------
#
# The checks of the binding's own rules on the REAL binding and plans are tests/test_partition_binding.py's. These run the whole call
# (prepare, the loading, the analysis) on the three-partition world with a binding made here (plumbing only: its copies, OUT names and
# sha256s are what the OUTs' identities are made to name). What they assert is typed in, or compared with the same analysis without a
# binding (the pre-existing path), or with ccea1637's script (git show): never derived from the helper that made the binding.

OLD_COMMIT = "ccea1637478c6395a598b981566b7b6b144eae3e"  # feat/e189-two-periods before a cited binding took --partitions


def _bound(world, copies):
    """Bound copies of the three partition OUTs (``bound/part-p``: the same names, identities naming the binding, partition p's plan and
    ``copies[p]``) with the binding and its three plans. Returns (OUTs, binding file, plan files, binding sha256, plan sha256s)."""
    tmp = world["tmp"]
    record = json.loads(world["record"].read_text(encoding="utf-8"))
    binding = {"schema": ex.BINDING_SCHEMA, "k": 3, "copies": {c: {} for c in sorted(set(copies))},
               "record": {"sha256": world["record_sha"], "rule": record.get("rule"), "n_questions": record["n_questions"],
                          "partitions": [[q["index"], q["n_questions"], q["ids_sha256"]] for q in record["partitions"]]}}
    bpath = tmp / "binding.json"
    bpath.write_text(json.dumps(binding, indent=1) + "\n", encoding="utf-8")
    bsha = _sha(bpath)
    (tmp / "bound").mkdir()
    outs, plans = [], []
    for p in range(3):
        plan = {"schema": ex.PLAN_SCHEMA, "partition": p, "binding_sha256": bsha, "k": 3, "out": f"part-{p}", "copy": copies[p],
                "n_questions": record["partitions"][p]["n_questions"]}
        path = tmp / f"plan-p{p}.json"
        path.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
        plans.append(path)
        out = tmp / "bound" / f"part-{p}"
        shutil.copytree(world["parts"][p], out)
        _edit_identity(out, lambda i, p=p: (i["partition"].update(copy=copies[p], binding_sha256=bsha, plan_sha256=_sha(plans[p])),
                                           i["store"].update(replica=copies[p])))
        outs.append(out)
    return outs, bpath, plans, bsha, [_sha(x) for x in plans]


def _bind_kw(bpath, plans):
    return {"partition_binding": bpath, "partition_plans": tuple(plans)}


def test_a_binding_with_a_proper_subset_is_the_subsets_analysis_and_records_the_binding(world):
    outs, bpath, plans, bsha, psha = _bound(world, [f"copy-{p}" for p in range(3)])  # the plans' copies are the record's own here
    for K in (HAND_K, None):
        bound = _run(world, outs[:2], partitions=(0, 1), K=K, **_bind_kw(bpath, plans))
        plain = _run(world, _cut(world, (0, 1)), partitions=(0, 1), K=K)  # the pre-existing subset path, no binding
        block = _block(bound)
        assert block.pop("binding") == {"sha256": bsha, "plans": [
            {"partition": p, "out": f"part-{p}", "copy": f"copy-{p}", "plan_sha256": psha[p]} for p in range(3)]}
        assert block["subset"]["partitions"] == [0, 1] and block["subset"] == _block(plain)["subset"]
        assert _dump(bound).replace(str(world["tmp"] / "bound"), str(world["tmp"])) == _dump(plain)  # nothing else differs
    acc = _run(world, outs[:2], partitions=(0, 1), K=HAND_K, **_bind_kw(bpath, plans))["arms"][ARM]["metrics"]["accuracy"]
    assert acc["estimate"] == pytest.approx(1 / 3) and acc["ci"] == pytest.approx([0.0, 2 / 3])  # the hand-derived subset numbers above
    for keep in ((0, 2), (1, 2)):  # any proper subset of at least two, not only the first two
        same = _run(world, [outs[p] for p in keep], partitions=keep, K=None, **_bind_kw(bpath, plans))
        twin = _run(world, _cut(world, keep), partitions=keep, K=None)
        assert _block(same).pop("binding")["sha256"] == bsha
        assert _dump(same).replace(str(world["tmp"] / "bound"), str(world["tmp"])) == _dump(twin)


def test_each_analysed_partition_is_checked_against_its_plan_in_a_binding_subset(world):
    outs, bpath, plans, bsha, psha = _bound(world, [f"bound-{p}" for p in range(3)])  # plan copies that are NOT the record's
    kw = _bind_kw(bpath, plans)
    assert _run(world, outs[:2], partitions=(0, 1), **kw)["arms"]  # accepted: each OUT on its plan's copy
    # the same OUTs without citing the binding are refused (their blocks name one), and the record's copies are not theirs
    _refused(world, outs[:2], r"part-0: its partition block names binding_sha256 and plan_sha256, but no binding is cited",
             partitions=(0, 1))
    _edit_identity(outs[1], lambda i: (i["partition"].update(copy="copy-1"), i["store"].update(replica="copy-1")))
    _refused(world, outs[:2], r"part-1: its store copy \(copy-1, replica copy-1\) is not partition 1's copy in the cited binding "
                              r"\(bound-1\)", partitions=(0, 1), **kw)
    _edit_identity(outs[1], lambda i: (i["partition"].update(copy="bound-1", plan_sha256=psha[0]), i["store"].update(replica="bound-1")))
    _refused(world, outs[:2], r"part-1: its partition block does not name partition 1's plan", partitions=(0, 1), **kw)
    _edit_identity(outs[1], lambda i: i["partition"].update(plan_sha256=psha[1]))
    assert _run(world, outs[:2], partitions=(0, 1), **kw)["arms"]  # the twin: restored


def test_a_binding_subset_refuses_what_a_proper_subset_may_not_be(world):
    outs, bpath, plans, bsha, psha = _bound(world, [f"bound-{p}" for p in range(3)])
    kw = _bind_kw(bpath, plans)
    assert _run(world, outs[:2], partitions=(0, 1), **kw)["arms"]  # baseline
    _refused(world, outs, r"--partitions names every partition of the record: with a cited binding that is the pooled call, which is "
                          r"made without --partitions \(a binding takes --partitions only for a proper subset\)",
             partitions=(2, 0, 1), **kw)
    assert _run(world, outs, **kw)["arms"]  # the pooled call, without the option
    _refused(world, outs[1:2], r"--partitions \[1\]: a partition is analysed only in a union of at least two", partitions=(1,), **kw)
    _refused(world, outs[:2], r"--partitions \[0, 3\]: not all of 0\.\.2, the record's partitions", partitions=(0, 3), **kw)
    _refused(world, outs, r"part-2: partition 2, which --partitions \[0, 1\] does not name", partitions=(0, 1), **kw)  # an unnamed OUT
    _refused(world, outs[:1], r"the union holds partitions \[0\], not each of the named \[0, 1\] once", partitions=(0, 1), **kw)
    _refused(world, outs[:2], r"exactly one plan per partition \(3\), not 2", partitions=(0, 1),
             **_bind_kw(bpath, plans[:2]))  # the binding is cited whole: the plans of the subset alone are refused
    _refused(world, outs[:2], r"given together or not at all", partitions=(0, 1), partition_binding=bpath)


def _old_script(tmp: Path):
    """feat/e189-two-periods' exploratory.py at OLD_COMMIT (git show), beside the other files of its directory, as a second module."""
    blob = f"{OLD_COMMIT}:results/e189-exploratory-20261022/exploratory.py"
    shown = subprocess.run(["git", "-C", str(ROOT), "show", blob], capture_output=True, check=False)
    assert shown.returncode == 0, f"the commit {OLD_COMMIT} is not in this repository: fetch it"
    where = tmp / "old-script" / HERE.parent.name
    where.mkdir(parents=True)
    for f in HERE.parent.iterdir():
        if f.name not in ("exploratory.py", "tests", "dry-run", "__pycache__"):
            (where / f.name).symlink_to(f)
    (where / "exploratory.py").write_bytes(shown.stdout)
    spec = importlib.util.spec_from_file_location("e189_exploratory_old_binding", where / "exploratory.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e189_exploratory_old_binding"] = module
    spec.loader.exec_module(module)
    return module, where


def test_the_pooled_call_with_a_binding_gives_the_commit_befores_results_byte_for_byte(world):
    old, where = _old_script(world["tmp"])
    Fo = old.load_frozen(ROOT)
    outs, bpath, plans, bsha, psha = _bound(world, [f"bound-{p}" for p in range(3)])
    for K in (HAND_K, None):
        cm, rec = _map()
        was = old.run(Fo, old.Call(tuple(outs), (world["questions"],), "s01", "dev", (world["voided"],), world["record"], (), 1, None,
                                   partition_binding=bpath, partition_plans=tuple(plans)),
                      K_clusters=K, cluster_map=cm, map_record=rec)
        now = _run(world, outs, K=K, **_bind_kw(bpath, plans))
        # The old binding commit predates replay diagnostics; this call has no replay inputs.
        assert now.pop("online_vs_replay") == {"rule": ex.ONLINE_VS_REPLAY_RULE, "arms": {}}
        assert now["stores"].pop("replay_class_check") == {
            "rule": ex.REPLAY_CLASS_RULE,
            "labels": {k: v.format(taken="-") for k, v in ex.REPLAY_CLASS_LABEL_TEMPLATES.items()},
            "routing": [], "arms": {},
        }
        assert "binding" in _block(now) and _dump(now) == _dump(was).replace(str(where), str(HERE.parent))
    # ccea1637 refused the call the new script accepts: a binding with --partitions
    cm, rec = _map()
    with pytest.raises(old.Refused, match=r"--partition-binding is for a whole union: not with --partitions"):
        old.run(Fo, old.Call(tuple(outs[:2]), (world["questions"],), "s01", "dev", (world["voided"],), world["record"], (), 1, None,
                             partitions=(0, 1), partition_binding=bpath, partition_plans=tuple(plans)),
                cluster_map=cm, map_record=rec)


def test_the_binding_and_the_subset_reach_the_call_together_from_the_command_line(world, monkeypatch):
    class Stop(Exception):
        def __init__(self, call):
            super().__init__("stop")
            self.call = call

    def fake(_F, call, **_kw):
        raise Stop(call)

    monkeypatch.setattr(ex, "run", fake)
    argv = ["analyse", "--code-root", str(ROOT), "--graph", "s01", "--split", "dev", "--questions", str(world["questions"]),
            "--voided", str(world["voided"]), "--partition-record", str(world["record"]), "--out", str(world["tmp"] / "o"),
            *[a for b in world["parts"] for a in ("--batch", str(b))], "--partition-binding", "/b/binding.json",
            *[a for i in range(3) for a in ("--partition-plan", f"/b/plan-p{i}.json")]]
    with pytest.raises(Stop) as seen:
        ex.main([*argv, "--partitions", "2,0"])
    call = seen.value.call
    assert call.partitions == (0, 2) and call.partition_binding == Path("/b/binding.json")
    assert call.partition_plans == tuple(Path(f"/b/plan-p{i}.json") for i in range(3))


# --- the store timeline (science, 25 Oct 00:10Z): the records' own endpoint and time are the ground truth ---------------
#
# Hand-derived facts, never the script's output:
# - the timeline orders every row with an endpoint (records/ and set-aside) by (started_at, record name), makes a run of
#   each stretch on one endpoint, and is clean only when the runs are exactly the chain (A -> B is [A, B]; A -> B -> A is
#   [A, B, A]) and each run's last end is no later than the next run's first start (equal is fine);
# - the first record of the run on a change's new endpoint gives the change's change_at;
# - in the fixture world moved from copy-a to copy-b (_moved_world), s01-pass and s01-flag are on copy-a and s01-abstain
#   and s01-noquery on copy-b; _store_change times copy-a's records from 09:10:00.000 a second apart and copy-b's from
#   09:18:45.000 (the manifest says 09:18:39.962).

A, B, C = "http://a.invalid:1/", "http://b.invalid:2/", "http://c.invalid:3/"
UTC = timezone.utc


def _row(name: str, endpoint, started, ended, run_id: str | None = None) -> "ex.StoreRow":
    return ex.StoreRow(name, run_id or name.rsplit("/", 1)[-1].split(".")[0], QWEN, "s01", endpoint, started, ended)


def _change(index: int, src: str, dst: str, before=(), at: str = "2026-10-23T09:00:02.500+00:00") -> "ex.StoreChange":
    return ex.StoreChange(index, at, src, dst, frozenset(before))


def _at(sec: float) -> str:
    """09:00:00.000 + sec seconds, in the runner's form."""
    return _iso(datetime(2026, 10, 23, 9, 0, 0, tzinfo=UTC) + timedelta(seconds=sec))


def test_a_clean_move_is_timed_by_its_first_record_on_the_new_endpoint():
    rows = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/a2.json", A, _at(2), _at(3)),
            _row("records/b1.json", B, _at(3), _at(4)),  # starts the instant the last of A ended: allowed
            _row("records/b2.json", B, _at(5), _at(6))]
    line = ex.store_timeline(rows, [_change(1, A, B, {"a1", "a2"})], "records")
    assert line.chain == (A, B)
    assert (line.clean, line.interleaved, line.overlap, line.chain_incomplete) == (True, False, False, False)
    assert (line.endpoint_outside_chain, line.rows_without_endpoint, line.unusable_times) == (False, False, False)
    assert line.unusable_rows == () and line.period_of == {"a1": 0, "a2": 0, "b1": 1, "b2": 1}
    (change,) = line.changes
    assert change.first_to_started == datetime(2026, 10, 23, 9, 0, 3, tzinfo=UTC)
    assert change.first_to_started_text == "2026-10-23T09:00:03.000+00:00" and change.manifest_at == _at(2.5)
    assert (change.first_precedes_manifest_at, change.manifest_lists_to_endpoint_record_as_before,
            change.manifest_omits_from_endpoint_record, change.setaside_changes_first_instant) == (False,) * 4
    assert ex.timeline_problems("OUT", line) == []


@pytest.mark.parametrize(("sequence", "chain", "clean", "interleaved", "incomplete", "outside"), [
    ("AB", "AB", True, False, False, False),
    ("AABB", "AB", True, False, False, False),
    ("ABA", "ABA", True, False, False, False),
    ("ABA", "AB", False, True, False, False),
    ("BAB", "AB", False, True, False, False),
    ("ABAB", "ABA", False, True, False, False),
    ("AB", "ABA", False, False, True, False),
    ("A", "AB", False, False, True, False),
    ("B", "AB", False, False, True, False),
    ("", "AB", False, False, True, False),
    ("ACB", "AB", False, False, False, True),
    ("AC", "AB", False, False, False, True),
])
def test_the_runs_of_the_records_must_be_the_chain_in_order(sequence, chain, clean, interleaved, incomplete, outside):
    ends = {"A": A, "B": B, "C": C}
    rows = [_row(f"records/r{i}.json", ends[e], _at(10 * i), _at(10 * i + 5)) for i, e in enumerate(sequence)]
    changes = [_change(i, ends[a], ends[b]) for i, (a, b) in enumerate(itertools.pairwise(chain), 1)]
    line = ex.store_timeline(rows, changes, "records")
    assert (line.clean, line.interleaved, line.chain_incomplete, line.endpoint_outside_chain) == (
        clean, interleaved, incomplete, outside)
    assert line.overlap is False
    assert [bool(ex.timeline_problems("OUT", line))] == [not clean]


def test_a_return_to_the_first_copy_is_timed_by_the_first_record_of_its_third_run():
    rows = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/b1.json", B, _at(10), _at(11)),
            _row("records/a2.json", A, _at(20), _at(21)), _row("records/a3.json", A, _at(25), _at(26))]
    line = ex.store_timeline(rows, [_change(1, A, B, {"a1"}), _change(2, B, A, {"a1", "b1"})], "records")
    assert line.chain == (A, B, A) and line.clean and line.period_of == {"a1": 0, "b1": 1, "a2": 2, "a3": 2}
    assert [c.first_to_started_text for c in line.changes] == [_at(10), _at(20)]


def test_an_old_copy_attempt_still_running_when_the_new_copy_took_one_is_an_overlap_not_a_clean_move():
    rows = [_row("records/a1.json", A, _at(0), _at(5.001)), _row("records/b1.json", B, _at(5), _at(6))]
    line = ex.store_timeline(rows, [_change(1, A, B)], "records")
    assert (line.clean, line.overlap, line.interleaved, line.chain_incomplete) == (False, True, False, False)
    (why,) = ex.timeline_problems("OUT", line)
    assert "(overlap)" in why and "a mix of copies, not a late timestamp" in why
    # a1 ends in the same millisecond b1 starts: equal is not an overlap
    ok = ex.store_timeline([_row("records/a1.json", A, _at(0), _at(5)), rows[1]], [_change(1, A, B)], "records")
    assert ok.clean and not ok.overlap
    # a longer-running earlier attempt of the old copy counts too, not only the last one to start
    long = [_row("records/a0.json", A, _at(0), _at(100)), _row("records/a1.json", A, _at(1), _at(2)), rows[1]]
    assert ex.store_timeline(long, [_change(1, A, B)], "records").overlap is True


def test_set_aside_rows_are_in_the_population_and_can_move_the_first_instant_or_interleave():
    base = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/b1.json", B, _at(10), _at(11))]
    early = ex.store_timeline([*base, _row("infrastructure/b0.json", B, _at(8), _at(9))], [_change(1, A, B)], "records")
    (change,) = early.changes
    assert early.clean and change.first_to_started_text == _at(8) and change.setaside_changes_first_instant is True
    assert early.period_of == {"a1": 0, "b1": 1}  # the set-aside row is not in records/
    late_b = ex.store_timeline([*base, _row("infrastructure/b2.json", B, _at(12), _at(13))], [_change(1, A, B)], "records")
    assert late_b.clean and late_b.changes[0].setaside_changes_first_instant is False
    mix = ex.store_timeline([*base, _row("infrastructure/a2.json", A, _at(12), _at(13))], [_change(1, A, B)], "records")
    assert mix.interleaved is True and not mix.clean
    only = ex.store_timeline([base[0], _row("harness/b0.json", B, _at(8), _at(9))], [_change(1, A, B)], "records")
    assert only.clean and only.changes[0].setaside_changes_first_instant is True  # no records/ row on B at all
    assert only.changes[0].first_precedes_manifest_at is None  # nothing in records/ to compare with the manifest
    gz = ex.store_timeline([base[0], _row("records/b1.json.gz", B, _at(10), _at(11))], [_change(1, A, B)], "records")
    assert gz.clean and gz.period_of == {"a1": 0, "b1": 1}


def test_the_manifests_placement_of_records_is_compared_as_booleans_only():
    rows = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/a2.json", A, _at(2), _at(3)),
            _row("records/b1.json", B, _at(10), _at(11)), _row("records/b2.json", B, _at(12), _at(13))]
    late = _change(1, A, B, {"a1", "b1"}, at=_at(20))  # written after b1 started; omits a2; lists b1 as made before
    (change,) = ex.store_timeline(rows, [late], "records").changes
    assert (change.first_precedes_manifest_at, change.manifest_lists_to_endpoint_record_as_before,
            change.manifest_omits_from_endpoint_record) == (True, True, True)
    same = _change(1, A, B, {"a1", "a2"}, at=_at(9))
    (change,) = ex.store_timeline(rows, [same], "records").changes
    assert (change.first_precedes_manifest_at, change.manifest_lists_to_endpoint_record_as_before,
            change.manifest_omits_from_endpoint_record) == (False, False, False)
    for at in ("not a time", None, "2026-10-23T09:00:09"):  # unknown, never False
        (change,) = ex.store_timeline(rows, [_change(1, A, B, {"a1", "a2"}, at=at)], "records").changes
        assert change.first_precedes_manifest_at is None and change.manifest_at == at


def test_a_row_without_a_usable_time_or_endpoint_makes_the_timeline_unusable_and_names_the_record():
    good = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/b1.json", B, _at(10), _at(11))]
    bad = {"no zone": _row("records/z1.json", A, "2026-10-23T09:00:05.000", "2026-10-23T09:00:06.000+00:00"),
           "ended before started": _row("records/z1.json", A, _at(5), _at(4)),
           "no ended_at": _row("records/z1.json", A, _at(5), None),
           "not a string": _row("records/z1.json", A, 5, _at(6))}
    for case, row in bad.items():
        line = ex.store_timeline([*good, row], [_change(1, A, B)], "records")
        assert (line.clean, line.unusable_times, line.unusable_rows) == (False, True, ("records/z1.json",)), case
        assert (line.interleaved, line.overlap, line.chain_incomplete, line.period_of) == (None, None, None, {}), case
        assert line.changes[0].first_to_started is None
        (why,) = ex.timeline_problems("OUT", line)
        assert "unusable_times" in why and "['records/z1.json']" in why and "mix of copies" not in why
    for endpoint in (None, "", 7):
        line = ex.store_timeline([*good, _row("records/n1.json", endpoint, _at(5), _at(6))], [_change(1, A, B)], "records")
        assert (line.clean, line.rows_without_endpoint, line.unusable_times) == (False, True, False)
        assert line.unusable_rows == ("records/n1.json",)
        assert "['records/n1.json']" in ex.timeline_problems("OUT", line)[0]
    many = [_row(f"records/z{i}.json", A, None, None) for i in range(7, 0, -1)]
    assert ex.store_timeline([*good, *many], [_change(1, A, B)], "records").unusable_rows == tuple(
        f"records/z{i}.json" for i in range(1, 6))  # at most five names, sorted: never a count


def test_one_run_id_in_records_of_two_runs_is_flagged_and_an_empty_change_list_is_refused():
    rows = [_row("records/a1.json", A, _at(0), _at(1), "same"), _row("records/b1.json", B, _at(10), _at(11), "same")]
    line = ex.store_timeline(rows, [_change(1, A, B)], "records")  # a flag, not a raise: the command must still print it
    assert (line.run_id_in_two_runs, line.clean) == (True, False)
    (why,) = ex.timeline_problems("OUT", line)
    assert "(run_id_in_two_runs)" in why and "same" not in why  # no run id
    twin = [_row("records/a1.json", A, _at(0), _at(1), "same"), _row("records/a1.json.gz", A, _at(0), _at(1), "same"),
            ex.dataclasses.replace(rows[1], run_id="other")]  # one record twice in ONE run is not it
    again = ex.store_timeline(twin, [_change(1, A, B)], "records")
    assert (again.run_id_in_two_runs, again.clean) == (False, True)
    unknown = ex.store_timeline([*rows, _row("records/z.json", A, None, None)], [_change(1, A, B)], "records")
    assert unknown.run_id_in_two_runs is None  # not evaluated while a record has no usable time
    with pytest.raises(ex.Refused, match="at least one manifest store change"):
        ex.store_timeline(rows, [], "records")


# --- the branch phase in the timeline (science, 25 Oct ~04:55Z) --------------------------------------------------------
#
# Hand-derived facts: a branch record carries the endpoint of its PARENT attempt (natural parents on the old copy, prompted
# parents on the new one), so once the OUT's runner entered the branch phase (the manifest's additions entry that added
# "phase branch", at T_b) the endpoints alternate within milliseconds, which no live switch can do. A row that started at
# or after T_b (equal counts) is a branch row: outside the runs, the order check, first_to_started and the cross-check
# booleans, with no period; it still needs an endpoint of the chain and usable times, and it never completes a chain.


def _branch(sec: float) -> "ex.BranchPhase":
    """A branch phase entered at 09:00:00 + sec seconds, its manifest text being the runner's form of that instant."""
    return ex.BranchPhase(datetime(2026, 10, 23, 9, 0, 0, tzinfo=UTC) + timedelta(seconds=sec), _at(sec))


def test_branch_rows_alternating_between_the_copies_after_the_branch_instant_are_clean_where_they_were_a_mix():
    main = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/a2.json", A, _at(2), _at(3)),
            _row("records/b1.json", B, _at(10), _at(11)),
            _row("records/b2.json", B, _at(12), _at(20))]  # still running when the branch phase begins (at 15)
    late = [_row("records/x1.json", B, _at(15.5), _at(30)), _row("records/x2.json", A, _at(16), _at(17)),
            _row("records/x3.json", B, _at(16.5), _at(18)), _row("records/x4.json", A, _at(17), _at(40))]
    change = [_change(1, A, B, {"a1", "a2"})]
    today = ex.store_timeline([*main, *late], change, "records")  # without the branch instant: B A B A after b2
    # (without the instant: runs A, B, A, B, A; the second run ends at 30, after the third starts at 16, so an overlap too)
    assert (today.clean, today.interleaved, today.overlap) == (False, True, True)
    line = ex.store_timeline([*main, *late], change, "records", branch_start=_branch(15))
    assert (line.clean, line.interleaved, line.overlap, line.chain_incomplete, line.endpoint_outside_chain) == (
        True, False, False, False, False)
    assert ex.timeline_problems("OUT", line) == []
    assert [r.endpoint for r in line.runs] == [A, B]  # two runs: branch rows are not in the listing
    assert (line.runs[0].first_started, line.runs[0].last_started, line.runs[0].last_ended) == (_at(0), _at(2), _at(3))
    assert (line.runs[1].first_started, line.runs[1].last_started, line.runs[1].last_ended) == (_at(10), _at(12), _at(20))
    assert line.period_of == {"a1": 0, "a2": 0, "b1": 1, "b2": 1, "x1": None, "x2": None, "x3": None, "x4": None}
    assert line.branch_phase_start == _at(15) and line.branch_row_endpoints == (A, B)
    (ch,) = line.changes
    assert ch.first_to_started_text == _at(10)
    assert (ch.first_precedes_manifest_at, ch.manifest_lists_to_endpoint_record_as_before,
            ch.manifest_omits_from_endpoint_record, ch.setaside_changes_first_instant) == (False,) * 4


def test_a_real_interleave_before_the_branch_instant_is_still_a_mix():
    rows = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/b1.json", B, _at(2), _at(3)),
            _row("records/a2.json", A, _at(4), _at(5)),  # a mix before the branch phase (at 15)
            _row("records/x1.json", B, _at(16), _at(17))]
    line = ex.store_timeline(rows, [_change(1, A, B)], "records", branch_start=_branch(15))
    assert (line.clean, line.interleaved) == (False, True)
    (why,) = ex.timeline_problems("OUT", line)
    assert "(interleaved)" in why and "a mix of copies, not a late timestamp" in why


def test_the_branch_instant_exempts_by_time_a_row_exactly_at_it_is_a_branch_row_one_just_before_is_not():
    base = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/b1.json", B, _at(10), _at(11))]
    at_instant = ex.store_timeline([*base, _row("records/x1.json", A, _at(15), _at(16))], [_change(1, A, B)], "records",
                                   branch_start=_branch(15))
    assert at_instant.clean and at_instant.period_of == {"a1": 0, "b1": 1, "x1": None}  # equal counts as branch
    just_before = ex.store_timeline([*base, _row("records/a2.json", A, _at(14.999), _at(16))], [_change(1, A, B)],
                                    "records", branch_start=_branch(15))
    assert (just_before.clean, just_before.interleaved) == (False, True)  # an old-copy row after the first new-copy row
    assert just_before.period_of == {"a1": 0, "b1": 1, "a2": 2}  # the runs A, B, A of the rows before the instant
    assert just_before.branch_row_endpoints == ()  # a branch phase with no branch row: an empty tuple, not None


def test_a_branch_row_must_still_name_an_endpoint_of_the_chain_and_have_usable_times():
    base = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/b1.json", B, _at(10), _at(11))]
    outside = ex.store_timeline([*base, _row("records/x1.json", C, _at(16), _at(17))], [_change(1, A, B)], "records",
                                branch_start=_branch(15))
    assert (outside.clean, outside.endpoint_outside_chain, outside.interleaved) == (False, True, False)
    assert outside.branch_row_endpoints == (C,)
    assert "(endpoint_outside_chain)" in ex.timeline_problems("OUT", outside)[0]
    for bad in (_row("records/x1.json", A, _at(16), None), _row("records/x1.json", A, _at(16), _at(15)),
                _row("records/x1.json", A, "2026-10-23T09:00:16.000", _at(17))):
        line = ex.store_timeline([*base, bad], [_change(1, A, B)], "records", branch_start=_branch(15))
        assert (line.clean, line.unusable_times, line.unusable_rows) == (False, True, ("records/x1.json",))
        assert (line.runs, line.branch_row_endpoints) == (None, None)  # the rows cannot all be classed
        assert line.branch_phase_start == _at(15)
    no_endpoint = ex.store_timeline([*base, _row("records/x1.json", None, _at(16), _at(17))], [_change(1, A, B)],
                                    "records", branch_start=_branch(15))
    assert (no_endpoint.clean, no_endpoint.rows_without_endpoint) == (False, True)


def test_a_branch_row_never_completes_a_chain():
    rows = [_row("records/a1.json", A, _at(0), _at(1)),  # the new copy has branch rows only
            _row("records/x1.json", B, _at(16), _at(17)), _row("records/x2.json", A, _at(17), _at(18))]
    line = ex.store_timeline(rows, [_change(1, A, B)], "records", branch_start=_branch(15))
    assert (line.clean, line.chain_incomplete, line.interleaved) == (False, True, False)
    assert line.changes[0].first_to_started is None and line.changes[0].first_to_started_text is None
    assert line.period_of == {"a1": 0, "x1": None, "x2": None} and line.branch_row_endpoints == (A, B)
    assert "(chain_incomplete)" in ex.timeline_problems("OUT", line)[0]
    nothing = ex.store_timeline(rows[1:], [_change(1, A, B)], "records", branch_start=_branch(15))
    assert (nothing.clean, nothing.chain_incomplete, nothing.runs) == (False, True, ())  # no row outside the branch phase


def test_a_return_chain_with_branch_rows_on_both_copies_is_clean_and_the_branch_rows_have_no_period():
    rows = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/b1.json", B, _at(10), _at(11)),
            _row("records/a2.json", A, _at(20), _at(21)),
            _row("records/x1.json", B, _at(30), _at(31)), _row("records/x2.json", A, _at(30.5), _at(33))]
    changes = [_change(1, A, B, {"a1"}), _change(2, B, A, {"a1", "b1"})]
    line = ex.store_timeline(rows, changes, "records", branch_start=_branch(30))
    assert line.chain == (A, B, A) and line.clean and line.branch_row_endpoints == (A, B)
    assert line.period_of == {"a1": 0, "b1": 1, "a2": 2, "x1": None, "x2": None}
    assert [c.first_to_started_text for c in line.changes] == [_at(10), _at(20)]
    assert [r.endpoint for r in line.runs] == [A, B, A]


def test_one_run_id_in_a_run_and_in_the_branch_phase_is_flagged_and_set_aside_branch_rows_are_exempt_too():
    base = [_row("records/a1.json", A, _at(0), _at(1), "same"), _row("records/b1.json", B, _at(10), _at(11))]
    twin = ex.store_timeline([*base, _row("records/x1.json", B, _at(16), _at(17), "same")], [_change(1, A, B)], "records",
                             branch_start=_branch(15))
    assert (twin.run_id_in_two_runs, twin.clean) == (True, False)
    assert "(run_id_in_two_runs)" in ex.timeline_problems("OUT", twin)[0] and "same" not in ex.timeline_problems("OUT", twin)[0]
    setaside_late = ex.store_timeline([*base[1:], _row("records/a1.json", A, _at(0), _at(1)),
                                       _row("infrastructure/x9.json", A, _at(20), _at(21))], [_change(1, A, B)], "records",
                                      branch_start=_branch(15))
    assert setaside_late.clean and setaside_late.branch_row_endpoints == (A,)  # (the exemption is by time, in any directory)
    assert "x9" not in setaside_late.period_of
    setaside_early = ex.store_timeline([*base, _row("infrastructure/x9.json", A, _at(12), _at(13))], [_change(1, A, B)],
                                       "records", branch_start=_branch(15))
    assert (setaside_early.clean, setaside_early.interleaved) == (False, True)


def test_without_a_branch_phase_the_timeline_is_what_it_was_and_a_late_branch_instant_changes_nothing_else():
    rows = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/b1.json", B, _at(10), _at(11)),
            _row("records/a2.json", A, _at(12), _at(13))]
    plain = ex.store_timeline(rows, [_change(1, A, B)], "records")
    assert (plain.clean, plain.interleaved) == (False, True)  # an alternation is a mix, as before
    assert plain.branch_phase_start is None and plain.branch_row_endpoints is None
    assert plain == ex.store_timeline(rows, [_change(1, A, B)], "records", branch_start=None)
    later = ex.store_timeline(rows, [_change(1, A, B)], "records", branch_start=_branch(99))  # no row is that late
    assert (later.branch_phase_start, later.branch_row_endpoints) == (_at(99), ())
    assert ex.dataclasses.replace(later, branch_phase_start=None, branch_row_endpoints=None) == plain


OUT_PATH = Path("/host/OUT-x")
RUN_LIKE = "run-0123abcd-4567-89ef"  # a manifest string a refusal must never repeat
# every wrong type of a PRESENT additions (only a missing key is no branch phase), each holding RUN_LIKE where it can
WRONG_ADDITIONS = {
    "null": None, "a dict": {"added": ["phase branch"], "at": RUN_LIKE}, "a string": RUN_LIKE, "a number": 7,
    "false": False, "a list holding a string": [RUN_LIKE], "a list holding null": [None],
    "a list holding a number": [7], "a list holding a list": [["phase branch"]],
    "added a string": [{"added": RUN_LIKE, "at": RUN_LIKE}], "added null": [{"added": None, "at": RUN_LIKE}],
    "added a number": [{"added": 7, "at": RUN_LIKE}], "added a dict": [{"added": {"k": RUN_LIKE}, "at": RUN_LIKE}],
    "added holding a number": [{"added": ["phase branch", 7], "at": RUN_LIKE}],
    "added holding null": [{"added": [None], "at": RUN_LIKE}],
    "added holding a list": [{"added": [["phase branch"]], "at": RUN_LIKE}],
}
FIXTURE_PHASE_BRANCH_AT = "2026-10-20T02:24:12.743+00:00"  # by hand: the "phase branch" entry of the fixture's manifest


def _additions(*added) -> "ex.Out":
    return ex.Out(OUT_PATH, None, {"additions": list(added)})


def test_the_branch_instant_is_the_at_of_the_additions_entry_that_added_phase_branch():
    for manifest in ({}, {"additions": []},  # only a MISSING key, or a list without the entry, is no branch phase
                     {"additions": [{"added": ["phase prompted"], "at": _at(1)}, {"at": _at(2), "routes": []},
                                    {"added": ["phase branching", "phase branch 2"], "at": _at(3)}]}):
        assert ex.manifest_branch_phase(ex.Out(OUT_PATH, None, manifest)) == (None, [])  # names must be equal, not alike
    spelled = "2026-10-23T09:00:15Z"  # kept as the manifest wrote it; parsed to the instant
    found, problems = ex.manifest_branch_phase(_additions({"added": ["phase prompted"], "at": _at(1)},
                                                          {"added": ["x", "phase branch"], "at": spelled, "routes": []}))
    assert problems == [] and found == ex.BranchPhase(datetime(2026, 10, 23, 9, 0, 15, tzinfo=UTC), spelled)


@pytest.mark.parametrize(("why", "manifest"), [
    ("two entries", {"additions": [{"added": ["phase branch"], "at": RUN_LIKE}, {"added": ["phase branch"], "at": _at(5)}]}),
    ("two identical entries", {"additions": [{"added": ["phase branch"], "at": _at(5)}] * 2}),
    ("at unparseable", {"additions": [{"added": ["phase branch"], "at": RUN_LIKE}]}),
    ("at without zone", {"additions": [{"added": ["phase branch"], "at": "2026-10-23T09:00:15"}]}),
    ("at missing", {"additions": [{"added": ["phase branch"]}]}),
    ("at not a string", {"additions": [{"added": ["phase branch"], "at": 20261023}]}),
    ("additions a dict", {"additions": {"added": ["phase branch"], "at": RUN_LIKE}}),
    ("additions a string", {"additions": RUN_LIKE}),
    ("additions zero", {"additions": 0}),
    ("additions false", {"additions": False}),
    ("entry a string", {"additions": [RUN_LIKE]}),
    ("entry a list", {"additions": [["phase branch"]]}),
    ("added a string", {"additions": [{"added": "phase branch", "at": RUN_LIKE}]}),
    ("added null", {"additions": [{"added": None, "at": RUN_LIKE}]}),
    ("added a dict", {"additions": [{"added": {"phase branch": RUN_LIKE}, "at": RUN_LIKE}]}),
])
def test_a_malformed_additions_list_is_a_problem_naming_the_out_and_never_a_manifest_string(why, manifest):
    found, problems = ex.manifest_branch_phase(ex.Out(OUT_PATH, None, manifest))
    assert found is None and problems, why
    assert all(str(OUT_PATH) in p and "its manifest's additions" in p and RUN_LIKE not in p for p in problems), problems
    assert len(problems) == len(set(problems))  # a repeated shape is said once


@pytest.mark.parametrize("why", list(WRONG_ADDITIONS))
def test_every_wrong_type_of_a_present_additions_is_a_problem_and_only_a_missing_key_is_none(why):
    found, problems = ex.manifest_branch_phase(ex.Out(OUT_PATH, None, {"additions": WRONG_ADDITIONS[why]}))
    assert found is None and len(problems) == 1, why  # a problem, not a TypeError and not "no branch phase"
    assert str(OUT_PATH) in problems[0] and "its manifest's additions" in problems[0] and RUN_LIKE not in problems[0]
    assert ex.manifest_branch_phase(ex.Out(OUT_PATH, None, {})) == (None, [])  # the one way to have none, besides a list


# --- the timeline in the fixture world: check_stores, the report and the refusals --------------------------------------


def _ids(out: Path, *questions: str) -> list[str]:
    """The run ids of the OUT's records/ records of these questions (every model and cell), sorted."""
    found = []
    for f in sorted((out / "records").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["question_id"] in questions:
            found.append(r["run_id"])
    return sorted(found)


def _record_of(out: Path, question: str, model: str = QWEN, arm: str = "none") -> tuple[Path, dict]:
    (hit,) = [(f, r) for f in sorted((out / "records").glob("*.json"))
              if (r := json.loads(f.read_text(encoding="utf-8")))["question_id"] == question
              and r["model"] == model and r["load"]["arm"] == arm]
    return hit


def _edit_manifest(out: Path, edit) -> None:
    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    edit(manifest)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    _recheck(out)


def _check(world, out: Path, files, entries, classes: Path | None = None):
    """check_stores on the one OUT directly (no frozen loading): its StoreCopies, or Refused. ``classes`` is the copy
    classes file (by default the fixtures' own)."""
    config = _store_config(world["tmp"] / "direct.json", *entries)
    classes = classes or _classes_file(world["tmp"] / "classes.json")
    return ex.check_stores(ex.load_attribution(), [ex.Out.read(out)], "s01", tuple(files), config, None, classes)


def _refused_check(world, out: Path, files, entries, match: str) -> None:
    with pytest.raises(ex.Refused, match=match):
        _check(world, out, files, entries)


def test_the_first_record_on_the_new_copy_is_the_change_at_and_the_shares_split_by_period(world, ref):
    m = _moved_world(world)
    _stamp_by_endpoint(m["out"], E0, E1, dst_from=datetime(2026, 10, 23, 9, 19, 7, 250000, tzinfo=UTC))
    starts = [json.loads(f.read_text(encoding="utf-8"))["started_at"] for f in _record_files(m["out"])
              if json.loads(f.read_text(encoding="utf-8"))["store"]["endpoint"] == E1]
    assert min(starts) == "2026-10-23T09:19:07.250+00:00"  # the fixture, by hand: the manifest says 09:18:39.962
    entries = [m["entries"][0], {**m["entries"][1], "change_at": "2026-10-23T09:19:07.250Z"}]  # another spelling
    st = _stores(world, [m["out"]], m["proofs"], entries)
    (change,) = st["store_changes"]
    assert change["manifest_at"] == change["at"] == "2026-10-23T09:18:39.962+00:00"
    assert change["first_record_on_new_endpoint"] == change["change_at_used"] == "2026-10-23T09:19:07.250+00:00"
    assert (change["first_precedes_manifest_at"], change["manifest_lists_to_endpoint_record_as_before"],
            change["manifest_omits_from_endpoint_record"], change["setaside_changes_first_instant"]) == (False,) * 4
    for cell in CELLS:
        split = change["shares_by_arm"][f"s01|{QWEN}|{cell}|prompt|-"]
        assert split == {"before": _rows({"copy-a": 2}), "after": _rows({"copy-b": 2})}
    assert st["reader"] == {"fields": ["run_id", "model", "store.name", "store.endpoint", "started_at", "ended_at"]}
    assert st["rule"].endswith("science, 25 Oct 00:10Z: the records' own recorded endpoint is the ground truth; a store "
                               "change's change_at is the start of the first record on the new endpoint; records of the "
                               "two copies interleaving is a mix and refuses. " + BRANCH_RULE_TEXT)
    assert change["branch_phase_start"] is None and set(split) == {"before", "after"}  # no branch phase: as before
    assert "pre_change_records" not in change and change["manifest_pre_change_records"] == 20


def test_a_late_manifest_timestamp_is_accepted_with_the_manifest_placement_disclosed_and_the_split_by_period(world, ref):
    m = _moved_world(world)
    out = m["out"]
    omitted = _record_of(out, "s01-pass")[1]["run_id"]  # a record on copy-a the manifest leaves out of its list
    inside = _record_of(out, "s01-noquery")[1]["run_id"]  # a record on copy-b it lists as made before the change
    listed = sorted([*(set(m["before"]) - {omitted}), inside])

    def late(manifest):
        change = manifest["store_changes"][0]
        change.update(at="2026-10-23T09:25:00.000+00:00", pre_change_run_ids=listed, pre_change_records=len(listed))

    _edit_manifest(out, late)  # written at 09:25, after the first record on copy-b (09:18:45)
    st = _stores(world, [out], m["proofs"], m["entries"])
    (change,) = st["store_changes"]
    assert (change["first_precedes_manifest_at"], change["manifest_lists_to_endpoint_record_as_before"],
            change["manifest_omits_from_endpoint_record"], change["setaside_changes_first_instant"]) == (
        True, True, True, False)
    assert change["manifest_at"] == "2026-10-23T09:25:00.000+00:00" and change["manifest_pre_change_records"] == 20
    # by period, not by the manifest's list: Qwen's natural cell has s01-pass (omitted) and s01-flag on copy-a before
    # and s01-abstain and s01-noquery (listed) on copy-b after; the manifest's list would split it 1+1 and 1+1
    assert change["shares_by_arm"][ARM] == {"before": _rows({"copy-a": 2}), "after": _rows({"copy-b": 2})}
    late_check = _storecheck(world["tmp"] / "late-sc.json", E1, "copy-b", checked_utc="2026-10-23T09:18:50Z")
    entry = {**m["entries"][1], "storecheck_sha256": _sha(late_check)}  # made five seconds after the first record on b
    _refused_stores(world, [out], [*m["a"], late_check, m["b"][1]], [m["entries"][0], entry],
                    r"\(copy-b\): its storecheck was made after the copy took the OUT's attempts")


def test_an_old_copy_attempt_that_ran_past_the_first_on_the_new_copy_is_refused_and_ending_with_it_is_not(world, ref):
    m = _moved_world(world)
    out, first_a = m["out"], m["before"][0]
    # started on copy-a at 09:18:40, five seconds before the first record on copy-b, and ended at 09:18:50
    _retime(out, lambda r, f: ("2026-10-23T09:18:40.000+00:00", "2026-10-23T09:18:50.000+00:00")
            if r["run_id"] == first_a else None)
    _refused_stores(world, [out], m["proofs"], m["entries"],
                    r"single: its records' own endpoints and times do not follow its manifest store changes "
                    r"\(overlap\): .*a mix of copies, not a late timestamp")
    _retime(out, lambda r, f: ("2026-10-23T09:18:40.000+00:00", "2026-10-23T09:18:45.000+00:00")
            if r["run_id"] == first_a else None)  # ends the instant the first record on copy-b starts
    timeline = _check(world, out, m["proofs"], m["entries"]).timelines[str(out)]
    assert timeline.clean and timeline.changes[0].first_to_started_text == "2026-10-23T09:18:45.000+00:00"


A_TWO = "copy-a (two periods)"  # the report's one label for copy-a used in two periods
RETURN_AT = "2026-10-23T09:30:00Z"  # by hand: the start of the first record of copy-a's third run (_aba_world below)
A2_CHECKED, A2_STARTED = "2026-10-23T09:25:00Z", "2026-10-23T09:25:05Z"  # the second period's own checks, before RETURN_AT
B_RETURN_AT = "2026-10-23T09:40:00Z"  # by hand: the start of the first record of copy-b's second period (abab only)
B2_CHECKED, B2_STARTED = "2026-10-23T09:35:00Z", "2026-10-23T09:35:05Z"


def _aba_world(world, *, fourth: str = "a", abab: bool = False) -> dict:
    """The fixture whole on copy-a (E0), then copy-b (E1), then copy-a again, by two recorded store changes: s01-pass
    on a from 09:10:00, s01-flag on b from 09:18:45, s01-abstain on a from 09:30:00 and s01-noquery from 09:31:00 on a
    (or, with ``fourth`` "b", from 09:40:00 on b: a b a b, which ``abab`` makes the chain too, by a third change, and
    gives copy-b its second_period entry too, with its own checks made at 09:35). Copy-a's two periods have each their
    own pair of checks (the second's made at 09:25), and its ordinary entry, as the chain's start, carries no
    change_at."""
    out, tmp = world["single"], world["tmp"]
    fourth = "b" if abab else fourth
    where = {"s01-pass": E0, "s01-flag": E1, "s01-abstain": E0, "s01-noquery": E1 if fourth == "b" else E0}
    _set_endpoints(out, lambda r: where[r["question_id"]])
    first = {"s01-pass": "09:10:00", "s01-flag": "09:18:45", "s01-abstain": "09:30:00",
             "s01-noquery": "09:40:00" if fourth == "b" else "09:31:00"}
    seen: Counter = Counter()

    def start_of(r, f):
        k = seen[r["question_id"]]
        seen[r["question_id"]] += 1
        return datetime.fromisoformat(f"2026-10-23T{first[r['question_id']]}+00:00") + timedelta(seconds=k)

    _store_change(out, E0, E1, _ids(out, "s01-pass"), stamp=False)
    _store_change(out, E1, E0, _ids(out, "s01-pass", "s01-flag"), stamp=False)
    if abab:
        _store_change(out, E0, E1, _ids(out, "s01-pass", "s01-flag", "s01-abstain"), stamp=False)
    _retime(out, start_of)
    a, b = _copy_proofs(tmp, "a", E0, "copy-a"), _copy_proofs(tmp, "b", E1, "copy-b")
    a2 = (_storecheck(tmp / "sc-a2.json", E0, "copy-a", checked_utc=A2_CHECKED),
          _load10(tmp / "ka-a2.json", E0, started_utc=A2_STARTED))
    world_ = {"out": out, "a": a, "b": b, "a2": a2, "proofs": (*a, *b, *a2),
              "entries": [_entry("single", E0, "copy-a", a),  # its first period: the chain's start, no change_at
                          _entry("single", E1, "copy-b", b, change_at="2026-10-23T09:18:45Z"),
                          _entry("single", E0, "copy-a", a2, change_at=RETURN_AT, second_period=True)]}  # its third run
    if abab:  # copy-b's return, by the third change
        b2 = (_storecheck(tmp / "sc-b2-abab.json", E1, "copy-b", checked_utc=B2_CHECKED),
              _load10(tmp / "ka-b2-abab.json", E1, started_utc=B2_STARTED))
        world_ |= {"b2": b2, "proofs": (*world_["proofs"], *b2),
                   "entries": [*world_["entries"], _entry("single", E1, "copy-b", b2, change_at=B_RETURN_AT,
                                                         second_period=True)]}
    return world_


def test_a_return_to_the_first_copy_is_accepted_as_two_periods_of_one_copy_timed_and_split_by_period(world, ref):
    w = _aba_world(world)
    st = _stores(world, [w["out"]], w["proofs"], w["entries"])
    one, two = st["store_changes"]
    assert (one["index"], one["from"], one["to"], one["first_record_on_new_endpoint"]) == (
        1, A_TWO, "copy-b", "2026-10-23T09:18:45.000+00:00")
    assert (two["index"], two["from"], two["to"], two["first_record_on_new_endpoint"]) == (
        2, "copy-b", A_TWO, "2026-10-23T09:30:00.000+00:00")
    # Qwen's natural cell has one record per question: pass -> a, flag -> b, abstain and noquery -> a; both of a's
    # periods are one row, one copy
    assert st["shares"][ARM] == _rows({A_TWO: 3, "copy-b": 1})
    assert one["shares_by_arm"][ARM] == {"before": _rows({A_TWO: 1}), "after": _rows({"copy-b": 1, A_TWO: 2})}
    assert two["shares_by_arm"][ARM] == {"before": _rows({A_TWO: 1, "copy-b": 1}), "after": _rows({A_TWO: 2})}
    assert sorted(st["copies"]) == [A_TWO, "copy-b"] and "periods" not in st["copies"]["copy-b"]
    copy = st["copies"][A_TWO]
    assert copy["periods"] == [
        {"period": 1, "storecheck_sha256": _sha(w["a"][0]), "known_answers_sha256": _sha(w["a"][1])},
        {"period": 2, "change_at": RETURN_AT, "storecheck_sha256": _sha(w["a2"][0]),
         "known_answers_sha256": _sha(w["a2"][1])}]
    assert ["second_period" in p for p in copy["proofs"]] == [False, True] and "second_period" not in str(
        st["copies"]["copy-b"]["proofs"])
    assert [p["change_at"] for p in copy["proofs"]] == [None, RETURN_AT]
    assert copy["listed_by"] == {str(w["out"]): ["its manifest store change 1 (from)",
                                                 "its manifest store change 2 (to)"]}
    assert "split" not in copy and st["config"]["matched"] == 3  # one copy, not unsplittable; three entries matched
    assert E0 not in json.dumps(st) and E1 not in json.dumps(st)


def test_a_b_a_b_on_a_chain_of_a_b_a_is_refused_as_a_mix(world, ref):
    w = _aba_world(world, fourth="b")
    _refused_stores(world, [w["out"]], w["proofs"], w["entries"],
                    r"single: its records' own endpoints and times do not follow .*\(interleaved\): .*a mix of copies")


# --- a copy used in two periods (science, 25 Oct 00:10Z, point iii): A -> B -> A -----------------------------------------


def _where(ep: str = E0) -> str:
    return re.escape(f"single {ex.endpoint_label(ep)}")


def test_the_later_period_needs_exactly_its_second_period_entry(world, ref):
    w = _aba_world(world)
    first, b, second = w["entries"]
    # none: the return has no entry. The old workaround, the first period's entry carrying the return's instant, is
    # no longer one
    for entries in ([first, b], [{**first, "change_at": RETURN_AT}, b]):
        _refused_stores(world, [w["out"]], w["proofs"], entries,
                        rf"{_where()}: a later period of the copy without its entry")
    # two for one return, at different instants (the same instant twice is refused where the config is read)
    more = (_storecheck(world["tmp"] / "sc-a3.json", E0, "copy-a", checked_utc="2026-10-23T09:26:00Z"),
            _load10(world["tmp"] / "ka-a3.json", E0, started_utc="2026-10-23T09:26:05Z"))
    third = _entry("single", E0, "copy-a", more, change_at="2026-10-23T09:31:00Z", second_period=True)
    _refused_stores(world, [w["out"]], [*w["proofs"], *more], [first, b, second, third],
                    rf"{_where()}: more second_period entries than the OUT has later periods of the copy")


@pytest.mark.parametrize("change_at", ["2026-10-23T09:30:00.001Z", "2026-10-23T09:29:59.999Z", "2026-10-23T09:31:00Z"])
def test_a_second_periods_change_at_other_than_its_first_record_is_refused_even_by_a_millisecond(world, ref, change_at):
    w = _aba_world(world)
    entries = [*w["entries"][:2], {**w["entries"][2], "change_at": change_at}]
    _refused_stores(world, [w["out"]], w["proofs"], entries,
                    rf"{_where()} \(copy-a\): its change_at is not the first record on the copy "
                    r"\(the records' own endpoint is the ground truth\)")


def test_a_second_periods_entry_of_another_replica_is_another_copy_and_refused(world, ref):
    w = _aba_world(world)
    other = (_storecheck(world["tmp"] / "sc-z.json", E0, "copy-z", checked_utc=A2_CHECKED),
             _load10(world["tmp"] / "ka-z.json", E0, started_utc=A2_STARTED))
    entries = [*w["entries"][:2], _entry("single", E0, "copy-z", other, change_at=RETURN_AT, second_period=True)]
    _refused_stores(world, [w["out"]], [*w["proofs"], *other], entries,
                    rf"{_where()} \(copy-z\): its replica is not the first period's \(copy-a\)")


def test_a_second_periods_own_checks_are_held_to_its_own_change_at(world, ref):
    w = _aba_world(world)
    late = _storecheck(world["tmp"] / "sc-a2-late.json", E0, "copy-a", checked_utc="2026-10-23T09:31:00Z")
    entries = [*w["entries"][:2], {**w["entries"][2], "storecheck_sha256": _sha(late)}]
    _refused_stores(world, [w["out"]], [*w["proofs"], late], entries,
                    rf"{_where()} \(copy-a\): its storecheck was made after the copy took the OUT's attempts")
    ka_late = _load10(world["tmp"] / "ka-a2-late.json", E0, started_utc="2026-10-23T09:31:00Z")
    entries = [*w["entries"][:2], {**w["entries"][2], "known_answers_sha256": _sha(ka_late)}]
    _refused_stores(world, [w["out"]], [*w["proofs"], ka_late], entries,
                    rf"{_where()} \(copy-a\): its known answers were run after the copy took the OUT's attempts")


def test_a_second_period_entry_for_an_endpoint_the_out_uses_in_one_period_only_is_refused(world, ref):
    one = "a second_period entry for an endpoint the OUT uses in one period only"
    m = _moved_world(world)  # copy-a, then copy-b: each used once
    again_b = _entry("single", E1, "copy-b", m["b"], change_at="2026-10-23T09:30:00Z", second_period=True)
    again_a = _entry("single", E0, "copy-a", m["a"], change_at="2026-10-23T09:30:00Z", second_period=True)
    _refused_stores(world, [m["out"]], m["proofs"], [*m["entries"], again_b], rf"{_where(E1)}: {one}")
    _refused_stores(world, [m["out"]], m["proofs"], [*m["entries"], again_a], rf"{_where(E0)}: {one}")


def test_a_second_period_entry_for_an_out_with_one_endpoint_and_no_store_change_is_refused(world, ref):
    out = world["single"]  # the frozen fixture names one endpoint: no timeline, one period
    a = _copy_proofs(world["tmp"], "a", E0, "copy-a")
    stray = _entry("single", E0, "copy-a", a, change_at=RETURN_AT, second_period=True)
    _refused_stores(world, [out], a, [_entry("single", E0, "copy-a", a), stray],
                    rf"{_where()}: a second_period entry for an endpoint the OUT uses in one period only")


def test_an_endpoint_that_is_the_chains_start_needs_no_change_at_and_a_change_at_there_is_not_a_rule(world, ref):
    # today's behaviour, kept: prove() asks a change_at only of an endpoint a change moved the OUT to, and the first
    # period of the chain's start is not moved to; with a change_at it is held to the checks' bounds and nothing else
    w = _aba_world(world)
    with_at = {**w["entries"][0], "change_at": CHANGE_AT}
    assert _stores(world, [w["out"]], w["proofs"], [with_at, *w["entries"][1:]])["shares"][ARM] == _rows(
        {A_TWO: 3, "copy-b": 1})
    late = _storecheck(world["tmp"] / "sc-a-late.json", E0, "copy-a", checked_utc="2026-10-23T09:30:00Z")
    _refused_stores(world, [w["out"]], [*w["proofs"], late], [{**with_at, "storecheck_sha256": _sha(late)},
                                                               *w["entries"][1:]],
                    rf"{_where()} \(copy-a\): its storecheck was made after the copy took the OUT's attempts")


def test_a_change_at_on_the_first_copy_of_a_plain_move_is_held_to_the_checks_bounds_only(world, ref):
    m = _moved_world(world)  # a -> b, no return
    with_at = {**m["entries"][0], "change_at": CHANGE_AT}
    assert _stores(world, [m["out"]], m["proofs"], [with_at, m["entries"][1]])["shares"][ARM] == _rows(
        {"copy-a": 2, "copy-b": 2})


def test_the_first_period_of_a_copy_a_change_moved_the_out_to_still_needs_its_change_at(world, ref):
    # a -> b -> c -> b: b's first period began with change 1 and needs that change's instant, its return change 3's
    out, tmp = world["single"], world["tmp"]
    where = {"s01-pass": E0, "s01-flag": E1, "s01-abstain": EX, "s01-noquery": E1}
    _set_endpoints(out, lambda r: where[r["question_id"]])
    first = {"s01-pass": "09:10:00", "s01-flag": "09:18:45", "s01-abstain": "09:30:00", "s01-noquery": "09:40:00"}
    seen: Counter = Counter()

    def start_of(r, f):
        k = seen[r["question_id"]]
        seen[r["question_id"]] += 1
        return datetime.fromisoformat(f"2026-10-23T{first[r['question_id']]}+00:00") + timedelta(seconds=k)

    _store_change(out, E0, E1, _ids(out, "s01-pass"), stamp=False)
    _store_change(out, E1, EX, _ids(out, "s01-pass", "s01-flag"), stamp=False)
    _store_change(out, EX, E1, _ids(out, "s01-pass", "s01-flag", "s01-abstain"), stamp=False)
    _retime(out, start_of)
    a, b, x = (_copy_proofs(tmp, n, ep, f"copy-{n}") for n, ep in (("a", E0), ("b", E1), ("x", EX)))
    b2 = (_storecheck(tmp / "sc-b2.json", E1, "copy-b", checked_utc="2026-10-23T09:35:00Z"),
          _load10(tmp / "ka-b2.json", E1, started_utc="2026-10-23T09:35:05Z"))
    entries = [_entry("single", E0, "copy-a", a), _entry("single", E1, "copy-b", b, change_at="2026-10-23T09:18:45Z"),
               _entry("single", EX, "copy-x", x, change_at="2026-10-23T09:30:00Z"),
               _entry("single", E1, "copy-b", b2, change_at="2026-10-23T09:40:00Z", second_period=True)]
    files = (*a, *b, *x, *b2)
    st = _stores(world, [out], files, entries)
    assert sorted(st["copies"]) == ["copy-a", "copy-b (two periods)", "copy-x"]
    assert [p["period"] for p in st["copies"]["copy-b (two periods)"]["periods"]] == [1, 2]
    assert st["copies"]["copy-b (two periods)"]["periods"][0]["change_at"] == "2026-10-23T09:18:45Z"  # kept when present
    no_at = {k: v for k, v in entries[1].items() if k != "change_at"}
    _refused_stores(world, [out], files, [entries[0], no_at, *entries[2:]], rf"{_where(E1)} \(copy-b\): .*without its change_at")
    # first-record rule (science, 25 Oct, Gemma e190 K=3): b's first period carrying its return's instant (09:40:00, the
    # old workaround) and x's entry five seconds late are timed by their first records, 09:18:45 and 09:30:00: the gaps,
    # by hand, are 09:18:45 - 09:40:00 = -1275 s and 09:30:00 - 09:30:05 = -5 s. b's second period keeps its equality.
    off = [entries[0], {**entries[1], "change_at": "2026-10-23T09:40:00Z"}, {**entries[2], "change_at": "2026-10-23T09:30:05Z"},
           entries[3]]
    one, two, three = _stores(world, [out], files, off)["store_changes"]
    assert [c["first_record_change_point"]["entry_gap_seconds"] for c in (one, two)] == [-1275.0, -5.0]
    assert [c["first_record_change_point"]["entry_change_at"] for c in (one, two)] == [
        "2026-10-23T09:40:00Z", "2026-10-23T09:30:05Z"]
    # the manifest's instant (the fixture writes 09:18:39.962 for every change), by hand: 09:18:45.000 - 09:18:39.962 =
    # 5.038 s and 09:30:00.000 - 09:18:39.962 = 11 min 20.038 s = 680.038 s
    assert [(c["first_record_change_point"]["manifest_at"], c["first_record_change_point"]["manifest_gap_seconds"])
            for c in (one, two)] == [(MANIFEST_AT, 5.038), (MANIFEST_AT, 680.038)]
    assert [c["first_record_change_point"]["change_point"] for c in (one, two)] == [
        "2026-10-23T09:18:45.000+00:00", "2026-10-23T09:30:00.000+00:00"]
    assert "first_record_change_point" not in three and three["first_record_on_new_endpoint"] == (
        "2026-10-23T09:40:00.000+00:00")
    _refused_stores(world, [out], files, [*off[:3], {**entries[3], "change_at": "2026-10-23T09:40:01Z"}],
                    rf"{_where(E1)} \(copy-b\): its change_at is not the first record on the copy")


def test_the_periods_identity_must_be_identical(world, ref):
    w = _aba_world(world)
    first, b, second = w["entries"]

    def with_second(name: str, **over):
        sc = _storecheck(world["tmp"] / f"{name}.json", E0, "copy-a", checked_utc=A2_CHECKED, **over)
        return [*w["proofs"], sc], [first, b, {**second, "storecheck_sha256": _sha(sc)}]

    # each period identical to the pinned reference (this is all check_storecheck compares) yet the two periods' own
    # records of the copy's stats differ: refused as the periods' difference
    files, entries = with_second("sc-a2-stats", stats_sha256="11" * 32)
    _refused_stores(world, [w["out"]], files, entries,
                    rf"{_where()}: the copy's identity differs between its periods \(the period without a change_at and "
                    rf"the period from {RETURN_AT}, both storecheck files\)")
    # a period that is not itself identical to the reference is its own refusal, and the periods are not compared
    for name, over, match in (
            ("not-identical", {"identical_to_reference": False, "differ": ["num-triples-normal"]},
             r"\(copy-a\): its storecheck is not identical to the reference stats"),
            ("other-reference", {"reference_sha256": "11" * 32},
             r"\(copy-a\): its storecheck compares with another reference than s01's recorded stats")):
        files, entries = with_second(f"sc-a2-{name}", **over)
        with pytest.raises(ex.Refused, match=match) as caught:
            _check(world, w["out"], files, entries)
        assert "differs between its periods" not in str(caught.value)
    files, entries = with_second("sc-a2-same")  # and the same stats with another check time is accepted
    assert _stores(world, [w["out"]], files, entries)["shares"][ARM] == _rows({A_TWO: 3, "copy-b": 1})
    gone = {k: v for k, v in json.loads((world["tmp"] / "sc-a2-same.json").read_text(encoding="utf-8")).items()
            if k != "stats_sha256"}  # a storecheck that does not record its stats cannot be set beside the other
    (world["tmp"] / "sc-a2-gone.json").write_text(json.dumps(gone), encoding="utf-8")
    gone_path = world["tmp"] / "sc-a2-gone.json"
    _refused_stores(world, [w["out"]], [*w["proofs"], gone_path], [first, b, {**second, "storecheck_sha256": _sha(gone_path)}],
                    rf"{_where()}: the period from {RETURN_AT}: its storecheck does not record both its reference_sha256 and "
                    r"its stats_sha256")


def test_periods_identity_compares_each_format_by_type_and_value_and_formats_only_through_the_reference(tmp_path):
    def entry(path: Path, **over) -> dict:
        return {"storecheck_sha256": _sha(path), "replica": "copy-a", **over}

    def files(*paths: Path) -> dict:
        return {_sha(p): (str(p), json.loads(p.read_text(encoding="utf-8"))) for p in paths}

    one = _ident(tmp_path / "one.json", {E0: dict(STATS)})
    same = _ident(tmp_path / "same.json", {E0: dict(STATS)}, checked_utc="2026-10-23T09:25:00Z")
    count = _ident(tmp_path / "count.json", {E0: {**STATS, "num-triples-normal": 4043}})
    typed = _ident(tmp_path / "typed.json", {E0: {**STATS, "num-permutations": 6.0}})  # 6 is an int in STATS
    short = _ident(tmp_path / "short.json", {E0: {k: v for k, v in STATS.items() if k != "num-triples-normal"}})
    elsewhere = _ident(tmp_path / "elsewhere.json", {EX: dict(STATS)})
    sc = _storecheck(tmp_path / "sc.json", E0, "copy-a")
    sc_other = _storecheck(tmp_path / "sc-other.json", E0, "copy-a", stats_sha256="11" * 32)
    every = files(one, same, count, typed, short, elsewhere, sc, sc_other)
    old = {"identity_format": "ident.py"}
    T2, T3 = "2026-10-23T09:30:00Z", "2026-10-23T09:40:00Z"  # the later periods' instants; the first has none
    assert ex.periods_identity([entry(one, **old), entry(same, change_at=T2, **old)], E0, every) == []
    for other, field in ((count, "num-triples-normal"), (typed, "num-permutations")):
        assert ex.periods_identity([entry(one, **old), entry(other, change_at=T2, **old)], E0, every) == [
            f"the copy's identity differs between its periods (the period without a change_at and the period from {T2}, "
            f"both ident.py files)"], field
    cannot = (f"the period from {T2}: its identity check's copy does not record the fields ident.py compares, so its "
              f"identity cannot be set beside the other periods'")
    assert ex.periods_identity([entry(one, **old), entry(short, change_at=T2, **old)], E0, every) == [cannot]
    assert ex.periods_identity([entry(one, **old), entry(elsewhere, change_at=T2, **old)], E0, every) == [cannot]
    # (the second: no entry of the endpoint in that file, so nothing to compare)
    # two storechecks: the same reference and stats, or not
    assert ex.periods_identity([entry(sc), entry(sc)], E0, every) == []
    assert ex.periods_identity([entry(sc), entry(sc_other, change_at=T2)], E0, every) == [
        f"the copy's identity differs between its periods (the period without a change_at and the period from {T2}, "
        f"both storecheck files)"]
    # an ident.py file and a storecheck share no field but the reference they are each held to: not compared here
    assert ex.periods_identity([entry(one, **old), entry(sc_other, change_at=T2)], E0, every) == []
    assert ex.periods_identity([entry(sc_other), entry(one, change_at=T2, **old), entry(sc, change_at=T3)], E0,
                               every) == [
        f"the copy's identity differs between its periods (the period without a change_at and the period from {T3}, "
        f"both storecheck files)"]


def _reference_aba(world, monkeypatch, *, second: str) -> dict:
    """The ABA fixture whole with every identity check held to one pinned reference FILE (as UniProt's and Rhea's old
    checks are): copy-a's first period by ident.py (the reference file's own copies block), its second period by
    ident.py or by a storecheck (``second``), copy-b by a storecheck naming the reference file's sha256."""
    w = _aba_world(world)
    tmp = world["tmp"]
    reference = _ident(tmp / "ref-stats.json", {E0: dict(STATS)}, checked_utc="2026-10-23T08:00:00Z")
    monkeypatch.setitem(ex.REFERENCE_STATS_SHA256, "s01", _sha(reference))
    monkeypatch.setitem(ex.REFERENCE_NAME_INDEX, "s01", NAME_INDEX)
    ka1, kb = _load10(tmp / "ka-a1.json", E0), _load10(tmp / "ka-b1.json", E1)
    sc_b = _storecheck(tmp / "sc-b1.json", E1, "copy-b", reference_sha256=_sha(reference))
    ka2 = _load10(tmp / "ka-a2n.json", E0, started_utc=A2_STARTED)
    if second == "ident.py":
        id2, extra = _ident(tmp / "ident-a2n.json", {E0: dict(STATS)}, checked_utc=A2_CHECKED), OLD
    else:
        id2, extra = _storecheck(tmp / "sc-a2n.json", E0, "copy-a", checked_utc=A2_CHECKED,
                                 reference_sha256=_sha(reference)), {}
    return {"out": w["out"], "proofs": [reference, ka1, sc_b, kb, id2, ka2],
            "entries": [_entry("single", E0, "copy-a", (reference, ka1), **OLD),
                        _entry("single", E1, "copy-b", (sc_b, kb), change_at="2026-10-23T09:18:45Z"),
                        _entry("single", E0, "copy-a", (id2, ka2), change_at=RETURN_AT, second_period=True, **extra)]}


@pytest.mark.parametrize("second", ["ident.py", "storecheck"])
def test_two_periods_of_old_format_and_new_format_identity_checks_are_accepted_through_the_one_reference(
        world, monkeypatch, second):
    w = _reference_aba(world, monkeypatch, second=second)
    st = _stores(world, [w["out"]], w["proofs"], w["entries"])
    assert st["shares"][ARM] == _rows({A_TWO: 3, "copy-b": 1})
    assert st["copies"][A_TWO]["not_checked"] == sorted(NOT_CHECKED)  # from the ident.py period(s)
    assert [p.get("identity_format") for p in st["copies"][A_TWO]["proofs"]] == [
        "ident.py", "ident.py" if second == "ident.py" else None]
    # a period not identical to that one reference is refused, in either format
    bad = (_ident(world["tmp"] / "ident-bad.json", {E0: {**STATS, "num-triples-normal": 4043}}, checked_utc=A2_CHECKED)
           if second == "ident.py" else _storecheck(world["tmp"] / "sc-bad.json", E0, "copy-a", checked_utc=A2_CHECKED,
                                                    reference_sha256="11" * 32))
    entries = [*w["entries"][:2], {**w["entries"][2], "storecheck_sha256": _sha(bad)}]
    match = (r"differs from s01's recorded reference in \['num-triples-normal'\]" if second == "ident.py"
             else r"compares with another reference than s01's recorded stats")
    _refused_stores(world, [w["out"]], [*w["proofs"], bad], entries, rf"{_where()} \(copy-a\): .*{match}")


def test_each_period_of_a_copy_named_by_host_name_is_proven_at_its_pinned_alias(world, monkeypatch):
    """Gemma's Rhea OUT shape: the OUT names rhea-c06 by host name in both periods and the recorded checks name its
    address, so each period's entry carries the pinned alias with its own pair and the same further points."""
    w = _aba_world(world)
    tmp = world["tmp"]
    kit = _alias_kit(world, monkeypatch, E0, EA)
    ident_1, ka_1 = _ident(tmp / "ident-1.json", {EA: dict(STATS)}), _load10(tmp / "ka-1.json", EA)
    ident_2 = _ident(tmp / "ident-2.json", {EA: dict(STATS)}, checked_utc=A2_CHECKED)
    ka_2 = _load10(tmp / "ka-2.json", EA, started_utc=A2_STARTED)
    b = (_storecheck(tmp / "sc-bb.json", E1, "copy-b", reference_sha256=_sha(kit["reference"])),
         _load10(tmp / "ka-bb.json", E1))
    one = _entry("single", E0, "copy-a", (ident_1, ka_1), **OLD, alias=kit["alias"], further=kit["further"])
    two = _entry("single", E0, "copy-a", (ident_2, ka_2), **OLD, alias=kit["alias"], further=kit["further"],
                 change_at=RETURN_AT, second_period=True)
    entries = [one, _entry("single", E1, "copy-b", b, change_at="2026-10-23T09:18:45Z"), two]
    files = [kit["reference"], kit["table"], kit["ka_ref"], kit["ident_r"], kit["ka_r"], ident_1, ka_1, ident_2, ka_2, *b]
    st = _stores(world, [w["out"]], files, entries)
    assert st["shares"][ARM] == _rows({A_TWO: 3, "copy-b": 1})
    alias = {"endpoint": f"endpoint sha256:{_h(EA)[:12]}", "record_sha256": _sha(kit["table"])}
    assert [p["alias"] for p in st["copies"][A_TWO]["proofs"]] == [alias, alias]
    assert [p["period"] for p in st["copies"][A_TWO]["periods"]] == [1, 2] and "192.0.2" not in json.dumps(st)
    # the later period without its alias: its ident.py file names the copy by address, which the OUT's endpoint is not
    bare = {k: v for k, v in two.items() if k not in ("alias", "further")}
    _refused_stores(world, [w["out"]], files, [one, entries[1], bare],
                    rf"{_where()} \(copy-a\): its identity check does not have exactly one entry of the OUT's endpoint")


def test_prove_tests_what_the_outs_identity_records_for_the_last_period_only(world, ref):
    w = _aba_world(world)
    out = w["out"]

    def records(sc: Path, replica: str = "copy-a") -> None:
        check = f"?cmd=stats equal to ref in 2 fields at {CHECKED_AT} (store check sha256 {_sha(sc)})"
        _edit_identity(out, lambda i: i["store"].update(endpoint=E0, replica=replica, identity_check=check))

    # the identity records the second period's storecheck (the real case: the cutover check of the return), so the
    # first period's entry, which carries other files, is not held to it
    records(w["a2"][0])
    assert _stores(world, [out], w["proofs"], w["entries"])["shares"][ARM] == _rows({A_TWO: 3, "copy-b": 1})
    # recorded: the first period's storecheck. The last period's entry is then the one refused, and only it
    records(w["a"][0])
    with pytest.raises(ex.Refused) as caught:
        _check(world, out, w["proofs"], w["entries"])
    assert str(caught.value).count("its storecheck is not the one the OUT's identity records for this endpoint") == 1
    # the replica the OUT's identity names is the last period's too (both periods are the one replica, so one refusal)
    records(w["a2"][0], replica="copy-z")
    with pytest.raises(ex.Refused) as caught:
        _check(world, out, w["proofs"], w["entries"])
    assert str(caught.value).count("(copy-a): the OUT's identity names copy-z at this endpoint") == 1


def test_without_a_later_period_every_ordinary_entry_is_still_held_to_what_the_outs_identity_records(world, ref):
    """The OUT is known by several paths (its local ones and its identity's), so an endpoint can have two ordinary
    entries; with no second_period entry each is tested against the identity, as before the two-period rule."""
    m = _moved_world(world)
    out = m["out"]
    check = f"?cmd=stats equal to ref in 2 fields at {CHECKED_AT} (store check sha256 {_sha(m['a'][0])})"
    _edit_identity(out, lambda i: i["store"].update(endpoint=E0, replica="copy-a", identity_check=check))
    assert _stores(world, [out], m["proofs"], m["entries"])["shares"][ARM] == _rows({"copy-a": 2, "copy-b": 2})
    other = (_storecheck(world["tmp"] / "sc-a-other.json", E0, "copy-a", checked_utc="2026-10-23T08:00:00Z"),
             _load10(world["tmp"] / "ka-a-other.json", E0))
    twin = _entry("single", E0, "copy-a", other)
    twin["out"] = str(out.absolute())  # the same OUT under its local path
    _refused_stores(world, [out], [*m["proofs"], *other], [*m["entries"], twin],
                    rf"{_where()} \(copy-a\): its storecheck is not the one the OUT's identity records")


def test_one_ordinary_entry_per_endpoint_of_an_out_known_by_several_paths_even_with_a_later_period(world, ref):
    """external reviewer's finding on the merge: an OUT is known by several paths and read_store_proofs keys its one first copy on
    the literal path, so a second ordinary entry of another replica under another path used to pass once a later period
    existed (the identity tests were then skipped for both, and the period logic read the first only)."""
    w = _aba_world(world)
    z = (_storecheck(world["tmp"] / "sc-z0.json", E0, "copy-z"), _load10(world["tmp"] / "ka-z0.json", E0))
    twin = _entry("single", E0, "copy-z", z)  # valid proof files of another replica at the same endpoint
    twin["out"] = str(w["out"].absolute())  # the same OUT under its local path
    why = rf"{_where()}: more than one ordinary entry for the OUT's endpoint"
    _refused_stores(world, [w["out"]], [*w["proofs"], *z], [*w["entries"], twin], why)
    # the same replica under the other path is as much a second first copy; and the order of the entries is no matter
    same = {**w["entries"][0], "out": str(w["out"].absolute())}
    _refused_stores(world, [w["out"]], w["proofs"], [same, *w["entries"]], why)
    with pytest.raises(ex.Refused) as caught:
        _check(world, w["out"], [*w["proofs"], *z], [*w["entries"], twin])
    assert not re.search(r"[0-9]+ ordinary", str(caught.value))  # and no count in the message
    # a later period's entry under the other path is still the OUT-name refusal, not this one
    elsewhere = {**w["entries"][2], "out": str(w["out"].absolute())}
    with pytest.raises(ex.Refused) as caught:
        _check(world, w["out"], w["proofs"], [*w["entries"][:2], elsewhere])
    assert "more than one ordinary entry" not in str(caught.value)


def test_a_second_period_entry_must_name_the_first_periods_out_as_it_does(world, ref):
    w = _aba_world(world)
    elsewhere = {**w["entries"][2], "out": str(w["out"].absolute())}  # the same OUT, another path it is known by
    _refused_stores(world, [w["out"]], w["proofs"], [*w["entries"][:2], elsewhere],
                    rf"{_where()} \(copy-a\): its OUT is not the first period's")


def test_a_copy_used_in_two_periods_and_also_behind_an_endpoint_is_refused(world, ref):
    w = _aba_world(world)
    behind = _entry("single", E0, "copy-y", _copy_proofs(world["tmp"], "y", E0, "copy-y"),
                    change_at="2026-10-23T09:50:00Z", behind_endpoint=True)
    files = [*w["proofs"], *_copy_proofs(world["tmp"], "y", E0, "copy-y")]
    _refused_stores(world, [w["out"]], files, [*w["entries"], behind],
                    rf"{_where()}: a copy used in two periods and also behind an endpoint")


def _gzip(path: Path) -> Path:
    gz = path.with_name(path.name + ".gz")
    gz.write_bytes(gzip.compress(path.read_bytes()))
    path.unlink()
    return gz


def _setaside(out: Path, record: dict, run_id: str, endpoint: str, started: str, *, gz: bool = False) -> Path:
    """A set-aside try (infrastructure/): the record under another run id, endpoint and start (the name is the
    runner's <key>.<run id>.json)."""
    r = {**record, "run_id": run_id, "store": {**record["store"], "endpoint": endpoint}, "started_at": started,
         "ended_at": _iso(datetime.fromisoformat(started) + timedelta(milliseconds=500))}
    (out / "infrastructure").mkdir(exist_ok=True)
    path = out / "infrastructure" / f"{'f' * 40}.{run_id}.json"
    if gz:
        path = path.with_name(path.name + ".gz")
        path.write_bytes(gzip.compress(json.dumps(r).encode("utf-8")))
    else:
        path.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    _recheck(out)
    return path


def test_set_aside_and_gzipped_records_are_in_the_timeline(world, ref):
    m = _moved_world(world)
    out = m["out"]
    record = _record_of(out, "s01-noquery")[1]  # a record on copy-b
    early = _setaside(out, record, "bbbbbbbb-0000-0000-0000-000000000001", E1, "2026-10-23T09:18:30.000+00:00")
    # a set-aside try on copy-b at 09:18:30 is the first record on it, 15 seconds before the first in records/. The entry
    # still records 09:18:45: since science's ruling of 25 Oct (Gemma e190 K=3) the change point is the first record,
    # 09:18:30, and the entry's instant is disclosed with the gap, 09:18:30 - 09:18:45 = -15 s, and the manifest's
    # (09:18:39.962) with its own, 09:18:30.000 - 09:18:39.962 = -9.962 s
    (change,) = _stores(world, [out], m["proofs"], m["entries"])["store_changes"]
    assert change["first_record_on_new_endpoint"] == change["change_at_used"] == "2026-10-23T09:18:30.000+00:00"
    assert {k: v for k, v in change["first_record_change_point"].items() if k != "rule"} == {
        "change_point": "2026-10-23T09:18:30.000+00:00", "manifest_at": MANIFEST_AT, "manifest_gap_seconds": -9.962,
        "entry_change_at": CHANGE_AT, "entry_gap_seconds": -15.0}
    entries = [m["entries"][0], {**m["entries"][1], "change_at": "2026-10-23T09:18:30.000Z"}]
    (change,) = _stores(world, [out], m["proofs"], entries)["store_changes"]
    assert change["first_record_on_new_endpoint"] == "2026-10-23T09:18:30.000+00:00"
    assert "first_record_change_point" not in change  # the entry records the first record: nothing to disclose
    assert (change["setaside_changes_first_instant"], change["first_precedes_manifest_at"]) == (True, False)
    early.unlink()
    # a set-aside try on copy-a made at 09:25, after every record on copy-b, is the copies interleaving
    _setaside(out, _record_of(out, "s01-pass")[1], "aaaaaaaa-0000-0000-0000-000000000002", E0,
              "2026-10-23T09:25:00.000+00:00")
    _refused_stores(world, [out], m["proofs"], m["entries"], r"\(interleaved\): .*a mix of copies, not a late timestamp")
    for f in (out / "infrastructure").glob("*"):
        f.unlink()
    # gzipped records count: the first record on copy-b is a .json.gz in records/, and a try on it is one too
    first_b = min(((json.loads(f.read_text(encoding="utf-8"))["started_at"], f) for f in _record_files(out)
                   if json.loads(f.read_text(encoding="utf-8"))["store"]["endpoint"] == E1))
    assert first_b[0] == "2026-10-23T09:18:45.000+00:00"
    gz_run_id = json.loads(first_b[1].read_text(encoding="utf-8"))["run_id"]
    gz = _gzip(first_b[1])
    other = _setaside(out, record, "bbbbbbbb-0000-0000-0000-000000000003", E1, "2026-10-23T09:21:00.000+00:00", gz=True)
    assert gz.name.endswith(".json.gz") and other.name.endswith(".json.gz")
    timeline = _check(world, out, m["proofs"], m["entries"]).timelines[str(out)]
    assert timeline.clean and timeline.period_of[gz_run_id] == 1
    assert timeline.changes[0].first_to_started_text == "2026-10-23T09:18:45.000+00:00"
    assert timeline.changes[0].setaside_changes_first_instant is False


def test_a_branch_style_record_made_later_on_the_new_copy_is_clean_and_not_a_count(world, ref, capsys):
    m = _moved_world(world)
    out = m["out"]
    f, record = next((f, r) for f in sorted((out / "records").glob("*.json"))
                     if (r := json.loads(f.read_text(encoding="utf-8")))["load"]["arm"] == "gen:any"
                     and r["store"]["endpoint"] == E1)
    assert record["run_id"] not in m["before"]  # a branch record of the checker arm, made after the move
    _retime(out, lambda r, g: "2026-10-23T09:40:00.000+00:00" if r["run_id"] == record["run_id"] else None)
    timeline = _check(world, out, m["proofs"], m["entries"]).timelines[str(out)]
    assert timeline.clean and timeline.period_of[record["run_id"]] == 1
    assert timeline.changes[0].first_to_started_text == "2026-10-23T09:18:45.000+00:00"
    assert (timeline.changes[0].manifest_lists_to_endpoint_record_as_before,
            timeline.changes[0].manifest_omits_from_endpoint_record) == (False, False)
    code, (obj,), raw, err = _cli(capsys, out)  # and the command's output holds no count of any kind
    assert (code, err, obj["verdict"], set(obj)) == (0, "", "CLEAN", CLI_KEYS)
    assert all(set(c) == CLI_CHANGE_KEYS for c in obj["changes"]) and _ints(obj) == [".changes[].index"]
    assert record["run_id"] not in raw  # the keys and the one int are the whole of it; no run id


# --- the branch phase in the fixture world: check_stores, the report and the command (science, 25 Oct ~04:55Z) ------

T_B = "2026-10-23T09:25:00.000+00:00"  # the manifest's phase-branch instant, after every record the fixture stamps
LATE_A, LATE_B = "2026-10-23T09:25:10.000+00:00", "2026-10-23T09:25:20.000+00:00"
LABEL_A, LABEL_B = f"endpoint sha256:{_h(E0)[:12]}", f"endpoint sha256:{_h(E1)[:12]}"


def _middle_record(out: Path, endpoint: str, keep) -> dict | None:
    """A records/ record on ``endpoint`` that ``keep`` accepts and that is neither the first nor the last of the
    endpoint's records by started_at, so retiming it leaves the copy's first and last instants as the fixture stamps them
    (copy-a from 09:10:00.000 to 09:10:19.000, copy-b from 09:18:45.000 to 09:19:00.000)."""
    records = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((out / "records").glob("*.json"))]
    starts = sorted(r["started_at"] for r in records if r["store"]["endpoint"] == endpoint)
    found = [r for r in records if r["store"]["endpoint"] == endpoint and keep(r) and starts[0] < r["started_at"] < starts[-1]]
    return found[0] if found else None


def _two_late_rows(out: Path, arm_of=lambda r: r["load"]["arm"]) -> tuple[str, dict, dict]:
    """(the arm cell, a copy-a record, a copy-b record) of one prompt arm of QWEN, retimed into the branch phase (LATE_A,
    LATE_B, both after T_B) with their endpoints untouched: a parent's endpoint on a row made long after the move."""
    for cell in CELLS:
        a = _middle_record(out, E0, lambda r: r["question_id"] == "s01-pass" and r["model"] == QWEN and arm_of(r) == cell)
        for question in ("s01-abstain", "s01-noquery"):
            b = _middle_record(out, E1, lambda r: r["question_id"] == question and r["model"] == QWEN and arm_of(r) == cell)
            if a is not None and b is not None:
                times = {a["run_id"]: LATE_A, b["run_id"]: LATE_B}
                _retime(out, lambda r, f: times.get(r["run_id"]))
                return cell, a, b
    raise AssertionError("the fixture has no two middle records of one prompt arm")


def test_branch_phase_rows_are_not_a_mix_in_the_call_the_report_and_the_command(world, ref, capsys):
    m = _moved_world(world)
    out = m["out"]
    cell, a, b = _two_late_rows(out)
    # without a phase-branch instant in the manifest: a copy-a row after the first copy-b row is a mix, as it was
    _refused_check(world, out, m["proofs"], m["entries"], r"\(interleaved\)")
    code, (obj,), raw, err = _cli(capsys, out)
    assert (code, obj["verdict"], obj["branch_phase_start"], obj["branch_row_endpoints"]) == (2, "STOP: interleaved", None, None)
    # with it: the two late rows are branch rows, whatever endpoint they record
    _set_branch_phase(out, T_B)
    line = _check(world, out, m["proofs"], m["entries"]).timelines[str(out)]
    assert line.clean and line.branch_phase_start == T_B
    assert line.branch_row_endpoints == (E1, E0)  # sorted as strings: http://copy-b... before http://store...
    assert line.period_of[a["run_id"]] is None and line.period_of[b["run_id"]] is None
    assert line.changes[0].first_to_started_text == "2026-10-23T09:18:45.000+00:00"
    st = _stores(world, [out], m["proofs"], m["entries"])
    (change,) = st["store_changes"]
    assert change["branch_phase_start"] == T_B
    assert change["first_record_on_new_endpoint"] == change["change_at_used"] == "2026-10-23T09:18:45.000+00:00"
    arm = f"s01|{QWEN}|{cell}|prompt|-"
    # the arm's four attempts: copy-a's s01-flag before, the other copy-b one after, the two late ones in the branch phase;
    # the late rows record copy-a and copy-b (their parents') but both ran their live retry on copy-b, the chain's last
    assert change["shares_by_arm"][arm] == {"before": _rows({"copy-a": 1}), "after": _rows({"copy-b": 1}),
                                            "branch_phase": _rows({"copy-b": 2})}
    for other in (c for c in CELLS if c != cell):  # the other arms have none: an empty bucket, not a missing one
        assert change["shares_by_arm"][f"s01|{QWEN}|{other}|prompt|-"] == {
            "before": _rows({"copy-a": 2}), "after": _rows({"copy-b": 2}), "branch_phase": _rows({})}
    assert all("branch_phase" in split for split in change["shares_by_arm"].values())
    # (science, 25 Oct ~04:55Z) a branch record counts for the copy current at its start, not the copy it records: the
    # copy-a-recording late row is copy-b's, so the arm's four attempts are copy-a 1 (s01-flag before) and copy-b 3
    assert st["shares"][arm] == _rows({"copy-a": 1, "copy-b": 3})
    assert st["rule"].endswith("; records of the two copies interleaving is a mix and refuses. " + BRANCH_RULE_TEXT)
    code, (obj,), raw, err = _cli(capsys, out)
    assert (code, err, obj["verdict"], set(obj)) == (0, "", "CLEAN", CLI_KEYS)
    assert obj["branch_phase_start"] == T_B and obj["branch_row_endpoints"] == sorted([LABEL_A, LABEL_B])
    assert [(r["endpoint"], r["first_started"], r["last_started"], r["last_ended"]) for r in obj["runs"]] == [
        (LABEL_A, "2026-10-23T09:10:00.000+00:00", "2026-10-23T09:10:19.000+00:00", "2026-10-23T09:10:19.500+00:00"),
        (LABEL_B, "2026-10-23T09:18:45.000+00:00", "2026-10-23T09:19:00.000+00:00", "2026-10-23T09:19:00.500+00:00")]
    assert all(set(c) == CLI_CHANGE_KEYS for c in obj["changes"]) and _ints(obj) == [".changes[].index"]
    assert a["run_id"] not in raw and b["run_id"] not in raw and "store.invalid" not in raw and "copy-b.invalid" not in raw


def test_branch_records_of_the_checker_arm_made_on_their_parents_copies_after_the_branch_instant_are_clean(world, ref, capsys):
    m = _moved_world(world)
    out = m["out"]
    a = _middle_record(out, E0, lambda r: r["load"]["arm"] == "gen:any")
    b = _middle_record(out, E1, lambda r: r["load"]["arm"] == "gen:any")
    assert a is not None and b is not None
    times = {a["run_id"]: LATE_A, b["run_id"]: LATE_B}
    _retime(out, lambda r, f: times.get(r["run_id"]))
    _refused_check(world, out, m["proofs"], m["entries"], r"\(interleaved\)")
    _set_branch_phase(out, T_B)
    line = _check(world, out, m["proofs"], m["entries"]).timelines[str(out)]
    assert line.clean and line.period_of[a["run_id"]] is None and line.period_of[b["run_id"]] is None
    assert line.branch_row_endpoints == (E1, E0)
    code, (obj,), raw, err = _cli(capsys, out)
    assert (code, err, obj["verdict"]) == (0, "", "CLEAN") and obj["branch_row_endpoints"] == sorted([LABEL_A, LABEL_B])


# --- the branch rows' live retry is attributed to the copy current at their start (science, 25 Oct ~04:55Z) --------------
#
# Hand-derived: harness/checkloop/online.py:542 (branch_attempt) copies the PARENT record's store into the branch attempt,
# so a branch row's recorded endpoint is its parent's copy; only its live retry ran on the runner's current store client,
# the copy current when the branch phase began, which is the chain's last (every store change precedes T_b).


def test_branch_rows_recording_either_copy_are_the_chains_last_copys_in_every_share(world, ref):
    m = _moved_world(world)
    out = m["out"]
    cell, a, b = _two_late_rows(out)  # a records copy-a, b records copy-b; both retimed after T_B
    _set_branch_phase(out, T_B)
    st = _stores(world, [out], m["proofs"], m["entries"])
    arm = f"s01|{QWEN}|{cell}|prompt|-"
    # the arm's four attempts: s01-flag on copy-a before the move (1), the other copy-b record after it (1), and the two
    # late rows, which recorded copy-a and copy-b and are both copy-b's (2): copy-a 1, copy-b 3 (it was 2 and 2)
    assert st["shares"][arm] == _rows({"copy-a": 1, "copy-b": 3})
    assert st["shares_by_out"][str(out)][arm] == _rows({"copy-a": 1, "copy-b": 3})
    for other in (c for c in CELLS if c != cell):  # no branch row: the recorded endpoint is the copy, as before
        assert st["shares"][f"s01|{QWEN}|{other}|prompt|-"] == _rows({"copy-a": 2, "copy-b": 2})
    (change,) = st["store_changes"]
    assert change["shares_by_arm"][arm] == {"before": _rows({"copy-a": 1}), "after": _rows({"copy-b": 1}),
                                            "branch_phase": _rows({"copy-b": 2})}


def test_branch_rows_recording_the_middle_copy_of_a_b_a_chain_are_the_first_copys_second_period(world, ref):
    w = _aba_world(world)
    out = w["out"]
    for cell in CELLS:  # a copy-b record that is neither the first nor the last of copy-b's: its retime moves no change_at
        b = _middle_record(out, E1, lambda r: r["question_id"] == "s01-flag" and r["model"] == QWEN
                           and r["load"]["arm"] == cell)
        if b is not None:
            break
    assert b is not None
    t_b, late = "2026-10-23T09:50:00.000+00:00", "2026-10-23T09:51:00.000+00:00"  # after every record _aba_world stamps
    _retime(out, lambda r, f: late if r["run_id"] == b["run_id"] else None)
    _set_branch_phase(out, t_b)
    line = _check(world, out, w["proofs"], w["entries"]).timelines[str(out)]
    assert line.clean and line.period_of[b["run_id"]] is None  # a branch row, in no period
    st = _stores(world, [out], w["proofs"], w["entries"])
    arm = f"s01|{QWEN}|{cell}|prompt|-"
    # the arm's s01-pass, s01-abstain and s01-noquery are on copy-a and its s01-flag row recorded copy-b, but that row is
    # a branch row: its live retry ran on the chain's last copy, copy-a (it was copy-a 3, copy-b 1)
    assert st["shares"][arm] == _rows({A_TWO: 4})
    one, two = st["store_changes"]
    assert one["shares_by_arm"][arm] == {"before": _rows({A_TWO: 1}), "after": _rows({A_TWO: 2}),
                                         "branch_phase": _rows({A_TWO: 1})}
    assert two["shares_by_arm"][arm] == {"before": _rows({A_TWO: 1}), "after": _rows({A_TWO: 2}),
                                         "branch_phase": _rows({A_TWO: 1})}


def test_an_out_with_a_branch_phase_and_no_store_changes_keeps_its_recorded_endpoint(world, ref):
    _set_branch_phase(world["single"], T_B)
    st = _stores(world, [world["single"]], (), ())
    assert st["store_changes"] == []
    for cell in CELLS:
        assert st["shares"][f"s01|{QWEN}|{cell}|prompt|-"] == _rows({LABEL_A: 4})


def _hand_line(endpoints: list[str], firsts: list, branch_sec: float = 15):
    """A StoreTimeline of the chain ``endpoints`` whose changes' first records on their new copies are ``firsts`` (a
    datetime, or anything else for a change the records do not reach), and the branch phase entered ``branch_sec``
    seconds after 09:00:00."""
    changes = tuple(ex.ChangeTimeline(i, a, b, _at(1), f, None, None, None, None, None)
                    for i, (a, b, f) in enumerate(zip(endpoints[:-1], endpoints[1:], firsts, strict=True), 1))
    line = ex.StoreTimeline(tuple(endpoints), changes, True, False, False, False, False, False, False, (), {})
    return line, _branch(branch_sec)


def test_the_branch_copy_is_the_last_copy_of_the_chain_when_every_change_precedes_the_branch_phase():
    t = lambda sec: datetime(2026, 10, 23, 9, 0, 0, tzinfo=UTC) + timedelta(seconds=sec)  # noqa: E731
    line, branch = _hand_line([A, B], [t(10)])
    assert ex.branch_copy_endpoint("OUT", line, branch) == B
    line, branch = _hand_line([A, B, A], [t(10), t(12)])  # a copy used in two periods: the last copy is the first one
    assert ex.branch_copy_endpoint("OUT", line, branch) == A
    line, branch = _hand_line([A, B, C], [t(10), t(14.999)])  # a change a millisecond before the branch phase is before it
    assert ex.branch_copy_endpoint("OUT", line, branch) == C


@pytest.mark.parametrize("first", ["at the branch instant", "after it", "none", "text"])
def test_a_change_not_shown_to_precede_the_branch_phase_is_refused_by_number_without_manifest_strings(first):
    t = lambda sec: datetime(2026, 10, 23, 9, 0, 0, tzinfo=UTC) + timedelta(seconds=sec)  # noqa: E731
    bad = {"at the branch instant": t(15), "after it": t(15.001), "none": None, "text": _at(10)}[first]
    line, branch = _hand_line([A, B, C], [t(10), bad])  # the second change is the one in question
    with pytest.raises(ex.Refused, match=r"^OUT: its store change 2 is not shown by the records to precede the branch phase") as caught:
        ex.branch_copy_endpoint("OUT", line, branch)
    message = str(caught.value)
    assert not re.search(r"store change [13]", message) and "2026-10" not in message and _at(1) not in message
    line, branch = _hand_line([A, B], [bad])  # and the first, when it is the one
    with pytest.raises(ex.Refused, match=r"^OUT: its store change 1 is not shown"):
        ex.branch_copy_endpoint("OUT", line, branch)


# --- the report's branch_phase_copies: the parents' copies and the branch copy with their speed classes -------------------

ATTRIBUTION_SENTENCE = ("a branch row records its parent attempt's copy (branch_attempt in harness/checkloop/online.py), "
                        "but its live retry ran on the branch copy, the chain's last copy: the row counts for that copy, "
                        "and a checker arm's retried output whose parent attempt ran on another copy is a unit row naming "
                        "both copies, its retry attempt counted for the branch copy in the attempt shares")  # by hand
COPY_KEYS = {"copy", "class", "label", "basis", "source"}


def _copy(label: str, entry: dict) -> dict:
    return {"copy": label, **entry}


def test_the_report_lists_per_out_the_parent_copies_sorted_by_label_and_the_branch_copy_with_their_classes(world, ref):
    m = _moved_world(world)
    out = m["out"]
    cell, a, b = _two_late_rows(out)  # one late row records copy-a, one copy-b: the two parents' copies
    _set_branch_phase(out, T_B)
    classes = _classes_file(world["tmp"] / "classes.json")
    st = _stores(world, [out], m["proofs"], m["entries"], copy_classes=classes)
    # the late rows' endpoints sort E1 (copy-b) before E0 (copy-a) as strings, so the sorting by label is what puts copy-a
    # first; the branch copy is the chain's last, copy-b
    assert st["branch_phase_copies"] == [{
        "out": str(out), "branch_phase_start": T_B,
        "parent_copies": [_copy("copy-a", CLASS_A), _copy("copy-b", CLASS_B)],
        "branch_copy": _copy("copy-b", CLASS_B), "attribution": ATTRIBUTION_SENTENCE}]
    (entry,) = st["branch_phase_copies"]
    assert set(entry) == {"out", "branch_phase_start", "parent_copies", "branch_copy", "attribution"}
    assert all(set(c) == COPY_KEYS for c in [*entry["parent_copies"], entry["branch_copy"]])
    assert _ints(st["branch_phase_copies"]) == [] and E0 not in json.dumps(st) and E1 not in json.dumps(st)
    assert st["config"]["copy_classes"] == {"path": str(classes), "sha256": _sha(classes)}


def test_one_parent_copy_is_one_entry(world, ref):
    m = _moved_world(world)
    out = m["out"]
    a = _middle_record(out, E0, lambda r: r["question_id"] == "s01-pass" and r["model"] == QWEN)
    assert a is not None
    _retime(out, lambda r, f: LATE_A if r["run_id"] == a["run_id"] else None)  # the only branch row, recording copy-a
    _set_branch_phase(out, T_B)
    (entry,) = _stores(world, [out], m["proofs"], m["entries"])["branch_phase_copies"]
    assert entry["parent_copies"] == [_copy("copy-a", CLASS_A)] and entry["branch_copy"] == _copy("copy-b", CLASS_B)


def test_a_branch_phase_that_no_row_reached_has_no_parent_copies_and_still_names_the_branch_copy(world, ref):
    m = _moved_world(world)
    _set_branch_phase(m["out"], "2026-10-23T09:59:00.000+00:00")  # after every record of the fixture
    (entry,) = _stores(world, [m["out"]], m["proofs"], m["entries"])["branch_phase_copies"]
    assert entry["parent_copies"] == [] and entry["branch_copy"] == _copy("copy-b", CLASS_B)
    assert entry["branch_phase_start"] == "2026-10-23T09:59:00.000+00:00"


def test_a_copy_used_in_two_periods_is_listed_under_its_period_label_with_its_replicas_class(world, ref):
    w = _aba_world(world)
    out = w["out"]
    for cell in CELLS:
        b = _middle_record(out, E1, lambda r: r["question_id"] == "s01-flag" and r["model"] == QWEN
                           and r["load"]["arm"] == cell)
        if b is not None:
            break
    _retime(out, lambda r, f: "2026-10-23T09:51:00.000+00:00" if r["run_id"] == b["run_id"] else None)
    _set_branch_phase(out, "2026-10-23T09:50:00.000+00:00")
    (entry,) = _stores(world, [out], w["proofs"], w["entries"])["branch_phase_copies"]
    assert entry["parent_copies"] == [_copy("copy-b", CLASS_B)]  # the row recorded copy-b, its parent's copy
    assert entry["branch_copy"] == _copy("copy-a (two periods)", CLASS_A)  # the chain's last: copy-a, one copy


def test_the_entries_are_sorted_by_out_whatever_the_order_the_outs_are_given_in(world, ref):
    m = _moved_world(world)
    _two_late_rows(m["out"])
    _set_branch_phase(m["out"], T_B)
    first, second = world["tmp"] / "z-out", world["tmp"] / "a-out"  # the same OUT twice, named so as to sort a before z
    shutil.copytree(m["out"], first)
    shutil.copytree(m["out"], second)
    config = _store_config(world["tmp"] / "two.json", *m["entries"])
    st = ex.check_stores(ex.load_attribution(), [ex.Out.read(first), ex.Out.read(second)], "s01", tuple(m["proofs"]),
                         config, None, _classes_file(world["tmp"] / "classes.json"))
    assert [e["out"] for e in st.branch_phase_copies] == [str(second), str(first)]
    assert st.branch_phase_copies[0]["branch_copy"] == st.branch_phase_copies[1]["branch_copy"] == _copy("copy-b", CLASS_B)


def test_an_out_without_a_branch_phase_or_without_store_changes_has_no_entry_and_needs_no_class(world, ref):
    m = _moved_world(world)  # a store change and no branch phase
    only_a = world["tmp"] / "only-a.json"
    doc = json.loads(_classes_file(only_a).read_text(encoding="utf-8"))
    del doc["copies"]["copy-b"]  # the table lacks copy-b: only an OUT with a branch phase needs it
    only_a.write_text(json.dumps(doc), encoding="utf-8")
    assert _stores(world, [m["out"]], m["proofs"], m["entries"], copy_classes=only_a)["branch_phase_copies"] == []
    _set_branch_phase(world["single"], T_B)  # a branch phase and the store change: now it is needed
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"], r"has no entry in the copy classes", copy_classes=only_a)


def test_a_replica_missing_from_the_copy_classes_refuses_the_call_and_the_check_stores_command(world, ref, s01_graph, capsys):
    m = _moved_world(world)
    out = m["out"]
    _two_late_rows(out)
    _set_branch_phase(out, T_B)
    only_a = world["tmp"] / "only-a.json"
    doc = json.loads(_classes_file(only_a).read_text(encoding="utf-8"))
    del doc["copies"]["copy-b"]
    only_a.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ex.Refused, match=rf"{re.escape(str(out))}: the replica copy-b of its copy copy-b has no entry in "
                                         r"the copy classes"):
        _check(world, out, m["proofs"], m["entries"], only_a)
    config = _store_config(world["tmp"] / "check-cli.json", *m["entries"])  # the command: a refusal (2), not an error (3)
    code = ex.check_stores_main(ROOT, "s01", [out], tuple(m["proofs"]), config, only_a)
    obj = json.loads(capsys.readouterr().out)
    assert code == 2 and obj["ok"] is False and "has no entry in the copy classes" in obj["refused"]


def test_a_copy_known_only_by_its_endpoint_has_no_replica_behind_it_and_refuses(world, ref):
    out = world["single"]  # each model on one endpoint (so no entry is needed and the labels stay endpoint labels)
    before = _set_endpoints(out, lambda r: E0 if r["model"] == QWEN else E1)
    _store_change(out, E0, E1, before)
    a = _middle_record(out, E0, lambda r: r["model"] == QWEN)
    assert a is not None
    _retime(out, lambda r, f: LATE_A if r["run_id"] == a["run_id"] else None)
    _set_branch_phase(out, T_B)
    with pytest.raises(ex.Refused, match=rf"{re.escape(str(out))}: its copy endpoint sha256:[0-9a-f]{{12}} has no replica "
                                         r"behind it, so it has no speed class"):
        _check(world, out, (), [])


def test_a_malformed_copy_classes_file_refuses_the_call_with_ex_refused(world, ref):
    m = _moved_world(world)
    good = json.loads(_classes_file(world["tmp"] / "good.json").read_text(encoding="utf-8"))
    bad = {"not json": "{", "another schema": json.dumps({**good, "schema": "e189-copy-classes/2"}),
           "an entry without its basis": json.dumps({**good, "copies": {"copy-a": {k: v for k, v in CLASS_A.items()
                                                                                 if k != "basis"}}}),
           "a class that is not one": json.dumps({**good, "copies": {"copy-a": {**CLASS_A, "class": "partial"}}})}
    for case, text in bad.items():
        path = world["tmp"] / "bad.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ex.Refused, match=re.escape(str(path))):
            _check(world, m["out"], m["proofs"], m["entries"], path)  # read before anything else: with no branch phase too
        with pytest.raises(ex.Refused, match=re.escape(str(path))):
            ex.read_copy_classes(path)


def test_the_copy_classes_are_read_as_replica_entries_with_the_sha256_of_the_file(world):
    path = _classes_file(world["tmp"] / "classes.json")
    copies, sha = ex.read_copy_classes(path)
    assert copies == {"copy-a": CLASS_A, "copy-b": CLASS_B, "copy-c": CLASS_C} and sha == _sha(path)
    committed, committed_sha = ex.read_copy_classes(HERE.parent / "copy-classes.json")  # the committed one
    assert committed["rhea-c06"]["label"] == "fast" and committed_sha == _sha(HERE.parent / "copy-classes.json")


@pytest.mark.parametrize("label, replicas, expected", [
    ("copy-a", ["copy-a"], _copy("copy-a", CLASS_A)),
    ("copy-a (two periods)", ["copy-a"], _copy("copy-a (two periods)", CLASS_A)),  # one copy: its replica's class
    ("copy-a then copy-b (one endpoint, unsplittable)", ["copy-a", "copy-b"], {
        "copy": "copy-a then copy-b (one endpoint, unsplittable)", "class": "fast then slow",
        "label": "fast then slow (host rule, unmeasured)", "basis": "basis of copy-a then basis of copy-b",
        "source": "src then src"}),
])
def test_a_copy_object_is_the_replicas_entry_or_the_replicas_entries_joined_in_order(label, replicas, expected):
    classes = {"copy-a": CLASS_A, "copy-b": CLASS_B}
    assert ex.copy_object("OUT", label, replicas, classes) == expected


def test_a_copy_object_refuses_a_label_without_a_replica_and_a_replica_without_a_class_entry():
    classes = {"copy-a": CLASS_A, "copy-null": {"class": None, "label": "unmeasured (no host rule for corporate)",
                                                 "basis": "b", "source": "src"}}
    with pytest.raises(ex.Refused, match=r"^OUT: its copy endpoint sha256:abc has no replica behind it"):
        ex.copy_object("OUT", "endpoint sha256:abc", [], classes)
    with pytest.raises(ex.Refused, match=r"^OUT: the replica copy-z of its copy copy-z has no entry in the copy classes"):
        ex.copy_object("OUT", "copy-z", ["copy-z"], classes)
    with pytest.raises(ex.Refused, match=r"the replica copy-z of its copy copy-a then copy-z \(one endpoint, unsplittable\)"):
        ex.copy_object("OUT", "copy-a then copy-z (one endpoint, unsplittable)", ["copy-a", "copy-z"], classes)
    assert ex.copy_object("OUT", "copy-null", ["copy-null"], classes)["class"] is None  # a null class stays null
    assert ex.copy_object("OUT", "copy-a then copy-null (one endpoint, unsplittable)", ["copy-a", "copy-null"],
                          classes)["class"] == "fast then null"


# --- a retried output whose parent and retry ran on different copies (science, 25 Oct ~06:2xZ, addendum 1) ---------------
#
# Hand-derived on the fixture moved from copy-a (E0) to copy-b (E1): s01-pass and s01-flag are on copy-a, s01-abstain and
# s01-noquery on copy-b. Qwen's beta checker arms hold four units each, and the one retried output of each is s01-flag
# (its parent on copy-a, its retry record on copy-a too); the units are pass (copy-a), flag, abstain and noquery (copy-b).
# Retimed into the branch phase, the curated arm's retry record keeps recording copy-a (its parent's), but its live retry
# ran on the chain's last copy, copy-b: flag becomes "copy-a → copy-b", and the arm's five attempts are copy-a 2 (pass and
# the flag output) and copy-b 3 (the retry, abstain, noquery). The generic arm's retry stays in the first period, on its
# parent's copy: flag is copy-a alone, and its five attempts are copy-a 3, copy-b 2.

CURATED, GENERIC = f"s01|{QWEN}|beta|checker|curated", f"s01|{QWEN}|beta|checker|generic"
ARROW_AB = "copy-a → copy-b"
SAME, MIXED_CLASS = "mixed copies, one class", "MIXED CLASS"
CLASS_B_FAST = {"class": "fast", "label": "fast", "basis": "basis of copy-b", "source": "src"}
CLASS_NULL = {"class": None, "label": "unmeasured (no host rule for corporate)", "basis": "no host rule", "source": "src"}


def _retried_branch_world(world) -> dict:
    """The moved fixture with the curated arm's one retry record (Qwen, s01-flag) made in the branch phase."""
    m = _moved_world(world)
    out = m["out"]
    retry = _record_of(out, "s01-flag", QWEN, "fb:curated")[1]
    assert retry["store"]["endpoint"] == E0  # it records its parent's copy
    _retime(out, lambda r, f: LATE_A if r["run_id"] == retry["run_id"] else None)
    _set_branch_phase(out, T_B)
    return m


def _classes_with(world, name: str, **copies: dict) -> Path:
    return _classes_file(world["tmp"] / name, **copies)


def _checker_rows(st: dict, key: str, arm: str = CURATED) -> dict:
    return {label: row["n"] for label, row in st[key][arm]["rows"].items()}


def test_a_retried_output_with_its_retry_on_the_branch_copy_names_both_copies_and_one_class_is_not_a_class_mix(world, ref):
    m = _retried_branch_world(world)
    classes = _classes_with(world, "same.json", **{"copy-b": CLASS_B_FAST})  # copy-a and copy-b are both fast
    st = _stores(world, [m["out"]], m["proofs"], m["entries"], copy_classes=classes)
    out = str(m["out"])
    assert st["shares"][CURATED] == _rows({"copy-a": 1, ARROW_AB: 1, "copy-b": 2})
    assert st["class_shares"][CURATED] == _rows({"fast": 3, SAME: 1})
    assert st["attempt_shares"][CURATED] == _rows({"copy-a": 2, "copy-b": 3})
    # the per-OUT sections are the call's (one OUT)
    assert st["shares_by_out"][out][CURATED] == st["shares"][CURATED]
    assert st["class_shares_by_out"][out][CURATED] == st["class_shares"][CURATED]
    assert st["attempt_shares_by_out"][out][CURATED] == st["attempt_shares"][CURATED]
    # a retry on its parent's own copy is not listed: the generic arm's retried output is the copy alone
    assert st["shares"][GENERIC] == _rows({"copy-a": 2, "copy-b": 2})
    assert st["class_shares"][GENERIC] == _rows({"fast": 4})
    assert st["attempt_shares"][GENERIC] == _rows({"copy-a": 3, "copy-b": 2})
    # a prompt arm: one attempt per unit, no unit row naming two copies
    prompt = f"s01|{QWEN}|beta|prompt|-"
    assert st["shares"][prompt] == st["attempt_shares"][prompt] == _rows({"copy-a": 2, "copy-b": 2})
    # the change's split keeps its three buckets, by the unit's own attempt (the parent): the retried output is before
    (change,) = st["store_changes"]
    assert change["shares_by_arm"][CURATED] == {"before": _rows({"copy-a": 1, ARROW_AB: 1}),
                                               "after": _rows({"copy-b": 2}), "branch_phase": _rows({})}


def test_a_retried_output_whose_copies_have_different_classes_is_a_mixed_class(world, ref):
    m = _retried_branch_world(world)
    st = _stores(world, [m["out"]], m["proofs"], m["entries"])  # the fixtures' classes: copy-a fast, copy-b slow
    assert st["shares"][CURATED] == _rows({"copy-a": 1, ARROW_AB: 1, "copy-b": 2})
    assert st["class_shares"][CURATED] == _rows({"fast": 1, MIXED_CLASS: 1, "slow (host rule, unmeasured)": 2})
    assert st["attempt_shares"][CURATED] == _rows({"copy-a": 2, "copy-b": 3})
    assert st["class_shares"][GENERIC] == _rows({"fast": 2, "slow (host rule, unmeasured)": 2})  # the single-copy labels


@pytest.mark.parametrize("copies, expected", [
    ({"copy-b": CLASS_NULL}, {"fast": 1, MIXED_CLASS: 1, CLASS_NULL["label"]: 2}),  # one copy with no class
    ({"copy-a": CLASS_NULL, "copy-b": CLASS_NULL}, {MIXED_CLASS: 1, CLASS_NULL["label"]: 3}),  # both: unknown is not one class
])
def test_a_copy_with_a_null_class_makes_its_retried_output_a_mixed_class(world, ref, copies, expected):
    m = _retried_branch_world(world)
    st = _stores(world, [m["out"]], m["proofs"], m["entries"], copy_classes=_classes_with(world, "null.json", **copies))
    assert st["class_shares"][CURATED] == _rows(expected)
    assert st["shares"][CURATED] == _rows({"copy-a": 1, ARROW_AB: 1, "copy-b": 2})  # the rows do not depend on classes


def test_a_two_copy_unit_with_a_copy_missing_from_the_table_refuses_the_call(world, ref):
    m = _retried_branch_world(world)
    only_a = _classes_with(world, "only-a.json")
    doc = json.loads(only_a.read_text(encoding="utf-8"))
    del doc["copies"]["copy-b"]
    only_a.write_text(json.dumps(doc), encoding="utf-8")
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"], r"has no entry in the copy classes", copy_classes=only_a)


def test_a_single_endpoint_models_units_carry_the_no_copy_class_label_and_are_no_class_mix(world, ref):
    st = _stores(world, [world["single"]], (), ())  # no store change, no proof entry, no replica behind any label
    label = "no copy class (no entry in the copy classes)"
    for arm in (CURATED, GENERIC):
        assert st["class_shares"][arm] == _rows({label: 4})
    assert "MIXED CLASS" not in json.dumps(st["class_shares"]) and "MIXED CLASS" not in json.dumps(st["class_shares_by_out"])


def test_a_retry_on_the_parents_own_copy_is_that_copy_and_its_class_label(world, ref):
    m = _moved_world(world)  # no branch phase: the curated retry record stays on copy-a, the output's copy
    st = _stores(world, [m["out"]], m["proofs"], m["entries"])
    assert st["shares"][CURATED] == _rows({"copy-a": 2, "copy-b": 2})
    assert st["class_shares"][CURATED] == _rows({"fast": 2, "slow (host rule, unmeasured)": 2})
    assert st["attempt_shares"][CURATED] == _rows({"copy-a": 3, "copy-b": 2})


def test_a_retry_attributed_to_the_chains_last_copy_of_a_b_a_names_the_copy_in_two_periods(world, ref):
    w = _aba_world(world)  # pass on copy-a, flag on copy-b, abstain and noquery on copy-a again
    out = w["out"]
    retry = _record_of(out, "s01-flag", QWEN, "fb:curated")[1]
    assert retry["store"]["endpoint"] == E1  # the parent's copy, copy-b
    _retime(out, lambda r, f: "2026-10-23T09:51:00.000+00:00" if r["run_id"] == retry["run_id"] else None)
    _set_branch_phase(out, "2026-10-23T09:50:00.000+00:00")
    st = _stores(world, [out], w["proofs"], w["entries"])
    arrow = f"copy-b → {A_TWO}"  # the retry's copy is the chain's last: copy-a, whose label names its two periods
    assert st["shares"][CURATED] == _rows({A_TWO: 3, arrow: 1})
    assert st["class_shares"][CURATED] == _rows({"fast": 3, MIXED_CLASS: 1})  # copy-b slow, copy-a fast
    assert st["attempt_shares"][CURATED] == _rows({A_TWO: 4, "copy-b": 1})  # pass, abstain, noquery, the retry; the output


def test_several_retries_on_several_copies_list_the_retry_copies_sorted_and_not_the_parents_own():
    from types import SimpleNamespace as NS

    at = ex.load_attribution()
    key, unit = NS(use="checker", source="curated"), NS(attempt=NS(run_id="r1"), checked=NS(retried=True))
    table = {"A": ("fast", "fast"), "B": ("fast", "fast"), "C": ("slow", "slow (host rule, unmeasured)"),
             "N": (None, "unmeasured (no host rule for corporate)")}

    def cls(label: str) -> tuple[str | None, str]:
        return table.get(label, (None, ex.NO_COPY_CLASS))  # a label with no table entry: as label_class gives it

    def dep(parent: str, retries: list[str]) -> ex.Deployments:
        served = {"r1": parent, **{f"x{i}": c for i, c in enumerate(retries)}}
        return ex.Deployments(served, {"r1": {"fb:curated": list(served)[1:]}} if retries else {},
                              {"curated": {"fb:curated"}}, Counter(), 0, set())

    def store(parent: str, retries: list[str]) -> ex.StoreUnit:
        return ex.store_unit_row(unit, key, dep(parent, retries), "prompt", cls)

    unit_ = store("A", ["C", "B", "A"])  # retries on C, B and the parent's own copy A
    assert unit_ == ex.StoreUnit("A → B, C", "MIXED CLASS", ("A", "C", "B", "A"))
    assert store("A", ["B", "B"]) == ex.StoreUnit("A → B", "mixed copies, one class", ("A", "B", "B"))
    assert store("A", ["A", "A"]) == ex.StoreUnit("A", "fast", ("A", "A", "A"))  # all on the parent's copy: that copy
    assert store("C", ["C"]) == ex.StoreUnit("C", "slow (host rule, unmeasured)", ("C", "C"))
    assert store("N", ["N"]) == ex.StoreUnit("N", "unmeasured (no host rule for corporate)", ("N", "N"))
    assert store("A", ["N"]).class_label == "MIXED CLASS" and store("N", ["A"]).class_label == "MIXED CLASS"
    assert store("A", ["B", "C", "N"]).row == "A → B, C, N"
    # a copy with no table entry (analysis-lead-3, 25 Oct): on a unit of ONE copy the explicit label, and no class mix; on a
    # unit of SEVERAL copies a refusal naming the copy alone, whichever copy lacks the entry
    assert store("Z", ["Z"]) == ex.StoreUnit("Z", "no copy class (no entry in the copy classes)", ("Z", "Z"))
    assert store("Z", ["Z"]).class_label != "MIXED CLASS" and store("Z", ["Z", "Z"]).row == "Z"
    for parent, retries in (("A", ["Z"]), ("Z", ["A"]), ("A", ["B", "Z"]), ("N", ["Z"]), ("Z", ["Y"])):
        with pytest.raises(ex.Refused, match=r"^the copy Z of a retried output on several copies has no entry in the copy "
                                              r"classes$"):
            store(parent, retries)
    assert store("N", ["A"]).class_label == "MIXED CLASS"  # a null class in the table is a class mix, as specified
    # no retry record: as in the deployment section, the row and the class label say so; the output's attempt still counts
    assert store("A", []) == ex.StoreUnit(ex.RETRY_WITHOUT_RECORD, ex.RETRY_WITHOUT_RECORD, ("A",))
    # not retried (a prompt arm, or a plain run's checker record), or no record at all
    prompt = ex.store_unit_row(unit, NS(use="prompt", source=None), dep("A", ["B"]), "prompt", cls)
    assert prompt == ex.StoreUnit("A", "fast", ("A",))
    plain = ex.Deployments({"r1": "A", "x0": "B"}, {"r1": {"fb:curated": ["x0"]}}, {"curated": {"fb:curated"}}, Counter(), 0,
                           {"r1"})
    assert ex.store_unit_row(unit, key, plain, "prompt", cls) == ex.StoreUnit("A", "fast", ("A",))
    none = ex.Deployments({}, {}, {"curated": {"fb:curated"}}, Counter(), 0, set())
    assert ex.store_unit_row(unit, key, none, "prompt", cls) == ex.StoreUnit(None, None, (None,))
    # the deployment section's rule is untouched: the same unit with a retry elsewhere is 'mixed', and so is the plain
    # reading of the same two copies
    assert ex.unit_row(at, unit, key, dep("A", ["B"]), "prompt") == at.MIXED
    assert ex.unit_row(at, unit, key, dep("A", ["C", "B", "A"]), "prompt") == at.MIXED
    assert ex.unit_row(at, unit, key, dep("A", ["A"]), "prompt") == "A"
    assert ex.unit_row(at, unit, key, dep("A", []), "prompt") == ex.RETRY_WITHOUT_RECORD


def test_the_deployment_section_still_says_mixed_for_the_unit_the_store_section_names_by_both_copies(world, ref):
    m = _retried_branch_world(world)
    out = m["out"]
    retry = _record_of(out, "s01-flag", QWEN, "fb:curated")[1]["run_id"]
    _serve(out, lambda r: FP_B if r["run_id"] == retry else FP_A)  # the retry was served by another deployment too
    config = _store_config(world["tmp"] / "proofs.json", *m["entries"])
    result = _run(world, [out], record=False, ledgers=(world["ledger"],), store_proofs=tuple(m["proofs"]),
                  store_config=config, copy_classes=_classes_file(world["tmp"] / "classes.json"),
                  admissions=world["admit_b"])
    curated = result["deployments"]["shares"][CURATED]["rows"]
    assert curated["mixed"]["n"] == 1 and sorted(r["n"] for r in curated.values()) == [1, 3]  # unchanged: 'mixed'
    assert result["stores"]["shares"][CURATED] == _rows({"copy-a": 1, ARROW_AB: 1, "copy-b": 2})
    assert "mixed" not in json.dumps(result["stores"]["shares"])  # the store section never says 'mixed' for a unit row


def test_a_retry_in_another_out_than_its_parent_is_refused_before_the_store_section_needs_them_in_one_out(world, ref):
    """store_section reads a retry's class through its OUT's replicas, which holds because the frozen input loading refuses a
    retry whose parent is not in its own batch: the premise, pinned on the moved fixture with the curated retry alone in a
    second OUT."""
    m = _moved_world(world)
    out = m["out"]
    f, _ = _record_of(out, "s01-flag", QWEN, "fb:curated")
    other = world["tmp"] / "other"
    shutil.copytree(out, other)
    for g in (other / "records").glob("*.json"):
        if g.name != f.name:
            g.unlink()
    f.unlink()
    _recheck(out)
    _recheck(other)
    _refused(world, [out, other], r"its parent .* is not a record of its cell's parent 'beta' in the batch", record=False,
             ledgers=(world["ledger"],))


@pytest.mark.parametrize("run_id", [None, 7, ["x"], {"a": 1}])
def test_a_records_row_made_in_the_branch_phase_without_a_string_run_id_is_refused_not_labelled_by_its_recorded_copy(
        world, ref, s01_graph, capsys, run_id):
    """External review of T4 (r1): the timeline places a row by its run id, so a branch-phase row whose run id is no string has
    no period, the timeline can still be clean, and the row would silently keep the copy it records (its parent's)."""
    m = _moved_world(world)
    out = m["out"]
    cell, a, b = _two_late_rows(out)  # a records copy-a and b copy-b, both retimed into the branch phase
    _set_branch_phase(out, T_B)
    _check(world, out, m["proofs"], m["entries"])  # as it stands, accepted
    f, record = next((f, r) for f in sorted((out / "records").glob("*.json"))
                     if (r := json.loads(f.read_text(encoding="utf-8")))["run_id"] == a["run_id"])
    f.write_text(json.dumps({**record, "run_id": run_id}), encoding="utf-8")
    _recheck(out)
    where = re.escape(str(out))
    with pytest.raises(ex.Refused, match=rf"^{where}: records/ rows made in the branch phase have no run id that is a "
                                          r"string, so their copy cannot be attributed") as caught:
        _check(world, out, m["proofs"], m["entries"])
    assert f"records/{f.name}" in str(caught.value) and str(b["run_id"]) not in str(caught.value)
    config = _store_config(world["tmp"] / "check-cli.json", *m["entries"])  # the command refuses it too: exit 2
    assert ex.check_stores_main(ROOT, "s01", [out], tuple(m["proofs"]), config, _classes_file(world["tmp"] / "c.json")) == 2
    obj = json.loads(capsys.readouterr().out)
    assert obj["ok"] is False and "no run id that is a string" in obj["refused"]


def test_a_row_without_a_string_run_id_outside_the_branch_phase_is_not_this_checks_business(world, ref):
    single = world["tmp"] / "single-plain"  # an OUT with a branch phase and no store changes: copied before the move
    shutil.copytree(world["single"], single)
    m = _moved_world(world)
    out = m["out"]
    _two_late_rows(out)
    _set_branch_phase(out, T_B)
    f, record = next((f, r) for f in sorted((out / "records").glob("*.json"))  # a row before the branch phase
                     if (r := json.loads(f.read_text(encoding="utf-8")))["started_at"] < T_B)
    f.write_text(json.dumps({**record, "run_id": None}), encoding="utf-8")
    _recheck(out)
    assert _check(world, out, m["proofs"], m["entries"]).timelines[str(out)].clean
    # and the OUT with a branch phase and no store changes has no attribution to make
    _set_branch_phase(single, "2026-10-19T00:00:00.000+00:00")  # before every record: all are branch rows
    g, other = next((g, r) for g in sorted((single / "records").glob("*.json"))
                    if (r := json.loads(g.read_text(encoding="utf-8"))))
    g.write_text(json.dumps({**other, "run_id": None}), encoding="utf-8")
    _recheck(single)
    assert _check(world, single, (), []).run_row


@pytest.mark.parametrize("bad", [[], ["x"], {}, {"a": 1}, 3, None, True, "", " "])
def test_a_copy_class_source_of_any_wrong_type_refuses_through_check_stores_with_exit_2_not_3(world, ref, s01_graph, capsys, bad):
    """External review of T4 (r1): "source": [] raised TypeError in the loader, so the command exited 3, not 2."""
    m = _moved_world(world)
    doc = json.loads(_classes_file(world["tmp"] / "good.json").read_text(encoding="utf-8"))
    doc["copies"]["copy-a"]["source"] = bad
    path = world["tmp"] / "bad-source.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ex.Refused, match=re.escape(str(path))):
        ex.read_copy_classes(path)
    config = _store_config(world["tmp"] / "check-cli.json", *m["entries"])
    assert ex.check_stores_main(ROOT, "s01", [m["out"]], tuple(m["proofs"]), config, path) == 2
    obj = json.loads(capsys.readouterr().out)
    assert obj["ok"] is False and str(path) in obj["refused"] and "error" not in obj


@pytest.mark.parametrize("replicas, expected", [
    (["copy-a"], ("fast", "fast")),
    (["copy-b"], ("slow", "slow (host rule, unmeasured)")),
    (["copy-a", "copy-b"], (None, "fast then slow (host rule, unmeasured)")),  # one endpoint behind two copies of two classes
    (["copy-a", "copy-a"], ("fast", "fast then fast")),
    (["copy-null"], (None, "unmeasured (no host rule for corporate)")),
    (["copy-a", "copy-null"], (None, "fast then unmeasured (no host rule for corporate)")),
    ([], (None, "no copy class (no entry in the copy classes)")),  # an endpoint label: no replica behind it
    (["copy-z"], (None, "no copy class (no entry in the copy classes)")),  # a replica the table lacks
])
def test_a_copy_labels_class_and_class_label_are_read_from_the_replicas_behind_it(replicas, expected):
    classes = {"copy-a": CLASS_A, "copy-b": CLASS_B, "copy-null": CLASS_NULL}
    assert ex.label_class(replicas, classes) == expected


def test_a_malformed_branch_phase_refuses_the_call_and_the_command_says_stop_without_the_manifest_string(world, ref, capsys):
    m = _moved_world(world)
    out = m["out"]
    cases = {
        "two entries": lambda man: man["additions"].extend([{"added": ["phase branch"], "at": T_B},
                                                            {"added": ["phase branch"], "at": RUN_LIKE}]),
        "at unparseable": lambda man: man["additions"].append({"added": ["phase branch"], "at": RUN_LIKE}),
        "additions not a list": lambda man: man.update(additions=RUN_LIKE),
        "added not a list": lambda man: man["additions"].append({"added": RUN_LIKE, "at": RUN_LIKE}),
    }
    for case, edit in cases.items():
        _edit_manifest(out, lambda man: man.update(additions=[]))
        _edit_manifest(out, edit)
        with pytest.raises(ex.Refused, match=rf"{re.escape(str(out))}: its manifest's additions") as caught:
            _check(world, out, m["proofs"], m["entries"])
        assert RUN_LIKE not in str(caught.value) and "interleaved" not in str(caught.value), case
        code, (obj,), raw, err = _cli(capsys, out)
        assert code == 2, case  # a STOP, not an unreadable OUT
        assert obj == {"out": str(out), "manifest_sha256": _sha(out / "manifest.json"), "branch_phase_malformed": True,
                       "verdict": "STOP: branch_phase_malformed"}, case
        assert RUN_LIKE not in raw + err and ex.STOP_LINES["branch_phase_malformed"] in err, case
        assert "STOP interleaved" not in err and "manifest_store_changes_malformed" not in err
    _edit_manifest(out, lambda man: man.update(additions=[]))  # no entry at all is no branch phase, not a malformed one
    assert _check(world, out, m["proofs"], m["entries"]).timelines[str(out)].branch_phase_start is None


def test_a_malformed_branch_phase_and_malformed_store_changes_are_both_named_by_the_command(world, capsys):
    m = _moved_world(world)
    out = m["out"]
    _edit_manifest(out, lambda man: (man.update(store_changes="x"), man["additions"].append({"added": ["phase branch"],
                                                                                              "at": RUN_LIKE})))
    code, (obj,), raw, err = _cli(capsys, out)
    assert code == 2 and obj == {
        "out": str(out), "manifest_sha256": _sha(out / "manifest.json"), "manifest_store_changes_malformed": True,
        "branch_phase_malformed": True, "verdict": "STOP: manifest_store_changes_malformed, branch_phase_malformed"}
    assert ex.STOP_LINES["manifest_store_changes_malformed"] in err and ex.STOP_LINES["branch_phase_malformed"] in err
    assert RUN_LIKE not in raw + err


def _with_and_without_store_changes(world) -> tuple[dict[str, Path], dict]:
    """The OUTs by kind: a copy of the fixture whole as it stands (one endpoint, no store changes) and the OUT moved from
    copy-a to copy-b by one store change; and the moved world (its proofs and entries)."""
    plain = world["tmp"] / "single-plain"
    shutil.copytree(world["single"], plain)  # before the move edits the fixture
    m = _moved_world(world)
    return {"without": plain, "with": m["out"]}, m


@pytest.mark.parametrize("why", list(WRONG_ADDITIONS))
def test_a_wrong_typed_additions_refuses_the_call_for_an_out_with_or_without_store_changes(world, ref, why):
    outs, m = _with_and_without_store_changes(world)
    for kind, files, entries in (("without", (), []), ("with", m["proofs"], m["entries"])):
        out = outs[kind]
        _edit_manifest(out, lambda man: man.update(additions=WRONG_ADDITIONS[why]))
        with pytest.raises(ex.Refused, match=rf"{re.escape(str(out))}: its manifest's additions") as caught:
            _check(world, out, files, entries)
        assert RUN_LIKE not in str(caught.value), (kind, why)
        # a malformed additions says its own problem alone, not a timeline verdict made without the instant
        assert "interleaved" not in str(caught.value) and "chain_incomplete" not in str(caught.value), (kind, why)


@pytest.mark.parametrize("why", list(WRONG_ADDITIONS))
def test_a_wrong_typed_additions_is_stop_branch_phase_malformed_in_the_command_for_every_out(world, capsys, why):
    outs, _m = _with_and_without_store_changes(world)
    for out in outs.values():
        _edit_manifest(out, lambda man: man.update(additions=WRONG_ADDITIONS[why]))
    code, objs, raw, err = _cli(capsys, outs["without"], outs["with"])
    assert code == 2, why  # a STOP, not an unreadable OUT (3), and never a clean exit (0)
    assert [o["out"] for o in objs] == [str(outs["without"]), str(outs["with"])]
    for kind, obj in zip(("without", "with"), objs, strict=True):
        assert obj == {"out": str(outs[kind]), "manifest_sha256": _sha(outs[kind] / "manifest.json"),
                       "branch_phase_malformed": True, "verdict": "STOP: branch_phase_malformed"}, (kind, why)
    assert RUN_LIKE not in raw + err and err == ex.STOP_LINES["branch_phase_malformed"] + "\n"


@pytest.mark.parametrize("why", ["null", "a string", "added holding a number"])
def test_a_wrong_typed_additions_is_a_refusal_of_the_check_stores_command_too(world, ref, s01_graph, capsys, why):
    outs, m = _with_and_without_store_changes(world)
    for kind, files, entries in (("without", (), []), ("with", m["proofs"], m["entries"])):
        out = outs[kind]
        _edit_manifest(out, lambda man: man.update(additions=WRONG_ADDITIONS[why]))
        code, obj, raw, err = _check_cli(capsys, world, [out], files, entries)
        assert code == 2 and obj["ok"] is False, (kind, why)  # refused (2), not a clean verdict and not an error (3)
        assert "its manifest's additions" in obj["refused"] and RUN_LIKE not in raw + err, (kind, why)


def test_only_a_missing_additions_key_is_no_branch_phase_for_every_out(world, ref, capsys):
    outs, _m = _with_and_without_store_changes(world)
    for out in outs.values():
        _edit_manifest(out, lambda man: man.pop("additions"))
    code, (plain, moved), _raw, err = _cli(capsys, outs["without"], outs["with"])
    assert (code, err) == (0, "")
    assert plain == {"out": str(outs["without"]), "store_changes": False, "several_endpoints": False,
                     "branch_phase_start": None}
    assert moved["verdict"] == "CLEAN" and moved["branch_phase_start"] is None and moved["branch_row_endpoints"] is None
    assert _check(world, outs["without"], (), []).timelines == {}


def test_an_out_without_store_changes_carries_its_branch_phase_instant_verbatim(world, ref, capsys):
    out = world["single"]
    code, (obj,), _raw, err = _cli(capsys, out)  # the fixture's own manifest enters the branch phase at a fixed instant
    assert (code, err, obj) == (0, "", {"out": str(out), "store_changes": False, "several_endpoints": False,
                                        "branch_phase_start": FIXTURE_PHASE_BRANCH_AT})
    _set_branch_phase(out, "2026-10-23T09:00:07Z")  # another spelling is kept as the manifest wrote it
    code, (obj,), _raw, err = _cli(capsys, out)
    assert (code, err, obj["branch_phase_start"]) == (0, "", "2026-10-23T09:00:07Z")
    _set_branch_phase(out, None)  # a list without the entry: no branch phase, null
    code, (obj,), _raw, err = _cli(capsys, out)
    assert (code, err, obj["branch_phase_start"]) == (0, "", None)


def test_a_return_chain_reports_the_branch_phase_bucket_in_both_changes_and_the_branch_rows_are_in_neither_period(world, ref):
    w = _aba_world(world)  # copy-a, then copy-b, then copy-a again; by hand: the arm has one record per question
    out = w["out"]
    a = _middle_record(out, E0, lambda r: r["question_id"] == "s01-pass" and r["model"] == QWEN and r["load"]["arm"] == "none")
    b = _middle_record(out, E1, lambda r: r["question_id"] == "s01-flag" and r["model"] == QWEN and r["load"]["arm"] == "none")
    assert a is not None and b is not None  # neither is the first row of its copy, so every change_at stays what it was
    times = {a["run_id"]: "2026-10-23T09:50:10.000+00:00", b["run_id"]: "2026-10-23T09:50:20.000+00:00"}
    _retime(out, lambda r, f: times.get(r["run_id"]))
    _set_branch_phase(out, "2026-10-23T09:50:00.000+00:00")  # after the third run's rows, on their parents' copies
    line = _check(world, out, w["proofs"], w["entries"]).timelines[str(out)]
    assert line.clean and line.period_of[a["run_id"]] is None and line.period_of[b["run_id"]] is None
    assert [c.first_to_started_text for c in line.changes] == ["2026-10-23T09:18:45.000+00:00",
                                                                "2026-10-23T09:30:00.000+00:00"]
    st = _stores(world, [out], w["proofs"], w["entries"])
    one, two = st["store_changes"]
    assert one["branch_phase_start"] == two["branch_phase_start"] == "2026-10-23T09:50:00.000+00:00"
    # the natural arm's four attempts: pass (recording copy-a) and flag (recording copy-b) are branch rows now, abstain and
    # noquery are in the third run, so neither change has an attempt before it and the branch rows are in neither side;
    # (science, 25 Oct ~04:55Z) both branch rows ran their live retry on the chain's last copy, copy-a: its two periods are
    # one copy, so the branch bucket is copy-a 2 (it was copy-a 1 and copy-b 1) and the arm's shares copy-a 4 (3 and 1)
    for change in (one, two):
        assert change["shares_by_arm"][ARM] == {"before": _rows({}), "after": _rows({A_TWO: 2}),
                                                "branch_phase": _rows({A_TWO: 2})}
    assert st["shares"][ARM] == _rows({A_TWO: 4})


def _twin_run_id(out: Path) -> str:
    """The same run id in records/ a second time, on copy-b later: a record of copy-a's run reappearing in copy-b's."""
    _file, record_a = _record_of(out, "s01-pass")  # on copy-a, in the first run
    twin = {**record_a, "store": {**record_a["store"], "endpoint": E1}, "started_at": "2026-10-23T09:30:00.000+00:00",
            "ended_at": "2026-10-23T09:30:00.500+00:00"}
    (out / "records" / f"{'e' * 40}.{record_a['run_id']}.json").write_text(json.dumps(twin), encoding="utf-8")
    _recheck(out)
    return record_a["run_id"]


def test_one_run_id_in_two_runs_is_refused_naming_the_out(world, ref):
    m = _moved_world(world)
    run_id = _twin_run_id(m["out"])
    with pytest.raises(ex.Refused, match=rf"{re.escape(str(m['out']))}: its records' own endpoints and times do not "
                                         rf"follow its manifest store changes \(run_id_in_two_runs\)") as caught:
        _check(world, m["out"], m["proofs"], m["entries"])
    assert run_id not in str(caught.value)


def test_a_record_without_a_usable_time_or_endpoint_is_refused_by_its_name(world, ref):
    m = _moved_world(world)
    out = m["out"]
    f, record = _record_of(out, "s01-noquery")
    del record["ended_at"]
    f.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    _recheck(out)
    name = re.escape(f"records/{f.name}")
    _refused_check(world, out, m["proofs"], m["entries"], rf"\(unusable_times\).*\['{name}'\]")
    record["ended_at"], record["started_at"] = "2026-10-23T09:18:44.000+00:00", "2026-10-23T09:18:45.000+00:00"
    f.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")  # ended before it started
    _refused_check(world, out, m["proofs"], m["entries"], rf"\(unusable_times\).*\['{name}'\]")
    record["started_at"], record["ended_at"] = "2026-10-23T09:18:45.000", "2026-10-23T09:18:46.000"  # no zone
    f.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    _refused_check(world, out, m["proofs"], m["entries"], rf"\(unusable_times\).*\['{name}'\]")
    record["started_at"], record["ended_at"] = "2026-10-23T09:18:45.000+00:00", "2026-10-23T09:18:46.000+00:00"
    record["store"]["endpoint"] = None  # a record naming no endpoint
    f.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    _refused_check(world, out, m["proofs"], m["entries"], rf"\(rows_without_endpoint\).*\['{name}'\]")
    with pytest.raises(ex.Refused) as caught:
        _check(world, out, m["proofs"], m["entries"])
    assert str(caught.value).count("records/") == 1 and "records name" not in str(caught.value)  # one name, no count


# --- the first record is the change point (science, 25 Oct, Gemma e190 K=3) ----------------------------------------------
#
# Hand-derived facts, never the script's output: in _moved_world copy-b's first record starts 09:18:45.000 (DST_FROM), its
# checks were made at 09:00:00 and 09:00:05, and the manifest says 09:18:39.962. The ordinary entry of a period a change
# began may record another instant than the first record; the change point is then the first record, the entry's proof
# checks are held to it, and the two instants and their gap (first record minus recorded, in seconds) are disclosed.
# A second_period entry keeps its exact equality (it is paired with its move by sorting on the instants).

FIRST_RECORD = "2026-10-23T09:18:45.000+00:00"
RULE_START = "science, 25 Oct (Gemma e190 K=3): a store change's change point is the start of the first record on the new copy"


@pytest.mark.parametrize(("recorded", "gap"), [
    ("2026-10-23T09:18:45.001Z", -0.001),  # a millisecond after the first record: 45.000 - 45.001
    ("2026-10-23T09:18:44.999Z", 0.001),
    ("2026-10-23T09:18:45.000+01:00", 3600.0),  # 08:18:45Z, an hour before
    ("2026-10-22T09:18:45Z", 86400.0),  # a day before (no tolerance: the ruling gives none)
    ("2026-10-23T09:18:39.962+00:00", 5.038),  # the manifest's own instant: 45.000 - 39.962
    ("2026-10-23T09:26:00Z", -435.0),  # 7 min 15 s after the first record, past every fixture record's start
])
def test_an_entry_recording_another_instant_than_the_first_record_is_timed_by_the_first_record_and_discloses_the_gap(
        world, ref, recorded, gap):
    m = _moved_world(world)
    exact = _stores(world, [m["out"]], m["proofs"], m["entries"])
    off = _stores(world, [m["out"]], m["proofs"], [m["entries"][0], {**m["entries"][1], "change_at": recorded}],
                  name="off.json")
    (change,), (exact_change,) = off["store_changes"], exact["store_changes"]
    assert "first_record_change_point" not in exact_change
    disclosure = change["first_record_change_point"]
    # the manifest's own instant is the fixture's 09:18:39.962, so its gap is 09:18:45.000 - 09:18:39.962 = 5.038 s
    assert {k: v for k, v in disclosure.items() if k != "rule"} == {
        "change_point": FIRST_RECORD, "manifest_at": MANIFEST_AT, "manifest_gap_seconds": 5.038,
        "entry_change_at": recorded, "entry_gap_seconds": gap}
    assert type(disclosure["entry_gap_seconds"]) is float and type(disclosure["manifest_gap_seconds"]) is float
    assert disclosure["rule"].startswith(RULE_START)
    assert change["first_record_on_new_endpoint"] == change["change_at_used"] == FIRST_RECORD
    # nothing downstream moves: the shares, the split by period and every other field equal the exact entry's call, but
    # for the disclosure itself, the entry's own text in its proof, and the config file's path and hash
    def plain(stores: dict) -> dict:
        stores = json.loads(json.dumps(stores))
        stores["config"] = {k: v for k, v in stores["config"].items() if k not in ("path", "sha256")}
        for info in stores["copies"].values():
            for proof in info["proofs"]:
                proof["change_at"] = None
        for c in stores["store_changes"]:
            c.pop("first_record_change_point", None)
        return stores
    assert plain(off) == plain(exact)
    assert off["shares"] == exact["shares"] and change["shares_by_arm"] == exact_change["shares_by_arm"]


def test_an_entry_recording_the_first_record_in_any_spelling_carries_no_disclosure(world, ref):
    m = _moved_world(world)
    for spelled in (CHANGE_AT, "2026-10-23T09:18:45.000Z", "2026-10-23T10:18:45.000+01:00", "2026-10-23T09:18:45+00:00"):
        entries = [m["entries"][0], {**m["entries"][1], "change_at": spelled}]
        (change,) = _stores(world, [m["out"]], m["proofs"], entries)["store_changes"]
        assert "first_record_change_point" not in change, spelled


def test_the_disclosure_is_the_two_instants_and_the_gap_and_nothing_else(world, ref):
    m = _moved_world(world)
    entries = [m["entries"][0], {**m["entries"][1], "change_at": "2026-10-23T09:18:30Z"}]
    st = _stores(world, [m["out"]], m["proofs"], entries)
    (change,) = st["store_changes"]
    assert set(change["first_record_change_point"]) == {"change_point", "manifest_at", "manifest_gap_seconds",
                                                        "entry_change_at", "entry_gap_seconds", "rule"}
    assert change["first_record_change_point"]["entry_gap_seconds"] == 15.0
    assert change["first_record_change_point"]["manifest_gap_seconds"] == 5.038
    assert _ints(change["first_record_change_point"]) == []  # a float, no count
    assert E0 not in json.dumps(st) and E1 not in json.dumps(st)
    assert "(Gemma e190 K=3)" in change["first_record_change_point"]["rule"]
    # the rule rides on the change that needed it: the section's own rule is the one every call carries
    assert st["rule"].endswith("two copies interleaving is a mix and refuses. " + BRANCH_RULE_TEXT)


def test_the_proof_checks_are_held_to_the_first_record_not_to_the_entrys_instant(world, ref):
    m = _moved_world(world)
    tmp, out = world["tmp"], m["out"]
    # an entry 45 s after the first record (09:18:45): a check made between them was made after the copy took attempts
    sc = _storecheck(tmp / "sc-b-mid.json", E1, "copy-b", checked_utc="2026-10-23T09:19:00Z")
    entry = _entry("single", E1, "copy-b", (sc, m["b"][1]), change_at="2026-10-23T09:19:30Z")
    _refused_stores(world, [out], [*m["a"], sc, m["b"][1]], [m["entries"][0], entry],
                    r"\(copy-b\): its storecheck was made after the copy took the OUT's attempts")
    ka = _load10(tmp / "ka-b-mid.json", E1, started_utc="2026-10-23T09:19:00Z")
    entry = _entry("single", E1, "copy-b", (m["b"][0], ka), change_at="2026-10-23T09:19:30Z")
    _refused_stores(world, [out], [*m["a"], m["b"][0], ka], [m["entries"][0], entry],
                    r"\(copy-b\): its known answers were run after the copy took the OUT's attempts")
    # an entry 15 s before the first record: a check between them was made before the copy took attempts (09:18:40),
    # one in the first record's own millisecond too (not after it), one a second later was not
    for checked, ok in (("2026-10-23T09:18:40Z", True), ("2026-10-23T09:18:45Z", True), ("2026-10-23T09:18:46Z", False)):
        sc = _storecheck(tmp / f"sc-b-{checked[-3:-1]}.json", E1, "copy-b", checked_utc=checked)
        entry = _entry("single", E1, "copy-b", (sc, m["b"][1]), change_at="2026-10-23T09:18:30Z")
        files, entries = [*m["a"], sc, m["b"][1]], [m["entries"][0], entry]
        if ok:
            (change,) = _stores(world, [out], files, entries)["store_changes"]
            assert change["first_record_change_point"]["entry_gap_seconds"] == 15.0
        else:
            _refused_stores(world, [out], files, entries,
                            r"\(copy-b\): its storecheck was made after the copy took the OUT's attempts")


def test_an_entry_off_the_first_record_still_refuses_an_old_copy_record_after_it(world, ref):
    m = _moved_world(world)
    out, first_a = m["out"], m["before"][0]
    entries = [m["entries"][0], {**m["entries"][1], "change_at": "2026-10-23T09:18:30Z"}]
    assert _stores(world, [out], m["proofs"], entries)["store_changes"][0]["first_record_change_point"]  # the twin
    # an old-copy record starting at 09:20:00, after the first on copy-b (09:18:45): a mix whatever the entry records
    _retime(out, lambda r, f: ("2026-10-23T09:20:00.000+00:00", "2026-10-23T09:20:00.500+00:00")
            if r["run_id"] == first_a else None)
    _refused_stores(world, [out], m["proofs"], entries,
                    r"single: its records' own endpoints and times do not follow its manifest store changes "
                    r"\(interleaved\): .*a mix of copies, not a late timestamp")
    # one that starts five seconds before it and ends at 09:18:50 ran past it: an overlap
    _retime(out, lambda r, f: ("2026-10-23T09:18:40.000+00:00", "2026-10-23T09:18:50.000+00:00")
            if r["run_id"] == first_a else None)
    _refused_stores(world, [out], m["proofs"], entries, r"\(overlap\): .*a mix of copies, not a late timestamp")
    # and the entry's instant after the first record changes nothing about that
    late = [m["entries"][0], {**m["entries"][1], "change_at": "2026-10-23T09:19:30Z"}]
    _refused_stores(world, [out], m["proofs"], late, r"\(overlap\): .*a mix of copies, not a late timestamp")


def test_a_copy_behind_the_endpoint_is_ordered_after_the_change_point_not_after_the_entrys_instant(world, ref):
    # Opus review B1. The first copy's checks are held to the first record (09:18:45), so the switch to the copy behind
    # the same endpoint must come strictly after THAT, whatever instant the first entry records. By hand, with F =
    # 09:18:45 and T_e the first entry's instant, T_s the switch instant of copy-c behind the endpoint:
    m = _moved_world(world)
    out, tmp = m["out"], world["tmp"]
    c = _copy_proofs(tmp, "c", E1, "copy-c")
    switch_order = r"its copies' switch instants are not strictly after one another"

    def check(recorded, switch):
        entries = [m["entries"][0], {**m["entries"][1], "change_at": recorded},
                   _entry("single", E1, "copy-c", c, behind_endpoint=True, change_at=switch)]
        return _check(world, out, (*m["proofs"], *c), entries)

    for recorded, switch in (("2026-10-23T09:18:00Z", "2026-10-23T09:18:30Z"),  # T_e < T_s < F: before the change point
                             ("2026-10-23T09:18:00Z", CHANGE_AT),  # T_s = F: not strictly after
                             (CHANGE_AT, CHANGE_AT)):  # the exact entry, as it always was
        with pytest.raises(ex.Refused, match=switch_order):
            check(recorded, switch)
    # F < T_s <= T_e is valid by the change point (it was refused for T_e != F before, and for T_s <= T_e by the order)
    for recorded, gap in (("2026-10-23T09:20:00Z", -75.0), ("2026-10-23T09:19:00Z", -15.0)):  # F - T_e
        stores = check(recorded, "2026-10-23T09:19:00Z")
        assert stores.labels[(str(out), E1)] == "copy-b then copy-c (one endpoint, unsplittable)"
        assert stores.first_record_points[(str(out), 1)].entry_gap_seconds == gap
    stores = check(CHANGE_AT, "2026-10-23T09:19:00Z")  # the exact entry, the switch after it: as it was
    assert stores.first_record_points == {} and stores.labels[(str(out), E1)].endswith("(one endpoint, unsplittable)")


@pytest.mark.parametrize(("lines", "refused"), [
    # the only ssh came up at 09:15, after T_e and before F: running when the copy took attempts
    (("2026-10-23T09:15:00Z e184-copy-b ssh pid 101 127.0.0.1:48782 -> 192.0.2.7:28782",), False),
    # an ssh up at 09:00 that exited at 09:15 ran at T_e but not at F: refused, as the first record is the bound
    (("2026-10-23T09:00:00Z e184-copy-b ssh pid 101 127.0.0.1:48782 -> 192.0.2.7:28782",
      "2026-10-23T09:15:00Z e184-copy-b ssh pid 101 exited rc=255 after 900 s"), True)])
def test_a_forward_is_held_to_the_first_record_for_an_entry_recording_an_earlier_instant(world, ref, lines, refused):
    # Opus review N3. T_e = 09:10:00, F = 09:18:45.000, so the gap is F - T_e = 525 s
    early = "2026-10-23T09:10:00Z"
    f = _forward_world(world, lines)
    a, b = f["entries"]
    entries = [a, {**b, "change_at": early}]
    if refused:
        _refused_stores(world, [f["out"]], f["proofs"], entries,
                        rf"single {ex.endpoint_label(FWD)} \(copy-b\): .*no ssh of its forward was running when the "
                        r"copy took the OUT's attempts")
    else:
        (change,) = _stores(world, [f["out"]], f["proofs"], entries)["store_changes"]
        assert change["first_record_change_point"]["entry_gap_seconds"] == 525.0


def test_an_ident_py_check_is_held_to_the_first_record_for_an_entry_recording_a_later_instant(world, old_world):
    # Opus review N3. T_e = 09:30:00, F = 09:18:45.000: the gap F - T_e is -675 s
    tmp, proofs, (a, b) = world["tmp"], old_world["proofs"], old_world["entries"]
    late = {**b, "change_at": "2026-10-23T09:30:00Z"}
    made = _ident(tmp / "ident-b-mid.json", {E1: dict(STATS)}, checked_utc="2026-10-23T09:19:00Z")  # after F, before T_e
    _refused_stores(world, [old_world["out"]], [*proofs, made], [a, {**late, "storecheck_sha256": _sha(made)}],
                    rf"single {ex.endpoint_label(E1)} \(copy-b\): .*its identity check was made after the copy took "
                    r"the OUT's attempts")
    before = _ident(tmp / "ident-b-pre.json", {E1: dict(STATS)}, checked_utc="2026-10-23T09:18:40Z")
    (change,) = _stores(world, [old_world["out"]], [*proofs, before],
                        [a, {**late, "storecheck_sha256": _sha(before)}])["store_changes"]
    assert change["first_record_change_point"]["entry_gap_seconds"] == -675.0


def test_check_stores_notes_on_stderr_a_recorded_instant_more_than_an_hour_from_the_first_record(
        world, ref, s01_graph, capsys):
    note = ("check-stores: note: an entry's recorded change_at or its manifest store change's at is more than an "
            "hour from the first record on its copy (see first_record_change_points)\n")
    m = _moved_world(world)
    for recorded, gap, noted in (("2026-10-22T09:18:45Z", 86400.0, True), ("2026-10-23T08:18:44Z", 3601.0, True),
                                 ("2026-10-23T08:18:45Z", 3600.0, False), ("2026-10-23T09:18:30Z", 15.0, False)):
        entries = [m["entries"][0], {**m["entries"][1], "change_at": recorded}]
        code, obj, raw, err = _check_cli(capsys, world, [m["out"]], m["proofs"], entries)
        assert code == 0 and obj["first_record_change_points"]["moves"][0]["entry_gap_seconds"] == gap, recorded
        assert err == (note if noted else ""), recorded
    code, obj, raw, err = _check_cli(capsys, world, [m["out"]], m["proofs"], m["entries"])
    assert (code, err, set(obj)) == (0, "", CHECK_KEYS)  # the exact entry: no key, no note
    # the manifest's instant counts too: written 3601 s after the first record (10:18:46) with the entry 15 s off, the note
    # is there; 3600 s (10:18:45), it is not
    for written, gap, noted in (("2026-10-23T10:18:46.000+00:00", -3601.0, True),
                                ("2026-10-23T10:18:45.000+00:00", -3600.0, False)):
        _edit_manifest(m["out"], lambda manifest, w=written: manifest["store_changes"][0].update(at=w))
        entries = [m["entries"][0], {**m["entries"][1], "change_at": "2026-10-23T09:18:30Z"}]
        code, obj, raw, err = _check_cli(capsys, world, [m["out"]], m["proofs"], entries)
        assert code == 0 and obj["first_record_change_points"]["moves"][0]["manifest_gap_seconds"] == gap, written
        assert err == (note if noted else ""), written


def test_the_manifests_recorded_instant_and_its_gap_are_disclosed_beside_the_entrys(world, ref):
    # Opus review B2, science's ruling: "the manifest's recorded instant disclosed beside" the first record, "the gap
    # between the two instants". The manifest's instant is the store change's own `at`, apart from the entry's change_at.
    # By hand, F = 09:18:45.000. A manifest written AFTER the first record (09:25:00) and an entry recording 09:18:30:
    #   manifest gap 09:18:45 - 09:25:00 = -375.0 s, entry gap 09:18:45 - 09:18:30 = +15.0 s
    m = _moved_world(world)
    out = m["out"]
    _edit_manifest(out, lambda manifest: manifest["store_changes"][0].update(at="2026-10-23T09:25:00.000+00:00"))
    entries = [m["entries"][0], {**m["entries"][1], "change_at": "2026-10-23T09:18:30Z"}]
    st = _stores(world, [out], m["proofs"], entries)
    (change,) = st["store_changes"]
    assert {k: v for k, v in change["first_record_change_point"].items() if k != "rule"} == {
        "change_point": FIRST_RECORD, "manifest_at": "2026-10-23T09:25:00.000+00:00", "manifest_gap_seconds": -375.0,
        "entry_change_at": "2026-10-23T09:18:30Z", "entry_gap_seconds": 15.0}
    assert change["manifest_at"] == "2026-10-23T09:25:00.000+00:00" and change["first_precedes_manifest_at"] is True
    # the same instants whichever zone the manifest writes: 10:25:00+01:00 is 09:25:00Z, so the gap is the same, and the
    # text is the manifest's own
    _edit_manifest(out, lambda manifest: manifest["store_changes"][0].update(at="2026-10-23T10:25:00+01:00"))
    (change,) = _stores(world, [out], m["proofs"], entries, name="zone.json")["store_changes"]
    assert (change["first_record_change_point"]["manifest_at"],
            change["first_record_change_point"]["manifest_gap_seconds"]) == ("2026-10-23T10:25:00+01:00", -375.0)
    # the exact entry discloses nothing, whatever the manifest says (its `at` is already beside every move's change point)
    (change,) = _stores(world, [out], m["proofs"], m["entries"], name="exact.json")["store_changes"]
    assert "first_record_change_point" not in change and change["manifest_at"] == "2026-10-23T10:25:00+01:00"


def test_an_entry_off_the_first_record_needs_a_manifest_instant_to_disclose(world, ref):
    m = _moved_world(world)
    out = m["out"]
    off = [m["entries"][0], {**m["entries"][1], "change_at": "2026-10-23T09:18:30Z"}]
    for bad in ("not a time", None, "2026-10-23T09:18:39.962"):  # no instant, none at all, or one without a zone
        _edit_manifest(out, lambda manifest, bad=bad: manifest["store_changes"][0].update(at=bad))
        _refused_stores(world, [out], m["proofs"], off,
                        r"\(copy-b\): its manifest store change 1 records no instant \(an ISO time with a zone\) "
                        r"to disclose beside the first record")
        # an entry that records the first record still passes: nothing is disclosed, as before this rule
        assert _stores(world, [out], m["proofs"], m["entries"], name="exact.json")["store_changes"]


def test_a_gap_is_disclosed_only_for_an_instant_in_the_one_exact_form(world, ref):
    # external reviewer r5, r6: datetime reads forms whose value is not the one the text gives (nine fraction digits truncated to six,
    # the basic form 091839962123456, a comma). The disclosure takes YYYY-MM-DDTHH:MM:SS[.f to ffffff] and Z or an offset,
    # and refuses the rest. By hand, F = 09:18:45.000: against 09:18:39.962123 the gap is 5.037877 s, exact
    m = _moved_world(world)
    out = m["out"]
    off = [m["entries"][0], {**m["entries"][1], "change_at": "2026-10-23T09:18:30Z"}]
    for at, gap in (("2026-10-23T09:18:39.962123+00:00", 5.037877), ("2026-10-23T09:18:39Z", 6.0),
                    ("2026-10-23T10:18:39.5+01:00", 5.5)):
        _edit_manifest(out, lambda manifest, at=at: manifest["store_changes"][0].update(at=at))
        (change,) = _stores(world, [out], m["proofs"], off)["store_changes"]
        assert change["first_record_change_point"]["manifest_gap_seconds"] == gap, at
    manifest_refusal = (r"\(copy-b\): its manifest store change 1 records an instant not written as "
                        r"YYYY-MM-DDTHH:MM:SS\[\.ffffff\] with Z or an offset, so its gap to the first record cannot be "
                        r"disclosed exactly")
    for at in ("2026-10-23T09:18:39.962123456+00:00", "2026-10-23T091839962123456+00:00", "2026-10-23T09:18:39,962+00:00",
               "20261023T091839Z", "2026-10-23 09:18:39+00:00", "2026-10-23T09:18:39.+00:00",
               "2026-10-23T09:18:39+00:60", "2026-10-23T09:18:39+01:75"):  # an offset's minutes are normalised, not refused
        _edit_manifest(out, lambda manifest, at=at: manifest["store_changes"][0].update(at=at))
        _refused_stores(world, [out], m["proofs"], off, manifest_refusal)
        assert _stores(world, [out], m["proofs"], m["entries"], name="exact.json")["store_changes"]  # nothing disclosed
    _edit_manifest(out, lambda manifest: manifest["store_changes"][0].update(at=MANIFEST_AT))
    entry_refusal = (r"\(copy-b\): the entry's change_at or the first record's start is not written as "
                     r"YYYY-MM-DDTHH:MM:SS\[\.ffffff\] with Z or an offset, so the gaps cannot be disclosed exactly")
    for bad in ("2026-10-23T09:18:30.0000001Z", "2026-10-23T091830Z", "2026-10-23T09:18:30,5Z",
                "2026-10-23T09:18:30+00:60"):
        _refused_stores(world, [out], m["proofs"], [m["entries"][0], {**m["entries"][1], "change_at": bad}], entry_refusal)


def test_the_manifest_store_change_of_a_move_is_matched_by_its_number_and_its_endpoints():
    changes = [_change(1, A, B), _change(2, B, C)]
    ct = ex.store_timeline([_row("records/a1.json", A, _at(0), _at(1)), _row("records/b1.json", B, _at(10), _at(11)),
                            _row("records/c1.json", C, _at(20), _at(21))], changes, "records").changes
    assert ex.manifest_instant_of(changes, ct[1]) == (
        ("2026-10-23T09:00:02.500+00:00", datetime(2026, 10, 23, 9, 0, 2, 500000, tzinfo=UTC)), None)
    for wrong in ([], [changes[0]], [*changes, changes[1]],  # none, only the other change, two for one move
                  [changes[0], _change(2, B, A)]):  # the right number, other endpoints
        found, why = ex.manifest_instant_of(wrong, ct[1])
        assert found is None and why == ("its manifest store change 2 is not matched by exactly one of the OUT's manifest "
                                         "store changes, so the manifest's recorded instant cannot be disclosed "
                                         "beside the first record")
    found, why = ex.manifest_instant_of([changes[0], _change(2, B, C, at="soon")], ct[1])
    assert found is None and why == ("its manifest store change 2 records no instant (an ISO time with a zone) to "
                                     "disclose beside the first record")


def test_an_entry_naming_two_outs_of_the_call_keeps_the_exact_equality(world, ref):
    # external reviewer r1: an entry is selected for an OUT by its name alone, so one entry could prove the copies of two OUTs that
    # share a name, each with its own first record. Hand-derived: the twin is the OUT again with copy-b's first record
    # at 09:19:00 (the original's is 09:18:45); the one entry names "/host/single", which both OUTs are known by.
    m = _moved_world(world)
    out, tmp = m["out"], world["tmp"]
    twin = tmp / "twin"
    shutil.copytree(out, twin)
    _stamp_by_endpoint(twin, E0, E1, dst_from=datetime(2026, 10, 23, 9, 19, 0, tzinfo=UTC))
    classes = _classes_file(tmp / "classes-twin.json")

    def check(outs, entries):
        return ex.check_stores(ex.load_attribution(), [ex.Out.read(o) for o in outs], "s01", tuple(m["proofs"]),
                               _store_config(tmp / "twin.json", *entries), None, classes)

    off = [m["entries"][0], {**m["entries"][1], "change_at": "2026-10-23T09:18:30Z"}]
    assert check([out], off).first_record_points  # alone, the OUT's name is its own: the relaxed rule applies
    assert check([out], m["entries"]).first_record_points == {}  # and the exact entry discloses nothing
    for entries in (m["entries"], off):  # with the twin the equality holds: refused for the twin (09:19:00), and for both
        with pytest.raises(ex.Refused) as caught:
            check([out, twin], entries)
        assert f"{twin} {ex.endpoint_label(E1)} (copy-b): its change_at is not the first record on the copy" in str(caught.value)
    # an entry equal to each OUT's first record would be the same instant twice; the twin alone is a name of its own
    assert check([twin], [m["entries"][0], {**m["entries"][1], "change_at": "2026-10-23T09:18:45Z"}]
                 ).first_record_points[(str(twin), 1)].entry_gap_seconds == 15.0  # 09:19:00 - 09:18:45


def test_an_old_copy_record_starting_in_the_first_records_millisecond_is_a_mix_in_either_name_order():
    first = _row("records/m1.json", B, _at(10), _at(11))
    for name, flag in (("records/a9.json", "overlap"), ("records/z9.json", "interleaved")):
        rows = [_row("records/a1.json", A, _at(0), _at(1)), _row(name, A, _at(10), _at(10.5)), first]
        line = ex.store_timeline(rows, [_change(1, A, B)], "records")
        assert not line.clean and getattr(line, flag) is True, name
        assert ex.timeline_problems("OUT", line)  # by name before the first record, an overlap; after it, a third run
    # unchanged by the first-record rule, and not "after" the change point: a zero-duration old-copy record in the first
    # record's own millisecond that sorts before it ends in that millisecond, which the overlap test allows
    zero = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/a9.json", A, _at(10), _at(10)), first]
    assert ex.store_timeline(zero, [_change(1, A, B)], "records").clean


def test_no_record_on_the_new_copy_can_start_before_the_change_point():
    # the change point is the earliest start on the new copy's run, whatever the order the rows are given in, and a row
    # of the new copy before an old-copy row makes the copies alternate: not clean, and no change point is then used
    rows = [_row("records/b3.json", B, _at(14), _at(15)), _row("infrastructure/b0.json", B, _at(9), _at(9.5)),
            _row("records/a1.json", A, _at(0), _at(1)), _row("records/b1.json", B, _at(10), _at(11))]
    line = ex.store_timeline(rows, [_change(1, A, B)], "records")
    (change,) = line.changes
    assert line.clean and change.first_to_started == datetime(2026, 10, 23, 9, 0, 9, tzinfo=UTC)
    assert all(parse_start >= change.first_to_started for parse_start in
               (ex.parse_utc(r.started) for r in rows if r.endpoint == B))
    rows.append(_row("records/b9.json", B, _at(-5), _at(-4)))  # a new-copy row before the old copy's first
    alternating = ex.store_timeline(rows, [_change(1, A, B)], "records")
    assert not alternating.clean and alternating.interleaved is True
    assert alternating.changes[0].first_to_started is None  # the chain is not reached in order: no change point at all


def test_branch_rows_do_not_move_the_change_point_of_an_entry_off_the_first_record(world, ref):
    m = _moved_world(world)
    out = m["out"]
    cell, a, b = _two_late_rows(out)
    _set_branch_phase(out, T_B)
    exact = _stores(world, [out], m["proofs"], m["entries"])
    for recorded, gap in (("2026-10-23T09:18:30Z", 15.0), ("2026-10-23T09:26:00Z", -435.0)):  # before and after T_b
        entries = [m["entries"][0], {**m["entries"][1], "change_at": recorded}]
        off = _stores(world, [out], m["proofs"], entries, name="off.json")
        (change,), (exact_change,) = off["store_changes"], exact["store_changes"]
        assert change["first_record_change_point"]["entry_gap_seconds"] == gap
        assert change["first_record_change_point"]["change_point"] == FIRST_RECORD and change["branch_phase_start"] == T_B
        # the late rows are branch rows (copy-b's live retries) and the shares are the exact entry's call's
        assert {k: v for k, v in change.items() if k != "first_record_change_point"} == exact_change
        assert off["shares"] == exact["shares"] and off["branch_phase_copies"] == exact["branch_phase_copies"]
        arm = f"s01|{QWEN}|{cell}|prompt|-"
        assert change["shares_by_arm"][arm]["branch_phase"] == _rows({"copy-b": 2})


def test_the_check_stores_verdict_lists_the_instants_and_gap_only_for_an_entry_off_the_first_record(
        world, ref, s01_graph, capsys):
    m = _moved_world(world)
    entries = [m["entries"][0], {**m["entries"][1], "change_at": "2026-10-23T09:18:30Z"}]
    code, obj, raw, err = _check_cli(capsys, world, [m["out"]], m["proofs"], entries)
    assert (code, err) == (0, "")
    assert set(obj) == CHECK_KEYS | {"first_record_change_points"}
    # the copy's period began at the first record, not at the entry's instant
    assert obj["copies"]["copy-b"]["periods"] == [FIRST_RECORD] and obj["copies"]["copy-a"]["periods"] == [None]
    points = obj["first_record_change_points"]
    assert set(points) == {"rule", "moves"} and points["rule"].startswith(RULE_START)
    assert points["moves"] == [{"out": "single", "to": "copy-b", "change_point": FIRST_RECORD,
                                "manifest_at": MANIFEST_AT, "manifest_gap_seconds": 5.038,
                                "entry_change_at": "2026-10-23T09:18:30Z", "entry_gap_seconds": 15.0}]
    assert _ints(obj) == [] and E0 not in raw and E1 not in raw and str(m["out"]) not in raw
    # with the first record as the entry's instant the verdict is the one it was: no key, the entry's own text
    code, obj, raw, err = _check_cli(capsys, world, [m["out"]], m["proofs"], m["entries"])
    assert (code, err) == (0, "") and set(obj) == CHECK_KEYS and obj["copies"]["copy-b"]["periods"] == [CHANGE_AT]


# --- the store-timeline command: one JSON object per OUT, no counts ---------------------------------------------------

CLI_KEYS = {"out", "manifest_sha256", "chain", "changes", "clean", "interleaved", "overlap", "chain_incomplete",
            "endpoint_outside_chain", "rows_without_endpoint", "unusable_times", "run_id_in_two_runs", "unusable_rows",
            "runs", "verdict", "branch_phase_start", "branch_row_endpoints"}  # the last two: the branch phase (proposed)
CLI_RUN_KEYS = {"endpoint", "first_started", "last_started", "last_ended", "in_records_dir", "in_setaside"}
CLI_CHANGE_KEYS = {"index", "from", "to", "manifest_at", "first_record_on_new_endpoint", "first_precedes_manifest_at",
                   "manifest_lists_to_endpoint_record_as_before", "manifest_omits_from_endpoint_record",
                   "setaside_changes_first_instant"}


def _cli(capsys, *outs, root: Path = ROOT):
    code = ex.main(["store-timeline", "--code-root", str(root), *map(str, outs)])
    seen = capsys.readouterr()
    return code, [json.loads(line) for line in seen.out.splitlines()], seen.out, seen.err


def _ints(value, path="") -> list[str]:
    """Where an int (not a bool) sits in a JSON value."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [path]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _ints(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [p for v in value for p in _ints(v, f"{path}[]")]
    return []


def test_the_command_prints_a_clean_out_by_labels_and_without_counts(world, capsys):
    m = _moved_world(world)
    code, (obj,), raw, err = _cli(capsys, m["out"])
    assert (code, err) == (0, "")
    assert set(obj) == CLI_KEYS and obj["verdict"] == "CLEAN" and obj["out"] == str(m["out"])
    assert obj["branch_phase_start"] is None and obj["branch_row_endpoints"] is None  # no branch phase in the manifest
    assert obj["manifest_sha256"] == _sha(m["out"] / "manifest.json")
    assert obj["chain"] == [f"endpoint sha256:{_h(E0)[:12]}", f"endpoint sha256:{_h(E1)[:12]}"]
    (change,) = obj["changes"]
    assert set(change) == CLI_CHANGE_KEYS
    assert change == {"index": 1, "from": obj["chain"][0], "to": obj["chain"][1], "manifest_at": MANIFEST_AT,
                      "first_record_on_new_endpoint": "2026-10-23T09:18:45.000+00:00",
                      "first_precedes_manifest_at": False, "manifest_lists_to_endpoint_record_as_before": False,
                      "manifest_omits_from_endpoint_record": False, "setaside_changes_first_instant": False}
    assert (obj["clean"], obj["interleaved"], obj["overlap"], obj["chain_incomplete"], obj["endpoint_outside_chain"],
            obj["rows_without_endpoint"], obj["unusable_times"], obj["unusable_rows"]) == (
        True, False, False, False, False, False, False, [])
    # copy-a's 20 records start a second apart from 09:10:00, copy-b's 16 from 09:18:45 (each ends half a second on)
    assert obj["runs"] == [
        {"endpoint": obj["chain"][0], "first_started": "2026-10-23T09:10:00.000+00:00",
         "last_started": "2026-10-23T09:10:19.000+00:00", "last_ended": "2026-10-23T09:10:19.500+00:00",
         "in_records_dir": True, "in_setaside": False},
        {"endpoint": obj["chain"][1], "first_started": "2026-10-23T09:18:45.000+00:00",
         "last_started": "2026-10-23T09:19:00.000+00:00", "last_ended": "2026-10-23T09:19:00.500+00:00",
         "in_records_dir": True, "in_setaside": False}]
    assert all(set(r) == CLI_RUN_KEYS for r in obj["runs"])
    assert _ints(obj) == [".changes[].index"]  # no count of any kind
    assert "invalid" not in raw and "http" not in raw  # labels only, no host
    assert not [r for r in m["before"] if r in raw]  # no run id
    assert raw.endswith("\n") and raw.count("\n") == 1 and list(json.loads(raw)) == sorted(json.loads(raw))


def test_the_command_says_stop_with_exit_2_for_a_mix_and_still_prints_every_out(world, capsys):
    m = _moved_world(world)
    mixed = world["parts"][1]  # s01-flag and s01-abstain: flag on copy-a, abstain on copy-b, then one flag record late
    before = _set_endpoints(mixed, lambda r: E0 if r["question_id"] == "s01-flag" else E1)
    _store_change(mixed, E0, E1, before)
    late = _record_of(mixed, "s01-flag")[1]["run_id"]
    _retime(mixed, lambda r, f: "2026-10-23T09:20:00.000+00:00" if r["run_id"] == late else None)
    code, objs, raw, err = _cli(capsys, mixed, m["out"], world["parts"][0])
    assert code == 2
    assert [o["out"] for o in objs] == [str(mixed), str(m["out"]), str(world["parts"][0])]
    stop, clean, none = objs
    assert stop["verdict"] == "STOP: interleaved" and stop["clean"] is False and stop["interleaved"] is True
    assert (stop["overlap"], stop["chain_incomplete"], stop["endpoint_outside_chain"]) == (False, False, False)
    assert clean["verdict"] == "CLEAN"
    assert none == {"out": str(world["parts"][0]), "store_changes": False, "several_endpoints": False,
                    "branch_phase_start": FIXTURE_PHASE_BRANCH_AT}
    assert err == ("STOP interleaved: records of the copies alternate in time, a mix of copies, not a late timestamp\n")
    assert _ints(stop) == [".changes[].index"] and "invalid" not in raw


def test_the_command_names_every_refusal_class_once_and_exits_3_for_an_unreadable_out(world, capsys, tmp_path):
    m = _moved_world(world)
    empty = tmp_path / "empty-out"
    empty.mkdir()
    f, record = _record_of(m["out"], "s01-noquery")
    record["started_at"] = None
    f.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    code, objs, _raw, err = _cli(capsys, m["out"], empty)
    assert code == 3  # 3 wins over 2
    assert objs[0]["verdict"] == "STOP: unusable_times" and objs[0]["unusable_rows"] == [f"records/{f.name}"]
    assert (objs[0]["interleaved"], objs[0]["overlap"], objs[0]["chain_incomplete"]) == (None, None, None)
    assert objs[0]["runs"] is None  # the order is unknown
    assert objs[1] == {"out": str(empty), "unreadable": "FileNotFoundError"}
    assert err.splitlines() == [f"unreadable OUT {empty}: FileNotFoundError",
                                "STOP unusable_times: a record lacks a usable started_at or ended_at"]
    code, objs, _raw, err = _cli(capsys, empty)
    assert (code, err) == (3, f"unreadable OUT {empty}: FileNotFoundError\n")
    code, _objs, _raw, err = _cli(capsys, m["out"], root=tmp_path)  # a code root that is not the frozen analysis
    assert (code, err) == (3, "store-timeline: the frozen analysis or the attribution check cannot be loaded (Refused)\n")


def test_the_command_reports_an_out_without_store_changes_and_whether_its_rows_name_several_endpoints(world, capsys):
    out = world["single"]
    code, (one,), _raw, err = _cli(capsys, out)
    assert (code, err, one) == (0, "", {"out": str(out), "store_changes": False, "several_endpoints": False,
                                        "branch_phase_start": FIXTURE_PHASE_BRANCH_AT})
    _set_endpoints(out, lambda r: E0 if r["question_id"] in BEFORE_Q else E1)  # two endpoints, no recorded change
    code, (two,), _raw, err = _cli(capsys, out)
    assert (code, err, two) == (0, "", {"out": str(out), "store_changes": False, "several_endpoints": True,
                                        "branch_phase_start": FIXTURE_PHASE_BRANCH_AT})


def test_the_command_stops_for_a_chain_the_records_do_not_reach_and_for_a_malformed_manifest(world, capsys):
    m = _moved_world(world)
    _set_endpoints(m["out"], lambda r: E0)  # nothing on copy-b: the manifest says the OUT moved
    code, (obj,), _raw, err = _cli(capsys, m["out"])
    assert (code, obj["verdict"], obj["chain_incomplete"], obj["interleaved"]) == (2, "STOP: chain_incomplete", True, False)
    assert obj["changes"][0]["first_record_on_new_endpoint"] is None
    assert obj["runs"] == [{"endpoint": obj["chain"][0], "first_started": "2026-10-23T09:10:00.000+00:00",
                            "last_started": "2026-10-23T09:19:00.000+00:00", "last_ended": "2026-10-23T09:19:00.500+00:00",
                            "in_records_dir": True, "in_setaside": False}]  # one run: all 36 records now on copy-a
    assert err.startswith("STOP chain_incomplete:") and err.count("\n") == 1
    _set_endpoints(m["out"], lambda r: EX)  # records on a copy no store change lists
    code, (obj,), _raw, err = _cli(capsys, m["out"])
    assert (code, obj["verdict"], obj["endpoint_outside_chain"]) == (2, "STOP: endpoint_outside_chain", True)
    _edit_manifest(m["out"], lambda man: man.update(store_changes={"not": "a list"}))
    code, (obj,), _raw, err = _cli(capsys, m["out"])
    assert (code, set(obj), obj["verdict"]) == (2, {"out", "manifest_sha256", "manifest_store_changes_malformed",
                                                     "verdict"}, "STOP: manifest_store_changes_malformed")
    assert err == "STOP manifest_store_changes_malformed: the manifest's store_changes do not join up\n"


# --- review r1 of the store timeline (external reviewer gpt-6-sol, 25 Oct): five blocking items, each with a test that fails on the
# code it was written against ------------------------------------------------------------------------------------------


def test_a_declared_change_the_records_do_not_reach_refuses_in_the_call_though_no_model_has_two_endpoints(world, ref):
    m = _moved_world(world)
    _set_endpoints(m["out"], lambda r: E0)  # the manifest says A -> B; every record is on A: no (graph, model) has two
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"], r"\(chain_incomplete\)")
    _refused_stores(world, [m["out"]], (), [], r"\(chain_incomplete\)")  # and no entry is needed for it to refuse


def test_a_declared_change_with_another_change_at_is_timed_by_its_first_record_though_no_model_has_two_endpoints(world, ref):
    """Qwen's records all on copy-a, Gemma's all on copy-b: the OUT moved, yet no (graph, model) names two endpoints."""
    out, tmp = world["single"], world["tmp"]
    before = _set_endpoints(out, lambda r: E0 if r["model"] == QWEN else E1)
    _store_change(out, E0, E1, before)  # Qwen's records from 09:10:00, Gemma's from 09:18:45
    a, b = _copy_proofs(tmp, "a", E0, "copy-a"), _copy_proofs(tmp, "b", E1, "copy-b")
    entries = [_entry("single", E0, "copy-a", a), _entry("single", E1, "copy-b", b, change_at=CHANGE_AT)]
    st = _stores(world, [out], (*a, *b), entries)
    assert st["several_endpoints"] == [] and st["store_changes"][0]["first_record_on_new_endpoint"] == _iso(DST_FROM)
    # since science's ruling of 25 Oct (Gemma e190 K=3) an entry off the first record is timed by the first record and
    # discloses the gap (by hand: 09:18:45.000 - 09:18:45.001 = -0.001 s, 09:18:45.000 - 09:18:44.999 = +0.001 s)
    for wrong, gap in (("2026-10-23T09:18:45.001Z", -0.001), ("2026-10-23T09:18:44.999Z", 0.001)):
        st = _stores(world, [out], (*a, *b), [entries[0], {**entries[1], "change_at": wrong}])
        (change,) = st["store_changes"]
        assert (change["first_record_change_point"]["entry_change_at"],
                change["first_record_change_point"]["entry_gap_seconds"]) == (wrong, gap)
        assert change["first_record_on_new_endpoint"] == _iso(DST_FROM)


def test_a_b_a_b_by_three_changes_is_one_second_period_entry_per_later_move_each_to_its_own_change(world, ref):
    # T1's test of this chain asserted a refusal "not available in this build" for b, moved to twice. The general rule
    # replaced it: b and a each have two periods, one second_period entry each, at its own change's first record
    w = _aba_world(world, abab=True)
    first_a, first_b, second_a, second_b = w["entries"]
    st = _stores(world, [w["out"]], w["proofs"], w["entries"])
    B_TWO = "copy-b (two periods)"
    one, two, three = st["store_changes"]
    assert [(c["from"], c["to"], c["first_record_on_new_endpoint"]) for c in (one, two, three)] == [
        (A_TWO, B_TWO, "2026-10-23T09:18:45.000+00:00"), (B_TWO, A_TWO, "2026-10-23T09:30:00.000+00:00"),
        (A_TWO, B_TWO, "2026-10-23T09:40:00.000+00:00")]
    # Qwen's natural cell, one record per question: pass -> a, flag -> b, abstain -> a, noquery -> b
    assert st["shares"][ARM] == _rows({A_TWO: 2, B_TWO: 2})
    assert one["shares_by_arm"][ARM] == {"before": _rows({A_TWO: 1}), "after": _rows({B_TWO: 2, A_TWO: 1})}
    assert two["shares_by_arm"][ARM] == {"before": _rows({A_TWO: 1, B_TWO: 1}), "after": _rows({A_TWO: 1, B_TWO: 1})}
    assert three["shares_by_arm"][ARM] == {"before": _rows({A_TWO: 2, B_TWO: 1}), "after": _rows({B_TWO: 1})}
    assert [p["change_at"] for p in st["copies"][B_TWO]["periods"]] == ["2026-10-23T09:18:45Z", B_RETURN_AT]
    assert [p["change_at"] for p in st["copies"][A_TWO]["proofs"]] == [None, RETURN_AT]
    assert st["config"]["matched"] == 4
    # b's return missing: refused for b alone (a's is there); a's missing: for a alone
    with pytest.raises(ex.Refused, match=rf"{_where(E1)}: a later period of the copy without its entry") as caught:
        _check(world, w["out"], w["proofs"], [first_a, first_b, second_a])
    assert ex.endpoint_label(E0) not in str(caught.value)
    with pytest.raises(ex.Refused, match=rf"{_where(E0)}: a later period of the copy without its entry") as caught:
        _check(world, w["out"], w["proofs"], [first_a, first_b, second_b])
    assert ex.endpoint_label(E1) not in str(caught.value)
    # each is held to its own change: the entries' instants swapped between the two copies
    swapped = [first_a, first_b, {**second_a, "change_at": B_RETURN_AT}, {**second_b, "change_at": RETURN_AT}]
    with pytest.raises(ex.Refused) as caught:
        _check(world, w["out"], w["proofs"], swapped)
    assert str(caught.value).count("its change_at is not the first record on the copy") == 2


def _runs_world(world) -> dict:
    """The fixture whole on a, b, a, b, a by four changes, five runs by hand: s01-pass of Qwen on a from 09:10:00,
    s01-pass of Gemma on b from 09:20:00, s01-flag on a from 09:30:00, s01-abstain on b from 09:40:00 and s01-noquery
    on a from 09:50:00 (a second apart within a run). So a has three periods and b two, and Qwen's natural cell has pass,
    flag and noquery on a and abstain on b. Each period has its own checks, made before it began."""
    out, tmp = world["single"], world["tmp"]

    def run_of(r: dict) -> int:
        return {"s01-flag": 2, "s01-abstain": 3, "s01-noquery": 4}.get(r["question_id"], 0 if r["model"] == QWEN else 1)

    _set_endpoints(out, lambda r: E0 if run_of(r) % 2 == 0 else E1)
    seen: Counter = Counter()

    def start_of(r, f):
        k = seen[run_of(r)]
        seen[run_of(r)] += 1
        return datetime(2026, 10, 23, 9, 10 + 10 * run_of(r), 0, tzinfo=timezone.utc) + timedelta(seconds=k)

    ids_upto = [sorted(json.loads(f.read_text(encoding="utf-8"))["run_id"] for f in _record_files(out)
                       if run_of(json.loads(f.read_text(encoding="utf-8"))) <= j) for j in range(4)]
    for j in range(4):
        _store_change(out, E0 if j % 2 == 0 else E1, E1 if j % 2 == 0 else E0, ids_upto[j], stamp=False)
    _retime(out, start_of)
    starts = ["2026-10-23T09:20:00Z", "2026-10-23T09:30:00Z", "2026-10-23T09:40:00Z", "2026-10-23T09:50:00Z"]
    files, entries = [], []
    for n, (ep, replica) in enumerate([(E0, "copy-a"), (E1, "copy-b"), (E0, "copy-a"), (E1, "copy-b"), (E0, "copy-a")]):
        pair = (_storecheck(tmp / f"sc-run{n}.json", ep, replica, checked_utc=f"2026-10-23T09:0{n + 1}:00Z"),
                _load10(tmp / f"ka-run{n}.json", ep, started_utc=f"2026-10-23T09:0{n + 1}:05Z"))
        files += pair
        over = {} if n == 0 else {"change_at": starts[n - 1]}
        entries.append(_entry("single", ep, replica, pair, second_period=n > 1, **over))
    return {"out": out, "proofs": files, "entries": entries}


def test_a_copy_in_three_periods_has_one_entry_per_later_move_and_is_labelled_by_their_number(world, ref):
    w = _runs_world(world)
    a1, b1, a2, b2, a3 = w["entries"]
    st = _stores(world, [w["out"]], w["proofs"], w["entries"])
    three, two = "copy-a (three periods)", "copy-b (two periods)"
    assert sorted(st["copies"]) == [three, two]
    assert st["shares"][ARM] == _rows({three: 3, two: 1})
    assert [(p["period"], p.get("change_at")) for p in st["copies"][three]["periods"]] == [
        (1, None), (2, "2026-10-23T09:30:00Z"), (3, "2026-10-23T09:50:00Z")]
    assert [(p["period"], p.get("change_at")) for p in st["copies"][two]["periods"]] == [
        (1, "2026-10-23T09:20:00Z"), (2, "2026-10-23T09:40:00Z")]
    assert [("second_period" in p) for p in st["copies"][three]["proofs"]] == [False, True, True]
    assert st["config"]["matched"] == 5
    # a's third period is missing: the pair that is there is not enough, and it is not paired by position
    _refused_stores(world, [w["out"]], w["proofs"], [a1, b1, a2, b2], rf"{_where()}: a later period of the copy without its entry")
    # one entry too many
    extra = _entry("single", E0, "copy-a", (_storecheck(world["tmp"] / "sc-run5.json", E0, "copy-a", checked_utc="2026-10-23T09:06:00Z"),
                                            _load10(world["tmp"] / "ka-run5.json", E0, started_utc="2026-10-23T09:06:05Z")),
                   second_period=True, change_at="2026-10-23T09:55:00Z")
    _refused_stores(world, [w["out"]], [*w["proofs"], world["tmp"] / "sc-run5.json", world["tmp"] / "ka-run5.json"],
                    [*w["entries"], extra], rf"{_where()}: more second_period entries than the OUT has later periods")
    # each later period is held to its own change: a's third period a millisecond late refuses it alone. The entries
    # are a set, paired with the changes by their instants, so two entries swapping their instants stay the same set
    with pytest.raises(ex.Refused) as caught:
        _check(world, w["out"], w["proofs"], [a1, b1, a2, b2, {**a3, "change_at": "2026-10-23T09:50:00.001Z"}])
    assert str(caught.value).count("its change_at is not the first record on the copy") == 1
    assert _check(world, w["out"], w["proofs"], [a1, b1, {**a2, "change_at": "2026-10-23T09:50:00Z"}, b2,
                                                 {**a3, "change_at": "2026-10-23T09:30:00Z"}]).labels


def test_a_copy_moved_to_once_is_not_refused_by_that_rule_a_to_b(world, ref):
    m = _moved_world(world)
    assert _check(world, m["out"], m["proofs"], m["entries"]).timelines[str(m["out"])].clean


def test_a_copy_moved_to_once_is_not_refused_by_that_rule_a_to_b_to_a(world, ref):
    # Updated in the merge with the two-period rule: T1's version let the returning a carry the return's change_at in
    # its one entry; a's return now has its second_period entry, and the copy b, moved to once, still needs its
    # change_at
    w = _aba_world(world)
    assert _check(world, w["out"], w["proofs"], w["entries"]).timelines[str(w["out"])].clean
    late = {k: v for k, v in w["entries"][1].items() if k != "change_at"}
    _refused_check(world, w["out"], w["proofs"], [w["entries"][0], late, w["entries"][2]],
                   r"\(copy-b\): .*without its change_at")


@pytest.mark.parametrize("value", [{}, "", 0, False, 0.0])
def test_an_explicit_falsey_store_changes_that_is_not_a_list_is_malformed(value):
    changes, problems = ex.manifest_store_changes(ex.Out(Path("OUT"), None, {"store_changes": value}))
    assert changes == [] and len(problems) == 1 and "its manifest's store_changes is not a list" in problems[0]


@pytest.mark.parametrize("manifest", [{}, {"store_changes": None}, {"store_changes": []}])
def test_a_missing_null_or_empty_store_changes_means_no_changes(manifest):
    assert ex.manifest_store_changes(ex.Out(Path("OUT"), None, manifest)) == ([], [])


@pytest.mark.parametrize("value", [{}, "", 0, False])
def test_the_command_and_the_call_refuse_an_explicit_falsey_store_changes_that_is_not_a_list(world, capsys, ref, value):
    out = world["single"]
    _edit_manifest(out, lambda man: man.update(store_changes=value))
    code, (obj,), _raw, err = _cli(capsys, out)
    assert (code, set(obj), obj["verdict"]) == (2, {"out", "manifest_sha256", "manifest_store_changes_malformed",
                                                     "verdict"}, "STOP: manifest_store_changes_malformed")
    assert err.startswith("STOP manifest_store_changes_malformed:")
    _refused_stores(world, [out], (), [], r"its manifest's store_changes is not a list")  # one endpoint, still refused


@pytest.mark.parametrize("drop", ["null", "empty", "missing"])
def test_the_command_and_the_call_take_null_empty_or_missing_store_changes_as_no_changes(world, capsys, ref, drop):
    out = world["single"]
    _edit_manifest(out, lambda man: man.pop("store_changes", None) if drop == "missing"
                   else man.update(store_changes=None if drop == "null" else []))
    code, (obj,), _raw, err = _cli(capsys, out)
    assert (code, err, obj) == (0, "", {"out": str(out), "store_changes": False, "several_endpoints": False,
                                        "branch_phase_start": FIXTURE_PHASE_BRANCH_AT})
    assert _stores(world, [out], (), [])["store_changes"] == []  # runs as before


def test_a_timeline_refusal_of_the_call_carries_no_count(world, ref):
    m = _moved_world(world)
    f, record = _record_of(m["out"], "s01-pass")  # a record of copy-a made after the first on copy-b: a mix
    record["started_at"], record["ended_at"] = "2026-10-23T09:20:00.000+00:00", "2026-10-23T09:20:00.500+00:00"
    f.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ex.Refused) as caught:
        _check(world, m["out"], m["proofs"], m["entries"])
    text = str(caught.value)
    assert text.startswith("store copies not listed or not proven (") and "(interleaved)" in text
    assert not re.search(r"\b[0-9]+ store copies", text)


def test_the_command_says_stop_for_one_run_id_in_two_runs_and_not_unreadable(world, capsys):
    m = _moved_world(world)
    run_id = _twin_run_id(m["out"])
    code, (obj,), raw, err = _cli(capsys, m["out"])
    assert code == 2 and set(obj) == CLI_KEYS
    assert (obj["verdict"], obj["run_id_in_two_runs"], obj["clean"]) == ("STOP: run_id_in_two_runs", True, False)
    assert (obj["interleaved"], obj["overlap"], obj["chain_incomplete"], obj["unusable_times"]) == (False,) * 4
    assert err == "STOP run_id_in_two_runs: one run id names records in different runs of the endpoints\n"
    assert run_id not in raw and _ints(obj) == [".changes[].index"]


# --- the runs listing of the command (a diagnostic for a STOP): the run sequence in start order, no counts ---------------


def test_the_runs_of_a_clean_move_name_the_rows_that_attain_each_extreme():
    rows = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/a2.json", A, _at(2), _at(3.5)),
            _row("records/a3.json", A, _at(1), _at(4)),  # starts between a1 and a2 but ends last of A
            _row("records/b1.json", B, _at(10), _at(11)), _row("infrastructure/s1.json", B, _at(12), _at(20))]
    line = ex.store_timeline(rows, [_change(1, A, B)], "records")
    assert line.clean
    assert line.runs == (ex.RunSummary(A, _at(0), _at(2), _at(4), True, False),
                         ex.RunSummary(B, _at(10), _at(12), _at(20), True, True))


def test_the_runs_of_a_b_a_show_the_late_rows_of_the_third_run_and_where_they_are_kept():
    base = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/b1.json", B, _at(10), _at(11))]
    in_records = ex.store_timeline([*base, _row("records/a2.json", A, _at(20), _at(21))], [_change(1, A, B)], "records")
    in_setaside = ex.store_timeline([*base, _row("aborted/a2.json", A, _at(20), _at(21))], [_change(1, A, B)], "records")
    for line, kept in ((in_records, (True, False)), (in_setaside, (False, True))):
        assert line.interleaved is True and not line.clean
        assert line.runs == (ex.RunSummary(A, _at(0), _at(0), _at(1), True, False),
                             ex.RunSummary(B, _at(10), _at(10), _at(11), True, False),
                             ex.RunSummary(A, _at(20), _at(20), _at(21), *kept))


def test_the_runs_take_the_text_of_the_row_that_attains_each_extreme_in_start_then_name_order():
    """Three rows of A start at one instant in three spellings; a1 and a2 end at one instant in two spellings."""
    s1, s2, s3 = "2026-10-23T09:00:00.000+00:00", "2026-10-23T09:00:00Z", "2026-10-23T09:00:00.000000+00:00"
    e1, e2, e3 = "2026-10-23T09:00:03.000+00:00", "2026-10-23T09:00:03Z", "2026-10-23T09:00:02.000+00:00"
    rows = [_row("records/a3.json", A, s3, e3), _row("records/a1.json", A, s1, e1), _row("records/a2.json", A, s2, e2),
            _row("records/b1.json", B, _at(10), _at(11))]
    line = ex.store_timeline(rows, [_change(1, A, B)], "records")
    # in (started_at, name) order a1, a2, a3: the first start is a1's text, the last start a3's; the latest end is
    # shared by a1 and a2, and the last of them in that order, a2, gives its text
    assert line.runs[0] == ex.RunSummary(A, s1, s3, e2, True, False)
    assert line.clean and line.runs[1] == ex.RunSummary(B, _at(10), _at(10), _at(11), True, False)


def test_the_runs_are_not_known_while_a_record_has_no_usable_time_and_empty_without_any_row():
    rows = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/z1.json", A, None, None)]
    assert ex.store_timeline(rows, [_change(1, A, B)], "records").runs is None
    assert ex.store_timeline([], [_change(1, A, B)], "records").runs == ()


@pytest.mark.parametrize("late", ["records", "setaside"])
def test_the_command_lists_the_runs_of_a_mix_with_the_late_rows_of_the_third(world, capsys, late):
    m = _moved_world(world)
    out = m["out"]
    a_files = [f for f in _record_files(out) if json.loads(f.read_text(encoding="utf-8"))["store"]["endpoint"] == E0]
    if late == "records":  # the last of copy-a's 20 records (09:10:19) made at 09:20:00 instead, after copy-b's
        _retime(out, lambda r, f: "2026-10-23T09:20:00.000+00:00" if f == a_files[-1] else None)
        first_a_last, first_a_end = "2026-10-23T09:10:18.000+00:00", "2026-10-23T09:10:18.500+00:00"
        third = {"first_started": "2026-10-23T09:20:00.000+00:00", "last_started": "2026-10-23T09:20:00.000+00:00",
                 "last_ended": "2026-10-23T09:20:00.500+00:00", "in_records_dir": True, "in_setaside": False}
    else:  # a set-aside try on copy-a at 09:25:00
        _setaside(out, _record_of(out, "s01-pass")[1], "aaaaaaaa-0000-0000-0000-000000000009", E0,
                  "2026-10-23T09:25:00.000+00:00")
        first_a_last, first_a_end = "2026-10-23T09:10:19.000+00:00", "2026-10-23T09:10:19.500+00:00"
        third = {"first_started": "2026-10-23T09:25:00.000+00:00", "last_started": "2026-10-23T09:25:00.000+00:00",
                 "last_ended": "2026-10-23T09:25:00.500+00:00", "in_records_dir": False, "in_setaside": True}
    code, (obj,), raw, err = _cli(capsys, out)
    label_a, label_b = f"endpoint sha256:{_h(E0)[:12]}", f"endpoint sha256:{_h(E1)[:12]}"
    assert (code, obj["verdict"], obj["interleaved"]) == (2, "STOP: interleaved", True)
    assert obj["runs"] == [
        {"endpoint": label_a, "first_started": "2026-10-23T09:10:00.000+00:00", "last_started": first_a_last,
         "last_ended": first_a_end, "in_records_dir": True, "in_setaside": False},
        {"endpoint": label_b, "first_started": "2026-10-23T09:18:45.000+00:00",
         "last_started": "2026-10-23T09:19:00.000+00:00", "last_ended": "2026-10-23T09:19:00.500+00:00",
         "in_records_dir": True, "in_setaside": False},
        {"endpoint": label_a, **third}]
    assert set(obj) == CLI_KEYS and all(set(r) == CLI_RUN_KEYS for r in obj["runs"])
    assert _ints(obj) == [".changes[].index"] and "invalid" not in raw  # no count, no host
    assert not [r for r in m["before"] if r in raw] and "aaaaaaaa" not in raw  # no run id


# --- the check-stores command: the call's store check alone, its verdict as one JSON object --------------------------

CHECK_KEYS = {"ok", "copies", "outs_with_store_changes"}
CHECK_COPY_KEYS = {"endpoints", "outs", "periods"}
STORE_REFUSAL = ("store copies not listed or not proven (several store endpoints per graph and model are analysed only "
                 "listed in their OUT and identity-proven): ")


@pytest.fixture
def s01_graph(monkeypatch):
    """The fixtures' graph as a graph a call may name (a real call's graph is one of GRAPHS or the frozen config's)."""
    monkeypatch.setitem(ex.GRAPHS, "s01", {"store": "s01", "questions_sha256": "0" * 64, "map": "singletons"})


def _check_cli(capsys, world, outs, files, entries, *, graph: str = "s01", root: Path = ROOT):
    """check_stores_main on these OUTs with a store-proofs config of the entries (and the fixtures' copy classes): (exit
    code, the one JSON object, the raw stdout, stderr)."""
    config = _store_config(world["tmp"] / "check-cli.json", *entries)
    code = ex.check_stores_main(root, graph, outs, tuple(files), config,
                                _classes_file(world["tmp"] / "classes.json"))
    seen = capsys.readouterr()
    return code, json.loads(seen.out), seen.out, seen.err


def test_the_check_stores_command_prints_the_copies_of_a_clean_move_by_labels_alone(world, ref, s01_graph, capsys):
    m = _moved_world(world)
    code, obj, raw, err = _check_cli(capsys, world, [m["out"]], m["proofs"], m["entries"])
    assert (code, err) == (0, "")
    # copy-a is the chain's start (its period has no change_at), copy-b began at the change
    assert obj == {"ok": True, "outs_with_store_changes": ["single"],
                   "copies": {"copy-a": {"endpoints": [f"endpoint sha256:{_h(E0)[:12]}"], "outs": ["single"],
                                         "periods": [None]},
                              "copy-b": {"endpoints": [f"endpoint sha256:{_h(E1)[:12]}"], "outs": ["single"],
                                         "periods": [CHANGE_AT]}}}
    assert set(obj) == CHECK_KEYS and all(set(c) == CHECK_COPY_KEYS for c in obj["copies"].values())
    assert _ints(obj) == [] and E0 not in raw and E1 not in raw and "invalid" not in raw  # no integer, no host
    assert str(m["out"]) not in raw  # the directory name only


def test_the_check_stores_command_lists_a_copy_in_two_periods_by_its_instants_and_refuses_it_without_its_entry(
        world, ref, s01_graph, capsys):
    w = _aba_world(world)
    code, obj, raw, err = _check_cli(capsys, world, [w["out"]], w["proofs"], w["entries"])
    assert (code, err) == (0, "")
    assert obj["copies"][A_TWO] == {"endpoints": [f"endpoint sha256:{_h(E0)[:12]}"], "outs": ["single"],
                                    "periods": [None, RETURN_AT]}  # one element per period: the chain start, the return
    assert obj["copies"]["copy-b"]["periods"] == ["2026-10-23T09:18:45Z"] and sorted(obj["copies"]) == [A_TWO, "copy-b"]
    assert set(obj) == CHECK_KEYS and obj["outs_with_store_changes"] == ["single"]
    assert _ints(obj) == []  # instants and nulls only: no integer anywhere
    assert E0 not in raw and E1 not in raw and _sha(w["a"][0]) not in raw  # no host, no file hash
    code, obj, raw, err = _check_cli(capsys, world, [w["out"]], w["proofs"], w["entries"][:2])
    assert code == 2 and set(obj) == {"ok", "refused"} and obj["ok"] is False
    assert obj["refused"] == (STORE_REFUSAL + f"{w['out']} endpoint sha256:{_h(E0)[:12]}: a later period of the copy "
                              f"without its entry (a second_period entry for each return to it)")
    assert err == ""


def test_the_check_stores_command_keeps_outs_that_disagree_on_a_labels_periods_apart():
    def stores(**starts):
        labels = {(f"/x/{name}", E0): "copy-a" for name in starts}
        return ex.StoreCopies({}, {}, labels, {}, [], {}, period_starts={(f"/x/{n}", E0): v for n, v in starts.items()})

    same = ex._check_stores_object(stores(one=[None], two=[None]))
    assert same["copies"] == {"copy-a": {"endpoints": [f"endpoint sha256:{_h(E0)[:12]}"], "outs": ["one", "two"],
                                         "periods": [None]}}
    apart = ex._check_stores_object(stores(one=[None], two=[CHANGE_AT]))
    label = f"endpoint sha256:{_h(E0)[:12]}"
    assert apart["copies"] == {
        f"copy-a [one, {label}]": {"endpoints": [label], "outs": ["one"], "periods": [None]},
        f"copy-a [two, {label}]": {"endpoints": [label], "outs": ["two"], "periods": [CHANGE_AT]}}


def test_the_check_stores_command_refuses_a_mix_of_copies_with_the_timeline_message(world, ref, s01_graph, capsys):
    m = _moved_world(world)
    f = next(p for p in (m["out"] / "records").glob("*.json") if p.name.split(".")[1] == m["before"][0])
    r = json.loads(f.read_text(encoding="utf-8"))
    r["started_at"], r["ended_at"] = "2026-10-23T09:20:00.000+00:00", "2026-10-23T09:20:00.500+00:00"
    f.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")  # an old-copy record after the first on the new copy
    _recheck(m["out"])
    code, obj, raw, err = _check_cli(capsys, world, [m["out"]], m["proofs"], m["entries"])
    assert code == 2 and obj["ok"] is False and set(obj) == {"ok", "refused"}
    assert obj["refused"].startswith(STORE_REFUSAL + f"{m['out']}: its records' own endpoints and times do not follow "
                                                     f"its manifest store changes (interleaved): ")
    assert "a mix of copies, not a late timestamp" in obj["refused"]
    assert not any(run_id in obj["refused"] for run_id in m["before"])  # no run id in the message


def test_the_check_stores_command_refuses_a_graph_without_a_recorded_reference(world, s01_graph, capsys):
    m = _moved_world(world)  # no ``ref``: s01 is not among the pinned graphs
    code, obj, raw, err = _check_cli(capsys, world, [m["out"]], m["proofs"], m["entries"])
    assert code == 2 and obj["ok"] is False
    assert (f"{m['out']} endpoint sha256:{_h(E0)[:12]} (copy-a): s01 has no recorded reference stats here "
            f"(REFERENCE_STATS_SHA256, or its private pin)") in obj["refused"]


def test_the_check_stores_command_validates_the_graph_as_analyse_does(world, ref, capsys):
    # s01 is no graph of GRAPHS and not the frozen config's (here, without the s01_graph fixture): refused as analyse's
    # cluster_map_for refuses it, with a single endpoint and no entry, where check_stores alone would pass
    out = world["single"]
    cfg_graph = ex.frozen_config(F)[0].graph
    code, obj, raw, err = _check_cli(capsys, world, [out], (), [])
    assert code == 2 and obj == {"ok": False, "refused": (
        f"s01: no cluster map (graphs {sorted(ex.GRAPHS)} and the frozen config's {cfg_graph}); another graph's map is "
        f"science's decision, never singletons by default")}
    with pytest.raises(ex.Refused) as same:  # the very refusal analyse gives
        ex.cluster_map_for(F, ex.frozen_config(F)[0], "s01")
    assert obj["refused"] == str(same.value)
    code, obj, raw, err = _check_cli(capsys, world, [out], (), [], graph=cfg_graph)  # the frozen config's graph passes
    assert (code, obj["ok"]) == (0, True)


def test_the_check_stores_command_names_the_store_of_the_graph_as_a_call_does(world, ref, capsys):
    # SIB's records are on the UniProt store, which is what a call checks the entries against
    m = _moved_world(world)
    code, obj, raw, err = _check_cli(capsys, world, [m["out"]], m["proofs"], m["entries"],
                                     graph="sib-uniprot-sp-2026_03")
    assert code == 2 and "(copy-a): an entry of graph s01, not the call's uniprot-sp-2026_03" in obj["refused"]


def test_the_check_stores_command_refuses_the_privately_pinned_graph_and_takes_no_pin(world, ref, capsys):
    m = _moved_world(world)
    code, obj, raw, err = _check_cli(capsys, world, [m["out"]], m["proofs"], m["entries"], graph="a15")
    assert code == 2 and obj == {"ok": False, "refused": (
        "a15: check-stores does not run for a privately pinned graph (a15): its reference pin and its records stay in "
        "the private tree")} and err == ""
    # the command has no --reference-pin: the private graph's pin is not its business
    with pytest.raises(SystemExit) as caught:
        ex.main(["check-stores", "--code-root", str(ROOT), "--graph", "a15", "--batch", str(m["out"]),
                 "--reference-pin", str(world["tmp"] / "pin.json")])
    assert caught.value.code == 2 and "unrecognized arguments: --reference-pin" in capsys.readouterr().err


def test_the_check_stores_command_exits_3_with_the_class_alone_for_an_unreadable_out_and_a_bad_code_root(
        world, ref, s01_graph, capsys, tmp_path):
    m = _moved_world(world)
    # an OUT that cannot be read, and a code root that is not the frozen analysis: the exception class alone
    code, obj, raw, err = _check_cli(capsys, world, [tmp_path / "no-such-out"], (), [])
    assert (code, obj) == (3, {"ok": False, "error": "FileNotFoundError"})
    assert err == "check-stores: failed (FileNotFoundError)\n" and str(tmp_path) not in raw
    code, obj, raw, err = _check_cli(capsys, world, [m["out"]], (), [], root=tmp_path)
    assert (code, obj) == (3, {"ok": False, "error": "Refused"}) and err == "check-stores: failed (Refused)\n"


def test_a_refusal_names_a_manifest_store_change_by_its_position_and_prints_no_manifest_string(world, ref, s01_graph,
                                                                                               capsys):
    SECRET = "run-abc-123"  # a run-id-like string in the manifest, which no refusal may carry
    m = _moved_world(world)
    out = m["out"]

    def mangle(manifest):
        change = manifest["store_changes"][0]
        change["at"] = change["note"] = SECRET
        change["pre_change_run_ids"] = [SECRET, 5]  # not run ids of its pre_change_records
        change["from"] = {"endpoint": 5, "store": {"endpoint": [SECRET]}}  # a side that names no endpoint

    _edit_manifest(out, mangle)
    code, obj, raw, err = _check_cli(capsys, world, [out], m["proofs"], m["entries"])
    assert code == 2 and SECRET not in raw + err
    assert f"{out}: its manifest store change 1: its from side has no endpoint, or more than one (endpoint and " \
           f"store.endpoint must name one)" in obj["refused"]
    assert f"{out}: its manifest store change 1: its pre_change_run_ids are not its pre_change_records records' run ids" \
        in obj["refused"]


def test_a_chain_that_does_not_join_names_its_changes_by_position_too(world, ref, s01_graph, capsys):
    SECRET = "run-abc-123"
    w = _aba_world(world)
    out = w["out"]

    def mangle(manifest):
        second = manifest["store_changes"][1]
        second["at"] = SECRET
        second["from"] = _side(out, EX)  # starts at another endpoint than the first moved to

    _edit_manifest(out, mangle)
    code, obj, raw, err = _check_cli(capsys, world, [out], w["proofs"], w["entries"])
    assert code == 2 and SECRET not in raw + err
    assert (f"{out}: its manifest store change 2 does not start where change 1 moved to, with change 1's records "
            f"among those made before it") in obj["refused"]


def test_a_failure_in_formatting_the_verdict_is_an_exit_3_with_the_class_alone(world, ref, s01_graph, capsys, monkeypatch):
    m = _moved_world(world)

    def boom(stores):
        raise TypeError("not a thing to sort: /private/path 7")

    monkeypatch.setattr(ex, "_check_stores_object", boom)
    code, obj, raw, err = _check_cli(capsys, world, [m["out"]], m["proofs"], m["entries"])
    assert (code, obj) == (3, {"ok": False, "error": "TypeError"})
    assert raw == '{"error": "TypeError", "ok": false}\n' and err == "check-stores: failed (TypeError)\n"
    assert "Traceback" not in raw + err and "/private/path" not in raw + err  # the class, not the message


def test_two_labels_of_one_call_that_cannot_be_sorted_are_an_exit_3_not_a_traceback(world, ref, s01_graph, capsys,
                                                                                     tmp_path):
    # two OUTs of one endpoint whose identities name the replica by a number and by a string: the labels cannot be
    # ordered. No proof is needed for one endpoint, so check_stores passes and the formatting fails
    one = world["single"]
    two = tmp_path / "single-two"
    shutil.copytree(one, two)
    _edit_identity(one, lambda i: i["store"].update(endpoint=E0, replica=7))
    _edit_identity(two, lambda i: i["store"].update(endpoint=E0, replica="x"))
    code, obj, raw, err = _check_cli(capsys, world, [one, two], (), [])
    assert (code, obj) == (3, {"ok": False, "error": "TypeError"}) and "Traceback" not in raw + err


def test_the_check_stores_command_is_a_subcommand_with_the_analyse_arguments(world, ref, s01_graph, capsys):
    out = world["single"]  # one endpoint, no store change: the committed config is read and nothing needs a proof
    code = ex.main(["check-stores", "--code-root", str(ROOT), "--graph", "s01", "--batch", str(out)])
    seen = capsys.readouterr()
    obj = json.loads(seen.out)
    assert (code, seen.err) == (0, "") and obj == {
        "ok": True, "outs_with_store_changes": [],
        "copies": {f"endpoint sha256:{_h(E0)[:12]}": {"endpoints": [f"endpoint sha256:{_h(E0)[:12]}"],
                                                       "outs": ["single"], "periods": [None]}}}
    m = _moved_world(world)  # two endpoints and no entry of the committed config for this OUT: refused, exit 2
    code = ex.main(["check-stores", "--code-root", str(ROOT), "--graph", "s01", "--batch", str(m["out"]),
                    "--store-proof", str(m["proofs"][0])])
    obj = json.loads(capsys.readouterr().out)
    assert code == 2 and obj["ok"] is False and "no store-proofs.json entry proves its copy" in obj["refused"]


def test_the_number_audit_of_store_refusals_rejects_a_number_and_allows_the_listed_names():
    for number in ("its forward record names 2 mappings", "unreadable record files: 3", "entry 4 lacks one of x"):
        with pytest.raises(AssertionError, match="a number in a store refusal"):
            _no_number_in(number)
    _no_number_in(f"/tmp/pytest-of-x/pytest-12/test_a0/single endpoint sha256:{'ab' * 6} (copy-b2): s01 has no "
                  f"recorded reference stats here (REFERENCE_STATS_SHA256) | unreadable record files "
                  f"['records/ab.cd.json'] at 2026-10-23T09:18:45.000+00:00 of rhea-142 with entry180-store-check/2")


# --- the endpoint copies (analysis lead 4, 25 Oct ~08:00Z): the copy behind an endpoint no proof entry covers -------------


def _endpoint_table(path: Path, *rows: tuple[str, str, str]) -> Path:
    """An endpoint-copies file written by hand: rows of (graph, endpoint, replica)."""
    doc = {"schema": "e189-endpoint-copies/1", "rule": "a test", "copies": [
        {"graph": g, "endpoint_sha256": _h(ep), "replica": r, "basis": "a test",
         "sources": [{"path": "/identity.json", "sha256": "ab" * 32}]} for g, ep, r in rows]}
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return path


def _without_table_record(stores: dict) -> dict:
    """The store section with the endpoint-copies file's record (path and sha256) taken out."""
    out = json.loads(json.dumps(stores))
    del out["config"]["endpoint_copies"]
    return out


def test_a_single_endpoint_out_with_no_replica_is_named_by_the_endpoint_copies_and_gets_that_replicas_class(world, ref):
    out = str(world["single"])
    named = _stores(world, [world["single"]], (), [], endpoint_copies=_endpoint_table(world["tmp"] / "t.json", ("s01", E0, "copy-a")))
    other = _stores(world, [world["single"]], (), [], endpoint_copies=_endpoint_table(world["tmp"] / "t2.json", ("s01", EX, "copy-a")))
    label = ex.endpoint_label(E0)
    assert label.startswith("endpoint sha256:") and set(other["copies"]) == {label}  # not in the table: as before
    assert set(named["copies"]) == {"copy-a"}  # in the table: the replica
    assert named["copies"]["copy-a"] == {"endpoints": [label], "outs": [out], "listed_by": {out: []}, "proofs": []}
    for arm, share in named["shares"].items():  # every row of every arm is on copy-a, and its class is copy-a's
        k = share["attempts"]
        assert share["rows"] == {"copy-a": {"n": k, "of": k, "share": 1.0}} and other["shares"][arm]["rows"].keys() == {label}
        assert named["class_shares"][arm]["rows"] == {"fast": {"n": k, "of": k, "share": 1.0}}
        assert other["class_shares"][arm]["rows"].keys() == {"no copy class (no entry in the copy classes)"}
    assert named["config"]["endpoint_copies"] == {"path": str(world["tmp"] / "t.json"), "sha256": _sha(world["tmp"] / "t.json")}
    assert E0 not in json.dumps(named)  # the endpoint itself is named nowhere


def test_the_endpoint_copies_change_nothing_but_the_labels_and_classes_of_an_unproven_single_endpoint_out(world, ref):
    named = _stores(world, [world["single"]], (), [], endpoint_copies=_endpoint_table(world["tmp"] / "t.json", ("s01", E0, "copy-a")))
    other = _stores(world, [world["single"]], (), [], endpoint_copies=_endpoint_table(world["tmp"] / "t2.json", ("s01", EX, "copy-a")))

    def relabel(x):  # OLD's label as the replica, class rows aside: by hand, the one place a label is written
        if isinstance(x, dict):
            return {("copy-a" if k == ex.endpoint_label(E0) else k): relabel(v) for k, v in x.items()}
        if isinstance(x, list):
            return [relabel(v) if not isinstance(v, str) else v for v in x]
        return x

    a, b = relabel(_without_table_record(other)), _without_table_record(named)
    a["config"].pop("copy_classes"), b["config"].pop("copy_classes")
    for section in ("class_shares", "class_shares_by_out"):  # class-labelled rows: the totals only
        for d in (a, b):
            d[section] = json.loads(json.dumps(d[section]))
    a["copies"]["copy-a"]["endpoints"] = [ex.endpoint_label(E0)]  # (the endpoint labels stay endpoint labels)
    assert {k: v for k, v in a.items() if not k.startswith("class_shares")} == {
        k: v for k, v in b.items() if not k.startswith("class_shares")}


def test_the_endpoint_copies_are_never_a_proof_an_unproven_multi_endpoint_out_is_still_refused(world, ref):
    m = _moved_world(world)
    both = _endpoint_table(world["tmp"] / "t.json", ("s01", E0, "copy-a"), ("s01", E1, "copy-b"))
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"][:1], r"no store-proofs.json entry proves its copy",
                    endpoint_copies=both)  # E1 has a table entry and no proof entry
    _refused_stores(world, [m["out"]], (), [], r"no store-proofs.json entry proves its copy", endpoint_copies=both)


def test_with_its_proofs_a_moved_out_is_the_same_whether_or_not_the_table_names_its_copies(world, ref):
    m = _moved_world(world)
    plain = _stores(world, [m["out"]], m["proofs"], m["entries"], endpoint_copies=_endpoint_table(world["tmp"] / "o.json", ("s01", EX, "copy-x")))
    named = _stores(world, [m["out"]], m["proofs"], m["entries"],
                    endpoint_copies=_endpoint_table(world["tmp"] / "t.json", ("s01", E0, "copy-a"), ("s01", E1, "copy-b")))
    assert _without_table_record(plain) == _without_table_record(named)  # shares, copies, store changes, order check: all


def test_an_endpoint_copy_that_names_another_replica_than_a_proof_entry_refuses(world, ref):
    m = _moved_world(world)
    wrong = _endpoint_table(world["tmp"] / "t.json", ("s01", E0, "copy-c"))
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"], r"the endpoint copies disagree with the store proofs: "
                    r"the endpoint copies name copy-c for endpoint sha256:[0-9a-f]{12} of s01, the store proofs copy-a",
                    endpoint_copies=wrong)
    moved = _endpoint_table(world["tmp"] / "t2.json", ("s01", E1, "copy-a"))  # copy-a is proven at E0, not at E1
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"], r"the store proofs for another endpoint or graph",
                    endpoint_copies=moved)


def test_an_endpoint_copy_that_names_another_replica_than_the_outs_identity_refuses_and_an_agreeing_one_is_accepted(world, ref):
    out = world["single"]
    ident = _identity(out)
    ident["store"].update(endpoint=E0, replica="copy-c", identity_check="checked")
    _write_identity(out, ident)
    plain = _stores(world, [out], (), [], endpoint_copies=_endpoint_table(world["tmp"] / "o.json", ("s01", EX, "copy-x")))
    assert set(plain["copies"]) == {"copy-c"}  # the identity's replica labels the endpoint without a table entry (as before)
    agree = _stores(world, [out], (), [], endpoint_copies=_endpoint_table(world["tmp"] / "t.json", ("s01", E0, "copy-c")))
    assert set(agree["copies"]) == {"copy-c"} and agree["copies"] == plain["copies"]
    _refused_stores(world, [out], (), [], r"its deployment identity names copy-c, the endpoint copies copy-a",
                    endpoint_copies=_endpoint_table(world["tmp"] / "w.json", ("s01", E0, "copy-a")))


def test_an_endpoint_copy_whose_replica_has_no_class_entry_refuses(world, ref):
    _refused_stores(world, [world["single"]], (), [], r"the endpoint copies name copy-nowhere, which has no entry in the "
                    r"copy classes", endpoint_copies=_endpoint_table(world["tmp"] / "t.json", ("s01", E0, "copy-nowhere")))


def test_an_endpoint_copy_the_identity_names_too_still_needs_its_class_entry(world, ref):
    """external reviewer r1: the missing-class refusal held only where the identity named no replica."""
    out = world["single"]
    ident = _identity(out)
    ident["store"].update(endpoint=E0, replica="copy-nowhere", identity_check="checked")
    _write_identity(out, ident)
    plain = _stores(world, [out], (), [], endpoint_copies=_endpoint_table(world["tmp"] / "o.json", ("s01", EX, "copy-x")))
    assert set(plain["copies"]) == {"copy-nowhere"}  # no table entry: the identity's replica, classless, as before
    assert all(set(v["rows"]) == {"no copy class (no entry in the copy classes)"} for v in plain["class_shares"].values())
    _refused_stores(world, [out], (), [], r"the endpoint copies name copy-nowhere, which has no entry in the copy classes",
                    endpoint_copies=_endpoint_table(world["tmp"] / "t.json", ("s01", E0, "copy-nowhere")))


def test_an_endpoint_copy_that_agrees_with_a_proof_entry_still_needs_its_class_entry(world, ref):
    """external reviewer r2: the missing-class refusal also holds where the endpoint has a proof entry (the proven path)."""
    m = _moved_world(world)
    only_b = world["tmp"] / "only-b.json"
    only_b.write_text(json.dumps({"schema": "e189-copy-classes/1", "rule": "a test", "sources": {
        "src": {"path": "/src.json", "sha256": "ab" * 32}}, "copies": {"copy-b": CLASS_B}}), encoding="utf-8")
    agree = _endpoint_table(world["tmp"] / "t.json", ("s01", E0, "copy-a"))  # copy-a is proven at E0, the table agrees
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"], r"the endpoint copies name copy-a, which has no entry in "
                    r"the copy classes", endpoint_copies=agree, copy_classes=only_b)
    # without the table the proven moved OUT needs no class entry of copy-a outside a branch phase (as before)
    other = _endpoint_table(world["tmp"] / "o.json", ("s01", EX, "copy-x"))
    assert _stores(world, [m["out"]], m["proofs"], m["entries"], endpoint_copies=other, copy_classes=only_b)["copies"]


def test_an_endpoint_copy_of_another_graph_names_nothing_here(world, ref):
    other = _stores(world, [world["single"]], (), [],
                    endpoint_copies=_endpoint_table(world["tmp"] / "t.json", ("another-graph", E0, "copy-a")))
    assert set(other["copies"]) == {ex.endpoint_label(E0)}


def test_a_malformed_endpoint_copies_file_refuses_the_call(world, ref):
    bad = world["tmp"] / "bad.json"
    bad.write_text(json.dumps({"schema": "e189-endpoint-copies/1", "rule": "x", "copies": [{"graph": "s01"}]}), encoding="utf-8")
    _refused_stores(world, [world["single"]], (), [], r"bad.json: an endpoint copy is not graph", endpoint_copies=bad)


def test_the_conflicts_between_the_endpoint_copies_and_the_store_proofs_by_hand():
    def p(replica, endpoint, graph="g", **more):
        return {"graph": graph, "endpoint_sha256": endpoint, "replica": replica, **more}

    table = [{"graph": "g", "endpoint_sha256": "1" * 64, "replica": "r1"}]
    assert ex.endpoint_copy_conflicts(table, []) == []
    assert ex.endpoint_copy_conflicts(table, [p("r1", "1" * 64), p("r1", "1" * 64, second_period=True)]) == []  # the same copy
    assert ex.endpoint_copy_conflicts(table, [p("r9", "9" * 64), p("r2", "1" * 64, graph="h")]) == []  # unrelated, other graph
    assert len(ex.endpoint_copy_conflicts(table, [p("r2", "1" * 64)])) == 1  # another replica at the endpoint
    assert len(ex.endpoint_copy_conflicts(table, [p("r1", "1" * 64), p("r2", "1" * 64, behind_endpoint=True)])) == 1  # a copy behind it
    assert len(ex.endpoint_copy_conflicts(table, [p("r1", "2" * 64)])) == 1  # r1 proven at another endpoint
    assert len(ex.endpoint_copy_conflicts(table, [p("r1", "1" * 64, graph="h")])) == 1  # r1 proven for another graph


def test_the_committed_endpoint_copies_agree_with_the_committed_store_proofs_and_have_their_classes():
    table, sha = ex.read_endpoint_copies(HERE.parent / "endpoint-copies.json")
    proofs, _ = ex.read_store_proofs(HERE.parent / "store-proofs.json")
    classes, _ = ex.read_copy_classes(HERE.parent / "copy-classes.json")
    assert ex.endpoint_copy_conflicts(table, proofs) == [] and sha == _sha(HERE.parent / "endpoint-copies.json")
    assert {e["replica"] for e in table} <= set(classes)


# --- the branch-only OUT (science, 25 Oct ~09:5xZ, option (a): change-log entry 150's UniProt OUTs) ----------------------
#
# Hand-derived facts. Such an OUT holds the model's Tier 1 parents copied by file, which ran on the old copy and are made
# before the OUT's store change, and its own branch rows, made after the change and the branch instant; every one of them
# records the parents' copy (online.py:542), so no record names the copy the OUT moved to. Its directory has
# order-branch.json and no other order file. The periods of its copies are the manifest's change instants: a branch row
# counts for the copy current at its start, the new copy at the instant itself. In the fixture world below every record
# names copy-a (E0). The records of s01-pass and s01-flag play the copied parents (ten records a question, all kinds of
# cell, by the fixture; a role given by question) and start from 09:10:00 a second apart, before the change
# (09:18:39.962); those of s01-abstain and s01-noquery play the branch rows and start from 09:18:45, after the branch
# instant (09:18:40). So the parents are copy-a's and the branch rows copy-b's, though all name copy-a.

BO_BRANCH = "2026-10-23T09:18:40.000+00:00"  # the branch instant: after the manifest's change, before the first branch row
BO_SECOND_AT = "2026-10-23T09:19:30.000+00:00"  # the second change of the three-copy world (copy-b to copy-x)
BO_PARENTS = BEFORE_Q  # s01-pass, s01-flag: the copied parents
LEFT_OUT = ("order-natural.json", "order-prompted.json")  # what a copied-parents builder says it left out, by hand
BO_RULE_TEXT = ("a branch-only OUT (a manifest branch phase and store changes, its builder's declaration of the Tier 1 "
                "parents it copied with the order files it left out, and order-branch.json the only order file of its "
                "directory) takes the periods of its copies from the instants of its manifest's store changes, because "
                "its records name its parents' copies alone: in place of the chain's last copy of the branch-phase "
                "rule, each branch row's live retry is attributed to the copy current at its start (a change's at "
                "begins its new copy's period, and a start at that instant belongs to the new copy); a branch row "
                "before the first change, a row made before the branch phase that records another copy than the one "
                "current at its start, a row made before the branch phase that is not a copied parent the first change "
                "lists, that is a second row of one, or that started at or after that change (or a listed parent "
                "without such a row), instants that are not strictly increasing, an attempt that runs across a change, "
                "one run id in two periods, or an identity whose store or store changes do not follow the chain's "
                "copies to its last copy refuses; every copy of the chain needs its proof and its class (science, "
                "25 Oct ~09:5xZ)")
BO_ATTRIBUTION_TEXT = ("a branch row records its parent attempt's copy (branch_attempt in harness/checkloop/online.py), "
                       "but its live retry ran on the copy current at its start by the manifest's store changes: the "
                       "row counts for that copy, and a checker arm's retried output whose parent attempt ran on "
                       "another copy is a unit row naming both copies, its retry attempt counted for the copy of its "
                       "start in the attempt shares")


def _questions_of(out: Path) -> dict[str, str]:
    """run id -> question id, read from the OUT's records/ files."""
    found = {}
    for f in sorted((out / "records").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        found[r["run_id"]] = r["question_id"]
    return found


def _branch_only_world(world, *, three: bool = False, ret: bool = False, late_from: str = "09:18:45") -> dict:
    """The fixture whole as a branch-only OUT: order-branch.json alone, every record naming E0, s01-pass and s01-flag
    from 09:10:00 (before the change), s01-abstain from 09:18:45 and s01-noquery from ``late_from`` (a second apart in
    file order), the manifest's change from E0 to E1 at MANIFEST_AT (09:18:39.962), the branch phase at BO_BRANCH, the
    identity's store the chain's last copy, and the proofs of every copy of the chain. ``three``: a second change from
    E1 to EX at BO_SECOND_AT (09:19:30) with ``late_from`` "09:19:30" by default for s01-noquery; the copies are
    copy-a, copy-b and copy-x. ``ret``: the second change goes back from E1 to E0 (copy-a in its second period, with its
    own checks made at 09:19:00) at BO_SECOND_AT. The manifest carries its builder's declaration of the copied parents
    (``entry150``: as many records as the first change lists, the source's branch phase and order files left out)."""
    three = three or ret
    out, tmp = world["single"], world["tmp"]
    (out / "order-natural.json").unlink()
    (out / "order-prompted.json").unlink()
    _set_endpoints(out, lambda r: E0)
    question = _questions_of(out)
    starts = {"s01-pass": "09:10:00", "s01-flag": "09:10:00", "s01-abstain": "09:18:45",
              "s01-noquery": late_from if not three or late_from != "09:18:45" else "09:19:30"}
    seen: Counter = Counter()

    def start_of(r: dict, f: Path):
        q = r["question_id"]
        seen[q] += 1
        h, m, s = map(int, starts[q].split(":"))
        return datetime(2026, 10, 23, h, m, s, tzinfo=UTC) + timedelta(seconds=seen[q] - 1)

    _retime(out, start_of)
    parents = sorted(i for i, q in question.items() if q in BO_PARENTS)
    middle = sorted(i for i, q in question.items() if q == "s01-abstain")
    _store_change(out, E0, E1, parents, at=MANIFEST_AT, stamp=False)
    if three:
        _store_change(out, E1, E0 if ret else EX, sorted([*parents, *middle]), at=BO_SECOND_AT, stamp=False)
    _set_branch_phase(out, BO_BRANCH)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["entry150"] = {"copied": {"records": len(parents)},
                            "left_out": {"branch_phase": None, "files": ["runs.jsonl", *LEFT_OUT]}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    _recheck(out)
    ident = json.loads((out / "deployment-identity.json").read_text(encoding="utf-8"))
    ident["store"]["endpoint"] = (E0 if ret else EX) if three else E1
    _write_identity(out, ident)
    a, b = _copy_proofs(tmp, "a", E0, "copy-a"), _copy_proofs(tmp, "b", E1, "copy-b")
    entries = [_entry("single", E0, "copy-a", a), _entry("single", E1, "copy-b", b, change_at=MANIFEST_AT)]
    proofs = [*a, *b]
    if ret:  # copy-a's second period: its own pair of checks, made before the return
        a2 = (_storecheck(tmp / "sc-a2.json", E0, "copy-a", checked_utc="2026-10-23T09:19:00Z"),
              _load10(tmp / "ka-a2.json", E0, started_utc="2026-10-23T09:19:05Z"))
        entries.append(_entry("single", E0, "copy-a", a2, change_at=BO_SECOND_AT, second_period=True))
        proofs += a2
    elif three:
        x = _copy_proofs(tmp, "x", EX, "copy-x")
        entries.append(_entry("single", EX, "copy-x", x, change_at=BO_SECOND_AT))
        proofs += x
    return {"out": out, "question": question, "parents": parents, "middle": middle, "proofs": tuple(proofs),
            "entries": entries, "a": a, "b": b}


def _bo_classes(world) -> Path:
    return _classes_file(world["tmp"] / "bo-classes.json", **{"copy-x": {**CLASS_C, "basis": "basis of copy-x"}})


DECLARED = {"copied": {"records": 0}, "left_out": {"branch_phase": None, "files": ["runs.jsonl", *LEFT_OUT]}}
LISTED = [{"at": "2026-10-23T09:00:02.500+00:00", "pre_change_records": 0, "pre_change_run_ids": []}]  # (change 1 lists none)


def test_a_branch_only_out_is_recognised_by_its_order_files_and_its_manifests_declaration_of_copied_parents(tmp_path):
    manifest = {"additions": [{"added": ["phase branch"], "at": BO_BRANCH, "routes": []}], "entry150": DECLARED,
                "store_changes": LISTED}
    change = [_change(1, A, B)]  # (it lists no record as made before it: the declaration's records are 0)
    branch = ex.BranchPhase(datetime(2026, 10, 23, 9, 18, 40, tzinfo=UTC), BO_BRANCH)

    def out_with(name: str, *files: str, mf=None) -> "ex.Out":
        d = tmp_path / name
        d.mkdir()
        for f in files:
            (d / f).write_text("{}", encoding="utf-8")
        return ex.Out(d, None, manifest if mf is None else mf)

    only = out_with("only", "order-branch.json", "manifest.json", "records.txt")
    assert ex.is_branch_only(only, branch, change) is True
    # without a branch phase, or without a store change, the OUT is not one
    assert ex.is_branch_only(only, None, change) is False
    assert ex.is_branch_only(only, branch, []) is False
    # any other entry starting with "order" (any case) makes it an OUT of the ordinary kind
    for extra in ("order-natural.json", "order-prompted.json", "order.json", "order-branch.json.bak", "Order-Plan.json",
                  "ORDER-NATURAL.JSON", "orders"):
        assert ex.is_branch_only(out_with("with-" + extra, "order-branch.json", extra), branch, change) is False, extra
    # no order file at all, or only another one, or order-branch.json that is not a regular file
    assert ex.is_branch_only(out_with("none", "manifest.json"), branch, change) is False
    assert ex.is_branch_only(out_with("natural", "order-natural.json"), branch, change) is False
    assert ex.is_branch_only(out_with("plain", "order.json"), branch, change) is False
    assert ex.is_branch_only(out_with("upper", "ORDER-BRANCH.JSON"), branch, change) is False
    linked = out_with("linked")
    (tmp_path / "target.json").write_text("{}", encoding="utf-8")
    (linked.path / "order-branch.json").symlink_to(tmp_path / "target.json")
    assert ex.is_branch_only(linked, branch, change) is False
    directory = out_with("directory")
    (directory.path / "order-branch.json").mkdir()
    assert ex.is_branch_only(directory, branch, change) is False
    # a directory that cannot be listed is judged as every other OUT
    gone = ex.Out(tmp_path / "does-not-exist", None, manifest)
    assert ex.is_branch_only(gone, branch, change) is False
    # the declaration is needed too: an ordinary OUT whose order files are gone has none, and every part of it counts
    wrong = {"no entry150": None, "entry150 null": {"entry150": None}, "entry150 a list": {"entry150": []},
             "entry150 a string": {"entry150": "copied"}, "no copied": {"entry150": {"left_out": DECLARED["left_out"]}},
             "copied a list": {"entry150": {**DECLARED, "copied": [0]}},
             "no left_out": {"entry150": {"copied": DECLARED["copied"]}},
             "left_out null": {"entry150": {**DECLARED, "left_out": None}},
             "records one more": {"entry150": {**DECLARED, "copied": {"records": 1}}},
             "records a bool": {"entry150": {**DECLARED, "copied": {"records": False}}},
             "records a float": {"entry150": {**DECLARED, "copied": {"records": 0.0}}},
             "records a string": {"entry150": {**DECLARED, "copied": {"records": "0"}}},
             "no records": {"entry150": {**DECLARED, "copied": {}}},
             "no branch_phase": {"entry150": {**DECLARED, "left_out": {"files": DECLARED["left_out"]["files"]}}},
             "branch_phase named": {"entry150": {**DECLARED, "left_out": {**DECLARED["left_out"], "branch_phase": {}}}},
             "branch_phase false": {"entry150": {**DECLARED, "left_out": {**DECLARED["left_out"], "branch_phase": False}}},
             "natural not left out": {"entry150": {**DECLARED, "left_out": {"branch_phase": None,
                                                                              "files": ["order-prompted.json"]}}},
             "prompted not left out": {"entry150": {**DECLARED, "left_out": {"branch_phase": None,
                                                                               "files": ["order-natural.json"]}}},
             "files a string": {"entry150": {**DECLARED, "left_out": {"branch_phase": None,
                                                                        "files": "order-natural.json order-prompted.json"}}},
             "files null": {"entry150": {**DECLARED, "left_out": {"branch_phase": None, "files": None}}}}
    for name, extra in wrong.items():
        mf = {"additions": manifest["additions"], "store_changes": LISTED, **(extra or {})}
        assert ex.is_branch_only(out_with("w-" + name.replace(" ", "-"), "order-branch.json", mf=mf), branch, change) is False, name
    # left_out.branch_phase as the builder writes it: null where the source had no branch phase pinned, {"starts": n} where
    # the build removed it (n an exact count); any other form is not the builder's
    def with_phase(value):
        mf = {**manifest, "entry150": {**DECLARED, "left_out": {**DECLARED["left_out"], "branch_phase": value}}}
        return out_with(f"bp-{len(list(tmp_path.iterdir()))}", "order-branch.json", mf=mf)

    for good in (None, {"starts": 0}, {"starts": 3}):
        assert ex.is_branch_only(with_phase(good), branch, change) is True, good
    for bad in ({}, {"starts": -1}, {"starts": True}, {"starts": 1.0}, {"starts": "1"}, {"starts": None},
                {"starts": 1, "extra": 2}, {"count": 1}, [], [1], "none", 0, False):
        assert ex.is_branch_only(with_phase(bad), branch, change) is False, bad
    # the change lists run ids: the declaration must count them, each once (the parse keeps a set of them)
    def listing(*ids, records):
        mf = {**manifest, "entry150": {**DECLARED, "copied": {"records": records}},
              "store_changes": [{**LISTED[0], "pre_change_records": len(ids), "pre_change_run_ids": list(ids)}]}
        return out_with(f"l-{len(list(tmp_path.iterdir()))}", "order-branch.json", mf=mf), [_change(1, A, B, before=ids)]

    for ids, records, expected in [(("p",), 1, True), (("p", "q"), 2, True), (("p",), 0, False), (("p", "q"), 1, False),
                                   (("p", "p"), 1, False), (("p", "p"), 2, False)]:  # a run id listed twice is no count
        o, ch = listing(*ids, records=records)
        assert ex.is_branch_only(o, branch, ch) is expected, (ids, records)
    o, ch = listing("p", records=1)
    assert ex.is_branch_only(o, branch, [_change(2, B, C, before={"p"})]) is False  # the first change must be the parse of the first


# Hand-derived timeline: the chain A, B, C with change 1 (A to B) at 09:00:10 and change 2 (B to C) at 09:00:30, the branch
# phase entered at 09:00:20. The copied parents: a1 and a2 on A (starts 0, 5), b1 on B (start 12, after change 1). The
# branch rows, each recording its parent's copy: x1 (start 22, records A) and x2 (start 29, records B) are in B's period,
# x3 (start 30, exactly change 2's instant, records A) and x4 (start 45, records B) in C's. So the branch rows' periods
# are 1 and 2 (B, C), and the parents' 0, 0, 1.


def _bo_rows():
    return [_row("records/a1.json", A, _at(0), _at(1)), _row("records/a2.json", A, _at(5), _at(6)),
            _row("records/b1.json", B, _at(12), _at(13)),
            _row("records/x1.json", A, _at(22), _at(23)), _row("records/x2.json", B, _at(29), _at(29.5)),
            _row("records/x3.json", A, _at(30), _at(31)), _row("records/x4.json", B, _at(45), _at(46))]


def _bo_changes(at1: str | None = None, at2: str | None = None):
    return [_change(1, A, B, at=at1 or _at(10)), _change(2, B, C, at=at2 or _at(30))]


def _bo_line(rows=None, changes=None, branch_sec: float = 20):
    rows, changes, branch = rows or _bo_rows(), changes or _bo_changes(), _branch(branch_sec)
    return ex.branch_only_timeline(ex.store_timeline(rows, changes, "records", branch), rows, changes, "records", branch)


def test_a_branch_only_timeline_takes_the_periods_of_the_branch_rows_from_the_manifest_instants():
    rows, changes = _bo_rows(), _bo_changes()
    base = ex.store_timeline(rows, changes, "records", _branch(20))
    assert (base.clean, base.chain_incomplete) == (False, True)  # today's verdict: no non-branch row names C
    line = _bo_line()
    assert (line.branch_only, line.clean, line.interleaved, line.overlap, line.chain_incomplete) == (
        True, True, False, False, False)
    assert line.chain == (A, B, C) and line.branch_only_problems == ()
    assert line.period_of == {"a1": 0, "a2": 0, "b1": 1, "x1": None, "x2": None, "x3": None, "x4": None}
    assert line.branch_row_periods == (1, 2)
    assert line.branch_row_endpoints == (A, B)  # the copies the branch rows record (their parents')
    assert [run.endpoint for run in line.runs] == [A, B]  # the parents' runs: the chain's start, not all of it
    instants = [ex.parse_utc(_at(10)), ex.parse_utc(_at(30))]
    # the copy current at a start: the new copy at the change's own instant, the old one a millisecond before it (and one
    # before the first change has no copy to give: refused below)
    for sec, copy in [(10, B), (22, B), (29.999, B), (30, C), (45, C), (1000, C)]:
        assert ex.branch_only_copy(line, instants, _at(sec)) == copy, sec
    for refused in (_at(9.999), "not a time", None):
        with pytest.raises(ex.Refused, match="no copy current at its start"):
            ex.branch_only_copy(line, instants, refused)


def test_a_branch_only_timeline_without_rows_before_the_branch_phase_is_clean_and_one_with_all_of_the_chain_too():
    only_branch = [r for r in _bo_rows() if r.name.startswith("records/x")]  # the parents live in another OUT
    line = _bo_line(only_branch)
    assert (line.clean, line.runs, line.branch_row_periods) == (True, (), (1, 2))
    # parents on both of the first two copies, each in its own period: the runs are the chain's first two
    rows = [*_bo_rows()[:2], _row("records/b0.json", B, _at(11), _at(12)), *_bo_rows()[3:]]
    both = _bo_line(rows)
    assert both.clean and both.period_of["b0"] == 1 and [run.endpoint for run in both.runs] == [A, B]


def test_a_branch_row_before_the_first_change_is_refused_by_name_and_one_at_it_is_not():
    changes = _bo_changes()
    rows = [*_bo_rows(), _row("records/early.json", A, _at(7), _at(8))]
    line = _bo_line(rows, changes, branch_sec=6)  # the branch phase from 09:00:06: the x rows and the early one are branch rows
    assert not line.clean and line.branch_only_problems == (
        "branch rows started before the OUT's first manifest store change, so the copy they ran on is not defined: "
        "['records/early.json']",)
    assert ex.timeline_problems("out", line) == [
        "out: branch rows started before the OUT's first manifest store change, so the copy they ran on is not defined: "
        "['records/early.json']"]  # (no empty flag list in front of it)
    at_the_change = [*_bo_rows(), _row("records/early.json", A, _at(10), _at(11))]
    assert _bo_line(at_the_change, changes, branch_sec=6).clean  # a start at the instant is in the new copy's period


def test_a_row_before_the_branch_phase_must_record_the_copy_current_at_its_start():
    changes = _bo_changes()
    after_change = [r for r in _bo_rows() if r.name != "records/b1.json"] + [_row("records/old.json", A, _at(12), _at(13))]
    line = _bo_line(after_change, changes)  # an A record started after change 1, before the branch phase
    assert not line.clean and line.branch_only_problems == (
        "rows made before the branch phase record another copy than the one current at their start by the manifest's "
        "store changes: ['records/old.json']",)
    before_change = [_row("records/early-b.json", B, _at(5), _at(6)), *_bo_rows()[2:]]
    assert _bo_line(before_change, changes).branch_only_problems == (
        "rows made before the branch phase record another copy than the one current at their start by the manifest's "
        "store changes: ['records/early-b.json']",)  # a B record started before change 1
    tie = [r for r in _bo_rows() if r.name != "records/b1.json"] + [_row("records/tie.json", A, _at(10), _at(11))]
    assert _bo_line(tie, changes).branch_only_problems != ()  # a start at change 1's instant is B's, not A's
    ok_tie = [r for r in _bo_rows() if r.name != "records/b1.json"] + [_row("records/tie.json", B, _at(10), _at(11))]
    assert _bo_line(ok_tie, changes).clean


def test_copies_alternating_before_the_branch_phase_are_still_a_mix_in_a_branch_only_timeline():
    rows = [_row("records/a1.json", A, _at(0), _at(1)), _row("records/b1.json", B, _at(12), _at(13)),
            _row("records/a2.json", A, _at(14), _at(15)), _row("records/x1.json", A, _at(22), _at(23))]
    line = _bo_line(rows)
    assert (line.clean, line.interleaved, line.chain_incomplete) == (False, True, False)
    assert "(interleaved)" in ex.timeline_problems("out", line)[0]
    overlapping = [_row("records/a1.json", A, _at(0), _at(13)), _row("records/b1.json", B, _at(12), _at(14)),
                   _row("records/x1.json", A, _at(22), _at(23))]
    assert _bo_line(overlapping).overlap is True and not _bo_line(overlapping).clean


def test_a_branch_only_timeline_keeps_the_other_refusals_of_the_order_check():
    outside = [*_bo_rows(), _row("records/x5.json", "http://z.invalid:9/", _at(50), _at(51))]
    line = _bo_line(outside)
    assert (line.clean, line.endpoint_outside_chain) == (False, True)  # a branch row must still name a chain endpoint
    no_endpoint = [*_bo_rows(), _row("records/x6.json", None, _at(50), _at(51))]
    assert (_bo_line(no_endpoint).clean, _bo_line(no_endpoint).rows_without_endpoint) == (False, True)
    bad_time = [*_bo_rows(), _row("records/x7.json", B, "not a time", _at(51))]
    line = _bo_line(bad_time)
    assert (line.clean, line.unusable_times, line.branch_only, line.runs, line.branch_row_periods) == (
        False, True, True, None, ())
    two = [*_bo_rows(), _row("records/x8.json", A, _at(50), _at(51), run_id="a1")]  # a1's run id in the branch phase too
    assert (_bo_line(two).clean, _bo_line(two).run_id_in_two_runs) == (False, True)


def test_the_change_instants_of_a_branch_only_timeline_must_be_times_and_strictly_increasing():
    for at1, at2 in [(_at(30), _at(10)), (_at(10), _at(10))]:
        line = _bo_line(changes=_bo_changes(at1, at2))
        assert not line.clean and line.branch_only_problems == (
            "its manifest store changes' instants are not strictly increasing, so the periods of its copies are not "
            "defined",)
        assert line.period_of["x1"] is None and line.branch_row_periods == ()
    for bad in ("soon", None, 7, "2026-10-23T09:00:10"):  # not text, not a time, or a time without its zone
        line = _bo_line(changes=[_change(1, A, B, at=bad), _change(2, B, C, at=_at(30))])
        assert not line.clean and line.branch_only_problems == (
            "a manifest store change's at is not an ISO time with a zone, so the periods of its copies are not defined",)


def test_the_timeline_of_an_ordinary_out_is_what_it_was_and_has_no_branch_only_marks():
    rows, changes = _bo_rows(), _bo_changes()
    line = ex.store_timeline(rows, changes, "records", _branch(20))
    assert (line.branch_only, line.branch_only_problems, line.branch_row_periods) == (False, (), ())
    assert (line.clean, line.chain_incomplete, line.period_of["x1"]) == (False, True, None)
    assert ex.timeline_problems("out", line) == [
        "out: its records' own endpoints and times do not follow its manifest store changes (chain_incomplete): no "
        "record names a copy of its chain, or the last copies of it"]


def test_a_branch_only_out_is_accepted_with_its_rows_attributed_by_the_manifest_instant(world, ref):
    w = _branch_only_world(world)
    out = str(w["out"])
    assert sorted(p.name for p in w["out"].iterdir() if p.name.startswith("order")) == ["order-branch.json"]
    stores = _check(world, w["out"], w["proofs"], w["entries"])
    line = stores.timelines[out]
    assert (line.branch_only, line.clean, line.chain, line.branch_row_periods) == (True, True, (E0, E1), (1,))
    assert line.branch_phase_start == BO_BRANCH and line.branch_row_endpoints == (E0,)
    # copy-b is named by no record, yet it is labelled and proven; every record names copy-a
    assert stores.labels == {(out, E0): "copy-a", (out, E1): "copy-b"}
    assert sorted(stores.copies) == ["copy-a", "copy-b"]
    assert stores.copies["copy-b"]["listed_by"][out] == ["its deployment identity's store",
                                                        "its manifest store change 1 (to)"]
    assert stores.period_starts == {(out, E0): [None], (out, E1): [MANIFEST_AT]}
    # the parents' rows count for copy-a, the branch rows for copy-b (by the question each record answers, by hand)
    assert {r: stores.run_row[r] for r in w["question"]} == {
        r: ("copy-a" if q in BO_PARENTS else "copy-b") for r, q in w["question"].items()}
    (entry,) = stores.branch_phase_copies
    assert entry == {
        "out": out, "branch_phase_start": BO_BRANCH,
        "parent_copies": [{"copy": "copy-a", **{k: CLASS_A[k] for k in ("class", "label", "basis", "source")}}],
        "branch_copies": [{"copy": "copy-b", **{k: CLASS_B[k] for k in ("class", "label", "basis", "source")},
                           "period_start": MANIFEST_AT}],
        "attribution": BO_ATTRIBUTION_TEXT, "branch_only": True, "branch_only_rule": BO_RULE_TEXT}


def test_the_shares_of_a_branch_only_out_count_the_branch_rows_for_the_copy_current_at_their_start(world, ref):
    w = _branch_only_world(world)
    out = str(w["out"])
    st = _stores(world, [w["out"]], w["proofs"], w["entries"])
    # each initial prompt arm has four attempts, one per question: pass and flag are the parents (copy-a), abstain and
    # noquery the branch rows (copy-b), though all four record copy-a
    for model in MODELS:
        for cell in CELLS:
            arm = f"s01|{model}|{cell}|prompt|-"
            assert st["shares"][arm] == _rows({"copy-a": 2, "copy-b": 2})
            assert st["shares_by_out"][out][arm] == _rows({"copy-a": 2, "copy-b": 2})
    (change,) = st["store_changes"]
    assert (change["out"], change["index"], change["from"], change["to"]) == (out, 1, "copy-a", "copy-b")
    assert change["branch_phase_start"] == BO_BRANCH and change["manifest_at"] == MANIFEST_AT
    assert change["first_record_on_new_endpoint"] is None  # no record names copy-b
    assert change["change_at_used"] == MANIFEST_AT  # the manifest's instant, as written
    arm = f"s01|{QWEN}|none|prompt|-"
    assert change["shares_by_arm"][arm] == {"before": _rows({"copy-a": 2}), "after": _rows({}),
                                            "branch_phase": _rows({"copy-b": 2})}
    assert [c["branch_only"] for c in st["branch_phase_copies"]] == [True]
    # the rule sentence of the section is the ordinary one: the branch-only sentence rides on the OUT's own entry
    assert st["rule"].endswith("; records of the two copies interleaving is a mix and refuses. " + BRANCH_RULE_TEXT)
    assert BO_RULE_TEXT not in st["rule"]


def test_a_branch_only_chain_of_three_copies_attributes_each_branch_row_to_the_copy_current_at_its_start(world, ref):
    w = _branch_only_world(world, three=True)
    out = str(w["out"])
    # the branch phase begins between the changes' instants (09:18:39.962 and 09:19:30): s01-abstain's rows, from
    # 09:18:45, are copy-b's; s01-noquery's, from exactly 09:19:30 (the first of them at the second change's instant), are
    # copy-x's
    stores = _check(world, w["out"], w["proofs"], w["entries"], _bo_classes(world))
    line = stores.timelines[out]
    assert (line.clean, line.chain, line.branch_row_periods) == (True, (E0, E1, EX), (1, 2))
    assert stores.labels == {(out, E0): "copy-a", (out, E1): "copy-b", (out, EX): "copy-x"}
    where = {"s01-pass": "copy-a", "s01-flag": "copy-a", "s01-abstain": "copy-b", "s01-noquery": "copy-x"}
    assert {r: stores.run_row[r] for r in w["question"]} == {r: where[q] for r, q in w["question"].items()}
    (entry,) = stores.branch_phase_copies
    assert [(c["copy"], c["class"], c["period_start"]) for c in entry["branch_copies"]] == [
        ("copy-b", "slow", MANIFEST_AT), ("copy-x", "medium", BO_SECOND_AT)]
    assert [c["copy"] for c in entry["parent_copies"]] == ["copy-a"]
    # the first s01-noquery row a millisecond before the second change is copy-b's
    first = min((r for r, q in w["question"].items() if q == "s01-noquery"),
                key=lambda r: json.loads(next((w["out"] / "records").glob(f"*.{r}.json")).read_text())["started_at"])
    just_before = "2026-10-23T09:19:29.999+00:00"  # (its end is its start: a longer attempt would run across the change)
    _retime(w["out"], lambda r, f: (just_before, just_before) if r["run_id"] == first else None)
    moved = _check(world, w["out"], w["proofs"], w["entries"], _bo_classes(world))
    assert moved.run_row[first] == "copy-b"
    assert sum(1 for r, label in moved.run_row.items() if label == "copy-x") == 9  # the other nine of its ten records
    st = _stores(world, [w["out"]], w["proofs"], w["entries"], copy_classes=_bo_classes(world))
    arm = f"s01|{QWEN}|none|prompt|-"  # an arm's four attempts: pass and flag copy-a, abstain copy-b, noquery copy-x or -b
    assert st["shares"][arm]["attempts"] == 4
    assert st["shares"][arm]["rows"]["copy-a"]["n"] == 2


def _bo_refusal(world, w, match: str, *, entries=None, classes: Path | None = None) -> str:
    """check_stores on the branch-only world refuses with ``match``: the message."""
    with pytest.raises(ex.Refused, match=match) as caught:
        _check(world, w["out"], w["proofs"], w["entries"] if entries is None else entries, classes)
    return str(caught.value)


def test_an_out_with_another_order_file_or_without_a_branch_phase_is_judged_as_it_always_was(world, ref):
    old = (STORE_REFUSAL + f"{{out}}: its records' own endpoints and times do not follow its manifest store changes "
                           f"(chain_incomplete): no record names a copy of its chain, or the last copies of it")
    w = _branch_only_world(world)
    for extra in ("order-natural.json", "order-prompted.json"):
        shutil.copy2(FIXTURE / "batch" / extra, w["out"] / extra)
        _recheck(w["out"])
        assert ex.is_branch_only(ex.Out.read(w["out"]), ex.manifest_branch_phase(ex.Out.read(w["out"]))[0],
                                 ex.manifest_store_changes(ex.Out.read(w["out"]))[0]) is False
        assert _bo_refusal(world, w, r"\(chain_incomplete\)") == old.format(out=w["out"])
        (w["out"] / extra).unlink()
        assert _check(world, w["out"], w["proofs"], w["entries"]).timelines[str(w["out"])].branch_only is True
    _set_branch_phase(w["out"], None)  # no branch phase in the manifest: every row is an ordinary row
    assert _bo_refusal(world, w, r"\(chain_incomplete\)") == old.format(out=w["out"])


def test_a_branch_only_out_needs_every_copy_of_its_chain_proven_though_no_record_names_the_last(world, ref):
    w = _branch_only_world(world)
    out = w["out"]
    message = _bo_refusal(world, w, r"no store-proofs.json entry proves its copy", entries=w["entries"][:1])
    assert message == STORE_REFUSAL + f"{out} {ex.endpoint_label(E1)}: no store-proofs.json entry proves its copy"


def test_a_branch_only_chain_of_three_needs_its_middle_copy_proven_too(world, ref):
    w = _branch_only_world(world, three=True)
    entries = [w["entries"][0], w["entries"][2]]  # copy-a and copy-x, not copy-b
    message = _bo_refusal(world, w, r"no store-proofs.json entry proves its copy", entries=entries,
                          classes=_bo_classes(world))
    assert message == STORE_REFUSAL + f"{w['out']} {ex.endpoint_label(E1)}: no store-proofs.json entry proves its copy"


def test_a_branch_only_copys_change_at_is_the_manifests_instant_not_the_first_branch_rows_start(world, ref):
    w = _branch_only_world(world)
    entries = [w["entries"][0], {**w["entries"][1], "change_at": "2026-10-23T09:18:45Z"}]  # the first branch row's start
    message = _bo_refusal(world, w, r"not the instant of the manifest's store change", entries=entries)
    assert message == (STORE_REFUSAL + f"{w['out']} {ex.endpoint_label(E1)} (copy-b): its change_at is not the instant "
                                       f"of the manifest's store change (a branch-only OUT's records do not name the copy)")
    # the same instant written another way is the same instant
    same = [w["entries"][0], {**w["entries"][1], "change_at": "2026-10-23T09:18:39.962Z"}]
    assert _check(world, w["out"], w["proofs"], same).timelines[str(w["out"])].clean


@pytest.mark.parametrize("store", [{"out": "/host/single"}, {"out": "/host/single", "endpoint": E0},
                                   {"out": "/host/single", "endpoint": "http://copy-b.invalid:7040"}, None])
def test_a_branch_only_outs_identity_must_name_the_last_copy_of_the_chain_as_its_store(world, ref, store):
    w = _branch_only_world(world)
    ident = json.loads((w["out"] / "deployment-identity.json").read_text(encoding="utf-8"))
    ident["store"] = store
    _write_identity(w["out"], ident)
    message = _bo_refusal(world, w, r"deployment identity must name, as its store, the last copy")
    assert f"{w['out']}: a branch-only OUT's deployment identity must name, as its store, the last copy of its " \
           f"manifest's store changes (no record names it)" in message


def test_a_branch_only_out_without_an_identity_is_refused(world, ref):
    w = _branch_only_world(world)
    (w["out"] / "deployment-identity.json").unlink()
    _bo_refusal(world, w, r"deployment identity must name, as its store, the last copy")


def test_a_branch_row_before_the_first_change_refuses_the_call_with_its_name(world, ref):
    w = _branch_only_world(world)
    _set_branch_phase(w["out"], "2026-10-23T09:18:30.000+00:00")  # before the change (09:18:39.962)
    run = w["middle"][0]
    _retime(w["out"], lambda r, f: "2026-10-23T09:18:35.000+00:00" if r["run_id"] == run else None)
    message = _bo_refusal(world, w, r"started before the OUT's first manifest store change")
    name = next(p.name for p in (w["out"] / "records").glob(f"*.{run}.json"))
    assert message.endswith(f"{w['out']}: branch rows started before the OUT's first manifest store change, so the copy "
                            f"they ran on is not defined: ['records/{name}']")
    assert "do not follow its manifest store changes" not in message  # no empty list of flags in front of it


def test_a_row_made_before_the_branch_phase_must_record_the_copy_current_at_its_start_in_the_call(world, ref):
    w = _branch_only_world(world)
    # the branch phase begins later than the first branch rows: they are now ordinary rows, made after the change and
    # recording copy-a, which was no longer current
    _set_branch_phase(w["out"], "2026-10-23T09:19:00.000+00:00")
    message = _bo_refusal(world, w, r"record another copy than the one current at their start")
    later = [r for r, q in w["question"].items() if q in ("s01-abstain", "s01-noquery")]  # all from 09:18:45
    names = sorted(f"records/{p.name}" for r in later for p in (w["out"] / "records").glob(f"*.{r}.json"))
    assert str(names[:5]) in message


def test_a_parent_recording_the_new_copy_before_the_change_is_refused_in_the_call(world, ref):
    w = _branch_only_world(world)
    last = max(w["parents"], key=lambda r: json.loads(next((w["out"] / "records").glob(f"*.{r}.json")).read_text())[
        "started_at"])
    f = next((w["out"] / "records").glob(f"*.{last}.json"))
    r = json.loads(f.read_text(encoding="utf-8"))
    r["store"]["endpoint"] = E1  # the latest-starting parent says copy-b, though it started before the change
    f.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    _recheck(w["out"])
    message = _bo_refusal(world, w, r"record another copy than the one current at their start")
    assert f"['records/{f.name}']" in message


def test_the_change_instants_of_a_branch_only_chain_must_increase_in_the_call(world, ref):
    w = _branch_only_world(world, three=True)
    manifest = json.loads((w["out"] / "manifest.json").read_text(encoding="utf-8"))
    manifest["store_changes"][1]["at"] = "2026-10-23T09:18:30.000+00:00"  # before change 1 (09:18:39.962)
    (w["out"] / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    _recheck(w["out"])
    message = _bo_refusal(world, w, r"instants are not strictly increasing", classes=_bo_classes(world))
    assert message.endswith(f"{w['out']}: its manifest store changes' instants are not strictly increasing, so the "
                            f"periods of its copies are not defined")


def test_a_branch_only_copy_without_a_class_entry_refuses_the_call(world, ref):
    w = _branch_only_world(world)
    path = _classes_file(world["tmp"] / "no-b.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["copies"]["copy-b"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    message = _bo_refusal(world, w, r"has no entry in the copy classes", classes=path)
    assert message == f"{w['out']}: the replica copy-b of its copy copy-b has no entry in the copy classes"


def test_an_ordinary_out_with_a_branch_phase_keeps_its_report_and_attribution_unchanged(world, ref):
    m = _moved_world(world)
    _set_branch_phase(m["out"], T_B)
    stores = _check(world, m["out"], m["proofs"], m["entries"])
    line = stores.timelines[str(m["out"])]
    assert (line.branch_only, line.branch_only_problems, line.branch_row_periods) == (False, (), ())
    (entry,) = stores.branch_phase_copies
    assert set(entry) == {"out", "branch_phase_start", "parent_copies", "branch_copy", "attribution"}  # the old shape
    assert entry["branch_copy"]["copy"] == "copy-b"
    assert entry["attribution"] == (
        "a branch row records its parent attempt's copy (branch_attempt in harness/checkloop/online.py), but its live "
        "retry ran on the branch copy, the chain's last copy: the row counts for that copy, and a checker arm's retried "
        "output whose parent attempt ran on another copy is a unit row naming both copies, its retry attempt counted "
        "for the branch copy in the attempt shares")
    assert set(ex._check_stores_object(stores)) == CHECK_KEYS  # no branch_only_outs key
    st = _stores(world, [m["out"]], m["proofs"], m["entries"])
    (change,) = st["store_changes"]
    assert change["change_at_used"] == change["first_record_on_new_endpoint"] == "2026-10-23T09:18:45.000+00:00"
    assert '"branch_only"' not in json.dumps(st) and "a branch-only OUT" not in json.dumps(st)  # (not the path's name)


def test_the_check_stores_command_names_a_branch_only_out_and_its_copies_by_labels_alone(world, ref, s01_graph, capsys):
    w = _branch_only_world(world)
    code, obj, raw, err = _check_cli(capsys, world, [w["out"]], w["proofs"], w["entries"])
    assert (code, err) == (0, "")
    assert obj == {"ok": True, "outs_with_store_changes": ["single"], "branch_only_outs": ["single"],
                   "copies": {"copy-a": {"endpoints": [f"endpoint sha256:{_h(E0)[:12]}"], "outs": ["single"],
                                         "periods": [None]},
                              "copy-b": {"endpoints": [f"endpoint sha256:{_h(E1)[:12]}"], "outs": ["single"],
                                         "periods": [MANIFEST_AT]}}}
    assert _ints(obj) == [] and E0 not in raw and E1 not in raw and "invalid" not in raw
    code, obj, raw, err = _check_cli(capsys, world, [w["out"]], w["proofs"], w["entries"][:1])
    assert code == 2 and set(obj) == {"ok", "refused"} and "no store-proofs.json entry proves its copy" in obj["refused"]


def test_an_ordinary_out_whose_order_files_are_gone_is_not_taken_for_a_branch_only_one(world, ref):
    old = (STORE_REFUSAL + f"{{out}}: its records' own endpoints and times do not follow its manifest store changes "
                           f"(chain_incomplete): no record names a copy of its chain, or the last copies of it")
    for change in (lambda m: m.pop("entry150"), lambda m: m["entry150"]["copied"].update(records=1),
                   lambda m: m["entry150"]["left_out"].update(files=["order-natural.json"]),
                   lambda m: m["entry150"]["left_out"].update(branch_phase="the source's")):
        w = _branch_only_world(world)  # (its directory holds order-branch.json alone, as the lost-files OUT's would)
        manifest = json.loads((w["out"] / "manifest.json").read_text(encoding="utf-8"))
        change(manifest)
        (w["out"] / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
        _recheck(w["out"])
        assert _bo_refusal(world, w, r"\(chain_incomplete\)") == old.format(out=w["out"])
        world = _fresh(world)


def _fresh(world) -> dict:
    """The same fixture world again: the OUT (and its proof files) rebuilt from the fixture, nothing of the old one left."""
    shutil.rmtree(world["single"])
    single = world["single"]
    shutil.copytree(FIXTURE / "batch", single, ignore=shutil.ignore_patterns("claims"))
    _write_identity(single, _identity(single))
    return world


def test_a_branch_only_outs_identity_chain_must_join_up_though_only_one_endpoint_is_recorded(world, ref):
    w = _branch_only_world(world)
    ident = json.loads((w["out"] / "deployment-identity.json").read_text(encoding="utf-8"))
    store = ident["store"]  # the last copy, E1
    ident["store_changes"] = [{"from": {"endpoint": E0}, "to": {"endpoint": E0}}]  # ends at E0, not at its store
    _write_identity(w["out"], ident)
    message = _bo_refusal(world, w, r"its identity's store block is not its last identity store change's copy")
    assert message.endswith(f"{w['out']}: its identity's store block is not its last identity store change's copy")
    ident["store_changes"] = [{"from": {"endpoint": E0}, "to": store}]  # a chain that joins up is accepted
    _write_identity(w["out"], ident)
    assert _check(world, w["out"], w["proofs"], w["entries"]).timelines[str(w["out"])].branch_only is True


def test_an_unused_middle_copy_of_a_branch_only_chain_needs_its_class_entry_too(world, ref):
    w = _branch_only_world(world, three=True)
    seen: Counter = Counter()

    def later(r: dict, f: Path):  # s01-abstain's rows after the second change too: no branch row is in copy-b's period
        if r["question_id"] != "s01-abstain":
            return None
        seen["n"] += 1
        return datetime(2026, 10, 23, 9, 19, 40, tzinfo=UTC) + timedelta(seconds=seen["n"] - 1)

    _retime(w["out"], later)
    path = _classes_file(world["tmp"] / "no-b.json", **{"copy-x": {**CLASS_C, "basis": "basis of copy-x"}})
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["copies"]["copy-b"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    message = _bo_refusal(world, w, r"has no entry in the copy classes", classes=path)
    assert message == f"{w['out']}: the replica copy-b of its copy copy-b has no entry in the copy classes"
    line = _check(world, w["out"], w["proofs"], w["entries"], _bo_classes(world)).timelines[str(w["out"])]
    assert line.clean and line.branch_row_periods == (2,)  # with its class entry it passes, though no row is attributed to it


def test_a_branch_only_chain_that_returns_to_its_first_copy_attributes_by_the_period_position(world, ref):
    w = _branch_only_world(world, ret=True)
    out = str(w["out"])
    stores = _check(world, w["out"], w["proofs"], w["entries"])
    line = stores.timelines[out]
    assert (line.clean, line.chain, line.branch_row_periods) == (True, (E0, E1, E0), (1, 2))
    assert stores.labels == {(out, E0): "copy-a (two periods)", (out, E1): "copy-b"}
    # s01-pass and s01-flag (the parents) and s01-noquery (from the return, 09:19:30) are copy-a's; s01-abstain's rows
    # (from 09:18:45, between the two changes) are copy-b's
    where = {"s01-pass": "copy-a (two periods)", "s01-flag": "copy-a (two periods)", "s01-abstain": "copy-b",
             "s01-noquery": "copy-a (two periods)"}
    assert {r: stores.run_row[r] for r in w["question"]} == {r: where[q] for r, q in w["question"].items()}
    (entry,) = stores.branch_phase_copies
    assert [(c["copy"], c["period_start"]) for c in entry["branch_copies"]] == [
        ("copy-b", MANIFEST_AT), ("copy-a (two periods)", BO_SECOND_AT)]
    assert line.period_of[next(r for r, q in w["question"].items() if q == "s01-flag")] == 0


def test_branch_only_set_aside_rows_equal_starts_and_a_row_at_the_branch_instant_are_placed_by_hand():
    changes = _bo_changes()
    base_rows = [r for r in _bo_rows() if r.name != "records/x1.json"]
    rows = [*base_rows,
            _row("records/t1.json", A, _at(20), _at(21)),  # exactly at the branch instant: a branch row, period 1
            _row("records/t2.json", A, _at(22), _at(23)), _row("records/t3.json", B, _at(22), _at(24)),  # equal starts
            _row("infrastructure/s1.json", A, _at(35), _at(36), run_id="s1"),  # a set-aside branch row, period 2
            _row("infrastructure/s2.json", A, _at(3), _at(4), run_id="s2")]  # set-aside, before the branch phase, copy A
    line = _bo_line(rows, changes)
    assert line.clean and line.branch_row_periods == (1, 2)
    assert line.period_of["t1"] is None and line.period_of["t2"] is None and line.period_of["t3"] is None
    assert "s1" not in line.period_of and "s2" not in line.period_of  # (set-aside rows have no place in period_of)
    for row, why in [(_row("infrastructure/s3.json", B, _at(3), _at(4), run_id="s3"), "another copy"),  # B before change 1
                     ]:
        bad = _bo_line([*rows, row], changes)
        assert not bad.clean and "['infrastructure/s3.json']" in bad.branch_only_problems[0], why
    early = _bo_line([*rows, _row("infrastructure/s4.json", A, _at(7), _at(8), run_id="s4")], changes, branch_sec=6)
    assert not early.clean and "['infrastructure/s4.json']" in early.branch_only_problems[0]  # a set-aside branch row, early
    just_before = _bo_line([*base_rows, _row("records/j.json", B, _at(19.999), _at(21))], changes)
    assert just_before.clean and just_before.period_of["j"] == 1  # a pre-branch row (B, after change 1): period 1


def test_an_attempt_running_across_a_change_or_a_run_id_in_two_periods_is_refused_in_a_branch_only_timeline():
    changes = _bo_changes()
    base_rows = [r for r in _bo_rows() if r.name not in ("records/a2.json", "records/x2.json")]
    ok = [*base_rows, _row("records/a2.json", A, _at(5), _at(10)),  # ends exactly at change 1's instant: allowed
          _row("records/x2.json", B, _at(29), _at(30))]  # ends exactly at change 2's instant: allowed
    assert _bo_line(ok, changes).clean
    parent_across = [*base_rows, _row("records/a2.json", A, _at(5), _at(10.001)), _row("records/x2.json", B, _at(29), _at(30))]
    assert _bo_line(parent_across, changes).branch_only_problems == (
        "attempts ran across a change of copy (started in one copy's period and ended after the next began), a mix of "
        "copies, not a late timestamp: ['records/a2.json']",)
    branch_across = [*base_rows, _row("records/a2.json", A, _at(5), _at(6)), _row("records/x2.json", B, _at(29), _at(31))]
    assert _bo_line(branch_across, changes).branch_only_problems == (
        "attempts ran across a change of copy (started in one copy's period and ended after the next began), a mix of "
        "copies, not a late timestamp: ['records/x2.json']",)
    # a chain A, B, A with no row of B before the branch phase: the rows of A are one run for the base, which cannot see
    # one run id in the chain's first and third positions
    back = [_change(1, A, B, at=_at(10)), _change(2, B, A, at=_at(30))]
    dup = [_row("records/p1.json", A, _at(1), _at(2), run_id="dup"), _row("records/p2.json", A, _at(35), _at(36), run_id="dup"),
           _row("records/x1.json", B, _at(45), _at(46))]
    line = _bo_line(dup, back, branch_sec=40)
    assert not line.clean and line.run_id_in_two_runs is False  # (the base's flag cannot see it)
    assert line.branch_only_problems == (
        "one run id names records/ records made in different periods of the manifest's store changes: "
        "['records/p2.json']",)
    assert _bo_line([dup[0], _row("records/p2.json", A, _at(35), _at(36), run_id="other"), dup[2]], back,
                    branch_sec=40).clean  # the same rows with distinct run ids are clean


def test_the_names_of_early_rows_and_of_rows_recording_another_copy_are_cut_at_five_sorted():
    rows = [_row(f"records/e{k}.json", A, _at(7 + k / 10), _at(8)) for k in range(7)]  # seven branch rows before change 1
    line = _bo_line([*rows, *(r for r in _bo_rows() if r.name.startswith("records/a"))], branch_sec=6)
    assert line.branch_only_problems == (
        "branch rows started before the OUT's first manifest store change, so the copy they ran on is not defined: "
        "['records/e0.json', 'records/e1.json', 'records/e2.json', 'records/e3.json', 'records/e4.json']",)


def test_the_store_timeline_command_still_says_chain_incomplete_for_a_branch_only_out(world, ref, capsys):
    w = _branch_only_world(world)
    code, (obj,), raw, err = _cli(capsys, w["out"])  # the diagnostic tool is not changed by the branch-only rule
    assert (code, obj["verdict"], obj["chain_incomplete"], obj["clean"]) == (2, "STOP: chain_incomplete", True, False)
    assert "branch_only" not in obj


def test_the_rows_before_the_branch_phase_are_exactly_the_copied_parents_the_first_change_lists():
    branch = _branch(20)
    rows = [r for r in _bo_rows() if r.name != "records/b1.json"]  # (b1 started after the change: see below)
    listed = _change(1, A, B, before={"a1", "a2"}, at=_at(10))
    assert ex.copied_parents_problems(rows, [listed], branch, "records") == []
    # a listed parent that started at or after the first change's instant is no parent made before it (an exact
    # tie counts as after, as everywhere)
    after = [*rows, _row("records/b1.json", B, _at(12), _at(13))]
    late = _change(1, A, B, before={"a1", "a2", "b1"}, at=_at(10))
    assert ex.copied_parents_problems(after, [late], branch, "records") == [
        "copied parents that started at or after the first store change, which lists them as made before it: "
        "['records/b1.json']"]
    tie = [*rows, _row("records/t.json", B, _at(10), _at(10.5))]
    assert len(ex.copied_parents_problems(tie, [_change(1, A, B, before={"a1", "a2", "t"}, at=_at(10))], branch,
                                          "records")) == 1
    just = [*rows, _row("records/j.json", A, _at(9.999), _at(9.9995))]
    assert ex.copied_parents_problems(just, [_change(1, A, B, before={"a1", "a2", "j"}, at=_at(10))], branch,
                                      "records") == []
    # a second row (any directory) with a listed run id is an attempt of the OUT's own beside the copied one
    twice = [*rows, _row("infrastructure/a1-again.json", A, _at(2), _at(3), run_id="a1"),
             _row("records/a1-too.json", A, _at(3), _at(4), run_id="a1")]
    assert ex.copied_parents_problems(twice, [listed], branch, "records") == [
        "records made before the branch phase that share a copied parent's run id, so one of them is an attempt of the "
        "OUT's own: ['infrastructure/a1-again.json', 'records/a1-too.json', 'records/a1.json']"]
    # an own row before the branch phase (not listed), in any directory, named by record (at most five, sorted)
    own = [*rows, _row("records/own.json", A, _at(3), _at(4)), _row("infrastructure/own2.json", A, _at(4), _at(5), run_id="own2")]
    assert ex.copied_parents_problems(own, [listed], branch, "records") == [
        "records made before the branch phase that the manifest does not list as copied parents (the first store "
        "change's pre_change_run_ids): ['infrastructure/own2.json', 'records/own.json']"]
    many = [*rows, *(_row(f"records/o{k}.json", A, _at(1 + k / 10), _at(4)) for k in (6, 5, 4, 3, 2, 1, 0))]
    (message,) = ex.copied_parents_problems(many, [listed], branch, "records")
    assert message.endswith("pre_change_run_ids): ['records/o0.json', 'records/o1.json', 'records/o2.json', "
                            "'records/o3.json', 'records/o4.json']")
    # a listed parent with no records/ row made before the branch phase (a set-aside row is not one; nor is a later row)
    short = [r for r in rows if r.name != "records/a2.json"]
    assert ex.copied_parents_problems(short, [listed], branch, "records") == [
        "copied parents the manifest lists (the first store change's pre_change_run_ids) have no records/ record made "
        "before the branch phase"]
    in_set_aside = [*short, _row("infrastructure/a2.json", A, _at(5), _at(6), run_id="a2")]
    assert len(ex.copied_parents_problems(in_set_aside, [listed], branch, "records")) == 1
    later = [*short, _row("records/a2-late.json", A, _at(25), _at(26), run_id="a2")]
    assert len(ex.copied_parents_problems(later, [listed], branch, "records")) == 1  # (a2 started in the branch phase)
    # a run id that is no text before the branch phase is not a listed parent; a row without a usable start is left
    # to the timeline
    odd = [*rows, ex.StoreRow("records/odd.json", None, QWEN, "s01", A, _at(2), _at(3)),
           ex.StoreRow("records/bad.json", "bad", QWEN, "s01", A, "not a time", _at(3))]
    assert ex.copied_parents_problems(odd, [listed], branch, "records") == [
        "records made before the branch phase that the manifest does not list as copied parents (the first store "
        "change's pre_change_run_ids): ['records/odd.json']"]


def test_a_branch_row_run_id_in_two_periods_is_refused_as_one_in_two_periods_before_the_branch_phase_is():
    changes = _bo_changes()
    # x2 starts in B's period (29) and x4 in C's (45): the labels are keyed by run id, so a shared one would keep the last
    rows = [r if r.name != "records/x4.json" else _row("records/x4.json", B, _at(45), _at(46), run_id="x2")
            for r in _bo_rows()]
    line = _bo_line(rows, changes)
    assert not line.clean and line.branch_only_problems == (
        "one run id names records/ records made in different periods of the manifest's store changes: "
        "['records/x4.json']",)
    same = [r if r.name != "records/x4.json" else _row("records/x4.json", B, _at(31), _at(32), run_id="x3")
            for r in _bo_rows()]  # x3 and x4 in one period (C's) sharing a run id: not a period problem
    assert _bo_line(same, changes).branch_only_problems == ()
    many = [_row(f"records/w{k}.json", B, _at(21 + k / 10), _at(21.05 + k / 10), run_id="w0") for k in range(7)]
    many += [_row(f"records/v{k}.json", B, _at(46 + k), _at(46.5 + k), run_id="w0") for k in (6, 5, 4, 3, 2, 1, 0)]
    (message,) = _bo_line([*_bo_rows()[:3], *many], changes).branch_only_problems
    assert message.endswith("['records/v0.json', 'records/v1.json', 'records/v2.json', 'records/v3.json', "
                            "'records/v4.json']")  # sorted, then cut at five, whatever the order the rows came in
    across = [_row(f"records/z{k}.json", B, _at(25), _at(31 + k)) for k in (6, 5, 4, 3, 2, 1, 0)]
    (message,) = _bo_line([*_bo_rows()[:3], *across], changes).branch_only_problems
    assert message.endswith("['records/z0.json', 'records/z1.json', 'records/z2.json', 'records/z3.json', "
                            "'records/z4.json']")


def test_branch_only_copy_has_no_copy_without_a_change_instant():
    line = _bo_line()
    with pytest.raises(ex.Refused, match="no copy current at its start"):
        ex.branch_only_copy(line, [], _at(10))


def test_an_own_row_before_the_branch_phase_refuses_the_branch_only_call_by_record_name(world, ref):
    w = _branch_only_world(world)
    own = w["middle"][0]  # an s01-abstain record moved back before the change, on copy-a: consistent with the periods
    _retime(w["out"], lambda r, f: "2026-10-23T09:10:30.000+00:00" if r["run_id"] == own else None)
    message = _bo_refusal(world, w, r"does not list as copied parents")
    name = next(p.name for p in (w["out"] / "records").glob(f"*.{own}.json"))
    assert message.endswith(f"{w['out']}: records made before the branch phase that the manifest does not list as copied "
                            f"parents (the first store change's pre_change_run_ids): ['records/{name}']")
    # (a record name holds its run id, as in every store refusal; the message carries no other)


def test_a_listed_copied_parent_without_a_record_before_the_branch_phase_refuses_the_branch_only_call(world, ref):
    w = _branch_only_world(world)
    manifest = json.loads((w["out"] / "manifest.json").read_text(encoding="utf-8"))
    change = manifest["store_changes"][0]
    change["pre_change_run_ids"].append("ffffffff-0000-0000-0000-000000000001")  # a parent that is not in records/
    change["pre_change_records"] += 1
    manifest["entry150"]["copied"]["records"] += 1  # (the declaration still counts the listed ids exactly)
    (w["out"] / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    _recheck(w["out"])
    message = _bo_refusal(world, w, r"have no records/ record made before the branch phase")
    assert message.endswith(f"{w['out']}: copied parents the manifest lists (the first store change's pre_change_run_ids) "
                            f"have no records/ record made before the branch phase")
    assert "ffffffff" not in message


def test_a_branch_rows_label_is_not_overwritten_by_a_row_with_its_run_id_in_another_period(world, ref):
    w = _branch_only_world(world, three=True)
    first_mid, last_late = w["middle"][0], max(r for r, q in w["question"].items() if q == "s01-noquery")
    f = next((w["out"] / "records").glob(f"*.{last_late}.json"))
    r = json.loads(f.read_text(encoding="utf-8"))
    r["run_id"] = first_mid  # a noquery record (copy-x's period) now carries an abstain record's run id (copy-b's)
    f.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    _recheck(w["out"])
    message = _bo_refusal(world, w, r"in different periods of the manifest's store changes", classes=_bo_classes(world))
    assert f"['records/{f.name}']" in message


def test_a_second_row_beside_a_copied_parent_refuses_the_branch_only_call(world, ref):
    w = _branch_only_world(world)
    parent = w["parents"][0]
    f = next((w["out"] / "records").glob(f"*.{parent}.json"))
    (w["out"] / "infrastructure").mkdir(exist_ok=True)
    (w["out"] / "infrastructure" / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")  # an earlier try
    _recheck(w["out"])
    message = _bo_refusal(world, w, r"share a copied parent's run id")
    assert message.endswith(f"{w['out']}: records made before the branch phase that share a copied parent's run id, so "
                            f"one of them is an attempt of the OUT's own: ['infrastructure/{f.name}', 'records/{f.name}']")


def test_a_branch_only_outs_identity_history_must_follow_the_copies_of_the_manifest_chain(world, ref):
    w = _branch_only_world(world, three=True)
    ident = json.loads((w["out"] / "deployment-identity.json").read_text(encoding="utf-8"))
    store = ident["store"]  # the last copy, EX
    other = "http://copy-d.invalid:7060/"
    # a history that joins up but goes through another copy than the manifest's chain (E0, E1, EX)
    ident["store_changes"] = [{"from": {"endpoint": E0}, "to": {"endpoint": other}},
                              {"from": {"endpoint": other}, "to": store}]
    _write_identity(w["out"], ident)
    message = _bo_refusal(world, w, r"identity records store changes that do not follow", classes=_bo_classes(world))
    assert message.endswith(f"{w['out']}: a branch-only OUT's deployment identity records store changes that do not "
                            f"follow the copies of its manifest's store changes, in the same order")
    # the same copies in the same order are accepted; a history of the right copies in the wrong order is refused
    ident["store_changes"] = [{"from": {"endpoint": E0}, "to": {"endpoint": E1}},
                              {"from": {"endpoint": E1}, "to": store}]
    _write_identity(w["out"], ident)
    assert _check(world, w["out"], w["proofs"], w["entries"], _bo_classes(world)).timelines[str(w["out"])].clean
    ident["store_changes"] = [{"from": {"endpoint": E1}, "to": {"endpoint": E0}},
                              {"from": {"endpoint": E0}, "to": store}]
    _write_identity(w["out"], ident)
    _bo_refusal(world, w, r"identity records store changes that do not follow", classes=_bo_classes(world))


def test_the_retried_output_of_a_branch_only_out_is_a_unit_row_naming_the_chain_start_and_the_new_copy(world, ref):
    """The headline output for the e150 OUTs, end to end. By hand: s01-pass and s01-flag are the copied parents (copy-a),
    s01-abstain and s01-noquery the branch rows (copy-b, though all record copy-a). Qwen's beta checker arms hold four
    units each; the one retried output of each is s01-flag, whose output is a parent (copy-a). Its curated retry record
    is made in the branch phase (09:19:00, after the change and the branch instant) and records its parent's copy-a, but
    its live retry ran on copy-b: the unit is 'copy-a -> copy-b' and the arm's five attempts are copy-a 2 (pass and the
    flag output) and copy-b 3 (the retry, abstain, noquery). The generic arm's retry stays a parent (before the change,
    copy-a): its flag unit is copy-a alone, and its five attempts are copy-a 3 and copy-b 2."""
    w = _branch_only_world(world)
    out = w["out"]
    retry = _record_of(out, "s01-flag", QWEN, "fb:curated")[1]
    assert retry["store"]["endpoint"] == E0  # it records its parent's copy
    _retime(out, lambda r, f: "2026-10-23T09:19:00.000+00:00" if r["run_id"] == retry["run_id"] else None)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    change = manifest["store_changes"][0]
    change["pre_change_run_ids"].remove(retry["run_id"])  # a live retry of the branch phase is no copied parent
    change["pre_change_records"] -= 1
    manifest["entry150"]["copied"]["records"] -= 1
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    _recheck(out)
    st = _stores(world, [out], w["proofs"], w["entries"])  # the fixtures' classes: copy-a fast, copy-b slow
    assert st["shares"][CURATED] == _rows({"copy-a": 1, ARROW_AB: 1, "copy-b": 2})
    assert st["class_shares"][CURATED] == _rows({"fast": 1, MIXED_CLASS: 1, "slow (host rule, unmeasured)": 2})
    assert st["attempt_shares"][CURATED] == _rows({"copy-a": 2, "copy-b": 3})
    assert st["shares"][GENERIC] == _rows({"copy-a": 2, "copy-b": 2})
    assert st["class_shares"][GENERIC] == _rows({"fast": 2, "slow (host rule, unmeasured)": 2})
    assert st["attempt_shares"][GENERIC] == _rows({"copy-a": 3, "copy-b": 2})
    key = str(out)
    assert st["shares_by_out"][key][CURATED] == st["shares"][CURATED]
    assert st["attempt_shares_by_out"][key][CURATED] == st["attempt_shares"][CURATED]
    # the per-change split is by the unit's own attempt: pass and the flag output are before the change; abstain and
    # noquery are branch rows (no period), whatever copy they ran on
    (split,) = st["store_changes"]
    assert split["shares_by_arm"][CURATED] == {"before": _rows({"copy-a": 1, ARROW_AB: 1}), "after": _rows({}),
                                               "branch_phase": _rows({"copy-b": 2})}
    assert [c["branch_only"] for c in st["branch_phase_copies"]] == [True]


def test_a_branch_only_out_needs_its_chain_start_copy_proven_though_every_record_names_it(world, ref):
    w = _branch_only_world(world)
    message = _bo_refusal(world, w, r"no store-proofs.json entry proves its copy", entries=w["entries"][1:])
    assert message == STORE_REFUSAL + f"{w['out']} {ex.endpoint_label(E0)}: no store-proofs.json entry proves its copy"

# --- a privately pinned graph's own copy-class table (analysis lead 5, 25 Oct): --private-copy-classes ------------------
# Synthetic only: the fixture's graph s01 is made a privately pinned graph (as a15 is), the copies are the fixtures' copy-x.


@pytest.fixture(autouse=True)
def _private_a15_root(tmp_path, monkeypatch):
    """The private a15 trees (ex.PRIVATE_A15_ROOTS) made this test's own temporary directory (mode 700, as pytest makes
    it): the tables these tests write live under it. The guard's own tests set their own roots."""
    tmp_path.chmod(0o700)
    monkeypatch.setattr(ex, "PRIVATE_A15_ROOTS", ((tmp_path, lambda host: True),))


@pytest.fixture
def private_s01(monkeypatch):
    """s01 made a privately pinned graph (PRIVATE_REFERENCE_PINS); these tests give no reference pin and read none."""
    monkeypatch.setitem(ex.PRIVATE_REFERENCE_PINS, "s01", "0" * 64)


def _private_table(tmp: Path, *, name: str = "pc", mode: int = 0o600, inside_git: bool = False,
                   copies: dict | None = None, text: str | None = None) -> tuple[Path, str]:
    """A private copy-class table written by hand (not by copy_classes.build) in a private directory, outside git unless
    ``inside_git``: (its path, its sha256)."""
    d = tmp / (f"{name}-repo" if inside_git else f"{name}-private")
    d.mkdir()
    if inside_git:
        (d / ".git").mkdir()
        (d / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    path = d / "private-classes.json"
    doc = {"schema": "e189-copy-classes/1", "rule": "a test",
           "sources": {"src": {"path": "/src.json", "sha256": "ab" * 32}},
           "copies": {"copy-x": CLASS_C} if copies is None else copies}
    path.write_text(text if text is not None else json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(mode)
    return path, _sha(path)


def _parse_classes(module, raw):
    return module.loads(raw)


def _names_of_classes(doc):
    return doc["copies"]


def test_a_private_table_is_read_for_a_privately_pinned_graph_by_its_pin_outside_git_owner_only(tmp_path, private_s01,
                                                                                              monkeypatch):
    flag = "--private-copy-classes"
    path, sha = _private_table(tmp_path)
    table = ex.read_private_table("s01", flag, path, sha, _parse_classes, _names_of_classes)
    assert (table.name, table.sha256) == ("private-classes.json", sha) and table.doc["copies"] == {"copy-x": CLASS_C}
    assert ex.read_private_table("s01", flag, None, None, _parse_classes, _names_of_classes) is None  # not given: nothing read

    def refused(match, graph="s01", given=(path, sha)):
        with pytest.raises(ex.Refused, match=match) as stop:
            ex.read_private_table(graph, flag, given[0], given[1], _parse_classes, _names_of_classes)
        assert str(tmp_path) not in str(stop.value) and "private-classes" not in str(stop.value)  # no path, no directory
        return str(stop.value)

    refused(r"given together or not at all", given=(path, None))
    refused(r"given together or not at all", given=(None, sha))
    refused(r"--private-copy-classes is given only for a privately pinned graph \(a15, s01\)", graph="rhea-142")
    for bad_sha in ("xyz", sha.upper(), sha[:-1], sha + "0", ""):
        refused(r"-sha256 is not a sha256 in lowercase hex", given=(path, bad_sha))
    refused(r"is not its pinned one \(sha256 " + "0" * 12 + r"\)", given=(path, "0" * 64))  # the pin given, not the file's hash
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")  # changed after it was pinned
    refused(r"is not its pinned one", given=(path, sha))
    for n, mode in enumerate((0o644, 0o700, 0o400, 0o660)):
        bad, bad_sha = _private_table(tmp_path, name=f"mode{n}", mode=mode)
        refused(r"must be a regular file readable and writable by its owner alone", given=(bad, bad_sha))
    inside, inside_sha = _private_table(tmp_path, name="git", inside_git=True)
    outside = r"the table given by --private-copy-classes must be outside any git work tree and not a symlink$"
    refused(outside, given=(inside, inside_sha))
    good, good_sha = _private_table(tmp_path, name="target")
    link = tmp_path / "link.json"
    link.symlink_to(good)
    refused(outside, given=(link, good_sha))
    refused(r"must be a regular file", given=(good.parent, good_sha))  # a directory
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo, 0o600)  # opened non-blocking: refused at once, never waited on
    refused(r"must be a regular file", given=(fifo, good_sha))
    with monkeypatch.context() as patched:  # the opened file's own name inside git: the second check refuses
        real = ex.outside_git
        calls = []

        def second_check_fails(candidate):
            calls.append(candidate)
            if len(calls) == 2:
                raise ex.Refused(f"{candidate}: inside the git work tree /x")
            return real(candidate)

        patched.setattr(ex, "outside_git", second_check_fails)
        refused(outside, given=(good, good_sha))
        assert len(calls) == 2 and calls[1] == good  # the name the kernel gave the opened file is the file itself
    unverifiable = r"where the table given by --private-copy-classes was opened cannot be verified here"
    with monkeypatch.context() as patched:  # an unlinked file's name, or a name that is another file's
        patched.setattr(ex.os, "readlink", lambda p: str(good) + " (deleted)")
        refused(unverifiable, given=(good, good_sha))
    with monkeypatch.context() as patched:
        patched.setattr(ex.os, "readlink", lambda p: str(inside))
        refused(unverifiable, given=(good, good_sha))
    with monkeypatch.context() as patched:
        def no_proc(p):
            raise OSError("no /proc")
        patched.setattr(ex.os, "readlink", no_proc)
        refused(unverifiable, given=(good, good_sha))
    unlinked = good.parent / "table.json (deleted)"  # a hard link of that very name: the kernel's own suffix is refused
    os.link(good, unlinked)
    with monkeypatch.context() as patched:
        patched.setattr(ex.os, "readlink", lambda p: str(unlinked))
        refused(unverifiable, given=(good, good_sha))
    with monkeypatch.context() as patched:  # an unreadable .git on the way up: a fixed refusal, never the OS error
        def unreadable(candidate):
            raise PermissionError(13, "Permission denied", str(candidate))

        patched.setattr(ex, "outside_git", unreadable)
        refused(r"the table given by --private-copy-classes cannot be checked for a git work tree$", given=(good, good_sha))
    for n, bad_name in enumerate(("/private/dir/replica-secret", "a b", "", "x" * 65, "-lead", "a/b", "é")):
        bad, bad_sha = _private_table(tmp_path, name=f"name{n}", copies={bad_name: CLASS_C})
        if bad_name == "":  # (the loader refuses an empty name before the names check)
            refused(r"is not a valid table", given=(bad, bad_sha))
        else:
            refused(r"names a copy that is not a plain name$", given=(bad, bad_sha))
    plain, plain_sha = _private_table(tmp_path, name="plain", copies={"a15.like-name_2": CLASS_C, "X9": CLASS_C})
    assert set(ex.read_private_table("s01", flag, plain, plain_sha, _parse_classes, _names_of_classes).doc["copies"]) == {
        "a15.like-name_2", "X9"}
    refused(r"cannot be opened", given=(tmp_path / "nowhere.json", good_sha))
    for n, bad_text in enumerate(("not json", "[]", json.dumps({"schema": "e189-copy-classes/1"}))):
        bad, bad_sha = _private_table(tmp_path, name=f"text{n}", text=bad_text)
        refused(r"is not a valid table \(its loader refused it\)$", given=(bad, bad_sha))  # the loader's text is not repeated
    wrong_label, wrong_sha = _private_table(tmp_path, name="label", copies={"copy-x": {**CLASS_C, "label": "slow"}})
    assert "copy-x" not in refused(r"is not a valid table \(its loader refused it\)$", given=(wrong_label, wrong_sha))


STORE_SECTION_BEFORE_THE_OPTION = "2a341d29db3c7b3128e641031b89038ffdc658cfe479e2fb45960e566be1b51c"


def test_a_call_without_the_private_table_gives_byte_for_byte_the_store_section_of_the_code_before_the_option(world):
    """The digest was computed by running this very call on commit 1ad3587a (before --private-copy-classes), the call's
    temporary directory written as <TMP>; it is no output of this code. No private graph, no private option."""
    plain = _stores(world, [world["single"]], (), [],
                    endpoint_copies=_endpoint_table(world["tmp"] / "t.json", ("s01", E0, "copy-a")))
    # The stored digest predates replay diagnostics; no routing or replay rows were supplied.
    assert plain.pop("replay_class_check") == {
        "rule": ex.REPLAY_CLASS_RULE,
        "labels": {k: v.format(taken="-") for k, v in ex.REPLAY_CLASS_LABEL_TEMPLATES.items()},
        "routing": [], "arms": {},
    }
    text = json.dumps(plain, sort_keys=True).replace(str(world["tmp"]), "<TMP>")
    assert hashlib.sha256(text.encode()).hexdigest() == STORE_SECTION_BEFORE_THE_OPTION


def _single_with_copy_x(world, private=None, **kw):
    """The single-endpoint OUT, its endpoint named copy-x by an endpoint-copies table, with the private table if given."""
    extra = {} if private is None else {"private_copy_classes": private[0], "private_copy_classes_sha256": private[1]}
    return dict(batches=[world["single"]], files=(), entries=[],
                endpoint_copies=_endpoint_table(world["tmp"] / "t.json", ("s01", E0, "copy-x")), **extra, **kw)


def test_the_private_table_gives_a_copy_the_committed_classes_lack_its_class_and_the_results_name_its_basename_and_sha(
        world, private_s01):
    base = _single_with_copy_x(world)
    _refused_stores(world, base["batches"], (), [], r"the endpoint copies name copy-x, which has no entry in the copy "
                    r"classes", endpoint_copies=base["endpoint_copies"])  # without the private table, as before
    private = _private_table(world["tmp"])
    given = _single_with_copy_x(world, private)
    st = _stores(world, given["batches"], (), [], endpoint_copies=given["endpoint_copies"],
                 private_copy_classes=private[0], private_copy_classes_sha256=private[1])
    assert set(st["copies"]) == {"copy-x"}
    for arm, share in st["shares"].items():
        k = share["attempts"]
        assert share["rows"] == {"copy-x": {"n": k, "of": k, "share": 1.0}}
        assert st["class_shares"][arm]["rows"] == {"medium": {"n": k, "of": k, "share": 1.0}}  # the private entry's class
    assert st["config"]["private_copy_classes"] == {"name": "private-classes.json", "sha256": private[1]}
    assert set(st["config"]["private_copy_classes"]) == {"name", "sha256"}
    assert str(private[0].parent) not in json.dumps(st)  # the table's directory and path are named nowhere
    classes_file = world["tmp"] / "classes.json"
    assert st["config"]["copy_classes"] == {"path": str(classes_file), "sha256": _sha(classes_file)}  # the committed one


def test_a_call_without_the_private_table_has_the_store_section_it_always_had_and_an_unused_table_changes_only_its_record(
        world, private_s01):
    endpoint_copies = _endpoint_table(world["tmp"] / "t.json", ("s01", E0, "copy-a"))  # copy-a: a committed-style entry
    plain = _stores(world, [world["single"]], (), [], endpoint_copies=endpoint_copies)
    assert set(plain["config"]) == {"path", "sha256", "entries", "matched", "reference", "copy_classes", "endpoint_copies"}
    unused = _private_table(world["tmp"], copies={"copy-z": CLASS_C})
    with_table = _stores(world, [world["single"]], (), [], endpoint_copies=endpoint_copies,
                         private_copy_classes=unused[0], private_copy_classes_sha256=unused[1])
    record = with_table["config"].pop("private_copy_classes")
    assert record == {"name": "private-classes.json", "sha256": unused[1]} and with_table == plain
    # a copy in both tables: refused unless the two entries are identical, and then the results are as without
    same = _private_table(world["tmp"], name="same", copies={"copy-a": CLASS_A})
    identical = _stores(world, [world["single"]], (), [], endpoint_copies=endpoint_copies,
                        private_copy_classes=same[0], private_copy_classes_sha256=same[1])
    assert identical["config"].pop("private_copy_classes")["sha256"] == same[1] and identical == plain
    other = _private_table(world["tmp"], name="other", copies={"copy-a": CLASS_B})
    _refused_stores(world, [world["single"]], (), [], r"^a copy is in the committed copy classes and in the private "
                    r"table with different entries$", endpoint_copies=endpoint_copies, private_copy_classes=other[0],
                    private_copy_classes_sha256=other[1])


def test_the_private_table_options_are_refused_for_a_graph_that_is_not_privately_pinned_and_unless_given_together(
        world, monkeypatch):
    private = _private_table(world["tmp"])
    _refused_stores(world, [world["single"]], (), [], r"--private-copy-classes is given only for a privately pinned "
                    r"graph", private_copy_classes=private[0], private_copy_classes_sha256=private[1])  # s01 is public here
    monkeypatch.setitem(ex.PRIVATE_REFERENCE_PINS, "s01", "0" * 64)
    _refused_stores(world, [world["single"]], (), [], r"--private-copy-classes and --private-copy-classes-sha256 are "
                    r"given together or not at all", private_copy_classes=private[0])
    _refused_stores(world, [world["single"]], (), [], r"--private-copy-classes and --private-copy-classes-sha256 are "
                    r"given together or not at all", private_copy_classes_sha256=private[1])
    _refused_stores(world, [world["single"]], (), [], r"is not its pinned one", private_copy_classes=private[0],
                    private_copy_classes_sha256="1" * 64)


def test_the_command_line_carries_the_private_table_options_to_the_call_and_check_stores_takes_none(capsys):
    args = argparse.Namespace(batch=[Path("o")], questions=[Path("q")], graph="g", split="dev", voided=[Path("v")],
                              partition_record=None, ledger=[], n_reps=None, tier_if_missing=None, cluster_map=None,
                              front_log=[], store_proof=[], reference_pin=None, partitions=None, partition_binding=None,
                              partition_plan=[], private_copy_classes=Path("/p/t.json"),
                              private_copy_classes_sha256="ab" * 32, private_endpoint_copies=None,
                              private_endpoint_copies_sha256=None, replay_routing=[])
    call = ex._call(args)
    assert (call.private_copy_classes, call.private_copy_classes_sha256) == (Path("/p/t.json"), "ab" * 32)
    args.private_copy_classes = args.private_copy_classes_sha256 = None
    assert (ex._call(args).private_copy_classes, ex._call(args).private_copy_classes_sha256) == (None, None)
    for command in ("analyse", "parity"):
        with pytest.raises(SystemExit) as stop:
            ex.main([command, "--help"])
        assert stop.value.code == 0
        out = capsys.readouterr().out
        assert "--private-copy-classes PRIVATE_COPY_CLASSES" in out and "--private-copy-classes-sha256" in out
    with pytest.raises(SystemExit) as stop:  # check-stores refuses a privately pinned graph altogether: no such option
        ex.main(["check-stores", "--code-root", str(ROOT), "--graph", "s01", "--batch", "b",
                 "--private-copy-classes", "t.json"])
    assert stop.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err


# --- the private endpoint-to-copy table: --private-endpoint-copies (the private counterpart of the endpoint copies) -----


def _private_endpoints(tmp: Path, *rows: tuple[str, str, str], name: str = "pe", mode: int = 0o600) -> tuple[Path, str]:
    """A private endpoint-copies table (rows of graph, endpoint, replica) in a private directory: (path, sha256)."""
    d = tmp / f"{name}-private"
    d.mkdir()
    path = _endpoint_table(d / "private-endpoints.json", *rows)
    path.chmod(mode)
    return path, _sha(path)


def _with_private(world, *, classes=None, endpoints=None, **kw):
    """The single-endpoint OUT run as a privately pinned graph's, with the private tables given."""
    extra = {}
    if classes is not None:
        extra.update(private_copy_classes=classes[0], private_copy_classes_sha256=classes[1])
    if endpoints is not None:
        extra.update(private_endpoint_copies=endpoints[0], private_endpoint_copies_sha256=endpoints[1])
    return extra


def test_the_private_endpoint_copies_give_a_privately_pinned_graphs_endpoint_its_replica_and_so_its_private_class(
        world, private_s01):
    tmp = world["tmp"]
    other = _endpoint_table(tmp / "committed.json", ("s01", EX, "copy-a"))  # the committed table: not this endpoint
    classes = _private_table(tmp)  # copy-x: medium
    endpoints = _private_endpoints(tmp, ("s01", E0, "copy-x"))
    label = ex.endpoint_label(E0)
    # the class table alone is inert: nothing maps the endpoint to a replica, so the label has no replica behind it
    inert = _stores(world, [world["single"]], (), [], endpoint_copies=other, **_with_private(world, classes=classes))
    assert set(inert["copies"]) == {label}
    for arm, share in inert["class_shares"].items():
        assert share["rows"].keys() == {"no copy class (no entry in the copy classes)"}
    # both tables: the endpoint is copy-x and its class is the private entry's
    st = _stores(world, [world["single"]], (), [], endpoint_copies=other,
                 **_with_private(world, classes=classes, endpoints=endpoints))
    assert set(st["copies"]) == {"copy-x"}
    for arm, share in st["shares"].items():
        k = share["attempts"]
        assert share["rows"] == {"copy-x": {"n": k, "of": k, "share": 1.0}}
        assert st["class_shares"][arm]["rows"] == {"medium": {"n": k, "of": k, "share": 1.0}}
    assert st["copies"]["copy-x"] == {"endpoints": [label], "outs": [str(world["single"])],
                                      "listed_by": {str(world["single"]): []}, "proofs": []}  # never a proof
    assert st["config"]["private_copy_classes"] == {"name": "private-classes.json", "sha256": classes[1]}
    assert st["config"]["private_endpoint_copies"] == {"name": "private-endpoints.json", "sha256": endpoints[1]}
    assert st["config"]["endpoint_copies"] == {"path": str(other), "sha256": _sha(other)}  # the committed one, as given
    dumped = json.dumps(st)
    assert E0 not in dumped and str(endpoints[0].parent) not in dumped and str(classes[0].parent) not in dumped
    # the endpoint table with a replica of the committed classes needs no private class table
    committed_replica = _private_endpoints(tmp, ("s01", E0, "copy-a"), name="pe-a")
    named = _stores(world, [world["single"]], (), [], endpoint_copies=_endpoint_table(tmp / "c2.json", ("s01", EX, "copy-b")),
                    **_with_private(world, endpoints=committed_replica))
    assert set(named["copies"]) == {"copy-a"} and "private_copy_classes" not in named["config"]
    for arm, share in named["class_shares"].items():
        assert share["rows"].keys() == {"fast"}
    # and a replica no table has a class for refuses, as for the committed endpoint copies
    _refused_stores(world, [world["single"]], (), [], r"the endpoint copies name a replica of the private endpoint "
                    r"copies, which has no entry in the copy classes", endpoint_copies=other,
                    **_with_private(world, endpoints=endpoints))  # (the private table's replica is not named)


def test_the_private_endpoint_copies_are_merged_as_one_document_and_stay_of_the_calls_graph(world, private_s01):
    tmp = world["tmp"]
    both = _endpoint_table(tmp / "both.json", ("s01", E0, "copy-a"))
    same_endpoint = _private_endpoints(tmp, ("s01", E0, "copy-b"), name="pe1")
    merge_refusal = r"^the private endpoint copies and the committed ones name an endpoint twice, or a replica for two " \
                    r"endpoints or graphs$"
    _refused_stores(world, [world["single"]], (), [], merge_refusal, endpoint_copies=both,
                    **_with_private(world, endpoints=same_endpoint))  # an endpoint in both tables
    two_endpoints = _private_endpoints(tmp, ("s01", EX, "copy-a"), name="pe2")
    _refused_stores(world, [world["single"]], (), [], merge_refusal, endpoint_copies=both,
                    **_with_private(world, endpoints=two_endpoints))  # a replica at two endpoints
    other_graph = _private_endpoints(tmp, ("another-graph", E0, "copy-x"), name="pe3")
    _refused_stores(world, [world["single"]], (), [], r"its private endpoint copies name another graph",
                    endpoint_copies=_endpoint_table(tmp / "o.json", ("s01", EX, "copy-a")),
                    **_with_private(world, endpoints=other_graph))
    # an unused private table changes only its record
    unused = _private_endpoints(tmp, ("s01", EX, "copy-x"), name="pe4")
    plain = _stores(world, [world["single"]], (), [], endpoint_copies=both)
    with_table = _stores(world, [world["single"]], (), [], endpoint_copies=both, **_with_private(world, endpoints=unused))
    assert with_table["config"].pop("private_endpoint_copies") == {"name": "private-endpoints.json", "sha256": unused[1]}
    assert with_table == plain
    # the identity's replica against the merged table: a disagreement refuses as for the committed one
    out = world["single"]
    ident = _identity(out)
    ident["store"].update(endpoint=E0, replica="copy-c", identity_check="checked")
    _write_identity(out, ident)
    _refused_stores(world, [out], (), [], r"its deployment identity names copy-c, the endpoint copies a replica of the "
                    r"private endpoint copies", endpoint_copies=_endpoint_table(tmp / "w.json", ("s01", EX, "copy-b")),
                    **_with_private(world, endpoints=_private_endpoints(tmp, ("s01", E0, "copy-a"), name="pe5")))


def test_the_private_endpoint_copies_options_are_refused_for_a_graph_that_is_not_privately_pinned_and_unless_given_together(
        world, monkeypatch):
    endpoints = _private_endpoints(world["tmp"], ("s01", E0, "copy-a"))
    _refused_stores(world, [world["single"]], (), [], r"--private-endpoint-copies is given only for a privately pinned "
                    r"graph", **_with_private(world, endpoints=endpoints))  # s01 is public here
    monkeypatch.setitem(ex.PRIVATE_REFERENCE_PINS, "s01", "0" * 64)
    _refused_stores(world, [world["single"]], (), [], r"--private-endpoint-copies and --private-endpoint-copies-sha256 "
                    r"are given together or not at all", private_endpoint_copies=endpoints[0])
    _refused_stores(world, [world["single"]], (), [], r"--private-endpoint-copies and --private-endpoint-copies-sha256 "
                    r"are given together or not at all", private_endpoint_copies_sha256=endpoints[1])
    _refused_stores(world, [world["single"]], (), [], r"is not its pinned one", private_endpoint_copies=endpoints[0],
                    private_endpoint_copies_sha256="1" * 64)
    bad = world["tmp"] / "bad-private"
    bad.mkdir()
    path = bad / "t.json"
    path.write_text(json.dumps({"schema": "e189-endpoint-copies/1", "rule": "x", "copies": [{"graph": "s01"}]}), encoding="utf-8")
    path.chmod(0o600)
    _refused_stores(world, [world["single"]], (), [], r"the table given by --private-endpoint-copies is not a valid table "
                    r"\(its loader refused it\)$", private_endpoint_copies=path, private_endpoint_copies_sha256=_sha(path))
    _refused_stores(world, [world["single"]], (), [], r"must be a regular file readable and writable by its owner alone",
                    **_with_private(world, endpoints=_private_endpoints(world["tmp"], ("s01", E0, "copy-a"), name="pe-mode",
                                                                       mode=0o644)))
    for n, bad_name in enumerate(("/private/dir/replica-secret", "two words", "a/b")):  # a name is printed as a label
        _refused_stores(world, [world["single"]], (), [], r"names a copy that is not a plain name$",
                        **_with_private(world, endpoints=_private_endpoints(world["tmp"], ("s01", E0, bad_name),
                                                                           name=f"pe-name{n}")))


def test_the_command_line_carries_the_private_endpoint_copies_options_to_the_call(capsys):
    args = argparse.Namespace(batch=[Path("o")], questions=[Path("q")], graph="g", split="dev", voided=[Path("v")],
                              partition_record=None, ledger=[], n_reps=None, tier_if_missing=None, cluster_map=None,
                              front_log=[], store_proof=[], reference_pin=None, partitions=None, partition_binding=None,
                              partition_plan=[], private_copy_classes=None, private_copy_classes_sha256=None,
                              private_endpoint_copies=Path("/p/e.json"), private_endpoint_copies_sha256="cd" * 32,
                              replay_routing=[])
    call = ex._call(args)
    assert (call.private_endpoint_copies, call.private_endpoint_copies_sha256) == (Path("/p/e.json"), "cd" * 32)
    assert (call.private_copy_classes, call.private_copy_classes_sha256) == (None, None)
    for command in ("analyse", "parity"):
        with pytest.raises(SystemExit) as stop:
            ex.main([command, "--help"])
        assert stop.value.code == 0
        out = capsys.readouterr().out
        assert "--private-endpoint-copies PRIVATE_ENDPOINT_COPIES" in out and "--private-endpoint-copies-sha256" in out


def test_a_class_entry_only_the_private_table_has_shows_fixed_words_for_its_basis_and_source_in_the_results(world, monkeypatch):
    m = _moved_world(world)  # copies a and b are proof-named, so the branch-phase report prints their class objects
    out = m["out"]
    _two_late_rows(out)
    _set_branch_phase(out, T_B)
    pin = _pin(world["tmp"], monkeypatch)  # s01 privately pinned, with the reference the proofs compare with
    committed = world["tmp"] / "committed-classes.json"
    committed.write_text(json.dumps({"schema": "e189-copy-classes/1", "rule": "a test", "sources": {
        "src": {"path": "/src.json", "sha256": "ab" * 32}}, "copies": {"copy-c": CLASS_C}}), encoding="utf-8")
    secret = {"basis": "SECRET basis of the private table /secret/dir", "source": "src"}
    private = _private_table(world["tmp"], copies={"copy-a": {**CLASS_A, **secret}, "copy-b": {**CLASS_B, **secret}})
    st = _stores(world, [out], m["proofs"], m["entries"], copy_classes=committed, reference_pin=pin,
                 private_copy_classes=private[0], private_copy_classes_sha256=private[1])
    (entry,) = st["branch_phase_copies"]
    shown = {c["copy"]: c for c in [*entry["parent_copies"], entry["branch_copy"]]}
    assert shown == {
        "copy-a": {"copy": "copy-a", "class": "fast", "label": "fast", "basis": ex.PRIVATE_BASIS, "source": ex.PRIVATE_SOURCE},
        "copy-b": {"copy": "copy-b", "class": "slow", "label": "slow (host rule, unmeasured)", "basis": ex.PRIVATE_BASIS,
                   "source": ex.PRIVATE_SOURCE}}  # the class and the label stay; the table's words do not
    dumped = json.dumps(st)
    assert "SECRET" not in dumped and "/secret/dir" not in dumped and str(private[0].parent) not in dumped
    assert st["config"]["private_copy_classes"] == {"name": "private-classes.json", "sha256": private[1]}
    # a copy both tables describe, identically, keeps the committed entry as it is
    both = world["tmp"] / "both-classes.json"
    both.write_text(json.dumps({"schema": "e189-copy-classes/1", "rule": "a test", "sources": {
        "src": {"path": "/src.json", "sha256": "ab" * 32}}, "copies": {"copy-a": CLASS_A}}), encoding="utf-8")
    same = _private_table(world["tmp"], name="same", copies={"copy-a": CLASS_A, "copy-b": {**CLASS_B, **secret}})
    st = _stores(world, [out], m["proofs"], m["entries"], copy_classes=both, reference_pin=pin,
                 private_copy_classes=same[0], private_copy_classes_sha256=same[1])
    (entry,) = st["branch_phase_copies"]
    shown = {c["copy"]: c for c in [*entry["parent_copies"], entry["branch_copy"]]}
    assert shown["copy-a"] == _copy("copy-a", CLASS_A) and shown["copy-b"]["basis"] == ex.PRIVATE_BASIS


def test_a_private_endpoint_entry_that_disagrees_with_a_store_proofs_entry_refuses_without_naming_anything_of_it(
        world, monkeypatch):
    m = _moved_world(world)  # E0 is proven as copy-a, E1 as copy-b
    pin = _pin(world["tmp"], monkeypatch)  # s01 privately pinned, with the reference the proofs compare with
    private = _private_endpoints(world["tmp"], ("s01", E0, "copy-z"), name="pe-conflict")  # copy-z at a proven endpoint
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"], r"^the private endpoint copies disagree with the "
                    r"store proofs$", reference_pin=pin, **_with_private(world, endpoints=private))
    # the committed table's conflict is named as ever
    _refused_stores(world, [m["out"]], m["proofs"], m["entries"], r"^the endpoint copies disagree with the store proofs: "
                    r".*copy-z", reference_pin=pin, endpoint_copies=_endpoint_table(world["tmp"] / "c.json", ("s01", E0, "copy-z")))
    agreeing = _private_endpoints(world["tmp"], ("s01", E0, "copy-a"), name="pe-agree")  # the proof's own replica: accepted
    st = _stores(world, [m["out"]], m["proofs"], m["entries"], reference_pin=pin, **_with_private(world, endpoints=agreeing))
    assert set(st["copies"]) == {"copy-a", "copy-b"}


# --- the private a15 trees (science, 25 Oct): a private table and a call's output live only under them -----------------


REFUSAL_OF_THE_GUARD = (r"^the table must be under a private a15 tree \(the spoke's or SiteB' private a15 directory\) "
                        r"on its host$")


def test_require_private_a15_path_accepts_only_what_is_under_a_private_root_of_this_host(tmp_path, monkeypatch):
    root = tmp_path / "private" / "a15"
    root.mkdir(parents=True)
    root.chmod(0o700)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    spoke = lambda host: host == "the-spoke"  # noqa: E731
    monkeypatch.setattr(ex, "PRIVATE_A15_ROOTS", ((root, spoke),))
    monkeypatch.setattr(ex, "_hostname", lambda: "the-spoke")
    assert ex.require_private_a15_path(root / "classes" / "out", "x") == (root / "classes" / "out").resolve()  # not there yet
    assert ex.require_private_a15_path(root, "x") == root.resolve()
    (root / "link-out").symlink_to(elsewhere)
    (elsewhere / "link-in").symlink_to(root)
    refused = [root / ".." / "elsewhere" / "t.json",  # a .. folded out of the tree
               elsewhere / "t.json", root.parent / "t.json", root.parent,
               root / "link-out" / "t.json",  # a link out of the tree
               elsewhere / "link-in" / "t.json",  # a link in from outside
               Path("/home/user/project-reviews/analysis-lead-5-20261025/private/t.json"),  # the hub's
               Path("/home/user/project-results-owner/t.json"), Path("/home/user/t.json")]
    for path in refused:
        with pytest.raises(ex.Refused, match=REFUSAL_OF_THE_GUARD) as stop:
            ex.require_private_a15_path(path, "the table")
        assert str(tmp_path) not in str(stop.value) and "www2027" not in str(stop.value)  # no path in the refusal
    monkeypatch.chdir(elsewhere)  # a relative path is the working directory's: refused from outside the tree, not from inside
    with pytest.raises(ex.Refused):
        ex.require_private_a15_path(Path("t.json"), "x")
    monkeypatch.chdir(root)
    assert ex.require_private_a15_path(Path("t.json"), "x") == (root / "t.json").resolve()
    # the HUB with a directory of the same path, owner-only: the host's name refuses it (a path alone cannot)
    monkeypatch.setattr(ex, "_hostname", lambda: "agents-host-2.cluster-a.example")
    for path in (root / "t.json", root):
        with pytest.raises(ex.Refused, match=REFUSAL_OF_THE_GUARD):
            ex.require_private_a15_path(path, "the table")
    monkeypatch.setattr(ex, "_hostname", lambda: "the-spoke")
    root.chmod(0o755)  # a root that is not owner-only is no private tree
    with pytest.raises(ex.Refused):
        ex.require_private_a15_path(root / "t.json", "x")
    root.chmod(0o700)
    alias = tmp_path / "alias-root"  # a root that is itself a symlink to an owner-only directory elsewhere
    alias.symlink_to(root)
    monkeypatch.setattr(ex, "PRIVATE_A15_ROOTS", ((alias, spoke),))
    with pytest.raises(ex.Refused):
        ex.require_private_a15_path(alias / "t.json", "x")
    monkeypatch.setattr(ex, "PRIVATE_A15_ROOTS", ((tmp_path / "no-such-root", spoke),))  # a root absent from this host
    with pytest.raises(ex.Refused):
        ex.require_private_a15_path(tmp_path / "no-such-root" / "t.json", "x")
    monkeypatch.setattr(ex, "PRIVATE_A15_ROOTS", ())
    with pytest.raises(ex.Refused):
        ex.require_private_a15_path(root / "t.json", "x")


def test_the_committed_private_a15_roots_are_the_spokes_and_siteb_private_trees_and_refuse_every_hub_path(monkeypatch):
    monkeypatch.undo()  # the real constants, not this module's fixture
    (spoke, on_spoke), (siteb, on_siteb) = ex.PRIVATE_A15_ROOTS
    assert (spoke, siteb) == (Path("/home/user/project/private/a15"),
                               Path("/scratch/user/project/private/a15"))
    # the host reports its short name or its fully qualified one by configuration: the first dot-separated label decides
    for host in ("agents-host", "agents-host.cluster-a.example", "agents-host.example"):
        assert on_spoke(host)
    for host in ("agents-host-2.cluster-a.example", "agents-host-2", "agents-host-x", "x.agents-host", "", ".agents-host"):
        assert not on_spoke(host)  # the hub is not the spoke
    for host in ("login01", "login01.cluster-b.example", "login12.cluster-b.example", "t0005", "t0005.cluster-b.example"):
        assert on_siteb(host)  # a login node or a compute node, short or fully qualified
    for host in ("agents-host", "agents-host.cluster-a.example", "agents-host-2.cluster-a.example", "t05",
                 "t00055", "login", "login01x", "xlogin01", "x.login01", ""):
        assert not on_siteb(host)
    for path in ("/home/user/project-reviews/analysis-lead-5-20261025/private/a15-copy-classes.json",
                 "/home/user/project-results-owner/x", "/home/user/project-anything/x",
                 "/home/user/wt/e189-private-classes/x", "/tmp/x", "/home/user/x",
                 "/home/user/project/private/x", "/home/user/project/private/a15-other/x"):
        with pytest.raises(ex.Refused):
            ex.require_private_a15_path(Path(path), "x")


def test_a_private_table_outside_the_private_a15_trees_is_refused_before_it_is_opened(tmp_path, private_s01, monkeypatch):
    path, sha = _private_table(tmp_path)
    monkeypatch.setattr(ex, "PRIVATE_A15_ROOTS", ((tmp_path / "another-private-tree", lambda host: True),))
    opened = []
    monkeypatch.setattr(ex.os, "open", lambda *a, **k: opened.append(a) or (_ for _ in ()).throw(AssertionError("opened")))
    with pytest.raises(ex.Refused, match=r"^s01: the table given by --private-copy-classes must be under a private a15 "
                                         r"tree \(the spoke's or SiteB' private a15 directory\) on its host$") as stop:
        ex.read_private_table("s01", "--private-copy-classes", path, sha, _parse_classes, _names_of_classes)
    assert opened == [] and str(tmp_path) not in str(stop.value)


def test_the_command_refuses_an_output_outside_the_private_a15_trees_for_a_call_with_a_private_table_before_reading_anything(
        tmp_path, monkeypatch, capsys):
    loaded = []

    def sentinel(code_root):
        loaded.append(code_root)
        raise ex.Refused("the call went past the output guard")

    monkeypatch.setattr(ex, "load_frozen", sentinel)
    base = ["analyse", "--code-root", str(tmp_path), "--graph", "s01", "--split", "dev", "--batch", str(tmp_path),
            "--questions", str(tmp_path), "--voided", str(tmp_path)]
    outside = tmp_path.parent / "not-a-private-tree" / "out"
    for option in (["--private-copy-classes", "t.json", "--private-copy-classes-sha256", "0" * 64],
                   ["--private-endpoint-copies", "t.json", "--private-endpoint-copies-sha256", "0" * 64]):
        assert ex.main([*base, *option, "--out", str(outside)]) == 2
        err = capsys.readouterr().err
        assert err.startswith("refused: --out of a call with a private table must be under a private a15 tree")
        assert str(tmp_path) not in err and loaded == []  # no path, and nothing was loaded or read
        assert ex.main([*base, *option, "--out", str(tmp_path / "out")]) == 2  # under the private root: past the guard
        assert capsys.readouterr().err == "refused: the call went past the output guard\n" and len(loaded) == 1
        loaded.clear()
    assert ex.main([*base, "--out", str(outside)]) == 2  # no private table: the output is not guarded
    assert capsys.readouterr().err == "refused: the call went past the output guard\n" and len(loaded) == 1


def test_a_table_whose_opened_file_is_in_no_private_tree_is_refused_though_the_path_given_was_in_one(tmp_path, private_s01,
                                                                                                    monkeypatch):
    good, sha = _private_table(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-the-private-tree"
    outside.mkdir()
    linked = outside / "linked.json"
    os.link(good, linked)  # the same file (device and inode) under a name that is in no private tree
    monkeypatch.setattr(ex.os, "readlink", lambda p: str(linked))  # as if a parent had been redirected after the first check
    with pytest.raises(ex.Refused, match=r"^s01: the table given by --private-copy-classes as opened must be under a "
                                         r"private a15 tree \(the spoke's or SiteB' private a15 directory\) on its host$") as stop:
        ex.read_private_table("s01", "--private-copy-classes", good, sha, _parse_classes, _names_of_classes)
    assert str(tmp_path) not in str(stop.value) and str(outside) not in str(stop.value)
    shutil.rmtree(outside)


def _stub_the_analysis(monkeypatch):
    """main() with the frozen code and the analysis stubbed: only its output handling is under test."""
    from types import SimpleNamespace
    result = {"arms": [1], "settings": {"n_questions": 1}, "exploratory": {"clusters": {"clusters": 1}, "graph": "s01",
                                                                             "partition_union": None},
              "deployments": {"pairs_compared": 0, "triggered_pairs": []}, "stores": {"copies": {}}}
    loaded = []
    monkeypatch.setattr(ex, "load_frozen", lambda root: loaded.append(root) or SimpleNamespace(refusals=ex.Refused))
    monkeypatch.setattr(ex, "run", lambda F, call: result)
    return result, loaded


def _analyse(tmp_path, out, *private):
    return ex.main(["analyse", "--code-root", str(tmp_path), "--graph", "s01", "--split", "dev", "--batch", str(tmp_path),
                    "--questions", str(tmp_path), "--voided", str(tmp_path), *private, "--out", str(out)])


PRIVATE_PATH_OPTIONS = ("--private-copy-classes", "t.json", "--private-copy-classes-sha256", "0" * 64)


def test_a_call_with_a_private_table_writes_its_results_into_the_checked_directory_owner_only_and_names_no_path(
        tmp_path, monkeypatch, capsys):
    result, loaded = _stub_the_analysis(monkeypatch)
    out = tmp_path / "out"
    assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 0
    written = out / "results.json"
    assert json.loads(written.read_text(encoding="utf-8")) == result and written.stat().st_mode & 0o777 == 0o600
    assert out.stat().st_mode & 0o777 == 0o700
    stdout = capsys.readouterr().out
    assert stdout.startswith("results.json: 1 arms") and str(tmp_path) not in stdout  # the file, not where it is
    assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 0  # a second call overwrites its own file
    capsys.readouterr()
    # without a private option nothing changes: the path is printed and the mode is the default
    plain = tmp_path.parent / f"{tmp_path.name}-plain-out"
    assert _analyse(tmp_path, plain) == 0
    assert capsys.readouterr().out.startswith(f"{plain / 'results.json'}: 1 arms")
    shutil.rmtree(plain)


def test_an_existing_results_file_of_any_kind_is_replaced_never_written_through_waited_on_or_kept(tmp_path, monkeypatch, capsys):
    """A symlink, a hard link to a file elsewhere, a FIFO and a file of another mode: each is replaced by a fresh
    owner-only regular file; what it linked to is untouched; nothing blocks."""
    import signal
    result, _ = _stub_the_analysis(monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    elsewhere = tmp_path.parent / f"{tmp_path.name}-elsewhere"
    elsewhere.mkdir()
    victim = elsewhere / "victim.json"
    results = out / "results.json"

    def fresh_and_victim_untouched():
        assert results.is_file() and not results.is_symlink() and results.stat().st_nlink == 1
        assert json.loads(results.read_text(encoding="utf-8")) == result and results.stat().st_mode & 0o777 == 0o600
        assert victim.read_text(encoding="utf-8") == "keep" if victim.exists() else True
        assert [f.name for f in out.iterdir()] == ["results.json"]  # no temporary left behind

    def run_once():
        signal.alarm(20)  # a write that waits on a FIFO would hang here: fail instead
        try:
            assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 0
        finally:
            signal.alarm(0)
        capsys.readouterr()

    victim.write_text("keep", encoding="utf-8")
    results.symlink_to(victim)
    run_once()
    fresh_and_victim_untouched()
    results.unlink()
    os.link(victim, results)  # a hard link to a file outside the private tree
    assert results.stat().st_nlink == 2
    run_once()
    fresh_and_victim_untouched()
    results.unlink()
    os.mkfifo(results, 0o600)  # a FIFO: opening it for writing would wait for a reader
    run_once()
    fresh_and_victim_untouched()
    results.chmod(0o644)  # an existing file of another mode
    run_once()
    fresh_and_victim_untouched()
    results.unlink()
    results.mkdir()  # a directory of that name cannot be replaced: a fixed refusal, no temporary left
    (results / "x").write_text("x", encoding="utf-8")
    assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 2
    err = capsys.readouterr().err
    assert err == "refused: the output file of a call with a private table cannot replace the file there\n"
    assert sorted(f.name for f in out.iterdir()) == ["results.json"] and (results / "x").exists()
    shutil.rmtree(elsewhere)


def test_a_directory_swapped_for_a_link_after_the_guard_cannot_redirect_the_write(tmp_path, monkeypatch, capsys):
    _stub_the_analysis(monkeypatch)
    elsewhere = tmp_path.parent / f"{tmp_path.name}-elsewhere"
    elsewhere.mkdir()
    swapped = tmp_path / "swapped"
    swapped.symlink_to(elsewhere)  # the guard hands back a link: the directory is opened without following it
    monkeypatch.setattr(ex, "check_private_out", lambda path: swapped)  # (the preflight would refuse a link itself)
    assert _analyse(tmp_path, tmp_path / "anything", *PRIVATE_PATH_OPTIONS) == 2
    assert capsys.readouterr().err == "refused: the output directory of a call with a private table cannot be opened or made\n"
    assert list(elsewhere.iterdir()) == []
    shutil.rmtree(elsewhere)


def test_the_output_directory_is_made_one_level_inside_a_checked_parent_and_no_failure_names_a_path(tmp_path, monkeypatch, capsys):
    _stub_the_analysis(monkeypatch)
    fixed = r"^the output directory of a call with a private table cannot be opened or made$"
    assert _analyse(tmp_path, tmp_path / "new", *PRIVATE_PATH_OPTIONS) == 0  # one level, made owner-only
    assert (tmp_path / "new").stat().st_mode & 0o777 == 0o700 and (tmp_path / "new" / "results.json").is_file()
    capsys.readouterr()
    # the writer itself (the command's preflight refuses these earlier): two levels, a file for the parent or the directory
    for path in (tmp_path / "a" / "b", tmp_path / "file" / "b", tmp_path / "file"):
        (tmp_path / "file").write_text("x", encoding="utf-8")
        with pytest.raises(ex.Refused, match=fixed) as stop:
            ex.write_in_private_dir(path, "results.json", "x", create=True)
        assert str(tmp_path) not in str(stop.value)
    assert not (tmp_path / "a").exists()
    # a parent that is changed after the guard: its own name, as opened, is checked again and nothing is made through it
    outside = tmp_path.parent / f"{tmp_path.name}-outside-parent"
    outside.mkdir()
    real_readlink = os.readlink
    monkeypatch.setattr(ex.os, "readlink", lambda p: str(outside) if str(p).startswith("/proc/self/fd/") else real_readlink(p))
    assert _analyse(tmp_path, tmp_path / "made-through-a-redirect", *PRIVATE_PATH_OPTIONS) == 2
    err = capsys.readouterr().err
    assert err.startswith("refused: the output directory of a call with a private table as opened must be under a private "
                          "a15 tree") and str(tmp_path) not in err
    assert list(outside.iterdir()) == [] and not (tmp_path / "made-through-a-redirect").exists()
    shutil.rmtree(outside)


def test_the_directory_really_opened_is_checked_against_the_private_trees_again(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside-dir"
    outside.mkdir()
    monkeypatch.setattr(ex.os, "readlink", lambda p: str(outside))
    with pytest.raises(ex.Refused, match=r"^the output directory of a call with a private table as opened must be under "
                                         r"a private a15 tree") as stop:
        ex.write_in_private_dir(out, "results.json", "x")
    assert str(tmp_path) not in str(stop.value) and list(out.iterdir()) == []
    monkeypatch.setattr(ex.os, "readlink", lambda p: (_ for _ in ()).throw(OSError("no /proc")))
    with pytest.raises(ex.Refused, match=r"cannot be verified here"):
        ex.write_in_private_dir(out, "results.json", "x")
    shutil.rmtree(outside)


def test_every_private_table_option_alone_triggers_the_output_guard_before_anything_is_loaded(tmp_path, monkeypatch, capsys):
    _, loaded = _stub_the_analysis(monkeypatch)
    outside = tmp_path.parent / f"{tmp_path.name}-not-private" / "out"
    for option in ("--private-copy-classes", "--private-endpoint-copies"):
        assert _analyse(tmp_path, outside, option, "t.json") == 2  # the path alone, without its sha256
        assert capsys.readouterr().err.startswith("refused: --out of a call with a private table must be under a private a15 tree")
    for option in ("--private-copy-classes-sha256", "--private-endpoint-copies-sha256"):
        assert _analyse(tmp_path, outside, option, "0" * 64) == 2  # the sha256 alone, without its path
        assert capsys.readouterr().err.startswith("refused: --out of a call with a private table must be under a private a15 tree")
    assert loaded == []


def test_a_failed_write_sync_or_mode_is_a_fixed_refusal_with_no_temporary_left_and_the_mode_is_exact(tmp_path, monkeypatch, capsys):
    _stub_the_analysis(monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    fixed = "refused: the output file of a call with a private table cannot be written there\n"
    for name in ("fsync", "fchmod"):
        with monkeypatch.context() as patched:
            patched.setattr(ex.os, name, lambda *a, **k: (_ for _ in ()).throw(OSError(28, "No space left on device")))
            assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 2
        err = capsys.readouterr().err
        assert err == fixed and str(tmp_path) not in err and list(out.iterdir()) == []
    with monkeypatch.context() as patched:  # the write itself fails
        patched.setattr(ex.os, "write", lambda *a, **k: (_ for _ in ()).throw(OSError(5, "I/O error")))
        assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 2
    assert capsys.readouterr().err == fixed and list(out.iterdir()) == []
    with monkeypatch.context() as patched:  # a short write: the rest is written, not dropped
        real_write = os.write
        patched.setattr(ex.os, "write", lambda fd, view: real_write(fd, bytes(view)[:7]))
        assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 0
    capsys.readouterr()
    assert json.loads((out / "results.json").read_text(encoding="utf-8"))["arms"] == [1]
    old_umask = os.umask(0o277)  # a umask that would leave 0o400 on a file created with mode 600
    try:
        assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 0
    finally:
        os.umask(old_umask)
    assert (out / "results.json").stat().st_mode & 0o777 == 0o600


def test_the_directory_is_checked_again_just_before_the_rename_and_a_made_directory_that_does_not_check_out_is_removed(
        tmp_path, monkeypatch, capsys):
    _stub_the_analysis(monkeypatch)
    outside = tmp_path.parent / f"{tmp_path.name}-moved-to"
    outside.mkdir()
    real_readlink = os.readlink
    seen = []

    def moved_after(n_good: int):
        def readlink(p):
            if not str(p).startswith("/proc/self/fd/"):
                return real_readlink(p)
            seen.append(p)
            return real_readlink(p) if len(seen) <= n_good else str(outside)
        return readlink

    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(ex.os, "readlink", moved_after(1))  # the open check passes, the check before the rename sees it moved
    assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 2
    err = capsys.readouterr().err
    assert err.startswith("refused: the output directory of a call with a private table as opened must be under a private "
                          "a15 tree") and str(tmp_path) not in err
    assert list(out.iterdir()) == [] and list(outside.iterdir()) == []  # no results.json, no temporary
    seen.clear()
    fresh = tmp_path / "fresh"
    monkeypatch.setattr(ex.os, "readlink", moved_after(1))  # the parent checks out; the made directory does not: removed
    assert _analyse(tmp_path, fresh, *PRIVATE_PATH_OPTIONS) == 2
    capsys.readouterr()
    assert not fresh.exists() and list(outside.iterdir()) == []
    shutil.rmtree(outside)


def _open_descriptors() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_an_unencodable_result_a_failing_name_or_close_and_an_unforeseen_exception_are_each_one_fixed_outcome_with_nothing_leaked(
        tmp_path, monkeypatch, capsys):
    result, _ = _stub_the_analysis(monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    written = "refused: the output file of a call with a private table cannot be written there\n"
    created = "refused: the output file of a call with a private table cannot be created there\n"
    before = _open_descriptors()
    result["arms"] = ["\ud800"]  # an unpaired surrogate: ensure_ascii=False keeps it, UTF-8 cannot encode it
    assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 2
    assert capsys.readouterr().err == "refused: the output of a call with a private table cannot be encoded as UTF-8\n"
    assert list(out.iterdir()) == []
    result["arms"] = [1]
    for failure in (OSError(5, "no entropy"), NotImplementedError("no urandom")):
        with monkeypatch.context() as patched:
            patched.setattr(ex.secrets, "token_hex", lambda n, failure=failure: (_ for _ in ()).throw(failure))
            assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 2
        assert capsys.readouterr().err == created and list(out.iterdir()) == []
    real_close = os.close
    with monkeypatch.context() as patched:  # closing the file fails: after really closing it
        patched.setattr(ex.os, "close", lambda fd: (real_close(fd), (_ for _ in ()).throw(OSError(5, "I/O error")))[0])
        assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 2
    assert capsys.readouterr().err == written and list(out.iterdir()) == [] and _open_descriptors() == before
    directories = set()
    real_open = os.open
    with monkeypatch.context() as patched:  # closing only the directory's own descriptor fails: ignored, the write stands
        def recording_open(path, flags, *a, **k):
            fd = real_open(path, flags, *a, **k)
            if flags & os.O_DIRECTORY:
                directories.add(fd)
            return fd

        patched.setattr(ex.os, "open", recording_open)
        patched.setattr(ex.os, "close", lambda fd: (real_close(fd), (directories.discard(fd), (_ for _ in ()).throw(OSError(5, "I/O error"))) if fd in directories else None)[0])
        assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 0
    assert capsys.readouterr().out.startswith("results.json: ") and json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert _open_descriptors() == before
    with monkeypatch.context() as patched:  # anything unforeseen: its class only, never its text (which would hold a path)
        patched.setattr(ex, "write_in_private_dir", lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"secret {tmp_path}")))
        assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 3
    err = capsys.readouterr().err
    assert err == "results of a call with a private table: not written (RuntimeError)\n" and str(tmp_path) not in err


def test_no_unforeseen_error_escapes_the_guard_the_table_reader_or_the_writer_as_anything_but_a_fixed_refusal(
        tmp_path, private_s01, monkeypatch, capsys):
    guard = r"^the table must be under a private a15 tree \(the spoke's or SiteB' private a15 directory\) on its host$"
    with pytest.raises(ex.Refused, match=guard):  # a NUL byte in the path: ValueError inside the guard
        ex.require_private_a15_path(Path("a\0b"), "the table")
    secret = f"secret {tmp_path}"
    for failure in (RuntimeError(secret), OSError(13, "Permission denied", str(tmp_path)), ValueError(secret)):
        with monkeypatch.context() as patched:
            patched.setattr(ex.Path, "resolve", lambda self, *a, failure=failure, **k: (_ for _ in ()).throw(failure))
            with pytest.raises(ex.Refused, match=guard) as stop:
                ex.require_private_a15_path(tmp_path / "t.json", "the table")
            assert str(tmp_path) not in str(stop.value) and "secret" not in str(stop.value)
    _stub_the_analysis(monkeypatch)  # the early --out guard of the command: a NUL byte is a refusal, not a traceback
    assert ex.main(["analyse", "--code-root", str(tmp_path), "--graph", "s01", "--split", "dev", "--batch", str(tmp_path),
                    "--questions", str(tmp_path), "--voided", str(tmp_path), *PRIVATE_PATH_OPTIONS, "--out", "a\0b"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("refused: --out of a call with a private table must be under a private a15 tree") and "\0" not in err
    # the table reader: a file close that fails after the bytes were read
    path, sha = _private_table(tmp_path)
    real_close = os.close
    flag = "--private-copy-classes"
    with monkeypatch.context() as patched:
        patched.setattr(ex.os, "close", lambda fd: (real_close(fd), (_ for _ in ()).throw(OSError(5, "I/O", str(path))))[0])
        with pytest.raises(ex.Refused, match=r"^s01: the table given by --private-copy-classes could not be read$") as stop:
            ex.read_private_table("s01", flag, path, sha, _parse_classes, _names_of_classes)
        assert str(tmp_path) not in str(stop.value)
    # the output directory: a close that fails while a refusal is being raised (check fails, then the close fails too)
    out = tmp_path / "out"
    out.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside-close"
    outside.mkdir()
    real_readlink = os.readlink
    with monkeypatch.context() as patched:
        patched.setattr(ex.os, "readlink", lambda p: str(outside) if str(p).startswith("/proc/self/fd/") else real_readlink(p))
        patched.setattr(ex.os, "close", lambda fd: (real_close(fd), (_ for _ in ()).throw(OSError(5, "I/O", str(out))))[0])
        with pytest.raises(ex.Refused, match=r"^the output file of a call with a private table could not be written$") as stop:
            ex.write_in_private_dir(out, "results.json", "x")
        assert str(tmp_path) not in str(stop.value)
    shutil.rmtree(outside)
    assert [f.name for f in out.iterdir()] == []


def test_a_root_with_a_symlinked_ancestor_accepts_the_configured_and_the_physical_name_and_still_refuses_what_it_should(
        tmp_path, private_s01, monkeypatch, capsys):
    """The kernel names an opened file by its physical path: when a directory above the configured root is a symlink the
    physical name is not under the configured root. Both spellings are accepted, the checks on the opened table and
    directory pass, and a link out, a link in and a .. out are refused still."""
    physical = tmp_path / "real" / "private" / "a15"
    physical.mkdir(parents=True)
    physical.chmod(0o700)
    (tmp_path / "link").symlink_to(tmp_path / "real")  # an ancestor of the configured root is a symlink
    configured = tmp_path / "link" / "private" / "a15"
    spoke = lambda host: True  # noqa: E731
    monkeypatch.setattr(ex, "PRIVATE_A15_ROOTS", ((configured, spoke),))
    assert not configured.is_symlink() and configured.resolve() == physical.resolve()
    for given in (configured / "t.json", physical / "t.json", configured, configured / "new" / "t.json"):
        assert ex.require_private_a15_path(given, "x") == (physical / given.relative_to(configured if str(given).startswith(str(configured)) else physical)).resolve()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "link-in").symlink_to(configured)
    (configured / "link-out").symlink_to(elsewhere)
    for bad in (elsewhere / "t.json", elsewhere / "link-in" / "t.json", configured / "link-out" / "t.json",
                configured / ".." / ".." / "t.json", physical / ".." / "t.json", tmp_path / "link" / "t.json"):
        with pytest.raises(ex.Refused):
            ex.require_private_a15_path(bad, "x")
    # a table under it is read (the opened file's physical name is checked again), the results are written there
    table = configured / "classes"
    table.mkdir()
    path = table / "private-classes.json"
    path.write_text(json.dumps({"schema": "e189-copy-classes/1", "rule": "a test", "sources": {
        "src": {"path": "/src.json", "sha256": "ab" * 32}}, "copies": {"copy-x": CLASS_C}}, indent=1) + "\n", encoding="utf-8")
    path.chmod(0o600)
    read = ex.read_private_table("s01", "--private-copy-classes", path, _sha(path), _parse_classes, _names_of_classes)
    assert read.doc["copies"] == {"copy-x": CLASS_C}
    _stub_the_analysis(monkeypatch)
    assert _analyse(tmp_path, configured / "results-out", "--private-copy-classes", str(path),
                    "--private-copy-classes-sha256", _sha(path)) == 0
    written = physical / "results-out" / "results.json"
    assert written.is_file() and written.stat().st_mode & 0o777 == 0o600
    capsys.readouterr()


def test_the_output_of_a_call_with_a_private_table_must_be_outside_git_and_somewhere_that_can_be_written(
        tmp_path, monkeypatch, capsys):
    _, loaded = _stub_the_analysis(monkeypatch)
    start = r"^refused: --out of a call with a private table "
    inside = tmp_path / "repo"
    (inside / ".git").mkdir(parents=True)
    (inside / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    for out in (inside / "out", inside):
        assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 2
        err = capsys.readouterr().err
        assert re.match(start + r"must be outside any git work tree\n$", err) and str(tmp_path) not in err
    assert _analyse(tmp_path, tmp_path / "a" / "b", *PRIVATE_PATH_OPTIONS) == 2  # neither it nor its parent exists
    err = capsys.readouterr().err
    assert re.match(start + r"must be an existing directory or a new one in an existing directory\n$", err)
    (tmp_path / "plain").write_text("x", encoding="utf-8")
    for out in (tmp_path / "plain", tmp_path / "plain" / "b"):  # a file; a directory below a file
        assert _analyse(tmp_path, out, *PRIVATE_PATH_OPTIONS) == 2
        assert re.match(start + r"must be an existing directory or a new one", capsys.readouterr().err)
    (tmp_path / "dangling").symlink_to(tmp_path.parent / f"{tmp_path.name}-nowhere-outside")  # a link out, to nothing
    assert _analyse(tmp_path, tmp_path / "dangling", *PRIVATE_PATH_OPTIONS) == 2
    assert re.match(start + r"must be under a private a15 tree", capsys.readouterr().err) and loaded == []  # all refused first
    assert _analyse(tmp_path, tmp_path / "ok", *PRIVATE_PATH_OPTIONS) == 0  # a new directory in an existing one
    assert _analyse(tmp_path, tmp_path / "ok", *PRIVATE_PATH_OPTIONS) == 0  # an existing one
    (tmp_path / "inner-link").symlink_to(tmp_path / "ok")  # a link that stays inside the tree: judged by where it ends
    assert _analyse(tmp_path, tmp_path / "inner-link", *PRIVATE_PATH_OPTIONS) == 0
    assert len(loaded) == 3 and (tmp_path / "ok" / "results.json").is_file()


def test_the_success_line_of_a_call_with_a_private_table_leaves_out_the_store_copy_count_and_a_plain_call_keeps_it(
        tmp_path, monkeypatch, capsys):
    _stub_the_analysis(monkeypatch)
    assert _analyse(tmp_path, tmp_path / "out", *PRIVATE_PATH_OPTIONS) == 0
    assert capsys.readouterr().out == ("results.json: 1 arms over 1 questions in 1 clusters (s01); 0 arm pairs compared, "
                                       "0 over 10 points\n")
    plain = tmp_path.parent / f"{tmp_path.name}-plain-line"
    assert _analyse(tmp_path, plain) == 0
    assert capsys.readouterr().out == (f"{plain / 'results.json'}: 1 arms over 1 questions in 1 clusters (s01); 0 arm pairs "
                                       f"compared, 0 over 10 points; 0 store copies\n")
    shutil.rmtree(plain)


def test_an_empty_private_sha256_option_alone_triggers_the_output_guard_and_a_closed_parent_leaks_no_descriptor(
        tmp_path, monkeypatch, capsys):
    _, loaded = _stub_the_analysis(monkeypatch)
    outside = tmp_path.parent / f"{tmp_path.name}-not-private-either" / "out"
    for option in ("--private-copy-classes-sha256", "--private-endpoint-copies-sha256"):
        assert _analyse(tmp_path, outside, option, "") == 2  # an empty string is a given option
        assert capsys.readouterr().err.startswith("refused: --out of a call with a private table must be under a private a15 tree")
    assert loaded == []
    before = _open_descriptors()
    real_close, real_open, directories = os.close, os.open, set()
    with monkeypatch.context() as patched:  # a new output directory: its parent's close fails right after the mkdir
        def recording_open(path, flags, *a, **k):
            fd = real_open(path, flags, *a, **k)
            if flags & os.O_DIRECTORY:
                directories.add(fd)
            return fd

        patched.setattr(ex.os, "open", recording_open)
        patched.setattr(ex.os, "close", lambda fd: (real_close(fd), (directories.discard(fd), (_ for _ in ()).throw(OSError(5, "I/O"))) if fd in directories else None)[0])
        assert _analyse(tmp_path, tmp_path / "made-here", *PRIVATE_PATH_OPTIONS) == 0
    capsys.readouterr()
    assert (tmp_path / "made-here" / "results.json").is_file() and _open_descriptors() == before

