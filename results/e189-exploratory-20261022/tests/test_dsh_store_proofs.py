"""Tests of the six DeepSeek opaque partitions' entries in store-proofs.json (analysis lead 5, 25 Oct ~09:5xZ): the OUTs
opaque-deepseek-v4-flash-0731-dsh-p0of6 .. p5of6, each a whole new OUT on one of the arm64 in-GPU copies
opaque-rhea-siteb-22 .. -25 (entry 202: ruled not identical to the x86 copies, so each is proven by its OWN identity storecheck
and that copy's check-copies known answers, never by an x86 copy's proof).

The expectation is typed by hand from sources the entries were not built from: the committed plans of the partitions
(feat/deepseek-opaque-partitions, plan-p<i>.json: the copy and the endpoint, here as the sha256 of the endpoint string), the
sha256 of the storecheck each OUT's identity records, and the sha256 of the check-copies files. The checks against the cited
files run where those files are (the hub) and skip elsewhere; each also runs the analysis' own proof (exploratory.prove)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
ROOT = RESULTS.parents[1]

_spec = importlib.util.spec_from_file_location("e189_exploratory_dsh_proofs_under_test", RESULTS / "exploratory.py")
ex = importlib.util.module_from_spec(_spec)
sys.modules["e189_exploratory_dsh_proofs_under_test"] = ex
_spec.loader.exec_module(ex)

GRAPH = "opaque-rhea-142"
RUNS = "/scratch/user/project/runs/test-20261021/"
# partition -> (copy, sha256 of the plan's endpoint string, sha256 of the storecheck its identity records,
#               sha256 of the check-copies file nearest before its identity). Typed by hand.
PLANS = {
    0: ("opaque-rhea-siteb-22", "24717311ff5196c2a1ce5984e187c375fcbddf233509f47a329e145981681f95",
        "6c6547688d638c38e8a576a29f030a9ccd5bff894a6ec539895add9fee8b7e67",
        "d3687c759e3ea315afb76342088e5aa38c02937df3933ac464e601f48e298da2"),
    1: ("opaque-rhea-siteb-25", "76a9e4136c220e8f1fea30085076760e6273fe85a6a9deab0902064228c3ab3b",
        "9f04dcfb41a798476b605d0ef6031798047d9cc49df4a107b440c70304dee04d",
        "01fc704202ee34bd50c82bb35b041ac31aac3339f3512f2122d7892d0b033fd4"),
    2: ("opaque-rhea-siteb-24", "7b3a1cb9f81c487cff014463152bbdcdeb7da30819dfd9171dfd48bc239efa87",
        "c31f3f00b52053110dc174f711e1112f76adcf59a360fd2a6a4b261dbd372b38",
        "d3687c759e3ea315afb76342088e5aa38c02937df3933ac464e601f48e298da2"),
    3: ("opaque-rhea-siteb-22", "24717311ff5196c2a1ce5984e187c375fcbddf233509f47a329e145981681f95",
        "16bead5740fafd25305f244abf469be43e97a18068b888ca15b0fe71e943943c",
        "d3687c759e3ea315afb76342088e5aa38c02937df3933ac464e601f48e298da2"),
    4: ("opaque-rhea-siteb-23", "582c24669751232535d4d4bf3abead0fae77c2a929c11435f21ed5bfd4f7af44",
        "0446695c5e6e74e362f7ec57f2c31f5fc03c50302ddd98cbbdbc135fd07e37a0",
        "01fc704202ee34bd50c82bb35b041ac31aac3339f3512f2122d7892d0b033fd4"),
    5: ("opaque-rhea-siteb-25", "76a9e4136c220e8f1fea30085076760e6273fe85a6a9deab0902064228c3ab3b",
        "b5b892a87bf6fd31f4cf685b83a8d4448f76aae152326d5ff2ee8024f388fc72",
        "01fc704202ee34bd50c82bb35b041ac31aac3339f3512f2122d7892d0b033fd4"),
}
RULING_SHA = "63ad6b06136b33eb7e39b3ba78e69c68155297c0e5eb3c1e745e44e310deb3d7"  # entry 202's design change-log
# the launch step's hub copies of the six OUTs' deployment identities (typed by hand: sha256sum of each file)
IDENTITIES = Path("/home/user/project-reviews/deepseek-direct/opaque/private/identities")
IDENTITY_SHA = {
    0: "772916b6e989178cd0544c2b52cf57846b6cc7de31a8572987fc6c89af653029",
    1: "4a5ac62b124c96f66fe990d1b3662efc08e9cf902b76566af41285d44d54447e",
    2: "5a32b903a55a06cd656a19ad94a29495558b4b46692077cd2f82636c574a752f",
    3: "26151044ab1c252546ccb7a0159f4b57e1459c09caa288379018de0b64cbef7b",
    4: "3b7a2cbded8e38792b9d4e2207a440effbcecc1f46f0ae699def53e8d15d3361",
    5: "b931f3ad5bc2e46ed745550bb7f845f743722b4e3b9b957784e7a2607e31f152"}
BASE = "67c9c041"  # the merge that held 43 entries and no dsh entry
FACTS = Path("/home/user/project-reviews/store-copies-20261021/ct9/dsh-facts")


def _entries() -> list[dict]:
    return json.loads((RESULTS / "store-proofs.json").read_text(encoding="utf-8"))["proofs"]


def _dsh() -> dict[int, dict]:
    found = {int(e["out"].rsplit("-p", 1)[1].split("of")[0]): e for e in _entries() if "-dsh-p" in e["out"]}
    assert len(found) == len([e for e in _entries() if "-dsh-p" in e["out"]])  # one entry per OUT
    return found


def test_the_six_entries_are_the_plans_copies_with_the_hand_typed_hashes():
    dsh = _dsh()
    assert sorted(dsh) == [0, 1, 2, 3, 4, 5]
    for i, (copy, endpoint_sha, storecheck_sha, known_sha) in PLANS.items():
        e = dsh[i]
        assert e["out"] == f"{RUNS}opaque-deepseek-v4-flash-0731-dsh-p{i}of6"
        assert (e["graph"], e["replica"], e["endpoint_sha256"]) == (GRAPH, copy, endpoint_sha)
        assert (e["storecheck_sha256"], e["known_answers_sha256"]) == (storecheck_sha, known_sha)
        # a whole new OUT on its first copy: no move, no forward, no alias, no ident.py, no later period
        assert not {"change_at", "behind_endpoint", "second_period", "forward", "alias", "further", "identity_format",
                    "not_checked"} & set(e)
        assert "entry 202" in e["change_log"] and "never an x86 copy's proof" in e["change_log"]
        src = e["sources"]
        assert src["storecheck"]["sha256"] == storecheck_sha and src["known_answers"]["sha256"] == known_sha
        # the stager (driver/stage_partition_union.sh) accepts no other source key, so the identity copy and the ruling are cited
        # in the endpoint text and the change_log, each with its sha256
        assert set(src) == {"endpoint", "storecheck", "known_answers"}
        assert IDENTITY_SHA[i] in src["endpoint"] and f"-dsh-p{i}of6.deployment-identity.json" in src["endpoint"]
        assert RULING_SHA in e["change_log"]
    # p0 and p3 share copy -22, p1 and p5 share -25: one storecheck per OUT, never one OUT's for another
    assert len({e["storecheck_sha256"] for e in dsh.values()}) == 6


def test_the_file_reads_and_no_other_entry_changed():
    entries, _ = ex.read_store_proofs(RESULTS / "store-proofs.json")
    assert len(entries) == 49 and len([e for e in entries if "-dsh-p" in e["out"]]) == 6
    shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE}:results/e189-exploratory-20261022/store-proofs.json"],
                           capture_output=True, check=False)
    if shown.returncode != 0:
        pytest.skip(f"{BASE} is not in this repository")
    base = json.loads(shown.stdout)["proofs"]
    assert len(base) == 43 and entries[:43] == base  # the six are appended; every earlier entry is as it was


def _cited(entry: dict) -> dict[str, tuple[str, object]]:
    """The entry's cited files by sha256, each checked against the sha256 the entry names (a skip when absent)."""
    files = {}
    for kind in ("storecheck", "known_answers"):
        path = Path(entry["sources"][kind]["hub"])
        if not path.is_file():
            pytest.skip(f"{path.name} is not on this machine")
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry[f"{kind}_sha256"] == entry["sources"][kind]["sha256"]
        files[entry[f"{kind}_sha256"]] = (str(path), json.loads(raw))
    return files


def _identity(entry: dict) -> dict:
    i = int(entry["out"].rsplit("-p", 1)[1].split("of")[0])
    path = IDENTITIES / f"tier2-opaque-deepseek-v4-flash-0731-dsh-p{i}of6.deployment-identity.json"
    if not path.is_file():
        pytest.skip(f"{path.name} is not on this machine")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == IDENTITY_SHA[i]
    return json.loads(raw)["store"]


@pytest.mark.parametrize("i", sorted(PLANS))
def test_each_entry_passes_the_analysis_own_proof_against_its_cited_files(i):
    e = _dsh()[i]
    files = _cited(e)
    store = _identity(e)
    # the OUT's identity (a hub copy) agrees with the plan on the copy and the endpoint, and records this very storecheck
    assert store["replica"] == e["replica"] == PLANS[i][0]
    assert hashlib.sha256(store["endpoint"].encode()).hexdigest() == e["endpoint_sha256"] == PLANS[i][1]
    recorded = ex.RECORDED_CHECK.search(store["identity_check"])
    assert recorded and recorded["sha"] == e["storecheck_sha256"] == PLANS[i][2]
    assert ex.prove(e, GRAPH, store["endpoint"], files, store["replica"], recorded["sha"], False) == []


def test_a_wrong_file_is_refused_by_the_same_proof():
    """The proof can fail: the earlier check-copies runs that were not ok, another copy's run, another OUT's storecheck."""
    e = _dsh()[0]
    files = _cited(e)
    store = _identity(e)
    recorded = ex.RECORDED_CHECK.search(store["identity_check"])["sha"]
    notok = FACTS / "check-copies-20261023T133940Z.json"  # ok=false for the copy: never cited
    if not notok.is_file():
        pytest.skip("the dsh-facts files are not on this machine")
    raw = notok.read_bytes()
    bad = {**e, "known_answers_sha256": hashlib.sha256(raw).hexdigest()}
    problems = ex.prove(bad, GRAPH, store["endpoint"], {**files, bad["known_answers_sha256"]: (str(notok), json.loads(raw))},
                        store["replica"], recorded, False)
    assert any("not a passed check" in p for p in problems), problems
    # a run that has no entry of this copy
    other = FACTS / "check-copies-20261023T192236Z.json"
    raw = other.read_bytes()
    bad = {**e, "known_answers_sha256": hashlib.sha256(raw).hexdigest()}
    problems = ex.prove(bad, GRAPH, store["endpoint"], {**files, bad["known_answers_sha256"]: (str(other), json.loads(raw))},
                        store["replica"], recorded, False)
    assert any("does not have exactly one entry" in p for p in problems), problems
    # another OUT's storecheck of the same copy (p3 is on -22 too): not the one this OUT's identity records
    sibling = _dsh()[3]
    files3 = _cited(sibling)
    bad = {**e, "storecheck_sha256": sibling["storecheck_sha256"]}
    problems = ex.prove(bad, GRAPH, store["endpoint"], {**files, **files3}, store["replica"], recorded, False)
    assert any("not the one the OUT's identity records" in p for p in problems), problems


def test_the_four_copies_are_arm64_step_copies_of_the_graph_by_the_registry():
    registry = Path("/home/user/project-reviews/store-copies-20261021/copies.json")
    if not registry.is_file():
        pytest.skip("the copy registry is not on this machine")
    copies = json.loads(registry.read_text(encoding="utf-8"))["step_copies_20261023"]["copies"]
    for copy in {c for c, *_ in PLANS.values()}:
        assert copies[copy]["store"] == GRAPH and copies[copy]["arch"] == "aarch64"
        assert copies[copy]["image"]["sif_held"][0].endswith("qlever-9bebc60c51-arm64.sif")
