"""Tests of the partition binding (``--partition-binding``/``--partition-plan``: DeepSeek's dsh union) and of the pinned Coder dh2
record schema in exploratory.py (read_record, read_binding, binding_problems, check_union, prepare's option rules).

What is tested, and against what:
- The REAL files are read from git (the experiments repo, as the driver tests do): opaque-k6.json (feat/entry180-partitions), the
  binding and plans (origin/feat/deepseek-opaque-partitions), the frozen tree's questions file (847cc275), and the tracked
  copy of the Coder record (partition-records/opaque-coder-k2.json). Their acceptance is asserted with expectations typed in
  below from a DIFFERENT source: the six dsh OUTs' deployment identities (plan_sha256, binding_sha256, copy), not from the plan files.
- Every refusal test mutates a copy of the real binding, plans, record or identity values and asserts the one intended refusal; the
  unmutated baseline is asserted accepted in the same test, so a refusal can not come from the baseline. A mutated binding is
  re-serialised and its plans re-pointed at the new binding sha256 (only that bookkeeping field), so the rule under test is the one
  that fires.
- check_union is called with a stub frame whose load_batch returns no records: only its identity-level rules run (the record-level rules
  are the analysis run's). The OUT identities here are fixtures made from the REAL plan values (no OUT is read); the real identities
  are checked by the dry script PREC2-check-real.py of the review, which runs the same functions over the copied identity files.
- A binding cited with --partitions (science, 25 Oct ~11:5xZ: a PROPER subset only; the pooled call stays the reported one) is tested
  on the same real files: SLOW {0, 3, 4} and MEDIUM {1, 2, 5} are accepted with the expectations typed in below; every refusal has its
  accepted baseline in the same test and its full literal text; every mutated-OUT case of the pooled call is refused again, with the
  same text, in a subset call; and the pooled call is compared with exploratory.py as it was at ccea1637 (git show, a second module):
  equal Union values and equal refusal texts, so the change is shown not to touch it.
Run: the e189 suite's python (numpy) from the repository root: ``python -m pytest results/e189-exploratory-20261022/tests``."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
EXP = Path("/home/user/repos/www-2027/experiments-repo")
FROZEN = "847cc27563a3c7a505c8ec0fd647c9314acaae8c"
BINDING_REF = "origin/feat/deepseek-opaque-partitions:results/deepseek-opaque-partitions-20261023/out/partitions"
K6_SRC = "feat/entry180-partitions:results/entry180-partitions-20261022/out/partitions/opaque-k6.json"
GRAPH = "opaque-rhea-142"
OLD_COMMIT = "ccea1637478c6395a598b981566b7b6b144eae3e"  # feat/e189-two-periods before a cited binding took --partitions
SLOW, MEDIUM = (0, 3, 4), (1, 2, 5)  # the descriptive subsets science named beside the pooled call

_spec = importlib.util.spec_from_file_location("e189_exploratory_binding", RESULTS / "exploratory.py")
ex = importlib.util.module_from_spec(_spec)
sys.modules["e189_exploratory_binding"] = ex
_spec.loader.exec_module(ex)

# typed in from the six dsh OUTs' deployment identities (partition.binding_sha256 / plan_sha256 / copy), not from the plan files
BINDING_SHA = "015aa5e74a391b266db346f480978f46d7f706aad2d022aa48160f173f14326b"
DSH = [  # (partition, plan_sha256, copy)
    (0, "0d0cf40446fcc60976a66bca893a3730589814c313afcb92c43d553b45cc20cc", "example-copy-24"),
    (1, "7efb050424205cb849b1c0eabe495a2f00ac65b79d6b1bcad8953ea44b85267d", "example-copy-27"),
    (2, "18f9bd250b7d3df31ef5cf88f4d4424ec7ec36d0fad5ad0aa3e302c353373bca", "example-copy-26"),
    (3, "6b5cff90edc2e4000cf90b3439c29db557aef04fdf9ca4ddf736ba7efb1c9c88", "example-copy-24"),
    (4, "d4a045074cdec610e350cf0b439f9dc0087eb0427fc18d19b3b48c4ceb40fcd7", "example-copy-25"),
    (5, "671bc3743ede5f89b78bb294abff53cc0ca71302b7f6a9027ba0b1fce37e5ac7", "example-copy-27"),
]
OUTP = "opaque-deepseek-v4-flash-0731-dsh-p{}of6"
CODER_SHA = "639bb65af66828821bb774bf9e65b6194510fe343587c145a1615fb66b856714"
CODER_TRACKED = RESULTS / "partition-records" / "opaque-coder-k2.json"
HOST_RUNS = "/scratch/user/project/runs/test-20261021"


def _show(ref_path: str) -> bytes:
    return subprocess.run(["git", "-C", str(EXP), "show", ref_path], check=True, capture_output=True).stdout


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture(scope="module")
def design() -> tuple[list[str], str]:
    raw = _show(f"{FROZEN}:data/questions/{GRAPH}/questions.jsonl")
    rows = {}
    for line in raw.decode().splitlines():
        if line.strip():
            q = json.loads(line)
            rows[q["id"]] = q
    return [i for i, q in rows.items() if q.get("split") == "test"], _sha(raw)


@pytest.fixture()
def real(tmp_path):
    """The real record, binding and plans as bytes, and a writer for (possibly mutated) copies."""
    class Real:
        record = _show(K6_SRC)
        binding = _show(f"{BINDING_REF}/binding.json")
        plans = [_show(f"{BINDING_REF}/plan-p{i}.json") for i in range(6)]
        dir = tmp_path

        @staticmethod
        def write(record=None, binding=None, plans=None, repoint=True):
            """Write the files (the real ones where a part is None); a mutated binding gets its plans' binding_sha256 re-pointed
            at its new sha256 when ``repoint`` (so only the mutation under test can refuse)."""
            rb = record if record is not None else Real.record
            bb = binding if binding is not None else Real.binding
            pl = list(plans) if plans is not None else list(Real.plans)
            if repoint and binding is not None:
                new = []
                for raw in pl:
                    d = json.loads(raw)
                    d["binding_sha256"] = _sha(bb)
                    new.append(json.dumps(d, indent=1).encode())
                pl = new
            paths = {"record": tmp_path / "rec.json", "binding": tmp_path / "binding.json"}
            paths["record"].write_bytes(rb)
            paths["binding"].write_bytes(bb)
            paths["plans"] = []
            for i, raw in enumerate(pl):
                p = tmp_path / f"plan-{i}.json"
                p.write_bytes(raw)
                paths["plans"].append(p)
            return paths

    return Real


def _mutate_json(raw: bytes, fn) -> bytes:
    d = json.loads(raw)
    fn(d)
    return json.dumps(d, indent=1).encode()


# --- read_record: the pinned Coder record -------------------------------------------------------------------------------

def test_the_tracked_coder_record_is_the_pinned_file_and_is_accepted(tmp_path, design) -> None:
    ids, qsha = design
    raw = CODER_TRACKED.read_bytes()
    assert _sha(raw) == CODER_SHA == ex.PINNED_RECORDS["coder-dh2-partition/1"]
    original = Path("/home/user/wt/coder-example-copy-54/results/coder-example-copy-55/out/partitions/opaque-coder-k2.json")
    if original.exists():  # the untracked source file on this host: the tracked copy is byte-identical to it
        assert original.read_bytes() == raw
    record, sha = ex.read_record(CODER_TRACKED)
    assert sha == CODER_SHA and record["schema"] == "coder-dh2-partition/1"
    assert ex.check_record(record, ids, GRAPH, "test", qsha) == 2  # the rule over the real question set, recomputed


def test_a_record_in_the_coder_schema_is_accepted_only_as_the_pinned_file(tmp_path) -> None:
    raw = CODER_TRACKED.read_bytes()
    p = tmp_path / "r.json"
    p.write_bytes(raw)
    assert ex.read_record(p)[1] == CODER_SHA  # control
    for name, mutated in (("one byte more", raw + b"\n"),
                          ("a changed field", _mutate_json(raw, lambda d: d.__setitem__("ruling", "x"))),
                          ("an entry180 record relabelled", _mutate_json(_show(K6_SRC), lambda d: d.__setitem__("schema", "coder-dh2-partition/1")))):
        p.write_bytes(mutated)
        with pytest.raises(ex.Refused, match="accepted only as the one pinned file"):
            ex.read_record(p)


def test_the_other_schemas_are_read_as_before_and_an_unknown_one_is_refused(tmp_path) -> None:
    p = tmp_path / "r.json"
    p.write_bytes(_show(K6_SRC))
    assert ex.read_record(p)[0]["schema"] == "entry180-partition/1"
    for schema in ("gemma-partition/1", "deepseek-opaque-partition/1", None):
        p.write_bytes(_mutate_json(_show(K6_SRC), lambda d, s=schema: d.__setitem__("schema", s)))
        with pytest.raises(ex.Refused, match="not one of"):
            ex.read_record(p)
    p.write_bytes(b"[]")
    with pytest.raises(ex.Refused, match="not one of"):
        ex.read_record(p)
    assert "coder-dh2-partition/1" not in ex.RECORD_SCHEMAS  # never accepted by schema name alone


# --- read_binding: the real files accepted, mutated copies refused -----------------------------------------------------------

def test_the_real_binding_and_plans_are_accepted_for_the_real_record(real) -> None:
    f = real.write()
    record, rsha = ex.read_record(f["record"])
    assert rsha == "98158f127c0c7545d5c7255cfa7760dc6b9ff6a9d3b80b7929354cb003dc1cb7"
    binding = ex.read_binding(f["binding"], f["plans"], record, rsha)
    assert binding.sha256 == BINDING_SHA
    assert sorted(binding.plans) == list(range(6))
    for i, plan_sha, copy_ in DSH:
        pl = binding.plans[i]
        assert (pl.partition, pl.out, pl.copy, pl.sha256) == (i, OUTP.format(i), copy_, plan_sha)


def _read(real, f):
    record, rsha = ex.read_record(f["record"])
    return ex.read_binding(f["binding"], f["plans"], record, rsha)


BINDING_MUTATIONS = [  # (name, mutation of the parsed binding, expected refusal)
    ("another binding schema", lambda d: d.__setitem__("schema", "deepseek-opaque-binding/2"), "not a binding in schema"),
    ("K 5", lambda d: d.__setitem__("k", 5), "binding's K is 5"),
    ("K as text", lambda d: d.__setitem__("k", "6"), "binding's K is '6'"),
    ("another record sha256", lambda d: d["record"].__setitem__("sha256", "0" * 64), "not for the cited partition record"),
    ("no record block", lambda d: d.pop("record"), "not for the cited partition record"),
    ("a partition's count", lambda d: d["record"]["partitions"][2].__setitem__(1, 40), "record block"),
    ("a partition's ids digest", lambda d: d["record"]["partitions"][5].__setitem__(2, "0" * 64), "record block"),
    ("a partition missing", lambda d: d["record"]["partitions"].pop(), "record block"),
    ("another rule", lambda d: d["record"].__setitem__("rule", "mod K"), "record block"),
    ("another question count", lambda d: d["record"].__setitem__("n_questions", 209), "record block"),
    ("no copies", lambda d: d.__setitem__("copies", {}), "names no store copies"),
    ("copies as a list", lambda d: d.__setitem__("copies", ["example-copy-24"]), "names no store copies"),
    ("a copy name with a slash", lambda d: d.__setitem__("copies", {"a/b": {}}), "names no store copies"),
]


@pytest.mark.parametrize("name,fn,text", BINDING_MUTATIONS, ids=[m[0] for m in BINDING_MUTATIONS])
def test_a_mutated_binding_is_refused(real, name, fn, text) -> None:
    _read(real, real.write())  # the baseline is accepted
    f = real.write(binding=_mutate_json(real.binding, fn))
    with pytest.raises(ex.Refused, match=re.escape(text)):
        _read(real, f)


def test_a_binding_for_another_record_is_refused(real) -> None:
    f = real.write(record=real.record + b"\n")  # the record cited is not the one the binding names
    with pytest.raises(ex.Refused, match="not for the cited partition record"):
        _read(real, f)
    other = _show("feat/entry190-gemma-partitions:results/entry190-gemma-partitions-20261023/out/partitions/opaque-gemma-k3.json")
    with pytest.raises(ex.Refused, match="binding's K is 6"):
        _read(real, real.write(record=other))


PLAN_MUTATIONS = [  # (name, plan index, mutation, expected refusal)
    ("another plan schema", 2, lambda d: d.__setitem__("schema", "deepseek-opaque-plan/1"), "not a plan in schema"),
    ("partition 0 named twice", 1, lambda d: d.__setitem__("partition", 0), "named once"),
    ("a partition out of range", 5, lambda d: d.__setitem__("partition", 6), "named once"),
    ("a partition as a boolean", 1, lambda d: d.__setitem__("partition", True), "named once"),
    ("another binding sha256", 3, lambda d: d.__setitem__("binding_sha256", "0" * 64), "binding_sha256 is not the cited binding's"),
    ("another K", 3, lambda d: d.__setitem__("k", 4), "K or question count"),
    ("another question count", 4, lambda d: d.__setitem__("n_questions", 34), "K or question count"),
    ("an OUT path", 0, lambda d: d.__setitem__("out", "/net/x/" + OUTP.format(0)), "not a plain name"),
    ("no OUT", 0, lambda d: d.pop("out"), "not a plain name"),
    ("a copy that is not the binding's", 2, lambda d: d.__setitem__("copy", "example-copy-14"), "one of the binding's copies"),
    ("a copy with a slash", 2, lambda d: d.__setitem__("copy", "example-copy-24/"), "one of the binding's copies"),
    ("an OUT shared by two plans", 4, lambda d: d.__setitem__("out", OUTP.format(3)), "distinct OUTs"),
]


@pytest.mark.parametrize("name,i,fn,text", PLAN_MUTATIONS, ids=[m[0] for m in PLAN_MUTATIONS])
def test_a_mutated_plan_is_refused(real, name, i, fn, text) -> None:
    plans = list(real.plans)
    plans[i] = _mutate_json(plans[i], fn)
    f = real.write(plans=plans)
    with pytest.raises(ex.Refused, match=re.escape(text)):
        _read(real, f)


def test_the_plan_set_must_be_one_per_partition(real) -> None:
    record, rsha = ex.read_record(real.write()["record"])
    f = real.write()
    with pytest.raises(ex.Refused, match="exactly one plan per partition"):
        ex.read_binding(f["binding"], f["plans"][:5], record, rsha)  # a plan missing
    with pytest.raises(ex.Refused, match="exactly one plan per partition"):
        ex.read_binding(f["binding"], [*f["plans"], f["plans"][0]], record, rsha)  # a plan twice
    with pytest.raises(ex.Refused, match="named once"):
        ex.read_binding(f["binding"], [*f["plans"][:5], f["plans"][0]], record, rsha)  # six files, partition 0 twice, 5 missing
    shuffled = [f["plans"][i] for i in (3, 0, 5, 1, 4, 2)]
    assert sorted(ex.read_binding(f["binding"], shuffled, record, rsha).plans) == list(range(6))  # the order of the files is free


def test_a_binding_is_not_read_for_a_record_without_partitions(real) -> None:
    f = real.write()
    for bad in ({"k": 6}, {"k": 6, "partitions": "x"}, {"k": 0, "partitions": []}, {"k": 6, "partitions": [{}] * 5},
                {"k": 2, "partitions": [{"index": 1}, {"index": 0}]}):
        with pytest.raises(ex.Refused, match="no K partitions to bind"):
            ex.read_binding(f["binding"], f["plans"], bad, "0" * 64)


# --- check_union: the OUT-level rules, with the plan's copy in place of the record's -------------------------------------------

STUB = types.SimpleNamespace(
    inputs=types.SimpleNamespace(load_batch=lambda path, cells: types.SimpleNamespace(records=[], branch=[])),
    gate=types.SimpleNamespace(grid_cells=lambda path: None))


def _check() -> str:  # a store block's passed identity check in the builders' form
    return "?cmd=stats equal to http://192.0.2.101:7033 in 13 fields at 2026-10-23T18:14:55Z (store check sha256 " + "a" * 64 + ")"


def _outs(record: dict, rsha: str, binding_sha: str | None, plans: list[tuple[int, str, str]], prefix: str, k: int) -> list:
    """OUT objects (an identity with the partition block and store block the loader reads; the manifest empty), one per partition.
    ``plans``: (partition, plan sha256 or '', copy)."""
    outs = []
    for i, plan_sha, copy_ in plans:
        name = f"{prefix}-p{i}of{k}"
        part = {"copy": copy_, "ids_sha256": record["partitions"][i]["ids_sha256"], "index": i, "k": k,
                "n_questions": record["partitions"][i]["n_questions"], "record_sha256": rsha}
        if binding_sha is not None:
            part.update(binding_sha256=binding_sha, plan_sha256=plan_sha)
        ident = {"out": f"{HOST_RUNS}/{name}", "partition": part,
                 "store": {"replica": copy_, "endpoint": f"http://192.0.2.160:289{i}/", "identity_check": _check()}}
        outs.append(ex.Out(Path("/stage") / name, ident, {}))
    return outs


def _dsh_world(real, tmp_path_f=None):
    f = real.write()
    record, rsha = ex.read_record(f["record"])
    binding = ex.read_binding(f["binding"], f["plans"], record, rsha)
    return record, rsha, binding


def _union(design, record, rsha, outs, binding=None, subset=None):
    ids, qsha = design
    return ex.check_union(STUB, outs, record, rsha, ids, GRAPH, "test", qsha, subset, binding)


def test_check_union_with_the_binding_takes_the_plans_copies(real, design) -> None:
    record, rsha, binding = _dsh_world(real)
    outs = _outs(record, rsha, BINDING_SHA, DSH, "opaque-deepseek-v4-flash-0731-dsh", 6)
    union = _union(design, record, rsha, outs, binding)  # baseline: accepted
    assert union.k == 6 and union.binding["sha256"] == BINDING_SHA
    assert [p["plan_sha256"] for p in union.binding["plans"]] == [d[1] for d in DSH]
    assert [p["copy"] for p in union.binding["plans"]] == [d[2] for d in DSH]
    # the same OUTs without the binding: refused (their copies are not the record's), and say why
    with pytest.raises(ex.Refused) as e:
        _union(design, record, rsha, outs)
    assert "no binding is cited" in str(e.value) and "is not partition 0's copy in the record" in str(e.value)


def _mutate_outs(outs, i, fn):
    outs = list(outs)
    o = outs[i]
    ident = copy.deepcopy(o.identity)
    path = o.path
    ident, path = fn(ident, path)
    outs[i] = ex.Out(path, ident, o.manifest)
    return outs


def setp(key, value):  # set a field of the identity's partition block
    def fn(ident, path):
        ident["partition"][key] = value
        return ident, path
    return fn


def popp(key):  # remove a field of the identity's partition block
    def fn(ident, path):
        del ident["partition"][key]
        return ident, path
    return fn


def set_copy(partition_copy=None, replica=None):
    def fn(ident, path):
        if partition_copy is not None:
            ident["partition"]["copy"] = partition_copy
        if replica is not None:
            ident["store"]["replica"] = replica
        return ident, path
    return fn


def set_out(value):  # the identity's own `out` (None removes it)
    def fn(ident, path):
        if value is None:
            del ident["out"]
        else:
            ident["out"] = value
        return ident, path
    return fn


def rename_to(name):  # the OUT directory's name
    def fn(ident, path):
        return ident, path.parent / name
    return fn


OUT_MUTATIONS = [  # (name, partition, mutation of (identity, path) -> (identity, path), expected refusal)
    ("the binding unnamed", 1, popp("binding_sha256"), "does not name the cited binding"),
    ("another binding named", 1, setp("binding_sha256", "0" * 64), "does not name the cited binding"),
    ("the plan unnamed", 2, popp("plan_sha256"), "does not name partition 2's plan"),
    ("another partition's plan named (p0's for p3: the same copy)", 3, setp("plan_sha256", DSH[0][1]), "does not name partition 3's plan"),
    ("the record's copy (another copy of the record's list)", 0, set_copy("example-copy-14", "example-copy-14"),
     "is not partition 0's copy in the cited binding (example-copy-24)"),
    ("the partition copy right, the store replica another", 4, set_copy(replica="example-copy-24"),
     "is not partition 4's copy in the cited binding"),
    ("the store replica right, the partition copy another", 5, set_copy(partition_copy="example-copy-26"),
     "is not partition 5's copy in the cited binding"),
    ("the OUT directory named for another partition", 2, rename_to(OUTP.format(3)), "is not partition 2's plan OUT"),
    ("the identity's out another OUT", 2, set_out(f"{HOST_RUNS}/{OUTP.format(1)}"), "is not partition 2's plan OUT"),
    ("the identity's out missing", 2, set_out(None), "is not partition 2's plan OUT"),
]


@pytest.mark.parametrize("name,part,fn,text", OUT_MUTATIONS, ids=[m[0] for m in OUT_MUTATIONS])
def test_check_union_refuses_a_mutated_out_under_a_binding(real, design, name, part, fn, text) -> None:
    record, rsha, binding = _dsh_world(real)
    outs = _outs(record, rsha, BINDING_SHA, DSH, "opaque-deepseek-v4-flash-0731-dsh", 6)
    _union(design, record, rsha, outs, binding)  # baseline accepted
    with pytest.raises(ex.Refused) as e:
        _union(design, record, rsha, _mutate_outs(outs, part, fn), binding)
    assert text in str(e.value), str(e.value)


def test_check_union_refuses_a_binding_outs_without_the_binding_and_the_reverse(real, design) -> None:
    """A union whose OUTs do not name the binding (Qwen opaque's, which share the record) is refused when a binding is cited, and
    the dsh OUTs are refused when none is cited: no fallback in either direction."""
    record, rsha, binding = _dsh_world(real)
    qwen_like = _outs(record, rsha, None, [(i, "", c) for i, _, c in DSH], "opaque-deepseek-v4-flash-0731-dsh", 6)
    with pytest.raises(ex.Refused) as e:
        _union(design, record, rsha, qwen_like, binding)
    assert "does not name the cited binding" in str(e.value)
    # a union of OUTs on the RECORD's own copies, no binding named in their blocks, is accepted without a binding (the old path)
    record_copies = [(p["index"], "", p["copy"]) for p in record["partitions"]]
    plain = _outs(record, rsha, None, record_copies, "opaque-qwen3.8-27b-e180", 6)
    assert _union(design, record, rsha, plain).binding is None


def test_a_binding_with_a_partial_set_of_plans_is_refused_whatever_the_call(real, design) -> None:
    """A binding is cited whole: check_union still refuses a Binding that holds fewer plans than the record has partitions, for the pooled
    call and for a subset, even one whose partitions all have a plan."""
    record, rsha, binding = _dsh_world(real)
    outs = _outs(record, rsha, BINDING_SHA, DSH, "opaque-deepseek-v4-flash-0731-dsh", 6)
    partial = ex.Binding(binding.sha256, {i: pl for i, pl in binding.plans.items() if i < 5})
    for subset in (None, (0, 1, 2)):
        with pytest.raises(ex.Refused, match="one plan per partition"):
            _union(design, record, rsha, outs[:3] if subset else outs, partial, subset=subset)
    assert _union(design, record, rsha, outs[:3], binding, subset=(0, 1, 2)).subset == (0, 1, 2)  # the whole binding: accepted


# --- a binding cited with --partitions: a proper subset only -------------------------------------------------------------------

PREFIX = "opaque-deepseek-v4-flash-0731-dsh"


def _subset_outs(record, rsha, parts):
    """The dsh OUT fixtures (the real plans' values) of ``parts`` alone: a subset call is given no other OUT."""
    return _outs(record, rsha, BINDING_SHA, [d for d in DSH if d[0] in parts], PREFIX, 6)


@pytest.mark.parametrize("subset", [SLOW, MEDIUM, (0, 1), (2, 5), (1, 2, 3, 4, 5)])
def test_a_proper_subset_is_accepted_under_the_binding_with_only_its_outs(real, design, subset) -> None:
    record, rsha, binding = _dsh_world(real)
    union = _union(design, record, rsha, _subset_outs(record, rsha, subset), binding, subset=subset)
    assert union.subset == subset and union.k == 6 and union.record_sha256 == rsha
    assert list(union.index_of.items()) == [(f"/stage/{OUTP.format(i)}", i) for i in subset]  # the named partitions' OUTs, each once
    # the binding block is the whole binding as read (every plan), the sha256 and the plans' values typed in from the OUT identities
    assert union.binding["sha256"] == BINDING_SHA
    assert [(p["partition"], p["out"], p["copy"], p["plan_sha256"]) for p in union.binding["plans"]] == [
        (i, OUTP.format(i), copy_, plan_sha) for i, plan_sha, copy_ in DSH]


def test_the_pooled_call_with_the_binding_stays_the_call_without_the_option(real, design) -> None:
    record, rsha, binding = _dsh_world(real)
    outs = _subset_outs(record, rsha, range(6))
    pooled = _union(design, record, rsha, outs, binding)
    assert pooled.subset is None and sorted(pooled.index_of.values()) == list(range(6))
    for subset in (SLOW, MEDIUM):  # a subset's union is not the pooled one: its partitions only
        sub = _union(design, record, rsha, _subset_outs(record, rsha, subset), binding, subset=subset)
        assert sub.subset == subset and sub != pooled


def test_a_subset_with_a_partition_outside_the_binding_is_refused(real, design) -> None:
    record, rsha, binding = _dsh_world(real)
    assert _union(design, record, rsha, _subset_outs(record, rsha, (0, 3)), binding, subset=(0, 3)).subset == (0, 3)  # baseline
    for subset in ((0, 3, 6), (6, 7), (0, 3, -1)):
        with pytest.raises(ex.Refused) as e:
            _union(design, record, rsha, _subset_outs(record, rsha, (0, 3)), binding, subset=subset)
        assert str(e.value) == f"--partitions {sorted(subset)}: not all of 0..5, the record's partitions", str(e.value)
    # the binding itself names K = 6 partitions: a partition outside it is outside 0..K-1 of the record it is bound to
    assert json.loads(real.binding)["k"] == 6


def test_a_subset_naming_every_partition_is_refused_under_a_binding_but_not_without_one(real, design) -> None:
    record, rsha, binding = _dsh_world(real)
    outs = _subset_outs(record, rsha, range(6))
    assert _union(design, record, rsha, outs, binding).subset is None  # baseline: the pooled call, without the option
    for subset in ((0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0)):
        with pytest.raises(ex.Refused) as e:
            _union(design, record, rsha, outs, binding, subset=subset)
        assert str(e.value) == ("--partitions names every partition of the record: with a cited binding that is the pooled call, which is "
                                "made without --partitions (a binding takes --partitions only for a proper subset)"), str(e.value)
    # without a binding naming every partition is still the call without the option (the module docstring's contract), unchanged
    plain = _outs(record, rsha, None, [(p["index"], "", p["copy"]) for p in record["partitions"]], "opaque-qwen3.8-27b-e180", 6)
    assert _union(design, record, rsha, plain, subset=(0, 1, 2, 3, 4, 5)).subset is None


def test_a_subset_of_one_partition_is_refused_under_a_binding(real, design) -> None:
    record, rsha, binding = _dsh_world(real)
    assert _union(design, record, rsha, _subset_outs(record, rsha, (3, 4)), binding, subset=(3, 4)).subset == (3, 4)  # baseline
    for subset in ((3,), ()):
        with pytest.raises(ex.Refused) as e:
            _union(design, record, rsha, _subset_outs(record, rsha, (3,)), binding, subset=subset)
        assert str(e.value) == f"--partitions {list(subset)}: a partition is analysed only in a union of at least two", str(e.value)


def test_a_subset_out_whose_copy_differs_from_its_plans_is_refused(real, design) -> None:
    record, rsha, binding = _dsh_world(real)
    outs = _subset_outs(record, rsha, SLOW)  # partitions 0, 3, 4 in this order; partition 3's plan copy is example-copy-24
    assert _union(design, record, rsha, outs, binding, subset=SLOW).subset == SLOW  # baseline
    assert record["partitions"][3]["copy"] == "example-copy-30"  # the record's own copy of partition 3, typed from the real record
    for what, fn in (("the record's copy", set_copy("example-copy-30", "example-copy-30")),
                     ("partition 4's plan copy", set_copy("example-copy-25", "example-copy-25")),
                     ("a copy that is not the binding's", set_copy("example-copy-14", "example-copy-14")),
                     ("the store replica alone", set_copy(replica="example-copy-25"))):
        with pytest.raises(ex.Refused) as e:
            _union(design, record, rsha, _mutate_outs(outs, 1, fn), binding, subset=SLOW)
        assert f"/stage/{OUTP.format(3)}: its store copy (" in str(e.value) and "is not partition 3's copy in the cited binding (example-copy-24)" in str(e.value), (what, str(e.value))


def test_a_subset_refuses_an_out_of_a_partition_it_does_not_name_and_a_named_one_missing(real, design) -> None:
    record, rsha, binding = _dsh_world(real)
    assert _union(design, record, rsha, _subset_outs(record, rsha, SLOW), binding, subset=SLOW).subset == SLOW  # baseline
    with pytest.raises(ex.Refused) as e:  # partition 5 is a partition of the binding and of the record, but not named
        _union(design, record, rsha, _subset_outs(record, rsha, (0, 3, 4, 5)), binding, subset=SLOW)
    assert str(e.value) == (f"/stage/{OUTP.format(5)}: partition 5, which --partitions [0, 3, 4] does not name (a subset's union holds only "
                            "the named partitions)"), str(e.value)
    with pytest.raises(ex.Refused) as e:  # a named partition without its OUT
        _union(design, record, rsha, _subset_outs(record, rsha, (0, 3)), binding, subset=SLOW)
    assert str(e.value) == "the union holds partitions [0, 3], not each of the named [0, 3, 4] once", str(e.value)
    with pytest.raises(ex.Refused) as e:  # the pooled call still needs every partition's OUT
        _union(design, record, rsha, _subset_outs(record, rsha, SLOW), binding)
    assert str(e.value) == "the union holds partitions [0, 3, 4], not each of 0..5 once", str(e.value)


@pytest.mark.parametrize("name,part,fn,text", OUT_MUTATIONS, ids=[m[0] for m in OUT_MUTATIONS])
def test_a_mutated_out_is_refused_in_a_subset_call_exactly_as_in_the_pooled_call(real, design, name, part, fn, text) -> None:
    """Every OUT mutation of the pooled call, applied to the same OUT inside the subset (SLOW or MEDIUM) that holds its partition: the
    same refusal, text for text (the other OUTs of the subset add no problem)."""
    record, rsha, binding = _dsh_world(real)
    subset = SLOW if part in SLOW else MEDIUM
    pooled = _subset_outs(record, rsha, range(6))
    outs = _subset_outs(record, rsha, subset)
    _union(design, record, rsha, outs, binding, subset=subset)  # baseline accepted
    with pytest.raises(ex.Refused) as pooled_refusal:
        _union(design, record, rsha, _mutate_outs(pooled, part, fn), binding)
    with pytest.raises(ex.Refused) as subset_refusal:
        _union(design, record, rsha, _mutate_outs(outs, subset.index(part), fn), binding, subset=subset)
    assert text in str(subset_refusal.value), str(subset_refusal.value)
    assert str(subset_refusal.value) == str(pooled_refusal.value)


def test_the_binding_files_are_cited_whole_for_a_subset_too(real) -> None:
    """The loader is unchanged: a subset call is given the binding and ALL its plans (the plans of the partitions it does not analyse are
    read and checked, never matched to an OUT), and fewer plans than the binding's K are refused as in the pooled call."""
    f = real.write()
    record, rsha = ex.read_record(f["record"])
    for subset_plans in (f["plans"][:3], [f["plans"][i] for i in SLOW]):
        with pytest.raises(ex.Refused) as e:
            ex.read_binding(f["binding"], subset_plans, record, rsha)
        assert str(e.value) == "the binding needs exactly one plan per partition (6), not 3 (--partition-plan)", str(e.value)


@pytest.fixture(scope="module")
def old(tmp_path_factory):
    """exploratory.py as it was at OLD_COMMIT (git show), beside the other files of its directory, as a second module."""
    where = tmp_path_factory.mktemp("old") / RESULTS.name
    where.mkdir()
    for f in RESULTS.iterdir():
        if f.name not in ("exploratory.py", "tests", "dry-run", "__pycache__"):
            (where / f.name).symlink_to(f)
    blob = subprocess.run(["git", "-C", str(EXP), "show", f"{OLD_COMMIT}:results/e189-exploratory-20261022/exploratory.py"],
                          check=True, capture_output=True).stdout
    (where / "exploratory.py").write_bytes(blob)
    spec = importlib.util.spec_from_file_location("e189_exploratory_old", where / "exploratory.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e189_exploratory_old"] = module
    spec.loader.exec_module(module)
    return module


def _outcome(fn):
    """('union', its fields as plain data) or ('refused', its text): what a check_union call came to."""
    try:
        union = fn()
    except Exception as exc:  # noqa: BLE001 - the class differs between the two modules; the kind and the text are compared
        return type(exc).__name__, str(exc)
    return "union", dataclasses.asdict(union)


def test_the_pooled_call_is_what_it_was_at_the_commit_before(real, design, old) -> None:
    """The pooled call (no --partitions) with the binding: read_binding and check_union of this module and of ccea1637's give equal
    bindings, equal Union values and, for each OUT mutation, the same refusal text; the OLD module did refuse the subset the new one
    accepts (so the comparison would show a difference)."""
    f = real.write()
    ids, qsha = design
    record_new, rsha = ex.read_record(f["record"])
    record_old, rsha_old = old.read_record(f["record"])
    assert rsha == rsha_old and record_new == record_old
    bn = ex.read_binding(f["binding"], f["plans"], record_new, rsha)
    bo = old.read_binding(f["binding"], f["plans"], record_old, rsha)
    assert dataclasses.asdict(bn) == dataclasses.asdict(bo)
    outs = _subset_outs(record_new, rsha, range(6))

    def call(mod, binding, outs_, subset=None):
        return lambda: mod.check_union(STUB, [mod.Out(o.path, o.identity, o.manifest) for o in outs_], record_new, rsha, ids, GRAPH,
                                       "test", qsha, subset, binding)

    cases = [("the real fixtures", outs)] + [(m[0], _mutate_outs(outs, m[1], m[2])) for m in OUT_MUTATIONS]
    for name, case in cases:
        new_outcome, old_outcome = _outcome(call(ex, bn, case)), _outcome(call(old, bo, case))
        assert new_outcome == old_outcome, (name, new_outcome, old_outcome)
    assert _outcome(call(ex, bn, outs))[0] == "union"
    # the no-binding pooled call, too (a Qwen-like union on the record's own copies)
    plain = _outs(record_new, rsha, None, [(p["index"], "", p["copy"]) for p in record_new["partitions"]], "opaque-qwen3.8-27b-e180", 6)
    assert _outcome(call(ex, None, plain)) == _outcome(call(old, None, plain))
    assert _outcome(call(ex, None, plain, (0, 1, 2, 3, 4, 5))) == _outcome(call(old, None, plain, (0, 1, 2, 3, 4, 5)))
    # the one difference: a subset under the binding (ccea1637 refused it: "a binding is cited for the whole union")
    sub = _subset_outs(record_new, rsha, SLOW)
    assert _outcome(call(ex, bn, sub, SLOW))[0] == "union"
    assert _outcome(call(old, bo, sub, SLOW)) == ("Refused", "a binding is cited for the whole union, not a subset (--partitions)")


def test_check_union_for_the_coder_record_needs_no_binding_and_takes_the_records_copies(design) -> None:
    ids, qsha = design
    record, sha = ex.read_record(CODER_TRACKED)
    copies = [(p["index"], "", p["copy"]) for p in record["partitions"]]
    assert [c for _, _, c in copies] == ["example-copy-23"] * 2
    outs = _outs(record, sha, None, copies, "opaque-qwen3-coder-30b-a3b-dh2", 2)
    assert _union(design, record, sha, outs).k == 2
    # each partition's copy is still the record's: another copy is refused
    bad = _mutate_outs(outs, 1, lambda i, p: (i["partition"].__setitem__("copy", "example-copy-14") or i, p))
    with pytest.raises(ex.Refused, match="is not partition 1's copy in the record"):
        _union(design, record, sha, bad)


def test_the_dsh_binding_is_not_for_the_coder_record(real) -> None:
    f = real.write(record=CODER_TRACKED.read_bytes())
    record, rsha = ex.read_record(f["record"])
    with pytest.raises(ex.Refused, match="binding's K is 6, not the record's 2"):
        ex.read_binding(f["binding"], f["plans"], record, rsha)


# --- prepare(): the option rules (refused before any frame is used) ----------------------------------------------------------

def _call(**kw):
    base = dict(batches=(Path("/x"),), questions=(Path("/q"),), graph=GRAPH, split="test", voided=(Path("/v"),))
    return ex.Call(**{**base, **kw})


@pytest.mark.parametrize("kw,text", [
    (dict(partition_binding=Path("/b")), "given together or not at all"),
    (dict(partition_plans=(Path("/p"),)), "given together or not at all"),
    (dict(partition_binding=Path("/b"), partition_plans=(Path("/p"),)), "needs --partition-record"),
])
def test_prepare_refuses_inconsistent_binding_options(kw, text) -> None:
    with pytest.raises(ex.Refused, match=re.escape(text)):
        ex.prepare(None, _call(**kw))


def test_the_cli_carries_the_binding_options_into_the_call() -> None:
    src = (RESULTS / "exploratory.py").read_text()
    assert src.count('"--partition-binding"') == 1 and src.count('"--partition-plan"') == 1
    ns = types.SimpleNamespace(batch=[Path("/a")], questions=[Path("/q")], graph=GRAPH, split="test", voided=[Path("/v")],
                               partition_record=Path("/r"), ledger=[], n_reps=None, tier_if_missing=None, cluster_map=None,
                               front_log=[], store_proof=[], reference_pin=None, partitions=None,
                               partition_binding=Path("/b"), partition_plan=[Path("/p0"), Path("/p1")],
                               private_copy_classes=None, private_copy_classes_sha256=None,
                               private_endpoint_copies=None, private_endpoint_copies_sha256=None, replay_routing=[])
    call = ex._call(ns)
    assert call.partition_binding == Path("/b") and call.partition_plans == (Path("/p0"), Path("/p1"))
