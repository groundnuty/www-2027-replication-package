"""Synthetic checks for the gated store-change sides of a plain partition union (science's rule): a manifest store-change side
with a non-empty ``store_client`` object is taken in a plain union's call alone, when its ``endpoint``, ``store.endpoint`` and
``store_client.endpoint`` name one endpoint and the change's two sides' store_client are equal apart from their endpoint; every
other call refuses a gated side as it always did. Synthetic ids, endpoints and fixtures only."""

import ast
import copy
import hashlib
import inspect
import json
import random
from pathlib import Path

import pytest

import test_exploratory as TE
from test_exploratory import _private_a15_root, _store_refusals_carry_no_number  # noqa: F401  (autouse fixtures of that module)

ex = TE.ex
F = TE.F

EP_A, EP_B, EP_C = "http://gate-a.invalid:5102/", "http://gate-b.invalid:5103/", "http://gate-c.invalid:5104/"
SECRET_CLIENT = {"gate": "SECRET-GATE-NAME", "max_in_flight": 7, "timeouts": {"client": 60, "server": 120}}
TODAYS_TEXT = "a gated store (store_client), which this allowance does not cover"


def side(endpoint, client=SECRET_CLIENT, *, store_endpoint=None, client_endpoint=None, **over):
    """A store change side as the a15 partition manifests write one: endpoint, store (with its endpoint), store_client (its own
    endpoint among the gate's parameters). ``client`` None: an ungated side (store_client null, as the fixtures write it)."""
    built = {"endpoint": endpoint, "store": {"name": "synthetic-graph", "version": "v1",
                                             "endpoint": endpoint if store_endpoint is None else store_endpoint}}
    if client is None:
        built["store_client"] = None
    else:
        built["store_client"] = {**copy.deepcopy(client), "endpoint": endpoint if client_endpoint is None else client_endpoint}
    built.update(over)
    return built


def change(src, dst, before=("run-1", "run-2"), **over):
    return {"at": "2026-10-23T09:00:02.500+00:00", "note": "a test", "changed": ["endpoint", "store.endpoint", "store.version",
            "store_client.endpoint"], "from": src, "to": dst, "pre_change_records": len(before),
            "pre_change_run_ids": list(before), **over}


def out_of(*changes):
    return ex.Out(Path("OUT"), None, {"store_changes": list(changes)})


WHERE = "OUT: its manifest store change 1"


def refusal(*changes, plain_union=True):
    got, problems = ex.manifest_store_changes(out_of(*changes), plain_union=plain_union)
    return got, problems


# --- accepted --------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("client", [SECRET_CLIENT, {"gate": "g"}, {"a": [1, 2, {"b": None}], "c": 1.5, "d": True},
                                    {"nested": {"x": {"y": 1}}}])
def test_a_gated_change_with_one_endpoint_per_side_and_one_gate_regime_is_taken(client):
    got, problems = refusal(change(side(EP_A, client), side(EP_B, client)))
    assert problems == []
    assert got == [ex.StoreChange(1, "2026-10-23T09:00:02.500+00:00", EP_A, EP_B, frozenset({"run-1", "run-2"}))]


def test_a_gated_chain_of_two_changes_is_followed_as_an_ungated_one_is():
    got, problems = refusal(change(side(EP_A), side(EP_B)), change(side(EP_B), side(EP_C), ("run-1", "run-2", "run-3")))
    assert problems == [] and [(c.from_endpoint, c.to_endpoint) for c in got] == [(EP_A, EP_B), (EP_B, EP_C)]
    got, problems = refusal(change(side(EP_A), side(EP_B)), change(side(EP_A), side(EP_C)))  # still must start where it moved to
    assert any("does not start where change 1 moved to" in p for p in problems)


def test_the_gate_regime_is_compared_whatever_the_key_order_of_the_dicts():
    reordered = {k: SECRET_CLIENT[k] for k in reversed(list(SECRET_CLIENT))}
    got, problems = refusal(change(side(EP_A, SECRET_CLIENT), side(EP_B, reordered)))
    assert problems == [] and len(got) == 1


def test_a_gated_change_that_moves_nothing_is_still_refused():
    got, problems = refusal(change(side(EP_A), side(EP_A)))
    assert got == [] and problems == [f"{WHERE} moves nothing (its from and to name one endpoint)"]


# --- refused with the flag ---------------------------------------------------------------------------------------------------

ENDPOINTS_TEXT = ("a gated store without one endpoint (endpoint, store.endpoint and store_client.endpoint must all be present "
                  "and name one)")
INCONSISTENT = [
    ("side endpoint differs from store.endpoint", lambda s: s.update(store=dict(s["store"], endpoint=EP_C))),
    ("store.endpoint differs from client endpoint", lambda s: s["store_client"].update(endpoint=EP_C)),
    ("endpoint differs from both", lambda s: s.update(endpoint=EP_C)),
    ("endpoint missing", lambda s: s.pop("endpoint")),
    ("endpoint empty", lambda s: s.update(endpoint="")),
    ("endpoint not a string", lambda s: s.update(endpoint=7)),
    ("store missing", lambda s: s.pop("store")),
    ("store not an object", lambda s: s.update(store="x")),
    ("store.endpoint missing", lambda s: s["store"].pop("endpoint")),
    ("store.endpoint empty", lambda s: s["store"].update(endpoint="")),
    ("client endpoint missing", lambda s: s["store_client"].pop("endpoint")),
    ("client endpoint empty", lambda s: s["store_client"].update(endpoint="")),
    ("client endpoint not a string", lambda s: s["store_client"].update(endpoint=None)),
    ("client endpoint a list", lambda s: s["store_client"].update(endpoint=[EP_B])),
]


@pytest.mark.parametrize("which", ["from", "to"])
@pytest.mark.parametrize("name, mutate", INCONSISTENT, ids=[i[0] for i in INCONSISTENT])
def test_a_gated_side_without_one_endpoint_in_all_three_places_is_refused(name, mutate, which):
    ch = change(side(EP_A), side(EP_B))
    mutate(ch[which])
    got, problems = refusal(ch)
    assert got == [] and problems == [f"{WHERE}: its {which} side has {ENDPOINTS_TEXT}"]


@pytest.mark.parametrize("name, mutate", [
    ("a value changed", lambda c: c.update(max_in_flight=8)),
    ("a nested value changed", lambda c: c["timeouts"].update(server=121)),
    ("a key added", lambda c: c.update(extra=1)),
    ("a key removed", lambda c: c.pop("max_in_flight")),
    ("1 and true", lambda c: c.update(max_in_flight=True)),
    ("1 and 1.0", lambda c: c.update(max_in_flight=7.0)),
    ("a list for a scalar", lambda c: c.update(max_in_flight=[7])),
    ("a string for a number", lambda c: c.update(max_in_flight="7")),
])
def test_sides_whose_store_client_differ_beyond_the_endpoint_are_refused(name, mutate):
    to = side(EP_B)
    mutate(to["store_client"])
    got, problems = refusal(change(side(EP_A), to))
    assert got == []
    assert problems == [f"{WHERE}: its sides' store_client differ beyond their endpoint"]


@pytest.mark.parametrize("src, dst", [(side(EP_A), side(EP_B, None)), (side(EP_A, None), side(EP_B)),
                                      (side(EP_A), side(EP_B, store_client={})), (side(EP_A, store_client={}), side(EP_B))])
def test_a_store_client_on_one_side_only_is_refused(src, dst):
    got, problems = refusal(change(src, dst))
    assert got == [] and problems == [f"{WHERE}: its store_client is on one side only"]
    absent = [copy.deepcopy(built) for built in (src, dst)]
    for built in absent:  # the ungated side with no store_client key at all
        if not built["store_client"]:
            built.pop("store_client")
    got, problems = refusal(change(*absent))
    assert got == [] and problems == [f"{WHERE}: its store_client is on one side only"]


@pytest.mark.parametrize("junk", ["gate", ["a"], 5, True, 1.5])
@pytest.mark.parametrize("which", ["from", "to"])
def test_a_store_client_that_is_not_an_object_is_refused(junk, which):
    ch = change(side(EP_A), side(EP_B))
    ch[which]["store_client"] = junk
    got, problems = refusal(ch)
    assert got == [] and problems == [f"{WHERE}: its {which} side has a gated store whose store_client is not an object"]


def test_refusals_carry_no_endpoint_and_no_gate_value():
    cases = [change(side(EP_A), side(EP_B, client_endpoint=EP_C)), change(side(EP_A), side(EP_B, {**SECRET_CLIENT, "gate": "Z"})),
             change(side(EP_A), side(EP_B, None)), change(side(EP_A), dict(side(EP_B), store_client="SECRET-GATE-NAME"))]
    for ch in cases:
        _, problems = refusal(ch)
        assert problems
        for text in problems:
            assert not any(s in text for s in ("gate-a", "gate-b", "gate-c", "SECRET", "max_in_flight", "5102", "5103"))


def test_two_gated_changes_report_each_by_its_position():
    got, problems = refusal(change(side(EP_A), side(EP_B)), change(side(EP_B), side(EP_C, None), ("run-1", "run-2", "run-3")))
    assert [c.index for c in got] == [1] and problems == ["OUT: its manifest store change 2: its store_client is on one side only"]


# --- the flag's default and what it leaves alone ------------------------------------------------------------------------------

def test_the_flag_is_keyword_only_and_false_by_default():
    for function in (ex.manifest_store_changes, ex._side_endpoint):
        parameter = inspect.signature(function).parameters["plain_union"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is False
    with pytest.raises(TypeError):
        ex.manifest_store_changes(out_of(), True)
    with pytest.raises(TypeError):
        ex._side_endpoint({}, True)


@pytest.mark.parametrize("src, dst", [(side(EP_A), side(EP_B)), (side(EP_A, None), side(EP_B)), (side(EP_A), side(EP_B, None))])
def test_without_the_flag_a_gated_side_is_refused_with_todays_text(src, dst):
    for kwargs in ({}, {"plain_union": False}):
        got, problems = ex.manifest_store_changes(out_of(change(src, dst)), **kwargs)
        assert got == []
        gated = [s for s in ("from", "to") if (src, dst)[s == "to"].get("store_client")]
        assert problems == [f"{WHERE}: its {s} side has {TODAYS_TEXT}" for s in gated]
        for s, built in (("from", src), ("to", dst)):
            if built.get("store_client"):
                assert ex._side_endpoint(built) == (None, TODAYS_TEXT)
                assert ex._side_endpoint(built, plain_union=False) == (None, TODAYS_TEXT)


def test_an_ungated_change_is_the_same_with_and_without_the_flag():
    ungated = [change(side(EP_A, None), side(EP_B, None)),
               change({"endpoint": EP_A, "store": {"endpoint": EP_A}}, {"endpoint": EP_B, "store": {"endpoint": EP_B}}),
               change({"endpoint": EP_A, "store_client": {}}, {"endpoint": EP_B, "store_client": {}}),
               change({"endpoint": EP_A}, {"store": {"endpoint": EP_B}}),
               change({"endpoint": EP_A}, {"endpoint": EP_A}), change({"endpoint": EP_A}, {"endpoint": EP_B, "store": {"endpoint": EP_C}}),
               change("x", None), change({"endpoint": EP_A}, {"endpoint": EP_B}, before=["run-1"], pre_change_records=2)]
    for ch in ungated:
        assert ex.manifest_store_changes(out_of(ch), plain_union=True) == ex.manifest_store_changes(out_of(ch))


def test_every_caller_threads_the_flag_only_where_it_is_a_plain_union_call():
    """check_stores (the plain union's call, its ``plain_union`` parameter) passes the flag; timeline_report (the store-timeline
    tool, no union) does not: it keeps today's refusal of a gated side."""
    tree = ast.parse(Path(ex.__file__).read_text(encoding="utf-8"))
    calls = {}

    def walk(node, owner):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call) and getattr(child.func, "id", None) == "manifest_store_changes":
                calls.setdefault(owner, []).append([(k.arg, ast.unparse(k.value)) for k in child.keywords])
            if isinstance(child, ast.Call) and getattr(child.func, "id", None) == "_side_endpoint":
                calls.setdefault(owner + ":side", []).append([(k.arg, ast.unparse(k.value)) for k in child.keywords])
            walk(child, child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else owner)

    walk(tree, "<module>")
    assert calls == {"check_stores": [[("plain_union", "plain_union")]], "timeline_report": [[]],
                     "manifest_store_changes:side": [[("plain_union", "plain_union")]]}


# --- the whole call: a plain union whose OUT's records sit on two endpoints ------------------------------------------------------

PARENTS = "http://plain-copy-p.invalid:9200/"  # the first copy (the copied records' endpoint), a replica of the pinned tables
OWN = TE.PLAIN_ENDPOINTS  # each partition's own copy endpoint, as the pinned endpoint copies name them


def gated_union(tmp_path, monkeypatch, *, gated=True, regime_differs=False, branch=False):
    """A plain union whose partition 1 OUT holds the copied records of one question on the first copy and its own records on
    the partition's own copy: the manifest's one store change moves it between them (pre_change_run_ids = the copied records'
    ids; both sides gated); its identity's store is the second copy and its parents_store the first, as the a15 partitions'."""
    inputs = TE._plain_union_inputs(tmp_path, monkeypatch)
    inputs.update(TE._plain_union_private(tmp_path, monkeypatch, ("copy-0", "copy-1", "copy-p"), name="gated"))
    outs = TE._plain_union_outs(tmp_path, inputs, "union", endpoints=OWN)
    out = outs[1]
    copied = "t01-q2"
    TE._set_endpoints(out, lambda r: PARENTS if r["question_id"] == copied else OWN[1])
    before = [json.loads(f.read_text(encoding="utf-8"))["run_id"] for f in sorted((out / "records").glob("*.json"))
              if json.loads(f.read_text(encoding="utf-8"))["question_id"] == copied]
    assert before and len(before) < len(list((out / "records").glob("*.json")))
    sides = [side(PARENTS, SECRET_CLIENT), side(OWN[1], dict(SECRET_CLIENT, max_in_flight=8) if regime_differs else SECRET_CLIENT)]
    if not gated:
        sides = [side(PARENTS, None), side(OWN[1], None)]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["additions"] = [a for a in manifest.get("additions", []) if "phase branch" not in a.get("added", [])]
    manifest["store_changes"] = [change(sides[0], sides[1], before, at=TE.MANIFEST_AT)]  # (the fixture's own change form)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    TE._stamp_by_endpoint(out, PARENTS, OWN[1])  # the copied records first, the partition's own after the change point
    if branch:
        TE._set_branch_phase(out, "2026-10-23T09:19:00.000+00:00")
    TE._edit_identity(out, lambda i: i.update(parents_store={
        "out": str(out), "replica": "copy-p", "endpoint": PARENTS, "store_json": None, "store_json_sha256": None,
        "identity_check": TE.PLAIN_NOTE}))
    TE._recheck(out)
    return inputs, outs


def prepare(inputs, outs, tmp_path):
    return ex.prepare(F, TE._plain_union_call(inputs, outs, str(tmp_path / "union"), TE._plain_union_pins(outs)),
                      map_record=inputs["map_record"])


def test_the_gated_two_endpoint_plain_union_fixture_runs_through_check_stores(tmp_path, monkeypatch):
    inputs, outs = gated_union(tmp_path, monkeypatch)
    _, union, context = prepare(inputs, outs, tmp_path)
    assert union.k == 2
    assert sorted(context["stores"].copies) == ["copy-0", "copy-1", "copy-p"]


def test_the_same_fixture_with_the_gate_regime_changed_is_refused(tmp_path, monkeypatch):
    inputs, outs = gated_union(tmp_path, monkeypatch, regime_differs=True)
    with pytest.raises(ex.Refused, match=r"its manifest store change 1: its sides' store_client differ beyond their endpoint"):
        prepare(inputs, outs, tmp_path)


def test_the_ungated_twin_of_the_fixture_runs_as_before(tmp_path, monkeypatch):
    inputs, outs = gated_union(tmp_path, monkeypatch, gated=False)
    _, union, context = prepare(inputs, outs, tmp_path)
    assert sorted(context["stores"].copies) == ["copy-0", "copy-1", "copy-p"]


def test_a_gated_union_without_the_plain_options_is_refused_with_todays_text(tmp_path, monkeypatch):
    """check_stores of a non-plain call (plain_union False) refuses the gated sides: the flag is the call's, never the manifest's."""
    inputs, outs = gated_union(tmp_path, monkeypatch)
    manifest_outs = [ex.Out.read(o) for o in outs]
    for o in manifest_outs:
        got, problems = ex.manifest_store_changes(o)
        if o.path == outs[1]:
            assert got == [] and [p.split("side has ")[1] for p in problems] == [TODAYS_TEXT, TODAYS_TEXT]
        else:
            assert (got, problems) == ([], [])


def test_the_gated_fixture_with_a_branch_phase_after_the_change_runs_through_check_stores(tmp_path, monkeypatch):
    """The manifest enters the branch phase after the change's first record on the new copy (no branch row is in the fixture:
    the plain-run fixture has prompt rows only)."""
    inputs, outs = gated_union(tmp_path, monkeypatch, branch=True)
    _, union, context = prepare(inputs, outs, tmp_path)
    assert sorted(context["stores"].copies) == ["copy-0", "copy-1", "copy-p"]
