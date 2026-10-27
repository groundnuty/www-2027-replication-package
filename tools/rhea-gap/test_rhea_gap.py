"""Tests of rhea_gap.py on SYNTHETIC data only (no look, record or result is read). The frozen analysis modules (stats, metrics,
inputs, prereg) are imported from a frozen code tree: RHEA_GAP_CODE_ROOT, default the exploratory-exploratory worktree whose
harness/analysis, harness/score and scripts are byte-equal to 847cc275.

    RHEA_GAP_CODE_ROOT=... python -m pytest tests          (from the directory holding rhea_gap.py)
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
CODE_ROOT = Path(os.environ.get("RHEA_GAP_CODE_ROOT", "/work/code-root"))
sys.path.insert(0, str(CODE_ROOT))

import rhea_gap as G  # noqa: E402
from harness.analysis import inputs as I  # noqa: E402,N812
from harness.analysis import metrics as M  # noqa: E402,N812
from harness.analysis import prereg as P  # noqa: E402,N812
from harness.analysis import stats as S  # noqa: E402,N812

F = types.SimpleNamespace(stats=S, metrics=M)


# --- the bootstrap: known answers ---------------------------------------------------------------------------------


def singletons_K(n_q: int, n_boot: int = 400, seed: int = G.SEED) -> np.ndarray:
    index = S.QuestionIndex(tuple(f"q{i}" for i in range(n_q)), ("all",) * n_q)
    return S.resample_matrix(index, n_boot, seed=seed, scheme="unstratified")


def test_hand_computed_draws_and_estimate():
    # three hand-made draws over two single-question clusters: the draw rows are (1,0), (0,1), (1,1)
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    x_nat, x_opq, d = np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.ones(2)
    nat = S.ratio_draws(K, x_nat, d)
    opq = S.ratio_draws(K, x_opq, d)
    assert list(opq - nat) == [0.0, 1.0, 0.5]  # (1-1), (1-0), (1-0.5)
    gap = G.paired_gap(S, K, x_nat, d, x_opq, d)
    assert gap["estimate"] == pytest.approx(0.5)  # opaque 1.0 minus natural 0.5
    # inverted-CDF 2.5% and 97.5% quantiles of three draws: the smallest and the largest
    assert gap["ci"] == [0.0, 1.0]
    assert gap["undefined_share"] == 0.0


def test_constant_shift_gives_a_degenerate_interval_at_the_shift():
    K = singletons_K(40)
    rng = np.random.default_rng(1)
    x_nat = rng.integers(0, 2, 40).astype(float)
    d = np.ones(40)
    gap = G.paired_gap(S, K, x_nat, d, x_nat + 0.25, d)
    assert gap["estimate"] == pytest.approx(0.25)
    assert gap["ci"] == pytest.approx([0.25, 0.25], abs=1e-12)


def test_identical_arms_give_exactly_zero_where_an_unpaired_interval_is_wide():
    K = singletons_K(60)
    rng = np.random.default_rng(2)
    x = rng.integers(0, 2, 60).astype(float)
    d = np.ones(60)
    assert G.paired_gap(S, K, x, d, x.copy(), d)["ci"] == [0.0, 0.0]
    lo, hi = G.marginal_ci(S, K, x, d)
    assert hi - lo > 0.1  # the marginal interval of one arm is not degenerate: pairing is what removes it


def test_cluster_resampling_is_what_makes_the_paired_interval_degenerate():
    # cluster A = {q0, q1}: natural (0, 1), opaque (1, 0): per-question differences +1 and -1 that cancel inside the cluster;
    # the other clusters are singletons with equal values
    questions = [f"q{i}" for i in range(12)]
    cluster = {q: q for q in questions}
    cluster["q1"] = "q0"
    nat_map = P.ClusterMap(cluster)
    index = S.QuestionIndex(tuple(questions), ("all",) * 12)
    cfg = types.SimpleNamespace(seed=G.SEED, n_boot=500)
    clusters = nat_map.resampling(index, cfg)
    K_cluster = clusters.matrix[:, clusters.group]
    x_nat = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], dtype=float)
    x_opq = x_nat.copy()
    x_opq[0], x_opq[1] = 1.0, 0.0
    d = np.ones(12)
    assert G.paired_gap(S, K_cluster, x_nat, d, x_opq, d)["ci"] == pytest.approx([0.0, 0.0], abs=1e-12)
    K_question = singletons_K(12, 500)  # question-level resampling splits the cluster: not degenerate
    lo, hi = G.paired_gap(S, K_question, x_nat, d, x_opq, d)["ci"]
    assert lo < 0 < hi


def test_the_draws_are_the_frozen_cluster_bootstrap():
    questions = [f"q{i}" for i in range(9)]
    cluster = {q: f"c{i // 3}" for i, q in enumerate(questions)}  # three clusters of three
    cfg = types.SimpleNamespace(seed=G.SEED, n_boot=300)
    K = G.shared_K(None, F, cfg, P.ClusterMap(cluster), questions, ["all"] * 9)
    expected = S.resample_matrix(S.QuestionIndex(("c0", "c1", "c2"), ("all",) * 3), 300, seed=G.SEED, scheme="unstratified")
    assert K.shape == (300, 9)
    assert np.array_equal(K[:, [0, 3, 6]], expected)  # one column per cluster, the frozen draws
    assert np.array_equal(K[:, 0], K[:, 1]) and np.array_equal(K[:, 1], K[:, 2])  # members share their cluster's weight
    assert (K.sum(axis=1) == 9).all()  # every draw takes 3 clusters of 3 questions


# --- pairing ----------------------------------------------------------------------------------------------------------


def cell(graph: str, model: str, questions: list[str], arms: dict[str, list[float]], clusters: dict[str, str] | None = None,
         ok: bool = True) -> dict:
    clusters = clusters or {q: q for q in questions}
    return {
        "schema": G.CELL_SCHEMA, "name": f"{graph}-{model}", "graph": graph, "model": model, "ok": ok, "union": None,
        "index": {"questions": list(questions), "strata": ["all"] * len(questions), "cluster": [clusters[q] for q in questions]},
        "arms": {c: {"label": f"{graph}|{model}|{c}|prompt|-", "n_units": 3 * len(questions), "estimate": float(np.mean(v)),
                     "ci": None, "num": list(v), "den": [1.0] * len(questions),
                     "shares": {"attempts": 3 * len(questions), "without_record": 0,
                                "rows": {"dep-a": {"n": 3 * len(questions), "share": 1.0}}}}
                 for c, v in arms.items()}}


QS = [f"q{i}" for i in range(20)]
V = [float(i % 2) for i in range(20)]


def test_pair_arms_aligns_by_base_id_not_by_position():
    K = singletons_K(20)
    nat = cell(G.NAT, "m", QS, {"natural": V})
    perm = list(reversed(range(20)))
    opq = cell(G.OPQ, "m", [QS[i] for i in perm], {"natural": [V[i] + 0.0 for i in perm]})
    pair = G.pair_arms(S, K, nat, opq, "natural")
    assert pair["difference_opaque_minus_natural"]["estimate"] == 0.0
    assert pair["difference_opaque_minus_natural"]["ci"] == [0.0, 0.0]
    # a reordered but identical opaque arm is the same pair as an unshifted one
    same = G.pair_arms(S, K, nat, cell(G.OPQ, "m", QS, {"natural": V}), "natural")
    assert same["difference_opaque_minus_natural"] == pair["difference_opaque_minus_natural"]


def test_pair_arms_refuses_other_base_questions():
    K = singletons_K(20)
    nat = cell(G.NAT, "m", QS, {"natural": V})
    other = cell(G.OPQ, "m", [*QS[:19], "extra"], {"natural": V})
    with pytest.raises(G.GapRefused, match="base questions differ"):
        G.pair_arms(S, K, nat, other, "natural")


def test_pair_arms_refuses_different_clusters_for_a_shared_question():
    K = singletons_K(20)
    nat = cell(G.NAT, "m", QS, {"natural": V})
    clusters = {q: q for q in QS}
    clusters["q3"] = "q2"
    with pytest.raises(G.GapRefused, match="different clusters"):
        G.pair_arms(S, K, nat, cell(G.OPQ, "m", QS, {"natural": V}, clusters), "natural")


def test_pair_arms_refuses_a_shared_question_with_no_units():
    K = singletons_K(20)
    nat = cell(G.NAT, "m", QS, {"natural": V})
    opq = cell(G.OPQ, "m", QS, {"natural": V})
    opq["arms"]["natural"]["den"][4] = 0.0
    with pytest.raises(G.GapRefused, match="no units"):
        G.pair_arms(S, K, nat, opq, "natural")


def test_pair_arms_refuses_a_repeated_base_question():
    K = singletons_K(20)
    nat = cell(G.NAT, "m", QS, {"natural": V})
    dup = cell(G.OPQ, "m", QS, {"natural": V})
    dup["index"]["questions"][5] = dup["index"]["questions"][4]
    with pytest.raises(G.GapRefused, match="repeats"):
        G.pair_arms(S, K, nat, dup, "natural")


def test_build_pairs_lists_unpaired_models_and_arms_and_refusals():
    K = singletons_K(20)
    by = G.group_cells([
        cell(G.NAT, "both", QS, {"natural": V, "plain": V, "only-nat": V}),
        cell(G.OPQ, "both", QS, {"natural": V, "plain": V, "only-opq": V}),
        cell(G.NAT, "nat-only-model", QS, {"natural": V}),
        cell(G.OPQ, "opq-only-model", QS, {"natural": V}),
        cell(G.NAT, "failed", QS, {"natural": V}, ok=False),
        cell(G.OPQ, "failed", QS, {"natural": V}),
        cell(G.NAT, "mismatch", QS, {"natural": V}),
        cell(G.OPQ, "mismatch", [*QS[:19], "zz"], {"natural": V}),
    ])
    pairs, unpaired_models, unpaired_arms, refused, repro = G.build_pairs(S, K, by)
    assert [(p["model"], p["arm"]) for p in pairs] == [("both", "natural"), ("both", "plain")]
    assert sorted((u["model"], u["graph"]) for u in unpaired_models) == [
        ("nat-only-model", G.NAT), ("opq-only-model", G.OPQ)]
    assert sorted((u["arm"], u["graph"]) for u in unpaired_arms) == [("only-nat", G.NAT), ("only-opq", G.OPQ)]
    assert sorted(r["model"] for r in refused) == ["failed", "mismatch"]
    assert "cross-check failed" in next(r for r in refused if r["model"] == "failed")["reason"]
    assert set(repro) == {"both/natural", "both/plain"}


def test_two_cells_of_one_model_on_one_graph_are_refused():
    with pytest.raises(G.GapRefused, match="two cells"):
        G.group_cells([cell(G.NAT, "m", QS, {"natural": V}), {**cell(G.NAT, "m", QS, {"natural": V}), "name": "other"}])


def test_a_pair_holds_no_per_question_values():
    K = singletons_K(20)
    pair = G.pair_arms(S, K, cell(G.NAT, "m", QS, {"natural": V}), cell(G.OPQ, "m", QS, {"natural": V}), "natural")
    pair.pop("_same_order")
    text = G.dump(pair)
    assert '"num"' not in text and '"den"' not in text and "q7" not in text

    def longest_list(x):
        if isinstance(x, list):
            return max([len(x), *map(longest_list, x)])
        if isinstance(x, dict):
            return max([0, *map(longest_list, x.values())])
        return 0
    assert longest_list(pair) <= 2  # only [lo, hi] intervals


def test_render_text_shows_the_shares_beside_the_gap():
    K = singletons_K(20)
    pairs, um, ua, refused, _ = G.build_pairs(S, K, G.group_cells([cell(G.NAT, "m", QS, {"natural": V}),
                                                                   cell(G.OPQ, "m", QS, {"natural": V})]))
    table = {"bootstrap": {"n_boot": 400, "seed": G.SEED, "level": 0.95, "n_clusters": 20, "n_questions": 20},
             "pairs": pairs, "unpaired_models": um, "unpaired_arms": ua, "refused_pairs": refused}
    text = G.render_text(table)
    assert "dep-a 100.0% (60)" in text and "natural" in text and "+0.0" in text


# --- the accuracy of an arm, by the frozen code ---------------------------------------------------------------------


def attempt(qid: str, rep: int, correct: bool | None, cohort: str = "natural", base: str | None = None) -> I.Attempt:
    if correct is None:  # a missing attempt
        return I.Attempt(run_id=f"missing:{qid}:{rep}", graph="g", model="m", cohort=cohort, question_id=qid,
                         base_question_id=base or qid, template="t", split="test", repetition=rep, status="missing",
                         output=False, correct=False, f1=0.0, missing_reason="absent")
    return I.Attempt(run_id=f"r:{qid}:{rep}", graph="g", model="m", cohort=cohort, question_id=qid,
                     base_question_id=base or qid, template="t", split="test", repetition=rep, status="submitted",
                     output=True, correct=correct, f1=float(correct))


def test_arm_accuracy_is_the_per_question_mean_then_the_mean_over_questions():
    attempts = [attempt("a", 0, True), attempt("a", 1, True), attempt("b", 0, True), attempt("b", 1, False),
                attempt("c", 0, None), attempt("c", 1, False)]
    index = S.QuestionIndex(("a", "b", "c"), ("t|x",) * 3)
    arms = M.build_arms(attempts, [], {})
    ((key, units),) = arms.items()
    assert key.use == M.PROMPT
    vec, num, den, est = G.arm_accuracy(F, units, index)
    assert list(num) == [1.0, 0.5, 0.0] and list(den) == [1.0, 1.0, 1.0]  # the missing attempt counts wrong
    assert est == pytest.approx(0.5)
    assert list(vec["n"]) == [2, 2, 2]


def test_opaque_ids_pair_through_base_question_id():
    # an opaque question has its own id and the natural question's id as base_question_id: the arm is keyed by the base id
    attempts = [attempt("opaque-xyz", 0, True, base="nat-1"), attempt("opaque-xyz", 1, False, base="nat-1")]
    ((key, units),) = M.build_arms(attempts, [], {}).items()
    index = S.QuestionIndex(("nat-1",), ("t|x",))
    _, num, den, est = G.arm_accuracy(F, units, index)
    assert list(num) == [0.5] and est == 0.5


# --- the checks against a look --------------------------------------------------------------------------------------


def look_arm(est, ci, n):
    return {"metrics": {"accuracy": {"estimate": est, "ci": ci}}, "counts": {"N_att": n}}


def shares(n_with, n_without, **rows):
    return {"attempts": n_with, "without record": n_without,
            "rows": {d: {"n": n, "of": n_with, "share": n / n_with} for d, n in rows.items()}}


def test_arm_checks_pass_on_equal_and_fail_on_any_difference():
    sh = shares(9, 1, a=6, b=3)
    ok = G.arm_checks(0.5, [0.25, 0.75], 10, look_arm(0.5, [0.25, 0.75], 10), sh)
    assert all(ok.values())
    assert not G.arm_checks(0.5000001, [0.25, 0.75], 10, look_arm(0.5, [0.25, 0.75], 10), sh)["estimate_equals_look"]
    assert not G.arm_checks(0.5, [0.25, 0.76], 10, look_arm(0.5, [0.25, 0.75], 10), sh)["ci_equals_look"]
    assert not G.arm_checks(0.5, [0.25, 0.75], 11, look_arm(0.5, [0.25, 0.75], 10), sh)["n_att_equals_look"]
    assert not G.arm_checks(0.5, [0.25, 0.75], 11, look_arm(0.5, [0.25, 0.75], 11), sh)["shares_tied"]
    assert not any(G.arm_checks(0.5, None, 10, None, None).values())  # an arm the look lacks


def test_share_tied_needs_every_count_to_add_up():
    assert G.share_tied(shares(9, 1, a=6, b=3), 10)
    assert not G.share_tied(shares(9, 1, a=6, b=2), 10)  # rows do not add up to the attempts
    assert not G.share_tied(shares(9, 0, a=6, b=3), 10)  # attempts do not cover the units
    assert not G.share_tied({"rows": {}}, 10)


# --- the cells files ----------------------------------------------------------------------------------------------------

SINGLE = "|".join(["rhea-x", G.NAT, "/runs/rhea-x", "/q/questions.jsonl", "1", "0", "22", "16", "", "", "natural,prompted", "",
                   "/t/TIMEFILES.tsv:" + "a" * 64])


def test_parse_single_cell():
    spec = G.parse_cell_line(SINGLE)
    assert spec.name == "rhea-x" and spec.outs == ("/runs/rhea-x",) and spec.union is None and spec.cap_gb == 16
    with_store_records = SINGLE.replace("|22|16||", "|22|16|/p/a.json,/p/b.json|")
    assert G.parse_cell_line(with_store_records).store_records == ("/p/a.json", "/p/b.json")


@pytest.mark.parametrize("bad", [
    lambda s: s.replace("|1|0|", "|0|0|"),  # a cell without a ledger
    lambda s: s.replace("|1|0|", "|1|1|"),  # an a15 cell
    lambda s: s.replace("/runs/rhea-x", "runs/rhea-x"),  # a relative file name
    lambda s: s.replace("/runs/rhea-x", "/runs/../rhea-x"),
    lambda s: s.replace("/runs/rhea-x", "/runs/rhea x"),
    lambda s: s.replace("/runs/rhea-x", "/runs/a,/runs/b"),  # two OUTs on a single line
    lambda s: s.replace(G.NAT, "uniprot"),
    lambda s: s.replace("natural,prompted", "branch"),
    lambda s: s + "|x",  # a 14th field that is not a union spec
])
def test_parse_refuses_malformed_lines(bad):
    with pytest.raises(G.GapRefused):
        G.parse_cell_line(bad(SINGLE))


def union_line(prefix="opaque-qwen3-coder-30b-a3b-deepseek", k=2, rsha="639bb65af66828821bb774bf9e65b6194510fe343587c145a1615fb66b856714",
               graph=G.OPQ, binding=""):
    outs = ",".join(f"/u/{prefix}-p{p}of{k}" for p in range(k))
    spec = f"{prefix}:{k}:/u/record.json:{rsha}" + (f":{binding}" if binding else "")
    return "|".join(["u", graph, outs, "/q/q.jsonl", "1", "0", "30", "24", "/p/a.json", "", "natural,prompted", "",
                     "/t/T.tsv:" + "a" * 64, spec])


def test_parse_union_cell_and_its_refusals():
    spec = G.parse_cell_line(union_line())
    assert spec.union.k == 2 and len(spec.outs) == 2 and spec.union.binding is None
    for bad in (union_line(rsha="0" * 64),  # not the pinned record
                union_line(k=3), union_line(prefix="opaque-unknown"),
                union_line(graph=G.NAT),
                union_line(binding="/u/binding-dir"),  # a binding on a union without one
                union_line().replace("-p1of2", "-p2of2")):  # OUTs not p0..p(K-1)
        with pytest.raises(G.GapRefused):
            G.parse_cell_line(bad)
    deepseek_b = union_line("opaque-deepseek-v4-flash-0731-deepseek_b", 6, "98158f127c0c7545d5c7255cfa7760dc6b9ff6a9d3b80b7929354cb003dc1cb7",
                     binding="/u/deepseek_b")
    assert G.parse_cell_line(deepseek_b).union.binding == "/u/deepseek_b"
    with pytest.raises(G.GapRefused):  # the deepseek_b union needs its binding
        G.parse_cell_line(deepseek_b.rsplit(":", 1)[0])


def test_read_plan_and_load_spec_check_the_cells_file_pin(tmp_path):
    cells = tmp_path / "cells-b-rhea-20261025T184400Z.txt"
    cells.write_text(SINGLE + "\n")
    plan = tmp_path / "plan.tsv"
    plan.write_text(f"{cells}\t{G.sha256_file(cells)}\trhea-x\n")
    (row,) = G.read_plan(plan)
    spec, stamp = G.load_spec(row)
    assert spec.name == "rhea-x" and stamp == "20261025T184400Z"
    plan.write_text(f"{cells}\t{'0' * 64}\trhea-x\n")
    with pytest.raises(G.GapRefused, match="sha256 differs"):
        G.load_spec(G.read_plan(plan)[0])
    plan.write_text(f"{cells}\t{G.sha256_file(cells)}\trhea-x\n{cells}\t{G.sha256_file(cells)}\trhea-x\n")
    with pytest.raises(G.GapRefused, match="repeats"):
        G.read_plan(plan)


def test_the_look_results_path_of_a_single_and_a_union_cell():
    assert G.results_path(Path("/o"), "S", G.parse_cell_line(SINGLE)) == Path("/o/look-S/rhea-x/results.json")
    assert G.results_path(Path("/o"), "S", G.parse_cell_line(union_line())) == Path("/o/look-S/u/A/results.json")


def test_compact_shares_keeps_counts_and_shares_only():
    out = G.compact_shares(shares(9, 1, a=6, b=3))
    assert out == {"attempts": 9, "without_record": 1,
                   "rows": {"a": {"n": 6, "share": 6 / 9}, "b": {"n": 3, "share": 3 / 9}}}


def test_the_status_file_carries_no_result(tmp_path):
    args = types.SimpleNamespace(status=tmp_path / "s.jsonl")
    G.write_status(args, {"stage": "cell", "rc": 0, "ok": True})
    line = json.loads((tmp_path / "s.jsonl").read_text())
    assert line["schema"] == G.STATUS_SCHEMA and line["ok"] is True
    assert oct((tmp_path / "s.jsonl").stat().st_mode & 0o777) == "0o600"


# --- the reviewer's nits: real configuration rows only on both graphs; the expected model set ---------------------------------------


def test_configurations_on_both_graphs_leave_out_the_non_configurations():
    K = singletons_K(20)
    nat = cell(G.NAT, "m", QS, {"natural": V})
    opq = cell(G.OPQ, "m", QS, {"natural": V})
    for c in (nat, opq):
        c["arms"]["natural"]["shares"]["rows"] = {"dep-a": {"n": 30, "share": 0.5}, "mixed": {"n": 20, "share": 0.3},
                                                  "retry without record": {"n": 10, "share": 0.2}}
    not_configurations = ("mixed", "unattributed", "no own turns", "retry without record")
    assert G.pair_arms(S, K, nat, opq, "natural")["configurations_on_both_graphs"] == [
        "dep-a", "mixed", "retry without record"]  # nothing excluded without the list
    assert G.pair_arms(S, K, nat, opq, "natural", not_configurations)["configurations_on_both_graphs"] == ["dep-a"]
    pairs, *_ = G.build_pairs(S, K, G.group_cells([nat, opq]), not_configurations)
    assert pairs[0]["configurations_on_both_graphs"] == ["dep-a"]


def test_check_models_refuses_a_missing_extra_or_renamed_model():
    both = lambda names: G.group_cells([c for n in names for c in (cell(G.NAT, n, QS, {"a": V}), cell(G.OPQ, n, QS, {"a": V}))])  # noqa: E731
    G.check_models(both(["x", "y"]), ("x", "y"))
    with pytest.raises(G.GapRefused, match="lacks"):
        G.check_models(both(["x"]), ("x", "y"))
    with pytest.raises(G.GapRefused, match="unexpected"):
        G.check_models(both(["x", "y", "z"]), ("x", "y"))
    renamed = G.group_cells([cell(G.NAT, "x", QS, {"a": V}), cell(G.OPQ, "x-renamed", QS, {"a": V})])
    with pytest.raises(G.GapRefused):  # a model named differently on the two graphs does not land among the unpaired
        G.check_models(renamed, ("x",))
    assert len(G.EXPECTED_MODELS) == 8 and len(set(G.EXPECTED_MODELS)) == 8
