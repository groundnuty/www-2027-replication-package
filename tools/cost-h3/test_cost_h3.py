"""Tests of cost_h3.py on SYNTHETIC data only (shaped like harness/agent/record.py, harness/checkloop/ledger.py and
harness/sources/generated.py at e7ceaf0); no real record, ledger or draw is read. Run:  python3 -I -m unittest -v test_cost_h3
"""
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cost_h3 as C  # noqa: E402

TOOL = os.path.join(HERE, "cost_h3.py")
SENT = "SENTINEL_9d41c7"      # a string planted in every barred field
SENT_NUM = 31415.926535        # a number planted in barred numeric fields
SENT_INT = 7654321
GRAPH = "uniprot-sp-2026_03"


# -- synthetic builders ------------------------------------------------------------------------------------------


def turn_entry(n, started, inp, out, cache_r=0, cache_w=0, latency=1.5):
    return {"turn": n, "started_s": started, "input_tokens": inp, "output_tokens": out, "cache_read_tokens": cache_r,
            "cache_write_tokens": cache_w, "max_tokens": 16000, "latency_s": latency, "first_byte_s": 0.2, "wall_s": latency,
            "retries": 0, "retry_errors": [], "stop_reason": SENT, "truncated": False, "served_model": "m",
            "model_mismatch": False, "usage_raw": {"note": SENT, "x": SENT_INT}, "system_build_tag": None}


def base_record(model, per_turn, wall_s, checker=None, arm=None):
    return {
        "schema_version": "agent-run/22", "run_id": "rid-" + SENT, "question_id": "q-" + SENT, "base_question_id": None,
        "template": SENT, "split": "test", "repetition": 0, "condition": "none", "model": model,
        "channel": {"x": SENT}, "store": {"name": GRAPH}, "prompt": {"x": SENT}, "agent_config": {"x": SENT},
        "harness": {"x": SENT}, "started_at": "2026-10-22T00:00:00Z", "ended_at": "2026-10-22T00:10:00Z", "wall_s": wall_s,
        "transcript": [{"role": "user", "text": SENT}], "tool_calls": [{"name": "sparql_query", "result_raw": SENT}],
        "tokens": {"totals": {"input_tokens": SENT_INT}, "per_turn": per_turn},
        "outcome": {"status": SENT, "failure_type": SENT, "label": SENT, "turns": len(per_turn)},
        "final_rows": [[SENT]], "score": {"exact": SENT_NUM, "outcome": SENT}, "replay_of": None, "runner": None,
        "load": {"arm": arm, "x": SENT}, "checker": checker,
    }


def parent_turns(n=3):
    return [turn_entry(i + 1, 10.0 * i, 100 + i, 10 + i, 5, 0) for i in range(n)]


def make_checker(mode, arm_name, parent_id, retry):
    return {
        "settings": {"version": "checker-loop/2", "mode": mode, "arm": {"name": arm_name, "x": SENT}, "x": SENT},
        "branched_from": parent_id,
        "initial": {"outcome": {"status": SENT, "turns": 3}, "final_rows": [[SENT]], "score": {"exact": SENT_NUM},
                    "submission_index": SENT_INT, "matching": {}, "checked_query": SENT, "checked_query_status": SENT,
                    "verdict": {"status": SENT, "retry": SENT, "fired": [SENT], "fired_checks": [{"reason": SENT}]},
                    "reports": [{"evidence": {"budget_used": SENT_INT}}], "store_log": [SENT]},
        "retry": retry,
    }


class World:
    """A synthetic OUT directory: manifest.json and records/, with helpers to add parents and branch records."""

    def __init__(self, root, name, channels=None, cells=("fb:curated", "fb:union", "gen:any", "dep:generated-1")):
        self.dir = os.path.join(root, name)
        os.makedirs(os.path.join(self.dir, "records"))
        self.counter = 0
        channels = channels or {"claude-haiku-5-5": "claude-haiku-5-5", "dsk": "deepseek-ai/DeepSeek-V4-Flash-0731"}
        manifest = {"schema": "pilot-grid-manifest/1", "other": SENT,
                    "fingerprint": {"phases": {"branch": {"arms": {c: {"x": SENT} for c in cells}}, "natural": {"arms": {}}},
                                    "channels": {k: {"model": v, "x": SENT} for k, v in channels.items()}}}
        with open(os.path.join(self.dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh)

    def _name(self, run_id, claim=None):
        self.counter += 1
        claim = claim or f"{self.counter:040x}"
        return f"{claim}.{run_id}.json"

    def add(self, run_id, rec, claim=None):
        with open(os.path.join(self.dir, "records", self._name(run_id, claim)), "w") as fh:
            json.dump(rec, fh)

    def add_parent(self, run_id, model="claude-haiku-5-5", turns=None):
        turns = turns or parent_turns()
        self.add(run_id, base_record(model, turns, 30.0, None, arm="none"))
        return turns

    def add_branch(self, run_id, parent_id, parent_turns_, live, cell="fb:curated", mode="feedback", arm_name=None,
                   model="claude-haiku-5-5", wall_s=None, retry_started=30.0, retry_turns=None, retried=True, claim=None,
                   declared_started=None, infra_try_ids=None):
        """live: list of (input, output, cache_r, cache_w, latency) of the retry turns."""
        n0 = len(parent_turns_)
        entries = [dict(t) for t in parent_turns_]
        for i, (inp, out, cr, cw, lat) in enumerate(live):
            entries.append(turn_entry(n0 + i + 1, retry_started + 5.0 * i, inp, out, cr, cw, lat))
        if arm_name is None:
            arm_name = cell.split(":", 1)[1] if cell != "gen:any" else "curated"
        retry = None
        if retried:
            retry = {"message_kind": "checker_feedback", "message": SENT,
                     "started_s": retry_started if declared_started is None else declared_started,
                     "turns": len(live) if retry_turns is None else retry_turns, "nudges": 0, "submission_index": SENT_INT}
        wall = retry_started + 5.0 * len(live) + 2.0 if wall_s is None else wall_s
        rec = base_record(model, entries, wall, make_checker(mode, arm_name, parent_id, retry), arm=cell)
        if infra_try_ids is not None:
            rec["runner"] = {"infra_tries": len(infra_try_ids), "infra_try_run_ids": infra_try_ids}
        self.add(run_id, rec, claim)


def write_ledger(path, rows, header=None):
    with open(path, "w") as fh:
        fh.write(json.dumps(header or {"schema": "checker-ledger/2", "split": "test"}) + "\n")
        for r in rows:
            full = {"split": "test", "run_id": SENT, "question_id": SENT, "initial_correct": True, "status": SENT,
                    "substantive": True, "flagged_check_types": [SENT], "retry_status": SENT, "retry_correct": True,
                    "retry_usd": SENT_NUM, "result_rows": SENT_INT, "base_question_id": SENT, "fault_class": SENT,
                    "bank_item": SENT, "any_unknown": False, "checked_query": SENT, "data_fault": False,
                    "branch_local_only": False, "number_traceability": SENT, "questions_sha256": None,
                    "graph_outside_census": None, "graph": GRAPH, "cohort": "none", "check_s": None}
            full.update(r)
            fh.write(json.dumps(full) + "\n")


DEFAULT_ROW = {"model": "claude-haiku-5-5", "source": "curated", "use": "replay", "evidence_triples": 5}


def write_draws(d, costs=(1.0, 0.5, 0.5), extra_superseded_draw=False):
    os.makedirs(d)
    for i, cost in enumerate(costs, 1):
        rec = {"draw": i, "attempt": 1, "batch_id": SENT, "model_requested": SENT, "model_returned": "claude-opus-5",
               "status": "ok", "cache_hit": i > 1, "cost_usd": cost, "output_sha256": SENT, "response_id": SENT,
               "usage": {"input_tokens": 100 * i, "output_tokens": 1000 * i, "cache_read_input_tokens": 50 * i,
                         "cache_creation_input_tokens": 70 * i,
                         "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 70 * i}, "tier": SENT},
               "prices_usd_per_mtok": {"input": 5, "output": 25, "cache_read": 0.5, "cache_write_1h": 10, "batch_discount": 0.5}}
        with open(os.path.join(d, f"draw-{i}.json"), "w") as fh:
            json.dump(rec, fh)
        with open(os.path.join(d, f"draw-{i}.txt"), "w") as fh:
            fh.write(SENT + " reply text that must never be read")
    batches = {"request_sha256": SENT, "count": {"input_tokens": 12345, "max_tokens": 1, "fits": True},
               "batches": [{"group": "first", "draws": [1], "batch_id": SENT, "submitted_utc": "2026-10-23T10:00:00Z",
                            "ended_utc": "2026-10-23T10:05:00Z", "collected": True, "request_counts": {"x": SENT}},
                           {"group": "rest", "draws": [2, 3], "batch_id": SENT, "submitted_utc": "2026-10-23T10:06:00Z",
                            "ended_utc": "2026-10-23T10:16:30Z", "collected": True}]}
    with open(os.path.join(d, "batches.json"), "w") as fh:
        json.dump(batches, fh)
    sup = os.path.join(d, "superseded", "abc123")
    os.makedirs(sup)
    with open(os.path.join(sup, "SUPERSEDED.json"), "w") as fh:
        json.dump({"reason": SENT}, fh)
    if extra_superseded_draw:
        with open(os.path.join(sup, "draw-1.json"), "w") as fh:
            json.dump({"draw": 1}, fh)


def run_tool(root, outs, completes, ledger, draws, extra=(), ledger_sha=None):
    cmd = [sys.executable, "-I", TOOL, "--graph", GRAPH, "--ledger", ledger,
           "--out-json", os.path.join(root, "out.json"), "--out-txt", os.path.join(root, "out.txt"), *extra]
    for label, path in outs:
        cmd += ["--out", f"{label}={path}"]
    for label, path in completes:
        cmd += ["--complete", f"{label}={path}"]
    for label, path in draws:
        cmd += ["--draws", f"{label}={path}"]
    if ledger_sha:
        cmd += ["--ledger-sha256", ledger_sha]
    return subprocess.run(cmd, capture_output=True, text=True)


def marker(root, name):
    p = os.path.join(root, name)
    open(p, "w").close()
    return p


# -- tests -------------------------------------------------------------------------------------------------------


class Quantiles(unittest.TestCase):
    def test_known_answers(self):
        s = C.summary(list(range(1, 11)))
        self.assertEqual((s["n"], s["median"], s["p90"]), (10, 5.5, 9.1))
        s = C.summary([2, 1])
        self.assertEqual((s["n"], s["median"], s["p90"]), (2, 1.5, 1.9))
        s = C.summary([7])
        self.assertEqual((s["n"], s["median"], s["p90"]), (1, 7, 7))
        s = C.summary([1, 2, 3])
        self.assertEqual((s["median"], s["p90"]), (2, 2.8))
        self.assertEqual(C.summary([]), {"n": 0, "median": "not estimable", "p90": "not estimable"})

    def test_agrees_with_statistics_inclusive(self):
        rng = random.Random(5)
        for n in (2, 3, 7, 10, 11, 50, 101):
            vals = [rng.random() * 1000 for _ in range(n)]
            want = statistics.quantiles(vals, n=10, method="inclusive")[8]
            self.assertAlmostEqual(C.percentile(sorted(vals), 0.9), want, places=9)
            self.assertAlmostEqual(C.percentile(sorted(vals), 0.5), statistics.median(vals), places=9)


class Allowlist(unittest.TestCase):
    def test_record_view_refuses_barred_paths(self):
        v = C.RecordView(base_record("m", parent_turns(), 1.0, make_checker("feedback", "curated", "p", None), arm="fb:curated"))
        for path in [("outcome", "status"), ("outcome",), ("score",), ("score", "exact"), ("checker", "initial", "verdict"),
                     ("checker", "initial", "score"), ("checker", "initial"), ("checker", "retry", "message"),
                     ("checker", "retry", "submission_index"), ("final_rows",), ("transcript",), ("tool_calls",),
                     ("run_id",), ("question_id",), ("template",), ("checker", "settings"), ("tokens", "totals"),
                     ("tokens", "per_turn"), ("checker", "settings", "arm", "sources"), ("load",), ("channel",),
                     ("runner",), ("runner", "infra_try_run_ids"), ("runner", "infra_tries")]:
            with self.assertRaises(C.Forbidden, msg=str(path)):
                v.get(path)
        for path in [("checker", "initial"), ("outcome",), ("score",), ("checker", "retry", "message")]:
            with self.assertRaises(C.Forbidden):
                v.present(path)
        with self.assertRaises(C.Forbidden):
            v.keys(("checker", "initial"))

    def test_get_never_returns_a_container(self):
        v = C.RecordView({"model": {"a": 1}, "wall_s": [1, 2], "load": {"arm": {"x": 1}}})
        self.assertIsNone(v.get(("model",)))
        self.assertIsNone(v.get(("wall_s",)))
        self.assertIsNone(v.get(("load", "arm")))

    def test_ledger_manifest_draw_views_refuse(self):
        lv = C.LedgerView({"initial_correct": True})
        for name in ("initial_correct", "status", "substantive", "retry_correct", "retry_status", "flagged_check_types",
                     "run_id", "question_id", "retry_usd", "result_rows", "fault_class"):
            with self.assertRaises(C.Forbidden, msg=name):
                lv.get((name,))
        mv = C.ManifestView({})
        for path in [("fingerprint", "shared"), ("starts",), ("fingerprint", "phases", "natural", "arms")]:
            with self.assertRaises(C.Forbidden):
                mv.keys(path)
        dv = C.DrawView({})
        for path in [("output_sha256",), ("response_id",), ("batch_id",), ("stop_reason",), ("error",)]:
            with self.assertRaises(C.Forbidden):
                dv.get(path)

    def test_part_holds_only_allowed_values(self):
        with tempfile.TemporaryDirectory() as root:
            w = World(root, "o")
            pt = w.add_parent("par")
            w.add_branch("b1", "par", pt, [(10, 2, 0, 0, 1.0)])
            names = os.listdir(os.path.join(w.dir, "records"))
            part = None
            for n in names:
                p = C.load_part(os.path.join(w.dir, "records", n))
                if p.has_checker:
                    part = p
            dumped = repr({s: getattr(part, s) for s in part.__slots__})
            self.assertNotIn(SENT, dumped)
            self.assertNotIn(str(SENT_INT), dumped)
            self.assertNotIn("31415", dumped)


class Retry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, w_list, ledger_rows=None, draws=True, completes=True, extra=()):
        ledger = os.path.join(self.root, "ledger.jsonl")
        write_ledger(ledger, ledger_rows if ledger_rows is not None else [DEFAULT_ROW])
        dd = []
        if draws:
            d = os.path.join(self.root, "gen")
            if not os.path.isdir(d):
                write_draws(d)
            dd = [("generated", d)]
        outs = [(w.label, w.dir) for w in w_list]
        comp = [(w.label, marker(self.root, "done-" + w.label)) for w in w_list] if completes else []
        r = run_tool(self.root, outs, comp, ledger, dd, extra)
        return r

    def result(self):
        with open(os.path.join(self.root, "out.json")) as fh:
            return json.load(fh)

    def mk(self, name, **kw):
        w = World(self.root, name, **kw)
        w.label = name
        return w

    def test_known_answers_pooled_and_per_model(self):
        w = self.mk("o1")
        pt = w.add_parent("par1")
        w.add_parent("par2", model="deepseek-ai/DeepSeek-V4-Flash-0731")
        # prefix turns carry large tokens that must NOT be counted: parent_turns() has input 100.. per turn
        w.add_branch("b1", "par1", pt, [(1000, 100, 10, 5, 2.0), (500, 50, 0, 0, 3.0)], cell="fb:curated",
                     retry_started=30.0, wall_s=30.0 + 20.0)
        w.add_branch("b2", "par1", pt, [(300, 30, 0, 0, 1.0)], cell="fb:curated", retry_started=30.0, wall_s=30.0 + 8.0)
        pd = w.add_parent("par3", model="deepseek-ai/DeepSeek-V4-Flash-0731", turns=parent_turns(2))
        w.add_branch("b3", "par3", pd, [(700, 70, 0, 0, 4.0)], cell="fb:curated", model="deepseek-ai/DeepSeek-V4-Flash-0731",
                     retry_started=30.0, wall_s=30.0 + 14.0)
        w.add_branch("g1", "par1", pt, [(40, 4, 0, 0, 1.0)], cell="gen:any", mode="generic", retry_started=30.0, wall_s=33.0)
        w.add_branch("n1", "par1", pt, [], cell="fb:curated", retried=False)  # flagged by nothing: no retry
        r = self.go([w])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        res = self.result()
        cur = res["retry"]["by_cell"]["fb:curated"]
        self.assertEqual(cur["all"]["n_retried"], 3)
        # io tokens: b1 = 1000+100+500+50 = 1650; b2 = 330; b3 = 770 -> median 770, p90 = 770 + .8*(1650-770) = 1474
        self.assertEqual(cur["all"]["io_tokens"]["median"], 770)
        self.assertAlmostEqual(cur["all"]["io_tokens"]["p90"], 1474.0, places=3)
        # all tokens: b1 adds cache 15 -> 1665
        self.assertEqual(cur["all"]["all_tokens"]["median"], 770)
        self.assertAlmostEqual(cur["all"]["all_tokens"]["p90"], 770 + 0.8 * (1665 - 770), places=3)
        # latency: 20, 8, 14
        self.assertEqual(cur["all"]["latency_s"]["median"], 14)
        self.assertAlmostEqual(cur["all"]["latency_s"]["p90"], 14 + 0.8 * 6, places=3)
        # model call time: b1 5.0, b2 1.0, b3 4.0
        self.assertEqual(cur["all"]["model_call_s"]["median"], 4)
        self.assertEqual(cur["models"]["claude-haiku-5-5"]["n_retried"], 2)
        key = "deepseek-ai-deepseek-v4-flash-0731"  # the model id of the record, normalised
        self.assertEqual(cur["models"][key]["n_retried"], 1)
        self.assertEqual(cur["models"][key]["io_tokens"]["median"], 770)
        self.assertEqual(res["retry"]["by_cell"]["gen:any"]["all"]["n_retried"], 1)
        self.assertEqual(res["retry"]["by_source"]["generic"]["all"]["n_retried"], 1)
        self.assertEqual(res["retry"]["by_source"]["curated"]["all"]["n_retried"], 3)
        # a source with no retry in this world is simply absent, never a zero-with-denominator
        self.assertNotIn("fb:union", res["retry"]["by_cell"])

    def test_prefix_mismatch_parent_absent_and_structure_exclusions(self):
        w = self.mk("o2")
        pt = w.add_parent("par")
        # replayed prefix differs from the parent's
        other = [dict(t) for t in pt]
        other[1]["input_tokens"] += 1
        w.add_branch("bad_prefix", "par", other, [(5, 5, 0, 0, 1.0)])
        # parent not in the OUT
        w.add_branch("no_parent", "ghost", pt, [(5, 5, 0, 0, 1.0)])
        # retry.turns disagrees with the entries: it claims every entry is live (no prefix), or two live turns where
        # there is one (the claimed first live turn is a parent turn that started before retry.started_s)
        w.add_branch("bad_count", "par", pt, [(5, 5, 0, 0, 1.0)], retry_turns=4)
        w.add_branch("bad_count2", "par", pt, [(5, 5, 0, 0, 1.0)], retry_turns=2)
        # retry.started_s after the first live turn started
        w.add_branch("bad_boundary", "par", pt, [(5, 5, 0, 0, 1.0)], retry_started=30.0, declared_started=31.0)
        # cell not pinned, and cell/arm mismatch
        w.add_branch("not_pinned", "par", pt, [(5, 5, 0, 0, 1.0)], cell="fb:census-t5")
        w.add_branch("mismatch", "par", pt, [(5, 5, 0, 0, 1.0)], cell="fb:union", arm_name="curated")
        w.add_branch("ok", "par", pt, [(5, 5, 0, 0, 1.0)])
        r = self.go([w])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        res = self.result()
        d = res["retry"]["diagnostics"]
        self.assertEqual(res["retry"]["by_cell"]["fb:curated"]["all"]["n_retried"], 1)
        for key in ("excluded_parent_prefix_mismatch", "excluded_parent_absent", "excluded_count_mismatch",
                    "excluded_cell_not_pinned", "excluded_cell_mismatch"):
            self.assertEqual(d.get(key), 1, key)
        self.assertEqual(d.get("excluded_boundary_mismatch"), 2)

    def test_turn_numbering_and_unreported_tokens(self):
        w = self.mk("o3")
        pt = w.add_parent("par")
        w.add_branch("num", "par", pt, [(5, 5, 0, 0, 1.0)])
        # break the numbering of the last record's turns
        rec_dir = os.path.join(w.dir, "records")
        for n in os.listdir(rec_dir):
            p = os.path.join(rec_dir, n)
            with open(p) as fh:
                rec = json.load(fh)
            if n.split(".")[1] == "num":
                rec["tokens"]["per_turn"][-1]["turn"] = 99
                with open(p, "w") as fh:
                    json.dump(rec, fh)
        # a live turn without output tokens: latency still counts, tokens do not
        w.add_branch("unrep", "par", pt, [(5, None, 0, 0, 1.0)], retry_started=30.0, wall_s=36.0)
        r = self.go([w])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        res = self.result()
        b = res["retry"]["by_cell"]["fb:curated"]["all"]
        self.assertEqual(res["retry"]["diagnostics"].get("excluded_turn_numbering"), 1)
        self.assertEqual(b["n_retried"], 1)
        self.assertEqual(b["io_tokens"]["n"], 0)
        self.assertEqual(b["io_tokens"]["median"], "not estimable")
        self.assertEqual(b["latency_s"]["n"], 1)
        self.assertEqual(b["latency_s"]["median"], 6)

    def test_duplicate_run_ids_across_outs_count_once(self):
        a, b = self.mk("a"), self.mk("b")
        for w in (a, b):
            pt = w.add_parent("par")
            w.add_branch("same-run", "par", pt, [(5, 5, 0, 0, 1.0)])
        r = self.go([a, b])
        self.assertEqual(r.returncode, 0)
        res = self.result()
        self.assertEqual(res["retry"]["by_cell"]["fb:curated"]["all"]["n_retried"], 1)
        self.assertEqual(res["retry"]["diagnostics"]["duplicate_run_ids"], 1)

    def test_incomplete_out_is_not_opened(self):
        done, zzpending = self.mk("done"), self.mk("zzpending")
        pt = done.add_parent("par")
        done.add_branch("b", "par", pt, [(5, 5, 0, 0, 1.0)])
        import shutil
        shutil.rmtree(os.path.join(zzpending.dir, "records"))  # opening its records would crash the run
        ledger = os.path.join(self.root, "ledger.jsonl")
        write_ledger(ledger, [DEFAULT_ROW])
        r = run_tool(self.root, [("done", done.dir), ("zzpending", zzpending.dir)], [("done", marker(self.root, "m1"))], ledger, [])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        res = self.result()
        self.assertEqual(res["outs_excluded"], [{"label": "zzpending", "reason": "phase not complete: excluded"}])
        self.assertEqual([x["label"] for x in res["outs_included"]], ["done"])
        self.assertNotIn("zzpending", json.dumps(res["retry"]))

    def test_claim_digest_groups_resolve_by_infra_try_ids(self):
        w = self.mk("c")
        pt = w.add_parent("par")
        claim = "ab" * 20
        # two records of one attempt: the final one names the earlier try as its infra try -> one answer
        w.add_branch("try1", "par", pt, [], retried=True, claim=claim, retry_turns=0, retry_started=30.0, wall_s=95.0)
        w.add_branch("try2", "par", pt, [(6, 6, 0, 0, 1.0)], claim=claim, infra_try_ids=["try1"])
        # a second group that does not resolve (two records, none naming the other): left out whole
        c2 = "cd" * 20
        w.add_branch("amb1", "par", pt, [(5, 5, 0, 0, 1.0)], claim=c2)
        w.add_branch("amb2", "par", pt, [(7, 7, 0, 0, 1.0)], claim=c2)
        # and a plain single record
        w.add_branch("single", "par", pt, [(8, 8, 0, 0, 1.0)])
        r = self.go([w])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        res = self.result()
        d = res["retry"]["diagnostics"]
        self.assertEqual(res["retry"]["by_cell"]["fb:curated"]["all"]["n_retried"], 2)  # try2 and single
        self.assertEqual(res["retry"]["by_cell"]["fb:curated"]["all"]["io_tokens"]["median"], 14.0)  # (12 + 16) / 2
        self.assertEqual(d["claim_digest_groups_over_one_record"], 2)
        self.assertEqual(d["claim_digest_groups_ambiguous_left_out"], 1)

    def test_zero_live_turn_retry_is_counted_with_zero_tokens(self):
        w = self.mk("z")
        pt = w.add_parent("par")
        w.add_branch("zero", "par", pt, [], retry_started=30.0, wall_s=95.0, retry_turns=0)
        w.add_branch("one", "par", pt, [(5, 5, 0, 0, 1.0)], retry_started=30.0, wall_s=36.0)
        r = self.go([w])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        res = self.result()
        b = res["retry"]["by_cell"]["fb:curated"]["all"]
        self.assertEqual(b["n_retried"], 2)
        self.assertEqual(b["io_tokens"]["n"], 2)
        self.assertEqual(b["io_tokens"]["median"], 5.0)  # values 0 and 10
        self.assertEqual(b["latency_s"]["median"], 35.5)  # 65 and 6
        self.assertEqual(res["retry"]["diagnostics"]["retried_with_zero_live_turns"], 1)
        self.assertEqual(res["retry"]["diagnostics"].get("excluded_retry_malformed"), None)

    def test_a_copy_in_a_second_out_is_judged_on_its_own(self):
        a = self.mk("a", cells=("fb:union",))  # the cell is not pinned in OUT a
        b = self.mk("b")
        for w in (a, b):
            pt = w.add_parent("par")
            w.add_branch("same-run", "par", pt, [(5, 5, 0, 0, 1.0)])
        r = self.go([a, b])
        self.assertEqual(r.returncode, 0)
        res = self.result()
        self.assertEqual(res["retry"]["by_cell"]["fb:curated"]["all"]["n_retried"], 1)  # the valid copy, in b
        self.assertEqual(res["retry"]["diagnostics"]["excluded_cell_not_pinned"], 1)

    def test_out_labels_may_hold_dots(self):
        self.assertEqual(C.labelled(["uniprot-retry-glm-5.3-flash=/p/x"]), [("uniprot-retry-glm-5.3-flash", "/p/x")])
        for bad in ("=/p", "a b=/p", "a/b=/p", "x", "a=" ):
            with self.assertRaises(C.Bad):
                C.labelled([bad])

    def test_one_model_under_two_channel_names_is_one_key(self):
        a = self.mk("a", channels={"haiku": "claude-haiku-5-5"})
        b = self.mk("b", channels={"haiku-p2": "claude-haiku-5-5"})
        for w in (a, b):
            pt = w.add_parent("par-" + w.label)
            w.add_branch("br-" + w.label, "par-" + w.label, pt, [(5, 5, 0, 0, 1.0)])
        r = self.go([a, b])
        self.assertEqual(r.returncode, 0)
        self.assertEqual(sorted(self.result()["retry"]["by_cell"]["fb:curated"]["models"]), ["claude-haiku-5-5"])

    def test_a_forbidden_read_stops_the_run_it_is_not_an_unreadable_file(self):
        import unittest.mock as mock
        w = self.mk("f")
        pt = w.add_parent("par")
        w.add_branch("b", "par", pt, [(5, 5, 0, 0, 1.0)])
        with mock.patch.object(C, "load_part", side_effect=C.Forbidden("location")):
            with self.assertRaises(C.Forbidden):
                C.process_out(w.dir, {})

    def test_output_is_invariant_to_every_barred_field(self):
        """Two worlds identical in the allowed fields and different in every barred field give identical figures."""
        outs = []
        for variant in ("A", "B"):
            root = os.path.join(self.root, variant)
            os.makedirs(root)
            w = World(root, "o")
            pt = w.add_parent("par")
            w.add_branch("b1", "par", pt, [(100, 10, 5, 1, 2.0), (200, 20, 0, 0, 3.0)])
            w.add_branch("b2", "par", pt, [(300, 30, 0, 0, 1.0)])
            rec_dir = os.path.join(w.dir, "records")
            for n in sorted(os.listdir(rec_dir)):
                p = os.path.join(rec_dir, n)
                with open(p) as fh:
                    rec = json.load(fh)
                if variant == "B":
                    for k in ("outcome", "score", "final_rows", "transcript", "tool_calls", "question_id", "template",
                              "channel", "prompt", "agent_config"):
                        rec[k] = {"changed": "B-" + k, "n": 99}
                    rec["tokens"]["totals"] = {"other": 1}
                    if rec.get("checker"):
                        ch = rec["checker"]
                        ch["initial"] = {"verdict": {"status": "other"}, "score": {"exact": False}, "final_rows": [["x"]]}
                        if ch.get("retry"):
                            ch["retry"].update(message="B message", message_kind="other", submission_index=1, nudges=3)
                    for t in rec["tokens"]["per_turn"]:
                        t["stop_reason"] = "B"
                        t["usage_raw"] = {"B": 1}
                        t["retry_errors"] = ["B"]
                    # the parent's per-turn entries change too, in the same way in parent and prefix (the prefix check holds)
                with open(p, "w") as fh:
                    json.dump(rec, fh)
            ledger = os.path.join(root, "l.jsonl")
            rows = [{"model": "claude-haiku-5-5", "source": "curated", "use": "replay", "evidence_triples": 5}]
            if variant == "B":
                rows[0].update(initial_correct=False, status="Z", substantive=False, retry_correct=False, retry_status="Q",
                               retry_usd=1.0, result_rows=3)
            write_ledger(ledger, rows)
            r = run_tool(root, [("o", w.dir)], [("o", marker(root, "m"))], ledger, [])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            with open(os.path.join(root, "out.json")) as fh:
                doc = json.load(fh)
            for k in ("run_utc", "outs_included"):
                doc.pop(k)
            doc["ledger"].pop("sha256")
            outs.append(doc)
        self.assertEqual(outs[0], outs[1])

    def test_an_earlier_try_in_another_out_is_dropped_in_both_orders(self):
        """OUT a holds the final try (naming try1 as its earlier try), OUT b holds a copy of try1: one answer, not two."""
        for order in (("a", "b"), ("b", "a")):
            root = os.path.join(self.root, "".join(order))
            os.makedirs(root)
            a, b = World(root, "a"), World(root, "b")
            a.label, b.label = "a", "b"
            pa = a.add_parent("par")
            a.add_branch("try2", "par", pa, [(6, 6, 0, 0, 1.0)], claim="ab" * 20, infra_try_ids=["try1"])
            pb = b.add_parent("par")
            b.add_branch("try1", "par", pb, [(5, 5, 0, 0, 1.0)], claim="ab" * 20)  # a copy of the earlier try
            ledger = os.path.join(root, "l.jsonl")
            write_ledger(ledger, [DEFAULT_ROW])
            worlds = {"a": a, "b": b}
            r = run_tool(root, [(k, worlds[k].dir) for k in order], [(k, marker(root, "m" + k)) for k in order], ledger, [])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            with open(os.path.join(root, "out.json")) as fh:
                res = json.load(fh)
            self.assertEqual(res["retry"]["by_cell"]["fb:curated"]["all"]["n_retried"], 1, order)
            self.assertEqual(res["retry"]["by_cell"]["fb:curated"]["all"]["io_tokens"]["median"], 12.0, order)  # try2 alone
            self.assertEqual(res["retry"]["diagnostics"]["dropped_named_as_earlier_try"], 1, order)

    def test_zero_live_turn_series_and_negative_latency_are_reported(self):
        w = self.mk("e")
        pt = w.add_parent("par")
        w.add_branch("zero", "par", pt, [], retry_started=30.0, wall_s=95.0, retry_turns=0)
        w.add_branch("one", "par", pt, [(5, 5, 0, 0, 1.0)], retry_started=30.0, wall_s=36.0)
        w.add_branch("neg", "par", pt, [(7, 7, 0, 0, 1.0)], retry_started=30.0, wall_s=20.0)  # wall_s before retry start
        r = self.go([w])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        res = self.result()
        b = res["retry"]["by_cell"]["fb:curated"]["all"]
        self.assertEqual(b["n_retried"], 3)
        self.assertEqual(b["io_tokens"]["n"], 3)
        self.assertEqual(b["io_tokens_excluding_zero_live_turns"]["n"], 2)
        self.assertEqual(b["io_tokens_excluding_zero_live_turns"]["median"], 12.0)  # 10 and 14
        self.assertEqual(b["all_tokens_excluding_zero_live_turns"]["n"], 2)
        self.assertEqual(b["latency_s"]["n"], 2)  # the negative one is left out
        d = res["retry"]["diagnostics"]
        self.assertEqual((d["retried_with_zero_live_turns"], d["retried_with_negative_latency_left_out_of_latency"]), (1, 1))

    def test_no_sentinel_reaches_any_output(self):
        w = self.mk("s")
        pt = w.add_parent("parent-xq7")
        for i in range(4):
            w.add_branch(f"branch-xq7-{i}", "parent-xq7", pt, [(10 * i + 1, 3, 0, 0, 1.0)])
        rows = [{"model": "claude-haiku-5-5", "source": "curated", "use": "replay", "evidence_triples": 5},
                {"model": "claude-haiku-5-5", "source": "union", "use": "replay", "evidence_triples": 9},
                {"model": "claude-haiku-5-5", "source": "generic_retry/curated", "use": "checker", "evidence_triples": 4}]
        r = self.go([w], ledger_rows=rows)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        blob = r.stdout + r.stderr
        for name in ("out.json", "out.txt"):
            with open(os.path.join(self.root, name)) as fh:
                blob += fh.read()
        for forbidden in (SENT, str(SENT_INT), "31415", "reply text", "rid-", "q-" + SENT):
            self.assertNotIn(forbidden, blob)
        # no per-record or run label either: the synthetic run ids and claim digests
        for n in os.listdir(os.path.join(w.dir, "records")):
            self.assertNotIn(n.split(".")[0], blob)
            self.assertNotIn(n.split(".")[1], blob)

    def test_outputs_are_mode_600_and_errors_are_one_fixed_line(self):
        w = self.mk("m")
        pt = w.add_parent("par")
        w.add_branch("b", "par", pt, [(5, 5, 0, 0, 1.0)])
        r = self.go([w])
        self.assertEqual(r.returncode, 0)
        for name in ("out.json", "out.txt"):
            self.assertEqual(os.stat(os.path.join(self.root, name)).st_mode & 0o777, 0o600)
        bad = subprocess.run([sys.executable, "-I", TOOL, "--graph", GRAPH, "--ledger", "/nonexistent/secret-name",
                              "--out-json", os.path.join(self.root, "x.json"), "--out-txt", os.path.join(self.root, "x.txt")],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1)
        self.assertEqual(bad.stdout.strip(), "error: the figures could not be computed (no detail is printed)")
        self.assertNotIn("secret-name", bad.stdout + bad.stderr)
        self.assertNotIn("Traceback", bad.stdout + bad.stderr)
        worse = subprocess.run([sys.executable, "-I", TOOL, "--bogus", "/p/secret"], capture_output=True, text=True)
        self.assertEqual(worse.returncode, 2)
        self.assertNotIn("secret", worse.stdout + worse.stderr)


class Ledger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def run_ledger(self, rows, sha=None, header=None):
        path = os.path.join(self.root, "l.jsonl")
        write_ledger(path, rows, header)
        return C.process_ledger(path, GRAPH, sha)

    def test_filters_and_separation(self):
        rows = []
        for i, v in enumerate([10, 20, 30, 40, 50]):
            rows.append({"model": "deepseek-ai/DeepSeek-V4-Flash-0731", "source": "census-t5", "use": "replay", "evidence_triples": v})
        rows += [{"model": "claude-haiku-5-5", "source": "census-t5", "use": "replay", "evidence_triples": 100}]
        rows += [{"model": "claude-haiku-5-5", "source": "census-t5", "use": "replay", "evidence_triples": None}]
        rows += [{"model": "claude-haiku-5-5", "source": "census-t5", "use": "checker", "evidence_triples": 7}]
        rows += [{"model": "claude-haiku-5-5", "source": "census-t5/no_heuristic", "use": "replay", "evidence_triples": 1}]
        rows += [{"model": "bank", "source": "census-t5", "use": "replay", "evidence_triples": 2, "cohort": "bank"}]
        rows += [{"model": "claude-haiku-5-5", "source": "census-t5", "use": "replay", "evidence_triples": 3, "graph": "rhea"}]
        rows += [{"model": "claude-haiku-5-5", "source": "generic_retry/curated", "use": "checker", "evidence_triples": 11}]
        out = self.run_ledger(rows)
        rep = out["evidence_triples"]["replay"]["census-t5"]
        self.assertEqual(rep["all"]["n"], 6)  # 5 + 1; the None row, the online row, the variant, bank and rhea rows are out
        self.assertEqual(rep["models"]["deepseek-ai-deepseek-v4-flash-0731"]["median"], 30)
        self.assertEqual(rep["models"]["claude-haiku-5-5"]["n"], 1)
        self.assertEqual(out["evidence_triples"]["online"]["census-t5"]["all"]["n"], 1)
        self.assertEqual(out["evidence_triples"]["online"]["generic_retry/curated"]["all"]["median"], 11)
        t = out["tally"]
        self.assertEqual((t["evidence_not_recorded"], t["rows_variant_source"], t["rows_bank"], t["rows_other_graph"]),
                         (1, 1, 1, 1))
        self.assertEqual(out["check_s"], "not recorded")
        self.assertEqual(t["check_s_not_none"], 0)

    def test_check_s_recorded_is_reported_not_hidden(self):
        out = self.run_ledger([{"model": "claude-haiku-5-5", "source": "curated", "use": "replay", "evidence_triples": 1,
                                "check_s": 2.5}])
        self.assertEqual(out["tally"]["check_s_not_none"], 1)
        self.assertEqual(out["check_s"]["n"], 1)

    def test_hash_and_schema_are_checked(self):
        import hashlib
        path = os.path.join(self.root, "l.jsonl")
        write_ledger(path, [{"model": "m", "source": "curated", "use": "replay", "evidence_triples": 1}])
        with open(path, "rb") as fh:
            good = hashlib.sha256(fh.read()).hexdigest()
        self.assertEqual(C.process_ledger(path, GRAPH, good)["sha256"], good)
        with self.assertRaises(C.Bad):
            C.process_ledger(path, GRAPH, "0" * 64)
        with self.assertRaises(C.Bad):
            self.run_ledger([], header={"schema": "something-else/1", "split": "test"})

    def test_model_names_agree_between_id_forms_and_an_empty_graph_is_refused(self):
        out = self.run_ledger([{"model": "deepseek-ai/DeepSeek-V4-Flash-0731", "source": "curated", "use": "replay",
                                "evidence_triples": 3},
                               {"model": "deepseek-ai-deepseek-v4-flash-0731", "source": "curated", "use": "replay",
                                "evidence_triples": 4}])
        models = out["evidence_triples"]["replay"]["curated"]["models"]
        self.assertEqual(list(models), ["deepseek-ai-deepseek-v4-flash-0731"])
        self.assertEqual(models["deepseek-ai-deepseek-v4-flash-0731"]["n"], 2)
        with self.assertRaises(C.Bad):  # no row of the graph: the graph name is wrong
            self.run_ledger([{"model": "m", "source": "curated", "use": "replay", "evidence_triples": 1, "graph": "rhea"}])


class Draws(unittest.TestCase):
    def test_draw_usage_and_batches(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "generated")
            write_draws(d)
            out = C.process_draws("generated", d)
            self.assertEqual(out["total_cost_usd"], 2.0)
            self.assertEqual([x["draw"] for x in out["draws"]], [1, 2, 3])
            self.assertEqual(out["draws"][1]["usage"]["input_tokens"], 200)
            self.assertEqual(out["draws"][2]["usage"]["cache_creation_1h_tokens"], 210)
            self.assertEqual([b["wall_s"] for b in out["batches"]], [300.0, 630.0])
            self.assertEqual(out["input_tokens_counted_before_submission"], 12345)
            self.assertEqual(out["superseded_draw_records"], 0)
            self.assertEqual(out["earlier_attempts"], [])
            self.assertNotIn(SENT, json.dumps(out))
            d2 = os.path.join(root, "g2")
            write_draws(d2, extra_superseded_draw=True)
            self.assertEqual(C.process_draws("g", d2)["superseded_draw_records"], 1)
            d3 = os.path.join(root, "g3")
            write_draws(d3)
            with open(os.path.join(d3, "draw-2.attempt-1.json"), "w") as fh:
                json.dump({"draw": 2, "attempt": 1, "status": "failed: " + SENT, "cost_usd": 0.25, "output_sha256": SENT,
                           "usage": {"input_tokens": 9, "output_tokens": 8}}, fh)
            out3 = C.process_draws("g", d3)
            self.assertEqual(out3["total_cost_usd"], 2.0)
            self.assertEqual(out3["total_cost_usd_with_earlier_attempts"], 2.25)
            self.assertEqual([(x["draw"], x["attempt"], x["status_ok"]) for x in out3["earlier_attempts"]], [(2, 1, False)])
            self.assertNotIn(SENT, json.dumps(out3))
            d4 = os.path.join(root, "g4")
            write_draws(d4)
            with open(os.path.join(d4, "draw-1.json")) as fh:
                rec = json.load(fh)
            rec["cost_usd"] = "free"
            rec["usage"]["input_tokens"] = "many"
            rec["usage"]["output_tokens"] = -5
            rec["cache_hit"] = "yes"
            rec["draw"] = True
            with open(os.path.join(d4, "draw-1.json"), "w") as fh:
                json.dump(rec, fh)
            out4 = C.process_draws("g", d4)
            first = out4["draws"][0]
            self.assertEqual((first["cost_usd"], first["usage"]["input_tokens"], first["usage"]["output_tokens"],
                              first["cache_hit"], first["draw"]), (None, None, None, None, None))
            self.assertEqual(out4["total_cost_usd"], 1.0)  # the two other draws only


if __name__ == "__main__":
    unittest.main()
