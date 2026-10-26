#!/usr/bin/env python3
"""The frozen fault probe of the test runs' infrastructure-fault rule (change-log entries 148 and 149, and the
amendment of 18:21; harness/agent/fault_probe.py): build the probe sets, assemble the fault-rule file, run a probe.

    # on the host of the dev records (read-only; writes only the probe sets):
    uv run python scripts/fault_probe.py build --runs-root ~/www-2027/runs --source dev-natural-20261020/uniprot \\
        --cut 2026-10-21T05:45:00+00:00 --seed 20261021 --out-dir data/runs/infra-fault/probe-sets
    # the fault-rule file and its pin in the frozen config:
    uv run python scripts/fault_probe.py rule --baselines data/runs/infra-fault/dev-baselines.json \\
        --probe-sets data/runs/infra-fault/probe-sets --out data/runs/infra-fault/fault-rule.json \\
        --pin data/freeze/frozen-config.json
    # a probe of one model's route (what the monitor runs once it holds the model):
    uv run python scripts/fault_probe.py run --model glm-5.3-flash --manifest OUT/manifest.json --out DIR \\
        [--expect-fingerprint FP ...] [--base-url-override http://127.0.0.1:PORT/v1] [--dry]

``run`` loads the fault-rule file only if its sha256 is the frozen config's pin (data/freeze/frozen-config.json
``fault_rule``) and the model's probe set only by the sha256 the rule binds; it rebuilds each request from the
repository (system prompt and tools checked against the recorded sha256) and sends the 20 requests one at a time
through the runner's own route for the model (scripts/run_pilot.py MODELS), at the base URL of the model's route in the
OUT's manifest (``--manifest``; GLM's names our SiteA server) or ``--base-url``, refusing a route that goes elsewhere;
with ``--manifest`` the replies' served fingerprints are compared with the ones the OUT recorded (served-models.json,
and those admitted beside it in admitted-fingerprints.json, change-log entry 167) unless ``--expect-fingerprint`` names
others. ``--base-url-override`` sends the requests to one backend's loopback listener on the front instead (entry 167),
the route still checked against the manifest's base URL and recorded as it; the listener gets the placeholder key.
Keys: the Gateway's dedicated key and Anthropic's are read from their mode-600 files under ``--key-dir`` into this
process's environment when it holds none (never printed); a base URL other than the Gateway's gets a placeholder key, so
the Gateway key never goes to another server. It writes DIR/rows.jsonl (counts and ids only) and DIR/summary.json (the
verdict, harness/agent/fault_probe.py judge), and exits 0 on a pass, 1 on a fail, 2 on a refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.agent import fault_probe as fp  # noqa: E402
from harness.agent import infra_fault as fault  # noqa: E402
from harness.agent.providers import MIXED_FINGERPRINTS  # noqa: E402

RULE_FILE = "data/runs/infra-fault/fault-rule.json"
PIN_FILE = "data/freeze/frozen-config.json"
PROBE_SETS = "data/runs/infra-fault/probe-sets"
# the panel P by the runner's names (data/analysis/prereg-config.json panel, by scripts/run_pilot.py MODELS' routes)
P_MODELS = ("claude-sonnet-5-5", "claude-haiku-5-5", "qwen3.8-27b", "glm-5.3-flash", "gemma-4-31b",
            "qwen3-coder-30b-a3b", "deepseek-v4-flash-0731")
GATEWAY_HOST = "gateway.example"
GATEWAY_KEY_FILE = "gateway_api_key_dedicated"  # AGENTS.md: every new run starts with the dedicated key
ANTHROPIC_KEY_FILE = "anthropic_api_key"
PLACEHOLDER_KEY = "self-hosted-no-key"
REFUSED = 2
ADMITTED_FILE = "admitted-fingerprints.json"  # scripts/run_pilot.py's (change-log entry 167)
ADMITTED_SCHEMA = "admitted-fingerprints/1"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def write_json(path: Path, doc: dict) -> str:
    raw = (json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


# --- build ----------------------------------------------------------------------------------------------


def _trim(rec: dict) -> dict:
    """What the selection and the dev counts read of a record (a record can be large)."""
    tr = rec.get("transcript") or []
    first_asst = next((i for i, t in enumerate(tr) if t.get("role") == "assistant"), len(tr))
    per_turn = ((rec.get("tokens") or {}).get("per_turn") or [{}])[:1]
    keep = ("run_id", "question_id", "started_at", "checker", "outcome", "agent_config", "store", "prompt", "model")
    return {**{k: rec.get(k) for k in keep}, "tokens": {"per_turn": per_turn}, "transcript": tr[:first_asst + 1]}


def cmd_build(args: argparse.Namespace) -> int:
    out = Path(args.runs_root).expanduser() / args.source
    names = fault.route_names(out)
    cut = fault.parse_time(args.cut)
    by_model: dict[str, list[tuple[str, dict]]] = {m: [] for m in args.models}
    for path in fault.record_files(out):
        rec = json.loads(path.read_text(encoding="utf-8"))
        name = names.get(rec.get("model"))
        if name not in by_model:
            continue
        arm = (rec.get("load") or {}).get("arm") or rec.get("condition")
        if fp.eligible(rec, name, arm, cut):
            by_model[name].append((arm, _trim(rec)))
    routes = {v: k for k, v in names.items()}
    for model, cands in by_model.items():
        readings = [fp.turn1_reading(r["transcript"][0].get("text") or "", fp.turn1(r)) for _, r in cands]
        ids = sorted(r["run_id"] for _, r in cands)
        dev = {**fp.dev_counts(readings), "source": args.source, "cut": cut.isoformat(),
               "run_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
               "reading": "each eligible record's recorded turn 1 (selection's filters but the healthy turn 1)"}
        source = {"out": args.source, "cut": cut.isoformat(), "seed": args.seed, "n": fp.PROBE_N,
                  "eligible": len(cands)}
        requests = fp.select(cands, args.seed, fp.PROBE_N)
        sha = write_json(Path(args.out_dir) / f"{model}.json",
                         fp.probe_set_doc(model, routes[model], source, requests, dev))
        print(f"{model}: {len(requests)} requests from {len(cands)} eligible records; dev turn 1 "
              f"no-call {dev['no_call']}/{dev['n']}, coverage<{fp.COVERAGE_OK} {dev['low_coverage']}/{dev['n']}, "
              f"missing {dev['missing']}/{dev['n']}; sha256 {sha}")
    return 0


def cmd_control(args: argparse.Namespace) -> int:
    """The positive control: a model's turn-1 counts on its prompted records from a fault's start on (GLM's server
    fault of 21 Oct, from 09:44Z), read as the probe reads them, so the rule can show the fault exceeds its k."""
    out = Path(args.runs_root).expanduser() / args.source
    names = fault.route_names(out)
    start = fault.parse_time(args.after)
    readings, ids = [], []
    for path in fault.record_files(out):
        rec = json.loads(path.read_text(encoding="utf-8"))
        name = names.get(rec.get("model"))
        if name != args.model:
            continue
        arm = (rec.get("load") or {}).get("arm") or rec.get("condition")
        if fp.eligible(rec, name, arm, start, before=False):
            readings.append(fp.turn1_reading(rec["transcript"][0].get("text") or "", fp.turn1(rec)))
            ids.append(rec["run_id"])
    doc = {"model": args.model, "source": args.source, "from": start.isoformat(), **fp.dev_counts(readings),
           "run_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest(),
           "reading": "each record's recorded turn 1, prompted arms, started at or after 'from', as the probe reads it"}
    sha = write_json(Path(args.out), doc)
    print(f"{args.model} from {start.isoformat()}: no-call {doc['no_call']}/{doc['n']}, coverage<{fp.COVERAGE_OK} "
          f"{doc['low_coverage']}/{doc['n']}, missing {doc['missing']}/{doc['n']}; sha256 {sha}")
    return 0


# --- rule -----------------------------------------------------------------------------------------------


CONFIRMATION = [
    "the model's frozen fault probe fails (harness/agent/fault_probe.py judge)",
    ("a served fingerprint other than the model's recorded ones (the probe's replies; none for a route that reports "
     "none, as Anthropic's)"),
    "a provider-confirmed restart or incident, recorded as an operator note (scripts/infra_fault_monitor.py confirm)",
]


def fault_control(control: tuple[str, str, dict], limits: dict[str, dict[str, dict]]) -> dict:
    """The positive control bound beside the thresholds: a fault's turn-1 counts (cmd_control) and, per count
    criterion, the chance that a probe of 20 requests at the fault's rate exceeds the model's k."""
    rel, sha, doc = control
    m = doc["model"]
    if m not in limits:
        raise ValueError(f"the control's model {m} is not in the rule")
    exceed = {}
    for c in fp.CRITERIA:
        k = limits[m][c]["k"]
        exceed[c] = None if k is None else round(float(fp.tail(fp.PROBE_N, Fraction(doc[c], doc["n"]), k)), 6)
    return {"file": rel, "sha256": sha, "model": m, "counts": {x: doc[x] for x in ("n", *fp.CRITERIA, "missing")},
            "source": {"out": doc["source"], "from": doc["from"], "run_ids_sha256": doc["run_ids_sha256"]},
            "exceed_k": exceed}


def rule_doc(baselines: dict, baselines_rel: str, baselines_sha: str, probe_sets: dict[str, tuple[str, str, dict]],
             models: tuple[str, ...], control: tuple[str, str, dict] | None = None) -> dict:
    missing = [m for m in models if m not in baselines["models"] or m not in probe_sets]
    if missing:
        raise ValueError(f"no dev rates or no probe set for {missing}")
    limits = {m: fp.thresholds(probe_sets[m][2]["dev_turn1"]) for m in models}
    return {
        "schema": fault.RULE_SCHEMA,
        "rule": ("change-log entries 129, 148 and 149 and the amendment of 18:21 (science, design log, 21 Oct): the "
                 "trigger only holds; a window needs a confirmation independent of the test attempts"),
        "models": list(models),
        "trigger": {"bin_s": fault.BIN_S, "bins": "clock-aligned 15-minute bins of UTC (:00, :15, :30, :45)",
                    "factor": fault.FACTOR, "min_events": fault.MIN_EVENTS, "compare": fault.RULE["compare"],
                    "metrics": {k: {"events": v[0], "of": v[1]} for k, v in fault.METRICS.items()},
                    "call_less_turn": fault.RULE["call_less_turn"], "no_submission": fault.RULE["no_submission"],
                    "left_out": fault.RULE["left_out"]},
        "dev_rates": {"file": baselines_rel, "sha256": baselines_sha, "sources": baselines["sources"],
                      "code": baselines.get("code"), "branch": baselines.get("branch"),
                      "models": {m: baselines["models"][m] for m in models}},
        "probe": {"n": fp.PROBE_N, "retries": fp.RETRIES, "coverage_ok": fp.COVERAGE_OK,
                  "false_fail": f"{fp.FALSE_FAIL.numerator}/{fp.FALSE_FAIL.denominator}",
                  "criteria": {"no_call": "turn 1 without a tool call",
                               "low_coverage": f"turn-1 question coverage below {fp.COVERAGE_OK} (reasoning and "
                                               f"visible text)"},
                  "threshold_rule": ("per model and criterion, p = (x + 1) / (n + 2) from the healthy dev counts x of "
                                     "n; k = the smallest count with P(Binomial(20, p) > k) <= false_fail, and k >= "
                                     "k_floor; null (not applicable) where k would be 20 (design ruling, design log, "
                                     "21 Oct 19:52)"),
                  "k_floor": fp.K_FLOOR,
                  "missing_detector": {"pattern": fp.MISSING_PATTERN, "flags": "IGNORECASE",
                                       "sha256": fp.MISSING_SHA256, "applies_to": "the visible text", "max": 0},
                  "stop_words": sorted(fp.STOP),
                  "thresholds": limits,
                  "dev_turn1": {m: probe_sets[m][2]["dev_turn1"] for m in models}},
        "probe_sets": {m: {"file": probe_sets[m][0], "sha256": probe_sets[m][1],
                           "requests": len(probe_sets[m][2]["requests"]), "source": probe_sets[m][2]["source"]}
                       for m in models},
        "fault_control": fault_control(control, limits) if control else None,
        "confirmation": CONFIRMATION,
        "window": fault.RULE["window"],
        "not_confirmed": "the attempts stand, scored as usual; the model resumes; the hold is recorded",
    }


def cmd_rule(args: argparse.Namespace) -> int:
    base_path = ROOT / args.baselines
    raw = base_path.read_bytes()
    sets = {}
    for m in args.models:
        rel = f"{args.probe_sets}/{m}.json"
        praw = (ROOT / rel).read_bytes()
        sets[m] = (rel, hashlib.sha256(praw).hexdigest(), json.loads(praw))
    control = None
    if args.control:
        craw = (ROOT / args.control).read_bytes()
        control = (args.control, hashlib.sha256(craw).hexdigest(), json.loads(craw))
    doc = rule_doc(json.loads(raw), args.baselines, hashlib.sha256(raw).hexdigest(), sets, tuple(args.models), control)
    if args.stores:
        doc["stores"] = json.loads((ROOT / args.stores).read_text(encoding="utf-8"))
    sha = write_json(ROOT / args.out, doc)
    print(f"{args.out}: sha256 {sha}")
    for m in args.models:
        t = doc["probe"]["thresholds"][m]
        print(f"  {m}: probe set {sets[m][1]}; no-call k {t['no_call']['k']} ({t['no_call']['dev_events']}/"
              f"{t['no_call']['dev_n']}), coverage k {t['low_coverage']['k']} ({t['low_coverage']['dev_events']}/"
              f"{t['low_coverage']['dev_n']})")
    if args.pin:
        pin_path = ROOT / args.pin
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
        pin["fault_rule"] = {"path": args.out, "sha256": sha}
        pin_path.write_text(json.dumps(pin, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{args.pin}: fault_rule pinned")
    return 0


# --- run ------------------------------------------------------------------------------------------------


def load_runner():
    spec = importlib.util.spec_from_file_location("run_pilot_for_probe", str(ROOT / "scripts" / "run_pilot.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_pilot_for_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def set_keys(base_url: str | None, provider: str, key_dir: Path) -> None:
    """The route's key into this process's environment, from its mode-600 file when the environment holds none."""
    def from_file(env: str, name: str) -> None:
        if (os.environ.get(env) or "").strip():
            return
        path = key_dir / name
        if path.exists():
            os.environ[env] = path.read_text(encoding="utf-8").strip()

    if provider == "anthropic":
        from_file("ANTHROPIC_API_KEY", ANTHROPIC_KEY_FILE)
    elif base_url and urlparse(base_url).hostname != GATEWAY_HOST:
        os.environ.pop("CENTREB_API_KEY", None)
        os.environ["LLM_GATEWAY_API_KEY"] = PLACEHOLDER_KEY
    else:
        from_file("LLM_GATEWAY_API_KEY", GATEWAY_KEY_FILE)


def rebuild(req: dict) -> tuple[str, tuple]:
    """A request's system prompt and tools from the repository, refused unless they hash to the recorded values."""
    from harness.agent.loop import tools_sha256
    from harness.agent.prompts import get_condition, load_block, sha256_text
    from harness.agent.tools import agent_tools

    block = load_block(req["ontology_block_sha256"]) if req.get("ontology_block_sha256") else None
    system = get_condition(req["condition"]).render(row_cap=req["row_cap"], timeout_s=req["sparql_timeout_s"],
                                                    ontology_block=block)
    if sha256_text(system) != req["system_sha256"]:
        raise ValueError(f"request {req['i']}: the system prompt does not rebuild to its recorded sha256")
    tools = agent_tools(req["max_result_chars"])
    if tools_sha256(tools) != req["tools_sha256"]:
        raise ValueError(f"request {req['i']}: the tools do not rebuild to their recorded sha256")
    return system, tools


def read_admitted(path: Path) -> dict[str, list[str]]:
    """Each model's admitted system fingerprints in ``path`` (OUT/admitted-fingerprints.json, change-log entry 167),
    sorted; {} without the file; ValueError when it is malformed anywhere: a copy of scripts/run_pilot.py read_admitted,
    which the runner's check reads, and of scripts/infra_fault_monitor.py's (tests/test_e167_admitted_fingerprints.py
    holds the three to one result)."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}
    doc = json.loads(raw)
    if not isinstance(doc, dict) or doc.get("schema") != ADMITTED_SCHEMA or not isinstance(doc.get("models"), dict):
        raise ValueError(f"not an {ADMITTED_SCHEMA} file with a models object")
    out = {}
    for model, entry in doc["models"].items():
        admitted = entry.get("admitted") if isinstance(entry, dict) else None
        if not isinstance(admitted, list):
            raise ValueError(f"{model}: no admitted list")  # noqa: TRY004 - the file's content
        fingerprints = set()
        for item in admitted:
            fp_ = item.get("system_fingerprint") if isinstance(item, dict) else None
            if not isinstance(fp_, str) or not fp_ or MIXED_FINGERPRINTS in fp_:
                raise ValueError(f"{model}: an admitted entry without one system fingerprint")
            fingerprints.add(fp_)
        out[model] = sorted(fingerprints)
    return out


def recorded_fingerprints(manifest: Path, model: str) -> list[str]:
    """The model's fingerprints as its OUT recorded them, the reference a probe's replies are compared with: the served
    one (OUT/served-models.json, beside the manifest: the first reply's) and those admitted beside it
    (OUT/admitted-fingerprints.json, change-log entry 167; a malformed file adds none, as the runner's check reads it,
    with a warning); [] when none is recorded (a route that reports none, nothing admitted)."""
    out = Path(manifest).parent
    found = []
    path = out / "served-models.json"
    if path.exists():
        entry = (json.loads(path.read_text(encoding="utf-8")).get("models") or {}).get(model) or {}
        fp_ = entry.get("system_fingerprint")
        found = [fp_] if isinstance(fp_, str) and fp_ else []
    try:
        admitted = read_admitted(out / ADMITTED_FILE).get(model, [])
    except (OSError, ValueError) as exc:
        print(f"warning: {out / ADMITTED_FILE} is ignored ({type(exc).__name__}: {exc})", file=sys.stderr)
        admitted = []
    return found + [f for f in admitted if f not in found]


def cmd_run(args: argparse.Namespace) -> int:
    try:
        rule = fault.load_rule(ROOT / args.rule, ROOT / args.pin)
        entry = rule["probe_sets"][args.model]
        raw = (ROOT / entry["file"]).read_bytes()
        questions = fp.read_questions(ROOT / json.loads(raw).get("questions_file", fp.QUESTIONS_FILE))
        pset = fp.load_probe_set(raw, entry["sha256"], args.model, questions)
        limits = rule["probe"]["thresholds"][args.model]
        expected_base = args.base_url
        if args.manifest is not None:
            routes = (json.loads(Path(args.manifest).read_text(encoding="utf-8")).get("fingerprint") or {}).get("routes")
            expected_base = (routes or {})[args.model]["base_url"]
            if not args.expect_fingerprint:  # the OUT's recorded fingerprint: a probe reply under another is a change
                args.expect_fingerprint = recorded_fingerprints(args.manifest, args.model)
        if args.out.exists():
            raise ValueError(f"{args.out} exists")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return REFUSED
    rp = load_runner()
    override = args.base_url_override
    if override is not None and rp.front_url_problem(override) is not None:
        print(f"refused: --base-url-override {rp.front_url_problem(override)}", file=sys.stderr)
        return REFUSED
    spec = rp.MODELS[args.model]
    provider = "anthropic" if "ANTHROPIC_API_KEY" in spec.key_envs else "openai-compatible"
    if expected_base and provider != "anthropic":
        os.environ["LLM_GATEWAY_BASE_URL"] = expected_base.rstrip("/")
    # the key of the URL the requests go to: a front listener gets the placeholder (it puts the backend's own)
    set_keys(override or expected_base, provider, Path(args.key_dir).expanduser())
    route = spec.factory()
    if override is not None:  # one backend through the front's listener for it (change-log entry 167)
        try:
            route = rp.FrontRoute(route, override)
        except (TypeError, ValueError) as exc:
            print(f"refused: --base-url-override: {exc}", file=sys.stderr)
            return REFUSED
    info = route.route_info()
    if route.model != pset["route_model"]:
        print(f"refused: the runner's route model {route.model} is not the probe set's {pset['route_model']}",
              file=sys.stderr)
        return REFUSED
    if expected_base and info.get("base_url", "").rstrip("/") != expected_base.rstrip("/"):
        print(f"refused: the route goes to {info.get('base_url')}, not {expected_base}", file=sys.stderr)
        return REFUSED
    try:
        built = [(req, *rebuild(req)) for req in pset["requests"]]
    except (OSError, ValueError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return REFUSED
    print(f"{args.model}: {len(built)} requests rebuilt and checked; route {info.get('base_url')} {route.model}"
          + (f", sent through {override}" if override is not None else ""))
    if args.dry:
        return 0
    from harness.agent.loop import same_model

    args.out.mkdir(parents=True)
    started = now()
    rows = []
    for req, system, tools in built:
        row = fp.send(req, system, tools, route, same_model)
        rows.append(row)
        with (args.out / "rows.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({k: row.get(k) for k in ("i", "arm", "t1_call", "coverage", "missing", "tries", "error",
                                                   "fingerprint", "wall_s")}), flush=True)
    verdict = fp.judge(rows, limits, args.expect_fingerprint)
    summary = {**verdict, "model": args.model, "started_at": started, "ended_at": now(),
               "rule_sha256": fault.rule_sha256(ROOT / args.rule), "probe_set_sha256": entry["sha256"],
               "route": {k: info.get(k) for k in ("provider", "base_url", "model", "settings")},
               "manifest": str(args.manifest) if args.manifest else None,
               "probe_url": info.get("front_url")}  # the backend listener the requests went to; None: the route's URL
    write_json(args.out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("model", "verdict", "reasons", "counts", "limits", "fingerprints",
                                              "fingerprint_change")}))
    return 0 if verdict["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="the probe sets from the dev records")
    b.add_argument("--runs-root", required=True)
    b.add_argument("--source", required=True, help="the dev OUT under --runs-root")
    b.add_argument("--cut", required=True, help="records started before this time (ISO with its zone)")
    b.add_argument("--seed", required=True)
    b.add_argument("--models", nargs="+", default=list(P_MODELS))
    b.add_argument("--out-dir", default=str(ROOT / PROBE_SETS))
    r = sub.add_parser("rule", help="the fault-rule file from the dev baselines and the probe sets")
    r.add_argument("--baselines", default="data/runs/infra-fault/dev-baselines.json")
    r.add_argument("--probe-sets", default=PROBE_SETS)
    r.add_argument("--stores", default=None, help="the store copies' section (JSON, under the repository)")
    r.add_argument("--control", default="data/runs/infra-fault/fault-control-glm.json",
                   help="the positive control (scripts/fault_probe.py control; under the repository)")
    r.add_argument("--models", nargs="+", default=list(P_MODELS))
    r.add_argument("--out", default=RULE_FILE)
    r.add_argument("--pin", default=None, help="write the rule's sha256 into this frozen-config pin")
    c = sub.add_parser("control", help="a fault window's turn-1 counts (the positive control)")
    c.add_argument("--runs-root", required=True)
    c.add_argument("--source", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--after", required=True, help="the fault's start (ISO with its zone)")
    c.add_argument("--out", required=True)
    u = sub.add_parser("run", help="probe one model's route")
    u.add_argument("--model", required=True)
    u.add_argument("--out", type=Path, required=True)
    where = u.add_mutually_exclusive_group()
    where.add_argument("--manifest", type=Path, default=None, help="the OUT's manifest.json: its route's base URL")
    where.add_argument("--base-url", default=None)
    u.add_argument("--base-url-override", default=None,
                   help="send the requests to this loopback URL (one backend's listener on the front, change-log entry "
                        "167); the route is still checked against --manifest or --base-url and recorded as it")
    u.add_argument("--expect-fingerprint", action="append", default=[], help="a recorded served fingerprint")
    u.add_argument("--rule", default=RULE_FILE)
    u.add_argument("--pin", default=PIN_FILE)
    u.add_argument("--key-dir", default="~/.config/www-2027")
    u.add_argument("--dry", action="store_true", help="rebuild and check every request; send nothing")
    args = ap.parse_args(argv)
    return {"build": cmd_build, "control": cmd_control, "rule": cmd_rule, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
