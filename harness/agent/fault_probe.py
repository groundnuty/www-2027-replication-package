"""The frozen fault probe (change-log entry 148 and its amendment; science, design log, 21 Oct 18:04 and 18:21): the
diagnostic that CONFIRMS an infrastructure fault after the monitor's trigger held a model (harness/agent/infra_fault.py).
It is independent of the test attempts: fixed dev requests, replayed byte for byte on the held model's route.

* The probe set, per model: PROBE_N fixed turn-1 requests of the model's own UniProt dev ``prompted`` records (arms
  curated and census), stored in the repository (data/runs/infra-fault/probe-sets/<model>.json, UniProt dev only) and
  bound by their sha256 in the fault-rule file. Selection (``select``), deterministic and blind to outcomes: the
  model's records in the source OUT that started before the cut (GLM's server fault began at 05:45Z on 21 Oct, so every
  model's set comes from before it), not an infrastructure failure, not a branch record, whose first transcript turn is
  the question and whose recorded turn 1 called a tool (a request the model answered healthily); one record per
  distinct request (the same question and arm render the same request); each arm in the order of
  sha256('<seed>|<run_id>'), the arms alternating curated, census, until PROBE_N.
* A request is rebuilt from the repository (the condition's system prompt with its stored ontology block, checked
  against the record's sha256; the tools at the model's result limit, checked against the record's tools sha256; the
  record's first user turn as recorded; its max_tokens) and sent once through the runner's own route for the model.
  A request that fails is sent again up to RETRIES times (section 4.3's re-runs of an infrastructure failure); a reply
  served by another model id is a failed request.
* Per reply: whether turn 1 called a tool; turn-1 question coverage (the question's words of 5+ letters, minus STOP,
  found in the reply's reasoning and visible text, lowercased: glm_health.py's reading, entry 124); the question-missing
  detector (MISSING, frozen, on the VISIBLE text only: on reasoning it fires on healthy replies); the served model and
  system fingerprint.
* The rule (``judge``; the amendments of 18:21 and 19:52): FAIL on any count criterion above its per-model threshold,
  on any question-missing reply (threshold 0), or on any request still failing after its retries. Each count
  criterion's threshold ``k`` (turn-1 no tool call; coverage below COVERAGE_OK) is the smallest count out of PROBE_N,
  and at least K_FLOOR, whose chance of being exceeded by a healthy model is at most FALSE_FAIL, at the add-one
  estimate (x + 1) / (n + 2) of the model's healthy dev counts x of n (``threshold``:
  exact binomial tail at that estimate, with x, n and p bound in the fault-rule file); a criterion whose ``k``
  would be PROBE_N can never fail and is not applicable (coverage for the models whose dev turn 1 restates nothing).
* Also confirmatory (judged by the monitor): a served fingerprint other than the model's recorded ones
  (``fingerprint_change``; a route that reports none, as Anthropic's, cannot show one).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any

PROBE_N = 20
RETRIES = 3  # section 4.3: an infrastructure failure is re-run up to 3 times
COVERAGE_OK = 0.9
FALSE_FAIL = Fraction(1, 100)  # a healthy model fails a count criterion with at most this chance (18:21 amendment)
K_FLOOR = 1  # no count criterion fails on a single event (19:52 ruling)
PROMPTED_ARMS = ("curated", "census")
PROBE_SET_SCHEMA = "fault-probe-set/2"
# the first user turn of every probe request is the frozen question's text (the agent loop's first turn); the set stores
# its sha256, not the text, and the text is read from the frozen question set at load time
QUESTIONS_FILE = "data/questions/uniprot-sp-2026_03/questions.jsonl"
STOP = frozenset(("what", "which", "that", "this", "with", "from", "have", "does", "were", "their", "there", "where",
                  "whose", "many", "much", "each", "list", "give", "these", "those", "than", "into", "about", "other",
                  "only", "also", "more", "most", "such", "some"))  # glm_health.py's stop list
# glm_health.py's detector (entry 124), unchanged, frozen by its sha256; on the visible text only (the 18:21 ruling)
MISSING_PATTERN = (r"no (actual |specific )?question|question (was|is|has been) (never|not|missing)|task is missing|"
                   r"truncat|not (been )?provided|never (actually )?provided|no question to answer|"
                   r"(question|prompt|message) (is |was )?(cut off|incomplete|empty|missing)|didn'?t (ask|provide)|"
                   r"there is no (user )?(question|query|task)|no (rdf )?(graph|knowledge base|knowledge graph|"
                   r"data source) (was |is )?(provided|available|given)")
MISSING = re.compile(MISSING_PATTERN, re.IGNORECASE)
MISSING_SHA256 = hashlib.sha256(MISSING_PATTERN.encode("utf-8")).hexdigest()
CRITERIA = ("no_call", "low_coverage")  # the count criteria, each with its per-model threshold
SELECTION = ("the model's records of the source OUT started before the cut, arm curated or census, not "
             "failed:infrastructure, not a branch record, first turn the question, recorded turn 1 with a tool call; "
             "one per distinct request; per arm by sha256('<seed>|<run_id>'), arms alternating curated, census")


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- the readings ---------------------------------------------------------------------------------


def coverage(question: str, reasoning: str, text: str) -> float:
    """The share of the question's words of 5+ letters (minus STOP) found in the reasoning and visible text."""
    both = ((reasoning or "") + (text or "")).lower()
    words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{4,}", question)} - STOP
    return sum(w in both for w in words) / len(words) if words else 1.0


def native_reasoning(native: Mapping[str, Any] | None) -> str:
    """A reply's reasoning text: the OpenAI wire's reasoning_content (or reasoning), or Anthropic's thinking blocks."""
    native = native or {}
    text = native.get("reasoning_content") or native.get("reasoning") or ""
    text = text if isinstance(text, str) else ""
    content = native.get("content")
    if isinstance(content, list):
        text += "".join(b.get("thinking") or "" for b in content if isinstance(b, dict) and b.get("type") == "thinking")
    return text


def question_missing(visible_text: str | None) -> bool:
    return bool(MISSING.search(visible_text or ""))


def turn1_reading(question: str, turn1: Mapping[str, Any]) -> dict[str, Any]:
    """A recorded or replayed turn 1 (transcript form: tool_calls, text, native) read as the rule reads it."""
    reasoning = native_reasoning(turn1.get("native"))
    text = turn1.get("text") or ""
    return {"t1_call": bool(turn1.get("tool_calls")), "coverage": coverage(question, reasoning, text),
            "missing": question_missing(text)}


# --- the thresholds -------------------------------------------------------------------------------


def tail(n: int, p: Fraction, k: int) -> Fraction:
    """P(X > k) for X ~ Binomial(n, p), exactly."""
    return sum((comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1, n + 1)), Fraction(0))


def add_one(events: int, denominator: int) -> Fraction:
    """The add-one estimate of a dev rate, (x + 1) / (n + 2) (design ruling, design log, 21 Oct 19:52): a dev count of 0
    is no evidence of a rate of 0."""
    if denominator <= 0 or not 0 <= events <= denominator:
        raise ValueError(f"dev counts {events}/{denominator}: no rate")
    return Fraction(events + 1, denominator + 2)


def threshold(events: int, denominator: int, n: int = PROBE_N, false_fail: Fraction = FALSE_FAIL) -> int | None:
    """The smallest k >= K_FLOOR with P(Binomial(n, add_one(events, denominator)) > k) <= false_fail (the ruling of
    19:52); None (not applicable) when that k is n, a criterion that can never fail. A criterion with no dev denominator
    is refused (ValueError)."""
    p = add_one(events, denominator)
    k = max(K_FLOOR, next(k for k in range(n + 1) if tail(n, p, k) <= false_fail))
    return None if k >= n else k


def dev_counts(readings: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """The count criteria's dev counts over recorded turn-1 readings (turn1_reading)."""
    rows = list(readings)
    return {"n": len(rows), "no_call": sum(not r["t1_call"] for r in rows),
            "low_coverage": sum(r["coverage"] < COVERAGE_OK for r in rows),
            "missing": sum(bool(r["missing"]) for r in rows)}


def thresholds(counts: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    """Each count criterion's threshold from the dev counts, with its numerator x, denominator n and estimate p."""
    out = {}
    for c in CRITERIA:
        p = add_one(counts[c], counts["n"])
        out[c] = {"k": threshold(counts[c], counts["n"]), "dev_events": counts[c], "dev_n": counts["n"],
                  "p": f"{counts[c] + 1}/{counts['n'] + 2}", "p_float": round(float(p), 6)}
    return out


# --- the probe set ---------------------------------------------------------------------------------


def eligible(rec: dict, name: str | None, arm: str | None, cut: datetime, *, before: bool = True) -> bool:
    """A record the probe set may come from (module docstring), whatever its outcome: started before ``cut``; with
    before=False, at or after it (a fault window's records, the positive control)."""
    if name is None or arm not in PROMPTED_ARMS or rec.get("checker") is not None:
        return False
    if (rec.get("outcome") or {}).get("failure_type") == "infrastructure":
        return False
    if (datetime.fromisoformat(rec["started_at"]) >= cut) is before:
        return False
    tr = rec.get("transcript") or []
    return bool(tr) and tr[0].get("role") == "user" and tr[0].get("kind") == "question"


def turn1(rec: dict) -> dict:
    return next((t for t in rec.get("transcript") or [] if t.get("role") == "assistant"), {})


def descriptor(rec: dict) -> dict[str, Any]:
    """A request of the probe set from its record: what rebuilds it from the repository, and the hashes it must
    rebuild to."""
    cfg = rec["agent_config"]
    first = rec["transcript"][0]
    per_turn = (rec.get("tokens") or {}).get("per_turn") or [{}]
    d = {"ref_run_id": rec["run_id"], "question_id": rec["question_id"], "condition": cfg["condition"],
         "ontology_block_sha256": cfg.get("ontology_block_sha256"), "row_cap": rec["store"]["row_cap"],
         "sparql_timeout_s": rec["store"]["sparql_timeout_s"], "max_result_chars": cfg["max_result_chars"],
         "max_tokens": cfg["max_tokens"], "max_wall_s": cfg["max_wall_s"], "first_turn": first,
         "system_sha256": rec["prompt"]["sha256"], "tools_sha256": rec["prompt"]["tools_sha256"],
         "rec_input_tokens": per_turn[0].get("input_tokens")}
    d["request_sha256"] = request_sha256(d)
    return d


def request_sha256(d: Mapping[str, Any]) -> str:
    """The request as sent, by what fixes its bytes: the system prompt's and tools' sha256, the first turn, max_tokens."""
    return sha256_bytes(canonical({k: d[k] for k in ("system_sha256", "tools_sha256", "first_turn", "max_tokens")}))


def select(candidates: Iterable[tuple[str, dict]], seed: str, n: int = PROBE_N) -> list[dict]:
    """``candidates``: (arm, record) of eligible records; the probe set's requests (module docstring). Refuses
    (ValueError) with fewer than ``n`` distinct healthy requests."""
    by_arm: dict[str, list[tuple[str, dict]]] = {a: [] for a in PROMPTED_ARMS}
    for arm, rec in candidates:
        if turn1(rec).get("tool_calls"):
            by_arm[arm].append((sha256_bytes(f"{seed}|{rec['run_id']}".encode()), rec))
    for v in by_arm.values():
        v.sort(key=lambda x: x[0])
    picked, seen, cursor = [], set(), dict.fromkeys(PROMPTED_ARMS, 0)
    while len(picked) < n and any(cursor[a] < len(by_arm[a]) for a in PROMPTED_ARMS):
        for arm in PROMPTED_ARMS:
            while cursor[arm] < len(by_arm[arm]) and len(picked) < n:
                d = descriptor(by_arm[arm][cursor[arm]][1])
                cursor[arm] += 1
                if d["request_sha256"] not in seen:
                    seen.add(d["request_sha256"])
                    picked.append({"arm": arm, **d})
                    break
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} distinct healthy requests, {n} needed")
    return picked


def stored(request: Mapping[str, Any]) -> dict:
    """A request as the set stores it: its first turn by the sha256 of its text (the frozen question's), not the text."""
    first = request["first_turn"]
    if set(first) != {"role", "kind", "text"} or (first["role"], first["kind"]) != ("user", "question"):
        raise ValueError(f"request {request.get('ref_run_id')}: its first turn is not a plain question turn")
    out = {k: v for k, v in request.items() if k != "first_turn"}
    out["question_sha256"] = sha256_bytes(first["text"].encode("utf-8"))
    return out


def probe_set_doc(model: str, route_model: str, source: dict, requests: list[dict], dev: dict[str, int]) -> dict:
    return {"schema": PROBE_SET_SCHEMA, "model": model, "route_model": route_model, "source": source,
            "selection": SELECTION, "dev_turn1": dev, "questions_file": QUESTIONS_FILE,
            "requests": [{"i": i, **stored(r)} for i, r in enumerate(requests, 1)]}


def load_probe_set(raw: bytes, expected_sha256: str, model: str, questions: Mapping[str, str]) -> dict:
    """A probe set from its bytes, refused unless they hash to the bound value and it is the model's; each request's
    first turn rebuilt from ``questions`` (question id -> text, the frozen question set), refused unless the text has
    the stored sha256 and the whole request its request_sha256."""
    value = sha256_bytes(raw)
    if value != expected_sha256:
        raise ValueError(f"probe set sha256 {value} is not the bound {expected_sha256}")
    doc = json.loads(raw)
    if doc.get("schema") != PROBE_SET_SCHEMA or doc.get("model") != model:
        raise ValueError(f"not a {PROBE_SET_SCHEMA} of {model}")
    if len(doc.get("requests") or []) != PROBE_N:
        raise ValueError(f"{len(doc.get('requests') or [])} requests, {PROBE_N} expected")
    for r in doc["requests"]:
        text = questions.get(r.get("question_id"))
        if text is None or sha256_bytes(text.encode("utf-8")) != r.get("question_sha256"):
            raise ValueError(f"request {r.get('i')}: question {r.get('question_id')} is not the stored text")
        r["first_turn"] = {"role": "user", "kind": "question", "text": text}
        if request_sha256(r) != r.get("request_sha256"):
            raise ValueError(f"request {r.get('i')}: its request_sha256 does not match its content")
    return doc


def read_questions(path: Path) -> dict[str, str]:
    """Question id -> question text of a question set (JSON lines)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            out[q["id"]] = q["question"]
    return out


# --- sending and judging ---------------------------------------------------------------------------


def send(request: Mapping[str, Any], system: str, tools: Any, route: Any,
         same_model: Callable[[str, str | None], bool | None], retries: int = RETRIES) -> dict[str, Any]:
    """One request through ``route`` (its ``step``), re-sent up to ``retries`` times while it fails; the row the rule
    reads (counts and ids only: no text)."""
    row: dict[str, Any] = {"i": request.get("i"), "arm": request.get("arm"), "question_id": request["question_id"],
                           "ref_run_id": request["ref_run_id"], "errors": []}
    for attempt in range(1 + retries):
        try:
            step = route.step(system, [dict(request["first_turn"])], tools, max_tokens=request["max_tokens"],
                              time_left_s=float(request["max_wall_s"]))
        except Exception as exc:  # noqa: BLE001 - any failure is the infrastructure's until its retries are spent
            row["errors"].append(f"{type(exc).__name__}: {exc}"[:300])
            continue
        if same_model(route.model, step.model) is False:
            row["errors"].append(f"served by {step.model!r}, not {route.model!r}")
            continue
        reading = turn1_reading(request["first_turn"].get("text") or "",
                                {"tool_calls": step.tool_calls, "text": step.text, "native": step.native})
        row.update(reading, tries=attempt + 1, served_model=step.model, fingerprint=step.system_fingerprint,
                   tool_calls=[c.get("name") for c in step.tool_calls or []], stop_reason=step.stop_reason,
                   input_tokens=(step.usage or {}).get("input_tokens"), wall_s=round(step.wall_s or 0.0, 3))
        row["coverage"] = round(row["coverage"], 4)
        return row
    row.update(tries=1 + retries, error=row["errors"][-1])
    return row


def judge(rows: list[dict], limits: Mapping[str, Mapping[str, Any]], expected_fingerprints: Iterable[str] = (),
          n: int = PROBE_N) -> dict[str, Any]:
    """The probe's verdict on its rows (module docstring). ``limits``: criterion -> {"k": int or None}; the
    fingerprint change is reported beside the verdict (confirmatory on its own, judged by the monitor)."""
    replies = [r for r in rows if "error" not in r]
    counts = {"n": len(rows), "replies": len(replies), "errors": len(rows) - len(replies),
              "no_call": sum(not r["t1_call"] for r in replies),
              "low_coverage": sum(r["coverage"] < COVERAGE_OK for r in replies),
              "missing": sum(bool(r["missing"]) for r in replies)}
    reasons = []
    if len(rows) != n:
        reasons.append(f"{len(rows)} requests sent, {n} expected")
    if counts["errors"]:
        reasons.append(f"{counts['errors']} request(s) still failing after {RETRIES} retries")
    for c in CRITERIA:
        k = (limits.get(c) or {}).get("k")
        if k is not None and counts[c] > k:
            reasons.append(f"{c} {counts[c]} of {n} > {k}")
    if counts["missing"]:
        reasons.append(f"{counts['missing']} question-missing repl{'y' if counts['missing'] == 1 else 'ies'}")
    expected = set(expected_fingerprints)
    seen = sorted({str(r["fingerprint"]) for r in replies if r.get("fingerprint")})
    change = None if not expected else any(f not in expected for f in seen)
    return {"pass": not reasons, "verdict": "pass" if not reasons else "fail", "reasons": reasons, "counts": counts,
            "limits": {c: (limits.get(c) or {}).get("k") for c in CRITERIA}, "fingerprints": seen,
            "expected_fingerprints": sorted(expected), "fingerprint_change": change}
