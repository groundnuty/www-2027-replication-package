#!/usr/bin/env python3
"""Entry 167: admit one endpoint (a backend) of a model into every OUT that addresses it (DESIGN.md B2), in seconds.

    python3 -I admit.py admit --model gemma-4-31b --base-url http://HOST:PORT/v1 --backend ID
        (--front-config FRONT.json [--out OUT ...] [--no-enable] | --out OUT [--out ...] --no-enable)
        [--key-file FILE] [--declared-revision SHA] [--upstream-url URL]
        [--reference-base-url URL [--reference-upstream-url URL] [--reference-key-file FILE]] [--code DEPLOY]
    python3 -I admit.py shape --model gemma-4-31b --base-url URL [--key-file FILE] [--repeat N] [--code DEPLOY]

The OUTs: every OUT the front's config (builder A's e167-front-config/1) lists for the model, the inventory its
enable gate reads, plus any --out. The enable is the front's own admin request on the config's admin_socket
({"cmd": "enable", "backend", "system_fingerprint", "model"}; its gate reads every OUT's file again), and PASS only
when the front's list then shows the backend enabled with this fingerprint at this identity URL. An admission that
enables needs --front-config; --out alone is allowed only with --no-enable (status ADMITTED, never PASS).

Every check runs before anything is written; a failure prints its reason, exits 1 and touches no file.
1. GET <server>/version is the model's vLLM version (model-references.json; an OUT's admitted file must agree).
2. GET <base>/models has an entry whose id is the route's served name, with max_model_len >= the longest dev prompt
   + the output cap (the largest of the runner's and every OUT's pinned cap). The longest dev prompt is the larger of
   longest-prompts.json's (the dev sources that may be in git) and the private file's (--longest-private: every dev
   source, mode 600 on the cluster host, never in git), which must be the file longest-prompts-private.sha256 pins; a
   missing or other private file refuses. A private value larger than the committed one is never written or printed:
   the evidence then names the private file's sha256 instead. Its root gives
   the checkpoint revision when it is a Hugging Face snapshot path (.../snapshots/<sha>), else --declared-revision does
   (recorded as declared); either must be the model's revision.
3. The smoke: the identity check's request (sitea_identity.py's agent-shaped smoke, built by the runner's own route
   for the model from the code at --code) with max_tokens 1, sent once with no retry; for Gemma that is
   sitec-gemma-qual/body-1tok.json, the body of the SiteA and SiteC gates. (freeze_day.py's "Say OK." probe gives
   other token counts.) Its usage.prompt_tokens must be the model's reference. Its one system_fingerprint (non-null,
   not mixed, vLLM's form, of that version) is the candidate's, and every chunk must name the served model. A streamed
   route's reply must also carry [DONE], a finish reason and the final usage-only chunk, as the runner's _read_stream
   requires.
4. One tool-call request: the same body at the route's output cap, so the runner's tools and route settings. Its reply
   is assembled by the frozen StreamAssembler, or taken as sent for a non-streamed route. Every call needs an id, one
   of the runner's tool names, and arguments that parse as a JSON object (providers.parse_tool_arguments). The
   fingerprint must be the smoke's, and the route's stream guards must not fire. Its shape is the key sets of the
   message, choice 0, the calls and their functions. It must equal every reference: the shapes recorded with the
   model's earlier admissions in the OUTs' files, and the live reply of --reference-base-url. That reference passes
   checks 1-4 itself and must BE an admitted backend: its fingerprint at its URL (--reference-upstream-url behind a
   forward) is an entry in every OUT, or the OUT's incumbent at its pinned route URL. With no reference at all, only a
   backend whose fingerprint is every OUT's baseline passes; it is compared with itself, and recorded so.
5. Every OUT must address the model: its manifest.json pins the model's route with the served name, the same route
   settings, an output cap, and the tools this code builds for the tool-result cap it pins (tools_sha256). It needs a
   baseline, read from served-models.json models.<m>.system_fingerprint (the only field read there). Its admitted
   file must be absent or well formed, because the runner ignores a malformed file whole.

6. A JOIN needs a behaviour verdict (science, design log, 11:54Z): a candidate that is not every OUT's incumbent (its
   baseline fingerprint at its pinned route URL) joins an OUT already running on another endpoint, and must first pass
   a short dev malfunction check (malfunction_check.sh). --behaviour-verdict names its verdict file
   (e167-behaviour-verdict/1): "verdict" "pass", for this "model" (the runner's name), "backend" (this id) and
   "backend_url_sha256" (this backend's identity URL, the upstream's behind a forward), "at" (UTC, "...Z") at most 24 h
   old and not in the future, and "inputs" (name -> sha256 of each input). A given verdict is checked before any request;
   a join without one is refused. Recorded in the candidate's evidence: the file's sha256, its schema, verdict, at and
   inputs. The incumbent needs none, and its evidence says so.

Pass: every OUT is locked (.admitted-fingerprints.json.lock) and its file re-read; a reply shape recorded there since
the checks is compared too. The OUT's baseline is appended first when it is not admitted yet; one entry covers both
when the candidate IS the baseline. The candidate is appended unless the same (fingerprint, backend id, backend URL)
is already there, the triple the front's gate looks for. Every OUT's new file is written to a temporary in it and
fsynced first (a failure there changes no OUT, exit 1); then each is renamed over its file and the directory fsynced;
then every OUT holding the entry is read back: the old entries a prefix of the new, the new ones in place. Only then
is the front asked to enable. A failed rename, directory fsync (the entry in place, its durability uncertain) or
read-back is an INCOMPLETE ADMISSION (exit 3): the OUTs that hold the entry keep it (entries are never removed: the
runner's replay needs them), the backend is not enabled, and running the same admission again completes it. A
refused or unconfirmed enable is exit 3 too. With --no-enable the status is ADMITTED, never PASS.

Stored: the backend URL only as the sha256 of its canonical form (scheme://host[:port]/path), the upstream's for a
backend reached through a forward (--upstream-url). The evidence holds counts, field names, fingerprints and hashes,
never reply text or call arguments. The key (--key-file) is read into this process only and sent only in the
Authorization header. With no key file, a fixed non-secret placeholder is sent. The dedicated Gateway key is refused,
through a link too.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import http.client
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

HERE = Path(__file__).resolve().parent
# the private longest-prompt file (every dev source; never in git) and the committed pin of its sha256
LONGEST_PRIVATE = Path("~/project/private/a15/e167/longest-prompts-private.json")
LONGEST_PRIVATE_PIN = HERE / "longest-prompts-private.sha256"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PIN_RE = re.compile(rb"([0-9a-f]{64})(?:  longest-prompts-private\.json)?\n?")  # sha256sum's line, or the bare digest
SCHEMA = "admitted-fingerprints/1"
ADMITTED_FILE = "admitted-fingerprints.json"
LOCK_FILE = ".admitted-fingerprints.json.lock"  # never the runner's OUT/.lock
SERVED_MODELS_FILE = "served-models.json"
MANIFEST_FILE = "manifest.json"
RULING = "entry 167"
MODEL_FIELDS = ("served_model", "vllm_version", "revision")
MIXED = " | "  # providers.MIXED_FINGERPRINTS: a reply spliced from servers that differ
PLACEHOLDER_KEY = "sitea-local-no-key"  # sitea_identity.py's: our vLLM servers run without --api-key
GATEWAY_KEY_FILE_NAME = "gateway_api_key_dedicated"
SMOKE_USER = "Run one SPARQL query that counts the triples in the default graph. Use the sparql_query tool."
SMOKE_GRAPH = "uniprot-sp-2026_03"  # sitea_identity.py's: the smoke's tool-result cap is the model's UniProt cap
FINGERPRINT_RE = re.compile(r"^vllm-(?P<version>.+?)(?:-tp(?P<tp>\d+))?(?P<ep>-ep)?-(?P<hash>[0-9a-f]{6,})$")
SNAPSHOT_RE = re.compile(r"/snapshots/([0-9a-f]{40})/*$")
BACKEND_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")  # the front's backend ids (front.py ID_RE)
SHAPE_KEYS = ("message_fields", "choice_fields", "tool_call_fields", "function_fields")
SELF = "self (the OUT's baseline)"
VERDICT_SCHEMA = "e167-behaviour-verdict/1"
VERDICT_MAX_AGE_S = 24 * 3600.0
VERDICT_AT_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
INCUMBENT_WHY = ("the incumbent: every OUT's baseline fingerprint at its pinned route URL (not a join; science, design log, "
                 "11:54Z)")
LOCK_WAIT_S = 30.0
MAX_BODY = 64 << 20


class Failed(Exception):
    """A check that did not pass; its message is the reason."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_url(url: str) -> str:
    """scheme://host[:port]/path: no credentials, query or fragment, lower-case scheme and host, no trailing slash."""
    p = urlsplit(url.strip())
    if p.scheme.lower() not in ("http", "https") or not p.hostname:
        raise Failed(f"not an http(s) URL with a host: {p.scheme or '?'}://...")
    host = p.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    if p.port:
        host = f"{host}:{p.port}"
    return urlunsplit((p.scheme.lower(), host, p.path.rstrip("/"), "", ""))


def url_sha256(url: str) -> str:
    return sha256_text(canonical_url(url))


def server_root(base_url: str) -> str:
    p = urlsplit(canonical_url(base_url))
    return f"{p.scheme}://{p.netloc}"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def loads_object(raw: bytes, what: str) -> dict:
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise Failed(f"{what}: the reply is not JSON") from None
    if not isinstance(obj, dict):
        raise Failed(f"{what}: the reply is not a JSON object")
    return obj


def read_key(path: Path | None) -> str:
    if path is None:
        return PLACEHOLDER_KEY
    if GATEWAY_KEY_FILE_NAME in (path.name, path.expanduser().resolve().name):  # a link to it too
        raise SystemExit("refusing: the dedicated Gateway key is never sent to a candidate endpoint")
    key = path.expanduser().read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit("refusing: the key file is empty")
    return key


# --- the code the requests are built from --------------------------------------------------------------------------

@dataclass
class Code:
    root: Path
    runner: Any
    providers: Any
    prompts: Any
    tools: Any
    commit: str | None
    loop: Any = None


def load_code(root: Path) -> Code:
    """The deploy's runner table and harness (its own files: the requests are the runner's). The routes are only asked
    for their request bodies, so no key belongs in their environment."""
    root = root.resolve()
    for name in ("LLM_GATEWAY_API_KEY", "CENTREB_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.pop(name, None)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import harness.agent.loop as loop
    import harness.agent.prompts as prompts
    import harness.agent.providers as providers
    import harness.agent.tools as tools

    for mod in (loop, prompts, providers, tools):
        if root not in Path(mod.__file__).resolve().parents:
            raise SystemExit(f"refusing: {mod.__name__} was imported from {mod.__file__}, not from {root}")
    spec = importlib.util.spec_from_file_location("run_pilot_deploy", root / "scripts" / "run_pilot.py")
    runner = importlib.util.module_from_spec(spec)
    sys.modules["run_pilot_deploy"] = runner  # its dataclasses look their module up while it loads
    spec.loader.exec_module(runner)
    commit = None
    if (root / "COMMIT").is_file():
        commit = (root / "COMMIT").read_text(encoding="utf-8").strip() or None
    else:
        done = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=False)
        commit = done.stdout.strip() or None if done.returncode == 0 else None
    return Code(root, runner, providers, prompts, tools, commit, loop)


@dataclass(frozen=True)
class Requests:
    smoke: dict
    tool_call: dict
    served_model: str
    output_cap: int
    route_settings: dict
    tool_names: frozenset
    streamed: bool
    include_usage: bool
    empty_tool_calls: bool
    guard_token_gap: bool
    system_sha256: str
    smoke_body_sha256: str
    request_timeout_s: float


def build_requests(code: Code, model: str) -> Requests:
    """The identity smoke's body for ``model``, as sitea_identity.smoke builds it from the runner's route, at the
    route's output cap (the tool-call request) and with max_tokens 1 (the smoke)."""
    spec = code.runner.MODELS.get(model)
    if spec is None:
        raise Failed(f"the runner has no model {model!r}")
    route = spec.factory()
    info = route.route_info()
    if info.get("provider") != "openai-compatible":
        raise Failed(f"{model} is not served through an OpenAI-compatible route (provider {info.get('provider')!r})")
    system = code.prompts.get_condition("none").render(row_cap=500, timeout_s=60.0, ontology_block=None)
    tools = code.tools.agent_tools(spec.result_chars_for(SMOKE_GRAPH))
    transcript = [{"role": "user", "turn": 0, "text": SMOKE_USER}]
    body = route.request(system, transcript, tools, spec.max_tokens)
    if "extra_body" in body:  # the SDK merges it into the body; a raw request must do the same
        body.update(body.pop("extra_body") or {})
    smoke = {**body, "max_tokens": 1}
    settings = json.loads(json.dumps(info["settings"]))
    return Requests(smoke=smoke, tool_call=body, served_model=route.model, output_cap=spec.max_tokens,
                    route_settings=settings, tool_names=frozenset(t.name for t in tools),
                    streamed=bool(body.get("stream")),
                    include_usage=bool((body.get("stream_options") or {}).get("include_usage")),
                    empty_tool_calls=bool(getattr(route, "stream_empty_tool_calls", False)),
                    guard_token_gap=bool(getattr(route, "guard_token_gap", False)),
                    system_sha256=sha256_text(system), smoke_body_sha256=sha256_text(json.dumps(smoke, sort_keys=True)),
                    request_timeout_s=float(settings.get("request_timeout_s") or 600.0))


# --- talking to a backend -------------------------------------------------------------------------------------------

class Transport(Protocol):
    def get(self, url: str) -> tuple[int | None, bytes]: ...

    def post(self, url: str, body: dict) -> tuple[int | None, bytes]: ...


class HttpTransport:
    """One request per call (http.client, no retry, no redirect); the key only in the Authorization header."""

    def __init__(self, key: str, timeout_s: float):
        self.__key = key
        self.timeout_s = timeout_s

    def _send(self, method: str, url: str, data: bytes | None, timeout: float) -> tuple[int | None, bytes]:
        p = urlsplit(url)
        cls = http.client.HTTPSConnection if p.scheme == "https" else http.client.HTTPConnection
        conn = cls(p.hostname, p.port, timeout=timeout)
        headers = {"Authorization": f"Bearer {self.__key}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        try:
            conn.request(method, (p.path or "/") + (f"?{p.query}" if p.query else ""), body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read(MAX_BODY + 1)
            if len(raw) > MAX_BODY:
                raise Failed(f"{method} {p.path}: the reply is over {MAX_BODY} bytes")
            return resp.status, raw
        except (OSError, http.client.HTTPException) as exc:
            raise Failed(f"{method} {p.path}: {type(exc).__name__}: {exc}") from None
        finally:
            conn.close()

    def get(self, url: str) -> tuple[int | None, bytes]:
        return self._send("GET", url, None, min(30.0, self.timeout_s))

    def post(self, url: str, body: dict) -> tuple[int | None, bytes]:
        return self._send("POST", url, json.dumps(body).encode("utf-8"), self.timeout_s)


@dataclass
class Reply:
    assembled: dict
    fingerprint: str
    usage: dict
    headless_calls: list = field(default_factory=list)
    token_gaps: list = field(default_factory=list)


def read_reply(code: Code, req: Requests, status: int | None, raw: bytes, what: str) -> Reply:
    """A chat reply as the runner would accept it, with exactly one fingerprint and the served model named."""
    if status != 200:
        raise Failed(f"{what}: HTTP {status}: {raw[:200].decode('utf-8', errors='replace')}")
    p = code.providers
    headless: list = []
    gaps: list = []
    if not req.streamed:
        assembled = loads_object(raw, what)
        fingerprints = [assembled.get("system_fingerprint")]
        models = [assembled.get("model")] if "model" in assembled else []
    else:
        try:
            lines = re.split(r"\r\n|\r|\n", raw.decode("utf-8"))
        except UnicodeDecodeError:
            raise Failed(f"{what}: the stream is not UTF-8") from None
        asm = p.StreamAssembler(empty_tool_calls=req.empty_tool_calls)
        counter = p.TokenGapCounter() if req.guard_token_gap else None
        done, models = False, []
        for data in p.sse_data(iter(lines)):
            if data == "[DONE]":
                done = True
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                raise Failed(f"{what}: the stream carried an event that is not JSON") from None
            if isinstance(chunk, dict) and chunk.get("error"):
                raise Failed(f"{what}: the server sent an error event after its 200")
            if isinstance(chunk, dict):
                asm.add(chunk)
                if counter is not None:
                    counter.add(chunk)
                if "model" in chunk:
                    models.append(chunk.get("model"))
        assembled = asm.result()
        if not done:
            raise Failed(f"{what}: the stream ended without [DONE]")
        if assembled is None:
            raise Failed(f"{what}: choice 0 never finished (no finish reason)")
        if req.include_usage and not asm.final_usage:
            raise Failed(f"{what}: no final usage-only chunk, which the route asks for (stream_options include_usage)")
        fingerprints = list(asm.fingerprints)
        headless = asm.calls_without_identity()
        gaps = counter.gaps if counter is not None else []
    named = [f for f in fingerprints if f not in (None, "")]
    if len(set(map(str, named))) != 1:
        raise Failed(f"{what}: " + ("no system_fingerprint" if not named else
                                    f"several system fingerprints in one reply: {', '.join(map(str, named))}"))
    fp = named[0]
    if not isinstance(fp, str) or MIXED in fp:
        raise Failed(f"{what}: the system fingerprint {fp!r} is not one server's")
    if not models or any(m != req.served_model for m in models):
        raise Failed(f"{what}: the reply names model {sorted({str(m) for m in models})}, not the served name "
                     f"{req.served_model!r}")
    usage = assembled.get("usage")
    if not isinstance(usage, dict) or count(usage.get("prompt_tokens")) is None:
        raise Failed(f"{what}: the reply has no usage.prompt_tokens")
    return Reply(assembled, fp, usage, headless, gaps)


def reply_shape(assembled: dict) -> dict:
    """The key sets the records keep of a reply: the message, choice 0, the calls and their functions."""
    choice = assembled["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls") or []
    return {"message_fields": sorted(message), "choice_fields": sorted(choice),
            "tool_call_fields": sorted({k for c in calls for k in c}),
            "function_fields": sorted({k for c in calls for k in (c.get("function") or {})})}


def shape_difference(got: dict, want: dict) -> str | None:
    parts = []
    for k in SHAPE_KEYS:
        g, w = set(got.get(k) or ()), set(want.get(k) or ())
        if g != w:
            parts.append(f"{k} " + ", ".join([*(f"+{x}" for x in sorted(g - w)), *(f"-{x}" for x in sorted(w - g))]))
    return "; ".join(parts) or None


def valid_shape(shape: Any) -> bool:
    return isinstance(shape, dict) and all(
        isinstance(shape.get(k), list) and all(isinstance(x, str) for x in shape[k]) for k in SHAPE_KEYS)


def tool_call(code: Code, req: Requests, t: Transport, base: str, what: str) -> tuple[Reply, dict, list[str]]:
    """The tool-call request: (the reply, its shape, the tool names it called), every call checked."""
    status, raw = t.post(base + "/chat/completions", req.tool_call)
    reply = read_reply(code, req, status, raw, what)
    choices = reply.assembled.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise Failed(f"{what}: the reply has no choice 0 message")
    calls = choice["message"].get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise Failed(f"{what}: no tool call (finish reason {choice.get('finish_reason')!r})")
    names = []
    for i, call in enumerate(calls):
        fn = call.get("function") if isinstance(call, dict) else None
        if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
            raise Failed(f"{what}: call {i} has no id")
        name = fn.get("name") if isinstance(fn, dict) else None
        if name not in req.tool_names:
            raise Failed(f"{what}: call {i} names {name!r}, not one of the runner's tools")
        args = fn.get("arguments")
        if not isinstance(args, str) or not args.strip():
            raise Failed(f"{what}: call {i} has no JSON arguments")
        parsed, why = code.providers.parse_tool_arguments(args)
        if parsed is None:
            raise Failed(f"{what}: call {i}'s arguments: {why}")
        names.append(name)
    if reply.headless_calls:
        raise Failed(f"{what}: the call-identity guard fires: {reply.headless_calls}")
    if reply.token_gaps:
        raise Failed(f"{what}: the token-gap guard fires: {reply.token_gaps[0]}")
    return reply, reply_shape(reply.assembled), names


# --- the OUTs -------------------------------------------------------------------------------------------------------

def malformed(doc: Any) -> str | None:
    """Why the runner would ignore this admitted file whole (builder C's rule), or None."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return f"its schema is not {SCHEMA}"
    models = doc.get("models")
    if not isinstance(models, dict):
        return "models is not an object"
    for m, entry in models.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("admitted"), list):
            return f"{m}: admitted is not a list"
        for i, e in enumerate(entry["admitted"]):
            fp = e.get("system_fingerprint") if isinstance(e, dict) else None
            if not isinstance(fp, str) or not fp or MIXED in fp:
                return f"{m}: entry {i} has no single system_fingerprint"
    return None


def read_admitted(out: Path, model: str, reference: dict) -> dict | None:
    """The OUT's admitted file (None when absent), refused when the runner would ignore it or when it names another
    identity for the model than the reference."""
    path = out / ADMITTED_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        doc = json.loads(text)
    except ValueError:
        doc, why = None, "not JSON"
    else:
        why = malformed(doc)
    if why is not None:
        raise Failed(f"{out}: {ADMITTED_FILE} is malformed ({why}): the runner ignores it whole, so nothing is "
                     f"written into it")
    entry = doc["models"].get(model)
    if entry is not None:
        for k in (*MODEL_FIELDS, "ruling"):
            want = RULING if k == "ruling" else reference[k]
            if entry.get(k) != want:
                raise Failed(f"{out}: {ADMITTED_FILE} names {k} {entry.get(k)!r} for {model}, not {want!r}")
    return doc


def read_baseline(out: Path, model: str) -> str:
    """The OUT's baseline fingerprint for the model: served-models.json models.<m>.system_fingerprint, nothing else."""
    try:
        data = json.loads((out / SERVED_MODELS_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise Failed(f"{out}: no {SERVED_MODELS_FILE}, so no baseline for {model} yet: admit after its first "
                     f"fingerprinted turn") from None
    except ValueError:
        raise Failed(f"{out}: {SERVED_MODELS_FILE} is not JSON") from None
    models = data.get("models") if isinstance(data, dict) else None
    entry = models.get(model) if isinstance(models, dict) else None
    fp = entry.get("system_fingerprint") if isinstance(entry, dict) else None
    del data, models, entry
    if not isinstance(fp, str) or not fp:
        raise Failed(f"{out}: {SERVED_MODELS_FILE} has no baseline fingerprint for {model} (no fingerprinted turn "
                     f"yet): admit after its first one")
    if MIXED in fp:
        raise Failed(f"{out}: the baseline fingerprint for {model} is mixed ({fp!r})")
    return fp


@dataclass
class Out:
    path: Path
    baseline: str
    pinned_url: str | None
    pinned_cap: int | None
    doc: dict | None

    def rows(self, model: str) -> list[dict]:
        return list(((self.doc or {}).get("models") or {}).get(model, {}).get("admitted", []))

    def admitted_backends(self, model: str) -> set[tuple[str, str | None]]:
        """(fingerprint, backend URL sha256) of every admitted backend, the incumbent at the OUT's pinned URL too."""
        pairs = {(r["system_fingerprint"], r.get("backend_url_sha256")) for r in self.rows(model)}
        if self.pinned_url:
            pairs.add((self.baseline, url_sha256(self.pinned_url)))
        return pairs


def read_out(code: Code, out: Path, model: str, req: Requests, reference: dict) -> Out:
    if not out.is_dir():
        raise Failed(f"{out}: not a directory")
    try:
        manifest = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise Failed(f"{out}: no {MANIFEST_FILE}, so which models it addresses is unknown") from None
    except ValueError:
        raise Failed(f"{out}: {MANIFEST_FILE} is not JSON") from None
    routes = (manifest.get("fingerprint") or {}).get("routes") if isinstance(manifest, dict) else None
    route = routes.get(model) if isinstance(routes, dict) else None
    del manifest, routes
    if not isinstance(route, dict):
        raise Failed(f"{out}: does not address {model} (its manifest pins no route for it)")
    if route.get("model") != req.served_model:
        raise Failed(f"{out}: pins route model {route.get('model')!r} for {model}, not {req.served_model!r}")
    if route.get("settings") != req.route_settings:
        pinned = route.get("settings") if isinstance(route.get("settings"), dict) else {}
        keys = sorted(k for k in set(pinned) | set(req.route_settings) if pinned.get(k) != req.route_settings.get(k))
        raise Failed(f"{out}: pins other route settings for {model} than the code at --code ({', '.join(keys)}): the "
                     f"requests would not be that OUT's")
    agent = route.get("agent_config") if isinstance(route.get("agent_config"), dict) else {}
    cap = count(agent.get("max_tokens"))
    if cap is None or cap < 1:
        raise Failed(f"{out}: pins no output cap (agent_config.max_tokens) for {model}")
    # the OUT's tools must be the ones this code builds for the tool-result cap it pins (the requests use the
    # identity smoke's UniProt cap, whose prompt-token count the reference is)
    chars = count(agent.get("max_result_chars"))
    pinned_tools = route.get("tools_sha256")
    if chars is None or not isinstance(pinned_tools, str):
        raise Failed(f"{out}: pins no tools (tools_sha256 and agent_config.max_result_chars) for {model}")
    built = tools_sha256(code, chars)
    if built != pinned_tools:
        raise Failed(f"{out}: pins tools {pinned_tools[:12]} for {model}, the code at --code builds {built[:12]} for "
                     f"its cap {chars}: the requests would not be that OUT's runner's")
    url = route.get("base_url") if isinstance(route.get("base_url"), str) and route.get("base_url") else None
    return Out(out, read_baseline(out, model), url, cap, read_admitted(out, model, reference))


def tools_sha256(code: Code, max_result_chars: int) -> str:
    """The runner's tools hash for a cap (loop.tools_sha256 over tools.agent_tools, as model_fingerprint pins it)."""
    return code.loop.tools_sha256(code.tools.agent_tools(max_result_chars))


# --- the admission --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    model: str
    base_url: str
    backend: str
    outs: tuple[Path, ...]
    reference: dict
    longest_prompt_tokens: int
    declared_revision: str | None = None
    # set when the longest prompt is the private file's (longest_prompt): its value is then kept out of every file
    # and report, and the evidence names this sha256 instead
    longest_private_sha256: str | None = None
    # the dev malfunction check's verdict file, required for a join (science, design log, 11:54Z; read_verdict)
    behaviour_verdict: Path | None = None
    upstream_url: str | None = None
    reference_base_url: str | None = None
    reference_upstream_url: str | None = None
    front: FrontAdmin | None = None  # None: not enabled (--no-enable), the status says ADMITTED


@dataclass
class Outcome:
    status: str  # PASS, FAIL (nothing written) or PARTIAL (written, but a read-back or the enable failed)
    reason: str | None
    exit_code: int
    report: dict
    written: list[str] = field(default_factory=list)


def check_reference(req: Requests, reference: dict) -> None:
    """The model's reference must be for the requests the code builds (else its token count says nothing)."""
    if reference.get("served_model") != req.served_model:
        raise Failed(f"the reference is for {reference.get('served_model')!r}, the route serves {req.served_model!r}")
    for k in MODEL_FIELDS:
        if not isinstance(reference.get(k), str) or not reference[k]:
            raise Failed(f"the reference has no {k}")
    if count(reference.get("smoke_prompt_tokens")) is None:
        raise Failed("the reference has no smoke_prompt_tokens")
    if reference.get("smoke_system_sha256") != req.system_sha256:
        raise Failed("the smoke's system prompt is not the one the reference's token count was measured with")
    if reference.get("smoke_body_sha256") is not None and reference["smoke_body_sha256"] != req.smoke_body_sha256:
        raise Failed("the smoke body the code builds is not the one the reference's token count was measured with")


def probe_candidate(code: Code, req: Requests, p: Params, t: Transport, cap: int) -> dict:
    """Checks 1-4 on the candidate (the reference comparison aside): its evidence, fingerprint and shape."""
    ref = p.reference
    base = canonical_url(p.base_url)
    status, raw = t.get(server_root(base) + "/version")
    if status != 200:
        raise Failed(f"GET /version: HTTP {status}")
    version = loads_object(raw, "GET /version").get("version")
    if version != ref["vllm_version"]:
        raise Failed(f"GET /version: vLLM version {version}, the model's is {ref['vllm_version']}")
    status, raw = t.get(base + "/models")
    if status != 200:
        raise Failed(f"GET /v1/models: HTTP {status}")
    data = loads_object(raw, "GET /v1/models").get("data")
    entries = [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []
    ids = [e.get("id") for e in entries]
    match = [e for e in entries if e.get("id") == req.served_model]
    if len(match) != 1:
        raise Failed(f"GET /v1/models: the served name {req.served_model!r} is "
                     f"{'not listed' if not match else 'listed more than once'} (ids {ids})")
    listed = match[0]
    window = count(listed.get("max_model_len"))
    needed = p.longest_prompt_tokens + cap
    if window is None or window < needed:
        if p.longest_private_sha256 is not None:
            raise Failed(f"GET /v1/models: the window max_model_len {window} is under the longest dev prompt (the "
                         f"private file's, sha256 {p.longest_private_sha256[:12]}) + the output cap {cap}")
        raise Failed(f"GET /v1/models: the window max_model_len {window} is under the longest dev prompt "
                     f"{p.longest_prompt_tokens} + the output cap {cap} = {needed}")
    root = listed.get("root") if isinstance(listed.get("root"), str) else ""
    snap = SNAPSHOT_RE.search(root)
    if snap:
        revision, source = snap.group(1), "snapshot_path"
        if p.declared_revision is not None and p.declared_revision != revision:
            raise Failed(f"the declared revision {p.declared_revision} is not the snapshot's {revision}")
    elif p.declared_revision is None:
        raise Failed("the listed root is not a Hugging Face snapshot path: give the checkpoint revision with "
                     "--declared-revision (recorded as declared)")
    else:
        revision, source = p.declared_revision, "declared"
    if revision != ref["revision"]:
        raise Failed(f"checkpoint revision {revision} ({source}), the model's is {ref['revision']}")

    status, raw = t.post(base + "/chat/completions", req.smoke)
    smoke = read_reply(code, req, status, raw, "smoke")
    fp = smoke.fingerprint
    parsed = FINGERPRINT_RE.match(fp)
    if parsed is None or parsed.group("version") != ref["vllm_version"]:
        raise Failed(f"smoke: the system fingerprint {fp!r} is not vLLM {ref['vllm_version']}'s")
    prompt = count(smoke.usage.get("prompt_tokens"))
    if prompt != ref["smoke_prompt_tokens"]:
        raise Failed(f"smoke: prompt tokens {prompt}, the model's reference is {ref['smoke_prompt_tokens']}")
    reply, shape, names = tool_call(code, req, t, base, "tool call")
    if reply.fingerprint != fp:
        raise Failed(f"tool call: system fingerprint {reply.fingerprint!r}, the smoke's was {fp!r} (one backend, "
                     f"one fingerprint)")
    smoke_choice = (smoke.assembled.get("choices") or [{}])[0]
    choice = reply.assembled["choices"][0]
    return {
        "system_fingerprint": fp, "shape": shape, "revision": revision, "revision_source": source,
        "evidence": {
            "vllm_version": version,
            "fingerprint_parts": {"version": parsed.group("version"), "tp": parsed.group("tp"),
                                  "ep": bool(parsed.group("ep"))},
            "models_listing": {"ids": ids, "root_sha256": sha256_text(root), "root_is_snapshot_path": bool(snap)},
            "window": ({"max_model_len": window, "longest_prompt_tokens": p.longest_prompt_tokens,
                        "output_cap": cap, "needed": needed} if p.longest_private_sha256 is None else
                       {"max_model_len": window, "longest_prompt_tokens": None, "output_cap": cap, "needed": None,
                        "longest_prompt_private_sha256": p.longest_private_sha256}),
            "smoke": {"prompt_tokens": prompt, "reference_prompt_tokens": ref["smoke_prompt_tokens"],
                      "completion_tokens": count(smoke.usage.get("completion_tokens")),
                      "finish_reason": smoke_choice.get("finish_reason") if isinstance(smoke_choice, dict) else None,
                      "body_sha256": req.smoke_body_sha256, "system_sha256": req.system_sha256},
            "tool_call": {"shape": shape, "tool_names": names, "calls": len(names),
                          "finish_reason": choice.get("finish_reason"),
                          "prompt_tokens": count(reply.usage.get("prompt_tokens")),
                          "completion_tokens": count(reply.usage.get("completion_tokens")),
                          "response_fields": sorted(reply.assembled),
                          "usage_fields": sorted(reply.usage)},
        }}


def recorded_shapes(row_lists: list[list[dict]], where: list[Path]) -> list[tuple[dict, str]]:
    """(shape, fingerprint) of every admitted entry that records its tool-call reply's shape."""
    recorded = []
    for rows, path in zip(row_lists, where):
        for row in rows:
            shape = ((row.get("evidence") or {}).get("tool_call") or {}).get("shape")
            if shape is None:
                continue
            if not valid_shape(shape):
                raise Failed(f"{path}: an admitted entry's recorded reply shape is malformed")
            recorded.append((shape, row["system_fingerprint"]))
    return recorded


def reference_shapes(code: Code, req: Requests, p: Params, outs: list[Out], candidate_fp: str,
                     t: Transport | None, cap: int) -> tuple[list[str], list[dict], list[str]]:
    """(the kinds, the shapes, the fingerprints) the candidate's shape is compared with."""
    kinds, shapes, fps = [], [], []
    if p.reference_base_url is not None:
        if t is None:
            raise Failed("--reference-base-url needs a transport for the reference")
        # the reference passes the same identity checks, and must BE an admitted backend: its fingerprint at its URL
        # (the upstream's behind a forward) admitted in every OUT, or the OUT's incumbent at its pinned route URL
        ref_p = dataclasses.replace(p, base_url=p.reference_base_url, upstream_url=p.reference_upstream_url)
        try:
            probed = probe_candidate(code, req, ref_p, t, cap)
        except Failed as exc:
            raise Failed(f"the reference backend: {exc}") from None
        ref_sha = url_sha256(p.reference_upstream_url or p.reference_base_url)
        for o in outs:
            if (probed["system_fingerprint"], ref_sha) not in o.admitted_backends(p.model):
                raise Failed(f"the reference backend ({probed['system_fingerprint']!r} at URL sha256 {ref_sha[:12]}) "
                             f"is not admitted in {o.path} (nor its incumbent at its pinned route URL)")
        kinds.append("live")
        shapes.append(probed["shape"])
        fps.append(probed["system_fingerprint"])
    recorded = recorded_shapes([o.rows(p.model) for o in outs], [o.path for o in outs])
    if recorded:
        kinds.append("recorded")
        for shape, fp in recorded:
            if shape not in shapes:
                shapes.append(shape)
            if fp not in fps:
                fps.append(fp)
    if not shapes:
        if all(o.baseline == candidate_fp for o in outs):
            return [SELF], [], [candidate_fp]
        raise Failed("no reference reply: no admitted entry of the model records a reply shape and no "
                     "--reference-base-url was given (only the OUTs' own baseline backend may be compared with itself)")
    return kinds, shapes, fps


def plan_entries(rows: list[dict], o: Out, candidate: dict, baseline_entry: dict) -> list[dict]:
    """What to append to an OUT's list: its baseline when not admitted yet (unless the candidate is it), then the
    candidate unless the same (fingerprint, backend id, backend URL) is already there: the triple the front's enable
    gate looks for, so a backend admitted under another id (the "baseline" entry at the same URL) still gets its own."""
    fps = {r["system_fingerprint"] for r in rows}
    add = []
    if o.baseline not in fps and candidate["system_fingerprint"] != o.baseline:
        add.append(baseline_entry)
    key = ("system_fingerprint", "backend", "backend_url_sha256")
    if not any(all(r.get(k) == candidate[k] for k in key) for r in rows):
        add.append(candidate)
    return add


@contextlib.contextmanager
def locked(paths: list[Path]):
    fds = []
    try:
        for path in sorted(paths, key=lambda q: str(q.resolve())):
            fd = os.open(path / LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
            fds.append(fd)
            deadline = time.monotonic() + LOCK_WAIT_S
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise Failed(f"{path}: another admission holds its lock") from None
                    time.sleep(0.2)
        yield
    finally:
        for fd in fds:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def prepare(path: Path, doc: dict) -> str:
    """The new file as a temporary beside ``path``, written and fsynced: the part of a write that can fail on a full
    disk or a permission, done for every OUT before any file is replaced. -> the temporary's path."""
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return tmp


def install(tmp: str, path: Path) -> None:
    """The prepared temporary renamed over ``path``: one change for the runner's reload by (inode, mtime, size)."""
    try:
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def sync_dir(path: Path) -> None:
    """The directory fsynced, so the rename survives a crash."""
    dfd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def write_atomic(path: Path, doc: dict) -> None:
    install(prepare(path, doc), path)
    sync_dir(path.parent)


def read_back(o: Out, model: str, old_doc: dict | None, old_rows: list[dict], added: list[dict]) -> str | None:
    try:
        doc = json.loads((o.path / ADMITTED_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"unreadable ({type(exc).__name__})"
    why = malformed(doc)
    if why is not None:
        return f"malformed ({why})"
    rows = (doc["models"].get(model) or {}).get("admitted", [])
    if rows[:len(old_rows)] != old_rows:
        return "an earlier entry changed"
    if rows[len(old_rows):] != added:
        return "the new entries are not as written"
    others = {m: e for m, e in doc["models"].items() if m != model}
    if others != {m: e for m, e in ((old_doc or {}).get("models") or {}).items() if m != model}:
        return "another model's entries changed"
    return None


def commit(p: Params, req_outs: list[Out], shape: dict, candidate_for, baseline_for,
           model_fields: dict) -> tuple[list[str], list[dict], str | None]:
    """Under every OUT's lock: re-read each file (another admission may have written since the checks), compare the
    candidate's shape with every shape it now records, plan the entries; then write every OUT's temporary (a failure
    there writes nothing: Failed), rename each over its file, and read every OUT back. -> (the OUTs holding the entry,
    per-OUT rows, the first problem or None). A problem after a rename is an incomplete admission: the OUTs already
    holding the entry keep it (append-only: an entry is never removed), the backend is not enabled, and the same
    admission run again completes it (an entry already there is not appended twice)."""
    rows_out: list[dict] = []
    with locked([o.path for o in req_outs]):
        plans = []
        for o in req_outs:
            doc = read_admitted(o.path, p.model, p.reference)
            rows = list(((doc or {}).get("models") or {}).get(p.model, {}).get("admitted", []))
            for recorded, fp in recorded_shapes([rows], [o.path]):
                diff = shape_difference(shape, recorded)
                if diff:
                    raise Failed(f"tool call: the reply shape differs from the one {o.path} records for {fp!r} "
                                 f"(written since the checks): {diff}")
            add = plan_entries(rows, o, candidate_for(o), baseline_for(o))
            new = {"schema": SCHEMA, "models": dict((doc or {}).get("models") or {})}
            if doc is not None:
                new.update({k: v for k, v in doc.items() if k not in ("schema", "models")})
            entry = dict(new["models"].get(p.model) or {**model_fields, "admitted": []})
            entry["admitted"] = rows + add
            new["models"][p.model] = entry
            # what the read-back compares with: copies, so nothing done to the written document can move them
            plans.append((o, json.loads(json.dumps(doc)), json.loads(json.dumps(rows)),
                          json.loads(json.dumps(add)), new))
            rows_out.append({"out": str(o.path), "baseline": o.baseline, "entries_added": len(add),
                             "added_backends": [a["backend"] for a in add]})
        temps: dict[int, str] = {}
        for i, (o, _, _, add, new) in enumerate(plans):
            if not add:
                continue
            try:
                temps[i] = prepare(o.path / ADMITTED_FILE, new)
            except OSError as exc:
                for tmp in temps.values():
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(tmp)
                raise Failed(f"{o.path}: the new file could not be written ({type(exc).__name__}: {exc}); no OUT "
                             f"was changed") from None
        holding, problem, renames_stopped = [], None, False
        for i, (o, _, _, add, _) in enumerate(plans):
            if i in temps:
                if renames_stopped:  # an earlier rename failed: the rest are not installed
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(temps[i])
                    continue
                try:
                    install(temps[i], o.path / ADMITTED_FILE)
                except OSError as exc:
                    problem = problem or f"{o.path}: the rename failed ({type(exc).__name__}: {exc})"
                    renames_stopped = True
                    continue
            # renamed now, or already in the file (a re-run): read back below either way. The directory is fsynced
            # in both cases, so a re-run after a failed fsync makes the earlier rename durable before it can PASS.
            holding.append(i)
            try:
                sync_dir(o.path)
            except OSError as exc:
                rows_out[i]["durable"] = False
                problem = problem or (f"{o.path}: the directory fsync failed ({type(exc).__name__}: {exc}): the "
                                      f"entry is in place, its durability uncertain")
                continue
            rows_out[i]["durable"] = True
        written = []
        for i in holding:
            o, doc, rows, add, _ = plans[i]
            why = read_back(o, p.model, doc, rows, add)
            rows_out[i]["read_back"] = why is None
            if why is None:
                written.append(str(o.path))
            elif problem is None:
                problem = f"{o.path}: read-back: {why}"
    return written, rows_out, problem


class FrontAdmin(Protocol):
    def request(self, req: dict) -> dict: ...


class UnixFrontAdmin:
    """The front's admin API (builder A, results/e167-front/front.py): one JSON object per line on its unix socket,
    one JSON reply per line."""

    def __init__(self, socket_path: Path, timeout_s: float = 60.0):
        self.socket_path = socket_path
        self.timeout_s = timeout_s

    def request(self, req: dict) -> dict:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout_s)
                s.connect(str(self.socket_path))
                s.sendall(json.dumps(req).encode("utf-8") + b"\n")
                data = b""
                while not data.endswith(b"\n"):
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    data += chunk
            reply = json.loads(data.decode("utf-8"))
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": f"the front's admin socket: {type(exc).__name__}: {exc}"}
        return reply if isinstance(reply, dict) else {"ok": False, "error": "the front's reply is not an object"}


def read_verdict(path: Path, model: str, backend: str, backend_sha: str) -> dict:
    """The behaviour verdict's evidence, or Failed: a passing e167-behaviour-verdict/1 for this model, backend id and
    identity URL hash, at most 24 h old, with the sha256 of each input."""
    from datetime import UTC, datetime

    def bad(why: str) -> Failed:
        return Failed(f"behaviour verdict {path}: {why}")

    try:
        raw = Path(path).read_bytes()
        doc = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise bad(f"unreadable ({type(exc).__name__})") from None
    if not isinstance(doc, dict) or doc.get("schema") != VERDICT_SCHEMA:
        raise bad(f"its schema is not {VERDICT_SCHEMA}")
    if doc.get("verdict") != "pass":
        raise bad(f"not passing (verdict {doc.get('verdict')!r})")
    if doc.get("model") != model:
        raise bad(f"for model {doc.get('model')!r}, not {model}")
    if doc.get("backend") != backend:
        raise bad(f"for backend {doc.get('backend')!r}, not {backend}")
    if doc.get("backend_url_sha256") != backend_sha:
        raise bad("for another backend URL (backend_url_sha256: the identity URL, the upstream's behind a forward)")
    at = doc.get("at")
    try:
        when = (datetime.strptime(at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
                if isinstance(at, str) and VERDICT_AT_RE.fullmatch(at) else None)
    except ValueError:
        when = None
    if when is None:
        raise bad(f"no time 'at' in the form YYYY-MM-DDTHH:MM:SSZ ({at!r})")
    age = (datetime.now(UTC) - when).total_seconds()
    if age > VERDICT_MAX_AGE_S or age < 0:
        raise bad(f"its time {at} is {'older than 24 h' if age > 0 else 'in the future'}")
    inputs = doc.get("inputs")
    if (not isinstance(inputs, dict) or not inputs
            or not all(isinstance(k, str) and isinstance(v, str) and SHA256_RE.fullmatch(v) for k, v in inputs.items())):
        raise bad("no inputs (name -> sha256 of each input file)")
    return {"file_sha256": hashlib.sha256(raw).hexdigest(), "schema": VERDICT_SCHEMA, "verdict": "pass", "at": at,
            "inputs": dict(inputs)}


def enable(front: FrontAdmin, model: str, backend: str, fingerprint: str, identity_sha256: str) -> dict:
    """The front's enable (its own gate reads every OUT's file again), then its list: the backend must now be enabled
    with this fingerprint at this identity URL. -> {"ok", ...} with the front's answers (ids and hashes only)."""
    reply = front.request({"cmd": "enable", "backend": backend, "system_fingerprint": fingerprint, "model": model})
    if reply.get("ok") is not True:
        return {"ok": False, "enable_reply": reply}
    listed = front.request({"cmd": "list"})
    rows = [b for b in (listed.get("backends") or []) if isinstance(b, dict) and b.get("backend") == backend] \
        if listed.get("ok") is True else []
    if len(rows) != 1:
        return {"ok": False, "enable_reply": reply, "error": f"the front lists {len(rows)} backends {backend!r}"}
    row = rows[0]
    want = {"state": "enabled", "system_fingerprint": fingerprint, "identity_url_sha256": identity_sha256}
    got = {k: row.get(k) for k in want}
    if got != want:
        return {"ok": False, "enable_reply": reply, "listed": got, "error": "the front's list does not show the "
                "backend enabled with this fingerprint at this identity URL"}
    return {"ok": True, "listed": got}


def admit(p: Params, code: Code, transport: Transport, reference_transport: Transport | None = None) -> Outcome:
    admission_id = uuid.uuid4().hex
    at = utc_now()
    report: dict = {"status": None, "reason": None, "model": p.model, "backend": p.backend, "admission_id": admission_id,
                    "at": at, "code_commit": code.commit, "outs": []}
    try:
        if not BACKEND_RE.fullmatch(p.backend) or p.backend == "baseline":
            raise Failed(f"--backend {p.backend!r}: the front's id form, 1-64 letters, digits, '.', '_' or '-' (not "
                         f"'baseline')")
        if not p.outs:
            raise Failed("no --out given")
        if len({o.resolve() for o in p.outs}) != len(p.outs):
            raise Failed("an OUT is given twice")
        req = build_requests(code, p.model)
        check_reference(req, p.reference)
        outs = [read_out(code, o, p.model, req, p.reference) for o in p.outs]
        backend_sha = url_sha256(p.upstream_url) if p.upstream_url else url_sha256(p.base_url)
        # a join (science, design log, 11:54Z): checked before any request when the verdict is given, or when the
        # backend's identity URL is not every OUT's pinned one (then it is a join whatever it serves)
        verdict = read_verdict(p.behaviour_verdict, p.model, p.backend, backend_sha) \
            if p.behaviour_verdict is not None else None
        at_pins = all(o.pinned_url and url_sha256(o.pinned_url) == backend_sha for o in outs)
        if verdict is None and not at_pins:
            raise Failed("a join of a running OUT needs its behaviour verdict (--behaviour-verdict: the dev "
                         "malfunction check's passing verdict for this model and endpoint; science, design log, 11:54Z)")
        cap = max([req.output_cap, *(o.pinned_cap for o in outs)])
        cand = probe_candidate(code, req, p, transport, cap)
        fp = cand["system_fingerprint"]
        report["system_fingerprint"] = fp
        kinds, shapes, ref_fps = reference_shapes(code, req, p, outs, fp, reference_transport, cap)
        for shape in shapes:
            diff = shape_difference(cand["shape"], shape)
            if diff:
                raise Failed(f"tool call: the reply shape differs from the reference's ({', '.join(kinds)}: "
                             f"{', '.join(ref_fps)}): {diff}")
        contacted = url_sha256(p.base_url)
        if verdict is None and not all(o.baseline == fp for o in outs):  # at the pinned URL, another fingerprint
            raise Failed(f"a join of a running OUT needs its behaviour verdict (--behaviour-verdict): the backend at "
                         f"the pinned URL serves {fp}, not every OUT's baseline (science, design log, 11:54Z)")
        tool_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        evidence = {"admission_id": admission_id, "tool_sha256": tool_sha, "code_commit": code.commit,
                    "contacted_url_sha256": contacted, "via_forward": p.upstream_url is not None,
                    "declared_revision": p.declared_revision, **cand["evidence"],
                    "reference": {"kind": kinds, "system_fingerprints": ref_fps, "shapes": shapes},
                    "behaviour_verdict": verdict if verdict is not None else {"required": False, "why": INCUMBENT_WHY}}
        report["checks"] = cand["evidence"]
        report["reference"] = evidence["reference"]
        model_fields = {**{k: p.reference[k] for k in MODEL_FIELDS}, "ruling": RULING}

        def candidate_for(o: Out) -> dict:
            return {"system_fingerprint": fp, "backend": p.backend, "backend_url_sha256": backend_sha,
                    "admitted_at": at, "revision_source": cand["revision_source"],
                    "evidence": {**evidence, "is_baseline": fp == o.baseline}}

        def baseline_for(o: Out) -> dict:
            return {"system_fingerprint": o.baseline, "backend": "baseline",
                    "backend_url_sha256": url_sha256(o.pinned_url) if o.pinned_url else None,
                    "admitted_at": at, "revision_source": "baseline",
                    "evidence": {"admission_id": admission_id, "is_baseline": True,
                                 "source": "served-models.json models.<m>.system_fingerprint: the OUT's incumbent, "
                                           "accepted by the runner since its first fingerprinted turn; the URL hash "
                                           "is the OUT's pinned route base URL's",
                                 "admitted_with": p.backend}}
    except Failed as exc:
        report.update(status="FAIL", reason=str(exc))
        return Outcome("FAIL", str(exc), 1, report)

    try:
        written, rows, problem = commit(p, outs, cand["shape"], candidate_for, baseline_for, model_fields)
    except Failed as exc:  # no lock, a file changed since the checks, or a temporary not written: before any rename
        report.update(status="FAIL", reason=str(exc))
        return Outcome("FAIL", str(exc), 1, report)
    report["outs"] = rows
    report["written"] = written
    if problem is not None:
        missing = [str(o.path) for o in outs if str(o.path) not in written]
        report.update(status="PARTIAL", reason=f"INCOMPLETE ADMISSION: {problem}. The entry is in {written or 'no OUT'} "
                                               f"and not in {missing}; the backend is NOT enabled. Fix the cause and run "
                                               f"the same admission again to complete it (entries already in place are "
                                               f"not appended twice; none is ever removed).")
        return Outcome("PARTIAL", report["reason"], 3, report, written)
    if p.front is not None:
        report["enable"] = enable(p.front, p.model, p.backend, fp, backend_sha)
        if report["enable"].get("ok") is not True:
            report.update(status="PARTIAL", reason=f"admitted in every OUT, but the front's enable failed: "
                                                   f"{report['enable']}")
            return Outcome("PARTIAL", report["reason"], 3, report, written)
    else:  # --no-enable: admitted in every OUT, deliberately not enabled; a status no one can take for PASS
        report["enable"] = {"skipped": "--no-enable"}
        report.update(status="ADMITTED")
        return Outcome("ADMITTED", None, 0, report, written)
    report.update(status="PASS")
    return Outcome("PASS", None, 0, report, written)


# --- the command line -----------------------------------------------------------------------------------------------

def emit(obj: dict, keys: list[str]) -> None:
    text = json.dumps(obj, indent=1, sort_keys=True)
    for key in keys:
        if key != PLACEHOLDER_KEY and key in text:
            raise SystemExit("refusing: the output would hold a key")
    print(text)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def longest_prompt(served: str, public: Path, private: Path, pin_file: Path) -> tuple[int, str | None]:
    """(the served model's longest dev prompt, the private file's sha256 when that value is the private file's).
    The larger of the committed file's and the private file's maximum; the private file must be the one the committed
    pin names, so an admission never runs on fewer dev sources than the pin's file holds."""
    try:
        pin = PIN_RE.fullmatch(pin_file.read_bytes())
    except OSError as exc:
        raise SystemExit(f"refusing: the pin {pin_file} is unreadable ({type(exc).__name__})") from None
    if pin is None:
        raise SystemExit(f"refusing: the pin {pin_file} is not one sha256 line of the private file")
    pinned = pin.group(1).decode("ascii")
    path = private.expanduser()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"refusing: the private longest-prompt file {path} is unreadable ({type(exc).__name__}): "
                         f"the admission runs on the cluster host that keeps it") from None
    got = hashlib.sha256(raw).hexdigest()
    if got != pinned:
        raise SystemExit(f"refusing: {path} has sha256 {got[:12]}, the pin {pin_file.name} names {pinned[:12]}")
    values = []
    for doc in (load_json(public), json.loads(raw)):
        models = doc.get("models") if isinstance(doc, dict) else None
        entry = models.get(served) if isinstance(models, dict) else None
        values.append(count(entry.get("max_prompt_tokens")) if isinstance(entry, dict) else None)
    pub, priv = values
    if pub is None and priv is None:
        raise SystemExit(f"refusing: no longest prompt for {served} in {public} or the private file")
    if priv is not None and (pub is None or priv > pub):
        return priv, got
    return pub, None


FRONT_CONFIG_SCHEMA = "e167-front-config/1"


def front_inventory(path: Path, model: str, served: str) -> tuple[list[Path], Path]:
    """(the OUTs the front's config lists for the model, which its enable gate reads every one of: the inventory;
    the front's admin socket)."""
    try:
        data = load_json(path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"refusing: the front config is unreadable ({type(exc).__name__})") from None
    if not isinstance(data, dict) or data.get("schema") != FRONT_CONFIG_SCHEMA:
        raise SystemExit(f"refusing: the front config's schema is not {FRONT_CONFIG_SCHEMA}")
    entry = (data.get("models") or {}).get(served) if isinstance(data.get("models"), dict) else None
    if not isinstance(entry, dict) or entry.get("name") != model:
        raise SystemExit(f"refusing: the front config has no entry {served!r} named {model!r}")
    outs = entry.get("outs")
    if not isinstance(outs, list) or not outs or not all(isinstance(o, str) and os.path.isabs(o) for o in outs):
        raise SystemExit(f"refusing: the front config lists no OUT (absolute paths) for {model}")
    sock = data.get("admin_socket")
    if not isinstance(sock, str) or not os.path.isabs(sock):
        raise SystemExit("refusing: the front config names no admin_socket (an absolute path)")
    return [Path(o) for o in outs], Path(sock)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("admit", "shape"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True, help="the runner's model name (e.g. gemma-4-31b)")
        s.add_argument("--base-url", required=True, help="the candidate's OpenAI base URL (http://HOST:PORT/v1)")
        s.add_argument("--key-file", type=Path, default=None)
        s.add_argument("--code", type=Path, default=HERE.parents[1], help="the deploy whose runner builds the requests")
    a = sub.choices["admit"]
    a.add_argument("--backend", required=True, help="the backend's id (the front's)")
    a.add_argument("--front-config", type=Path, default=None,
                   help="the front's config (e167-front-config/1): every OUT it lists for the model is admitted "
                        "into, and the enable goes to its admin_socket")
    a.add_argument("--out", type=Path, action="append", default=[],
                   help="an OUT that addresses the model (beside the front config's; alone only with --no-enable)")
    a.add_argument("--declared-revision", default=None, help="the checkpoint revision when root is no snapshot path")
    a.add_argument("--upstream-url", default=None, help="the backend's own URL when --base-url is a forward to it")
    a.add_argument("--reference-base-url", default=None, help="an already-admitted backend, for a live reference")
    a.add_argument("--reference-upstream-url", default=None, help="the reference's own URL behind a forward")
    a.add_argument("--reference-key-file", type=Path, default=None)
    a.add_argument("--references", type=Path, default=HERE / "model-references.json")
    a.add_argument("--longest", type=Path, default=HERE / "longest-prompts.json")
    a.add_argument("--longest-private", type=Path, default=LONGEST_PRIVATE,
                   help="the private longest-prompt file (every dev source; mode 600, never in git)")
    a.add_argument("--no-enable", action="store_true", help="admit only (status ADMITTED, never PASS)")
    a.add_argument("--behaviour-verdict", type=Path, default=None,
                   help="a join's dev malfunction check verdict (e167-behaviour-verdict/1; science, design log, 11:54Z)")
    sub.choices["shape"].add_argument("--repeat", type=int, default=1)
    args = ap.parse_args(argv)

    key = read_key(args.key_file)
    code = load_code(args.code)
    if args.cmd == "shape":  # a diagnosis: the tool-call reply's shape and fingerprint, N times, nothing written
        req = build_requests(code, args.model)
        t = HttpTransport(key, req.request_timeout_s)
        rows = []
        for _ in range(max(1, args.repeat)):
            try:
                reply, shape, names = tool_call(code, req, t, canonical_url(args.base_url), "tool call")
                rows.append({"system_fingerprint": reply.fingerprint, "shape": shape, "tool_names": names,
                             "prompt_tokens": count(reply.usage.get("prompt_tokens"))})
            except Failed as exc:
                rows.append({"failed": str(exc)})
        shapes = {json.dumps(r.get("shape"), sort_keys=True) for r in rows if "shape" in r}
        emit({"model": args.model, "replies": rows, "distinct_shapes": len(shapes)}, [key])
        return 0 if len(shapes) == 1 and all("shape" in r for r in rows) else 1

    refs = load_json(args.references).get("models") or {}
    if args.model not in refs:
        raise SystemExit(f"refusing: {args.references} has no reference for {args.model}")
    served = refs[args.model].get("served_model")
    # the pin is always the committed file (no option replaces it): a shorter private file cannot pass
    longest_tokens, longest_private = longest_prompt(served, args.longest, args.longest_private, LONGEST_PRIVATE_PIN)
    ref_key = read_key(args.reference_key_file) if args.reference_base_url else None
    if args.front_config is None and not args.no_enable:
        raise SystemExit("refusing: an enabled admission takes its OUTs and the front's admin socket from "
                         "--front-config (the front's inventory of every OUT that addresses the model); or --no-enable")
    outs, admin_socket = front_inventory(args.front_config, args.model, served) if args.front_config is not None \
        else ([], None)
    seen = {o.resolve() for o in outs}
    outs += [o for o in args.out if o.resolve() not in seen]
    p = Params(model=args.model, base_url=args.base_url, backend=args.backend, outs=tuple(outs),
               reference=refs[args.model], longest_prompt_tokens=longest_tokens,
               longest_private_sha256=longest_private, declared_revision=args.declared_revision,
               behaviour_verdict=args.behaviour_verdict, upstream_url=args.upstream_url,
               reference_base_url=args.reference_base_url, reference_upstream_url=args.reference_upstream_url,
               front=None if args.no_enable else UnixFrontAdmin(admin_socket))
    timeout = float(refs[args.model].get("request_timeout_s") or 600.0)
    got = admit(p, code, HttpTransport(key, timeout),
                HttpTransport(ref_key, timeout) if ref_key is not None else None)
    emit(got.report, [key, *([ref_key] if ref_key else [])])
    print(f"{got.status}: {got.reason}" if got.reason else got.status, file=sys.stderr)
    return got.exit_code


if __name__ == "__main__":
    sys.exit(main())
