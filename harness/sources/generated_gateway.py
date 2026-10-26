"""A second generator for the self-check crossing (E4.3): the Opus recipe sent to a HPCNet LLM Gateway model.

E4.3 (preregistration §6, design change-log entry 47) needs three drafts per graph from a second model family "by
exactly the Opus recipe" (``harness.sources.generated``): the same input bundle, read from the Opus bundle directory
and checked against its request hash; the same system prompt and user text, byte for byte; the same output budget
(max_tokens 100,000). Qwen3.8-27B's window cannot take the Rhea bundles, so the issue's fallback,
DeepSeek-V4-Flash-0731 (kept by design log's pilot), draws. Only what an OpenAI-compatible endpoint cannot carry
differs, and each draw record says so:

* no cache breakpoint and no thinking or effort setting: those are Anthropic parameters. The model's server
  default applies, as in DeepSeek's agent runs (design ruling, design log; every attempt records it as
  ``thinking``): a model that reasons by default spends its reasoning from max_tokens as Opus's thinking does;
* no sampling parameter, as in the Opus recipe: the server's default sampling applies. Each draw sends a fixed,
  recorded seed (``DRAW_SEEDS``), so a draw can be requested again identically;
* the draws stream, because a 100,000-token reply can outlast the gateway's 180-s idle cut. ``stream`` and
  ``stream_options`` are transport and are not part of the request hash.

Fit. The issue forbids truncating, and the Gateway has no token-counting endpoint. ``count_input`` therefore sends
the exact request with max_tokens equal to the model's whole window. A server that enforces its window rejects
that before generating anything, and the rejection names the server's own prompt count and window. Nothing is
drawn unless prompt + 100,000 <= window. A rejection that cannot be read, or a server that accepts the count
request, leaves the fit unknown, and nothing is drawn. A draw rejected for context overflow is recorded as "does
not fit" and is never retried, because retrying with less input would be truncation.

Integrity (study lead's rule for the DeepSeek draws, design log). A streamed reply can lose whole events
on the server's side and still end cleanly (stream-integrity probe, 19 Oct), and the per-chunk token-gap guard is
unusable on DeepSeek, so each draw is checked as a whole (``integrity_failures``): the body is read as raw
server-sent events, not through the SDK's iterator, which ends the same way on ``[DONE]`` and on a body that just
stops; the reply counts only with ``[DONE]``, the usage-only chunk that ends a stream asked for usage, finish
reason ``stop``, the requested model, a reported prompt count of at least the counted prompt tokens, and its
completion tokens against what was delivered, twice:

* re-tokenized (External review of 4ad3ef9, item 1; design ruling, design log): the delivered content and reasoning,
  exactly as they arrived, counted on the model's own tokenizer (``TOKENIZERS``: the Hugging Face file, pinned by
  the sha256 of its bytes and read offline; a model without one is never drawn), plus the chat-template tokens the
  server counts for a reply (the end-of-sentence token a stop samples; the ``</think>`` that ends reasoning sent in a
  reasoning field), must come within ``RETOKENIZED_BAND_PCT`` (5%) of ``completion_tokens`` either way. Measured on
  the stream-integrity probe's archived DeepSeek-0731 replies (results/e43-deepseek-retokenize-20261019/): the
  content of 246 of 249 intact streams re-tokenizes to exactly the server's running count at its last content
  chunk, two more differ by 1 and 2 tokens (0.2%, 0.3%), and the last, +26 tokens (12.7%), is a silent gap the probe
  counted as intact (one chunk's counter rose 27 tokens while it delivered " other"); whole replies whose tool calls
  carry only string parameters, written back in the model's tool-call markup, differ by +1 (the end-of-sentence
  token) in 324 of 347, +2 in 22 and +3 in 1, at most 2.4%. No probe reply carried reasoning, so the ``</think>``
  token is unmeasured. 5% of a 5,000-token draft is 250 tokens, so a smaller gap still passes this check;
* against the characters delivered (content plus reasoning) inside ``CHARS_PER_TOKEN_BAND``. The band, 1.5 to 6.0
  characters per token, is set from text measured on DeepSeek-V4-Flash-0731's own tokenizer: 2.85 to 3.71 on the
  twelve Opus redraw drafts (f076e1b), 1.77 to 2.63 on the four input bundles (the densest text a draft can quote:
  opaque identifiers and numbers), and 1.48 to 4.01 on the stream-integrity probe's 299 DeepSeek-0731 tool-call
  replies (whose tool-call markup tokens deliver no characters; a draft has none). It catches gross loss only.

Files. Every attempt keeps what arrived verbatim: ``draw-<i>.txt`` is the content channel exactly as delivered (the
Opus layout's raw reply) and ``draw-<i>.reasoning.txt`` the reasoning channel, when it carried anything. Nothing is
trimmed. The reply the parse check reads is derived from the content deterministically and recorded (``reply``): the
content itself, or, when a server without a reasoning parser left a think block inline, everything after its first
``</think>``, written to ``draw-<i>.reply.txt``, which ``generated.load_draw`` then reads.

The draw rule (``DRAW_RULE``). An attempt that fails the integrity check or whose reply is not a readable,
non-empty Turtle ontology is redrawn with the identical request (body and seed), every attempt kept as
draw-<i>.attempt-<k>.*; nothing is repaired or truncated by hand. An attempt the server never answered (an HTTP
error, a refused connection) is sent again identically too, after ``RETRY_WAIT_S``, and counted apart (re-sends,
not redraws: ``attempt_summary``). An attempt is recorded before it is sent, so one the process never finished is
kept as interrupted. The run stops, to be resumed by running it again, when a draw reaches ``max_attempts``, when
two attempts in a row return the identical failing reply (the identical request cannot change it), on a refused
key or request, and on a context overflow (never redrawn).

Pause (design ruling: pause when the natural phase launches if first bytes approach 60 s). Before every request
the run looks for a stop file (``--pause-file``) and stops while it exists; an attempt in flight finishes. Every
attempt records its first-byte time; one over ``max_first_byte_s`` pauses the run after that attempt, by writing
the stop file with the reason. A draw's first byte includes the prefill of its whole bundle, so the threshold is
set per bundle on the command line, never defaulted.

Per-bundle stop (design ruling, design log, 23:06). An attempt the gateway cut before any byte (its 504, or no byte
within ``GATEWAY_FIRST_BYTE_S``) records ``gateway_cut``. Two such attempts at one bundle, over all its draws, stop
that bundle as infeasible for the model on this gateway: ``draw-infeasible.json`` says why, the bundle is never sent
again, and the command exits ``EXIT_INFEASIBLE`` so a launcher goes on with the next bundle. The run-wide pause file
is not written. The ruling names Rhea and opaque Rhea, whose prefill may outlast the gateway; the rule applies to
every bundle, since a cut before the first byte cannot tell prefill from queueing.

Draws 1, 2 and 3 run in order, one request at a time. Draws go to their own directory beside the Opus one
(``generated-deepseek-v4-flash-0731/``), in the Opus file layout, so ``generated.load_draw`` reads them. The key
is read inside the process from its key file only (default: the dedicated key), never from the environment, and
is never logged. Compute on the Gateway is covered by a HPCNet grant.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import httpx
import openai
from tokenizers import Tokenizer

from harness.sources import generated as gen

GATEWAY_BASE_URL = "https://gateway.example/api/v1"
QWEN = "Qwen/Qwen3.8-27B"
DEEPSEEK = "deepseek-ai/DeepSeek-V4-Flash-0731"
MAX_TOKENS = gen.PARAMETERS["max_tokens"]
# One seed per draw, fixed before any draw (E4.3, draw i).
DRAW_SEEDS = {1: 4301, 2: 4302, 3: 4303}
# Served context windows (max_model_len), each read from the server's own rejection of one request with a known
# padding prompt and max_tokens 5,000,000, on 2026-10-19 (no generation). The fallback order of the issue is
# Qwen, then DeepSeek, then GLM.
CONTEXT_WINDOWS = {
    "Qwen/Qwen3.8-27B": 262_144,
    "deepseek-ai/DeepSeek-V4-Flash-0731": 700_000,
    "deepseek-ai/DeepSeek-V4.1-Flash": 1_048_576,
    "zai-org/GLM-5.3-Flash": 1_048_576,
}
DEFAULT_KEY_FILE = Path.home() / ".config" / "www-2027" / "gateway_api_key_dedicated"
DEFAULT_PAUSE_FILE = Path.home() / "www-2027" / "selfcheck-deepseek" / "PAUSE"
COUNT_FILE = "draw-input-count.json"  # matches draw-*, so generated.draw_files and retire_draws take it along
COUNT_TIMEOUT_S = 120.0
DRAW_TIMEOUT_S = 900.0  # per read of the stream, not per draw
CHARS_PER_TOKEN_BAND = (1.5, 6.0)  # delivered characters per completion token (module docstring)
RETOKENIZED_BAND_PCT = 5  # design ruling, design log: the re-tokenized reply within 5% of completion_tokens
# The model's own tokenizer: the file, and the sha256 of its uncompressed bytes. DeepSeek's is tokenizer.json of
# huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731, fetched 2026-10-19, the file the draw estimate counted inputs
# with (results/e43-deepseek-estimate-20261019/); kept gzipped in data/tokenizers/ and read offline.
TOKENIZER_DIR = Path(__file__).resolve().parents[2] / "data" / "tokenizers"
TOKENIZERS = {DEEPSEEK: ("deepseek-ai_DeepSeek-V4-Flash-0731.tokenizer.json.gz",
                         "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf")}
MAX_ATTEMPTS = 4  # per draw, every attempt counted: the first and at most three more
RETRY_WAIT_S = 120.0  # before an attempt follows a failed one
GATEWAY_FIRST_BYTE_S = 180.0  # the gateway answers 504 when no byte reached it by then (gateway, 19 Oct)
GATEWAY_CUTS_INFEASIBLE = 2  # attempts cut at the gateway that stop a bundle (design ruling, design log)
INFEASIBLE_FILE = "draw-infeasible.json"  # matches draw-*, so generated.draw_files and retire_draws take it along
EXIT_STOPPED, EXIT_INFEASIBLE = 3, 4  # the run stops / this bundle stops and the run goes on with the next
GRANT_BASIS = "HPCNet grant: no per-token charge on HPCNet LLM Gateway"
NOT_SENT = ("cache_control (Anthropic only), thinking and output_config.effort (Anthropic only; the server's default "
            "reasoning applies), no sampling parameter (as in the Opus recipe)")
THINKING = ("the server's default: no thinking, reasoning or chat-template setting is sent, the configuration of "
            "DeepSeek's agent runs (design ruling, design log)")
REPLY_AS_DELIVERED = "the content channel as delivered (draw-<i>.txt); nothing derived"
REPLY_AFTER_THINK = ("the content channel after its first </think> (a think block a server without a reasoning parser "
                     "left inline); nothing else removed")
DRAW_RULE = (
    "Study lead, design log: every draw streams and is checked as a whole ([DONE], the final usage chunk, "
    "finish_reason stop, the requested model, the counted prompt tokens, completion tokens against delivered "
    "characters and against the delivered text re-tokenized on the model's own tokenizer, within 5%) and parsed; a "
    "draw that fails the check or does not parse is redrawn with the identical request (body and seed), with every "
    "attempt recorded verbatim as draw-<i>.attempt-<k>.*; it is never repaired or truncated by hand. An attempt the "
    "server never answered is sent again identically and counted as a re-send; two attempts cut at the gateway "
    "before any byte stop the bundle as infeasible on this gateway. The draw is its first attempt that passes both."
)
GATEWAY_RULE = (
    "Design ruling, design log (23:06): if a bundle's prefill makes two attempts fail at the gateway (a gateway cut: "
    "its 504, or no byte within 180 s), that bundle stops and is reported as infeasible for the model on this "
    "gateway, without pushing past it; the run goes on with the other bundles. Counted over every attempt at the "
    "bundle's draws."
)
STOP_KINDS = ("does_not_fit", "key", "request")  # never sent again: the identical request cannot change the answer
RESEND_KINDS = ("infrastructure", "interrupted")
REDRAW_KINDS = ("integrity", "parse")

_WINDOW = (re.compile(r"maximum context length is (\d+)"), re.compile(r"max_model_len(?:=max_total_tokens)?=(\d+)"))
_PROMPT = (re.compile(r"\((\d+) in the messages"), re.compile(r"request has (\d+) input tokens"),
           re.compile(r"[Ii]nput length \((\d+)\)"), re.compile(r"prompt \(length (\d+)\)"))
_OVERFLOW = re.compile(r"context length|max_model_len|max_total_tokens", re.IGNORECASE)
_THINK_END = "</think>"


def gateway_request(bundle: dict, model: str = QWEN) -> dict:
    """The request body every draw of this bundle shares (each draw adds its seed): the Opus system and user text,
    byte for byte, and the Opus output budget."""
    return {"model": model, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": bundle["system"]}, {"role": "user", "content": bundle["user"]}]}


def default_out_dir(bundle_dir: Path, model: str) -> Path:
    """generated/ -> generated-deepseek-v4-flash-0731/, generated-dc/ -> generated-dc-deepseek-v4-flash-0731/."""
    return bundle_dir.parent / f"{bundle_dir.name}-{model.rsplit('/', 1)[-1].lower()}"


def gateway_client(key_file: Path = DEFAULT_KEY_FILE) -> openai.OpenAI:
    """A Gateway client whose key is read inside this process from ``key_file`` only (an older key left in the
    environment can never be used by mistake); the key is never printed, logged or put in the environment."""
    if not key_file.exists():
        raise RuntimeError(f"no key file at {key_file}")
    if key_file.stat().st_mode & 0o077:
        raise RuntimeError(f"the key file {key_file} is readable by others; it must be mode 600")
    return openai.OpenAI(api_key=key_file.read_text().strip(), base_url=GATEWAY_BASE_URL, max_retries=0)


def parse_rejection(message: str) -> tuple[int | None, int | None]:
    """(context window, prompt tokens) as a server's rejection states them, each None when it does not."""

    def first(patterns) -> int | None:
        for p in patterns:
            if m := p.search(message):
                return int(m.group(1))
        return None

    return first(_WINDOW), first(_PROMPT)


def _message(exc: openai.APIError) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        text = (err.get("message") if isinstance(err, dict) else None) or body.get("detail")
        if text:
            return str(text)
    return str(exc)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def count_input(bundle: dict, client, model: str = QWEN, window: int | None = None,
                timeout_s: float = COUNT_TIMEOUT_S) -> dict:
    """The server's own prompt count of the exact request, and whether it fits with the 100,000-token budget,
    read from the rejection of that request with max_tokens equal to the whole window (nothing is generated).
    ``fits`` is None when the server does not say."""
    window = CONTEXT_WINDOWS.get(model) if window is None else window
    if window is None:
        raise RuntimeError(f"no measured context window for {model}")
    body = gateway_request(bundle, model)
    result = {"input_tokens": None, "context_window": None, "max_tokens": MAX_TOKENS, "fits": None}
    for _ in range(2):
        try:
            stream = client.chat.completions.create(**{**body, "max_tokens": window}, stream=True, timeout=timeout_s)
        except openai.APIStatusError as exc:
            msg = _message(exc)
            served_window, n = parse_rejection(msg)
            if exc.status_code == 400 and n is None and served_window is not None and served_window != window:
                window = served_window  # max_tokens alone exceeded the served window: count again at that window
                continue
            fits = None if n is None or served_window is None else n + MAX_TOKENS <= served_window
            return {**result, "input_tokens": n, "context_window": served_window, "fits": fits,
                    "source": f"server rejection of the exact request with max_tokens = {window}",
                    "server_message": msg[:1000]}
        except (openai.APIError, httpx.HTTPError) as exc:
            return {**result, "source": f"no answer to the count request: {type(exc).__name__}", "server_message": None}
        stream.close()  # the server did not enforce the window: cut the reply off unread
        return {**result, "source": f"the server accepted max_tokens = {window}; the reply was cut off unread",
                "server_message": None}
    return {**result, "source": "the server named a different window twice", "server_message": None}


def sse_data(lines):
    """The ``data`` of each server-sent event in ``lines`` (one event per blank-line-ended block; comments and other
    fields skipped; several data lines joined by newlines, as the SSE format has it)."""
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if name == "data":
            data.append(value.removeprefix(" "))
    if data:
        yield "\n".join(data)


def read_stream(client, kwargs: dict, clock=time.monotonic) -> dict:
    """One streamed request read to its end, or to where it broke: everything that arrived, and how it ended.
    An HTTP error before the body (the server never answered) is raised; anything after it is recorded, never
    raised, so a partial reply is kept with its attempt."""
    content, reasoning = [], []
    out = {"done": False, "final_usage": False, "finish_reason": None, "models": [], "usage": None,
           "response_id": None, "reasoning_field": None, "transport_error": None, "error_event": None,
           "first_byte_s": None, "wall_s": None, "chunks": 0, "consumers": []}
    t0 = clock()
    last = None
    with client.chat.completions.with_streaming_response.create(**kwargs) as response:
        def timed(lines):
            for line in lines:
                if out["first_byte_s"] is None:
                    out["first_byte_s"] = round(clock() - t0, 3)
                yield line

        try:
            for data in sse_data(timed(response.iter_lines())):
                if data == "[DONE]":
                    out["done"] = True
                    break
                try:
                    ch = json.loads(data)
                except json.JSONDecodeError:
                    out["transport_error"] = "the stream carried an event that is not JSON"
                    break
                if not isinstance(ch, dict):
                    continue
                if ch.get("error"):  # the server failed after its 200
                    err = ch["error"]
                    out["error_event"] = str(err.get("message") if isinstance(err, dict) else err)[:300]
                    break
                out["chunks"] += 1
                last = ch
                out["response_id"] = out["response_id"] or ch.get("id")
                if ch.get("model") and ch["model"] not in out["models"]:
                    out["models"].append(ch["model"])
                consumer = {"node": ch.get("_consumer_node_name"), "slurm_job_id": ch.get("_consumer_slurm_job_id")}
                if any(consumer.values()) and consumer not in out["consumers"]:
                    out["consumers"].append(consumer)
                for choice in ch.get("choices") or []:
                    if choice.get("index", 0) != 0:
                        continue
                    delta = choice.get("delta") or {}
                    for field in ("reasoning_content", "reasoning"):  # vLLM's reasoning parser, by version
                        if delta.get(field):
                            reasoning.append(delta[field])
                            out["reasoning_field"] = field
                    if delta.get("content"):
                        content.append(delta["content"])
                    if choice.get("finish_reason") is not None:
                        out["finish_reason"] = choice["finish_reason"]
        except (httpx.HTTPError, openai.APIError, OSError) as exc:
            out["transport_error"] = f"the stream broke: {type(exc).__name__}: {str(exc)[:200]}"
    out["wall_s"] = round(clock() - t0, 3)
    # the usage that counts is the usage-only chunk ending the stream (the Gateway also sends running counts on chunks)
    if last is not None and not last.get("choices") and isinstance(last.get("usage"), dict):
        out["final_usage"], out["usage"] = True, last["usage"]
    elif last is not None and isinstance(last.get("usage"), dict):
        out["usage"] = last["usage"]  # a running count only: recorded, but the check needs the final one
    out["content"], out["reasoning"] = "".join(content), "".join(reasoning)
    return out


@functools.cache
def token_counter(model: str):
    """A function counting a text's tokens on ``model``'s own tokenizer, read offline from its pinned file (a special
    token written in the text, such as ``</think>``, counts as the one token the model emits). Refused, so that
    nothing is drawn unchecked, when the model has no pinned tokenizer, the file is missing or its bytes differ."""
    if model not in TOKENIZERS:
        raise RuntimeError(f"no pinned tokenizer for {model}: its draws cannot be re-tokenized, so none is sent")
    name, digest = TOKENIZERS[model]
    path = TOKENIZER_DIR / name
    if not path.exists():
        raise RuntimeError(f"no tokenizer file at {path}")
    data = gzip.decompress(path.read_bytes())
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError(f"{path} is not the pinned tokenizer (sha256 {digest})")
    tok = Tokenizer.from_str(data.decode("utf-8"))

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids) if text else 0

    count.tokenizer, count.sha256 = name.removesuffix(".gz"), digest
    return count


def retokenize(got: dict, count) -> dict:
    """The tokens a server counts for the delivered reply, were it whole: the content and reasoning exactly as they
    arrived, counted with ``count`` (``token_counter``), plus the template tokens it never delivers (the end-of-
    sentence token of a stop, the ``</think>`` ending reasoning sent in a reasoning field); with the offset from the
    reported completion tokens."""
    content = count(got["content"])
    reasoning = count(got["reasoning"])
    template = int(got["finish_reason"] == "stop") + int(bool(got["reasoning"] and got.get("reasoning_field")))
    out = {"tokenizer": count.tokenizer, "tokenizer_sha256": count.sha256, "content_tokens": content,
           "reasoning_tokens": reasoning, "template_tokens": template, "expected": content + reasoning + template,
           "band_pct": RETOKENIZED_BAND_PCT}
    tokens = (got.get("usage") or {}).get("completion_tokens")
    if tokens:
        out["offset"] = tokens - out["expected"]
        out["offset_pct"] = round(100 * out["offset"] / tokens, 1)
    return out


def retokenized_failure(retok: dict, tokens: int) -> str | None:
    """Why the re-tokenized reply is not within ``RETOKENIZED_BAND_PCT`` of ``tokens`` either way, or None."""
    off = tokens - retok["expected"]
    if abs(off) * 100 <= RETOKENIZED_BAND_PCT * tokens:  # integers: the edge is inside, exactly
        return None
    return (f"the delivered text re-tokenizes to {retok['expected']} tokens ({retok['content_tokens']} content, "
            f"{retok['reasoning_tokens']} reasoning, {retok['template_tokens']} template), {abs(off)} "
            f"{'fewer' if off > 0 else 'more'} than the {tokens} completion tokens ({100 * off / tokens:.1f}%, "
            f"outside {RETOKENIZED_BAND_PCT}%)")


def band_failure(chars: int, tokens: int) -> str | None:
    """Why ``chars`` delivered characters cannot be ``tokens`` completion tokens, or None."""
    low, high = CHARS_PER_TOKEN_BAND
    if low <= chars / tokens <= high:
        return None
    return (f"{chars} characters delivered for {tokens} completion tokens ({chars / tokens:.2f} per token, outside "
            f"{low} to {high})")


def integrity_failures(got: dict, model: str, counted: int | None, retok: dict) -> list[str]:
    """Why a read stream is not a whole reply (module docstring, "Integrity"); empty when it is. ``retok`` is the
    stream's ``retokenize``."""
    failed = []
    if got.get("transport_error"):
        failed.append(got["transport_error"])
    if got.get("error_event"):
        failed.append(f"the stream carried an error event: {got['error_event']}")
    if not got["done"]:
        failed.append("no [DONE]")
    if not got["final_usage"]:
        failed.append("no final usage chunk")
    if got["finish_reason"] != "stop":
        failed.append(f"finish_reason {got['finish_reason']}")
    served = got["models"][0] if len(got["models"]) == 1 else got["models"] or None
    if served != model:
        failed.append(f"model {served} differs from {model}")
    usage = got.get("usage") or {}
    read = usage.get("prompt_tokens")
    if read is None:  # an unreported count cannot show that the whole prompt was read
        failed.append("no prompt token count")
    elif counted is not None and read < counted:  # a proxy that cut the prompt still says "stop"
        failed.append(f"the server read {read} prompt tokens, fewer than the {counted} counted")
    tokens = usage.get("completion_tokens")  # without the final chunk, the last running count (failed above anyway)
    if not tokens:
        failed.append("no completion token count")
    else:
        chars = len(got["content"]) + len(got["reasoning"])
        failed += [why for why in (band_failure(chars, tokens), retokenized_failure(retok, tokens)) if why]
    return failed


def derive_reply(content: str) -> tuple[str, str, str]:
    """(reply, file suffix, derivation): the text the parse check reads, derived from the content channel, which is
    kept as delivered. A think block left inline (no reasoning parser on the server) is everything up to its first
    closing tag, even when the chat template opened it in the prompt; nothing else is removed."""
    if _THINK_END in content:
        return content.split(_THINK_END, 1)[1], ".reply.txt", REPLY_AFTER_THINK
    return content, ".txt", REPLY_AS_DELIVERED


def _attempt(out_dir: Path, index: int) -> int:
    """The attempt number of a new draw ``index``; every file of an earlier attempt is kept as
    draw-<i>.attempt-<k>.*."""
    attempt = 1
    while (out_dir / f"draw-{index}.attempt-{attempt}.json").exists():
        attempt += 1
    if (out_dir / f"draw-{index}.json").exists():
        for p in sorted(out_dir.glob(f"draw-{index}.*")):
            if not p.name.startswith(f"draw-{index}.attempt-"):
                p.rename(out_dir / p.name.replace(f"draw-{index}.", f"draw-{index}.attempt-{attempt}.", 1))
        attempt += 1
    return attempt


def _write_record(out_dir: Path, index: int, rec: dict) -> None:
    (out_dir / f"draw-{index}.json").write_text(json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n",
                                                encoding="utf-8")


def _no_reply(waited_s: float, status: int | None) -> dict:
    """What an attempt that ended before any byte records: how long it waited, and whether that was the gateway's
    cut (its 504, or no byte within its limit)."""
    return {"no_reply_after_s": round(waited_s, 3),
            "gateway_cut": status == 504 or waited_s >= GATEWAY_FIRST_BYTE_S}


def _write_raw(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))  # bytes: no newline translation, the text exactly as delivered


def _http_failure(exc: openai.APIStatusError) -> tuple[str, str]:
    """(kind, status) of a request the server refused before any reply."""
    msg = _message(exc)
    if exc.status_code == 400 and _OVERFLOW.search(msg):
        return "does_not_fit", f"failed: does not fit: {msg[:300]}"
    if exc.status_code in (401, 403):
        return "key", f"failed: key refused ({exc.status_code}): {msg[:300]}"
    if 400 <= exc.status_code < 500 and exc.status_code not in (408, 409, 429):
        return "request", f"failed: request refused ({exc.status_code}): {msg[:300]}"
    return "infrastructure", f"failed: no reply: server error {exc.status_code}: {msg[:300]}"


def draw(bundle: dict, out_dir: Path, client, index: int, model: str = QWEN, counted: int | None = None,
         timeout_s: float = DRAW_TIMEOUT_S, clock=time.monotonic, key_label: str | None = None) -> dict:
    """One attempt at draw ``index``: its content channel verbatim in draw-<i>.txt, its reasoning channel verbatim in
    draw-<i>.reasoning.txt, a derived reply in draw-<i>.reply.txt when one is derived, its record in draw-<i>.json,
    with the integrity check, then the parse check of the derived reply. ``counted`` is the server's prompt count of
    the same request (``count_input``)."""
    count_tokens = token_counter(model)
    body = gateway_request(bundle, model)
    seed = DRAW_SEEDS[index]
    attempt = _attempt(out_dir, index)
    rec = {
        "draw": index, "attempt": attempt, "seed": seed,
        "model_requested": model, "base_url": GATEWAY_BASE_URL, "key_file": key_label,
        "api": "OpenAI-compatible chat.completions, streamed, read as raw server-sent events",
        "parameters": {"max_tokens": MAX_TOKENS}, "not_sent": NOT_SENT, "thinking": THINKING,
        "request_sha256": gen._sha(body), "draw_request_sha256": gen._sha({**body, "seed": seed}),
        "bundle_request_sha256": bundle["request_sha256"], "user_sha256": _sha(bundle["user"]),
        "system_sha256": _sha(bundle["system"]),
        "input": "the Opus bundle's input/prompt-system.txt and input/prompt-user.txt, byte for byte",
        "use": "offline replay only (E4.3); never a prompt cell",
        "draw_rule": DRAW_RULE, "cost_usd": 0.0, "cost_basis": GRANT_BASIS, "started_utc": _utc(),
        "first_byte_s": None,
    }
    if "decontamination" in bundle:  # the map the input was renamed with: a draw is current only with it
        rec["map_sha256"] = bundle["decontamination"]["map_sha256"]
    # recorded before it is sent: an attempt the process never finished stays on disk as interrupted
    _write_record(out_dir, index, {**rec, "failure_kind": "interrupted",
                                   "status": "failed: interrupted: the process ended before this attempt was recorded"})
    content, reasoning, reply, suffix = "", "", "", ".txt"
    kwargs = {**body, "seed": seed, "stream": True, "stream_options": {"include_usage": True}, "timeout": timeout_s}
    t0 = clock()
    try:
        got = read_stream(client, kwargs, clock)
    except openai.APIStatusError as exc:
        rec["failure_kind"], rec["status"] = _http_failure(exc)
        rec.update(_no_reply(clock() - t0, exc.status_code))
    except (openai.APIError, httpx.HTTPError, OSError) as exc:
        rec["failure_kind"] = "infrastructure"
        rec["status"] = f"failed: no reply: {type(exc).__name__}: {str(exc)[:300]}"
        rec.update(_no_reply(clock() - t0, None))
    else:
        content, reasoning = got["content"], got["reasoning"]
        retok = retokenize(got, count_tokens)
        failed = integrity_failures(got, model, counted, retok)
        reply, suffix, derivation = derive_reply(content)
        turtle, turtle_from = gen.extract_turtle(reply)
        usage = got["usage"] or {}
        tokens = usage.get("completion_tokens")
        chars = len(content) + len(reasoning)
        rec["integrity"] = {"done": got["done"], "final_usage": got["final_usage"],
                            "finish_reason": got["finish_reason"], "completion_tokens": tokens,
                            "delivered_chars": chars, "chars_per_token": round(chars / tokens, 2) if tokens else None,
                            "band": list(CHARS_PER_TOKEN_BAND), "retokenized": retok, "failures": failed}
        read = usage.get("prompt_tokens")
        rec["prompt_tokens_check"] = "not reported by the server" if read is None else f"{read} read, {counted} counted"
        kind = "integrity" if failed else None
        if not failed:  # a draw is ok only if its reply is a non-empty ontology
            try:
                parsed = gen.parse_generated(turtle, f"generated-{index}", "check")
                if not parsed.classes and not parsed.properties:
                    failed, kind = ["the reply declares no class or property"], "parse"
            except Exception as exc:  # rdflib raises parser errors of many types
                failed, kind = [f"the reply is not a readable Turtle ontology: {str(exc)[:200]}"], "parse"
        served = got["models"][0] if len(got["models"]) == 1 else got["models"] or None
        rec.update({
            "model_returned": served, "response_id": got["response_id"], "finish_reason": got["finish_reason"],
            "usage": got["usage"], "output_sha256": _sha(content), "output_chars": len(content),
            "reply": {"file_suffix": suffix, "derivation": derivation, "chars": len(reply), "sha256": _sha(reply),
                      "turtle_from": turtle_from},
            "reasoning": {"chars": len(reasoning), "sha256": _sha(reasoning), "source": got["reasoning_field"] or "none"},
            "first_byte_s": got["first_byte_s"], "wall_s": got["wall_s"], "chunks": got["chunks"],
            "consumers": got["consumers"], "failure_kind": kind,
            "gateway_cut": got["first_byte_s"] is None and got["wall_s"] >= GATEWAY_FIRST_BYTE_S,
            "status": f"failed: {kind}: " + "; ".join(failed) if failed else "ok",
        })
    rec["ended_utc"] = _utc()
    _write_raw(out_dir / f"draw-{index}.txt", content)
    if reasoning:
        _write_raw(out_dir / f"draw-{index}.reasoning.txt", reasoning)
    if suffix != ".txt":
        _write_raw(out_dir / f"draw-{index}{suffix}", reply)
    _write_record(out_dir, index, rec)
    return rec


def _record(out_dir: Path, index: int) -> dict | None:
    p = out_dir / f"draw-{index}.json"
    return json.loads(p.read_text()) if p.exists() else None


def attempt_records(out_dir: Path, index: int) -> list[dict]:
    """Every attempt at draw ``index``, in order: the kept ones, then the current one."""
    recs, k = [], 1
    while (p := out_dir / f"draw-{index}.attempt-{k}.json").exists():
        recs.append(json.loads(p.read_text()))
        k += 1
    if cur := _record(out_dir, index):
        recs.append(cur)
    return recs


def attempt_summary(out_dir: Path) -> dict:
    """Per draw and in total: attempts, redraws (an attempt after an integrity or parse failure), re-sends (an
    attempt after one the server never answered or the process never finished) and attempts cut at the gateway;
    and why the bundle stopped as infeasible, if it did; for posting with the draws."""
    draws, totals = {}, {"draws_ok": 0, "attempts": 0, "redraws": 0, "resends": 0, "gateway_cuts": 0}
    for i in DRAW_SEEDS:
        recs = attempt_records(out_dir, i)
        if not recs:
            continue
        followed = [r.get("failure_kind") for r in recs[:-1]]  # every earlier attempt failed and was followed
        d = {"attempts": len(recs), "status": recs[-1]["status"],
             "redraws": sum(k in REDRAW_KINDS for k in followed), "resends": sum(k in RESEND_KINDS for k in followed),
             "gateway_cuts": sum(bool(r.get("gateway_cut")) for r in recs),
             "failed_attempts": [{"attempt": r["attempt"], "kind": r.get("failure_kind"), "status": r["status"]}
                                 for r in recs if r["status"] != "ok"]}
        draws[str(i)] = d
        totals["draws_ok"] += d["status"] == "ok"
        for key in ("attempts", "redraws", "resends", "gateway_cuts"):
            totals[key] += d[key]
    infeasible = out_dir / INFEASIBLE_FILE
    return {"draws": draws, "totals": totals,
            "infeasible": json.loads(infeasible.read_text()) if infeasible.exists() else None}


def _stop_reason(out_dir: Path, index: int, max_attempts: int) -> str | None:
    """Why draw ``index`` must not be attempted again, or None."""
    recs = attempt_records(out_dir, index)
    if not recs:
        return None
    if recs[-1].get("failure_kind") in STOP_KINDS:
        return f"draw {index}: {recs[-1]['status']}; the identical request is not sent again"
    if len(recs) >= max_attempts:
        return f"draw {index} failed {len(recs)} attempts, the most allowed ({max_attempts}); the run stops"
    if len(recs) >= 2:
        a, b = recs[-2], recs[-1]
        if (a.get("failure_kind") in REDRAW_KINDS and b.get("failure_kind") in REDRAW_KINDS
                and a.get("output_chars") and a.get("output_sha256") == b.get("output_sha256")
                and (a.get("reasoning") or {}).get("sha256") == (b.get("reasoning") or {}).get("sha256")):
            return (f"draw {index}: attempts {a['attempt']} and {b['attempt']} returned the identical failing reply; "
                    "the identical request cannot change it, and the run stops")
    return None


def _gateway_stop(out_dir: Path, model: str, request_sha: str, bundle: dict) -> str | None:
    """When the bundle's attempts cut at the gateway reach ``GATEWAY_CUTS_INFEASIBLE``: why the bundle stops as
    infeasible, written to ``INFEASIBLE_FILE``; else None."""
    cuts = [r for i in DRAW_SEEDS for r in attempt_records(out_dir, i) if r.get("gateway_cut")]
    if len(cuts) < GATEWAY_CUTS_INFEASIBLE:
        return None
    which = ", ".join(f"draw {r['draw']} attempt {r['attempt']}" for r in cuts)
    reason = f"{len(cuts)} attempts cut at the gateway before any byte ({which}): infeasible for {model} on this gateway"
    (out_dir / INFEASIBLE_FILE).write_text(json.dumps({
        "reason": reason, "rule": GATEWAY_RULE, "model": model, "request_sha256": request_sha,
        "bundle_request_sha256": bundle["request_sha256"], "stopped_utc": _utc(),
        "attempts": [{k: r.get(k) for k in ("draw", "attempt", "status", "no_reply_after_s", "first_byte_s")}
                     for r in cuts],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return reason


def _paused(pause_file: Path | None, log) -> bool:
    if pause_file is not None and pause_file.exists():
        log(f"paused: {pause_file} exists; remove it and run again to resume")
        return True
    return False


def run_gateway_draws(bundle: dict, out_dir: Path, client, model: str = QWEN, window: int | None = None,
                    log=print, timeout_s: float = DRAW_TIMEOUT_S, max_attempts: int = MAX_ATTEMPTS,
                    pause_file: Path | None = None, max_first_byte_s: float | None = None, sleep=time.sleep,
                    clock=time.monotonic, key_label: str | None = None) -> list[dict]:
    """Count, check the fit, then draws 1, 2 and 3 in order, each until it is ok under the draw rule or the run
    stops (module docstring). Returns this run's attempts. A directory holding another request's draws (another
    bundle or model) is refused, and so is a model without a pinned tokenizer; a bundle stopped as infeasible is
    never sent again."""
    drawable, why = gen.is_drawable(bundle)
    if not drawable:
        raise RuntimeError(why)
    request_sha = gen._sha(gateway_request(bundle, model))
    out_dir.mkdir(parents=True, exist_ok=True)
    if other := gen.draws_request_sha(out_dir) - {request_sha}:
        raise RuntimeError(f"{out_dir} holds draws of another request ({sorted(other)}); retire them first")
    for i in DRAW_SEEDS:
        rec = _record(out_dir, i)
        if rec and str(rec.get("status", "")).startswith("failed: does not fit"):
            raise RuntimeError(f"draw {i} does not fit the context window; it is never redrawn with less input")
    token_counter(model)  # refused here, before any request, when the draws could not be re-tokenized
    if (out_dir / INFEASIBLE_FILE).exists():
        reason = json.loads((out_dir / INFEASIBLE_FILE).read_text())["reason"]
        log(f"{out_dir}: stopped as infeasible ({reason}); this bundle is not sent again")
        return []
    if _paused(pause_file, log):
        return []
    state_path = out_dir / COUNT_FILE
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state is None or state["count"]["fits"] is None:  # an unknown fit is counted again; a known one is final
        state = {"request_sha256": request_sha, "model": model, "bundle_request_sha256": bundle["request_sha256"],
                 "counted_utc": _utc(), "count": count_input(bundle, client, model, window)}
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    count = state["count"]
    log(f"input {count}")
    if count["fits"] is None:
        raise RuntimeError(f"the fit is unknown ({count['source']}); nothing is drawn")
    if not count["fits"]:
        raise RuntimeError(f"the input does not fit the context window with the {MAX_TOKENS}-token budget: {count}")
    records: list[dict] = []
    for i in DRAW_SEEDS:
        while True:
            prev = _record(out_dir, i)
            if prev and prev.get("status") == "ok":
                break
            if stop := _stop_reason(out_dir, i, max_attempts):
                log(stop)
                return records
            if _paused(pause_file, log):
                return records
            if prev is not None:  # an attempt follows a failed one: give a passing outage or load time to clear
                sleep(RETRY_WAIT_S)
                if _paused(pause_file, log):
                    return records
            rec = draw(bundle, out_dir, client, i, model, count["input_tokens"], timeout_s, clock, key_label)
            records.append(rec)
            log(f"draw {i} attempt {rec['attempt']}: {rec['status']} (first byte {rec['first_byte_s']} s)")
            first_byte = rec.get("first_byte_s")
            if max_first_byte_s is not None and first_byte is not None and first_byte > max_first_byte_s:
                reason = (f"draw {i} attempt {rec['attempt']}: first byte {first_byte} s over the "
                          f"{max_first_byte_s} s threshold")
                if pause_file is not None:
                    pause_file.parent.mkdir(parents=True, exist_ok=True)
                    pause_file.write_text(json.dumps({"paused_utc": _utc(), "reason": reason}) + "\n",
                                          encoding="utf-8")
                log(f"paused: {reason}; remove {pause_file} and run again to resume")
                return records
            if reason := _gateway_stop(out_dir, model, request_sha, bundle):
                log(f"{reason}; this bundle stops, and the run goes on with the next bundle")
                return records
            if rec["status"] == "ok":
                break
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Gateway draws of the Opus generator recipe (E4.3 self-check crossing), one bundle per run.",
        epilog=f"exit 0: all three draws ok; {EXIT_INFEASIBLE}: this bundle stopped as infeasible on the gateway (go "
               f"on with the next bundle); {EXIT_STOPPED}: the run stops (paused, a stop rule, or a draw failed)")
    ap.add_argument("--bundle-dir", type=Path, required=True,
                    help="the Opus bundle directory, e.g. data/sources/rhea-142/generated/")
    ap.add_argument("--out", type=Path, help="default: <bundle-dir>-<model>, e.g. generated-deepseek-v4-flash-0731/")
    # no default: a run that forgot it would draw the UniProt bundles, which fit Qwen's window too, with Qwen
    ap.add_argument("--model", required=True, choices=sorted(CONTEXT_WINDOWS))
    ap.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    ap.add_argument("--count-only", action="store_true", help="count the input and check the fit; no draw")
    ap.add_argument("--summary", action="store_true", help="print the attempts, redraws and re-sends; no request")
    ap.add_argument("--pause-file", type=Path, default=DEFAULT_PAUSE_FILE,
                    help="no request is sent while this file exists")
    ap.add_argument("--max-first-byte-s", type=float,
                    help="pause after an attempt whose first byte took longer (required to draw: set per bundle)")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    args = ap.parse_args(argv)
    out = args.out or default_out_dir(args.bundle_dir, args.model)
    if args.summary:
        print(json.dumps(attempt_summary(out), indent=2, sort_keys=True))
        return 0
    if not args.count_only and args.max_first_byte_s is None:
        ap.error("drawing needs --max-first-byte-s (the first byte includes the bundle's prefill: set it per bundle)")
    bundle = gen.load_bundle(args.bundle_dir)  # checked against its request hash before anything is sent
    client = gateway_client(args.key_file)
    if args.count_only:
        print(json.dumps(count_input(bundle, client, args.model), sort_keys=True))
        return 0
    run_gateway_draws(bundle, out, client, args.model, log=lambda m: print(m, flush=True),
                    max_attempts=args.max_attempts, pause_file=args.pause_file,
                    max_first_byte_s=args.max_first_byte_s, key_label=args.key_file.name)
    summary = attempt_summary(out)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if summary["infeasible"]:
        return EXIT_INFEASIBLE
    return 0 if summary["totals"]["draws_ok"] == len(DRAW_SEEDS) else EXIT_STOPPED


if __name__ == "__main__":
    sys.exit(main())
