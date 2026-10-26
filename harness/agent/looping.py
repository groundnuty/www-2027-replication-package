"""Detecting an agent that loops instead of converging.

The rule, applied to the tool calls the model issued in an attempt (every entry
of the record's ``tool_calls``, in order, whatever its status):

  A. some SPARQL query is issued REPEAT_THRESHOLD (3) or more times, comparing
     queries after normalisation (runs of whitespace collapsed to one space,
     leading and trailing whitespace removed; case is kept, because IRIs and
     literals are case-sensitive); or
  B. the last WINDOW (4) tool calls contain at most MAX_DISTINCT (2) distinct
     calls, a call being its tool name plus its arguments (canonical JSON, with
     the SPARQL query normalised as in A).

``looping`` is A or B. The loop records the verdict only for capped attempts,
and the evidence for every attempt.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

REPEAT_THRESHOLD = 3
WINDOW = 4
MAX_DISTINCT = 2

_WS = re.compile(r"\s+")


def normalise_query(query: str) -> str:
    return _WS.sub(" ", query).strip()


def call_identity(call: dict) -> str:
    args = call.get("arguments")
    if isinstance(args, dict) and call.get("name") == "sparql_query" and isinstance(args.get("query"), str):
        args = {**args, "query": normalise_query(args["query"])}
    if args is None:
        args = {"arguments_raw": call.get("arguments_raw")}
    return json.dumps([call.get("name"), args], sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def loop_evidence(calls: list[dict]) -> dict[str, Any]:
    queries = Counter(
        normalise_query(c["arguments"]["query"])
        for c in calls
        if c.get("name") == "sparql_query" and isinstance(c.get("arguments"), dict)
        and isinstance(c["arguments"].get("query"), str)
    )
    max_repeats = max(queries.values(), default=0)
    window = [call_identity(c) for c in calls[-WINDOW:]] if len(calls) >= WINDOW else None
    distinct = len(set(window)) if window is not None else None
    repeated = max_repeats >= REPEAT_THRESHOLD
    cycling = distinct is not None and distinct <= MAX_DISTINCT
    return {
        "max_query_repeats": max_repeats,
        "last_window_distinct": distinct,
        "repeated_query": repeated,
        "cycling_window": cycling,
        "rule": f"query repeated >= {REPEAT_THRESHOLD} times, or last {WINDOW} calls over <= {MAX_DISTINCT} distinct",
    }


def is_looping(calls: list[dict]) -> bool:
    evidence = loop_evidence(calls)
    return evidence["repeated_query"] or evidence["cycling_window"]
