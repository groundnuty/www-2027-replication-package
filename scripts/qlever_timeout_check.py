#!/usr/bin/env python3
"""QLever time-out protocol: does a query that hits its time-out come back as an
explicit error, and does a query that finishes near the cap come back complete?

Uncommitted check script (design log question). Every query goes through two
paths: the harness client (harness/sparql.py, the code the agents and checks
use) and a raw HTTP/1.1 socket reader that records exactly what the server sent
(status, headers, chunked framing, the body's head and tail, how the connection
ended). One query is in flight at a time; a query the client abandons is given
until its server-side cap has passed before the next one is sent.

    uv run python scripts/qlever_timeout_check.py stats  --store rhea7039
    uv run python scripts/qlever_timeout_check.py time   --store rhea7039 --cap 60 'SELECT ...'
    uv run python scripts/qlever_timeout_check.py heavy  --store rhea7039 --out DIR
    uv run python scripts/qlever_timeout_check.py window --store rhea7039 --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.sparql import SPARQL_JSON, SparqlClient, SparqlError

QLEVER_JSON = "application/qlever-results+json"
# What QLever 9bebc60 appends to a streamed body when the export fails after
# the 200 status line has gone out (seen on the time-out).
MARKER = b"!!!!>># An error has occurred while exporting the query result"

STORES = {
    "uniprot7020": ("localhost", 7020),
    "rhea7039": ("localhost", 7039),
    "uniprot7022": ("localhost", 7022),
}


# --- raw HTTP ---------------------------------------------------------------

@dataclass
class Raw:
    status: int | None
    headers: dict[str, str]
    t_headers: float | None
    t_first_body: float | None
    t_end: float
    end_reason: str  # "eof", "reset", "sock_timeout", "wall_limit", "byte_cap"
    framing: str  # "chunked", "length", "close"
    framing_complete: bool  # final zero chunk seen / Content-Length reached
    body: bytes

    def summary(self, keep_body: bool = False) -> dict[str, Any]:
        body = self.body
        marker_at = body.find(MARKER)
        out: dict[str, Any] = {
            "status": self.status,
            "content_type": self.headers.get("content-type"),
            "transfer_encoding": self.headers.get("transfer-encoding"),
            "t_headers_s": _r(self.t_headers),
            "t_first_body_s": _r(self.t_first_body),
            "t_end_s": _r(self.t_end),
            "end_reason": self.end_reason,
            "framing": self.framing,
            "framing_complete": self.framing_complete,
            "body_bytes": len(body),
            "marker": marker_at >= 0,
        }
        if marker_at >= 0:
            tail = body[marker_at:].decode("utf-8", "replace")
            out["marker_message"] = tail.split("The error message was:", 1)[-1].strip()[:300]
            out["approx_rows_before_marker"] = _approx_rows(body[:marker_at])
            out["bytes_before_marker"] = marker_at
        parsed = _try_json(body)
        out["json_valid"] = parsed is not None
        if isinstance(parsed, dict):
            if "exception" in parsed:
                out["exception"] = str(parsed["exception"])[:300]
            bindings = parsed.get("results", {}).get("bindings") if isinstance(parsed.get("results"), dict) else None
            if isinstance(bindings, list):
                out["rows"] = len(bindings)
            if "res" in parsed:
                out["rows"] = len(parsed["res"])
            ri = parsed.get("runtimeInformation")
            if isinstance(ri, dict) and isinstance(ri.get("query_execution_tree"), dict):
                ri = ri["query_execution_tree"]
            if isinstance(ri, dict):
                out["root_cache_status"] = ri.get("cache_status")
                out["cache_statuses"] = sorted(_cache_statuses(ri))
            if isinstance(parsed.get("time"), dict):
                out["server_time_ms"] = parsed["time"]
        elif parsed is None and marker_at < 0 and self.status == 200:
            out["tail"] = body[-200:].decode("utf-8", "replace")
        if self.status != 200 or not out["json_valid"]:
            out["head"] = body[:240].decode("utf-8", "replace")
        if keep_body:
            out["_parsed"] = parsed
        return out


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 3)


def _approx_rows(prefix: bytes) -> int:
    # Bindings in SPARQL JSON are separated by "}},{" (end of the last term,
    # end of the binding); the first binding follows '"bindings":['.
    if b'"bindings":[{' not in prefix:
        return 0
    return prefix.count(b"}},{") + 1


def _try_json(body: bytes) -> Any:
    if not body or len(body) > 400 * 2**20:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


def _cache_statuses(node: dict) -> set[str]:
    out = {f"{node.get('description', '?')[:60]}: {node.get('cache_status')}"}
    for child in node.get("children") or []:
        out |= _cache_statuses(child)
    return out


def raw_post(host: str, port: int, query: str, *, timeout_param: str | None, accept: str = SPARQL_JSON,
             sock_timeout: float = 60.0, wall_limit: float | None = None,
             byte_cap: int = 300 * 2**20) -> Raw:
    """POST the query exactly as the harness client does, and record the reply as sent."""
    path = "/" + (f"?timeout={timeout_param}" if timeout_param else "")
    payload = query.encode("utf-8")
    request = (f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/sparql-query\r\n"
               f"Accept: {accept}\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode() + payload
    start = time.monotonic()
    buf = bytearray()
    t_headers = t_first_body = None
    header_end = -1
    end_reason = "eof"
    sock = socket.create_connection((host, port), timeout=sock_timeout)
    try:
        sock.sendall(request)
        while True:
            if wall_limit is not None and time.monotonic() - start > wall_limit:
                end_reason = "wall_limit"
                break
            try:
                chunk = sock.recv(1 << 20)
            except TimeoutError:
                end_reason = "sock_timeout"
                break
            except ConnectionResetError:
                end_reason = "reset"
                break
            if not chunk:
                break
            buf += chunk
            if header_end < 0:
                header_end = buf.find(b"\r\n\r\n")
                if header_end >= 0:
                    t_headers = time.monotonic() - start
                    if len(buf) > header_end + 4:
                        t_first_body = t_headers
            elif t_first_body is None:
                t_first_body = time.monotonic() - start
            if len(buf) > byte_cap:
                end_reason = "byte_cap"
                break
    finally:
        sock.close()
    t_end = time.monotonic() - start
    if header_end < 0:
        return Raw(None, {}, None, None, t_end, end_reason, "none", False, bytes(buf))
    head = bytes(buf[:header_end]).decode("latin-1").split("\r\n")
    status = int(head[0].split()[1])
    headers = {}
    for line in head[1:]:
        k, _, v = line.partition(":")
        headers[k.strip().lower()] = v.strip()
    rest = bytes(buf[header_end + 4:])
    if headers.get("transfer-encoding", "").lower() == "chunked":
        body, complete = _dechunk(rest)
        framing = "chunked"
    elif "content-length" in headers:
        n = int(headers["content-length"])
        body, complete, framing = rest[:n], len(rest) >= n, "length"
    else:
        body, complete, framing = rest, end_reason == "eof", "close"
    return Raw(status, headers, t_headers, t_first_body, t_end, end_reason, framing, complete, body)


def _dechunk(data: bytes) -> tuple[bytes, bool]:
    out = bytearray()
    i = 0
    while True:
        j = data.find(b"\r\n", i)
        if j < 0:
            return bytes(out), False
        size_text = data[i:j].split(b";")[0].strip()
        try:
            size = int(size_text, 16)
        except ValueError:
            return bytes(out), False
        if size == 0:
            return bytes(out), True
        if j + 2 + size > len(data):
            out += data[j + 2:]
            return bytes(out), False
        out += data[j + 2:j + 2 + size]
        i = j + 2 + size + 2


# --- the harness client -------------------------------------------------------

class _ForcedHttpTimeout(httpx.Client):
    """An httpx client whose per-request time-out is forced, to separate the
    HTTP-level time-out from the server's `timeout` parameter."""

    def __init__(self, http_timeout: float):
        super().__init__()
        self._forced = http_timeout

    def stream(self, *args, **kwargs):  # type: ignore[override]
        kwargs["timeout"] = self._forced
        return super().stream(*args, **kwargs)


def client_run(endpoint: str, query: str, cap: float, http_timeout: float | None = None) -> tuple[dict, dict | None]:
    http = _ForcedHttpTimeout(http_timeout) if http_timeout is not None else None
    client = SparqlClient(endpoint, timeout_s=cap, row_cap=None, http=http)
    start = time.monotonic()
    try:
        res = client.query(query)
    except SparqlError as exc:
        return ({"outcome": "error", "kind": exc.kind, "status": exc.status, "message": str(exc)[:300],
                 "elapsed_s": round(time.monotonic() - start, 3)}, None)
    bindings = res.raw.get("results", {}).get("bindings") if res.form == "SELECT" else None
    return ({"outcome": "rows", "rows": res.total_rows, "truncated": res.truncated,
             "elapsed_s": round(res.elapsed_s, 3), "boolean": res.boolean}, {"bindings": bindings})


def canon(bindings: list[dict]) -> list[str]:
    return [json.dumps(b, sort_keys=True, ensure_ascii=False) for b in bindings]


def digest(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def compare(ref: list[dict], got: list[dict]) -> dict:
    a, b = canon(ref), canon(got)
    return {"ref_rows": len(a), "rows": len(b), "ordered_equal": a == b, "multiset_equal": sorted(a) == sorted(b),
            "ref_digest": digest(a), "digest": digest(b), "sorted_digest_equal": digest(sorted(a)) == digest(sorted(b))}


def stats(host: str, port: int) -> dict:
    r = httpx.get(f"http://{host}:{port}/", params={"cmd": "stats"}, timeout=20)
    r.raise_for_status()
    return r.json()


# --- logging ------------------------------------------------------------------

class Log:
    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rec: dict) -> None:
        rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec}
        print(json.dumps(rec, ensure_ascii=False)[:1500], flush=True)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def nonce() -> str:
    return "nonce-" + uuid.uuid4().hex[:12]


def settle(sent_at: float, server_cap_s: float, margin: float = 3.0) -> None:
    """Wait until a possibly abandoned query's server-side cap has passed."""
    wait = sent_at + server_cap_s + margin - time.monotonic()
    if wait > 0:
        time.sleep(wait)


# --- query sets ---------------------------------------------------------------

UP = "PREFIX up: <http://purl.uniprot.org/core/> "
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OBO = "http://www.geneontology.org/formats/oboInOwl#"
CHEMROF = "https://w3id.org/chemrof/"
VALUES4 = 'VALUES ?v { "H2O" "ATP" "NAD(+)" "CoA" } ?s ?p ?x FILTER(STR(?x) = ?v)'
AUTH2 = 'VALUES ?v { "Smith J." "Zhang Y." } ?c up:author ?a FILTER(STR(?a) = ?v)'

# (id, cap_s, repeats, kind, query). "materialising": an operator that needs its
# whole input (sort, aggregate, distinct, ASK) sits on top, so nothing can be
# exported before the cap. "streaming": rows can leave the server while the
# rest of the query still runs. Caps of 1-2 s keep the dense streams small in bytes.
HEAVY = {
    "rhea": [
        ("RH1_values_str_orderby", 10, 1, "materialising", f"SELECT ?s ?p ?v WHERE {{ {VALUES4} }} ORDER BY ?s ?p ?v"),
        ("RH2_values_str_count", 10, 1, "materialising", f"SELECT (COUNT(*) AS ?n) WHERE {{ {VALUES4} }}"),
        ("RH3_values_str_groupby", 10, 1, "materialising", f"SELECT ?v (COUNT(*) AS ?n) WHERE {{ {VALUES4} }} GROUP BY ?v"),
        ("RH4_values_str_distinct", 10, 1, "materialising", f"SELECT DISTINCT ?p WHERE {{ {VALUES4} }}"),
        ("RH5_ask_nomatch", 10, 1, "materialising",
         'ASK { VALUES ?v { "zzqq1" "zzqq2" "zzqq3" "zzqq4" } ?s ?p ?x FILTER(STR(?x) = ?v) }'),
        ("RH6_values_str_stream", 10, 3, "streaming", f"SELECT ?s ?p ?v WHERE {{ {VALUES4} }}"),
        ("RH7_values1_sparse_stream", 10, 3, "streaming",
         'SELECT ?s ?p WHERE { VALUES ?v { "H2O" } ?s ?p ?x FILTER(STR(?x) = ?v) }'),
        ("RH8_agent_str_eq_limit20", 10, 3, "streaming",
         'SELECT ?s ?p ?o WHERE { ?s ?p ?o . FILTER(isLiteral(?o) && str(?o)="Mn(2+)") } LIMIT 20'),
        ("RH9_agent_contains_nomatch_limit20", 1, 2, "streaming",
         'SELECT ?s ?p ?o WHERE { ?s ?p ?o FILTER(CONTAINS(STR(?o), "zzqqxx")) } LIMIT 20'),
        ("RH10_dense_stream", 2, 3, "streaming", 'SELECT ?s ?p ?o WHERE { ?s ?p ?o FILTER(CONTAINS(STR(?o), "a")) }'),
        ("RH11_dense_stream_limit", 2, 2, "streaming",
         'SELECT ?s ?p ?o WHERE { ?s ?p ?o FILTER(CONTAINS(STR(?o), "a")) } LIMIT 3000000'),
        # The slow equality-against-VALUES plan with frequent matches: the first
        # rows are ready only a few seconds in, then the cap fires mid-export.
        ("RH12_slow_plan_stream", 10, 3, "streaming",
         'SELECT ?s ?p ?x WHERE { VALUES ?v { "H2O" } ?s ?p ?x FILTER(STR(?x) = ?v || CONTAINS(STR(?x), "ase")) }'),
        ("RH13_slow_plan_stream_limit", 10, 2, "streaming",
         ('SELECT ?s ?p ?x WHERE { VALUES ?v { "H2O" } ?s ?p ?x FILTER(STR(?x) = ?v || CONTAINS(STR(?x), "ase")) }'
          ' LIMIT 100000')),
    ],
    "uniprot": [
        ("UH1_values_str_orderby", 10, 1, "materialising", f"{UP}SELECT ?c ?a WHERE {{ {AUTH2} }} ORDER BY ?c ?a"),
        ("UH2_groupby_scan", 10, 1, "materialising",
         'SELECT ?p (COUNT(*) AS ?c) WHERE { ?s ?p ?o FILTER(STRSTARTS(STR(?o), "http://purl.uniprot.org/taxonomy/")) } GROUP BY ?p'),
        ("UH3_count_regex_scan", 10, 1, "materialising",
         'SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o FILTER(REGEX(STR(?o), "zzqq", "i")) }'),
        ("UH4_distinct_p_scan", 10, 1, "materialising",
         'SELECT DISTINCT ?p WHERE { ?s ?p ?o . FILTER(CONTAINS(LCASE(STR(?o)), "keyword")) }'),
        ("UH5_ask_nomatch", 10, 1, "materialising", 'ASK { ?s ?p ?o FILTER(CONTAINS(STR(?o), "zzqqxx")) }'),
        ("UH6_values_str_stream", 10, 3, "streaming", f"{UP}SELECT ?c ?a WHERE {{ {AUTH2} }}"),
        ("UH7_contains_sparse_stream", 10, 3, "streaming",
         'SELECT ?s ?p ?o WHERE { ?s ?p ?o FILTER(CONTAINS(STR(?o), "Wolbachia")) }'),
        ("UH8_agent_str_eq_limit20", 10, 3, "streaming",
         'SELECT ?s ?p ?o WHERE { ?s ?p ?o FILTER(str(?o) = "Zinc import ATP-binding protein ZnuC") } LIMIT 20'),
        ("UH9_dense_stream", 1, 3, "streaming", "SELECT ?s ?p ?o WHERE { ?s ?p ?o FILTER(isLiteral(?o)) }"),
        ("UH10_dense_stream_limit", 1, 2, "streaming",
         "SELECT ?s ?p ?o WHERE { ?s ?p ?o FILTER(isLiteral(?o)) } LIMIT 3000000"),
        ("UH12_slow_plan_stream", 10, 3, "streaming",
         UP + 'SELECT ?c ?a WHERE { VALUES ?v { "Smith J." } ?c up:author ?a FILTER(STR(?a) = ?v || CONTAINS(?a, "Zh")) }'),
        ("UH13_slow_plan_stream_limit", 10, 2, "streaming",
         (UP + 'SELECT ?c ?a WHERE { VALUES ?v { "Smith J." } ?c up:author ?a FILTER(STR(?a) = ?v || CONTAINS(?a, "Zh")) }'
          ' LIMIT 100000')),
    ],
}
# A shape recorded overrunning its own cap (cal-uniprot-r1 qwen record 7cef9dec,
# tool call 31: timeout=60s, no answer by the 70 s HTTP backstop, twice).
OVERRUN = ("UH11_recorded_overrun", 10,
           ("SELECT DISTINCT ?p WHERE {\n  ?p a <http://purl.uniprot.org/core/Protein> .\n  ?p ?p2 ?o .\n}"
            " GROUP BY ?p ORDER BY (COUNT(?p2)) LIMIT 1"))
# The HTTP-level time-out against the server's: server cap 20 s, HTTP 5 s.
BACKSTOP = {"rhea": "RH1_values_str_orderby", "uniprot": "UH1_values_str_orderby"}

# Window queries: (id, ordered, template). "{nonce}" sits in a FILTER on a leaf
# variable, so each run has its own cache key from that scan upwards. Where an
# expensive filter tests the same variable, the nonce goes into that filter's
# expression ("... && ?o != nonce"): as a FILTER of its own the planner put it
# above the expensive one, whose result was then served from cache. Ordered
# means ORDER BY over every projected column (a total order), so rows are
# compared in order, else as a multiset.
WINDOW = {
    "rhea": [
        ("R01b_count_contains_scan", True, 'SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o FILTER(CONTAINS(STR(?o), "ase") && ?o != "{nonce}") }'),
        ("R02_group_p_strlen", True, 'SELECT ?p (COUNT(*) AS ?n) (MAX(STRLEN(STR(?o))) AS ?m) WHERE { ?s ?p ?o FILTER(?o != "{nonce}") } GROUP BY ?p ORDER BY ?p'),
        ("R03_sort_scan_limit300k", True, 'SELECT ?s ?p ?o WHERE { ?s ?p ?o FILTER(?o != "{nonce}") } ORDER BY ?s ?p ?o LIMIT 300000'),
        ("R04_lazy_export_500k", False, 'SELECT ?s ?p ?o WHERE { ?s ?p ?o FILTER(?o != "{nonce}") } LIMIT 500000'),
        ("R07_distinct_p_literal", True, 'SELECT DISTINCT ?p ?o WHERE { ?s ?p ?o FILTER(isLiteral(?o)) FILTER(?o != "{nonce}") } ORDER BY ?p ?o LIMIT 200000'),
        ("R08_group_syn_minmax", True, f'SELECT ?s (COUNT(?o) AS ?n) (MIN(STR(?o)) AS ?lo) (MAX(STR(?o)) AS ?hi) WHERE {{ ?s <{OBO}hasSynonym> ?o FILTER(?o != "{{nonce}}") }} GROUP BY ?s ORDER BY ?s LIMIT 100000'),
        ("R11_ucase_sort_labels", True, f'SELECT ?s (UCASE(STR(?l)) AS ?u) WHERE {{ ?s <{RDFS}label> ?l FILTER(?l != "{{nonce}}") }} ORDER BY ?u ?s LIMIT 150000'),
        ("R19_optional_formula", True, f'SELECT ?c ?l ?f WHERE {{ ?c <{RDFS}label> ?l OPTIONAL {{ ?c <{CHEMROF}generalized_empirical_formula> ?f }} FILTER(?l != "{{nonce}}") }} ORDER BY ?c ?l ?f LIMIT 250000'),
        ("R20_datatype_group_scan", True, 'SELECT ?dt (COUNT(*) AS ?n) WHERE { ?s ?p ?o FILTER(isLiteral(?o)) BIND(DATATYPE(?o) AS ?dt) FILTER(?o != "{nonce}") } GROUP BY ?dt ORDER BY ?dt'),
        ("R21_chebi_iri_group_scan", True, 'SELECT ?p (COUNT(*) AS ?n) WHERE { ?s ?p ?o FILTER(isIRI(?o) && STRSTARTS(STR(?o), "http://purl.obolibrary.org/obo/CHEBI_")) FILTER(?o != <urn:x-{nonce}>) } GROUP BY ?p ORDER BY ?p'),
        ("R24_join_syn_xref_export", True, f'SELECT ?c ?syn ?x WHERE {{ ?c <{OBO}hasSynonym> ?syn . ?c <{OBO}hasDbXref> ?x FILTER(?x != "{{nonce}}") }} ORDER BY ?c ?syn ?x LIMIT 400000'),
        ("R25b_regex_scan_count", True, 'SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o FILTER(REGEX(STR(?o), "ph?osph", "i") && ?o != "{nonce}") }'),
        ("R26b_lcase_scan_rows", True, 'SELECT ?s ?p ?o WHERE { ?s ?p ?o FILTER(CONTAINS(LCASE(STR(?o)), "glucos") && ?o != "{nonce}") } ORDER BY ?s ?p ?o'),
        ("R27b_regex_lcase_scan_rows", True, 'SELECT ?s ?p ?o WHERE { ?s ?p ?o FILTER(REGEX(LCASE(STR(?o)), "a.*e.*i.*o.*u") && ?o != "{nonce}") } ORDER BY ?s ?p ?o'),
        ("R28_values1_scan", True, 'SELECT ?s ?p WHERE { VALUES ?v { "H2O" } ?s ?p ?x FILTER(STR(?x) = ?v) FILTER(?x != "{nonce}") } ORDER BY ?s ?p'),
        ("R30_sort_by_strlen", True, 'SELECT ?s ?p ?o (STRLEN(STR(?o)) AS ?len) WHERE { ?s ?p ?o FILTER(?o != "{nonce}") } ORDER BY DESC(?len) ?s ?p ?o LIMIT 100000'),
        ("R31_regex_alt_count", True, 'SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o FILTER(REGEX(STR(?o), "(a|e)(b|c|d)(e|i).*[0-9]{2}", "i")) FILTER(?o != "{nonce}") }'),
        ("R32_sha256_labels_sort", True, f'SELECT ?s (SHA256(STR(?o)) AS ?h) WHERE {{ ?s <{RDFS}label> ?o FILTER(?o != "{{nonce}}") }} ORDER BY ?h ?s LIMIT 200000'),
        ("R33_values2_scan", True, 'SELECT ?s ?p ?v WHERE { VALUES ?v { "H2O" "ATP" } ?s ?p ?x FILTER(STR(?x) = ?v) FILTER(?x != "{nonce}") } ORDER BY ?s ?p ?v'),
        ("R34_sort_export_600k", True, 'SELECT ?s ?p ?o WHERE { ?s ?p ?o FILTER(?o != "{nonce}") } ORDER BY ?o ?s ?p LIMIT 600000'),
    ],
    "uniprot": [
        ("U03_title_lcase", True, UP + 'SELECT ?c ?t WHERE { ?c up:title ?t FILTER(CONTAINS(LCASE(?t), "kinase") && ?t != "{nonce}") } ORDER BY ?c ?t'),
        ("U04_taxon_regex", True, UP + 'SELECT ?t ?n WHERE { ?t up:scientificName ?n FILTER(REGEX(?n, "^Strepto", "i") && ?n != "{nonce}") } ORDER BY ?t ?n'),
        ("U05_sequence_export", False, 'SELECT ?s ?v WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#value> ?v FILTER(?v != "{nonce}") } LIMIT 100000'),
        ("U06_annotation_type_group", True, UP + 'SELECT ?type (COUNT(?a) AS ?n) WHERE { ?p up:annotation ?a . ?a a ?type FILTER(?type != <urn:x-{nonce}>) } GROUP BY ?type ORDER BY ?type'),
        ("U07_author_group", True, UP + 'SELECT ?a (COUNT(?c) AS ?n) WHERE { ?c up:author ?a FILTER(?a != "{nonce}") } GROUP BY ?a ORDER BY DESC(?n) ?a LIMIT 20000'),
        ("U08_comment_contains", True, f'SELECT ?s ?c WHERE {{ ?s <{RDFS}comment> ?c FILTER(CONTAINS(?c, "zinc") && ?c != "{{nonce}}") }} ORDER BY ?s ?c'),
        ("U10_protein_organism_name", True, UP + 'SELECT ?p ?n WHERE { ?p up:organism ?o . ?o up:scientificName ?n FILTER(CONTAINS(?n, "coli") && ?n != "{nonce}") } ORDER BY ?p ?n'),
        ("U11_subclass_group", True, f'SELECT ?t (COUNT(?anc) AS ?n) WHERE {{ ?t <{RDFS}subClassOf> ?anc FILTER(?anc != <urn:x-{{nonce}}>) }} GROUP BY ?t ORDER BY DESC(?n) ?t LIMIT 100000'),
        ("U12_seealso_db_group", True, UP + f'SELECT ?db (COUNT(?x) AS ?n) WHERE {{ ?p <{RDFS}seeAlso> ?x . ?x up:database ?db FILTER(?db != <urn:x-{{nonce}}>) }} GROUP BY ?db ORDER BY ?db'),
        ("U17_fullname_regex", True, UP + 'SELECT ?s ?n WHERE { ?s up:fullName ?n FILTER(REGEX(?n, "kinase|phosphatase", "i") && ?n != "{nonce}") } ORDER BY ?s ?n'),
        ("U19_author_title_join_export", True, UP + 'SELECT ?c ?a ?t WHERE { ?c up:author ?a . ?c up:title ?t FILTER(?t != "{nonce}") } ORDER BY ?c ?a ?t LIMIT 300000'),
        ("U21_comment_regex_count", True, f'SELECT (COUNT(*) AS ?n) WHERE {{ ?s <{RDFS}comment> ?c FILTER(REGEX(?c, "(zinc|iron).*binding", "i") && ?c != "{{nonce}}") }}'),
        ("U23_othername_lcase", True, UP + 'SELECT ?s ?n WHERE { ?s up:otherName ?n FILTER(CONTAINS(LCASE(?n), "str") && ?n != "{nonce}") } ORDER BY ?s ?n'),
        ("U24_attribution_evidence_group", True, UP + 'SELECT ?e (COUNT(?a) AS ?n) WHERE { ?s up:attribution ?a . ?a up:evidence ?e FILTER(?e != <urn:x-{nonce}>) } GROUP BY ?e ORDER BY ?e'),
        ("U25_signature_sort_export", True, UP + 'SELECT ?p ?x WHERE { ?p up:signatureSequenceMatch ?x FILTER(?x != <urn:x-{nonce}>) } ORDER BY ?p ?x LIMIT 400000'),
        ("U26_comment_vowels_count", True, f'SELECT (COUNT(*) AS ?n) WHERE {{ ?s <{RDFS}comment> ?c FILTER(REGEX(LCASE(?c), "a.*e.*i.*o.*u.*y") && ?c != "{{nonce}}") }}'),
        ("U28_comment_kda_rows", True, f'SELECT ?s ?c WHERE {{ ?s <{RDFS}comment> ?c FILTER(REGEX(?c, "[0-9]{{3,}}.*kDa") && ?c != "{{nonce}}") }} ORDER BY ?s ?c'),
        ("U30_values1_author", True, UP + 'SELECT ?c ?a WHERE { VALUES ?v { "Smith J." } ?c up:author ?a FILTER(STR(?a) = ?v && ?a != "{nonce}") } ORDER BY ?c ?a'),
        ("U32_sequence_sort_export", True, 'SELECT ?s ?v WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#value> ?v FILTER(?v != "{nonce}") } ORDER BY ?s ?v LIMIT 250000'),
        ("U33_organism_mnemonic_join", True, UP + 'SELECT ?p ?o WHERE { ?p up:organism ?o . ?p up:mnemonic ?m FILTER(?m != "{nonce}") } ORDER BY ?o ?p'),
    ],
}
SHORT_7022 = {"heavy": ["UH1_values_str_orderby", "UH6_values_str_stream", "UH9_dense_stream"],
              "window": ["U03_title_lcase", "U08_comment_contains", "U19_author_title_join_export",
                         "U24_attribution_evidence_group", "U32_sequence_sort_export"]}


def family(store: str) -> str:
    return "rhea" if store.startswith("rhea") else "uniprot"


def endpoint(store: str) -> str:
    host, port = STORES[store]
    return f"http://{host}:{port}/"


# --- commands -----------------------------------------------------------------

def cmd_stats(args: argparse.Namespace) -> None:
    host, port = STORES[args.store]
    out = {"store": args.store, "stats": stats(host, port)}
    print(json.dumps(out, indent=1))
    if args.out:
        (Path(args.out) / f"stats-{args.store}.json").write_text(json.dumps(out, indent=1) + "\n")


def cmd_time(args: argparse.Namespace) -> None:
    """Time one query on the server (QLever JSON: server time, cache status, size)."""
    host, port = STORES[args.store]
    q = args.query.replace("{nonce}", nonce())
    raw = raw_post(host, port, q, timeout_param=f"{args.cap}s", accept=QLEVER_JSON, sock_timeout=args.cap + 15)
    print(json.dumps(raw.summary(), ensure_ascii=False))


def cmd_heavy(args: argparse.Namespace) -> None:
    host, port = STORES[args.store]
    ep = endpoint(args.store)
    log = Log(Path(args.out) / f"{args.store}-heavy.jsonl")
    shapes = HEAVY[family(args.store)]
    if args.only:
        shapes = [s for s in shapes if s[0] in args.only]
    for sid, cap, repeats, kind, q in shapes:
        for rep in range(1 if args.short else repeats):
            # Raw HTTP: exactly what the server sends, with the server's cap and a
            # generous socket time-out so the server, not the client, ends it.
            sent = time.monotonic()
            raw = raw_post(host, port, q, timeout_param=f"{cap}s", sock_timeout=cap + 60)
            r = raw.summary()
            r["overshoot_s"] = round(r["t_end_s"] - cap, 3)
            log.write({"store": args.store, "test": "heavy", "path": "raw", "shape": sid, "shape_kind": kind, "cap_s": cap,
                       "rep": rep, "query": q, **r})
            settle(sent, 0, 1.0)
            sent = time.monotonic()
            c, _ = client_run(ep, q, cap)
            log.write({"store": args.store, "test": "heavy", "path": "client", "shape": sid, "shape_kind": kind,
                       "cap_s": cap, "rep": rep, **c})
            if c.get("status") is None and c.get("kind") == "timeout":
                settle(sent, cap)  # the client gave up first; let the server's cap pass
            else:
                settle(sent, 0, 1.0)
    if args.backstop and not args.short:
        sid = BACKSTOP[family(args.store)]
        q = next(s[4] for s in HEAVY[family(args.store)] if s[0] == sid)
        sent = time.monotonic()
        raw = raw_post(host, port, q, timeout_param="20s", sock_timeout=5.0)
        log.write({"store": args.store, "test": "backstop", "path": "raw", "shape": sid, "server_cap_s": 20,
                   "http_timeout_s": 5, **raw.summary()})
        settle(sent, 20)
        sent = time.monotonic()
        c, _ = client_run(ep, q, 20, http_timeout=5.0)
        log.write({"store": args.store, "test": "backstop", "path": "client", "shape": sid, "server_cap_s": 20,
                   "http_timeout_s": 5, **c})
        settle(sent, 20)
    if args.overrun and family(args.store) == "uniprot":
        sid, cap, q = OVERRUN
        sent = time.monotonic()
        raw = raw_post(host, port, q, timeout_param=f"{cap}s", sock_timeout=285.0, wall_limit=285.0)
        r = raw.summary()
        r["overshoot_s"] = round(r["t_end_s"] - cap, 3)
        log.write({"store": args.store, "test": "overrun", "path": "raw", "shape": sid, "cap_s": cap, "query": q, **r})
        if r["end_reason"] != "eof":
            # No answer: the server may still be computing. (The first run on 7020
            # sent the client run 2 s later; it attached to that computation.)
            # Wait out the server's 300 s maximum before anything else is sent.
            log.write({"store": args.store, "test": "overrun", "path": "client", "shape": sid, "cap_s": cap,
                       "outcome": "skipped", "reason": "raw run got no answer; waiting out the server maximum"})
            settle(sent, 300.0, 15.0)
            return
        server_s = r["t_end_s"]
        settle(sent, 0, 2.0)
        sent = time.monotonic()
        c, _ = client_run(ep, q, cap)
        log.write({"store": args.store, "test": "overrun", "path": "client", "shape": sid, "cap_s": cap, **c})
        if c.get("status") is None and c.get("kind") == "timeout":
            settle(sent, server_s, 5.0)


def pick_cap(t: float) -> int | None:
    """The integer cap (s) that puts a run of t seconds nearest 70 % of it, within 45-92 %."""
    best = None
    for cap in range(1, 291):
        frac = t / cap
        if 0.45 <= frac <= 0.92 and (best is None or abs(frac - 0.7) < abs(t / best - 0.7)):
            best = cap
    return best


def cmd_window(args: argparse.Namespace) -> None:
    host, port = STORES[args.store]
    ep = endpoint(args.store)
    log = Log(Path(args.out) / f"{args.store}-window.jsonl")
    items = WINDOW[family(args.store)]
    if args.only:
        items = [w for w in items if w[0] in args.only]
    for wid, ordered, template in items:
        # 1. Calibration on the server (QLever JSON: time and cache status).
        q = template.replace("{nonce}", nonce())
        cal = raw_post(host, port, q, timeout_param="280s", accept=QLEVER_JSON, sock_timeout=290).summary()
        log.write({"store": args.store, "test": "window", "step": "calibrate", "id": wid, "template": template,
                   **{k: cal.get(k) for k in ("status", "t_end_s", "rows", "server_time_ms", "root_cache_status",
                                              "cache_statuses", "exception")}})
        if cal.get("status") != 200:
            continue
        t = cal["t_end_s"]
        # 2. Reference: the harness client with the largest cap the server allows.
        ref_meta, ref = client_run(ep, template.replace("{nonce}", nonce()), 290)
        log.write({"store": args.store, "test": "window", "step": "reference", "id": wid, "cap_s": 290, **ref_meta})
        if ref is None:
            continue
        t = max(t, ref_meta["elapsed_s"]) if ref_meta["elapsed_s"] < 2 * t else ref_meta["elapsed_s"]
        # 3. Capped runs through the client, re-aimed until one lands in 40-95 % of its cap.
        accepted = None
        for attempt in range(4):
            cap = pick_cap(t)
            if cap is None:
                log.write({"store": args.store, "test": "window", "step": "skip", "id": wid,
                           "reason": f"no integer cap puts {t:.3f} s in 45-92 %"})
                break
            meta, got = client_run(ep, template.replace("{nonce}", nonce()), cap)
            frac = meta["elapsed_s"] / cap
            rec = {"store": args.store, "test": "window", "step": "capped_client", "id": wid, "attempt": attempt,
                   "cap_s": cap, "frac": round(frac, 3), "in_window": 0.40 <= frac <= 0.95, **meta}
            if got is not None:
                rec["compare"] = compare(ref["bindings"], got["bindings"])
                rec["match"] = rec["compare"]["ordered_equal"] if ordered else rec["compare"]["multiset_equal"]
            log.write(rec)
            if got is not None and rec["in_window"]:
                accepted = cap
                break
            t = meta["elapsed_s"] if got is not None else cap * 1.3
        if accepted is None:
            continue
        # 4. The same cap over raw HTTP, SPARQL JSON: framing and rows as sent.
        raw = raw_post(host, port, template.replace("{nonce}", nonce()), timeout_param=f"{accepted}s",
                       sock_timeout=accepted + 60)
        rs = raw.summary(keep_body=True)
        parsed = rs.pop("_parsed")
        rec = {"store": args.store, "test": "window", "step": "capped_raw", "id": wid, "cap_s": accepted,
               "frac": round(rs["t_end_s"] / accepted, 3), **rs}
        if isinstance(parsed, dict) and isinstance(parsed.get("results"), dict):
            rec["compare"] = compare(ref["bindings"], parsed["results"]["bindings"])
            rec["match"] = rec["compare"]["ordered_equal"] if ordered else rec["compare"]["multiset_equal"]
        log.write(rec)
        # 5. The same cap over raw HTTP, QLever JSON: the run was computed, not served from cache.
        qj = raw_post(host, port, template.replace("{nonce}", nonce()), timeout_param=f"{accepted}s",
                      accept=QLEVER_JSON, sock_timeout=accepted + 60).summary()
        non_scan_cached = [c for c in qj.get("cache_statuses", [])
                           if not c.startswith(("IndexScan", "Values")) and not c.endswith(": computed")]
        log.write({"store": args.store, "test": "window", "step": "cache_check", "id": wid, "cap_s": accepted,
                   "frac": round(qj["t_end_s"] / accepted, 3),
                   **{k: qj.get(k) for k in ("status", "t_end_s", "rows", "server_time_ms", "root_cache_status",
                                             "exception", "marker")},
                   "rows_equal_ref_count": qj.get("rows") == len(ref["bindings"]),
                   "non_scan_cached": non_scan_cached})


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _cell(x: Any) -> str:
    return "" if x is None else str(x).replace("|", "\\|").replace("\n", " ")


def cmd_summarize(args: argparse.Namespace) -> None:
    """Tables per store from the JSONL logs: heavy shapes, back-stop, overrun, window."""
    out = Path(args.out)
    shape_kind = {sh[0]: sh[3] for shapes in HEAVY.values() for sh in shapes}
    shape_kind[OVERRUN[0]] = "materialising"
    summary: dict[str, Any] = {}
    md: list[str] = []
    for store in STORES:
        heavy, window = _load(out / f"{store}-heavy.jsonl"), _load(out / f"{store}-window.jsonl")
        if not heavy and not window:
            continue
        st = {"heavy": [], "window": [], "client_kinds_on_timeouts": {}}
        md.append(f"## {store}\n")
        if heavy:
            md.append("### Heavy shapes past the cap\n")
            md.append("| shape | kind | cap s | rep | path | status | end | framing complete | bytes | rows before cut | "
                      "error / marker text | client kind | elapsed s |")
            md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in heavy:
                if r["path"] == "raw":
                    text = r.get("marker_message") if r.get("marker") else (r.get("exception") or r.get("head", "")[:80])
                    rows = r.get("approx_rows_before_marker") if r.get("marker") else r.get("rows")
                    md.append(f"| {r['shape']} | {r['test']}/{shape_kind.get(r['shape'])} | {r.get('cap_s', r.get('server_cap_s'))} | "
                              f"{r.get('rep', '')} | raw | {_cell(r['status'])} | {r['end_reason']} | "
                              f"{r['framing_complete']} | {r['body_bytes']} | {_cell(rows)} | "
                              f"{'MARKER: ' if r.get('marker') else ''}{_cell(text)[:90]} |  | {r['t_end_s']} |")
                else:
                    md.append(f"| {r['shape']} | {r['test']}/{shape_kind.get(r['shape'])} | {r.get('cap_s', r.get('server_cap_s'))} | "
                              f"{r.get('rep', '')} | client | {_cell(r.get('status'))} |  |  |  | {_cell(r.get('rows'))} | "
                              f"{_cell(r.get('message'))[:90]} | {r.get('kind') if r['outcome'] == 'error' else 'ROWS'} | "
                              f"{r['elapsed_s']} |")
                    key = f"{r['test']}:{r['outcome']}:{r.get('kind')}:{r.get('status')}"
                    st["client_kinds_on_timeouts"][key] = st["client_kinds_on_timeouts"].get(key, 0) + 1
                st["heavy"].append({k: v for k, v in r.items() if k not in ("query", "head", "tail")})
            md.append("")
        if window:
            md.append("### Window queries (finish at 40-95 % of the cap)\n")
            md.append("| id | t calib s | cap s | client elapsed s | frac | rows | ref rows | match (client) | "
                      "raw status | raw frac | raw match | QLever-JSON frac | root cache | non-scan cached |")
            md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            by: dict[str, dict[str, Any]] = {}
            for r in window:
                d = by.setdefault(r["id"], {"attempts": []})
                if r["step"] == "capped_client":
                    d["attempts"].append(r)
                else:
                    d[r["step"]] = r
            n_ok = 0
            current = {w[0] for w in WINDOW[family(store)]}
            superseded = [wid for wid in by if wid not in current]
            for wid in superseded:
                del by[wid]
            for wid, d in by.items():
                acc = next((a for a in d["attempts"] if a.get("in_window") and a["outcome"] == "rows"), None)
                cal, ref, raw, cc = d.get("calibrate", {}), d.get("reference", {}), d.get("capped_raw", {}), d.get("cache_check", {})
                cached = [c for c in cc.get("non_scan_cached") or [] if not c.startswith("Values")]
                cc = {**cc, "non_scan_cached": cached}
                ok = bool(acc and acc.get("match") and raw.get("match") and cc.get("status") == 200
                          and not cached and cc.get("rows_equal_ref_count"))
                n_ok += ok
                row = {"id": wid, "t_calib_s": cal.get("t_end_s"), "ref_rows": ref.get("rows"), "ref_elapsed_s": ref.get("elapsed_s"),
                       "attempts": [{k: a.get(k) for k in ("cap_s", "elapsed_s", "frac", "in_window", "outcome", "kind", "status",
                                                          "rows", "match")} for a in d["attempts"]],
                       "accepted": None if not acc else {k: acc.get(k) for k in ("cap_s", "elapsed_s", "frac", "rows", "match", "compare")},
                       "raw": {k: raw.get(k) for k in ("status", "t_end_s", "frac", "framing_complete", "marker", "rows", "match")},
                       "cache_check": {k: cc.get(k) for k in ("status", "t_end_s", "frac", "root_cache_status", "non_scan_cached",
                                                              "rows_equal_ref_count", "rows")},
                       "pass": ok}
                st["window"].append(row)
                a = acc or (d["attempts"][-1] if d["attempts"] else {})
                md.append(f"| {wid} | {cal.get('t_end_s')} | {a.get('cap_s')} | {a.get('elapsed_s')} | {a.get('frac')} | "
                          f"{_cell(a.get('rows'))} | {_cell(ref.get('rows'))} | {_cell(a.get('match'))} | {_cell(raw.get('status'))} | "
                          f"{_cell(raw.get('frac'))} | {_cell(raw.get('match'))} | {_cell(cc.get('frac'))} | "
                          f"{_cell(cc.get('root_cache_status'))} | {len(cc.get('non_scan_cached') or [])} |")
            st["window_pass"] = f"{n_ok}/{len(by)}"
            st["window_superseded"] = superseded
            md.append(f"\nWindow queries passing every check: {n_ok}/{len(by)}\n")
            if superseded:
                md.append("Superseded (nonce in a FILTER of its own, so the expensive filter came from cache; rerun as the "
                          f"'b' ids with the nonce inside that filter): {', '.join(superseded)}\n")
        summary[store] = st
    for store in STORES:
        f = out / f"stats-{store}.json"
        if f.exists():
            summary.setdefault(store, {})["server"] = json.loads(f.read_text())["stats"]
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    (out / "tables.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'summary.json'} and {out / 'tables.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summarize").add_argument("--out", required=True)
    for name in ("stats", "time", "heavy", "window"):
        p = sub.add_parser(name)
        p.add_argument("--store", required=True, choices=sorted(STORES))
        p.add_argument("--out")
        if name == "time":
            p.add_argument("--cap", type=int, default=60)
            p.add_argument("query")
        if name in ("heavy", "window"):
            p.add_argument("--only", nargs="*")
            p.add_argument("--short", action="store_true", help="7022: the short confirmation set, one repeat")
        if name == "heavy":
            p.add_argument("--backstop", action="store_true")
            p.add_argument("--overrun", action="store_true")
    args = ap.parse_args()
    if getattr(args, "short", False) and not args.only:
        args.only = SHORT_7022[args.cmd]
    {"stats": cmd_stats, "time": cmd_time, "heavy": cmd_heavy, "window": cmd_window,
     "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
