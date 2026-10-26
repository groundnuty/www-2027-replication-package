"""SIB's UniProt example queries as held-out gold answers on the local SP+ store.

Three steps, each a script under ``scripts/``:

1. **prepare** (``e02_sib_prepare.py``, needs rdflib and the SIB checkout): read
   the ``examples/UniProt/*.ttl`` files, check prefixes, classify every
   query, pin the Rhea ``SERVICE`` calls to our Rhea 142 store (gold, variant
   a) and unwrap them over SP+ (diagnostic, variant b), and write the frozen
   catalogue ``queries.json`` plus ``edits.json`` and ``EDITS.md``.
2. **run** (``e02_sib_gold.py``, needs the store): post every final query to
   the endpoint and store one result file per query.
3. **copy to held-out** (``e02_sib_heldout.py``): copy the small result files
   into ``heldout/`` and record count and sha256 for the large ones.

The store is QLever without reasoning. A query without FROM or GRAPH sees the
union of all loaded graphs, and QLever executes SERVICE calls against the live
endpoints, so a query that still has a SERVICE to anything but our pinned Rhea
store is never pinned gold.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import statistics
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

from harness.sparql import SPARQL_JSON, query_form
from harness.store_identity import (
    EXPECTED_STORES,
    STRONG,
    StoreIdentityError,
    meta_identity,
    verify_store,
)
from harness.terms import XSD_STRING, BNode, IRI, Literal, row_to_json, term_from_json

SIB_REPO = "sib-swiss/sparql-examples"
SIB_COMMIT = "5d3fc1d7fdabf88950f01922a136a81df463be74"
HELDOUT_NAME = "sib-uniprot-5d3fc1d"

RHEA_ENDPOINTS = frozenset(
    {
        "https://sparql.rhea-db.org/sparql",
        "https://sparql.rhea-db.org/sparql/",
        "http://sparql.rhea-db.org/sparql",
        "http://sparql.rhea-db.org/sparql/",
    }
)
RHEA_GRAPH = "https://sparql.rhea-db.org/rhea"
CORE_GRAPH = "http://purl.uniprot.org/core"
_SP = "http://sparql.uniprot.org/"
LOADED_GRAPHS = (
    _SP + "uniprot",
    _SP + "taxonomy",
    _SP + "proteomes",
    _SP + "citations",
    _SP + "go",
    _SP + "enzymes",
    _SP + "keywords",
    _SP + "locations",
    _SP + "pathways",
    _SP + "tissues",
    _SP + "diseases",
    _SP + "journal",
    _SP + "databases",
    _SP + "chebi",
    RHEA_GRAPH,
    CORE_GRAPH,
)

# Gold (variant a) sends every Rhea SERVICE to our pinned Rhea 142 store. Its
# host changes per job, so the catalogue stores this placeholder IRI and the
# runner substitutes the concrete endpoint (--rhea-endpoint) at run time.
PINNED_RHEA = "urn:www2027:PINNED_RHEA"
PIN_REASON = "SERVICE endpoint pinned to the Rhea 142 store (same release as inside SP+)"
UNWRAP_REASON = (
    "diagnostic only, never gold: the SERVICE block unwrapped (its group pattern inlined) and run over "
    "SP+'s union default graph, which holds Rhea 142 and ChEBI from Rhea 142"
)

UP = "http://purl.uniprot.org/core/"
UNIPARC_UNIREF_GRAPHS = frozenset({_SP + "uniparc", _SP + "uniref"})
UNIPARC_UNIREF_NAMESPACES = ("http://purl.uniprot.org/uniparc/", "http://purl.uniprot.org/uniref/")
# Terms only UniParc and UniRef records use: up:Cluster and the properties whose
# core.owl domain or range is up:Cluster (UniRef), and up:sequenceFor (UniParc
# sequence -> entry). Swiss-Prot entries reach their sequences with up:sequence.
UNIPARC_UNIREF_TERMS = frozenset(
    UP + t
    for t in (
        "Cluster",
        "member",
        "memberOf",
        "commonTaxon",
        "seedFor",
        "representativeFor",
        "identity",
        "someMembersClassifiedWith",
        "sequenceFor",
    )
)

STATUSES = ("rows", "empty", "parse_error", "timeout", "http_error", "unsupported")
SUCCESS = ("rows", "empty")
NTRIPLES = "application/n-triples"
ACCEPT = {"SELECT": SPARQL_JSON, "ASK": SPARQL_JSON, "CONSTRUCT": NTRIPLES, "DESCRIBE": NTRIPLES}
XSD_BOOLEAN = "http://www.w3.org/2001/XMLSchema#boolean"

# Exclusive strata, in precedence order: a query lands in the first that applies.
# Every changed gold query is in "edited" (its "pinned" flag says whether it
# still calls a live endpoint).
STRATA = ("edited", "external_live", "uniparc_uniref", "unloaded_graph", "local")
FLAGS = (
    "external",
    "rhea_service",
    "needs_uniparc_uniref",
    "explicit_graph",
    "unloaded_graph",
    "unbounded_path",
    "prefix_added",
    "edited",
    "pinned",
)


# ---------------------------------------------------------------------------
# Extraction


class _TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def strip_html(text: str) -> str:
    """Text content of an HTML fragment, entities decoded, whitespace collapsed."""
    parser = _TextOnly()
    parser.feed(text)
    parser.close()
    return " ".join("".join(parser.parts).split())


def load_prefixes(prefixes_ttl: str | Path) -> dict[str, str]:
    """The ``sh:declare`` prefix map of SIB's ``prefixes.ttl``."""
    from rdflib import Graph, Namespace

    sh = Namespace("http://www.w3.org/ns/shacl#")
    g = Graph()
    g.parse(str(prefixes_ttl), format="turtle")
    out: dict[str, str] = {}
    for decl in g.subjects(sh.prefix, None):
        prefix = str(g.value(decl, sh.prefix))
        namespace = g.value(decl, sh.namespace)
        if namespace is not None:
            out[prefix] = str(namespace)
    return out


def read_example(path: str | Path) -> dict[str, Any]:
    """The fields of one SIB example file, as written by its authors."""
    from rdflib import RDF, RDFS, Graph, Namespace

    sh = Namespace("http://www.w3.org/ns/shacl#")
    spex = Namespace("https://purl.expasy.org/sparql-examples/ontology#")
    schema = Namespace("https://schema.org/")
    rdf_html = RDF.HTML

    path = Path(path)
    data = path.read_bytes()
    g = Graph()
    g.parse(data=data, format="turtle")
    subjects = set(g.subjects(RDF.type, sh.SPARQLExecutable))
    if len(subjects) != 1:
        raise ValueError(f"{path.name}: expected one sh:SPARQLExecutable, found {len(subjects)}")
    (s,) = subjects
    queries = [
        (str(p), str(o))
        for p in (sh.select, sh.ask, sh.construct, sh.describe, spex.describe)
        for o in g.objects(s, p)
    ]
    if len(queries) != 1:
        raise ValueError(f"{path.name}: expected one query, found {len(queries)}")
    comments = []
    for c in g.objects(s, RDFS.comment):
        is_html = getattr(c, "datatype", None) == rdf_html
        comments.append(
            {"text": strip_html(str(c)), "original": str(c), "lang": getattr(c, "language", None), "html": is_html}
        )
    # Primary question: plain-text comments before HTML ones, then by text. Every
    # comment is kept in "comments".
    comments.sort(key=lambda c: (c["html"], c["text"]))
    keywords = sorted({str(o) for p in (schema.keywords, schema.keyowrd) for o in g.objects(s, p)})
    return {
        "id": path.stem,
        "number": int(path.stem.split("_", 1)[0]),
        "file": path.name,
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "question": comments[0]["text"] if comments else "",
        "question_original": comments[0]["original"] if comments else "",
        "comments": comments,
        "labels": sorted(str(o) for o in g.objects(s, RDFS.label)),
        "keywords": keywords,
        "target": sorted(str(o) for o in g.objects(s, schema.target)),
        "federates_with": sorted(str(o) for o in g.objects(s, spex.federatesWith)),
        "query_predicate": queries[0][0],
        "query_original": queries[0][1],
    }


# ---------------------------------------------------------------------------
# Parse-tree helpers (rdflib's SPARQL grammar, no prefix resolution)


def _parse(query: str):
    from harness.sparql_grammar import parse_query

    return parse_query(query)


def _walk(node: Any) -> Iterable[Any]:
    from pyparsing import ParseResults
    from rdflib.plugins.sparql.parserutils import CompValue

    stack = [node]
    while stack:
        x = stack.pop()
        yield x
        if isinstance(x, CompValue):
            stack.extend(x.values())
        elif isinstance(x, (list, tuple, ParseResults)):
            stack.extend(x)


def _prologue(tree) -> tuple[dict[str, str], str | None]:
    prefixes: dict[str, str] = {}
    base = None
    for decl in tree[0]:
        if decl.name == "PrefixDecl":
            prefixes[decl.get("prefix") or ""] = str(decl.get("iri"))
        elif decl.name == "Base":
            base = str(decl.get("iri"))
    return prefixes, base


def parsed_form(query: str) -> str:
    name = _parse(query)[1].name
    return {"SelectQuery": "SELECT", "AskQuery": "ASK", "ConstructQuery": "CONSTRUCT", "DescribeQuery": "DESCRIBE"}[name]


def undeclared_prefixes(query: str) -> set[str]:
    """Prefixes the query uses but does not declare.

    This reads the parse tree rather than calling ``prepareQuery``, because
    rdflib resolves about thirty common prefixes (skos, schema, ...) by default
    and would hide a missing declaration.
    """
    from rdflib.plugins.sparql.parserutils import CompValue

    tree = _parse(query)
    declared = set(_prologue(tree)[0])
    used = {n.get("prefix") or "" for n in _walk(tree[1]) if isinstance(n, CompValue) and n.name == "pname"}
    return used - declared


def add_missing_prefixes(query: str, prefixes: dict[str, str]) -> tuple[str, dict | None, set[str]]:
    """Prepend declarations for used-but-undeclared prefixes.

    Returns the new query, the edit (or None) and the prefixes that have no
    declaration in ``prefixes`` either.
    """
    missing = undeclared_prefixes(query)
    known = sorted(p for p in missing if p in prefixes)
    unknown = missing - set(known)
    if not known:
        return query, None, unknown
    block = "".join(f"PREFIX {p}: <{prefixes[p]}>\n" for p in known)
    edit = {
        "kind": "prefix_added",
        "before": "",
        "after": block.rstrip("\n"),
        "reason": f"the query uses {', '.join(p + ':' for p in known)} without declaring it; "
        "declarations taken from SIB's examples/prefixes.ttl",
    }
    return block + query, edit, unknown


# ---------------------------------------------------------------------------
# Classification


def _resolve(node, prefixes: dict[str, str]) -> str | None:
    from rdflib import URIRef
    from rdflib.plugins.sparql.parserutils import CompValue

    if isinstance(node, URIRef):
        return str(node)
    if isinstance(node, CompValue) and node.name == "pname":
        ns = prefixes.get(node.get("prefix") or "")
        return None if ns is None else ns + (node.get("localname") or "")
    return None


def classify(query: str) -> dict[str, Any]:
    """Scope-aware facts about a query, read from its parse tree.

    Everything inside a SERVICE body runs on the remote endpoint, so it sets no
    local flag: its GRAPH clauses, paths and IRIs are recorded as remote.
    Comments never count because the parser drops them.
    """
    from pyparsing import ParseResults
    from rdflib import URIRef, Variable
    from rdflib.plugins.sparql.parserutils import CompValue

    tree = _parse(query)
    prefixes, _ = _prologue(tree)
    services: list[str] = []
    local_graphs: set[str] = set()
    local_iris: set[str] = set()
    paths = Counter()

    def visit(node: Any, remote: bool) -> None:
        if isinstance(node, CompValue):
            if node.name == "ServiceGraphPattern":
                term = node.get("term")
                services.append(str(term) if isinstance(term, URIRef) else f"?{term}")
                visit(node.get("graph"), True)
                return
            if not remote:
                if node.name == "GraphGraphPattern":
                    term = node.get("term")
                    local_graphs.add(str(term) if isinstance(term, URIRef) else f"?{term}")
                elif node.name == "DatasetClause":
                    local_graphs.add(str(node.get("default") or node.get("named")))
            if node.name == "PathElt" and node.get("mod") in ("+", "*"):
                paths["remote" if remote else "local"] += 1
            iri = _resolve(node, prefixes)
            if iri is not None and not remote:
                local_iris.add(iri)
            for value in node.values():
                visit(value, remote)
        elif isinstance(node, URIRef):
            if not remote:
                local_iris.add(str(node))
        elif isinstance(node, (list, tuple, ParseResults)):
            for value in node:
                visit(value, remote)
        elif isinstance(node, Variable):
            return

    visit(tree[1], False)
    rhea = [s for s in services if s in RHEA_ENDPOINTS]
    pinned_rhea = [s for s in services if s == PINNED_RHEA]
    external = sorted({s for s in services if s not in RHEA_ENDPOINTS and s != PINNED_RHEA})
    unloaded = sorted(g for g in local_graphs if not g.startswith("?") and g not in LOADED_GRAPHS)
    uniparc_uniref = sorted(
        {g for g in local_graphs if g in UNIPARC_UNIREF_GRAPHS}
        | {i for i in local_iris if i.startswith(UNIPARC_UNIREF_NAMESPACES) or i in UNIPARC_UNIREF_TERMS}
    )
    return {
        "services": sorted(set(services)),
        "external_endpoints": external,
        "rhea_service_blocks": len(rhea),
        "pinned_rhea_service_blocks": len(pinned_rhea),
        "local_graphs": sorted(local_graphs),
        "unloaded_graphs": unloaded,
        "uniparc_uniref_evidence": uniparc_uniref,
        "unbounded_paths_local": paths["local"],
        "unbounded_paths_in_service": paths["remote"],
    }


# ---------------------------------------------------------------------------
# Text-level SERVICE rewrite

_IRIREF = re.compile(r'<[^<>"{}|^`\\\x00-\x20]*>')
_SERVICE = re.compile(r"(?<![\w?$:])SERVICE(?![\w:-])", re.IGNORECASE)
_SILENT = re.compile(r"SILENT(?![\w:-])", re.IGNORECASE)


def _scan(q: str) -> str:
    """One kind letter per character: c code, i IRI, s string, m comment."""
    kinds = ["c"] * len(q)
    i, n = 0, len(q)

    def mark(a: int, b: int, k: str) -> None:
        kinds[a:b] = [k] * (b - a)

    while i < n:
        ch = q[i]
        if ch == "#":
            j = q.find("\n", i)
            j = n if j < 0 else j
            mark(i, j, "m")
            i = j
        elif ch in "\"'":
            if q.startswith(ch * 3, i):
                j = i + 3
                while j < n and not q.startswith(ch * 3, j):
                    j += 2 if q[j] == "\\" else 1
                j = min(n, j + 3)
            else:
                j = i + 1
                while j < n and q[j] not in (ch, "\n"):
                    j += 2 if q[j] == "\\" else 1
                j = min(n, j + 1)
            mark(i, j, "s")
            i = j
        elif ch == "<" and (m := _IRIREF.match(q, i)):
            mark(i, m.end(), "i")
            i = m.end()
        else:
            i += 1
    return "".join(kinds)


def _skip_space(q: str, kinds: str, i: int) -> int:
    while i < len(q) and (kinds[i] == "m" or (kinds[i] == "c" and q[i].isspace())):
        i += 1
    return i


def _match_brace(q: str, kinds: str, i: int) -> int:
    depth = 0
    for j in range(i, len(q)):
        if kinds[j] != "c":
            continue
        if q[j] == "{":
            depth += 1
        elif q[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced braces after SERVICE")


def service_blocks(query: str) -> list[dict[str, Any]]:
    """Every ``SERVICE [SILENT] <iri> { ... }`` block, with character offsets.

    ``start``/``header_end`` span the keyword up to the end of the endpoint
    IRI; ``open``/``close`` are the offsets of the body's braces.
    """
    kinds = _scan(query)
    blocks = []
    for m in _SERVICE.finditer(query):
        if kinds[m.start()] != "c":
            continue
        j = _skip_space(query, kinds, m.end())
        silent = _SILENT.match(query, j)
        if silent and kinds[j] == "c":
            j = _skip_space(query, kinds, silent.end())
        if j < len(query) and kinds[j] == "i":
            k = j
            while k < len(query) and kinds[k] == "i":
                k += 1
            endpoint = query[j + 1 : k - 1]
        else:
            var = re.match(r"[?$]\w+", query[j:])
            if not var:
                continue
            k = j + var.end()
            endpoint = var.group(0)
        open_ = _skip_space(query, kinds, k)
        if open_ >= len(query) or query[open_] != "{":
            continue
        blocks.append(
            {"start": m.start(), "header_end": k, "endpoint": endpoint, "open": open_, "close": _match_brace(query, kinds, open_)}
        )
    return blocks


def rhea_variants(query: str) -> tuple[str, str | None, list[dict], list[dict]]:
    """The two Rhea variants of a query.

    (a) gold: each ``SERVICE <https://sparql.rhea-db.org/sparql>`` keeps its
    body and points at the ``PINNED_RHEA`` placeholder; (b) diagnostic: each
    such block is unwrapped, ``SERVICE <...> { P }`` becoming ``{ P }``.
    Returns (query_a, query_b or None, edits_a, edits_b); with no Rhea SERVICE
    the query is returned unchanged and there is no diagnostic variant.
    """
    blocks = [b for b in service_blocks(query) if b["endpoint"] in RHEA_ENDPOINTS]
    if not blocks:
        return query, None, [], []
    gold, diag = query, query
    edits_a, edits_b = [], []
    for b in sorted(blocks, key=lambda b: b["start"], reverse=True):
        iri_start = query.rindex("<", b["start"], b["header_end"])
        gold = gold[:iri_start] + f"<{PINNED_RHEA}>" + gold[b["header_end"] :]
        diag = diag[: b["start"]] + diag[b["open"] :]
    for b in blocks:
        iri_start = query.rindex("<", b["start"], b["header_end"])
        block = query[b["start"] : b["close"] + 1]
        after_a = query[b["start"] : iri_start] + f"<{PINNED_RHEA}>" + query[b["header_end"] : b["close"] + 1]
        edits_a.append({"kind": "rhea_service_pinned", "before": block, "after": after_a, "reason": PIN_REASON})
        edits_b.append({"kind": "rhea_service_unwrapped", "before": block, "after": query[b["open"] : b["close"] + 1], "reason": UNWRAP_REASON})
    return gold, diag, edits_a, edits_b


_RHEA_ENDPOINT_OK = re.compile(r'https?://[^\s<>"{}|^`\\]+')


def substitute_rhea(query: str, rhea_endpoint: str | None) -> str:
    """The query with the PINNED_RHEA placeholder replaced by the concrete endpoint."""
    token = f"<{PINNED_RHEA}>"
    if token not in query:
        return query
    if not rhea_endpoint or not _RHEA_ENDPOINT_OK.fullmatch(rhea_endpoint):
        raise ValueError(f"this query needs a pinned Rhea endpoint (an http(s) URL), got {rhea_endpoint!r}")
    return query.replace(token, f"<{rhea_endpoint}>")


# ---------------------------------------------------------------------------
# Catalogue


def _stratum(flags: dict[str, bool]) -> str:
    if flags["edited"]:
        return "edited"
    if flags["external"]:
        return "external_live"
    if flags["needs_uniparc_uniref"]:
        return "uniparc_uniref"
    if flags["unloaded_graph"]:
        return "unloaded_graph"
    return "local"


def prepare_example(raw: dict[str, Any], prefixes: dict[str, str]) -> dict[str, Any]:
    """Prefix check, Rhea variants and classification for one extracted example."""
    ex = dict(raw)
    query = raw["query_original"]
    edits: list[dict] = []
    query, prefix_edit, unknown = add_missing_prefixes(query, prefixes)
    if prefix_edit:
        edits.append(prefix_edit)
    query, diagnostic, rhea_edits, diag_edits = rhea_variants(query)
    edits.extend(rhea_edits)
    for e in edits + diag_edits:
        e["id"] = ex["id"]
    final = classify(query)
    original = classify(raw["query_original"])
    diag_class = classify(diagnostic) if diagnostic is not None else None
    # Both variants must have no live Rhea SERVICE left; (a) one pinned block per original block.
    n_rhea = original["rhea_service_blocks"]
    if (
        final["rhea_service_blocks"]
        or final["pinned_rhea_service_blocks"] != n_rhea
        or (diag_class and (diag_class["rhea_service_blocks"] or diag_class["pinned_rhea_service_blocks"]))
    ):
        raise ValueError(f"{ex['id']}: Rhea variants did not produce the expected queries")
    form = parsed_form(query)
    text_form = query_form(query)
    if text_form != form:
        raise ValueError(f"{ex['id']}: form from text {text_form} differs from parse tree {form}")
    flags = {
        "external": bool(final["external_endpoints"]),
        "rhea_service": original["rhea_service_blocks"] > 0,
        "needs_uniparc_uniref": bool(final["uniparc_uniref_evidence"]),
        "explicit_graph": bool(final["local_graphs"]),
        "unloaded_graph": bool(final["unloaded_graphs"]),
        "unbounded_path": final["unbounded_paths_local"] > 0,
        "prefix_added": prefix_edit is not None,
        "edited": any(e["kind"] != "prefix_added" for e in edits),
        # Pinned: no SERVICE left except to our pinned Rhea store.
        "pinned": all(s == PINNED_RHEA for s in final["services"]),
    }
    ex.update(
        {
            "form": form,
            "query": query,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "edits": edits,
            "query_diagnostic": diagnostic,
            "query_diagnostic_sha256": hashlib.sha256(diagnostic.encode("utf-8")).hexdigest() if diagnostic else None,
            "diagnostic_edits": diag_edits,
            "classification_diagnostic": diag_class,
            "undeclared_prefixes_unresolved": sorted(unknown),
            "classification": final,
            "classification_original": original,
            "flags": flags,
            "stratum": _stratum(flags),
        }
    )
    return ex


def prepare(examples_dir: str | Path, sib_commit: str = SIB_COMMIT) -> dict[str, Any]:
    """The frozen catalogue of all UniProt examples under ``examples_dir``."""
    examples_dir = Path(examples_dir)
    prefixes_path = examples_dir / "prefixes.ttl"
    prefixes = load_prefixes(prefixes_path)
    files = sorted((examples_dir / "UniProt").glob("*.ttl"), key=lambda p: int(p.stem.split("_", 1)[0]))
    examples = [prepare_example(read_example(f), prefixes) for f in files]
    ids = [e["id"] for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate example ids")
    return {
        "source": {
            "repo": SIB_REPO,
            "commit": sib_commit,
            "path": "examples/UniProt",
            "n_examples": len(examples),
            "prefixes_sha256": hashlib.sha256(prefixes_path.read_bytes()).hexdigest(),
        },
        "store": {
            "loaded_graphs": list(LOADED_GRAPHS),
            "rhea_graph": RHEA_GRAPH,
            "rhea_endpoints": sorted(RHEA_ENDPOINTS),
            "pinned_rhea_placeholder": PINNED_RHEA,
        },
        "examples": examples,
    }


def edits_report(catalog: dict[str, Any]) -> dict[str, Any]:
    """The content of ``edits.json``: every gold edit, and the diagnostic variants (never gold)."""
    examples = catalog["examples"]
    return {
        "source": catalog["source"],
        "rule": "Any edit other than a pure prefix addition changes the gold query; those questions "
        "are listed in changed_ids and reported as their own stratum.",
        "changed_ids": [e["id"] for e in examples if e["flags"]["edited"]],
        "prefix_only_ids": [e["id"] for e in examples if e["flags"]["prefix_added"] and not e["flags"]["edited"]],
        "pinned_rhea_placeholder": {
            "iri": PINNED_RHEA,
            "substituted_by": "scripts/e02_sib_gold.py --rhea-endpoint URL (recorded as rhea_endpoint in each result's meta)",
        },
        "environment_ambiguous_rule": "For each query with a diagnostic variant, compare gold (a) and diagnostic (b) by "
        "canonical row hash and row count. A difference, or one variant failing while the other succeeds, marks the "
        "question environment_ambiguous: true, and it is excluded from the test set.",
        "diagnostic_ids": [e["id"] for e in examples if e["query_diagnostic"]],
        "diagnostic_edits": [
            {"id": ed["id"], "kind": ed["kind"], "before": ed["before"], "after": ed["after"], "reason": ed["reason"]}
            for e in examples
            for ed in e["diagnostic_edits"]
        ],
        "edits": [
            {"id": ed["id"], "kind": ed["kind"], "before": ed["before"], "after": ed["after"], "reason": ed["reason"]}
            for e in examples
            for ed in e["edits"]
        ],
    }


def _fence(text: str) -> str:
    return "```sparql\n" + text.strip("\n") + "\n```"


def edits_markdown(catalog: dict[str, Any]) -> str:
    examples = catalog["examples"]
    report = edits_report(catalog)
    by_id = {e["id"]: e for e in examples}
    strata = Counter(e["stratum"] for e in examples)
    src = catalog["source"]
    out = [
        "# Edits to the SIB UniProt gold queries",
        "",
        f"Source: `{src['repo']}` at commit `{src['commit']}`, `{src['path']}` ({src['n_examples']} examples).",
        "Generated by `scripts/e02_sib_prepare.py`; do not edit by hand.",
        "",
        "Rule: every question whose gold query is changed by any edit other than a pure prefix addition is listed "
        "below with its change, and is reported as its own stratum, `edited`. A changed query that still has a "
        "SERVICE to anything but our pinned Rhea store (pinned: no) calls a live endpoint, so its answers are not "
        "pinned gold either.",
        "",
        "## Strata",
        "",
        "Each query is in exactly one stratum, the first that applies:",
        "",
        "| Stratum | Meaning | Queries |",
        "|---|---|---|",
    ]
    meaning = {
        "edited": "the gold query was changed (listed below); check its pinned flag",
        "external_live": "unchanged, has a SERVICE to a non-Rhea endpoint; QLever calls it live, never pinned",
        "uniparc_uniref": "needs UniParc or UniRef data, which is not loaded; expected empty",
        "unloaded_graph": "names a graph that is not loaded; expected empty",
        "local": "unchanged and answered by the local store",
    }
    for s in STRATA:
        out.append(f"| `{s}` | {meaning[s]} | {strata.get(s, 0)} |")
    out += [
        "",
        "## The pinned Rhea endpoint",
        "",
        f"Gold (variant a) sends every `SERVICE <https://sparql.rhea-db.org/sparql>` to our Rhea 142 store, the same "
        f"release as the Rhea graph inside SP+. Its host changes per job, so the stored query names the placeholder "
        f"`<{PINNED_RHEA}>`; `scripts/e02_sib_gold.py --rhea-endpoint URL` substitutes the concrete endpoint at run time "
        "and records it as `rhea_endpoint` in each result's meta.",
        "",
        f"## Changed gold queries ({len(report['changed_ids'])})",
        "",
    ]
    if not report["changed_ids"]:
        out.append("None.")
    for qid in report["changed_ids"]:
        e = by_id[qid]
        others = [x for x in e["classification"]["services"] if x != PINNED_RHEA]
        live = "yes" if e["flags"]["pinned"] else "no: still calls " + ", ".join(others)
        out += [f"### `{qid}` (stratum `{e['stratum']}`, pinned: {live})", "", f"Question: {e['question']}", ""]
        for ed in e["edits"]:
            out += [f"- **{ed['kind']}**: {ed['reason']}", "", "Before:", "", _fence(ed["before"]), "", "After:", "", _fence(ed["after"]), ""]
    out += [f"## Prefix additions only ({len(report['prefix_only_ids'])})", ""]
    if report["prefix_only_ids"]:
        for qid in report["prefix_only_ids"]:
            added = "; ".join(ed["after"].replace("\n", "; ") for ed in by_id[qid]["edits"])
            out.append(f"- `{qid}`: {added}")
    else:
        out.append("None: every query declares every prefix it uses (checked on the parse tree, not with rdflib's default prefixes).")
    out += [
        "",
        f"## Diagnostic variant (b), never gold ({len(report['diagnostic_ids'])} queries)",
        "",
        "For each query with a Rhea SERVICE, variant (b) unwraps the SERVICE block (its group pattern is inlined) and "
        "runs over SP+'s union default graph. Its results are stored separately (`diagnostic/` under the run's output "
        "directory) and are never gold. " + report["environment_ambiguous_rule"],
        "",
    ]
    for qid in report["diagnostic_ids"]:
        e = by_id[qid]
        out += [f"### `{qid}`", ""]
        for ed in e["diagnostic_edits"]:
            out += ["Unwrapped block:", "", _fence(ed["after"]), ""]
    return "\n".join(out).rstrip("\n") + "\n"


def flag_counts(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(examples),
        "forms": dict(Counter(e["form"] for e in examples)),
        "flags": {f: sum(e["flags"][f] for e in examples) for f in FLAGS},
        "strata": {s: sum(e["stratum"] == s for e in examples) for s in STRATA},
    }


# ---------------------------------------------------------------------------
# N-Triples (stdlib only: the runner must not need rdflib for result parsing)

_NT_IRI = r'<((?:[^<>"{}|^`\\\x00-\x20]|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8})*)>'
_NT_BNODE = r"_:([\w](?:[\w.\-]*[\w\-])?)"
_NT_LIT = r'"((?:[^"\\\n\r]|\\.)*)"(?:@([A-Za-z]+(?:-[A-Za-z0-9]+)*)|\^\^' + _NT_IRI + ")?"
_NT_LINE = re.compile(
    r"^\s*(?:" + _NT_IRI + "|" + _NT_BNODE + r")\s+" + _NT_IRI + r"\s+(?:" + _NT_IRI + "|" + _NT_BNODE + "|" + _NT_LIT + r")\s*\.\s*(?:#.*)?$"
)
_ESC = re.compile(r"\\(?:u([0-9A-Fa-f]{4})|U([0-9A-Fa-f]{8})|(.))", re.DOTALL)
_ECHAR = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f", '"': '"', "'": "'", "\\": "\\"}


class NTriplesError(ValueError):
    pass


def _unescape(s: str) -> str:
    if "\\" not in s:
        return s

    def sub(m: re.Match) -> str:
        if m.group(1) or m.group(2):
            return chr(int(m.group(1) or m.group(2), 16))
        if m.group(3) in _ECHAR:
            return _ECHAR[m.group(3)]
        raise NTriplesError(f"bad escape \\{m.group(3)}")

    return _ESC.sub(sub, s)


def parse_ntriples_line(line: str):
    """One N-Triples line as an (s, p, o) tuple of harness terms, or None if blank."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    m = _NT_LINE.match(line)
    if not m:
        raise NTriplesError(f"not an N-Triples line: {stripped[:200]}")
    s_iri, s_bnode, p, o_iri, o_bnode, lex, lang, dt = m.groups()
    s = IRI(_unescape(s_iri)) if s_iri is not None else BNode(s_bnode)
    if o_iri is not None:
        o = IRI(_unescape(o_iri))
    elif o_bnode is not None:
        o = BNode(o_bnode)
    else:
        o = Literal(_unescape(lex), _unescape(dt) if dt else None, lang)
    return s, IRI(_unescape(p)), o


# ---------------------------------------------------------------------------
# Canonical rows and hashing


def _dump(cells: Any) -> str:
    return json.dumps(cells, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_row(row: list[dict | None]) -> str:
    """A row in term JSON form as a canonical string.

    Blank-node labels are not stable between runs, so every blank node becomes
    the same placeholder; an explicit xsd:string datatype is dropped (RDF 1.1
    treats it as a simple literal).
    """
    cells = []
    for c in row:
        if c is not None and c.get("type") == "bnode":
            c = {"type": "bnode", "value": ""}
        elif c is not None and c.get("datatype") == XSD_STRING:
            c = {k: v for k, v in c.items() if k != "datatype"}
        cells.append(c)
    return _dump(cells)


def rows_sha256(rows: Iterable[list[dict | None]]) -> str:
    """sha256 over the sorted canonical rows: independent of row order."""
    h = hashlib.sha256()
    for line in sorted(canonical_row(r) for r in rows):
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def boolean_sha256(value: bool) -> str:
    return hashlib.sha256(f"ASK {str(value).lower()}\n".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Runner core


def _truncate(text: str | None, n: int = 2000) -> str | None:
    if text is None:
        return None
    return text if len(text) <= n else text[:n] + f"... [{len(text) - n} more chars]"


_UNSUPPORTED = ("not supported", "unsupported", "not yet supported", "not implemented", "not yet implemented")


def _error_message(resp: httpx.Response, query: str | None) -> str:
    """The server's own error message, without the query text it may echo back.

    QLever answers errors as JSON with the message in ``exception`` and the
    query in ``query``; other servers send text, from which the sent query is cut.
    """
    text = resp.text
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict):
        for key in ("exception", "message", "error", "detail"):
            if isinstance(data.get(key), str):
                text = data[key]
                break
        else:
            text = json.dumps({k: v for k, v in data.items() if k != "query"})
    if query:
        text = text.replace(query, " ")
    return text


_TIMEOUT_MARKERS = ("timeout", "timed out", "time out", "cancelled", "canceled")


def failure_status(resp: httpx.Response, query: str | None = None) -> tuple[str, str]:
    """Status and error text for a non-200 response.

    The HTTP status decides first (406/415/501 unsupported, 408/504 timeout);
    then the server's message with any echoed query removed (timeout before
    unsupported); then 400 is a parse error and anything else an HTTP error.
    The recorded error is the full response text.
    """
    code = resp.status_code
    error = f"HTTP {code}: {resp.text}"
    if code in (406, 415, 501):
        return "unsupported", error
    if code in (408, 504):
        return "timeout", error
    message = _error_message(resp, query).lower()
    if any(m in message for m in _TIMEOUT_MARKERS):
        return "timeout", error
    if any(m in message for m in _UNSUPPORTED):
        return "unsupported", error
    if code == 400:
        return "parse_error", error
    return "http_error", error


def _outcome(status: str, elapsed: float, error: str | None = None, **kw: Any) -> dict[str, Any]:
    out = {
        "status": status,
        "elapsed_s": round(elapsed, 3),
        "error": _truncate(error),
        "columns": [],
        "rows": [],
        "boolean": None,
        "row_count": 0,
        "rows_sha256": None,
        "has_bnodes": False,
    }
    out.update(kw)
    return out


def _rows_outcome(columns: list[str], row_strings: list[str], elapsed: float) -> dict[str, Any]:
    has_bnodes = any('"type":"bnode"' in r for r in row_strings)
    needs_canon = has_bnodes or any(XSD_STRING in r for r in row_strings)
    if needs_canon:
        digest = rows_sha256(json.loads(r) for r in row_strings)
    else:  # stored strings are already canonical
        h = hashlib.sha256()
        for line in sorted(row_strings):
            h.update(line.encode("utf-8"))
            h.update(b"\n")
        digest = h.hexdigest()
    return _outcome(
        "rows" if row_strings else "empty",
        elapsed,
        columns=columns,
        rows=row_strings,
        row_count=len(row_strings),
        rows_sha256=digest,
        has_bnodes=has_bnodes,
    )


def _parse_body(form: str, content_type: str, body, elapsed: float) -> dict[str, Any]:
    if form in ("SELECT", "ASK"):
        try:
            raw = json.load(body)
        except ValueError as exc:
            return _outcome("http_error", elapsed, f"response is not JSON ({content_type}): {exc}")
        if form == "ASK":
            if not isinstance(raw.get("boolean"), bool):
                return _outcome("http_error", elapsed, "ASK response has no boolean")
            value = raw["boolean"]
            return _outcome("rows", elapsed, boolean=value, row_count=1, rows_sha256=boolean_sha256(value))
        try:
            columns = list(raw["head"].get("vars", []))
            bindings = raw["results"]["bindings"]
            row_strings = [
                _dump(row_to_json(tuple(term_from_json(b.get(v)) for v in columns))) for b in bindings
            ]
        except (KeyError, TypeError, ValueError) as exc:
            return _outcome("http_error", elapsed, f"malformed SPARQL JSON results: {exc}")
        del raw, bindings
        return _rows_outcome(columns, row_strings, elapsed)
    # CONSTRUCT / DESCRIBE
    ctype = content_type.split(";")[0].strip().lower()
    if ctype not in (NTRIPLES, "text/plain"):
        return _outcome("unsupported", elapsed, f"asked for {NTRIPLES}, server answered {content_type or 'no content type'}")
    seen: set[str] = set()
    rows: list[str] = []
    try:
        for n, raw_line in enumerate(body, 1):
            triple = parse_ntriples_line(raw_line.decode("utf-8"))
            if triple is None:
                continue
            s = _dump(row_to_json(triple))
            if s not in seen:  # an RDF graph is a set
                seen.add(s)
                rows.append(s)
    except (NTriplesError, UnicodeDecodeError) as exc:
        return _outcome("http_error", elapsed, f"line {n}: {exc}")
    return _rows_outcome(["subject", "predicate", "object"], rows, elapsed)


def execute(
    http: httpx.Client,
    endpoint: str,
    query: str,
    form: str,
    timeout_s: float,
    spool_dir: str | Path | None = None,
) -> dict[str, Any]:
    """POST one query (SPARQL 1.1 protocol) and classify the outcome.

    The body is spooled to a temporary file so that a very large result is
    not held twice in memory. ``elapsed_s`` runs from the request to the last
    byte of the response.
    """
    headers = {"Content-Type": "application/sparql-query", "Accept": ACCEPT[form]}
    start = time.monotonic()
    try:
        with http.stream(
            "POST",
            endpoint,
            content=query.encode("utf-8"),
            headers=headers,
            # QLever reads its own time-out from the request; the HTTP time-out
            # is a backstop a little above it (as in harness.sparql).
            params={"timeout": f"{timeout_s:g}s"},
            timeout=timeout_s + 10,
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                status, error = failure_status(resp, query)
                return _outcome(status, time.monotonic() - start, error)
            with tempfile.TemporaryFile(dir=spool_dir) as tmp:
                for chunk in resp.iter_bytes():
                    tmp.write(chunk)
                elapsed = time.monotonic() - start
                tmp.seek(0)
                return _parse_body(form, resp.headers.get("content-type", ""), tmp, elapsed)
    except httpx.TimeoutException as exc:
        return _outcome("timeout", time.monotonic() - start, f"client time-out after {timeout_s + 10:g}s: {exc!r}")
    except httpx.HTTPError as exc:
        return _outcome("http_error", time.monotonic() - start, f"transport error: {exc!r}")


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def result_path(out_dir: str | Path, qid: str) -> Path:
    return Path(out_dir) / "results" / f"{qid}.json.gz"


def meta_path(out_dir: str | Path, qid: str) -> Path:
    return Path(out_dir) / "meta" / f"{qid}.json"


def read_result(path: str | Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _result_bytes(header: dict[str, Any], row_strings: list[str]) -> bytes:
    head = json.dumps(header, ensure_ascii=False, sort_keys=True)
    sep = "\n" if row_strings else ""
    text = head[:-1] + ',"rows":[' + sep + ",\n".join(row_strings) + sep + "]}\n"
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:  # mtime=0: same rows, same bytes
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


VARIANTS = ("a", "b")  # a: gold; b: diagnostic (Rhea SERVICE unwrapped), never gold
DIAGNOSTIC_DIR = "diagnostic"
UNPINNED_DIR = "unpinned"  # variant-a results of queries that still call a live endpoint: never gold

# Run-time test-set classification of each query (exclusive, first that applies).
RESULT_STRATA = ("unpinned", "not_run", "failed", "empty_gold", "environment_ambiguous", "unverified", "primary")
BNODE_NOTE = (
    "the result has blank nodes; rows_sha256 maps every blank node to one placeholder, so it cannot tell "
    "apart answers that differ only in blank nodes"
)


def _variant_query(example: dict[str, Any], variant: str) -> tuple[str, str] | None:
    """(stored query text, its sha256) of one variant, or None if the example has no such variant."""
    if variant == "a":
        return example["query"], example["query_sha256"]
    if example.get("query_diagnostic"):
        return example["query_diagnostic"], example["query_diagnostic_sha256"]
    return None


def _uses_pinned_rhea(text: str) -> bool:
    return f"<{PINNED_RHEA}>" in text


def run_one(
    http: httpx.Client,
    endpoint: str,
    example: dict[str, Any],
    out_dir: Path,
    timeout_s: float,
    variant: str = "a",
    rhea_endpoint: str | None = None,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one variant of one catalogue entry; write its meta record and, on success, its result file.

    ``out_dir`` is the variant's own directory (the run directory for gold,
    its ``diagnostic/`` subdirectory for variant b). Only a variant-a result
    of a pinned query is gold; a variant-a result of a query that still calls
    a live endpoint is written under ``unpinned/`` and marked not gold.
    ``identity`` carries the SP+ and Rhea store identities of this run.
    """
    identity = identity or {}
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stored, stored_sha = _variant_query(example, variant)
    uses_pinned_rhea = _uses_pinned_rhea(stored)
    t0 = time.monotonic()
    sent = stored
    try:
        sent = substitute_rhea(stored, rhea_endpoint)
        outcome = execute(http, endpoint, sent, example["form"], timeout_s, spool_dir=out_dir)
    except Exception as exc:  # one bad query (e.g. out of memory) must not stop the run
        outcome = _outcome("http_error", time.monotonic() - t0, f"runner error: {exc!r}")
    meta = {
        "id": example["id"],
        "number": example["number"],
        "form": example["form"],
        "variant": variant,
        "gold": variant == "a" and bool(example["flags"]["pinned"]),
        "status": outcome["status"],
        "row_count": outcome["row_count"],
        "boolean": outcome["boolean"],
        "elapsed_s": outcome["elapsed_s"],
        "error": outcome["error"],
        "rows_sha256": outcome["rows_sha256"],
        "has_bnodes": outcome["has_bnodes"],
        "rows_sha256_note": BNODE_NOTE if outcome["has_bnodes"] else None,
        "edits": [e["kind"] for e in (example["edits"] if variant == "a" else example["diagnostic_edits"])],
        "flags": example["flags"],
        "stratum": example["stratum"],
        "external_endpoints": example["classification"]["external_endpoints"],
        "query_sha256": stored_sha,
        "query_sent_sha256": hashlib.sha256(sent.encode("utf-8")).hexdigest(),
        "rhea_endpoint": rhea_endpoint if uses_pinned_rhea else None,
        "rhea_identity": identity.get("rhea_identity") if uses_pinned_rhea else None,
        "rhea_identity_method": identity.get("rhea_identity_method") if uses_pinned_rhea else None,
        "sp_identity": identity.get("sp_identity"),
        "sp_identity_method": identity.get("sp_identity_method"),
        "endpoint": endpoint,
        "timeout_s": timeout_s,
        "started_at": started,
        "result_file": None,
    }
    if outcome["status"] in SUCCESS:
        try:
            store_dir = out_dir if meta["gold"] or variant != "a" else out_dir / UNPINNED_DIR
            _store_result(meta, outcome, example, out_dir, store_dir)
            if store_dir != out_dir:  # an older run may have left this live answer among the gold files
                result_path(out_dir, example["id"]).unlink(missing_ok=True)
        except Exception as exc:
            meta.update(status="http_error", error=f"could not store the result: {exc!r}", result_file=None)
    _write_atomic(meta_path(out_dir, example["id"]), (json.dumps(meta, indent=1, sort_keys=True) + "\n").encode())
    return meta


def _store_result(
    meta: dict[str, Any], outcome: dict[str, Any], example: dict[str, Any], out_dir: Path, store_dir: Path
) -> None:
    """Write the gzip result file under ``store_dir``; record its path (relative to ``out_dir``), size and sha256."""
    keys = (
        "id", "form", "variant", "gold", "status", "row_count", "boolean", "rows_sha256", "rows_sha256_note",
        "query_sha256", "rhea_endpoint", "rhea_identity", "rhea_identity_method", "sp_identity",
        "sp_identity_method", "endpoint", "started_at",
    )
    header = {k: meta[k] for k in keys}
    header["columns"] = outcome["columns"]
    header["pinned"] = example["flags"]["pinned"]
    path = result_path(store_dir, example["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _result_bytes(header, outcome["rows"])
    _write_atomic(path, data)
    meta["result_file"] = str(path.relative_to(out_dir))
    meta["result_file_bytes"] = len(data)
    meta["result_file_sha256"] = hashlib.sha256(data).hexdigest()


def variant_dir(out_dir: str | Path, variant: str) -> Path:
    return Path(out_dir) if variant == "a" else Path(out_dir) / DIAGNOSTIC_DIR


def run_all(
    examples: list[dict[str, Any]],
    endpoint: str,
    out_dir: str | Path,
    timeout_s: float = 120,
    parallel: int = 4,
    force: bool = False,
    http: httpx.Client | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    rhea_endpoint: str | None = None,
    variants: Iterable[str] = ("a",),
    identity_timeout_s: float = 600,
    allow_counts_identity: bool = False,
    expected_stores: dict[str, dict[str, Any]] = EXPECTED_STORES,
) -> list[dict[str, Any]]:
    """Run the chosen variants of every example that lacks a current result.

    Variant ``a`` is the gold query of every example, with the pinned Rhea
    placeholder replaced by ``rhea_endpoint``; variant ``b`` exists only for
    examples with a diagnostic query and is written under ``diagnostic/``.

    Before any query, the run establishes that ``endpoint`` serves the
    expected SP+ index and, if a selected query uses the pinned Rhea store,
    that ``rhea_endpoint`` serves the expected Rhea 142 index (QLever's
    ``name-index``; or, with ``allow_counts_identity``, the exact per-graph
    counts, recorded as the weaker identity). Otherwise it raises
    StoreIdentityError and runs nothing. Both are written to
    ``run_identity.json``.

    A result is current only if its query text and store identities match
    this run, it was made under the strong identity (or this run allows the
    weaker one), a diagnostic result shares its gold result's SP+ identity,
    and its file still has the recorded sha256; anything else is run again
    (all with ``force``).
    """
    variants = tuple(variants)
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants {sorted(unknown)}; use a and/or b")
    out_dir = Path(out_dir)
    candidates = [(e, v) for v in variants for e in examples if _variant_query(e, v) is not None]
    needs = sorted({e["id"] for e, v in candidates if _uses_pinned_rhea(_variant_query(e, v)[0])})
    if needs and not rhea_endpoint:
        raise ValueError(f"{len(needs)} queries need the pinned Rhea endpoint (rhea_endpoint / --rhea-endpoint): {', '.join(needs)}")
    if needs:
        substitute_rhea(f"<{PINNED_RHEA}>", rhea_endpoint)  # validate the URL once, before any request
    own = http is None
    http = http or httpx.Client()
    metas = []
    out_dir.mkdir(parents=True, exist_ok=True)

    def record(sp: dict | None, rhea: dict | None) -> None:
        data = {"sp": sp, "rhea": rhea, "allow_counts_identity": allow_counts_identity}
        _write_atomic(out_dir / "run_identity.json", (json.dumps(data, indent=1, sort_keys=True) + "\n").encode())

    try:
        sp = rhea = None
        try:
            sp = verify_store(http, endpoint, expected_stores["sp"], identity_timeout_s, allow_counts_identity)
            if needs:
                rhea = verify_store(http, rhea_endpoint, expected_stores["rhea"], identity_timeout_s, allow_counts_identity)
        except StoreIdentityError as exc:  # record the refusal, so no summary shows an older verification
            failed = {**(exc.identity or {"endpoint": rhea_endpoint if sp else endpoint}), "verified": False, "error": str(exc)}
            if sp is not None:
                record(sp, failed)
            else:
                record(failed, None)
            raise
        identity = {
            "sp_identity": sp["identity"],
            "sp_identity_method": sp["method"],
            "rhea_identity": rhea["identity"] if rhea else None,
            "rhea_identity_method": rhea["method"] if rhea else None,
            "allow_counts_identity": allow_counts_identity,
        }
        record(sp, rhea)
        jobs = []
        for e, v in candidates:
            vdir = variant_dir(out_dir, v)
            (vdir / "results").mkdir(parents=True, exist_ok=True)
            (vdir / "meta").mkdir(parents=True, exist_ok=True)
            if force or not _done(out_dir, e, v, identity, expected_stores):
                jobs.append((e, v, vdir))
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            for meta in pool.map(lambda j: run_one(http, endpoint, j[0], j[2], timeout_s, j[1], rhea_endpoint, identity), jobs):
                metas.append(meta)
                if progress:
                    progress(meta)
    finally:
        if own:
            http.close()
    return metas


def _load_meta(vdir: Path, example: dict[str, Any], query_sha256: str) -> dict[str, Any] | None:
    """The meta record for this query text, or None (missing or stale)."""
    p = meta_path(vdir, example["id"])
    if not p.exists():
        return None
    meta = json.loads(p.read_text())
    return meta if meta.get("query_sha256") == query_sha256 else None


def _identity_ok(meta: dict[str, Any], key: str, identity: dict[str, Any], expected: dict[str, dict[str, Any]]) -> bool:
    ident, method = meta_identity(meta, key, expected)
    if ident is None or ident != identity.get(f"{key}_identity"):
        return False
    return method == STRONG or bool(identity.get("allow_counts_identity"))


def _done(
    out_dir: Path,
    example: dict[str, Any],
    variant: str,
    identity: dict[str, Any],
    expected: dict[str, dict[str, Any]] = EXPECTED_STORES,
) -> bool:
    """A current result exists: same query text, same store identities, intact file.

    A result made under the weaker (counts) identity counts only in a run that
    allows it. A diagnostic result must share its gold result's SP+ identity.
    """
    vdir = variant_dir(out_dir, variant)
    text, sha = _variant_query(example, variant)
    meta = _load_meta(vdir, example, sha)
    if not meta or not meta.get("result_file"):
        return False
    if not _identity_ok(meta, "sp", identity, expected):
        return False
    if _uses_pinned_rhea(text) and not _identity_ok(meta, "rhea", identity, expected):
        return False
    if variant == "b":
        gold = _load_meta(out_dir, example, example["query_sha256"])
        if gold and meta_identity(gold, "sp", expected)[0] != meta_identity(meta, "sp", expected)[0]:
            return False
    path = vdir / meta["result_file"]
    return path.exists() and _sha256_file(path) == meta.get("result_file_sha256")


def compare_variants(gold: dict[str, Any] | None, diagnostic: dict[str, Any] | None) -> tuple[bool | None, str]:
    """(environment_ambiguous, reason) from the gold and diagnostic meta records.

    Both records must carry the same SP+ store identity (``sp_identity``);
    otherwise the comparison is undetermined (``identity_mismatch``), since
    the two answers may come from different snapshots. Ambiguous when both
    succeed with different canonical row hashes or row counts, or when one
    succeeds and the other fails. Undetermined (None) when a variant has not
    run, both failed, or the hashes agree but either result has blank nodes:
    the hash maps every blank node to one value, so agreement is no evidence
    (a difference still is).
    """
    if gold is None or diagnostic is None:
        return None, "not_run"
    if not gold.get("sp_identity") or gold.get("sp_identity") != diagnostic.get("sp_identity"):
        return None, "identity_mismatch"
    a_ok, b_ok = gold["status"] in SUCCESS, diagnostic["status"] in SUCCESS
    if a_ok and b_ok:
        same = gold["rows_sha256"] == diagnostic["rows_sha256"] and gold["row_count"] == diagnostic["row_count"]
        if not same:
            return True, "rows_differ"
        if gold.get("has_bnodes") or diagnostic.get("has_bnodes"):
            return None, "blank_nodes"
        return False, "same"
    if a_ok != b_ok:
        return True, "diagnostic_failed" if a_ok else "gold_failed"
    return None, "both_failed"


def result_stratum(record: dict[str, Any], has_diagnostic: bool) -> str:
    """Where a query's gold answer stands for the test set.

    ``primary`` (the primary test set) needs a pinned query, a non-empty gold
    answer (status ``rows``) and, for a query with a diagnostic variant, a
    comparison that found the variants the same. Pinned empty gold is
    ``empty_gold``; a diagnostic comparison that could not be made (not run,
    blank nodes, different store identities) is ``unverified``.
    """
    if not record["flags"]["pinned"]:
        return "unpinned"
    if record["status"] == "not_run":
        return "not_run"
    if record["status"] not in SUCCESS:
        return "failed"
    if record["status"] == "empty":
        return "empty_gold"
    if record.get("environment_ambiguous") is True:
        return "environment_ambiguous"
    if has_diagnostic and record.get("environment_ambiguous") is not False:
        return "unverified"
    return "primary"


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]


def _effective_identities(meta: dict[str, Any], expected: dict[str, dict[str, Any]] = EXPECTED_STORES) -> None:
    """Set sp/rhea identity and method on a meta record (mapping legacy counts fingerprints)."""
    meta["sp_identity"], meta["sp_identity_method"] = meta_identity(meta, "sp", expected)
    if meta.get("rhea_endpoint"):
        meta["rhea_identity"], meta["rhea_identity_method"] = meta_identity(meta, "rhea", expected)


def _weak(r: dict[str, Any]) -> bool:
    """Made under the weaker (counts) identity for SP+ or, where used, Rhea."""
    return r.get("sp_identity_method") != STRONG or bool(r.get("rhea_endpoint") and r.get("rhea_identity_method") != STRONG)


def summarize(examples: list[dict[str, Any]], out_dir: str | Path, catalog_sha256: str | None = None) -> dict[str, Any]:
    """Summary over the latest gold attempt of every catalogue entry, plus the Rhea comparison.

    Status, rows, time and hash come from the meta record of the entry's current
    query text (a record for an older text counts as not run); edits, flags and
    stratum always come from the catalogue. Entries with a diagnostic variant
    also carry its outcome and ``environment_ambiguous``.
    """
    out_dir = Path(out_dir)
    records = []
    for e in examples:
        meta = _load_meta(out_dir, e, e["query_sha256"]) or {
            "id": e["id"], "number": e["number"], "form": e["form"], "status": "not_run", "row_count": 0,
            "elapsed_s": None, "error": None, "rows_sha256": None, "result_file": None,
        }
        meta.update(edits=[x["kind"] for x in e["edits"]], flags=e["flags"], stratum=e["stratum"])
        _effective_identities(meta)
        meta["environment_ambiguous"], meta["comparison"], meta["diagnostic"] = None, None, None
        if e.get("query_diagnostic"):
            diag = _load_meta(variant_dir(out_dir, "b"), e, e["query_diagnostic_sha256"])
            if diag:
                _effective_identities(diag)
            gold = meta if meta["status"] != "not_run" else None
            meta["environment_ambiguous"], meta["comparison"] = compare_variants(gold, diag)
            if diag:
                keys = ("status", "row_count", "rows_sha256", "has_bnodes", "elapsed_s", "error", "result_file",
                        "result_file_sha256", "sp_identity", "sp_identity_method")
                meta["diagnostic"] = {k: diag.get(k) for k in keys}
        meta["gold"] = bool(meta.get("gold")) and bool(e["flags"]["pinned"])  # older metas marked live answers gold
        meta["result_stratum"] = result_stratum(meta, bool(e.get("query_diagnostic")))
        meta["primary_test"] = meta["result_stratum"] == "primary"
        meta["empty_gold"] = meta["result_stratum"] == "empty_gold"
        records.append(meta)
    statuses = list(STATUSES) + (["not_run"] if any(r["status"] == "not_run" for r in records) else [])

    def tally(rs: list[dict]) -> dict[str, int]:
        c = Counter(r["status"] for r in rs)
        return {"total": len(rs), **{s: c.get(s, 0) for s in statuses}}

    compared = [r for r in records if r["comparison"] is not None]
    ok_times = [r["elapsed_s"] for r in records if r["status"] in SUCCESS]
    ident_path = out_dir / "run_identity.json"
    run_identity = json.loads(ident_path.read_text()) if ident_path.exists() else None
    sp_ids = sorted({r["sp_identity"] for r in records if r.get("sp_identity")})
    rhea_ids = sorted({r["rhea_identity"] for r in records if r.get("rhea_identity")})
    with_results = [r for r in records if r.get("result_file")]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoints": sorted({r["endpoint"] for r in records if r.get("endpoint")}),
        "rhea_endpoints": sorted({r["rhea_endpoint"] for r in records if r.get("rhea_endpoint")}),
        "run_identity": run_identity,
        "rhea_verification": (run_identity or {}).get("rhea"),
        "sp_identities": sp_ids,
        "rhea_identities": rhea_ids,
        "mixed_identities": len(sp_ids) > 1 or len(rhea_ids) > 1,
        "identity_methods": {
            "sp": dict(sorted(Counter(str(r.get("sp_identity_method")) for r in with_results).items())),
            "rhea": dict(sorted(Counter(str(r.get("rhea_identity_method")) for r in with_results if r.get("rhea_endpoint")).items())),
        },
        "weak_identity_gold_results": sum(r["gold"] and _weak(r) for r in with_results),
        "primary_test": sum(r["primary_test"] for r in records),
        "empty_gold": sum(r["empty_gold"] for r in records),
        "primary_test_edited": sum(r["primary_test"] and r["stratum"] == "edited" for r in records),
        "result_strata": {s: sum(r["result_stratum"] == s for r in records) for s in RESULT_STRATA},
        "primary_test_by_stratum": {s: sum(r["primary_test"] and r["stratum"] == s for r in records) for s in STRATA},
        "primary_test_ids": [r["id"] for r in records if r["primary_test"]],
        "empty_gold_ids": [r["id"] for r in records if r["empty_gold"]],
        "gold_results": sum(r["gold"] and bool(r.get("result_file")) for r in records),
        "unpinned_results": sum((not r["gold"]) and bool(r.get("result_file")) for r in records),
        "catalog_sha256": catalog_sha256,
        "n_queries": len(records),
        "status_counts": tally(records),
        "by_stratum": {s: tally([r for r in records if r["stratum"] == s]) for s in STRATA},
        "by_flag": {f: tally([r for r in records if r["flags"][f]]) for f in FLAGS},
        "pinned_status_counts": tally([r for r in records if r["flags"]["pinned"]]),
        "environment_ambiguous": {
            "with_diagnostic": len(compared),
            "ambiguous": sum(r["environment_ambiguous"] is True for r in compared),
            "same": sum(r["environment_ambiguous"] is False for r in compared),
            "undetermined": sum(r["environment_ambiguous"] is None for r in compared),
            "by_reason": dict(sorted(Counter(r["comparison"] for r in compared).items())),
            "ambiguous_ids": [r["id"] for r in compared if r["environment_ambiguous"] is True],
        },
        "latency_successful": {
            "n": len(ok_times),
            "median_s": statistics.median(ok_times) if ok_times else None,
            "p95_s": _p95(ok_times),
        },
        "queries": records,
    }


TSV_COLUMNS = (
    "id", "number", "stratum", "result_stratum", "primary_test", "status", "gold", "row_count", "elapsed_s", "rows_sha256", *FLAGS,
    "environment_ambiguous", "comparison", "diag_status", "diag_row_count", "rhea_endpoint", "edits", "error",
)


def summary_tsv(summary: dict[str, Any]) -> str:
    lines = ["\t".join(TSV_COLUMNS)]
    for r in summary["queries"]:
        cells = []
        for col in TSV_COLUMNS:
            if col in FLAGS:
                v = int(r["flags"][col])
            elif col == "edits":
                v = ",".join(r.get("edits") or [])
            elif col == "error":
                v = " ".join((r.get("error") or "").split())[:300]
            elif col in ("gold", "primary_test"):
                v = int(r[col])
            elif col == "environment_ambiguous":
                v = None if r.get(col) is None else int(r[col])
            elif col in ("diag_status", "diag_row_count"):
                v = (r.get("diagnostic") or {}).get(col.removeprefix("diag_"))
            else:
                v = r.get(col)
            cells.append("" if v is None else str(v))
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def write_summary(examples: list[dict[str, Any]], out_dir: str | Path, catalog_sha256: str | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    summary = summarize(examples, out_dir, catalog_sha256)
    _write_atomic(out_dir / "summary.json", (json.dumps(summary, indent=1, sort_keys=True) + "\n").encode())
    _write_atomic(out_dir / "summary.tsv", summary_tsv(summary).encode())
    return summary


def _verified(results_dir: Path, rel: str, sha256: str | None) -> Path:
    src = results_dir / rel
    if not sha256 or _sha256_file(src) != sha256:
        raise ValueError(f"{src}: file sha256 does not match the recorded result_file_sha256; re-run the query")
    return src


def copy_to_heldout(
    results_dir: str | Path, dest: str | Path, cluster_path: str, max_bytes: int = 5 * 1024 * 1024
) -> dict[str, Any]:
    """Copy gold result files under ``max_bytes`` into ``dest``; record the rest.

    Gold means a variant-a result of a pinned query (``gold`` in its meta and
    ``pinned`` in the catalogue flags). Every gold file gets a manifest line
    with its row count, row hash, file size and sha256 (checked against the
    recorded one); files at or above the limit stay on cluster storage under
    ``cluster_path``. Results of unpinned queries (they called a live
    endpoint) and diagnostic results are never gold: they are listed with
    their hashes and never copied.
    """
    results_dir, dest = Path(results_dir), Path(dest)
    summary = json.loads((results_dir / "summary.json").read_text())
    gold_records = [r for r in summary["queries"] if r.get("result_file") and r.get("gold") is True]
    unidentified = [r["id"] for r in gold_records if not r.get("sp_identity") or (r.get("rhea_endpoint") and not r.get("rhea_identity"))]
    if summary.get("mixed_identities") or unidentified:
        raise ValueError(
            "refusing to copy: the gold results were not all made against one identified store "
            f"(SP+ {summary.get('sp_identities')}, Rhea {summary.get('rhea_identities')}, "
            f"without identity: {unidentified[:5]}); re-run first"
        )
    (dest / "results").mkdir(parents=True, exist_ok=True)
    root = cluster_path.rstrip("/")
    files, unpinned_files, diagnostic_files = [], [], []
    for r in summary["queries"]:
        if not r.get("result_file"):
            continue
        src = _verified(results_dir, r["result_file"], r.get("result_file_sha256"))
        entry = {
            "id": r["id"],
            "status": r["status"],
            "row_count": r["row_count"],
            "rows_sha256": r["rows_sha256"],
            "file_bytes": src.stat().st_size,
            "file_sha256": r["result_file_sha256"],
        }
        entry["primary_test"] = bool(r.get("primary_test"))
        entry["result_stratum"] = r.get("result_stratum")
        entry["sp_identity"], entry["sp_identity_method"] = r.get("sp_identity"), r.get("sp_identity_method")
        entry["rhea_identity"], entry["rhea_identity_method"] = r.get("rhea_identity"), r.get("rhea_identity_method")
        entry["weak_identity"] = _weak(r)
        if not (r.get("gold") is True and r["flags"]["pinned"]):
            unpinned_files.append({**entry, "in_git": False, "path": f"{root}/{r['result_file']}"})
            continue
        in_git = entry["file_bytes"] < max_bytes
        entry.update(
            in_git=in_git,
            path=r["result_file"] if in_git else f"{root}/{r['result_file']}",
            environment_ambiguous=r.get("environment_ambiguous"),  # true: excluded from the test set
        )
        if in_git:
            _write_atomic(dest / r["result_file"], src.read_bytes())
        files.append(entry)
    keep = {Path(f["path"]).name for f in files if f["in_git"]}
    pruned = sorted(f.name for f in (dest / "results").glob("*.json.gz") if f.name not in keep)
    for name in pruned:  # left by an earlier copy: not current pinned gold
        (dest / "results" / name).unlink()
    for r in summary["queries"]:
        d = r.get("diagnostic") or {}
        if not d.get("result_file"):
            continue
        rel = f"{DIAGNOSTIC_DIR}/{d['result_file']}"
        src = _verified(results_dir, rel, d.get("result_file_sha256"))
        diagnostic_files.append(
            {
                "id": r["id"],
                "status": d["status"],
                "row_count": d["row_count"],
                "rows_sha256": d["rows_sha256"],
                "file_bytes": src.stat().st_size,
                "file_sha256": d["result_file_sha256"],
                "in_git": False,  # diagnostic, never gold: stays on cluster storage
                "path": f"{root}/{rel}",
            }
        )
    storage = {
        "cluster_path": cluster_path,
        "max_bytes_in_git": max_bytes,
        "n_result_files": len(files),
        "n_in_git": sum(f["in_git"] for f in files),
        "n_on_cluster_only": sum(not f["in_git"] for f in files),
        "n_unpinned_files_on_cluster": len(unpinned_files),
        "n_primary_test": sum(f["primary_test"] for f in files),
        "n_weak_identity": sum(f["weak_identity"] for f in files),
        "pruned_from_heldout": pruned,
        "n_diagnostic_files_on_cluster": len(diagnostic_files),
    }
    manifest = {**storage, "files": files, "unpinned_files": unpinned_files, "diagnostic_files": diagnostic_files}
    _write_atomic(dest / "results_manifest.json", (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode())
    summary["storage"] = storage
    _write_atomic(dest / "summary.json", (json.dumps(summary, indent=1, sort_keys=True) + "\n").encode())
    _write_atomic(dest / "summary.tsv", (results_dir / "summary.tsv").read_bytes())
    return manifest
