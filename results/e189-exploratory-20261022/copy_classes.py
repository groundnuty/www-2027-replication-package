#!/usr/bin/env python3
"""The speed class of each store copy the exploratory analysis can attribute (science's ruling, 25 Oct ~05:40-05:50Z),
as the committed data file ``copy-classes.json`` that exploratory.py reads (read) and prints in its branch-phase report.

``build`` writes the file from the class sources; the functions below are pure (they take parsed sources) so that each
precedence step is tested on small fixtures. The copies covered are the replica names of store-proofs.json plus those
of endpoint-copies.json (the copies the analysis can attribute: a copy with a proof entry, and a copy named for the
endpoint an OUT with a single store endpoint used; that table never counts as a proof, see ``parse_endpoint_copies``)
plus those of extra-copies.json (a copy a look reads that neither file names, with its graph and the registry that
says so; it names no endpoint, host or port, see ``parse_extra_copies``).
A copy's class is the first source that gives one:

1. the v3 FINAL measurement, speed-class.json (``copies.<copy>.speed_class`` and ``.basis``), and the per-copy records
   (``copy``, ``speed_class``, ``basis``); only fast, medium or slow is a class, so v3's "partial" is no class and falls
   through; v3 and a record giving different classes for one copy refuse;
2. the reviewed tier2 replay table cell-classes.tsv (tab-separated: column 2 the copy id, 3 the class, 4 the note);
3. the standing host rule, from copy-hosts.json, for a copy in neither: its graph and host family give the class.

An explicit, reviewed list of overrides, class-overrides.json (``parse_overrides``), is the only exception to the order, and it is
narrow: a cell-table row that is informational (a similarity note, not a measurement) is ignored for the copies it names, so
they fall to the host rule; and the basis text of an entry that comes from the cell table is replaced by the text it gives.
A third kind, a measured class, gives a copy measured after the last v3 FINAL (and not yet issued as a per-copy record) the class
its measurement gives, citing the measurement's files, and only while no other source has a class for it.
Every override is checked against what it overrides, and a stale one refuses the build.

Every failure raises Refused (exit 2 on the command line): nothing is guessed, nothing is a warning or an empty result."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "e189-copy-classes/1"
HOSTS_SCHEMA = "e189-copy-hosts/1"
STORE_PROOFS_SCHEMA = "e189-store-proofs/1"
ENDPOINT_COPIES_SCHEMA = "e189-endpoint-copies/1"
ENDPOINT_COPY_KEYS = {"graph", "endpoint_sha256", "replica", "basis", "sources"}
EXTRA_COPIES_SCHEMA = "e189-extra-copies/1"
EXTRA_COPY_KEYS = {"replica", "graph", "basis", "sources"}
OVERRIDES_SCHEMA = "e189-class-overrides/2"
OVERRIDE_INFO_KEYS = {"replica", "table_rows", "why", "sources"}
OVERRIDE_BASIS_KEYS = {"replica", "table_rows", "basis", "why", "sources"}
OVERRIDE_MEASURED_KEYS = {"replica", "class", "basis", "why", "sources"}
CLASSES = ("fast", "medium", "slow")
HOSTS = ("sitea", "sitec", "siteb-x86", "arm64")
_BY_SITEA_SITEC = {"sitea": "fast", "sitec": "fast", "siteb-x86": "slow", "arm64": "medium"}
# The standing host rule (science, 25 Oct ~05:4xZ): the graph (a store name, as store-proofs.json names it) -> host -> class.
# None: no host rule for the graph (corporate): its copies are unmeasured and carry no class.
HOST_RULE: dict[str, dict[str, str] | None] = {
    "rhea-142": {"sitea": "fast", "sitec": "medium", "siteb-x86": "slow", "arm64": "medium"},
    "opaque-rhea-142": _BY_SITEA_SITEC,
    "uniprot-sp-2026_03": _BY_SITEA_SITEC,
    "text2sparql-ck26": None,
}
NO_RULE_LABEL = "unmeasured (no host rule for corporate)"
V3_SOURCE, CELLS_SOURCE, HOSTS_SOURCE, PROOFS_SOURCE = "speed-class-v3", "cell-classes", "copy-hosts", "store-proofs"
ENDPOINTS_SOURCE, EXTRAS_SOURCE, OVERRIDES_SOURCE = "endpoint-copies", "extra-copies", "class-overrides"
RECORD_PREFIX = "record:"  # a per-copy record's source name is this and its file name
RULE = (
    "science, 25 Oct ~05:40-05:50Z: a copy's speed class is the first source that gives one. 1. The v3 FINAL measurement "
    "(speed-class.json: copies.<copy>.speed_class and .basis) and the per-copy records (speed-class-<copy>*.json: copy, "
    "speed_class, basis): only fast, medium or slow is a class, so v3's partial is none and falls through; if v3 and a "
    "record both give a class for a copy and they differ the build refuses; v3's basis is used where it gives a class. "
    "2. The reviewed tier2 replay table cell-classes.tsv (external reviewer APPROVE of each rollout: the r21 table f4d0700e6679 plus r22's two "
    "uniprot-siteb-2 rows (sha256 bd3b83f8ec7e), r23's two uniprot-siteb-5 rows and r27's two uniprot-siteb-8 rows, "
    "sha256 6c4293552da7; r21's pre-r21 version a3a1881da084 differs only by the rhea-sitec-c13 "
    "row): a copy id on several lines has one class (else the build refuses), its basis is "
    "the note of its first line, and a note with the word unmeasured is that table's host rule (science, 24 Oct ~13:00Z), "
    "labelled '<class> (host rule, unmeasured)'. 3. The standing host rule (science, 25 Oct ~05:4xZ), for a copy in "
    "neither source, from copy-hosts.json (per copy its graph, host and basis): rhea-142 on sitea fast, sitec medium, "
    "siteb-x86 slow, arm64 medium; opaque-rhea-142 and uniprot-sp-2026_03 on sitea fast, sitec fast, siteb-x86 slow, "
    "arm64 medium; text2sparql-ck26 (corporate) has no host rule, so its class is null, labelled "
    f"'{NO_RULE_LABEL}'. Any other graph or host, or a copy found nowhere, refuses the build. The copies are the replica names of "
    "store-proofs.json plus those of endpoint-copies.json (the copies of the endpoints that OUTs with a single store "
    "endpoint used, whose deployment identities name no replica; science, 25 Oct ~05:3xZ: every copy a look reads has a "
    "class) plus those of extra-copies.json (analysis lead 5, 25 Oct ~09:2xZ and ~12:5xZ: the copies a look or a Tier 2 replay "
    "reads that neither file names, each with its graph and the registry that gives it; no endpoint, host or port); a copy in several of the "
    "three has one graph. The one exception to the order is the reviewed list class-overrides.json (analysis lead 5, 25 Oct "
    "~12:5xZ; science: a table row that is informational, not a measurement, is no class source): for the copies it names "
    "the table's row is ignored, so the copy falls to the host rule, and for the copies it gives a basis the entry from the "
    "table keeps its class and label and takes that basis; and for a copy measured after the last v3 FINAL and not yet issued as a "
    "per-copy record (uniprot-siteb-9, 25 Oct 13:29Z) a measured class gives it the class of the measurement, label the class, "
    "citing the measurement's files, only while v3, the records and the table have no class for it; an override with nothing to "
    "override refuses the build."
)
ENTRY_KEYS = {"class", "label", "basis", "source"}
_SHA = re.compile(r"[0-9a-f]{64}")


class Refused(Exception):
    """A class source or the committed file is malformed, or a copy has no class source (exploratory.py re-raises this
    as its own Refused)."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _object(raw: bytes, what: str) -> dict:
    try:
        doc = json.loads(raw)
    except ValueError:
        raise Refused(f"{what} is not JSON") from None
    if not isinstance(doc, dict):
        raise Refused(f"{what} is not a JSON object")
    return doc


def dumps(doc: Mapping[str, Any]) -> str:
    """The file's bytes: sorted keys, one space of indent, a final newline (the same document, the same bytes)."""
    return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


# --- the sources, parsed -------------------------------------------------------------------------------------------


def parse_v3(doc: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """speed-class.json -> {copy: (class, basis)} for the copies whose ``speed_class`` is a class (others fall through)."""
    copies = doc.get("copies")
    if not isinstance(copies, dict):
        raise Refused("the v3 speed-class file has no copies object")
    found = {}
    for copy, entry in copies.items():
        if isinstance(entry, dict) and entry.get("speed_class") in CLASSES:
            if not _text(entry.get("basis")):
                raise Refused(f"the v3 speed class of {copy} has no basis")
            found[copy] = (entry["speed_class"], entry["basis"])
    return found


def parse_record(name: str, doc: Mapping[str, Any]) -> tuple[str, str, str] | None:
    """A per-copy speed-class record -> (copy, class, basis), or None when its ``speed_class`` is not a class."""
    if not _text(doc.get("copy")):
        raise Refused(f"{name}: no copy named")
    if doc.get("speed_class") not in CLASSES:
        return None
    if not _text(doc.get("basis")):
        raise Refused(f"{name}: a speed class without a basis")
    return doc["copy"], doc["speed_class"], doc["basis"]


def parse_cell_rows(text: str) -> dict[str, list[tuple[str, str]]]:
    """cell-classes.tsv -> {copy: [(class, note) of each of its lines, in file order]}. Lines starting with # and blank lines
    are skipped; a line has exactly four tab-separated columns (endpoint, copy id, class, note), all filled, the class one of
    fast, medium, slow; a copy id on several lines has one class."""
    found: dict[str, list[tuple[str, str]]] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) != 4 or not all(c.strip() for c in columns):
            raise Refused(f"cell-classes line {number}: not four filled tab-separated columns")
        _, copy, speed, note = columns
        if speed not in CLASSES:
            raise Refused(f"cell-classes line {number}: {copy} has a class that is not one of {', '.join(CLASSES)}")
        if copy in found and found[copy][0][0] != speed:
            raise Refused(f"cell-classes: {copy} is on several lines with different classes")
        found.setdefault(copy, []).append((speed, note))
    return found


def parse_cell_classes(text: str) -> dict[str, tuple[str, str]]:
    """cell-classes.tsv -> {copy: (class, note of its first line)} (the checks of parse_cell_rows)."""
    return {copy: rows[0] for copy, rows in parse_cell_rows(text).items()}


def parse_hosts(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """copy-hosts.json (e189-copy-hosts/1) -> {copy: entry}: per copy its graph (one HOST_RULE names), host (one of
    HOSTS), basis and source {path, sha256}."""
    if set(doc) != {"schema", "rule", "copies"} or doc["schema"] != HOSTS_SCHEMA or not _text(doc["rule"]) \
            or not isinstance(doc["copies"], dict):
        raise Refused(f"the hosts file is not a {HOSTS_SCHEMA} file (schema, rule and copies, nothing else)")
    for copy, entry in doc["copies"].items():
        if not isinstance(entry, dict) or set(entry) != {"graph", "host", "basis", "source"}:
            raise Refused(f"the hosts entry of {copy} is not graph, host, basis and source")
        if not isinstance(entry["graph"], str) or entry["graph"] not in HOST_RULE:  # (an unhashable graph would raise)
            raise Refused(f"the hosts entry of {copy}: no host rule for its graph")
        if entry["host"] not in HOSTS:
            raise Refused(f"the hosts entry of {copy}: its host is not one of {', '.join(HOSTS)}")
        source = entry["source"]
        if not _text(entry["basis"]) or not (isinstance(source, dict) and set(source) == {"path", "sha256"}
                                              and _text(source["path"]) and isinstance(source["sha256"], str)
                                              and _SHA.fullmatch(source["sha256"])):
            raise Refused(f"the hosts entry of {copy} lacks its basis or its source (path and sha256)")
    return doc["copies"]


def proof_replicas(doc: Mapping[str, Any]) -> dict[str, str]:
    """store-proofs.json -> {replica: graph}: the copies the analysis can attribute."""
    proofs = doc.get("proofs") if doc.get("schema") == STORE_PROOFS_SCHEMA else None
    if not isinstance(proofs, list):
        raise Refused(f"the store proofs are not a {STORE_PROOFS_SCHEMA} file with a proofs list")
    found: dict[str, str] = {}
    for entry in proofs:
        if not isinstance(entry, dict) or not _text(entry.get("replica")) or not _text(entry.get("graph")):
            raise Refused("a store proof without a replica and a graph")
        if found.setdefault(entry["replica"], entry["graph"]) != entry["graph"]:
            raise Refused(f"{entry['replica']} is proven for two graphs")
    return found


def _sources_ok(sources: Any) -> bool:
    return (isinstance(sources, list) and bool(sources) and all(
        isinstance(s, dict) and set(s) == {"path", "sha256"} and _text(s["path"]) and isinstance(s["sha256"], str)
        and _SHA.fullmatch(s["sha256"]) is not None for s in sources))


def parse_endpoint_copies(doc: Any) -> list[dict[str, Any]]:
    """endpoint-copies.json (e189-endpoint-copies/1) -> its entries. Exactly schema, rule and a non-empty copies list;
    every entry exactly graph, endpoint_sha256 (64 hex: the sha256 of the endpoint string the records name, never the
    endpoint), replica, basis and sources (a non-empty list of {path, sha256}, 64 hex). No (graph, endpoint_sha256) is
    listed twice, and a replica is named for one endpoint of one graph only. The table names the copy behind an
    endpoint that an OUT with a single store endpoint used (its deployment identity names no replica); it NEVER counts
    as a proof of a copy: exploratory.check_stores reads it for the label of an endpoint that no proof entry covers, and
    a store-proofs entry or an identity that names another replica for that endpoint refuses."""
    if not isinstance(doc, dict) or set(doc) != {"schema", "rule", "copies"} or doc["schema"] != ENDPOINT_COPIES_SCHEMA \
            or not _text(doc["rule"]) or not isinstance(doc["copies"], list) or not doc["copies"]:
        raise Refused(f"the endpoint copies are not a {ENDPOINT_COPIES_SCHEMA} file (schema, rule and a non-empty "
                      f"copies list, nothing else)")
    keys: set[tuple[str, str]] = set()
    seen: dict[str, tuple[str, str]] = {}
    for entry in doc["copies"]:
        if not isinstance(entry, dict) or set(entry) != ENDPOINT_COPY_KEYS:
            raise Refused("an endpoint copy is not graph, endpoint_sha256, replica, basis and sources")
        graph, endpoint, replica = entry["graph"], entry["endpoint_sha256"], entry["replica"]
        if not _text(graph) or not _text(replica) or not _text(entry["basis"]):
            raise Refused("an endpoint copy lacks its graph, replica or basis")
        if not isinstance(endpoint, str) or _SHA.fullmatch(endpoint) is None:
            raise Refused(f"the endpoint copy of {replica}: endpoint_sha256 is not a sha256 (64 lowercase hex)")
        if not _sources_ok(entry["sources"]):
            raise Refused(f"the endpoint copy of {replica}: its sources are not a non-empty list of path and sha256")
        if (graph, endpoint) in keys:
            raise Refused(f"the endpoint copies list the endpoint sha256:{endpoint[:12]} of {graph} twice")
        keys.add((graph, endpoint))
        if seen.setdefault(replica, (graph, endpoint)) != (graph, endpoint):
            raise Refused(f"{replica} is named for two endpoints (or two graphs) in the endpoint copies")
    return doc["copies"]


def loads_endpoint_copies(raw: bytes) -> list[dict[str, Any]]:
    """An endpoint-copies document from its bytes, validated (read_endpoint_copies' parse step: a private table is
    validated by the same function, from the very bytes whose sha256 was checked)."""
    return parse_endpoint_copies(_object(raw, "the endpoint copies"))


def read_endpoint_copies(path: Path) -> tuple[list[dict[str, Any]], str]:
    """The committed endpoint-copies file, validated, and the sha256 of its bytes."""
    raw = Path(path).read_bytes()
    return loads_endpoint_copies(raw), hashlib.sha256(raw).hexdigest()


def merge_endpoint_copies(committed: Sequence[Mapping[str, Any]],
                          private: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The committed endpoint copies and a private table's, one list (the ``copies`` of two validated documents),
    validated as one document: no (graph, endpoint_sha256) twice, and a replica named for one endpoint of one graph
    only, whichever table names it. So an endpoint named by both tables is refused, identical or not. The refusal
    names no copy, graph or endpoint (the private table's are private)."""
    merged = [dict(entry) for entry in [*committed, *private]]
    try:
        return parse_endpoint_copies({"schema": ENDPOINT_COPIES_SCHEMA, "rule": "merged", "copies": merged})
    except Refused:  # the parser's refusals name the replica and the graph: not for a private table
        raise Refused("the private endpoint copies and the committed ones name an endpoint twice, or a replica for two "
                      "endpoints or graphs") from None


def endpoint_replicas(entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """endpoint-copies entries -> {replica: graph}: copies the analysis can attribute without a proof entry."""
    return {e["replica"]: e["graph"] for e in entries}


def parse_extra_copies(doc: Any) -> list[dict[str, Any]]:
    """extra-copies.json (e189-extra-copies/1) -> its entries. Exactly schema, rule and a non-empty copies list; every
    entry exactly replica, graph (one HOST_RULE names, so the host rule and the graph checks of the build can read it),
    basis and sources (a non-empty list of {path, sha256}, 64 hex: the registry that names the copy's graph). A replica
    is listed once. The file names a copy and its graph for the class table only: no endpoint, host or port (a test
    scans it), and it is no proof and no endpoint copy, so exploratory.py never reads it."""
    if not isinstance(doc, dict) or set(doc) != {"schema", "rule", "copies"} or doc["schema"] != EXTRA_COPIES_SCHEMA \
            or not _text(doc["rule"]) or not isinstance(doc["copies"], list) or not doc["copies"]:
        raise Refused(f"the extra copies are not a {EXTRA_COPIES_SCHEMA} file (schema, rule and a non-empty copies "
                      f"list, nothing else)")
    seen: set[str] = set()
    for entry in doc["copies"]:
        if not isinstance(entry, dict) or set(entry) != EXTRA_COPY_KEYS:
            raise Refused("an extra copy is not replica, graph, basis and sources")
        replica, graph = entry["replica"], entry["graph"]
        if not _text(replica) or not _text(entry["basis"]):
            raise Refused("an extra copy lacks its replica or basis")
        if not isinstance(graph, str) or graph not in HOST_RULE:  # (an unhashable graph would raise)
            raise Refused(f"the extra copy of {replica}: no host rule names its graph")
        if not _sources_ok(entry["sources"]):
            raise Refused(f"the extra copy of {replica}: its sources are not a non-empty list of path and sha256")
        if replica in seen:
            raise Refused(f"the extra copies list {replica} twice")
        seen.add(replica)
    return doc["copies"]


def read_extra_copies(path: Path) -> list[dict[str, Any]]:
    """The committed extra-copies file, validated (a file that cannot be read is Refused, never an OSError)."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise Refused(f"the extra copies file cannot be read ({type(exc).__name__})") from None
    return parse_extra_copies(_object(raw, "the extra copies"))


def extra_replicas(entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """extra-copies entries -> {replica: graph}."""
    return {e["replica"]: e["graph"] for e in entries}


def parse_overrides(doc: Any) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """class-overrides.json (e189-class-overrides/2) -> ({replica: entry} of the informational rows, of the bases, of the
    measured classes). Exactly schema, rule, informational_rows, bases and measured (each a list, possibly empty). An
    informational row is exactly replica, table_rows (a non-empty list of exactly {class, note}: ALL the cell-table lines of the
    copy, in file order, that the override was reviewed against; the build refuses when the table's lines for the copy differ in
    any way, so an override never outlives its rows and a line added later cannot slip past it), why and sources (a non-empty
    list of {path, sha256}, 64 hex); a basis is exactly replica, table_rows, basis, why and sources; a measured class is exactly
    replica, class (fast, medium or slow), basis, why and sources (the measurement's files). A replica is listed once in all (an
    informational row falls to the host rule, a basis keeps the table's entry, a measured class stands in for every source: one
    kind only)."""
    sections = (("informational_rows", OVERRIDE_INFO_KEYS), ("bases", OVERRIDE_BASIS_KEYS),
                ("measured", OVERRIDE_MEASURED_KEYS))
    if not isinstance(doc, dict) or set(doc) != {"schema", "rule", *(n for n, _ in sections)} \
            or doc["schema"] != OVERRIDES_SCHEMA or not _text(doc["rule"]) \
            or not all(isinstance(doc[n], list) for n, _ in sections):
        raise Refused(f"the class overrides are not a {OVERRIDES_SCHEMA} file (schema, rule, informational_rows, bases and "
                      f"measured, nothing else)")
    seen: set[str] = set()
    found: list[dict[str, dict[str, Any]]] = []
    for name, keys in sections:
        entries: dict[str, dict[str, Any]] = {}
        for entry in doc[name]:
            if not isinstance(entry, dict) or set(entry) != keys:
                raise Refused(f"a class override of {name} is not exactly {', '.join(sorted(keys))}")
            replica = entry["replica"]
            if not _text(replica) or not _text(entry["why"]) or (name != "informational_rows" and not _text(entry["basis"])):
                raise Refused(f"a class override of {name} lacks its replica, why or basis")
            if name == "measured" and entry["class"] not in CLASSES:
                raise Refused(f"the class override of {replica}: class is not one of {', '.join(CLASSES)}")
            if name != "measured":
                rows = entry["table_rows"]
                if not (isinstance(rows, list) and rows and all(
                        isinstance(r, dict) and set(r) == {"class", "note"} and r["class"] in CLASSES and _text(r["note"])
                        for r in rows)):
                    raise Refused(f"the class override of {replica}: table_rows are not a non-empty list of class "
                                  f"({', '.join(CLASSES)}) and note")
            if not _sources_ok(entry["sources"]):
                raise Refused(f"the class override of {replica}: its sources are not a non-empty list of path and sha256")
            if replica in seen:
                raise Refused(f"the class overrides name {replica} twice")
            seen.add(replica)
            entries[replica] = entry
        found.append(entries)
    return found[0], found[1], found[2]


def read_overrides(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """The committed class-overrides file, validated (a file that cannot be read is Refused, never an OSError)."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise Refused(f"the class overrides file cannot be read ({type(exc).__name__})") from None
    return parse_overrides(_object(raw, "the class overrides"))


# --- the precedence ------------------------------------------------------------------------------------------------


def classify(replica: str, graph: str, v3: Mapping[str, tuple[str, str]],
             records: Sequence[tuple[str, str, str, str]], cells: Mapping[str, tuple[str, str]],
             hosts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """One copy's entry {class, label, basis, source} by the precedence in RULE. ``records`` are (source name, copy,
    class, basis) in the order they are tried. v3 and the records must not disagree about a copy (checked by
    ``check_measurements``, which build runs first)."""
    if replica in v3:
        speed, basis = v3[replica]
        return {"class": speed, "label": speed, "basis": basis, "source": V3_SOURCE}
    for source, copy, speed, basis in records:
        if copy == replica:
            return {"class": speed, "label": speed, "basis": basis, "source": source}
    if replica in cells:
        speed, note = cells[replica]
        unmeasured = re.search(r"\bunmeasured\b", note) is not None
        return {"class": speed, "label": f"{speed} (host rule, unmeasured)" if unmeasured else speed, "basis": note,
                "source": CELLS_SOURCE}
    if replica not in hosts:
        raise Refused(f"{replica}: found in no class source and not in the hosts file")
    entry = hosts[replica]
    if entry["graph"] != graph:
        raise Refused(f"{replica}: the hosts file names another graph than its store proof does")
    by_host = HOST_RULE[graph]
    if by_host is None:
        return {"class": None, "label": NO_RULE_LABEL, "basis": entry["basis"], "source": HOSTS_SOURCE}
    speed = by_host[entry["host"]]
    return {"class": speed, "label": f"{speed} (host rule, unmeasured)", "basis": entry["basis"],
            "source": HOSTS_SOURCE}


def check_measurements(v3: Mapping[str, tuple[str, str]], records: Sequence[tuple[str, str, str, str]]) -> None:
    """v3 and the per-copy records agree about every copy they both give a class for (two records included)."""
    seen: dict[str, str] = {copy: speed for copy, (speed, _) in v3.items()}
    for source, copy, speed, _ in records:
        if seen.setdefault(copy, speed) != speed:
            raise Refused(f"{copy}: the measurements disagree on its class ({source} against another source)")


def build_doc(replicas: Mapping[str, str], v3: Mapping[str, tuple[str, str]],
              records: Sequence[tuple[str, str, str, str]], cells: Mapping[str, tuple[str, str]],
              hosts: Mapping[str, Mapping[str, Any]], sources: Mapping[str, Mapping[str, str]],
              overrides: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] = ({}, {}, {}),
              cell_rows: Mapping[str, Sequence[tuple[str, str]]] | None = None) -> dict[str, Any]:
    """The copy-classes document for ``replicas`` ({replica: graph}); ``sources`` are {name: {path, sha256}}. ``overrides`` are
    the parsed class overrides (parse_overrides): the informational rows are dropped from ``cells`` for the copies they name
    (each must be covered, have a table row and no measurement: v3 and the records come first, so an override there would
    do nothing), a basis replaces the basis of an entry that comes from the cell table (each copy must be covered), and a
    measured class gives a covered copy its class when NO other source does (it refuses if v3, a record or the table does: the
    measurement is then moot or needs review). Each informational row and basis holds ALL the table lines of its copy that it was reviewed against (``cell_rows``: {copy: [(class, note)]}, every
    line in file order; by default the one line ``cells`` keeps) and refuses when the table's lines differ in any way."""
    informational, bases, measured = overrides
    lines = cell_rows if cell_rows is not None else {copy: [row] for copy, row in cells.items()}

    def bound(replica: str, entry: Mapping[str, Any], what: str) -> None:
        """The table's lines of ``replica`` are the ones the override was reviewed against (every line, class and note)."""
        if replica not in lines:
            raise Refused(f"{replica}: its {what} override has no row in the cell table (a stale override)")
        if list(lines[replica]) != [(r["class"], r["note"]) for r in entry["table_rows"]]:
            raise Refused(f"{replica}: the cell table's lines are no longer the ones its {what} override was reviewed against "
                          f"(a class, a note or a line differs): review the lines, then the override")

    for replica in sorted(informational):
        if replica not in replicas:
            raise Refused(f"{replica}: an informational-row override for a copy the build does not cover")
        bound(replica, informational[replica], "informational-row")
        if replica in v3 or any(copy == replica for _, copy, _, _ in records):
            raise Refused(f"{replica}: its informational-row override is for a copy with a measurement, which comes first")
    check_measurements(v3, records)
    ordered = sorted(records)
    effective = {copy: row for copy, row in cells.items() if copy not in informational}
    for replica in sorted(measured):
        if replica not in replicas:
            raise Refused(f"{replica}: a measured-class override for a copy the build does not cover")
        if replica in v3 or any(copy == replica for _, copy, _, _ in records) or replica in lines:
            raise Refused(f"{replica}: its measured-class override is for a copy that now has a v3 row, a per-copy record or a "
                          f"cell-table row, which come first or need review: remove the override, or review it against them")
    copies = {r: classify(r, replicas[r], v3, ordered, effective, hosts) for r in sorted(replicas) if r not in measured}
    for replica, entry in measured.items():  # a measurement taken after the last v3 FINAL and not yet issued as a record
        copies[replica] = {"class": entry["class"], "label": entry["class"], "basis": entry["basis"], "source": OVERRIDES_SOURCE}
    copies = {r: copies[r] for r in sorted(copies)}
    for replica in sorted(bases):
        if replica not in copies:
            raise Refused(f"{replica}: a basis override for a copy the build does not cover")
        if copies[replica]["source"] != CELLS_SOURCE:
            raise Refused(f"{replica}: a basis override is for an entry that comes from the cell table, not from "
                          f"{copies[replica]['source']}")
        bound(replica, bases[replica], "basis")
        copies[replica] = {**copies[replica], "basis": bases[replica]["basis"]}
    return {"schema": SCHEMA, "rule": RULE, "sources": {k: dict(v) for k, v in sorted(sources.items())},
            "copies": copies}


# --- the committed file, read --------------------------------------------------------------------------------------


def validate(doc: Any) -> dict[str, Any]:
    """The document, or Refused: exactly schema, rule, sources and copies; every source a path and a sha256; every copy
    exactly class, label, basis and source, the class fast, medium, slow or null, the label the one that class and
    source allow, and the source one of the document's."""
    if not isinstance(doc, dict) or set(doc) != {"schema", "rule", "sources", "copies"}:
        raise Refused("the copy classes are not a document of schema, rule, sources and copies")
    if doc["schema"] != SCHEMA:
        raise Refused("the copy classes are of another schema")
    if not _text(doc["rule"]):
        raise Refused("the copy classes have no rule")
    sources, copies = doc["sources"], doc["copies"]
    if not isinstance(sources, dict) or not sources:
        raise Refused("the copy classes have no sources")
    for name, source in sources.items():
        if not (_text(name) and isinstance(source, dict) and set(source) == {"path", "sha256"} and _text(source["path"])
                and isinstance(source["sha256"], str) and _SHA.fullmatch(source["sha256"])):
            raise Refused("a source of the copy classes is not a path and a sha256")
    if not isinstance(copies, dict) or not copies:
        raise Refused("the copy classes have no copies")
    for copy, entry in copies.items():
        if not _text(copy) or not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
            raise Refused(f"the copy class of {copy} is not class, label, basis and source")
        speed = entry["class"]
        if speed is not None and speed not in CLASSES:
            raise Refused(f"the copy class of {copy} is not one of {', '.join(CLASSES)} or null")
        allowed = [NO_RULE_LABEL] if speed is None else [speed, f"{speed} (host rule, unmeasured)"]
        if entry["label"] not in allowed:
            raise Refused(f"the label of {copy} does not match its class")
        # (a source that is a list or an object is unhashable: ``in`` on the dict would raise TypeError, not refuse)
        if not _text(entry["basis"]) or not isinstance(entry["source"], str) or entry["source"] not in sources:
            raise Refused(f"the copy class of {copy} lacks its basis or names a source the file does not list")
    return doc


def loads(raw: bytes) -> dict[str, Any]:
    """A copy-classes document from its bytes, validated (read's parse step: a private table is validated by the same
    function, from the very bytes whose sha256 was checked)."""
    return validate(_object(raw, "the copy classes"))


def read(path: Path) -> tuple[dict[str, Any], str]:
    """The committed copy-classes file, validated, and the sha256 of its bytes."""
    raw = Path(path).read_bytes()
    return loads(raw), hashlib.sha256(raw).hexdigest()


def merge_classes(committed: Mapping[str, Mapping[str, Any]],
                  private: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The committed copy classes and a private table's, one table (the ``copies`` objects of two validated documents):
    every copy of either. A copy in both is refused unless its two entries are identical (class, label, basis and
    source); the committed entry is never changed by the private one, nor the other way round. The refusal names no
    copy (the private table's names are private)."""
    merged = {copy: dict(entry) for copy, entry in committed.items()}
    for copy, entry in private.items():
        if copy in merged and merged[copy] != dict(entry):
            raise Refused("a copy is in the committed copy classes and in the private table with different entries")
        merged[copy] = dict(entry)
    return merged


# --- the command ---------------------------------------------------------------------------------------------------


def build(store_proofs: Path, v3: Path, records: Sequence[Path], cell_classes: Path, hosts: Path,
          endpoint_copies: Path, extra_copies: Path, class_overrides: Path) -> dict[str, Any]:
    """Read the sources (a path is recorded as given) and build the document. The copies covered are store-proofs.json's
    replicas plus endpoint-copies.json's plus extra-copies.json's; a replica in several has one graph (else Refused)."""
    sources = {PROOFS_SOURCE: store_proofs, V3_SOURCE: v3, CELLS_SOURCE: cell_classes, HOSTS_SOURCE: hosts,
               ENDPOINTS_SOURCE: endpoint_copies, EXTRAS_SOURCE: extra_copies, OVERRIDES_SOURCE: class_overrides}
    parsed = []
    for path in records:
        name = RECORD_PREFIX + Path(path).name
        if name in sources:
            raise Refused(f"two records of one file name ({Path(path).name})")
        sources[name] = path
        found = parse_record(name, _object(Path(path).read_bytes(), name))
        if found is not None:
            parsed.append((name, *found))
    replicas = proof_replicas(_object(Path(store_proofs).read_bytes(), "the store proofs"))
    for replica, graph in endpoint_replicas(read_endpoint_copies(endpoint_copies)[0]).items():
        if replicas.setdefault(replica, graph) != graph:
            raise Refused(f"{replica} is of one graph in the store proofs and of another in the endpoint copies")
    for replica, graph in extra_replicas(read_extra_copies(extra_copies)).items():
        if replicas.setdefault(replica, graph) != graph:
            raise Refused(f"{replica} is of one graph in the extra copies and of another in the store proofs or the "
                          f"endpoint copies")
    overrides = read_overrides(class_overrides)  # (before the sources are hashed: a missing file is a refusal)
    cell_rows = parse_cell_rows(Path(cell_classes).read_text(encoding="utf-8"))
    cells = {copy: rows[0] for copy, rows in cell_rows.items()}  # (parse_cell_classes' map, from the same lines)
    return build_doc(replicas, parse_v3(_object(Path(v3).read_bytes(), "the v3 speed-class file")), parsed, cells,
                     parse_hosts(_object(Path(hosts).read_bytes(), "the hosts file")),
                     {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in sources.items()},
                     overrides, cell_rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write copy-classes.json from the class sources")
    b.add_argument("--store-proofs", type=Path, required=True, help="store-proofs.json: the copies covered")
    b.add_argument("--v3", type=Path, required=True, help="speed-class.json, the v3 FINAL measurement")
    b.add_argument("--record", type=Path, action="append", default=[],
                   help="a per-copy speed-class record (repeat; never a glob: a superseded file would match)")
    b.add_argument("--cell-classes", type=Path, required=True, help="cell-classes.tsv")
    b.add_argument("--hosts", type=Path, required=True, help="copy-hosts.json")
    b.add_argument("--endpoint-copies", type=Path, required=True,
                   help="endpoint-copies.json: the copies of the endpoints no proof entry covers (required: an "
                        "omitted table would silently leave those copies without a class)")
    b.add_argument("--extra-copies", type=Path, required=True,
                   help="extra-copies.json: the copies a look reads that neither store-proofs.json nor "
                        "endpoint-copies.json names (required: an omitted file would silently leave those copies "
                        "without a class)")
    b.add_argument("--class-overrides", type=Path, required=True,
                   help="class-overrides.json: the reviewed overrides of the precedence (required: an omitted file would "
                        "silently give an informational table row the class it was ruled not to give)")
    b.add_argument("--out", type=Path, required=True, help="the file to write (copy-classes.json)")
    args = parser.parse_args(argv)
    try:
        doc = build(args.store_proofs, args.v3, args.record, args.cell_classes, args.hosts,
                    args.endpoint_copies, args.extra_copies, args.class_overrides)
        validate(doc)
    except Refused as exc:
        print(f"copy_classes: refused: {exc}", file=sys.stderr)
        return 2
    args.out.write_text(dumps(doc), encoding="utf-8")
    print(f"copy_classes: wrote {args.out} (sha256 {sha256_file(args.out)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
