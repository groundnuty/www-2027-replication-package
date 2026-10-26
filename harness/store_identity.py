"""Which store an endpoint serves.

Strong identity: QLever's ``?cmd=stats`` reports ``name-index``, which
``stores/serve.sbatch`` sets with ``--index-description "<store>@<digest>"``,
where the digest is the sha256 of the store's committed ``build/index.sha256``
(the checksums of the exact index files served). A matching ``name-index`` ties
answers to those files.

Weaker identity, only when the caller allows it (``--allow-counts-identity``,
for servers started before the index description existed): the endpoint's
per-graph triple counts must equal the store's recorded counts exactly. The
identity record says which of the two was used.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from harness.sparql import SPARQL_JSON
from harness.terms import rows_from_sparql_json

SP_STORE = "uniprot-sp-2026_03"
RHEA_STORE = "rhea-142"

# Per-graph triple counts from stores/<store>/build/checks.json "per_graph" (feat/stores at fc6390d).
SP_2026_03_GRAPH_COUNTS = {
    "http://purl.uniprot.org/core": 2_835,
    "http://sparql.uniprot.org/chebi": 3_426_039,
    "http://sparql.uniprot.org/citations": 30_735_083,
    "http://sparql.uniprot.org/databases": 2_407,
    "http://sparql.uniprot.org/diseases": 91_009,
    "http://sparql.uniprot.org/enzymes": 182_712,
    "http://sparql.uniprot.org/go": 655_968,
    "http://sparql.uniprot.org/journal": 43_700,
    "http://sparql.uniprot.org/keywords": 13_875,
    "http://sparql.uniprot.org/locations": 6_781,
    "http://sparql.uniprot.org/pathways": 17_823,
    "http://sparql.uniprot.org/proteomes": 44_980_519,
    "http://sparql.uniprot.org/taxonomy": 70_047_936,
    "http://sparql.uniprot.org/tissues": 4_115,
    "http://sparql.uniprot.org/uniprot": 251_975_923,
    "https://sparql.rhea-db.org/rhea": 2_053_334,
}
RHEA_142_GRAPH_COUNTS = {
    "https://sparql.rhea-db.org/rhea": 2_053_334,
    "http://sparql.rhea-db.org/chebi": 3_426_039,
}

# index_digest = sha256 of the committed stores/<store>/build/index.sha256 (feat/stores at fc6390d).
EXPECTED_STORES: dict[str, dict[str, Any]] = {
    "sp": {
        "store": SP_STORE,
        "index_digest": "9484989e47e8903a78937402a0b3c69b43a12499721ea71d3c940ec9e62a62b1",
        "graph_counts": SP_2026_03_GRAPH_COUNTS,
    },
    "rhea": {
        "store": RHEA_STORE,
        "index_digest": "c0174e728db659c40c96b988a7c96c1a7f4fa3b396534ca97e16594271e38a33",
        "graph_counts": RHEA_142_GRAPH_COUNTS,
    },
}

STRONG = "index_description"
WEAK = "graph_counts"
WEAK_LEGACY = "graph_counts_legacy"  # results from before this module: a counts fingerprint only
GRAPH_COUNT_QUERY = "SELECT ?g (COUNT(*) AS ?n) WHERE { GRAPH ?g { ?s ?p ?o } } GROUP BY ?g"
_DESCRIBED = re.compile(r"^\S+@[0-9a-f]{64}$")


class StoreIdentityError(RuntimeError):
    """An endpoint is not the store the run must use, or its identity cannot be established."""

    def __init__(self, message: str, identity: dict[str, Any] | None = None):
        super().__init__(message)
        self.identity = identity


def expected_identity(spec: dict[str, Any]) -> str:
    return f"{spec['store']}@{spec['index_digest']}"


def legacy_counts_fingerprint(counts: dict[str, int]) -> str:
    """The counts fingerprint the runner recorded before index identities (sha256 of sorted graph/count lines)."""
    return hashlib.sha256("".join(f"{g}\t{n}\n" for g, n in sorted(counts.items())).encode()).hexdigest()


def read_name_index(http: httpx.Client, endpoint: str, timeout_s: float) -> tuple[str | None, str | None]:
    """QLever's ``name-index`` from ``?cmd=stats``, or (None, why not)."""
    try:
        resp = http.get(endpoint, params={"cmd": "stats"}, timeout=timeout_s)
    except httpx.HTTPError as exc:
        return None, f"stats request failed: {exc!r}"
    if resp.status_code != 200:
        return None, f"stats request answered HTTP {resp.status_code}"
    try:
        data = resp.json()
    except ValueError:
        return None, "stats answer is not JSON"
    name = data.get("name-index") if isinstance(data, dict) else None
    return (name, None) if isinstance(name, str) else (None, "stats answer has no name-index")


def graph_counts(http: httpx.Client, endpoint: str, timeout_s: float) -> dict[str, int]:
    """Triples per named graph of an endpoint (one GROUP BY)."""
    try:
        resp = http.post(
            endpoint,
            content=GRAPH_COUNT_QUERY.encode(),
            headers={"Content-Type": "application/sparql-query", "Accept": SPARQL_JSON},
            params={"timeout": f"{timeout_s:g}s"},
            timeout=timeout_s + 10,
        )
    except httpx.HTTPError as exc:
        raise StoreIdentityError(f"could not read per-graph triple counts from {endpoint}: {exc!r}") from exc
    if resp.status_code != 200:
        raise StoreIdentityError(f"could not read per-graph triple counts from {endpoint}: HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        _, rows = rows_from_sparql_json(resp.json())
        return {g.value: int(n.lexical) for g, n in rows}
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise StoreIdentityError(f"unexpected answer to the graph-count query from {endpoint}: {exc!r}") from exc


def verify_store(
    http: httpx.Client,
    endpoint: str,
    spec: dict[str, Any],
    timeout_s: float = 600,
    allow_counts: bool = False,
) -> dict[str, Any]:
    """Establish that ``endpoint`` serves the store ``spec`` describes, or raise StoreIdentityError.

    A ``name-index`` equal to "<store>@<digest>" is the strong identity. A
    ``name-index`` naming a different "<store>@<digest>" is always refused. If
    the server reports no such description, the exact per-graph counts are
    accepted only with ``allow_counts`` and recorded as the weaker identity.
    """
    want = expected_identity(spec)
    rec: dict[str, Any] = {
        "endpoint": endpoint,
        "store": spec["store"],
        "expected_identity": want,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    name, why = read_name_index(http, endpoint, timeout_s)
    rec["name_index"] = name
    if name == want:
        rec.update(identity=want, method=STRONG, weak=False, verified=True)
        return rec
    if name is not None and _DESCRIBED.match(name):
        raise StoreIdentityError(f"{endpoint} serves index {name}, expected {want}", {**rec, "verified": False})
    if not allow_counts:
        detail = f"name-index {name!r}" + (f"; {why}" if why else "")
        raise StoreIdentityError(
            f"{endpoint} does not report the index identity {want} ({detail}); pass --allow-counts-identity "
            "to accept the store's exact per-graph triple counts instead (weaker)",
            {**rec, "verified": False},
        )
    counts = graph_counts(http, endpoint, timeout_s)
    rec.update(graph_counts=dict(sorted(counts.items())), expected_graph_counts=dict(sorted(spec["graph_counts"].items())))
    if counts != spec["graph_counts"]:
        raise StoreIdentityError(
            f"{endpoint} is not the expected store {spec['store']}: graph counts {rec['graph_counts']}, "
            f"expected exactly {rec['expected_graph_counts']}",
            {**rec, "verified": False},
        )
    rec.update(identity=want, method=WEAK, weak=True, verified=True)
    return rec


def meta_identity(meta: dict[str, Any], key: str, expected: dict[str, dict[str, Any]] = EXPECTED_STORES) -> tuple[str | None, str | None]:
    """(identity, method) a meta record was made under, for ``key`` "sp" or "rhea".

    Records from before index identities carry only a counts fingerprint; one
    that equals the expected store's counts maps to that store's identity with
    the method ``graph_counts_legacy`` (as weak as ``graph_counts``).
    """
    if meta.get(f"{key}_identity"):
        return meta[f"{key}_identity"], meta.get(f"{key}_identity_method")
    fp = meta.get(f"{key}_fingerprint")
    if fp and fp == legacy_counts_fingerprint(expected[key]["graph_counts"]):
        return expected_identity(expected[key]), WEAK_LEGACY
    return None, None
