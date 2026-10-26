"""The pinned store, for the fault bank: running gold and mutated queries, and the oracle the
operators ask (absence, a constant's classes, class members, stale identifiers).

Answer effect (issue E1.3, criterion 4): a mutated query that runs is answer-neutral when the
scorer finds its rows exactly equal to the gold query's rows on the same store, under the
question's scoring rules, and answer-changing otherwise. A result cut by the row cap, or refused
as too large, is answer-changing (gold results hold at most a few dozen rows); the basis is kept so
it can be excluded. A mutation that rdflib cannot parse, or that the store refuses (parse error,
time-out, server error), is rejected and counted per operator. A transport failure is not a
rejection: the build stops, so an unreachable store cannot pass as a column of rejections.

Store identity (review #2 of 3afbfa1): results are cached per store. The store is identified
before anything runs, against the committed store files: strongly by the index description that
``stores/serve.sbatch`` sets (``<store>@<sha256 of build/index.sha256>``, read from QLever's
``?cmd=stats``), or, on a server that reports none, by its per-graph triple counts, which must equal
``stores/<store>/store.json`` exactly. Every cache entry carries that key, and a cache holding an
entry of another store, or of none, is refused rather than reused.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.faults.query import parse_error
from harness.score.rules import ScoringRules
from harness.score.scorer import score_rows
from harness.sparql import SparqlError
from harness.terms import IRI, Row, row_from_json, row_to_json

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASSOF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
MAX_CHAIN = 200  # data nodes followed per constant when reading its classes
STORES_DIR = Path(__file__).resolve().parents[2] / "stores"
GRAPH_COUNT_QUERY = "SELECT ?g (COUNT(*) AS ?n) WHERE { GRAPH ?g { ?s ?p ?o } } GROUP BY ?g"
NO_INDEX_NAME = "no index name specified"  # what QLever reports when started without --index-description


class StoreMismatch(RuntimeError):
    """The endpoint is not the pinned store, or cannot be shown to be."""


class CacheStoreMismatch(RuntimeError):
    """A result cache built against another store (or an unknown one)."""


@dataclass(frozen=True)
class PinnedStore:
    name: str
    identity: str | None  # the index description a reviewed server reports
    graph_counts: dict[str, int]

    @classmethod
    def load(cls, name: str, stores_dir: Path = STORES_DIR) -> PinnedStore:
        """From the committed ``stores/<name>/store.json`` and ``build/index.sha256``."""
        meta = json.loads((stores_dir / name / "store.json").read_text())
        identity = meta.get("identity")
        checksums = stores_dir / name / "build" / "index.sha256"
        if identity is None and checksums.exists():
            identity = f"{name}@{hashlib.sha256(checksums.read_bytes()).hexdigest()}"
        return cls(name, identity, {g: int(n) for g, n in meta["graphs"].items()})


def fetch_stats(endpoint: str) -> dict | None:
    """QLever's ``?cmd=stats``, or None when it cannot be read (then the counts decide)."""
    import httpx

    try:
        return httpx.get(endpoint.rstrip("/") + "/?cmd=stats", timeout=30).json()
    except (httpx.HTTPError, ValueError):
        return None


def store_identity(client: Any, pinned: PinnedStore, stats: dict | None) -> dict:
    """Which store answers, checked live (never from a cache). ``key`` is what cache entries carry."""
    name = (stats or {}).get("name-index")
    if name and name != NO_INDEX_NAME:
        if name != pinned.identity:
            raise StoreMismatch(f"the server reports index {name!r}, not the pinned {pinned.identity!r}")
        return {"basis": "index_description", "key": name, "name_index": name}
    res = client.query(GRAPH_COUNT_QUERY)
    counts = {row[0].value: int(row[1].lexical) for row in res.rows if row[0] is not None and row[1] is not None}
    if res.truncated or counts != pinned.graph_counts:
        raise StoreMismatch(f"per-graph triple counts {counts} are not the pinned {pinned.graph_counts} of {pinned.name}")
    fingerprint = hashlib.sha256(json.dumps(sorted(counts.items())).encode()).hexdigest()
    return {"basis": "graph_counts", "key": f"{pinned.name}@graph-counts:{fingerprint}",
            "graph_counts": dict(sorted(counts.items())), "name_index": name}


@dataclass
class Execution:
    status: str  # ok, error
    rows: list[Row] | None = None
    boolean: bool | None = None
    truncated: bool = False
    error_kind: str | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    columns: list[str] | None = None  # result variables, in row order (None in caches written before they were kept)

    def to_json(self) -> dict[str, Any]:
        return {"status": self.status, "rows": None if self.rows is None else [row_to_json(r) for r in self.rows],
                "boolean": self.boolean, "truncated": self.truncated, "error_kind": self.error_kind,
                "error": self.error, "elapsed_s": round(self.elapsed_s, 3), "columns": self.columns}

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> Execution:
        rows = None if obj["rows"] is None else [row_from_json(r) for r in obj["rows"]]
        return cls(obj["status"], rows, obj["boolean"], obj["truncated"], obj["error_kind"], obj["error"], obj["elapsed_s"],
                   obj.get("columns"))


def query_sha(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


@dataclass
class Runner:
    """One query at a time, each result cached by the query's hash (in memory, and in a JSONL file
    when given, so an interrupted build resumes without re-running anything). A file cache is
    keyed by ``store`` (``store_identity``'s key): entries of any other store are refused."""

    client: Any
    cache_path: Path | None = None
    store: str | None = None
    retry_wait_s: float = 10.0
    _cache: dict[str, Execution] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.cache_path is not None and self.store is None:
            raise ValueError("a file cache must name the store its results come from")
        if self.cache_path is not None and self.cache_path.exists():
            for n, line in enumerate(self.cache_path.read_text().splitlines(), 1):
                rec = json.loads(line)
                if rec.get("store") != self.store:
                    raise CacheStoreMismatch(f"{self.cache_path}:{n} was built against {rec.get('store')!r}, "
                                             f"not {self.store!r}; build into a fresh directory")
                self._cache[rec["sha"]] = Execution.from_json(rec["execution"])

    def run(self, query: str) -> Execution:
        key = query_sha(query)
        if key in self._cache:
            return self._cache[key]
        error = parse_error(query)
        if error is not None:
            ex = Execution("error", error_kind="parse_rdflib", error=error)
        else:
            ex = self._execute(query)
        self._cache[key] = ex
        if self.cache_path is not None:
            with self.cache_path.open("a") as fh:
                fh.write(json.dumps({"sha": key, "store": self.store, "execution": ex.to_json()}, sort_keys=True) + "\n")
        return ex

    def _execute(self, query: str) -> Execution:
        for attempt in range(4):
            try:
                res = self.client.query(query)
            except SparqlError as exc:
                if exc.kind == "transport":
                    if attempt == 3:
                        raise
                    time.sleep(self.retry_wait_s * (attempt + 1))
                    continue
                return Execution("error", error_kind=exc.kind, error=str(exc)[:500])
            return Execution("ok", list(res.rows), res.boolean, bool(res.truncated), elapsed_s=res.elapsed_s,
                             columns=list(res.columns))
        raise AssertionError("unreachable")


def bank_rules(rules: ScoringRules) -> ScoringRules:
    """The rules a bank label is decided under: the given settings with every scorer-rules-v5 rule off (change-log
    entry 120), so the frozen banks keep the v4 semantics they were labelled under and the new rules reach agent
    outputs only."""
    return rules.as_v4()


def answer_effect(gold: Execution, mutated: Execution, rules: ScoringRules) -> dict[str, Any]:
    rules = bank_rules(rules)
    if mutated.status != "ok":
        if mutated.error_kind == "too_large":
            return {"label": "answer_changing", "basis": "too_large"}
        reason = "parse_rdflib" if mutated.error_kind == "parse_rdflib" else f"store_{mutated.error_kind}"
        return {"label": "rejected", "reason": reason, "detail": mutated.error}
    if mutated.truncated:
        return {"label": "answer_changing", "basis": "truncated"}
    if mutated.boolean is not None or gold.boolean is not None:
        same = mutated.boolean == gold.boolean
        return {"label": "answer_neutral" if same else "answer_changing", "basis": "ask"}
    score = score_rows(mutated.rows or [], gold.rows or [], rules)
    out = {"label": "answer_neutral" if score.exact else "answer_changing", "basis": "scorer",
           "n_rows": score.n_answer_rows, "n_gold_rows": score.n_gold_rows, "n_matched": score.n_matched}
    return out


def _iris(ex: Execution) -> list[str]:
    if ex.status != "ok":
        raise RuntimeError(f"oracle query failed: {ex.error_kind}: {ex.error}")
    return [row[0].value for row in ex.rows or [] if row and isinstance(row[0], IRI)]


@dataclass
class SparqlOracle:
    """What the operators ask the store. Every query names the exact IRI it asks about, so none
    scans unboundedly."""

    runner: Runner
    data_graph: str
    named_classes: frozenset[str]
    stale_sources: tuple[str, ...] = ()
    rhea_graph: str = "https://sparql.rhea-db.org/rhea"

    def absent(self, iri: str) -> bool:
        q = f"ASK {{ {{ <{iri}> ?p ?o }} UNION {{ ?s <{iri}> ?o }} UNION {{ ?s ?p <{iri}> }} }}"
        ex = self.runner.run(q)
        if ex.status != "ok" or ex.boolean is None:
            raise RuntimeError(f"absence query failed for {iri}: {ex.error}")
        return not ex.boolean

    def classes_of(self, iri: str) -> tuple[str, ...]:
        """rdf:type objects in any graph, and data-graph subclass objects followed through data
        nodes (check contract rule 35); only named classes are kept."""
        found: set[str] = set()
        found.update(c for c in _iris(self.runner.run(f"SELECT DISTINCT ?c WHERE {{ <{iri}> <{RDF_TYPE}> ?c }}"))
                     if c in self.named_classes)
        frontier, seen = [iri], {iri}
        while frontier and len(seen) < MAX_CHAIN:
            node = frontier.pop()
            q = f"SELECT DISTINCT ?c WHERE {{ GRAPH <{self.data_graph}> {{ <{node}> <{RDFS_SUBCLASSOF}> ?c }} }}"
            for c in _iris(self.runner.run(q)):
                if c in self.named_classes:
                    found.add(c)
                elif c not in seen:
                    seen.add(c)
                    frontier.append(c)
        return tuple(sorted(found))

    def _members_pattern(self, cls: str) -> str:
        return (f"{{ ?k <{RDF_TYPE}> <{cls}> }} UNION {{ GRAPH <{self.data_graph}> {{ ?k <{RDFS_SUBCLASSOF}> <{cls}> }} }} "
                "FILTER(isIRI(?k))")

    def members(self, cls: str) -> int:
        ex = self.runner.run(f"SELECT (COUNT(DISTINCT ?k) AS ?n) WHERE {{ {self._members_pattern(cls)} }}")
        if ex.status != "ok":
            raise RuntimeError(f"member count failed for {cls}: {ex.error}")
        return int(ex.rows[0][0].lexical) if ex.rows and ex.rows[0][0] is not None else 0

    def member(self, cls: str, offset: int) -> str | None:
        ex = self.runner.run(f"SELECT DISTINCT ?k WHERE {{ {self._members_pattern(cls)} }} ORDER BY ?k OFFSET {offset} LIMIT 1")
        found = _iris(ex)
        return found[0] if found else None

    # The type-consistent swap reads membership in the graph's own convention (GraphProfile.membership):
    # rdf:type in any graph, and a subClassOf membership predicate in the data graph only (rule 35), so
    # on UniProt the data graph's entry-local subclass nodes are never members. ``members`` and
    # ``classes_of`` above stay as the type-wrong swaps read them.
    def _membership(self, node: str, cls: str, membership: str) -> str:
        pattern = f"{{ {node} <{RDF_TYPE}> {cls} }}"
        if membership != RDF_TYPE:
            pattern += f" UNION {{ GRAPH <{self.data_graph}> {{ {node} <{membership}> {cls} }} }}"
        return pattern

    def own_classes(self, iri: str, membership: str) -> tuple[str, ...]:
        """The IRIs a node is a member of, by the graph's membership predicate (meta classes included:
        the operator decides what is a class)."""
        q = f"SELECT DISTINCT ?c WHERE {{ {self._membership(f'<{iri}>', '?c', membership)} FILTER(isIRI(?c)) }}"
        return tuple(sorted(_iris(self.runner.run(q))))

    def _peers(self, cls: str, membership: str, exclude: str) -> str:
        return f"{self._membership('?k', f'<{cls}>', membership)} FILTER(isIRI(?k) && ?k != <{exclude}>)"

    def class_size(self, cls: str, membership: str, exclude: str) -> int:
        ex = self.runner.run(f"SELECT (COUNT(DISTINCT ?k) AS ?n) WHERE {{ {self._peers(cls, membership, exclude)} }}")
        if ex.status != "ok":
            raise RuntimeError(f"member count failed for {cls}: {ex.error}")
        return int(ex.rows[0][0].lexical) if ex.rows and ex.rows[0][0] is not None else 0

    def class_member(self, cls: str, membership: str, exclude: str, offset: int) -> str | None:
        ex = self.runner.run(f"SELECT DISTINCT ?k WHERE {{ {self._peers(cls, membership, exclude)} }} "
                             f"ORDER BY ?k OFFSET {offset} LIMIT 1")
        found = _iris(ex)
        return found[0] if found else None

    def _ask(self, query: str) -> bool:
        ex = self.runner.run(query)
        if ex.status != "ok" or ex.boolean is None:
            raise RuntimeError(f"store-fact query failed: {ex.error_kind}: {ex.error}")
        return ex.boolean

    def facts(self, iri: str) -> dict[str, bool]:
        """The store facts rule 32's data-node test reads (amended text, change-log entry 26)."""
        g = self.data_graph
        w3c = " && ".join(f'!STRSTARTS(STR(?p), "{ns}")' for ns in (
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://www.w3.org/2000/01/rdf-schema#",
            "http://www.w3.org/2002/07/owl#"))
        return {
            "membership_target": self._ask(f"ASK {{ {{ ?s <{RDF_TYPE}> <{iri}> }} UNION "
                                           f"{{ GRAPH <{g}> {{ ?s <{RDFS_SUBCLASSOF}> <{iri}> }} }} }}"),
            "modelled_as_class": self._ask(f"ASK {{ GRAPH <{g}> {{ <{iri}> <{RDFS_SUBCLASSOF}> ?o FILTER(isIRI(?o)) }} }}"),
            "data_subject": self._ask(f"ASK {{ GRAPH <{g}> {{ <{iri}> ?p ?o FILTER({w3c}) }} }}"),
            "used_as_predicate": self._ask(f"ASK {{ ?s <{iri}> ?o }}"),
        }

    def stale(self, iri: str) -> tuple[tuple[str, str], ...]:
        out: set[tuple[str, str]] = set()
        if "obo_replaced_by" in self.stale_sources:
            q = ("SELECT DISTINCT ?old WHERE { ?old <http://purl.obolibrary.org/obo/IAO_0100001> "
                 f"<{iri}> . ?old <http://www.w3.org/2002/07/owl#deprecated> true }}")
            out.update((o, "obo_replaced_by") for o in _iris(self.runner.run(q)))
        m = re.fullmatch(r"http://rdf\.rhea-db\.org/(\d+)", iri)
        if m and "rhea_obsolete_comment" in self.stale_sources:
            n = m.group(1)
            q = (f"SELECT DISTINCT ?old ?c WHERE {{ GRAPH <{self.rhea_graph}> {{ ?old <http://rdf.rhea-db.org/status> "
                 "<http://rdf.rhea-db.org/Obsolete> ; <http://www.w3.org/2000/01/rdf-schema#comment> ?c . "
                 f'FILTER(CONTAINS(STR(?c), "RHEA:{n}")) }} }}')
            ex = self.runner.run(q)
            if ex.status != "ok":
                raise RuntimeError(f"stale query failed for {iri}: {ex.error}")
            pattern = re.compile(rf"replaced by RHEA:{n}(?!\d)")
            out.update((row[0].value, "rhea_obsolete_comment") for row in ex.rows or []
                       if isinstance(row[0], IRI) and row[1] is not None and pattern.search(row[1].lexical))
        if "uniprot_replaces" in self.stale_sources:
            q = (f"SELECT DISTINCT ?old WHERE {{ GRAPH <{self.data_graph}> {{ <{iri}> "
                 "<http://purl.uniprot.org/core/replaces> ?old } }")
            out.update((o, "uniprot_replaces") for o in _iris(self.runner.run(q)))
        return tuple(sorted(out))
