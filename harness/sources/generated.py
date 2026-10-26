"""The LLM-generated source: claude-opus-5-5 writes an OWL ontology from the census.

Proposal §7.3: the generator "receives exactly what the data-derived source is
built from: the census of classes, predicates and co-occurrence, plus a fixed
sample of instances per class", and nothing else (not the curated schema). It
returns an OWL ontology in Turtle over the data's own identifiers. Three
independent draws are made from one input; draw 1 is fixed now, before any
run, as the draw the prompt cells use (``PROMPT_CELL_DRAW``).

The input bundle
    * the census record, serialised losslessly (``census_block``): every class
      with its observed superclasses, ``type_support`` (direct counts),
      ``class_support`` (membership counts), ``position_support`` (classes and
      datatypes observed at each predicate position), ``untyped_fillers``,
      every pattern with its support and reading, every derived domain and
      range, the threshold and the membership reading, with compact IRIs. The
      census's build provenance (queries, counts) is left out. These are the
      same figures the census prompt renders and the checker reads (rule 20);
    * the instance sample (``sample_instances``): strata are the census classes
      with at least ``min_members`` direct data members (a data node whose
      rdf:type or rdfs:subClassOf object is the class); from each stratum the
      ``k`` members with the smallest sha256(seed + "|" + IRI), ties by IRI,
      with all their outgoing triples in the census input (the first
      ``max_triples`` in (predicate, object) order).
    Nothing in it comes from the curated schema: the census input excludes the
    schema's own triples (rule 20), and there are no labels, comments or
    definitions of schema terms, only IRIs, counts and the data's own triples.

Draws run through the Message Batches API (50% price) with a 1-hour prompt-cache
breakpoint at the end of the shared input (``bundle_request``). Before any
submission, ``count_input`` measures the input with the free token-counting
endpoint and refuses a request whose input plus ``max_tokens`` exceeds the
model's context window. Draw 1 goes first; draws 2 and 3 follow in a second
batch so they can read the cache draw 1 wrote. No sampling parameters are sent
(Opus 5.5 rejects them), the effort is explicit, and no server-side fallback is
enabled, because a fallback would answer with a different model inside a
"claude-opus-5-5" draw. A draw whose stop reason is not ``end_turn``, or whose
reported model differs, is recorded as failed. ``ANTHROPIC_API_KEY`` is read
inside the process and never logged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from harness.sources.local import LocalSparqlClient
from harness.sources.model import ClassExpr, ClassInfo, OntologySource, Pattern, PropertyInfo
from harness.sources.owl_reader import read_owl_fragment
from harness.sources.render import PREFIX_TABLE, compact
from harness.sources.vocab import PREFIXES, RDF_TYPE, RDFS_SUBCLASSOF, StoreLayout, non_meta, ref, values
from harness.sparql import SparqlError
from harness.terms import IRI, BNode, Literal

MODEL = "claude-opus-5-5"
N_DRAWS = 3
PROMPT_CELL_DRAW = 1
PARAMETERS = {
    "max_tokens": 100000,  # bounded; the reply is asked to stay under about 50,000 tokens of Turtle, leaving room to think
    "thinking": {"type": "adaptive"},
    "output_config": {"effort": "high"},
}
CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}
SAMPLE_SEED = "www2027-e03-generated-v1"
# The census definition is check contract v2 with change-log entry 14 (science-agent 78fee91). Earlier bundles
# are kept under generated/superseded/ and never drawn. A bundle is drawable only once its exact request has
# been counted and fits rule 33's limit (``fit_bundle``); ``build_bundle`` alone never makes it drawable
# (External review of b0c3920, item 6).
BUNDLE_STATUS = "final: census per check contract v2 with change-log entry 14 (science-agent 78fee91)"
UNCOUNTED_STATUS = "not drawable: the input has not been counted against rule 33's limit (fit_bundle)"
RETRY_RULE = (
    "A failed draw (stop reason other than end_turn, another model, or a batch error) is retried only on an explicit "
    "rerun, with the identical request; the failed attempt is kept as draw-<i>.attempt-<k>.json/.txt. The draw is "
    "its first successful attempt; the prompt-cell draw is draw 1's first successful attempt. A failed draw 1 stops "
    "the run before draws 2 and 3 are submitted."
)
# Prices assumed for the cost record (claude-api skill model table, cached 2026-10-06), USD per million tokens.
PRICES = {"input": 4.00, "output": 20.00, "cache_write_1h": 8.00, "cache_read": 0.20, "batch_discount": 0.5}
SAMPLE_K = 3
SAMPLE_MAX_TRIPLES = 25

SYSTEM = "You are an ontology engineer. You write OWL 2 ontologies in Turtle."

INSTRUCTIONS = """Below are a census of an RDF knowledge graph and a sample of its nodes, both in JSON. Write an OWL ontology for this graph.

The census
- "membership" says how a node belongs to a class in this graph. Read it first: in this graph membership is expressed with {membership}.
- "classes" lists every class the graph uses directly, as [superclasses stated by the census (none: the census states no hierarchy of its own), class_support, type_support]: class_support is the number of nodes that belong to the class under the membership reading (counted up the class hierarchy shared by the graph's vocabularies); type_support is the number of subjects of the literal triple "?x {membership} C", exactly as asserted. A class that is not listed has type_support 0.
- "position_support" gives, for each predicate and position (subject, object), the most specific classes (and, for literal objects, the datatypes) whose support there is at least the census threshold, each with the number of triples whose filler there belongs to that class (counted up the shared class hierarchy). A class there may be a superclass in that hierarchy that the graph does not use directly, given when only its subclasses together reach the threshold. The superclasses of a given class also meet the threshold and are not repeated; classes and datatypes below the threshold are left out. "untyped_fillers" gives, for each predicate and position, the number of triples whose filler there has no class ("{untyped}").
- "patterns" lists [owl:Thing, predicate, class, null, subjects, "literal"]: the asserted triple pattern "?x predicate class" with any subject, and the number of its subjects (the membership predicate that is not this graph's convention).
- "properties" gives, for each predicate, the domain and range derived from the classes whose support at its positions is at least the census threshold, by the rule in "domain_range_rule".
- IRIs are written with the prefixes in "prefixes".

The sample
- "instances" are graph nodes drawn for each class that has at least {min_members} direct members ({k} per class, chosen by a fixed hash of their IRIs), with their outgoing triples. Literals are written as in Turtle.

The ontology you write
- Use the graph's own IRIs, exactly as they appear in the census and the sample. Do not rename them and do not invent IRIs for classes or properties.
- Declare the classes (owl:Class) and their hierarchy (rdfs:subClassOf).
- Declare each property as owl:ObjectProperty or owl:DatatypeProperty, with rdfs:domain and rdfs:range. Write a domain or range that is a union of classes with owl:unionOf.
- State which classes are disjoint (owl:disjointWith or owl:AllDisjointClasses).
- Give every class and property you declare an rdfs:label and an rdfs:comment.
- You do not have to repeat every class of the census: include the classes needed to describe the graph. The whole ontology must fit in one reply of at most about 50,000 tokens.
- Reply with the Turtle document only, in a single ```turtle fenced block, starting with its @prefix declarations."""


# The patterns line as the instructions state it, and its wording when the census also keeps literal patterns on
# the graph's own convention for meta classes (rule 35: ``?x rdf:type owl:Class`` on UniProt). A census without such
# patterns (Rhea's) is described exactly as before.
PATTERNS_NOTE = "(the membership predicate that is not this graph's convention)"
PATTERNS_NOTE_META = (
    "(the membership predicate that is not this graph's convention, and this graph's convention when the class is an "
    "RDF, RDFS or OWL meta class such as owl:Class: a meta class is never a class of the graph and is not in \"classes\")"
)


def _patterns_note(census: OntologySource) -> str:
    convention = RDFS_SUBCLASSOF if census.membership == "rdfs:subClassOf" else RDF_TYPE
    meta = any(p.reading == "literal" and p.predicate == convention for p in census.patterns)
    return PATTERNS_NOTE_META if meta else PATTERNS_NOTE


# --------------------------------------------------------------------------- census block


def _c(iri: str | None) -> str | None:
    return None if iri is None else compact(iri)[0]


def _expand(term: str | None, prefixes: dict[str, str]) -> str | None:
    if term is None:
        return None
    if term.startswith("<") and term.endswith(">"):
        return term[1:-1]
    prefix, local = term.split(":", 1)
    return prefixes[prefix] + local


def census_block(census: OntologySource) -> dict:
    """The census as the generator sees it: all its content, compact IRIs, no build provenance."""
    if census.kind != "census":
        raise ValueError("the generator's input is built from a census source")
    prov = census.provenance
    used = set()

    def c(iri):
        out = _c(iri)
        if out and not out.startswith("<"):
            used.add(out.split(":", 1)[0])
        return out

    ts, cs = census.type_support or {}, census.class_support or {}
    classes = {c(k): [[c(p) for p in v.parents], cs.get(k, 0), ts.get(k, 0)] for k, v in sorted(census.classes.items())}
    properties = {}
    for k, v in sorted(census.properties.items()):
        properties[c(k)] = {
            "kind": v.kind,
            "domain": [[c(m) for m in e.members] for e in v.domain],
            "range": [[c(m) for m in e.members] for e in v.range],
            "datatype_range": [c(d) for d in v.datatype_range],
        }
    patterns = [
        [c(x.subject_class), c(x.predicate), c(x.object_class), c(x.object_datatype), x.support, x.reading]
        for x in census.patterns
    ]
    positions = {
        c(p): {pos: {c(k): n for k, n in sorted(d.items())} for pos, d in sorted(v.items())}
        for p, v in sorted((census.position_support or {}).items())
    }
    untyped_fillers = {c(p): dict(sorted(v.items())) for p, v in sorted((census.untyped_fillers or {}).items())}
    ns = dict(PREFIX_TABLE)
    return {
        "graph": census.graph,
        "membership": prov.get("membership_reading", census.membership),
        "membership_convention": census.membership,
        "support": prov.get("support"),
        "support_threshold": census.support_threshold,
        "untyped": c(prov.get("untyped_sentinel")),
        "domain_range_rule": prov.get("domain_range_rule"),
        "prefixes": {p: ns[p] for p in sorted(used)},
        "classes": classes,
        "position_support": positions,
        "untyped_fillers": untyped_fillers,
        "properties": properties,
        "patterns": patterns,
    }


def census_from_block(block: dict, name: str, graph: str) -> OntologySource:
    """Inverse of ``census_block`` for the census's content (used to prove the block is lossless)."""
    px = block["prefixes"]
    e = lambda t: _expand(t, px)  # noqa: E731
    return OntologySource(
        name=name,
        kind="census",
        graph=graph,
        membership=block["membership_convention"],
        classes={e(k): ClassInfo(e(k), parents=[e(p) for p in v[0]]) for k, v in block["classes"].items()},
        class_support={e(k): v[1] for k, v in block["classes"].items()},
        type_support={e(k): v[2] for k, v in block["classes"].items()},
        position_support={
            e(p): {pos: {e(k): n for k, n in d.items()} for pos, d in v.items()}
            for p, v in block["position_support"].items()
        },
        untyped_fillers={e(p): dict(v) for p, v in block["untyped_fillers"].items()},
        properties={
            e(k): PropertyInfo(
                e(k), kind=v["kind"],
                domain=[ClassExpr(tuple(e(m) for m in x)) for x in v["domain"]],
                range=[ClassExpr(tuple(e(m) for m in x)) for x in v["range"]],
                datatype_range=[e(d) for d in v["datatype_range"]],
            )
            for k, v in block["properties"].items()
        },
        patterns=[Pattern(e(s), e(p), e(o), e(d), n, r) for s, p, o, d, n, r in block["patterns"]],
        support_threshold=block["support_threshold"],
    )


# --------------------------------------------------------------------------- instance sample


@dataclass
class InstanceSample:
    spec: dict
    instances: list[dict]  # {"iri", "sampled_for": [class...], "triples": [[p, o_turtle]...]}


def _turtle_term(cell) -> str:
    if isinstance(cell, IRI):
        return compact(cell.value)[0]
    if isinstance(cell, BNode):
        return "_:b"
    if isinstance(cell, Literal):
        text = json.dumps(cell.lexical, ensure_ascii=False)
        if cell.lang:
            return f"{text}@{cell.lang}"
        if cell.datatype and cell.datatype != "http://www.w3.org/2001/XMLSchema#string":
            return f"{text}^^{compact(cell.datatype)[0]}"
        return text
    return "null"


def _term_key(cell) -> tuple:
    if isinstance(cell, IRI):
        return (0, cell.value, "", "")
    if isinstance(cell, Literal):
        return (1, cell.lexical, cell.datatype or "", cell.lang or "")
    return (2, "", "", "")


def _seeded_hash(seed: str, iri: str) -> str:
    return hashlib.sha256(f"{seed}|{iri}".encode()).hexdigest()


def _complete_rows(client, query: str) -> list:
    """The rows of a sampling query, refusing a truncated result: a capped prefix shorter than the LIMIT would
    pass for exhaustion and drop a stratum, and capped triples would change the sample (External reviews of cf5c000).
    An answer the client marks possibly partial is refused the same way. Both are SparqlErrors (a RuntimeError), so
    a caller that can draw otherwise reads them as a query that did not answer."""
    result = client.query(query)
    if result.truncated:
        raise SparqlError(f"a sampling query returned a truncated result ({result.total_rows} rows); refusing it",
                          "too_large")
    if getattr(result, "possibly_partial", False):
        raise SparqlError("a sampling query's answer may be partial; refusing it", "possibly_partial")
    return result.rows


def _first_data_members(client, data: str, cls: str, seed: str, n: int,
                        is_whole: Callable[[int], bool] | None = None) -> list[str]:
    """The first ``n`` direct data members of ``cls`` (IRIs, the subject of a data-graph triple whose predicate is
    outside RDF, RDFS and OWL) in the seeded order (``_seeded_hash``, ties by IRI). The store orders the class's
    direct members by the same hash and returns a prefix; the prefix's data members are kept in order, and the
    prefix grows tenfold until it holds ``n`` of them or holds every member. Testing data-ness only on the prefix
    keeps the largest UniProt classes (8.7 million members) within the store's time limit.

    ``is_whole``, where given, decides whether an answer shorter than its LIMIT holds every member (from the caller's
    own count of them); one it rejects is refused as cut (SparqlError kind "cut"), since a cut answer can carry no flag
    (External review of c00244b, item 3). Without it a short answer is the whole class, as where no answer is ever cut."""
    limit = max(10 * n, 50)
    while True:
        q = PREFIXES + (
            f"SELECT ?x ?h WHERE {{ {{ SELECT DISTINCT ?x WHERE {{ GRAPH {data} {{ ?x rdf:type|rdfs:subClassOf {ref(cls)} }} "
            f"FILTER isIRI(?x) }} }} BIND(SHA256(CONCAT({json.dumps(seed)}, \"|\", STR(?x))) AS ?h) }} ORDER BY ?h ?x LIMIT {limit}"
        )
        rows = [(r[0].value, r[1].lexical) for r in _complete_rows(client, q)]
        for x, h in rows:
            if h != _seeded_hash(seed, x):
                raise ValueError(f"the store's SHA256 of {x} differs from the sample's own hash")
        if len(rows) < limit and is_whole is not None and not is_whole(len(rows)):
            raise SparqlError(f"a member answer of {len(rows)} rows under its LIMIT {limit} is not the whole class", "cut")
        xs = [x for x, _h in rows]
        is_data: set[str] = set()
        for i in range(0, len(xs), 2000):
            dq = PREFIXES + (f"SELECT DISTINCT ?x WHERE {{ {values('x', xs[i:i + 2000])} GRAPH {data} {{ ?x ?dp ?dv }} "
                             f"FILTER({non_meta('dp')}) }}")
            is_data |= {r[0].value for r in _complete_rows(client, dq)}
        members = [x for x in xs if x in is_data]
        if len(members) >= n or len(rows) < limit:
            return members[:n]
        limit *= 10


def sample_instances(
    client,
    layout: StoreLayout,
    census: OntologySource,
    k: int = SAMPLE_K,
    min_members: int | None = None,
    seed: str = SAMPLE_SEED,
    max_triples: int = SAMPLE_MAX_TRIPLES,
) -> InstanceSample:
    min_members = census.support_threshold if min_members is None else min_members
    data = ref(layout.data_graph)
    # A class's direct data members are at most the subjects of its literal membership triples, which the census
    # counts (type_support for the graph's convention, the literal patterns for the other predicate): a class
    # below min_members there cannot be a stratum, and is not queried.
    bound = dict(census.type_support or {})
    for pat in census.patterns:
        if pat.reading == "literal" and pat.object_class:
            bound[pat.object_class] = bound.get(pat.object_class, 0) + pat.support
    candidates = sorted(c for c in census.classes if bound.get(c, 0) >= min_members)
    # Per class, its first max(k, min_members) direct data members in the seeded order: the class is a stratum
    # when that many exist, and its first k are drawn. One bounded query per class scales to UniProt, where one
    # query over every class's members times out (``_first_data_members``).
    first = max(k, min_members)
    strata: list[str] = []
    picked: dict[str, list[str]] = {}
    for cls in candidates:
        rows = _first_data_members(client, data, cls, seed, first)
        if len(rows) >= min_members:
            strata.append(cls)
            for x in rows[:k]:
                picked.setdefault(x, []).append(cls)

    triples: dict[str, list[tuple]] = {x: [] for x in picked}
    if picked:
        q = PREFIXES + f"SELECT ?x ?p ?o WHERE {{ {values('x', picked)} GRAPH {data} {{ ?x ?p ?o }} }}"
        for x, p, o in _complete_rows(client, q):
            triples[x.value].append((p.value, o))
    instances = []
    for x in sorted(picked):
        rows = sorted(triples[x], key=lambda po: (po[0], _term_key(po[1])))[:max_triples]
        instances.append({
            "iri": compact(x)[0],
            "sampled_for": [compact(c)[0] for c in sorted(picked[x])],
            "triples": [[compact(p)[0], _turtle_term(o)] for p, o in rows],
        })
    spec = {
        "strata": "census classes with at least min_members direct data members (rdf:type or rdfs:subClassOf object)",
        "min_members": min_members,
        "k_per_class": k,
        "selection": "smallest sha256(seed + '|' + IRI) (hex, as SPARQL SHA256), ties by IRI",
        "seed": seed,
        "max_triples_per_instance": max_triples,
        "triples": "all outgoing triples of the instance in the data graph, ordered by (predicate, object)",
        "data_graph": layout.data_graph,
        "n_strata": len(strata),
        "n_instances": len(instances),
    }
    return InstanceSample(spec, instances)


# --------------------------------------------------------------------------- bundle


def _request(system: str, user: str) -> dict:
    """The exact request body of every draw: the whole input is the shared prefix, cached for an hour."""
    return {
        "model": MODEL, **PARAMETERS, "system": system,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user, "cache_control": CACHE_CONTROL}]}],
    }


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_bundle(census: OntologySource, sample: InstanceSample) -> dict:
    block = census_block(census)
    payload = {"census": block, "sample": {"spec": sample.spec, "instances": sample.instances}}
    data_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    instructions = INSTRUCTIONS.replace("{membership}", census.membership).replace("{untyped}", str(block["untyped"])).replace(
        "{min_members}", str(sample.spec["min_members"])).replace("{k}", str(sample.spec["k_per_class"])).replace(
        PATTERNS_NOTE, _patterns_note(census))
    user = instructions + "\n\n<census_and_sample>\n" + data_text + "\n</census_and_sample>\n"
    return {
        "status": UNCOUNTED_STATUS,
        "model": MODEL,
        "parameters": PARAMETERS,
        "cache_control": CACHE_CONTROL,
        "api": "Message Batches (draw 1, then draws 2 and 3), 1-hour cache breakpoint at the end of the input",
        "fallbacks": "none (a fallback would answer with another model)",
        "retry_rule": RETRY_RULE,
        "n_draws": N_DRAWS,
        "prompt_cell_draw": PROMPT_CELL_DRAW,
        "census": {"name": census.name, "digest": census.digest(), "support_threshold": census.support_threshold},
        "sample_spec": sample.spec,
        "system": SYSTEM,
        "user": user,
        "request_sha256": _sha(_request(SYSTEM, user)),
        "user_sha256": hashlib.sha256(user.encode()).hexdigest(),
    }


INPUT_LIMIT = 800_000  # rule 33: above this many input tokens the instance sample is reduced
DRAW_THRESHOLD = 5  # rule 33: the census a generator input is drawn from (census.DEFAULT_THRESHOLD)


def reduce_sample(sample: InstanceSample, k: int) -> InstanceSample:
    """The same sample with at most ``k`` instances per stratum: per stratum, the first ``k`` of its
    instances in the sample's own order (smallest sha256(seed|IRI)), so a reduced sample is a subset."""
    if k >= sample.spec["k_per_class"]:
        return sample
    seed = sample.spec["seed"]
    from harness.sources.decontaminate import _expand

    keep: dict[str, list[str]] = {}
    by_stratum: dict[str, list[dict]] = {}
    for inst in sample.instances:
        for c in inst["sampled_for"]:
            by_stratum.setdefault(c, []).append(inst)
    for c, insts in by_stratum.items():
        ranked = sorted(insts, key=lambda i: (_seeded_hash(seed, _expand(i["iri"])), _expand(i["iri"])))
        for i in ranked[:k]:
            keep.setdefault(i["iri"], []).append(c)
    instances = [{**i, "sampled_for": sorted(keep[i["iri"]])} for i in sample.instances if i["iri"] in keep]
    spec = {**sample.spec, "k_per_class": k, "n_instances": len(instances),
            "reduced_from": sample.spec["k_per_class"], "reduction": "rule 33 fallback: uniform per class"}
    return InstanceSample(spec, instances)


def fit_bundle(census: OntologySource, sample: InstanceSample, count, limit: int = INPUT_LIMIT) -> tuple[dict, dict]:
    """Rule 33's fallback, stated before measuring: build the bundle, count its exact request with
    ``count(bundle) -> input tokens`` (the free counting endpoint); while it exceeds ``limit``, reduce the
    per-class sample uniformly, down to one instance. The record says what was tried; ``fits`` is False if
    even one instance per class does not fit, and then nothing is drawn."""
    tried = []
    for k in range(sample.spec["k_per_class"], 0, -1):
        bundle = build_bundle(census, reduce_sample(sample, k))
        n = count(bundle)
        tried.append({"k_per_class": k, "input_tokens": n})
        if n <= limit:
            bundle["input_fit"] = {"limit": limit, "tried": tried, "fits": True, "fallback_used": k < sample.spec["k_per_class"]}
            bundle["status"] = BUNDLE_STATUS
            return bundle, bundle["input_fit"]
    bundle["input_fit"] = {"limit": limit, "tried": tried, "fits": False, "fallback_used": True}
    bundle["status"] = "too large: rule 33 fallback exhausted; nothing is drawn"
    return bundle, bundle["input_fit"]


def bundle_request(bundle: dict) -> dict:
    return _request(bundle["system"], bundle["user"])


def supersede_input(out_dir: Path, superseded_by: str | None, reason: str) -> Path | None:
    """Move the bundle under out_dir/input/ to out_dir/superseded/<its request sha256 prefix>/ with a note, so it is
    never drawn; None when there is none."""
    inp = out_dir / "input"
    old = inp / "bundle.json"
    if not old.exists():
        return None
    prev = json.loads(old.read_text())
    dest = out_dir / "superseded" / prev.get("request_sha256", "unknown")[:16]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    inp.rename(dest)
    (dest / "SUPERSEDED.json").write_text(json.dumps({
        "superseded_by": superseded_by,
        "reason": reason,
        "previous_status": prev.get("status"),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


IDENTITIES_FILE = "census-identities.json"
IDENTITY_RULE = ("design ruling in the design log: the store's identity record is provenance, never content; a census rebuilt "
                 "with byte-identical content keeps every bundle built from it drawable, and the identities it was read "
                 "under are recorded here, beside the bundle; only a content difference retires a bundle")


def census_content_digest(census: OntologySource) -> str:
    """sha256 of the census's canonical JSON form without each vocabulary's store record
    (``provenance["vocabularies"][i]["store"]``: the serving index's name, stats URL and counts as the server reports
    them), which is provenance (design ruling in the design log). Everything else, other provenance included, is content."""
    obj = census.to_json()
    prov = dict(obj.get("provenance") or {})
    if isinstance(prov.get("vocabularies"), list):
        prov["vocabularies"] = [{k: x for k, x in v.items() if k != "store"} if isinstance(v, dict) else v
                                for v in prov["vocabularies"]]
    obj["provenance"] = prov
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _store_records(census: OntologySource) -> list[dict]:
    return [{"graph": v.get("graph"), "store": v.get("store")} for v in census.provenance.get("vocabularies") or []
            if isinstance(v, dict)]


def _identities(bundle_dir: Path) -> dict | None:
    path = bundle_dir / "input" / IDENTITIES_FILE
    return json.loads(path.read_text()) if path.exists() else None


def _same_census(bundle_digest: str | None, census: OntologySource, bundle_dir: Path) -> bool:
    """Whether ``census`` is the census a bundle recording ``bundle_digest`` was built from, or that census re-read
    under another store identity: the same digest, or both digests recorded beside the bundle with one content
    digest, which ``census`` has."""
    digest = census.digest()
    if digest == bundle_digest:
        return True
    rec = _identities(bundle_dir)
    if not rec or rec.get("census") != census.name:
        return False
    seen = {i.get("digest") for i in rec.get("identities", [])}
    return bundle_digest in seen and digest in seen and census_content_digest(census) == rec.get("content_digest")


def _record_identity(bundle_dir: Path, built_from: OntologySource, census: OntologySource) -> None:
    """Add the store records of ``built_from`` (the census the bundle was built from) and of ``census`` (its content
    re-read under another identity) to input/census-identities.json, in the order seen."""
    content = census_content_digest(census)
    rec = _identities(bundle_dir)
    if not rec or rec.get("census") != census.name or rec.get("content_digest") != content:
        rec = {"rule": IDENTITY_RULE, "census": census.name, "content_digest": content, "identities": []}
    for src in (built_from, census):
        if all(i.get("digest") != src.digest() for i in rec["identities"]):
            rec["identities"].append({"digest": src.digest(), "stores": _store_records(src)})
    (bundle_dir / "input" / IDENTITIES_FILE).write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n",
                                                        encoding="utf-8")


def retire_stale_bundle(out_dir: Path, sources_dir: Path, census: OntologySource, new_request: str | None = None,
                        name: str | None = None, reason: str | None = None) -> Path | None:
    """Before a new census is written (External review of the sources snapshot, item 2): the bundle under out_dir/input/
    is retired, with any draws made from it, when it was built from a census of other content than ``census`` or
    when ``new_request`` replaces it, so no failure later in the build leaves it drawable against the new census. A
    bundle of the same census and request stays. A census of the same content read under another store identity (the
    design ruling in the design log) is the same census: the bundle stays, and both identities are recorded beside it
    (input/census-identities.json), the old one read from the census file it was built from, still on disk here.
    Returns where a retired bundle went, or None."""
    old = out_dir / "input" / "bundle.json"
    if not old.exists():
        return None
    prev = json.loads(old.read_text())
    recorded = prev.get("census") or {}
    same_request = new_request in (None, prev.get("request_sha256"))
    if same_request and recorded.get("digest") == census.digest():
        return None
    if same_request and recorded.get("name") == census.name:
        path = sources_dir / f"{census.name}.json"
        on_disk = OntologySource.from_json(json.loads(path.read_text())) if path.exists() else None
        if (on_disk is not None and _same_census(recorded.get("digest"), on_disk, out_dir)
                and census_content_digest(on_disk) == census_content_digest(census)):
            _record_identity(out_dir, on_disk, census)
            return None
    sha = prev.get("request_sha256", "unknown")
    why = reason or (f"the census changed (bundle built from census {recorded.get('digest')}, the "
                     f"new census is {census.digest()}); this bundle and its draws are never drawn or current")
    if sha in draws_request_sha(out_dir):
        return retire_draws(out_dir, sources_dir, name or f"superseded-{sha[:16]}", why)
    return supersede_input(out_dir, new_request, why)


def write_bundle(bundle: dict, out_dir: Path, input_tokens: int | None = None) -> Path:
    """Write the bundle under out_dir/input/. A different earlier bundle there is moved to
    out_dir/superseded/<its request sha256>/ with a note; it is never drawn."""
    inp = out_dir / "input"
    old = inp / "bundle.json"
    if old.exists() and json.loads(old.read_text()).get("request_sha256") != bundle["request_sha256"]:
        supersede_input(out_dir, bundle["request_sha256"], "the generator's input changed (its census, sample or "
                                                           "instructions); this bundle is never drawn")
    inp.mkdir(parents=True, exist_ok=True)
    record = {k: v for k, v in bundle.items() if k not in ("system", "user")}
    if input_tokens is not None:
        record["input_tokens_o200k"] = input_tokens
    (inp / "prompt-system.txt").write_text(bundle["system"], encoding="utf-8")
    (inp / "prompt-user.txt").write_text(bundle["user"], encoding="utf-8")
    (inp / "bundle.json").write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (inp / "SHA256").write_text(f"{bundle['request_sha256']}  request (model, parameters, system, messages)\n"
                                f"{bundle['user_sha256']}  prompt-user.txt\n", encoding="utf-8")
    return inp


def load_bundle(out_dir: Path) -> dict:
    """The bundle written under out_dir/input/, checked against its recorded hash."""
    inp = out_dir / "input"
    bundle = json.loads((inp / "bundle.json").read_text())
    bundle["system"] = (inp / "prompt-system.txt").read_text(encoding="utf-8")
    bundle["user"] = (inp / "prompt-user.txt").read_text(encoding="utf-8")
    if _sha(_request(bundle["system"], bundle["user"])) != bundle["request_sha256"]:
        raise ValueError("the bundle's prompt files do not match its request_sha256")
    return bundle


# --------------------------------------------------------------------------- draws and parsing

_FENCE = re.compile(r"```(?:turtle|ttl)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_turtle(text: str) -> tuple[str, str]:
    """The Turtle document in a reply: the first fenced block, else the whole text."""
    m = _FENCE.search(text)
    if m:
        return m.group(1), "fenced block"
    return text, "whole reply"


def parse_generated(turtle: str, name: str, graph: str, provenance: dict | None = None,
                    membership: str = "rdfs:subClassOf") -> OntologySource:
    """Read a generated Turtle ontology with the same OWL reader as the curated source."""
    gname = "urn:www2027:generated"
    client = LocalSparqlClient.from_graphs({gname: turtle}, fmt="turtle")
    frag = read_owl_fragment(client, [gname], None)
    return OntologySource(
        name=name,
        kind="generated",
        graph=graph,
        membership=membership,
        classes=frag.classes,
        properties=frag.properties,
        disjoint=frag.disjoint,
        existentials=frag.existentials,
        unsupported=frag.unsupported,
        # rule 32 (a): the typings, left out (None: every listed class is declared) when they cover every class, as
        # for the curated source, so a draft that types all its classes keeps its JSON
        declared_classes=None if set(frag.declared) == set(frag.classes) else frag.declared,
        described=frag.described or None,  # read as the curated source's are (design ruling, 19 Oct 23:02)
        provenance={
            "builder": "harness.sources.generated.parse_generated (owl_reader.read_owl_fragment on rdflib)",
            "turtle_sha256": hashlib.sha256(turtle.encode()).hexdigest(),
            "excluded": frag.excluded,
            **(provenance or {}),
        },
    )


def draw_cost(usage: dict) -> float:
    """USD for one batch draw from its usage, at ``PRICES`` (batch discount applied)."""
    cc = usage.get("cache_creation") or {}
    write_1h = cc.get("ephemeral_1h_input_tokens")
    write_5m = cc.get("ephemeral_5m_input_tokens", 0) or 0
    if write_1h is None:
        write_1h, write_5m = usage.get("cache_creation_input_tokens", 0) or 0, 0
    total = (
        (usage.get("input_tokens", 0) or 0) * PRICES["input"]
        + write_1h * PRICES["cache_write_1h"]
        + write_5m * PRICES["input"] * 1.25
        + (usage.get("cache_read_input_tokens", 0) or 0) * PRICES["cache_read"]
        + (usage.get("output_tokens", 0) or 0) * PRICES["output"]
    )
    return round(total * PRICES["batch_discount"] / 1e6, 4)


def anthropic_client(key_file: Path | None = None):
    """A client whose key is read inside this process (from ANTHROPIC_API_KEY, else from key_file); the
    key is never printed or written anywhere."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        if key_file is None or not key_file.exists():
            raise RuntimeError("ANTHROPIC_API_KEY is not set and no key file was found")
        os.environ["ANTHROPIC_API_KEY"] = key_file.read_text().strip()
    import anthropic

    return anthropic.Anthropic()


def count_input(bundle: dict, client, limit: int | None = None) -> dict:
    """Count the input with the free token-counting endpoint and check it fits: at most ``limit`` input tokens
    (rule 33: 800,000) and, with ``max_tokens``, within the model's context window."""
    limit = INPUT_LIMIT if limit is None else limit
    req = bundle_request(bundle)
    counted = client.messages.count_tokens(model=req["model"], system=req["system"], messages=req["messages"],
                                           thinking=req["thinking"], output_config=req["output_config"])
    model = client.models.retrieve(req["model"])
    window = getattr(model, "max_input_tokens", None)
    n = counted.input_tokens
    fits = window is not None and n <= limit and n + req["max_tokens"] <= window
    return {"input_tokens": n, "max_tokens": req["max_tokens"], "context_window": window, "input_limit": limit,
            "model_max_output": getattr(model, "max_tokens", None), "fits": fits}


def _dump(obj) -> dict:
    return obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)


def submit_batch(bundle: dict, indices: list[int], client) -> str:
    req = bundle_request(bundle)
    batch = client.messages.batches.create(requests=[{"custom_id": f"draw-{i}", "params": req} for i in indices])
    return batch.id


def wait_batch(client, batch_id: str, poll_s: float = 60.0, log=print):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        log(f"batch {batch_id}: {batch.processing_status} {_dump(batch.request_counts)}")
        time.sleep(poll_s)


def collect_batch(bundle: dict, batch_id: str, out_dir: Path, client) -> list[dict]:
    """Write draw-<i>.txt (the raw reply) and draw-<i>.json (the record) for every result."""
    records = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in client.messages.batches.results(batch_id):
        index = int(result.custom_id.split("-")[1])
        current = out_dir / f"draw-{index}.json"
        prev = json.loads(current.read_text()) if current.exists() else None
        if prev is not None and prev.get("batch_id") == batch_id:
            attempt = int(prev.get("attempt", 1))  # collected before (a resumed run): rewrite it, same attempt
        else:
            attempt = 1
            while (out_dir / f"draw-{index}.attempt-{attempt}.json").exists():
                attempt += 1
            if prev is not None:  # an earlier attempt from another batch (failed, or a retry): keep it
                for ext in ("json", "txt"):
                    old = out_dir / f"draw-{index}.{ext}"
                    if old.exists():
                        old.rename(out_dir / f"draw-{index}.attempt-{attempt}.{ext}")
                attempt += 1
        rec = {
            "draw": index,
            "attempt": attempt,
            "prompt_cell_draw": index == bundle["prompt_cell_draw"],
            "batch_id": batch_id,
            "custom_id": result.custom_id,
            "model_requested": bundle["model"],
            "parameters": bundle["parameters"],
            "cache_control": bundle.get("cache_control"),
            "request_sha256": bundle["request_sha256"],
            "input": "input/prompt-system.txt and input/prompt-user.txt (the exact request: input/bundle.json)",
            "result_type": result.result.type,
        }
        if "decontamination" in bundle:  # the map the input was renamed with: a draw is current only with it
            rec["map_sha256"] = bundle["decontamination"]["map_sha256"]
        text = ""
        if result.result.type == "succeeded":
            message = result.result.message
            text = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
            usage = _dump(message.usage)
            failed = []
            if message.stop_reason != "end_turn":
                failed.append(f"stop_reason {message.stop_reason}")
            if message.model != bundle["model"]:
                failed.append(f"model {message.model} differs from {bundle['model']}")
            if not failed:  # a draw is ok only if its reply is a non-empty ontology
                try:
                    turtle, _ = extract_turtle(text)
                    parsed = parse_generated(turtle, f"generated-{index}", "check")
                    if not parsed.classes and not parsed.properties:
                        failed.append("the reply declares no class or property")
                except Exception as exc:  # rdflib raises parser errors of many types
                    failed.append(f"the reply is not a readable Turtle ontology: {str(exc)[:200]}")
            rec.update({
                "cache_hit": (usage.get("cache_read_input_tokens") or 0) > 0,
                "model_returned": message.model, "response_id": message.id, "stop_reason": message.stop_reason,
                "usage": usage, "cost_usd": draw_cost(usage), "prices_usd_per_mtok": PRICES,
                "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "status": "failed: " + "; ".join(failed) if failed else "ok",
            })
        else:
            rec["status"] = f"failed: batch result {result.result.type}"
            rec["error"] = _dump(result.result).get("error") if hasattr(result.result, "error") else None
        (out_dir / f"draw-{index}.txt").write_text(text, encoding="utf-8")
        (out_dir / f"draw-{index}.json").write_text(json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n",
                                                    encoding="utf-8")
        records.append(rec)
    return records


def run_batch_draws(bundle: dict, out_dir: Path, client, poll_s: float = 60.0, log=print, retry: bool = False,
                    sources_dir: Path | None = None) -> list[dict]:
    """Count, check the fit, then draw 1 in one batch and draws 2 and 3 in a second batch, so they can read
    the cache draw 1 wrote. A submitted batch (batches.json) is resumed, never resubmitted. A failed draw is
    redrawn only with ``retry`` (``RETRY_RULE``). Nothing is submitted unless the bundle is drawable and built from
    the graph's current census (``census_current``; ``sources_dir`` defaults to the parent of ``out_dir``)."""
    drawable, why = is_drawable(bundle)
    if not drawable:
        raise RuntimeError(why)
    current, why = census_current(bundle, out_dir.parent if sources_dir is None else sources_dir, out_dir)
    if not current:
        raise RuntimeError(f"not drawn: {why}")
    state_path = out_dir / "batches.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"request_sha256": bundle["request_sha256"]}
    if state["request_sha256"] != bundle["request_sha256"]:
        raise RuntimeError("batches.json belongs to another bundle")
    state.setdefault("batches", [])

    def save():
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    if "count" not in state:
        state["count"] = count_input(bundle, client)
        save()
    log(f"input {state['count']}")
    if not state["count"]["fits"]:
        raise RuntimeError(f"the input does not fit the context window: {state['count']}")
    records = []
    for group, indices in (("first", [1]), ("rest", [2, 3])):
        if group == "rest" and not _draw_ok(out_dir, 1):
            log("draw 1 has no successful attempt; draws 2 and 3 are not submitted")
            break
        pending = [b for b in state["batches"] if b["group"] == group and not b.get("collected")]
        if pending:
            entry = pending[-1]
        else:
            todo = [i for i in indices if not _draw_ok(out_dir, i)]
            if not todo:
                continue
            if any(b["group"] == group for b in state["batches"]) and not retry:
                log(f"draws {todo} failed before; rerun with --retry to redraw them with the identical request")
                break
            entry = {"group": group, "draws": todo, "batch_id": submit_batch(bundle, todo, client),
                     "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            state["batches"].append(entry)
            save()
        batch = wait_batch(client, entry["batch_id"], poll_s, log)
        entry["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry["request_counts"] = _dump(batch.request_counts)
        got = collect_batch(bundle, entry["batch_id"], out_dir, client)
        entry["collected"] = True
        save()
        records += got
        for r in got:
            if r["draw"] > 1 and r.get("status") == "ok" and not r.get("cache_hit"):
                log(f"warning: draw {r['draw']} did not read the cache draw 1 wrote (paid the full input)")
        if group == "first" and any(r["status"] != "ok" for r in got):
            log("draw 1 failed; draws 2 and 3 are not submitted")
            break
    return records


def _payload(user: str) -> dict:
    start = user.index("<census_and_sample>\n") + len("<census_and_sample>\n")
    return json.loads(user[start:user.index("\n</census_and_sample>")])


def census_current(bundle: dict, sources_dir: Path, bundle_dir: Path | None = None) -> tuple[bool, str]:
    """Whether a bundle was built from the graph's current census file (External review of the sources snapshot, item
    2): the census it records (``<sources_dir>/<census name>.json``) has the recorded digest, or is that census re-read
    under another store identity as recorded beside the bundle (``census-identities.json``; design ruling on issue
    15), and the census in its prompt is exactly that file's ``census_block``. A decontaminated bundle is current when
    the natural bundle it was renamed from (``<sources_dir>/generated``, checked against its own request hash) is the
    one it records, and that bundle is current. ``bundle_dir``: a natural bundle's directory (default
    ``<sources_dir>/generated``)."""
    natural = bundle
    nat_dir = sources_dir / "generated" if bundle_dir is None else bundle_dir
    dc = bundle.get("decontamination")
    if dc is not None:
        nat_dir = sources_dir / "generated"
        if not (nat_dir / "input" / "bundle.json").exists():
            return False, f"the decontaminated bundle's natural bundle is missing ({nat_dir}/input)"
        try:
            natural = load_bundle(nat_dir)
        except ValueError as e:
            return False, f"the natural bundle at {nat_dir}: {e}"
        if natural["request_sha256"] != dc.get("natural_request_sha256"):
            return False, (f"the decontaminated bundle was renamed from natural request {dc.get('natural_request_sha256')}, "
                           f"not the current {natural['request_sha256']}")
    recorded = natural.get("census") or {}
    path = sources_dir / f"{recorded.get('name')}.json"
    if not path.exists():
        return False, f"no census file {path} to check the bundle against"
    census = OntologySource.from_json(json.loads(path.read_text()))
    if not _same_census(recorded.get("digest"), census, nat_dir):
        return False, (f"the bundle was built from census {recorded.get('digest')}, but {path} is census "
                       f"{census.digest()} (content {census_content_digest(census)})")
    if _payload(natural["user"]).get("census") != census_block(census):
        return False, f"the census in the bundle's prompt is not {path}'s"
    return True, "current"


def is_drawable(bundle: dict) -> tuple[bool, str]:
    """A bundle is drawable only with a final status and a passing rule-33 count of its exact request."""
    fit = bundle.get("input_fit") or {}
    if not str(bundle.get("status", "")).startswith("final"):
        return False, f"bundle is {bundle.get('status')}"
    threshold = (bundle.get("census") or {}).get("support_threshold")
    if threshold != DRAW_THRESHOLD:  # rule 33: never another threshold's census (design log's t2 and t20 are offline only)
        return False, f"the bundle's census has threshold {threshold}, not {DRAW_THRESHOLD}"
    if not fit.get("fits") or not fit.get("tried") or fit["tried"][-1]["input_tokens"] > fit.get("limit", INPUT_LIMIT):
        return False, "the bundle has no passing rule-33 count of its exact request"
    return True, "drawable"


def draw_files(out_dir: Path) -> list[Path]:
    """Every file of the draws made in ``out_dir`` (records, texts, failed attempts, the batch state)."""
    files = [p for p in out_dir.glob("draw-*") if p.is_file()]
    if (out_dir / "batches.json").exists():
        files.append(out_dir / "batches.json")
    return sorted(files)


def draws_request_sha(out_dir: Path) -> set[str]:
    """The request hashes the draws in ``out_dir`` were made from."""
    out = set()
    for p in out_dir.glob("draw-*.json"):
        sha = json.loads(p.read_text()).get("request_sha256")
        if sha:
            out.add(sha)
    if (out_dir / "batches.json").exists():
        out.add(json.loads((out_dir / "batches.json").read_text()).get("request_sha256"))
    return out - {None}


def retire_draws(out_dir: Path, sources_dir: Path, name: str, reason: str, source_prefix: str = "generated") -> Path:
    """Move the draws in ``out_dir`` (and, when they were made from it, the input bundle), with the generated
    sources, renders and profiles built from them (``<source_prefix>-<i>``: generated-dc for decontaminated draws),
    to ``out_dir/name/``, so none of them stays a current source (External review of b0c3920, item 7). Profiles are
    copied, not moved: the next build rewrites them."""
    import shutil

    dest = out_dir / name
    if dest.exists():
        raise FileExistsError(f"{dest} exists; retire under another name")
    shas = draws_request_sha(out_dir)
    files = draw_files(out_dir)
    if not files:
        raise FileNotFoundError(f"no draws in {out_dir}")
    dest.mkdir(parents=True)
    moved = []
    for f in files:
        shutil.move(str(f), dest / f.name)
        moved.append(f.name)
    inp = out_dir / "input" / "bundle.json"
    if inp.exists() and json.loads(inp.read_text()).get("request_sha256") in shas:
        shutil.move(str(out_dir / "input"), dest / "input")
        moved.append("input/")
    draws = sorted(int(m.group(1)) for f in moved for m in [re.match(r"draw-(\d+)\.json\Z", f)] if m)
    for i in draws:
        g = f"{source_prefix}-{i}"
        for src, rel in ((sources_dir / f"{g}.json", f"{g}.json"),
                         (sources_dir / "renders" / f"{g}.txt", f"renders/{g}.txt"),
                         (sources_dir / "renders" / f"{g}.manifest.json", f"renders/{g}.manifest.json")):
            if src.exists():
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), dest / rel)
                moved.append(rel)
    for prof in ("profiles.json", "PROFILES.md"):
        if (sources_dir / prof).exists():
            shutil.copy2(sources_dir / prof, dest / prof)
    (dest / "SUPERSEDED.json").write_text(json.dumps({
        "draws": draws, "request_sha256": sorted(shas)[0] if len(shas) == 1 else sorted(shas), "reason": reason,
        "moved": sorted(moved),
        "profiles": "profiles.json and PROFILES.md as they stood with these draws",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def _draw_ok(out_dir: Path, index: int) -> bool:
    p = out_dir / f"draw-{index}.json"
    return p.exists() and json.loads(p.read_text()).get("status") == "ok"


def load_draw(out_dir: Path, index: int, graph: str, membership: str = "rdfs:subClassOf") -> OntologySource | None:
    """The generated source of a successful draw, or None. The reply is draw-<i>.txt, or the file a draw record names
    as the reply it derived from that raw text (``reply.file_suffix``: a Gateway draw whose think block came inline)."""
    rec_path = out_dir / f"draw-{index}.json"
    if not rec_path.exists():
        return None
    record = json.loads(rec_path.read_text())
    txt_path = out_dir / f"draw-{index}{(record.get('reply') or {}).get('file_suffix', '.txt')}"
    if not txt_path.exists():
        return None
    text = txt_path.read_bytes().decode("utf-8")
    if record.get("status") != "ok" or not text.strip():
        return None
    turtle, where = extract_turtle(text)
    return parse_generated(turtle, f"generated-{index}", graph, {"draw": record, "turtle_from": where}, membership)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generated-source draws from an input bundle written by build_sources.")
    ap.add_argument("--out", type=Path, required=True, help="e.g. data/sources/rhea-142/generated/")
    ap.add_argument("--key-file", type=Path, default=Path.home() / ".config" / "www-2027" / "anthropic_api_key")
    ap.add_argument("--count-only", action="store_true", help="count the input tokens and check the fit; no draw")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--retry", action="store_true", help="redraw failed draws with the identical request")
    args = ap.parse_args(argv)
    bundle = load_bundle(args.out)
    client = anthropic_client(args.key_file)
    if args.count_only:
        print(json.dumps(count_input(bundle, client), sort_keys=True))
        return 0
    records = run_batch_draws(bundle, args.out, client, args.poll, log=lambda m: print(m, flush=True), retry=args.retry)
    for r in records:
        print(f"draw {r['draw']}: {r['status']} usage={r.get('usage')} cost=${r.get('cost_usd')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
