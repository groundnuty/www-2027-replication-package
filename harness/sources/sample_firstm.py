"""The generator's instance sample where the recipe's ordered member query does not answer (census rulings of 20 Oct
00:17).

The recipe (``generated.sample_instances``) takes from each stratum the first ``max(k, min_members)`` direct data members
in the order of sha256(seed|IRI), ties by IRI, the store sorting the class's members by that hash. Where that query does
not answer within the store's limits, the ruling: the class's k members are drawn uniformly by the seed from its FIRST M
members in the engine's natural order, M the largest that answers within the limits, recorded as a difference from the
other graphs' recipe.

So here:

* every class is drawn by the recipe's own query where that answers. An answer shorter than its LIMIT is the whole class
  only when it has as many rows as the class has IRI members by a separate count of the same population; any other
  short answer is cut (External review of c00244b, item 3) and the class goes on as one the recipe could not order;
* at the first class whose recipe query does not answer, M is found once, on the largest class: the largest LIMIT
  whose plain member query (the recipe's member pattern, no ORDER BY) answers completely. The rungs of ``ladder`` below
  the class's size are tried upwards, then the size itself (the ceiling is the class, not the ladder); between the
  last that answers and the first that does not, M is bisected to within ``RESOLUTION`` members (item 4). A class on
  which M does not answer steps down the ladder. An answer is complete only with ``m`` rows or with the class's every
  IRI member, counted by the same pattern and filter as the member query (item 5: a census count may include
  blank-node members, the member query reads IRIs only): a cut answer may carry no flag, so any other row count
  counts as not answered, and so does an answer the client marks possibly partial;
* a class the recipe cannot order has its first M members ranked by the same hash client-side and tested for data-ness
  in that order with the recipe's own test, until ``max(k, min_members)`` data members are found; a class with no more
  members than M is thereby read whole, and its draw is exactly the recipe's;
* the member query is sent a second time, outside any cache, and the two answers compared (the natural order is the
  engine's, so its stability is checked, not assumed). If they differ, the same members are read with the rdf:type
  pattern alone (the same member set when the class has no literal rdfs:subClassOf members in the census) and checked
  the same way; a class whose pool is unstable in both forms is not drawn.

The sample keeps the recipe's form; its spec says which rule drew how many strata, and ``record`` (private) holds, per
class, the rule, the bound, M, the pool's form, size and hash, the stability check and the data-ness tests.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

from harness.sources import generated as gen
from harness.sources.model import OntologySource
from harness.sources.render import compact
from harness.sources.vocab import PREFIXES, RDFS_SUBCLASSOF, StoreLayout, non_meta, ref, values
from harness.sparql import SparqlError
from harness.terms import IRI

LADDER = (10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000)
RESOLUTION = 1_000  # members: M is bisected to within this
NOT_ANSWERED = frozenset({"timeout", "too_large", "transport", "http", "cut", "possibly_partial"})
LARGE_RULE = (
    "a class whose members the store could not order within its limits (the recipe's member query did not answer, or "
    "answered cut) is drawn from its first M members in the store's natural order (the recipe's member pattern with "
    "LIMIT M and no ORDER BY; the rdf:type pattern alone where that order was not stable), ranked by the same seeded "
    "hash, data-ness tested in that order; M is the largest LIMIT whose member query answered completely on the largest "
    "class, found by the ladder with the class's size as its ceiling and bisected to within pool_resolution members "
    "(census rulings of 20 Oct 00:17). This differs from the other graphs' recipe for these classes only."
)


def _members_query(data: str, cls: str, limit: int, form: str = "recipe") -> str:
    pattern = "rdf:type|rdfs:subClassOf" if form == "recipe" else "rdf:type"
    distinct = "DISTINCT " if form == "recipe" else ""
    return PREFIXES + (f"SELECT {distinct}?x WHERE {{ GRAPH {data} {{ ?x {pattern} {ref(cls)} }} "
                       f"FILTER isIRI(?x) }} LIMIT {limit}")


def _count_query(data: str, cls: str, form: str = "recipe") -> str:
    """The IRI members the member query of ``form`` reads, counted: the same pattern, filter and distinctness."""
    pattern = "rdf:type|rdfs:subClassOf" if form == "recipe" else "rdf:type"
    count = "COUNT(DISTINCT ?x)" if form == "recipe" else "COUNT(*)"
    return PREFIXES + f"SELECT ({count} AS ?n) WHERE {{ GRAPH {data} {{ ?x {pattern} {ref(cls)} }} FILTER isIRI(?x) }}"


def _not_answered(exc: SparqlError) -> bool:
    return exc.kind in NOT_ANSWERED


def _pool(client, data: str, cls: str, limit: int, form: str = "recipe") -> list[str]:
    result = client.query(_members_query(data, cls, limit, form))
    if result.truncated:
        raise SparqlError("a member query returned a truncated result", "too_large")
    if getattr(result, "possibly_partial", False):
        raise SparqlError("a member query's answer may be partial", "possibly_partial")
    return [r[0].value for r in result.rows if isinstance(r[0], IRI)]


def member_count(client, data: str, cls: str, form: str = "recipe") -> int | None:
    """The class's IRI members as the member query of ``form`` reads them; None when the count does not answer."""
    try:
        result = client.query(_count_query(data, cls, form))
    except SparqlError as exc:
        if not _not_answered(exc):
            raise
        return None
    rows = result.rows
    if result.truncated or getattr(result, "possibly_partial", False) or len(rows) != 1 or rows[0][0] is None:
        return None
    return int(rows[0][0].lexical)


def _pool_sha(pool: list[str]) -> str:
    return hashlib.sha256("\n".join(pool).encode()).hexdigest()


def _first_data_members_of(client, data: str, pool: list[str], seed: str, n: int) -> tuple[list[str], int]:
    """The first ``n`` data members of ``pool`` in the seeded order, with the recipe's data-ness test, and how many
    members were tested."""
    ranked = sorted(pool, key=lambda x: (gen._seeded_hash(seed, x), x))
    found: list[str] = []
    tested = 0
    for i in range(0, len(ranked), 2000):
        chunk = ranked[i:i + 2000]
        q = PREFIXES + (f"SELECT DISTINCT ?x WHERE {{ {values('x', chunk)} GRAPH {data} {{ ?x ?dp ?dv }} "
                        f"FILTER({non_meta('dp')}) }}")
        is_data = {r[0].value for r in gen._complete_rows(client, q)}
        tested += len(chunk)
        found += [x for x in chunk if x in is_data]
        if len(found) >= n:
            break
    return found[:n], tested


def _complete(rows: int, m: int, whole: int | None) -> bool:
    """A member answer is used only when it is the full LIMIT (``m`` rows) or the whole class (as many rows as the
    class has IRI members by the same pattern, ``whole``; None when that count did not answer). Any other count may be
    a cut answer, which may carry no flag, so it is not trusted."""
    return rows == m or (whole is not None and rows == whole)


def find_pool_limit(client, data: str, cls: str, bound: int, ladder=LADDER, whole: int | None = None,
                    resolution: int = RESOLUTION) -> tuple[int | None, list[dict]]:
    """M on ``cls``: the largest LIMIT whose member query answers completely. The rungs of ``ladder`` below the
    class's size (``whole``, its IRI members, else the census count ``bound``) are tried upwards, then the size; between
    the last that answers (or 0) and the first that does not, M is bisected until the two are ``resolution`` or fewer
    apart. None when nothing answers."""
    size = whole if whole is not None else bound
    tried: list[dict] = []

    def answers(m: int) -> bool:
        try:
            got = len(_pool(client, data, cls, m))
        except SparqlError as exc:
            if not _not_answered(exc):
                raise
            tried.append({"m": m, "answered": False, "kind": exc.kind})
            return False
        if not _complete(got, m, whole):
            tried.append({"m": m, "answered": True, "rows": got, "cut": True})
            return False
        tried.append({"m": m, "answered": True, "rows": got})
        return True

    low, high = 0, None
    for m in [r for r in ladder if r < size] + [size]:
        if not answers(m):
            high = m
            break
        low = m
    while high is not None and high - low > resolution:
        mid = (low + high) // 2
        if answers(mid):
            low = mid
        else:
            high = mid
    return (low or None), tried


def _stable_pool(client, recheck, data: str, cls: str, limit: int, ladder, forms,
                 whole: dict[str, int | None]) -> tuple[list[str] | None, dict]:
    """The class's first members at the largest rung up to ``limit`` that answers completely, in the first form whose
    answer is the same when sent again; ``whole`` gives each form's count of the class's IRI members."""
    info: dict = {"forms_tried": [], "cut_answers": 0}
    for form in forms:
        m, pool = limit, None
        while pool is None:
            try:
                pool = _pool(client, data, cls, m, form)
                if not _complete(len(pool), m, whole[form]):
                    info["cut_answers"] += 1
                    pool = None
                    raise SparqlError("a cut member answer", "too_large")
            except SparqlError as exc:
                if not _not_answered(exc):
                    raise
                lower = [r for r in ladder if r < m]
                if not lower:
                    break
                m = lower[-1]
        if pool is None:
            info["forms_tried"].append({"form": form, "answered": False})
            continue
        try:
            stable = _pool(recheck, data, cls, m, form) == pool
        except SparqlError as exc:
            if not _not_answered(exc):
                raise
            stable = None  # the re-check did not answer: stability unknown, the pool is not used
        info["forms_tried"].append({"form": form, "answered": True, "m": m, "rows": len(pool), "stable": stable,
                                    "pool_sha256": _pool_sha(pool)})
        if stable:
            info.update(form=form, m=m, pool_rows=len(pool), whole_class=len(pool) == whole[form],
                        pool_sha256=_pool_sha(pool), stable=True)
            return pool, info
    info["stable"] = False
    return None, info


def sample_instances_firstm(
    client,
    layout: StoreLayout,
    census: OntologySource,
    k: int = gen.SAMPLE_K,
    min_members: int | None = None,
    seed: str = gen.SAMPLE_SEED,
    max_triples: int = gen.SAMPLE_MAX_TRIPLES,
    ladder=LADDER,
    recheck_client=None,
    log: Callable[[str], None] | None = None,
    resolution: int = RESOLUTION,
) -> tuple[gen.InstanceSample, dict]:
    """The sample and its private record. ``recheck_client`` sends the stability re-check (by default ``client``;
    give the uncached client when ``client`` caches answers)."""
    say = log or (lambda _m: None)
    recheck = recheck_client or client
    min_members = census.support_threshold if min_members is None else min_members
    data = ref(layout.data_graph)
    bound = dict(census.type_support or {})
    via_subclass = set()
    for pat in census.patterns:
        if pat.reading == "literal" and pat.object_class:
            bound[pat.object_class] = bound.get(pat.object_class, 0) + pat.support
            if pat.predicate == RDFS_SUBCLASSOF:
                via_subclass.add(pat.object_class)
    candidates = sorted((c for c in census.classes if bound.get(c, 0) >= min_members), key=lambda c: (bound[c], c))
    first = max(k, min_members)
    strata: list[str] = []
    picked: dict[str, list[str]] = {}
    per_class: list[dict] = []
    limit: int | None = None
    ladder_tried: list[dict] = []
    limit_found = False
    counts: dict[tuple[str, str], int | None] = {}

    def whole(cls: str, form: str = "recipe") -> int | None:
        if (cls, form) not in counts:
            counts[(cls, form)] = member_count(client, data, cls, form)
        return counts[(cls, form)]

    for i, cls in enumerate(candidates):
        entry = {"class_sha256": hashlib.sha256(cls.encode()).hexdigest()[:16], "bound": bound[cls]}
        try:
            members = gen._first_data_members(client, data, cls, seed, first,
                                              is_whole=lambda rows, c=cls: rows == whole(c))
            entry["rule"] = "recipe"
        except SparqlError as exc:
            if not _not_answered(exc):
                raise
            members = None
            entry["recipe_not_answered"] = exc.kind
            say(f"  sample: the recipe did not answer on class {i + 1}/{len(candidates)}")
        if members is None:
            if not limit_found:
                largest = candidates[-1]
                limit, ladder_tried = find_pool_limit(client, data, largest, bound[largest], ladder, whole(largest),
                                                      resolution)
                limit_found = True
            if limit is None:
                entry.update(rule="not_drawn", why="no LIMIT of the member query answered")
                per_class.append(entry)
                continue
            forms = ("recipe",) if cls in via_subclass else ("recipe", "type_only")
            entry["iri_members"] = {f: whole(cls, f) for f in forms}
            pool, info = _stable_pool(client, recheck, data, cls, limit, ladder, forms, entry["iri_members"])
            entry.update(info)
            if pool is None:
                entry.update(rule="not_drawn", why="no stable member list within the ladder")
                per_class.append(entry)
                continue
            entry["rule"] = "first_m"
            members, entry["data_tested"] = _first_data_members_of(client, data, pool, seed, first)
        entry["members_found"] = len(members)
        entry["stratum"] = len(members) >= min_members
        per_class.append(entry)
        if len(members) >= min_members:
            strata.append(cls)
            for x in members[:k]:
                picked.setdefault(x, []).append(cls)

    triples: dict[str, list[tuple]] = {x: [] for x in picked}
    todo = sorted(picked)
    for i in range(0, len(todo), 50):  # the recipe's one query, sent in chunks: the same rows
        q = PREFIXES + f"SELECT ?x ?p ?o WHERE {{ {values('x', todo[i:i + 50])} GRAPH {data} {{ ?x ?p ?o }} }}"
        for x, p, o in gen._complete_rows(client, q):
            triples[x.value].append((p.value, o))
    instances = []
    for x in sorted(picked):
        rows = sorted(triples[x], key=lambda po: (po[0], gen._term_key(po[1])))[:max_triples]
        instances.append({
            "iri": compact(x)[0],
            "sampled_for": [compact(c)[0] for c in sorted(picked[x])],
            "triples": [[compact(p)[0], gen._turtle_term(o)] for p, o in rows],
        })
    n_large = sum(1 for e in per_class if e["rule"] == "first_m" and e["stratum"])
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
    if limit_found:
        spec.update(large_classes=LARGE_RULE, pool_limit=limit, pool_ladder=list(ladder), pool_resolution=resolution,
                    n_large_strata=n_large)
    record = {"candidates": len(candidates), "first": first, "pool_limit": limit, "pool_resolution": resolution,
              "ladder": ladder_tried,
              "per_class": per_class,
              "rules": {r: sum(1 for e in per_class if e["rule"] == r) for r in ("recipe", "first_m", "not_drawn")},
              "unstable_pools": sum(1 for e in per_class for f in e.get("forms_tried", []) if f.get("stable") is False),
              "type_only_pools": sum(1 for e in per_class if e.get("form") == "type_only"),
              "cut_answers": sum(e.get("cut_answers", 0) for e in per_class)
              + sum(1 for t in ladder_tried if t.get("cut")),
              "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()}
    return gen.InstanceSample(spec, instances), record
