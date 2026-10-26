"""Question set for the private graph a15: generic machinery only.

Nothing about the graph lives here. Two private files on the run host hold it
(mode 600 under ~/project/private/a15/, never in git):

- the vocabulary (``A15_VOCAB``, default a15-vocab.json): placeholder -> IRI
  ("iri"), placeholder -> value ("value"), the names question texts must not
  use ("names"), the predicates whose triples are counted ("predicates"), the
  keys of the engine-suspect classes ("suspect_class_keys",
  harness/engine_suspect.py), the endpoint, and the path of the template file
  ("templates_module");
- the template file (``A15_TEMPLATES``, or the vocabulary's
  "templates_module"): a Python module that defines ``TEMPLATES`` (the
  question templates, built with the helpers below), ``KNOBS`` (knobs added to
  the generic ones), ``NOTES`` (manifest notes), ``STORE_IDENTITY`` (the
  store's identity in the manifest), ``PREPARE`` (what ``prepare`` draws),
  ``PREPARE_CHECKS`` (what ``prepare`` checks) and ``REQUIRED_KEYS``. It is
  refused unless it is mode 600.

Every gold is confirmed by a differently written query (design ruling (a),
design log): its variant's ``confirm_pattern``, written by the template file.
The generator refuses a variant without one, or whose confirmation is the
gold's own pattern counted or wrapped, and requires exact agreement
(harness.qgen.generate); the profile also refuses golds of the engine-suspect
shape (harness/engine_suspect.py, suspect classes from the vocabulary).

Patterns use ``string.Template`` placeholders; the vocabulary renders them
(``VOCAB``), with ``$FRAME`` and ``$FRAME_SMALL`` (and any value set named in
``PREPARE``) filled by ``prepare`` before sampling.

``prepare`` runs the template file's checks of what its templates assume about
the graph on every run, then draws a seeded frame of entities of one class from
the complete, ordered list (checked against a COUNT) and fills the value sets.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from harness.engine_suspect import VOCAB_KEYS, suspect, suspect_classes_from_vocab
from harness.qgen.base import (
    Anchor,
    PoolRow,
    Profile,
    TemplateError,
    Variant,
    check_iri,
    sparql_literal,
    usable_label,
)
from harness.qgen.readings import Reading
from harness.qgen.sampling import rng_for
from harness.qgen.text_guard import register_private_names

GRAPH_ID = "a15"
ID_PREFIX = "a15"
PRIVATE = Path("~/project/private/a15").expanduser()
DEFAULT_VOCAB = PRIVATE / "a15-vocab.json"
DEFAULT_TEMPLATES = PRIVATE / "a15_templates.py"
# The pinned triple count of the deployed store (public: design log, the proposal).
PINNED_TRIPLES = 945_694_878

PREFIXES = (
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
)
FRAME_KEYS = ("FRAME", "FRAME_SMALL")

# Generic knobs; the template file adds its own.
GENERIC_KNOBS: dict[str, Any] = {
    # Sampling frame: entities drawn with the seed; small pools use the first frame_small of them.
    "frame_size": 1000,
    "frame_small": 200,
    # Answer sizes for list questions, and the most threshold options per entity.
    "max_rows": 30,
    "max_threshold_options": 3,
    # A pool query slower than this fails the run: a slow answer from this store can be incomplete without a
    # time-out flag (design log), so pools are kept fast and checked.
    "pool_max_seconds": 60,
    # A gold query slower than this fails the run.
    "gold_max_seconds": 10,
    "variant_weights": {},
}


class Vocab(Mapping[str, str]):
    """Placeholder -> rendered term: IRIs as ``<...>``, values as SPARQL literals, runtime values as given.

    Read from the private file on first use; tests call ``use`` with an
    invented vocabulary.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] | None = None
        self._runtime: dict[str, str] = {}

    def use(self, data: Mapping[str, Any]) -> None:
        iris = dict(data.get("iri", {}))
        for key, iri in iris.items():
            try:
                check_iri(iri)
            except TemplateError as exc:
                raise TemplateError(f"vocabulary entry {key} is not a safe IRI") from exc
        self._data = {"iri": iris, "value": dict(data.get("value", {})), "names": list(data.get("names", ())),
                      "endpoint": data.get("endpoint"), "predicates": list(data.get("predicates", ())),
                      "templates_module": data.get("templates_module")}
        if VOCAB_KEYS in data:  # passed on only when present, so a vocabulary without it is refused, not empty
            self._data[VOCAB_KEYS] = list(data[VOCAB_KEYS])
        self._runtime = {}

    def _loaded(self) -> dict[str, Any]:
        if self._data is None:
            path = Path(os.environ.get("A15_VOCAB", DEFAULT_VOCAB))
            self.use(json.loads(path.read_text()))
        return self._data  # type: ignore[return-value]

    def require(self, keys: Iterable[str]) -> None:
        data = self._loaded()
        missing = [k for k in keys if k not in data["iri"] and k not in data["value"]]
        if missing:
            raise TemplateError(f"the a15 vocabulary lacks {len(missing)} required entries")

    def set_runtime(self, **values: str) -> None:
        self._runtime.update(values)

    def iri(self, key: str) -> str:
        return self._loaded()["iri"][key]

    @property
    def names(self) -> list[str]:
        return self._loaded()["names"]

    @property
    def predicates(self) -> list[str]:
        return self._loaded()["predicates"]

    @property
    def endpoint(self) -> str | None:
        return self._loaded()["endpoint"]

    @property
    def templates_module(self) -> str | None:
        return self._loaded()["templates_module"]

    @property
    def suspect_classes(self) -> list[str]:
        """The engine-suspect classes: the IRIs of the "suspect_class_keys" (harness.engine_suspect)."""
        data = self._loaded()
        return suspect_classes_from_vocab({k: data[k] for k in ("iri", VOCAB_KEYS) if k in data})

    def __getitem__(self, key: str) -> str:
        data = self._loaded()
        if key in data["iri"]:
            return f"<{data['iri'][key]}>"
        if key in data["value"]:
            return sparql_literal(data["value"][key])
        if key in self._runtime:
            return self._runtime[key]
        if key in FRAME_KEYS or key in _runtime_names():
            raise TemplateError(f"{key} is not prepared yet (a15.prepare runs before sampling)")
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        data = self._loaded()
        return iter(dict.fromkeys([*data["iri"], *data["value"], *FRAME_KEYS, *_runtime_names()]))

    def __len__(self) -> int:
        return len(list(iter(self)))


VOCAB = Vocab()

# --------------------------------------------------------------------------
# The private template file

_SPEC: Any = None


def use_spec(module: Any) -> None:
    """Use a template module object (tests); None reloads the private file on next use."""
    global _SPEC
    _SPEC = module


def spec() -> Any:
    """The template module: given by use_spec, or loaded from the private file (mode 600 required)."""
    global _SPEC
    if _SPEC is None:
        path = Path(os.environ.get("A15_TEMPLATES") or VOCAB.templates_module or DEFAULT_TEMPLATES).expanduser()
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise TemplateError("the a15 template file must be readable by its owner only (mode 600)")
        loader_spec = importlib.util.spec_from_file_location("a15_private_templates", path)
        module = importlib.util.module_from_spec(loader_spec)
        sys.modules["a15_private_templates"] = module
        loader_spec.loader.exec_module(module)
        _SPEC = module
    return _SPEC


def _runtime_names() -> list[str]:
    if _SPEC is None:
        return []
    return [vs["name"] for vs in getattr(_SPEC, "PREPARE", {}).get("value_sets", ())]


class _LazyTemplates(Sequence):
    """The template file's TEMPLATES, loaded on first use (importing the profiles needs no private file)."""

    def _items(self) -> tuple:
        return tuple(spec().TEMPLATES)

    def __getitem__(self, i):  # type: ignore[override]
        return self._items()[i]

    def __len__(self) -> int:
        return len(self._items())

    def __iter__(self):
        return iter(self._items())


class _LazyMapping(Mapping[str, Any]):
    """Generic entries plus the template file's, loaded on first use."""

    def __init__(self, generic: Mapping[str, Any], attribute: str):
        self._generic, self._attribute = generic, attribute

    def _items(self) -> dict[str, Any]:
        return {**self._generic, **getattr(spec(), self._attribute, {})}

    def __getitem__(self, key: str) -> Any:
        return self._items()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items())

    def __len__(self) -> int:
        return len(self._items())


TEMPLATES = _LazyTemplates()
KNOBS = _LazyMapping(GENERIC_KNOBS, "KNOBS")
NOTES = _LazyMapping({
    "private_files": "templates, wording, notes and vocabulary are read at run time from mode-600 files on the run host",
    "store_check": "triples counted per predicate of the private vocabulary and summed, against the pinned count; no GRAPH ?g scan",
}, "NOTES")

# --------------------------------------------------------------------------
# Helpers for the template file


def phrase(text: str) -> str:
    """A clause regex for an exact phrase."""
    return r"(?<![\w-])" + re.escape(text) + r"(?![\w-])"


def pattern_reading(name: str, description: str, clause: str, pattern: str) -> Reading:
    return Reading(name, description, clause, pattern=pattern)


def other_kind_anchor(entity_param: str, label_param: str, kind_phrase: str) -> Reading:
    """Reading: the quoted label names any entity carrying it, of whatever kind.

    The alternative gold resolves the entity by its label alone; the clause is
    the kind stated next to the label (``kind_phrase`` «label»).
    """

    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        iri, label = params.get(entity_param), params.get(label_param)
        if not iri or not label or f"<{iri}>" not in sparql:
            return None
        var = f"?homonym_{entity_param}"
        out = sparql.replace(f"<{iri}>", var)
        m = re.search(r"WHERE\s*\{", out)
        if m is None:
            return None
        return out[: m.end()] + f"\n  {var} rdfs:label {sparql_literal(str(label))} ." + out[m.end():]

    return Reading(
        "other_kind_anchor",
        "the quoted label names any entity carrying it, of whatever kind",
        phrase(kind_phrase) + rf" «{label_param}»",
        transform=transform,
    )


def label_anchor(kind: str, class_key: str, label_param: str) -> Anchor:
    """An entity named by a label unique among the entities of one class (a vocabulary key)."""
    return Anchor(kind, (label_param,), f"?e a ${class_key} ; rdfs:label ${label_param}_lit .")


def avariant(**kwargs: Any) -> Variant:
    """A Variant with the a15 prefixes and the private vocabulary as its placeholders.

    Its confirmation query is the ``confirm_pattern`` the template file gives; a variant without one is refused
    when the set is generated (generate.check_confirm_forms).
    """
    kwargs.setdefault("membership_mode", "n.a.")
    return Variant(prefixes=PREFIXES, graphs=VOCAB, **kwargs)


def engine_suspect(query: str) -> bool:
    """Whether a query has the shape the store answers wrongly (suspect classes from the private vocabulary)."""
    return suspect(query, VOCAB.suspect_classes)


def stable(items: Iterable[Any], salt: str) -> list[Any]:
    """A seed-independent scrambled order, so a cap does not favour low identifiers."""
    return sorted(items, key=lambda x: hashlib.sha256(f"{salt}|{x}".encode()).hexdigest())


def size_stratum(n: int) -> str:
    return "1" if n == 1 else "2" if n == 2 else "3-5" if n <= 5 else "6+"


def _int(row: PoolRow, key: str) -> int:
    return int(float(row[key]))


def group_rows(rows: list[PoolRow], key: str) -> OrderedDict[str, list[PoolRow]]:
    out: OrderedDict[str, list[PoolRow]] = OrderedDict()
    for row in rows:
        out.setdefault(row[key], []).append(row)
    return out


# Expand functions: candidate-query rows -> question parameters. Column and
# parameter names are given by the template file.


def entities(iri_col: str, label_col: str, param: str, label_param: str, stratum: bool = True):
    """One option per labelled entity; a count column "n" sets the answer-size stratum."""

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        for row in rows:
            label = row.get(label_col)
            if label and usable_label(label):
                out = {param: check_iri(row[iri_col]), label_param: label}
                if stratum and row.get("n") not in (None, ""):
                    out["_stratum"] = size_stratum(_int(row, "n"))
                yield out

    return expand


def entities_below_total(iri_col: str, label_col: str, param: str, label_param: str, total_col: str):
    """Labelled entities whose count "n" is below their total (the filter leaves something out)."""

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        for row in rows:
            if usable_label(row[label_col]) and _int(row, total_col) > _int(row, "n"):
                yield {param: check_iri(row[iri_col]), label_param: row[label_col],
                       "_stratum": size_stratum(_int(row, "n"))}

    return expand


def stratified_entities(iri_col: str, label_col: str, param: str, label_param: str, stratum_col: str,
                        stratum_key: str):
    """Labelled entities in two strata: the column equals the vocabulary IRI ``stratum_key`` ("a") or not ("b")."""

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        marker = VOCAB.iri(stratum_key)
        for row in rows:
            if usable_label(row[label_col]):
                yield {param: check_iri(row[iri_col]), label_param: row[label_col],
                       "_stratum": "a" if row.get(stratum_col) == marker else "b"}

    return expand


def thresholds_between(values: list[float], max_rows: int, limit: int, salt: str) -> list[int]:
    """Integer thresholds strictly between two values present, keeping 1..max_rows above and dropping at least one.

    No threshold equals a value in the data, so "above" and "at least" read the same.
    """
    distinct = sorted(set(values))
    options = []
    for lo, hi in pairwise(distinct):
        t = int(lo) + 1
        if not lo < t < hi:
            continue
        above = sum(1 for v in values if v > t)
        if 1 <= above <= max_rows and above < len(values):
            options.append(t)
    return stable(options, salt)[:limit]


def threshold_options(entity_col: str, label_col: str, value_col: str, param: str, label_param: str,
                      below: bool = False):
    """Per entity, up to max_threshold_options thresholds between its values (rows: entity, label, value)."""

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        k = knobs or KNOBS
        for entity, group in group_rows(rows, entity_col).items():
            label = group[0][label_col]
            if not usable_label(label):
                continue
            values = [float(r[value_col]) for r in group]
            flipped = [-v for v in values] if below else values
            for t in thresholds_between(flipped, int(k["max_rows"]), int(k["max_threshold_options"]), entity):
                yield {param: check_iri(entity), label_param: label, "threshold": -t if below else t}

    return expand


def knob_thresholds(iri_col: str, label_col: str, param: str, label_param: str, knob: str):
    """Per entity, every threshold listed in a knob, in a stable scrambled order."""

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        k = knobs or KNOBS
        for row in rows:
            if usable_label(row[label_col]):
                for t in stable(k[knob], row[iri_col]):
                    yield {param: check_iri(row[iri_col]), label_param: row[label_col], "threshold": int(t)}

    return expand


def attribute_options(iri_col: str, label_col: str, param: str, label_param: str, attributes: tuple):
    """Per entity, one option per attribute: (vocabulary key, phrase, unit clause) triples."""

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        for row in rows:
            if not usable_label(row[label_col]):
                continue
            for key, attribute_phrase, unit in stable(attributes, row[iri_col]):
                yield {param: check_iri(row[iri_col]), label_param: row[label_col],
                       "attribute": VOCAB[key], "attribute_phrase": attribute_phrase, "unit": unit, "_stratum": key}

    return expand


def pairs(iri_col: str, label_col: str, compared: tuple, columns: Mapping[str, str]):
    """Each entity with the next in a stable scrambled order, on the first attribute (key, phrase) where they differ.

    ``columns`` maps each attribute key to its column in the rows. Both orders
    (higher, lower) become options; the parameters are a, b and their labels.
    """

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        usable = [r for r in rows if usable_label(r[label_col])]
        order = stable(usable, "pairs")
        for a, b in pairwise(order):
            for key, attribute_phrase in stable(compared, a[iri_col]):
                col = columns[key]
                if float(a[col]) == float(b[col]):
                    continue
                for direction, sort in (("higher", "DESC(?value)"), ("lower", "ASC(?value)")):
                    yield {"a": check_iri(a[iri_col]), "a_label": a[label_col], "b": check_iri(b[iri_col]),
                           "b_label": b[label_col], "attribute": VOCAB[key], "attribute_phrase": attribute_phrase,
                           "direction": direction, "order": sort}
                break

    return expand


def _with_value(params: dict[str, Any], name: str, value: str) -> dict[str, Any]:
    lit = sparql_literal(value)
    return {**params, name: value, f"{name}_lit": lit, f"{name}_typed": lit + "^^xsd:string"}


def entity_value(iri_col: str, label_col: str, param: str, label_param: str, value_col: str, value_param: str):
    """A labelled entity with a literal value from the same row (plain and typed literal forms included)."""

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        for row in rows:
            if usable_label(row[label_col]) and usable_label(row[value_col]):
                yield _with_value({param: check_iri(row[iri_col]), label_param: row[label_col],
                                   "_stratum": size_stratum(_int(row, "n"))}, value_param, row[value_col])

    return expand


# --------------------------------------------------------------------------
# Run-time preparation and the store check


def prepare(client: Any, seed: int, knobs: Mapping[str, Any] = KNOBS) -> dict[str, Any]:
    """Run the template file's checks, draw the sampling frame, fill the value sets and load the private names.

    ``PREPARE_CHECKS(client)`` in the template file checks, on every run, what its templates assume about the
    graph (raising TemplateError when an assumption fails) and returns counts for the manifest; a template file
    without it is refused. ``PREPARE`` names the frame class ("frame_class") and the value sets ("value_sets":
    runtime name, property key, knob of the largest entity count, and optionally "observe": a callable given the
    set's complete value list that returns counts for the manifest).
    """
    module = spec()
    plan = getattr(module, "PREPARE", {})
    VOCAB.require(getattr(module, "REQUIRED_KEYS", ()))
    checks = getattr(module, "PREPARE_CHECKS", None)
    if checks is None:
        raise TemplateError("the template file defines no PREPARE_CHECKS")
    checked = dict(checks(client))
    frame_class = VOCAB[plan["frame_class"]]
    total = int(client.query(f"SELECT (COUNT(?a) AS ?n) WHERE {{ ?a a {frame_class} . }}").rows[0][0].lexical)
    members = [r[0].value for r in client.query(f"SELECT ?a WHERE {{ ?a a {frame_class} . }} ORDER BY ?a").rows]
    if len(members) != total or len(set(members)) != total:
        raise TemplateError(f"the frame list has {len(members)} entities, the count {total}")
    frame = rng_for(seed, "a15-frame").sample(members, min(int(knobs["frame_size"]), total))
    small = frame[: int(knobs["frame_small"])]
    VOCAB.set_runtime(
        FRAME=" ".join(f"<{check_iri(g)}>" for g in sorted(frame)),
        FRAME_SMALL=" ".join(f"<{check_iri(g)}>" for g in sorted(small)),
    )
    register_private_names(VOCAB.names)
    frame_values = " ".join(f"<{g}>" for g in sorted(frame))
    value_sets = {}
    for vs in plan.get("value_sets", ()):
        prop = VOCAB[vs["property"]]
        values = {r[0].lexical for r in client.query(f"SELECT DISTINCT ?v WHERE {{ ?x {prop} ?v . }}").rows}
        n_values = int(client.query(f"SELECT (COUNT(DISTINCT ?v) AS ?n) WHERE {{ ?x {prop} ?v . }}")
                       .rows[0][0].lexical)
        if len(values) != n_values:
            raise TemplateError(f"the value list of {vs['name']} has {len(values)} values, the count {n_values}")
        observed = dict(vs["observe"](values)) if vs.get("observe") else {}
        # Values of the frame held by at most a knob's count of entities: two queries, intersected here, so each
        # stays fast.
        in_frame = {r[0].lexical for r in client.query(
            f"SELECT DISTINCT ?v WHERE {{ VALUES ?f {{ {frame_values} }} ?f {prop} ?v . }}").rows}
        sized = {r[0].lexical for r in client.query(
            f"SELECT ?v WHERE {{ ?x {prop} ?v . }} GROUP BY ?v HAVING (COUNT(?x) <= {int(knobs[vs['max_count_knob']])})"
        ).rows}
        chosen = sorted(in_frame & sized)
        VOCAB.set_runtime(**{vs["name"]: " ".join(sparql_literal(v) for v in chosen)})
        value_sets[vs["name"]] = {"values": len(values), **observed, "in_frame_and_sized": len(chosen)}
    return {
        **checked,
        "frame_total": total,
        "frame_size": len(frame),
        "frame_small_size": len(small),
        "frame_sha256": hashlib.sha256("\n".join(sorted(frame)).encode()).hexdigest(),
        "value_sets": value_sets,
        "private_names": len(VOCAB.names),
    }


def store_count(client: Any) -> dict[str, int]:
    """Triples of the data graph, as the sum of per-predicate counts (each bounded by the predicate's index)."""
    if not VOCAB.predicates:
        raise TemplateError("the a15 vocabulary lists no predicates to count")
    total = 0
    for pred in VOCAB.predicates:
        rows = client.query(f"SELECT (COUNT(*) AS ?n) WHERE {{ ?s <{check_iri(pred)}> ?o . }}").rows
        total += int(rows[0][0].lexical)
    return {"data graph": total}


PROFILE = Profile(
    graph_id=GRAPH_ID,
    id_prefix=ID_PREFIX,
    templates=TEMPLATES,
    store_identity=_LazyMapping({}, "STORE_IDENTITY"),
    two_tier=True,
    lookup_share=1 / 3,
    knobs=KNOBS,
    notes=NOTES,
    forbid_database_names=True,
    global_answer_uniqueness=True,
    audit_rules=True,
    store_count=store_count,
    pinned_triples={"data graph": PINNED_TRIPLES},
    prepare=prepare,
    confirm_golds=True,
    engine_suspect=engine_suspect,
)
