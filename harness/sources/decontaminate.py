"""Decontaminated generations (design log, criterion 4): the generator's input with its names replaced.

One map, from one source term to one opaque term, is built from the census and the instance sample
and applied to both, so the generator works over opaque names and cannot recall a public ontology
from them. Its output is mapped back through the inverse map. A term in the output that the map does
not know was invented by the model: it stays as it is and is reported, and an output IRI in an
original namespace (one the renamed input never showed) is reported apart, as a recall signal.

Policy, as for opaque Rhea (#7, ``stores/opaque/rename.py``):

* IRIs: every IRI is renamed wherever it occurs, except the terms actually defined in the published RDF, RDFS,
  OWL 2 and XSD vocabularies (rule 36 as narrowed: the shared list data/vocab/w3c-terms.json, the compiler's own);
  an undefined IRI in those namespaces (an invented ``rdfs:labelOf``) is renamed like any other:
  IRIs used as predicates to ``<base>pNNNNNNN``, other IRIs to ``<base>eNNNNNNN``, with the number a
  seeded random permutation within each kind (no order of the originals survives). The seed is
  required (scripts/build_decontaminated.py --seed) and recorded in the bundle's provenance and the held-out
  map's header.
* Literals (design log ruling, proposal §7.3: ``literals="strings"``): the lexical form of every literal whose
  datatype is not a W3C numeric, date-time or boolean type is masked (plain and language-tagged strings,
  labels and comments, and typed identifiers such as ``"P04637"^^<...#Accession>``), with one placeholder per
  distinct lexical form (same string, same token, whatever its language tag), so equalities and co-occurrence
  survive; a masked literal becomes ``sNNNNNNN`` and keeps its language tag and its datatype, renamed like any
  other IRI when it is not a W3C one (External review of b0c3920, items 4 and 8). Numbers, booleans and dates stay.
  (``"names"`` masks only label and comment values.)
* The census block's graph name becomes ``g0000001``; its explanatory texts (membership reading,
  support, derivation rule) are kept and must name no graph (``check_leaks`` looks).

The map is a function on terms, so co-reference survives. ``verify`` checks it independently of how
it was built: it is one-to-one, no opaque term equals an original term, it covers every name of the current
input (``input_originals``), and the renamed input holds no original IRI, namespace or renamed string of the map
or of the current input. A map is a pure function of the input, the seed and the policy, so a map written
earlier is reused only when it equals the map rebuilt from the current input (``same_map``;
scripts/build_decontaminated.py compares the file itself, recorded policy included). Mapping a decontaminated
ontology back and comparing it with the census proves nothing (it agrees by construction); the diagnostic is
overlap with the curated ontology (design log, criterion 5).
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from harness.checks.compiler import W3C_TERMS_PATH, w3c_terms
from harness.sources.generated import InstanceSample
from harness.sources.model import ClassExpr, ClassInfo, OntologySource, Pattern, PropertyInfo
from harness.sources.render import PREFIX_TABLE, compact
from harness.sources.vocab import (
    COMMENT_PROPERTIES,
    LABEL_PROPERTIES,
    MEMBERSHIP_READING_BASE,
    NAMED_RESTRICTIONS_READING,
    OWL,
    RDF,
    RDFS,
    SKOS,
    XSD,
)

W3C_NAMESPACES = (RDF, RDFS, OWL, XSD)  # the namespaces of the kept terms: never an original namespace
OPAQUE_BASE = "http://example.org/opaque/"
NAME_PROPERTIES = frozenset(LABEL_PROPERTIES) | frozenset(COMMENT_PROPERTIES) | {SKOS + "altLabel"}
WIDTH = 7
_TOKEN = re.compile(r"\A(?:[pegs])[0-9]{%d}\Z" % WIDTH)


def kept(iri: str) -> bool:
    """Rule 36 (narrowed): only a term defined in the published RDF, RDFS, OWL 2 or XSD vocabulary is kept."""
    return iri in w3c_terms()


def kept_terms_sha256() -> str:
    return hashlib.sha256(W3C_TERMS_PATH.read_bytes()).hexdigest()


def _expand(term: str) -> str:
    if term.startswith("<") and term.endswith(">"):
        return term[1:-1]
    prefix, local = term.split(":", 1)
    return dict(PREFIX_TABLE)[prefix] + local


def _parse_literal(term: str) -> tuple[str, str | None, str | None]:
    """A Turtle literal as the sample writes it (double-quoted), or single-quoted: lexical, datatype IRI, language
    tag."""
    quote = term[0]
    end = term.rfind(quote)
    if quote == '"':
        lexical = json.loads(term[: end + 1])
    else:  # 'x': the same escapes, with \' for the quote
        lexical = json.loads('"' + term[1:end].replace("\\'", "'").replace('"', '\\"') + '"')
    rest = term[end + 1:]
    if rest.startswith("^^"):
        return lexical, _expand(rest[2:]), None
    if rest.startswith("@"):
        return lexical, None, rest[1:]
    return lexical, None, None


def _format_literal(lexical: str, datatype: str | None, lang: str | None) -> str:
    text = json.dumps(lexical, ensure_ascii=False)
    if lang:
        return f"{text}@{lang}"
    if datatype:
        return f"{text}^^{compact(datatype)[0]}"
    return text


# W3C datatypes whose literals are values, not names: kept as they are (every other literal's lexical form is masked
# under the "strings" policy).
VALUE_TYPES = frozenset(XSD + t for t in (
    "decimal", "integer", "nonPositiveInteger", "negativeInteger", "long", "int", "short", "byte",
    "nonNegativeInteger", "unsignedLong", "unsignedInt", "unsignedShort", "unsignedByte", "positiveInteger",
    "float", "double", "boolean", "date", "dateTime", "dateTimeStamp", "time", "gYear", "gYearMonth", "gMonth",
    "gMonthDay", "gDay", "duration", "dayTimeDuration", "yearMonthDuration"))


def is_value(datatype: str | None) -> bool:
    """A literal kept as it is: a W3C numeric, date-time or boolean type."""
    return datatype in VALUE_TYPES


@dataclass
class OpaqueMap:
    iris: dict[str, str] = field(default_factory=dict)  # original IRI -> opaque IRI
    strings: dict[str, str] = field(default_factory=dict)  # lexical form -> opaque lexical (tags kept per literal)
    seed: int = 0
    base: str = OPAQUE_BASE
    literals: str = "names"
    graph_name: tuple[str, str] = ("", "g0000001")

    def iri(self, iri: str | None) -> str | None:
        if iri is None or kept(iri):
            return iri
        return self.iris[iri]

    def inverse_iris(self) -> dict[str, str]:
        return {v: k for k, v in self.iris.items()}

    def inverse_strings(self) -> dict[str, str]:
        return {v: k for k, v in self.strings.items()}

    def to_json(self, header: dict | None = None) -> dict:
        return {
            "header": {"seed": self.seed, "literals": self.literals, **(header or {})},
            "policy": {"kept": "the terms defined in the published RDF, RDFS, OWL 2 and XSD vocabularies (rule 36), "
                               "data/vocab/w3c-terms.json", "kept_terms_sha256": kept_terms_sha256(),
                       "base": self.base, "literals": self.literals,
                       "width": WIDTH, "name_properties": sorted(NAME_PROPERTIES),
                       "value_types": sorted(VALUE_TYPES), "strings_keyed_by": "lexical form"},
            "seed": self.seed,
            "graph_name": list(self.graph_name),
            "iris": dict(sorted(self.iris.items())),
            "strings": [[lex, tok] for lex, tok in sorted(self.strings.items(), key=lambda kv: kv[1])],
        }

    @staticmethod
    def from_json(data: dict) -> "OpaqueMap":
        m = OpaqueMap(seed=data["seed"], base=data["policy"]["base"], literals=data["policy"]["literals"],
                      graph_name=tuple(data["graph_name"]))
        m.iris = dict(data["iris"])
        if any(len(row) != 2 for row in data["strings"]):
            raise ValueError("a map whose strings are keyed by (lexical, language) predates the lexical-only policy")
        m.strings = {lex: tok for lex, tok in data["strings"]}
        return m

    def to_bytes(self, header: dict | None = None) -> bytes:
        """The held-out map file's exact bytes (gzipped JSON, no timestamp), so its sha256 is known before writing."""
        text = json.dumps(self.to_json(header), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        return gzip.compress(text.encode("utf-8"), mtime=0)

    def write(self, path: Path, header: dict | None = None) -> None:
        """The held-out map (heldout/...): never shown to agents or to the generator."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_bytes(header)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.chmod(path, 0o600)  # an existing file keeps its mode under O_CREAT
        with open(fd, "wb") as fh:
            fh.write(data)


def _census_iris(c: OntologySource) -> tuple[set[str], set[str]]:
    """(IRIs used as predicates, all IRIs) of a census."""
    preds = set(c.properties) | {p.predicate for p in c.patterns} | set(c.position_support or {}) | set(c.untyped_fillers or {})
    out = set(preds) | set(c.classes) | set(c.type_support or {}) | set(c.class_support or {})
    for ci in c.classes.values():
        out |= set(ci.parents)
    for pi in c.properties.values():
        out |= set(pi.parents) | set(pi.datatype_range)
        for e in pi.domain + pi.range:
            out |= set(e.members)
    for p in c.patterns:
        out |= {x for x in (p.subject_class, p.object_class, p.object_datatype) if x}
    for d in (c.position_support or {}).values():
        for pos in d.values():
            out |= set(pos)
    return preds, out


def _sample_terms(sample: InstanceSample) -> tuple[set[str], set[str], list[tuple[str, str | None, str | None, str]]]:
    preds, iris, literals = set(), set(), []
    for inst in sample.instances:
        iris.add(_expand(inst["iri"]))
        iris |= {_expand(c) for c in inst["sampled_for"]}
        for p, o in inst["triples"]:
            pe = _expand(p)
            preds.add(pe)
            if o.startswith('"'):
                lex, dt, lang = _parse_literal(o)
                literals.append((lex, dt, lang, pe))
                if dt:
                    iris.add(dt)
            elif o != "_:b":
                iris.add(_expand(o))
    return preds, iris | preds, literals


def masked(datatype: str | None, predicate: str, literals: str) -> bool:
    """Whether a literal's lexical form is masked under the policy."""
    if literals == "strings":
        return not is_value(datatype)
    return predicate in NAME_PROPERTIES and not is_value(datatype)


def _input_terms(census: OntologySource, sample: InstanceSample, literals: str) -> tuple[set[str], set[str], set[str]]:
    """(IRIs used as predicates, other IRIs, masked lexical forms) of the input, kept terms left out."""
    cp, ci = _census_iris(census)
    sp, si, lits = _sample_terms(sample)
    preds = {i for i in cp | sp if not kept(i)}
    others = {i for i in ci | si if not kept(i)} - preds
    return preds, others, {lex for lex, dt, _lang, p in lits if masked(dt, p, literals)}


def input_originals(census: OntologySource, sample: InstanceSample, literals: str = "strings") -> tuple[set[str], set[str]]:
    """Every name of the current input that the map must rename: (IRIs, masked lexical forms)."""
    preds, others, strings = _input_terms(census, sample, literals)
    return preds | others, strings


def same_map(a: OpaqueMap, b: OpaqueMap) -> bool:
    """The same map, whatever its header says (the header records where it was built)."""
    ja, jb = a.to_json(), b.to_json()
    ja.pop("header"), jb.pop("header")
    return ja == jb


def build_map(census: OntologySource, sample: InstanceSample, literals: str = "strings", *, seed: int,
              base: str = OPAQUE_BASE) -> OpaqueMap:
    if literals not in ("names", "strings"):
        raise ValueError("literals is 'names' or 'strings'")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("the map's seed is required (an integer), so the map can be rebuilt and audited")
    preds, others, masked_strings = _input_terms(census, sample, literals)
    if base.startswith(W3C_NAMESPACES) or any(i.startswith(base) for i in preds | others):
        raise ValueError(f"an original IRI is already in the opaque namespace {base}")
    rng = random.Random(seed)
    m = OpaqueMap(seed=seed, base=base, literals=literals, graph_name=(census.graph, "g0000001"))
    for kind, terms in (("p", preds), ("e", others)):
        terms = sorted(terms)
        numbers = rng.sample(range(1, len(terms) + 1), len(terms))
        for t, n in zip(terms, numbers, strict=True):
            m.iris[t] = f"{base}{kind}{n:0{WIDTH}d}"
    chosen = sorted(masked_strings)
    numbers = rng.sample(range(1, len(chosen) + 1), len(chosen))
    for key, n in zip(chosen, numbers, strict=True):
        m.strings[key] = f"s{n:0{WIDTH}d}"
    return m


def rename_census(census: OntologySource, m: OpaqueMap) -> OntologySource:
    f = m.iri
    out = OntologySource(
        name=census.name + "-opaque",
        kind="census",
        graph=m.graph_name[1],
        membership=census.membership,
        classes={f(k): ClassInfo(f(k), parents=sorted(f(p) for p in v.parents)) for k, v in census.classes.items()},
        properties={
            f(k): PropertyInfo(
                f(k), kind=v.kind, parents=sorted(f(p) for p in v.parents),
                domain=[ClassExpr(tuple(sorted(f(x) for x in e.members))) for e in v.domain],
                range=[ClassExpr(tuple(sorted(f(x) for x in e.members))) for e in v.range],
                datatype_range=[f(d) for d in v.datatype_range],
            )
            for k, v in census.properties.items()
        },
        patterns=sorted((Pattern(f(p.subject_class), f(p.predicate), f(p.object_class), f(p.object_datatype), p.support,
                                 p.reading) for p in census.patterns),
                        key=lambda x: (x.reading, x.predicate, x.subject_class, x.object_class or "", x.object_datatype or "")),
        support_threshold=census.support_threshold,
        type_support={f(k): n for k, n in (census.type_support or {}).items()},
        class_support={f(k): n for k, n in (census.class_support or {}).items()},
        untyped_fillers={f(k): dict(v) for k, v in (census.untyped_fillers or {}).items()},
        position_support={f(k): {pos: {f(c): n for c, n in d.items()} for pos, d in v.items()}
                          for k, v in (census.position_support or {}).items()},
        provenance={k: copy.deepcopy(census.provenance[k]) for k in
                    ("support", "domain_range_rule", "untyped_sentinel") if k in census.provenance},
    )
    # The reading without any graph-specific note (Rhea's names rh:Compound_N): the generic base, and the
    # generic note on named restrictions where the original had it.
    reading = census.provenance.get("membership_reading", "")
    out.provenance["membership_reading"] = MEMBERSHIP_READING_BASE + "." + (
        NAMED_RESTRICTIONS_READING if NAMED_RESTRICTIONS_READING in reading else "")
    out.classes = dict(sorted(out.classes.items()))
    out.properties = dict(sorted(out.properties.items()))
    return out


def rename_sample(sample: InstanceSample, m: OpaqueMap) -> InstanceSample:
    def term(t: str) -> str:
        if t.startswith('"'):
            lex, dt, lang = _parse_literal(t)
            if not is_value(dt) and lex in m.strings:
                lex = m.strings[lex]
            return _format_literal(lex, m.iri(dt), lang)
        if t == "_:b":
            return t
        return compact(m.iri(_expand(t)))[0]

    instances = []
    for inst in sample.instances:
        instances.append({
            "iri": term(inst["iri"]),
            "sampled_for": sorted(term(c) for c in inst["sampled_for"]),
            "triples": sorted(([term(p), term(o)] for p, o in inst["triples"])),
        })
    instances.sort(key=lambda i: i["iri"])
    spec = dict(sample.spec)
    spec["data_graph"] = m.graph_name[1]
    return InstanceSample(spec, instances)


DATABASE_NAMES = re.compile(r"rhea|chebi|uniprot|pubmed|obolibrary|expasy", re.I)


def _namespace(iri: str) -> str:
    cut = max(iri.rfind("#"), iri.rfind("/"))
    return iri[: cut + 1]


def _literal_values(text: str) -> set[str] | None:
    """Every string value of a bundle's census-and-sample payload, with the sample's Turtle literals read
    to their lexical forms; None if the text holds no payload."""
    if "<census_and_sample>" not in text:
        return None
    start = text.index("<census_and_sample>\n") + len("<census_and_sample>\n")
    payload = json.loads(text[start:text.index("\n</census_and_sample>")])
    out: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                out.add(k)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            if x[:1] in ('"', "'"):  # a sample literal: only a literal of a non-value type can carry a masked string
                try:
                    lex, dt, _lang = _parse_literal(x)
                except (ValueError, json.JSONDecodeError, KeyError):
                    out.add(x)
                else:
                    if not is_value(dt):
                        out.add(lex)
            else:
                out.add(x)

    walk(payload)
    return out


def check_leaks(text: str, m: OpaqueMap, originals: set[str] | None = None, strings: set[str] | None = None) -> list[str]:
    """Names of the original input found in a renamed text: original IRIs (full or prefixed), their
    namespaces, the graph name, database names, and masked strings (compared with the payload's string values
    and literals when the text is a bundle; otherwise with the text, for strings that are not digits only).
    The originals are the map's and, when given, the current input's (``originals``, ``strings``), so a name the
    map does not know is still looked for. Empty means no leak found."""
    leaks = []
    iris = set(m.iris) | {i for i in (originals or ()) if not kept(i)}
    namespaces = {ns for _, ns in PREFIX_TABLE if not ns.startswith(W3C_NAMESPACES) and ns != m.base}
    used_ns = {ns for ns in namespaces if any(i.startswith(ns) for i in iris)}
    # The namespace of every original IRI, whatever the prefix table knows (UniProt's store also holds the rhea,
    # chebi, GO, taxonomy and enzyme graphs). The W3C namespaces hold the kept terms, so they are no leak; an
    # undefined term in them is found by its own name, full or prefixed.
    for ns in sorted({_namespace(i) for i in iris} - used_ns):
        if len(ns) > len("http://x/") and not ns.startswith(W3C_NAMESPACES) and ns in text:
            leaks.append(f"namespace {ns}")
    for ns in sorted(used_ns):
        prefix = next(p for p, n in PREFIX_TABLE if n == ns)
        if ns in text or re.search(rf'(?<![\w/]){re.escape(prefix)}:[\w.-]', text):
            leaks.append(f"namespace {ns}")
    for iri in sorted(iris):
        if iri.startswith(W3C_NAMESPACES):
            short = compact(iri)[0]
            if re.search(rf"{re.escape(iri)}(?![\w.-])|(?<![\w/#:.-]){re.escape(short)}(?![\w.-])", text):
                leaks.append(f"IRI {iri}")
        elif iri in text:
            leaks.append(f"IRI {iri}")
    if m.graph_name[0] and m.graph_name[0] in text:
        leaks.append(f"graph name {m.graph_name[0]}")
    for name in sorted({x.lower() for x in DATABASE_NAMES.findall(text)}):
        leaks.append(f"database name {name}")
    values_ = _literal_values(text)
    for lex in sorted(set(m.strings) | set(strings or ())):
        if values_ is not None:
            if lex in values_:
                leaks.append(f"string {lex[:60]!r}")
        elif len(lex) >= 4 and not lex.isdigit() and json.dumps(lex, ensure_ascii=False)[1:-1] in text:
            leaks.append(f"string {lex[:60]!r}")
    return leaks


def verify(m: OpaqueMap, renamed_text: str, current: tuple[set[str], set[str]] | None = None) -> dict:
    """Independent checks of the map: one-to-one, disjoint from the originals, covering every name of the current
    input (``current``: ``input_originals`` of the input renamed), and no leak in the renamed input of the map's
    originals or the current input's."""
    iri_values = list(m.iris.values())
    str_values = list(m.strings.values())
    cur_iris, cur_strings = current if current is not None else (set(), set())
    originals = set(m.iris) | set(m.strings) | cur_iris | cur_strings
    uncovered = {"iris": len({i for i in cur_iris if not kept(i)} - set(m.iris)), "strings": len(cur_strings - set(m.strings))}
    report = {
        "iris": len(m.iris),
        "strings": len(m.strings),
        "one_to_one": len(set(iri_values)) == len(iri_values) and len(set(str_values)) == len(str_values),
        "opaque_disjoint_from_originals": not (set(iri_values) | set(str_values)) & originals,
        "tokens_well_formed": all(_TOKEN.match(v[len(m.base):]) for v in iri_values) and all(_TOKEN.match(v) for v in str_values),
        "current_input_checked": current is not None,
        "uncovered_by_map": uncovered,
        "leaks": check_leaks(renamed_text, m, cur_iris, cur_strings),
    }
    report["ok"] = report["one_to_one"] and report["opaque_disjoint_from_originals"] and report["tokens_well_formed"] \
        and not any(uncovered.values()) and not report["leaks"]
    return report


def stale_draw(record: dict, bundle: dict, map_sha256: str) -> str | None:
    """Why a decontaminated draw is not current, or None: it must have been made from the current bundle's exact
    request and with the map the bundle records, which must be the map given (External reviews of cf5c000)."""
    dc = bundle.get("decontamination") or {}
    if dc.get("map_sha256") != map_sha256:
        return f"the bundle was renamed with map {dc.get('map_sha256')}, not the given map {map_sha256}"
    if record.get("request_sha256") != bundle.get("request_sha256"):
        return f"made from request {record.get('request_sha256')}, not the current {bundle.get('request_sha256')}"
    if record.get("map_sha256") != map_sha256:
        return f"made with map {record.get('map_sha256')}, not the current {map_sha256}"
    return None


def map_back_turtle(turtle: str, m: OpaqueMap) -> tuple[str, dict]:
    """A generated ontology over opaque names, with every mapped IRI put back (design log ruling: IRIs only).
    Literals are never mapped back: the labels and comments the model writes are its own, and are counted (with
    how many copy a placeholder token of the input). Terms outside the map stay as they are and are reported."""
    from rdflib import Graph, Literal as RLit, URIRef

    g = Graph()
    g.parse(data=turtle, format="turtle")
    inv_i = m.inverse_iris()
    placeholders = set(m.strings.values())
    invented, original_namespace = set(), set()
    out = Graph()
    namespaces = {ns for _, ns in PREFIX_TABLE if not ns.startswith(W3C_NAMESPACES)}
    text_props = {URIRef(x) for x in NAME_PROPERTIES}
    texts = {"labels": 0, "comments": 0, "copied_placeholders": 0}

    def back(t):
        if isinstance(t, URIRef):
            s_ = str(t)
            if s_ in inv_i:
                return URIRef(inv_i[s_])
            if s_.startswith(m.base):
                invented.add(s_)
            elif not kept(s_) and any(s_.startswith(ns) for ns in namespaces):
                original_namespace.add(s_)
        return t

    for s_, p_, o_ in g:
        if p_ in text_props and isinstance(o_, RLit):
            texts["labels" if str(p_).endswith(("label", "prefLabel", "altLabel")) else "comments"] += 1
            texts["copied_placeholders"] += str(o_) in placeholders
        out.add((back(s_), back(p_), back(o_)))
    report = {
        "triples": len(g),
        "invented_opaque_terms": sorted(invented),
        "original_namespace_terms": sorted(original_namespace),
        "model_texts": texts,
        "rule": "IRIs are mapped back; literals never are (the model's own labels and comments are counted); terms not "
                "in the map stay as they are and are reported (design log, criterion 4 and its literal-policy ruling)",
    }
    return out.serialize(format="turtle"), report
