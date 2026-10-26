"""Map questions and gold answers to the opaque copy of a graph, and back.

The renaming is the held-out map written by ``stores/opaque/rename.py`` (one
consistent, one-to-one map on RDF terms). This module applies it to a question
file: the question text, the gold SPARQL and the gold rows. Every mapping is
checked by mapping back:

- the question text maps back to the original text exactly;
- the gold SPARQL maps back to the original query, compared token by token
  with prefixed names expanded (PREFIX lines for non-W3C namespaces are dropped
  from the opaque query, because their namespace IRIs would name the source);
- the gold rows map back to the original gold rows exactly;
- with ``--endpoint``, each mapped gold query runs on the opaque store and its
  rows, mapped back, must equal the original gold rows exactly.

Question text is mapped in two steps:

1. From the record's ``params``, not by searching the text: each renamed
   parameter value is replaced where it stands as a whole token. A ChEBI CURIE
   (``CHEBI:15377``) becomes the opaque identifier literal (``TA:N'``), whose
   number is also the number in the class's opaque IRI and opaque
   ``oboInOwl:id`` (design log, ruling 4). Parameters the map leaves unchanged (a
   label that is also a kept formula, for instance) are reported as leaks.
2. A fixed substitution (``SUBSTITUTIONS``) takes the database names out of
   the template prose ("Rhea" becomes "the reaction graph", "ChEBI" becomes
   "the chemical class hierarchy"; design log, ruling 2). Concept words such as
   "EC number" stay. A database name left over stops the mapping. The
   map-back inverts both steps.

The map stays out of agents' reach (``heldout/``). The opaque question file
carries only opaque values in ``params``.

Usage:
  python -m harness.opaque --map heldout/opaque-rhea-142/map.tsv.gz \\
      --questions questions.jsonl --out opaque.jsonl --report report.json \\
      [--endpoint http://localhost:7032/]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from harness.terms import IRI, BNode, Cell, Literal, Row, row_from_json, row_to_json

W3C_NAMESPACES = (
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2001/XMLSchema#",
)
XSD = "http://www.w3.org/2001/XMLSchema#"
INTEGER_TYPES = {XSD + t for t in (
    "int", "integer", "long", "short", "byte", "nonNegativeInteger", "positiveInteger",
    "nonPositiveInteger", "negativeInteger", "unsignedInt", "unsignedLong", "unsignedShort", "unsignedByte")}
# Opaque tokens: <base>[gpe]NNNNNNN for IRIs, sNNNNNNN for strings.
TOKEN = re.compile(r"(?<![A-Za-z0-9_])(?:[gpesuvw][0-9]{7}|T[A-Z]:[0-9]{7})(?![A-Za-z0-9_])")

# How each question parameter maps. IRIs are recognised by their value.
# label_c, name_c and the reference names ref_1, ref_2 (with their ref_N_entity
# IRIs) come with the 0.4 generator; its *_homonym, any_status and having_max
# parameters are booleans and integers, which pass through unchanged.
LITERAL_PARAMS = frozenset({"label", "name", "accession", "label_a", "label_b", "label_c", "name_a", "name_b",
                            "name_c", "ref_1", "ref_2"})
CURIE_PARAMS = {"curie": ("CHEBI:", "http://purl.obolibrary.org/obo/CHEBI_")}
# Identifiers that are an IRI's local name and have no literal in the data (EC
# numbers): they become the local name of the opaque IRI, which an agent turns
# back into the IRI with the opaque base, as it would build the EC IRI.
IRI_LOCAL_PARAMS = {"ec_number": "http://purl.uniprot.org/enzyme/"}
# '|'-joined entity IRIs (the generator's key for pair questions).
IRI_LIST_PARAMS = frozenset({"key"})
PLAIN_PARAMS = frozenset({"variant", "op", "threshold", "comparison", "k", "k_words"})
# Fields that carry the original question id or condition name by design.
ID_FIELDS = frozenset({"id", "original_id", "base_question_id", "graph"})
# Words in template prose that name the source vocabulary (reported).
PROSE_MARKERS = ("ChEBI", "CHEBI", "Rhea", "RHEA", "PubMed", "EC number", "InChIKey", "SMILES", "UniProt")
# Database names that must not survive the substitution below.
DATABASE_NAMES = re.compile(r"Rhea|RHEA|ChEBI|CHEBI|PubMed|UniProt")

# The fixed substitution that takes database names out of the opaque question
# text (ruling 2 in the design log). It covers every phrase the templates use, as
# enumerated from the question file; a name left over stops the mapping, so a
# new template phrase cannot slip through. {X} is a label with its identifier,
# carried over unchanged:
#   LID: "label" (identifier)      IDL: identifier ("label")
# The substitution is invertible: no target occurs in a question before
# substitution, and every question's map-back inverts it exactly.
_LID = r'"[^"]*" \([^()]*\)'
_IDL = r'\S+ \("[^"]*"\)'
SUBSTITUTIONS: tuple[tuple[str, str, str | None], ...] = (
    # (source, target, pattern of {X} or None)
    ("the ChEBI class {X}, as recorded in ChEBI", "the class {X}, as recorded in the chemical class hierarchy", _LID),
    ("the ChEBI class {X}", "the class {X} in the chemical class hierarchy", _LID),
    ("The Rhea generic compound {X}", "The generic compound {X} in the reaction graph", _LID),
    ("the Rhea reaction {X}", "the reaction {X} in the reaction graph", _IDL),
    ("ChEBI classes, not Rhea compounds",
     "classes in the chemical class hierarchy, not the reaction graph's own compound entries", None),
    ("the Rhea reaction hierarchy", "the reaction hierarchy of the reaction graph", None),
    ("the ChEBI subclass hierarchy", "the subclass hierarchy of the chemical class hierarchy", None),
    ("the ChEBI hierarchy", "the chemical class hierarchy", None),
    ("the ChEBI class IRI", "the class IRI in the chemical class hierarchy", None),
    ("whose ChEBI class", "whose class in the chemical class hierarchy", None),
    ("its ChEBI class", "its class in the chemical class hierarchy", None),
    ("per ChEBI class", "per class in the chemical class hierarchy", None),
    ("a ChEBI class", "a class in the chemical class hierarchy", None),
    ("ChEBI classes", "classes in the chemical class hierarchy", None),
    ("Rhea small-molecule compounds", "small-molecule compounds in the reaction graph", None),
    ("the Rhea compound IRI", "the compound IRI in the reaction graph", None),
    ("Rhea reactions", "reactions in the reaction graph", None),
    ("does Rhea", "does the reaction graph", None),
    ("recorded in Rhea", "recorded in the reaction graph", None),
    ("recorded in ChEBI", "recorded in the chemical class hierarchy", None),
    ("that ChEBI gives", "that the chemical class hierarchy gives", None),
    ("the PubMed IRI", "the publication IRI", None),
)


class MappingError(ValueError):
    """A value the renaming cannot map, or a round trip that fails."""


def _tag(datatype: str | None, lang: str | None) -> str:
    """Lookup tag of a literal: "" (simple or xsd:string), "@lang", "int" or the datatype."""
    if lang:
        return "@" + lang
    if datatype in (None, "", XSD + "string"):
        return ""
    if datatype in INTEGER_TYPES:
        return "int"  # QLever returns xsd:long as xsd:int; the map keys the source datatype
    return datatype


def _unescape_field(value: str) -> str:
    return re.sub(r"\\(.)", lambda m: {"t": "\t", "n": "\n", "r": "\r"}.get(m.group(1), m.group(1)), value)


@dataclass
class OpaqueMap:
    iris: dict[str, str]
    literals: dict[tuple[str, str], str]
    header: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.iris_inv = {v: k for k, v in self.iris.items()}
        self.literals_inv = {(v, tag): k for (k, tag), v in self.literals.items()}
        if len(self.iris_inv) != len(self.iris) or len(self.literals_inv) != len(self.literals):
            raise MappingError("the map is not one-to-one")
        self.base = self.header.get("opaque_iri_base", "http://example.org/opaque/")

    @classmethod
    def load(cls, path: str) -> "OpaqueMap":
        iris: dict[str, str] = {}
        lits: dict[tuple[str, str], str] = {}
        header: dict[str, Any] = {}
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", newline="\n") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("# {"):
                    header = json.loads(line[2:])
                    continue
                if not line or line.startswith("#"):
                    continue
                kind, original, tag, new = line.split("\t")
                if kind == "I":
                    iris[_unescape_field(original)] = new
                else:
                    tag = _unescape_field(tag)
                    norm = tag if tag.startswith("@") else _tag(tag or None, None)
                    lits[(_unescape_field(original), norm)] = new
        return cls(iris, lits, header)

    # -- terms
    def iri(self, value: str, inverse: bool = False) -> str:
        table = self.iris_inv if inverse else self.iris
        if value in table:
            return table[value]
        if value.startswith(W3C_NAMESPACES):
            return value
        raise MappingError(f"IRI not in the map: {value}")

    def literal(self, lexical: str, datatype: str | None, lang: str | None, inverse: bool = False) -> str:
        table = self.literals_inv if inverse else self.literals
        return table.get((lexical, _tag(datatype, lang)), lexical)

    def term(self, cell: Cell, inverse: bool = False) -> Cell:
        if cell is None or isinstance(cell, BNode):
            return cell
        if isinstance(cell, IRI):
            return IRI(self.iri(cell.value, inverse))
        return Literal(self.literal(cell.lexical, cell.datatype, cell.lang, inverse), cell.datatype, cell.lang)

    def row(self, row: Row, inverse: bool = False) -> Row:
        return tuple(self.term(c, inverse) for c in row)

    def curie(self, value: str, prefix: str, namespace: str) -> str:
        """The opaque identifier literal for a CURIE such as CHEBI:15377 (ruling 4 in the design log).

        The CURIE has the form of the identifier literals with that prefix
        (rh:accession "CHEBI:57618", oboInOwl:hasAlternativeId), so it maps by
        the same rule: opaque prefix plus the permuted number, which is also
        the number in the class's opaque IRI.
        """
        if not value.startswith(prefix) or not value[len(prefix):].isdigit():
            raise MappingError(f"not a {prefix} CURIE: {value!r}")
        rel = self.header.get("identifier_relations", {})
        rules = [r for r in rel.get("literals", []) if r["prefix"] == prefix and namespace in r.get("iri", [])]
        families = [f["opaque"] for sp in rel.get("spaces", {}).values() for f in sp["iris"] if f["prefix"] == namespace]
        if len({r["opaque"] for r in rules}) != 1 or len(families) != 1:
            raise MappingError(f"the map defines no single identifier rule for {prefix} CURIEs")
        opaque_iri = self.iri(namespace + value[len(prefix):])
        local = opaque_iri[len(self.base) + len(families[0]):]
        if not opaque_iri.startswith(self.base + families[0]) or not local.isdigit():
            raise MappingError(f"{value} does not map into the {prefix} identifier family")
        out = rules[0]["opaque"] + local
        known = self.literals.get((value, ""))
        if known is not None and known != out:
            raise MappingError(f"{value}: the literal map and the identifier rule disagree")
        return out


# ---------------------------------------------------------------- rows

def canonical(rows: Iterable[Row]) -> list[list[dict[str, str] | None]]:
    """Rows in JSON form, sorted by canonical JSON (as the question generator stores them)."""
    as_json = [row_to_json(r) for r in rows]
    return sorted(as_json, key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False))


def map_rows(m: OpaqueMap, rows_json: list[list[Any]], inverse: bool = False) -> list[list[Any]]:
    return canonical(m.row(row_from_json(r), inverse) for r in rows_json)


# ---------------------------------------------------------------- SPARQL

_IRIREF = r"<[^<>\"{}|^`\\\x00-\x20]*>"
_PNAME = r"(?:[A-Za-z][A-Za-z0-9_.-]*)?:(?:[A-Za-z0-9_:%-](?:[A-Za-z0-9_.:%-]*[A-Za-z0-9_:%-])?)?"
_STRING = (r'"""(?:[^"\\]|\\.|"(?!""))*"""' r"|'''(?:[^'\\]|\\.|'(?!''))*'''"
           r'|"(?:[^"\\\n]|\\.)*"' r"|'(?:[^'\\\n]|\\.)*'")
_TOKEN = re.compile(
    r"(?P<ws>\s+)|(?P<comment>#[^\n]*)"
    rf"|(?P<string>(?:{_STRING})(?:@[A-Za-z]+(?:-[A-Za-z0-9]+)*|\^\^(?:{_IRIREF}|{_PNAME}))?)"
    rf"|(?P<iri>{_IRIREF})|(?P<var>[?$][A-Za-z0-9_]+)|(?P<pname>{_PNAME})"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_]*)|(?P<num>[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)|(?P<other>.)",
    re.S,
)
_PREFIX_DECL = re.compile(r"\A\s*PREFIX\s+([A-Za-z][A-Za-z0-9_.-]*)?:\s*<([^>]*)>", re.I)
_BASE_DECL = re.compile(r"\A\s*BASE\s*<([^>]*)>", re.I)
_SPARQL_ESC = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f", '"': '"', "'": "'", "\\": "\\"}


def _split_prologue(query: str) -> tuple[dict[str, str], str]:
    prefixes: dict[str, str] = {}
    rest = query
    while True:
        m = _PREFIX_DECL.match(rest)
        if m:
            prefixes[m.group(1) or ""] = m.group(2)
            rest = rest[m.end():]
            continue
        if _BASE_DECL.match(rest):
            raise MappingError("BASE declarations are not supported")
        stripped = rest.lstrip()
        if stripped.startswith("#"):
            rest = stripped.split("\n", 1)[1] if "\n" in stripped else ""
            continue
        return prefixes, rest


def _tokens(body: str) -> list[tuple[str, str]]:
    out = []
    pos = 0
    while pos < len(body):
        m = _TOKEN.match(body, pos)
        kind = m.lastgroup
        out.append((kind, m.group()))
        pos = m.end()
    return out


def _expand(pname: str, prefixes: dict[str, str]) -> str:
    pre, local = pname.split(":", 1)
    if pre not in prefixes:
        raise MappingError(f"undeclared prefix in {pname}")
    return prefixes[pre] + local.replace("\\", "")


def _parse_string(token: str, prefixes: dict[str, str]) -> tuple[str, str | None, str | None, str]:
    """(lexical form, datatype, lang, raw suffix) of a SPARQL string token."""
    q = token[:3] if token[:3] in ('"""', "'''") else token[0]
    end = token.rindex(q)
    body, suffix = token[len(q):end], token[end + len(q):]
    lexical = re.sub(r"\\(.)", lambda m: _SPARQL_ESC.get(m.group(1), "\\" + m.group(1)), body)
    datatype = lang = None
    if suffix.startswith("@"):
        lang = suffix[1:]
    elif suffix.startswith("^^"):
        dt = suffix[2:]
        datatype = dt[1:-1] if dt.startswith("<") else _expand(dt, prefixes)
    return lexical, datatype, lang, suffix


def _quote(lexical: str) -> str:
    return '"' + lexical.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r") + '"'


def map_sparql(m: OpaqueMap, query: str, inverse: bool = False) -> str:
    """The query with every renamed IRI and string constant replaced.

    Prefixed names of W3C namespaces stay prefixed; every other IRI is written
    in full, and PREFIX lines for non-W3C namespaces are dropped. Unquoted
    numbers are never mapped.
    """
    prefixes, body = _split_prologue(query)
    out: list[str] = []
    used_w3c: set[str] = set()
    for kind, text in _tokens(body):
        if kind == "iri":
            out.append(f"<{m.iri(text[1:-1], inverse)}>")
        elif kind == "pname":
            iri = _expand(text, prefixes)
            if iri.startswith(W3C_NAMESPACES):
                used_w3c.add(text.split(":", 1)[0])
                out.append(text)
            else:
                out.append(f"<{m.iri(iri, inverse)}>")
        elif kind == "string":
            lexical, datatype, lang, suffix = _parse_string(text, prefixes)
            new = m.literal(lexical, datatype, lang, inverse)
            if suffix.startswith("^^") and not suffix.startswith("^^<"):
                used_w3c.add(suffix[2:].split(":", 1)[0])
            out.append(text if new == lexical else _quote(new) + suffix)
        else:
            out.append(text)
    head = "".join(f"PREFIX {p}: <{prefixes[p]}>\n" for p in sorted(used_w3c) if p in prefixes
                   and prefixes[p].startswith(W3C_NAMESPACES))
    return head + "".join(out).lstrip("\n")


def normal_form(query: str) -> list[tuple[str, Any]]:
    """Tokens without whitespace, comments or prologue; IRIs expanded; strings as terms."""
    prefixes, body = _split_prologue(query)
    out: list[tuple[str, Any]] = []
    for kind, text in _tokens(body):
        if kind in ("ws", "comment"):
            continue
        if kind == "pname":
            out.append(("iri", _expand(text, prefixes)))
        elif kind == "iri":
            out.append(("iri", text[1:-1]))
        elif kind == "string":
            lexical, datatype, lang, _ = _parse_string(text, prefixes)
            out.append(("literal", (lexical, _tag(datatype, lang))))
        elif kind == "word":
            out.append((kind, text.upper()))
        else:
            out.append((kind, text))
    return out


# ---------------------------------------------------------------- text and params

def _replace_all(text: str, pairs: dict[str, str]) -> tuple[str, Counter]:
    """Replace each key standing as a whole token, longest first, in one pass."""
    if not pairs:
        return text, Counter()
    alternation = "|".join(re.escape(k) for k in sorted(pairs, key=len, reverse=True))
    pattern = re.compile(rf"(?<![A-Za-z0-9_])(?:{alternation})(?![A-Za-z0-9_])")
    hits: Counter = Counter()

    def sub(mm: re.Match) -> str:
        hits[mm.group()] += 1
        return pairs[mm.group()]

    return pattern.sub(sub, text), hits


def map_params(m: OpaqueMap, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Opaque params, the text replacements they imply, and the parameters left as they were."""
    out: dict[str, Any] = {}
    text_pairs: dict[str, str] = {}
    kept: list[str] = []
    for key, value in params.items():
        if not isinstance(value, str) or key in PLAIN_PARAMS:
            out[key] = value
            continue
        if key in IRI_LIST_PARAMS:
            out[key] = "|".join(m.iri(v) for v in value.split("|"))
        elif value.startswith(("http://", "https://")):
            out[key] = m.iri(value)
        elif key in IRI_LOCAL_PARAMS:
            opaque_iri = m.iri(IRI_LOCAL_PARAMS[key] + value)
            if not opaque_iri.startswith(m.base):
                raise MappingError(f"{key}={value!r} does not map to an opaque IRI")
            out[key] = opaque_iri[len(m.base):]
        elif key in CURIE_PARAMS:
            out[key] = m.curie(value, *CURIE_PARAMS[key])
        elif key in LITERAL_PARAMS:
            out[key] = m.literal(value, None, None)
        else:
            raise MappingError(f"parameter {key!r} has no mapping rule")
        if out[key] == value:
            kept.append(key)
        elif key in LITERAL_PARAMS or key in CURIE_PARAMS or key in IRI_LOCAL_PARAMS:
            if text_pairs.get(value, out[key]) != out[key]:
                raise MappingError(f"parameter value {value!r} maps two ways")
            text_pairs[value] = out[key]
    return out, text_pairs, kept


def map_anchors(m: OpaqueMap, anchors: list[dict[str, Any]], pairs: dict[str, str]) -> list[dict[str, Any]]:
    """The generator's anchor list (entity and the name it is called by), mapped like the params."""
    out = []
    for a in anchors:
        b = dict(a)
        if "entity" in b:
            b["entity"] = m.iri(b["entity"])
        if "name" in b:
            b["name"] = pairs.get(b["name"], m.literal(b["name"], None, None))
        out.append(b)
    return out


METADATA_NAMES = re.compile(r"rhea|chebi|pubmed|uniprot", re.I)


def _metadata_names(record: dict[str, Any]) -> list[str]:
    """Every database name outside the question text and the id fields, as "field: match".

    Policy: agents see only the question text, so names in metadata (variant
    names, answer column names, gold SPARQL variables) are listed, not renamed.
    """
    found: list[str] = []

    def walk(v: Any, path: str) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{path}.{k}")
        elif isinstance(v, list):
            for x in v:
                walk(x, f"{path}[]")
        elif isinstance(v, str):
            found.extend(f"{path}: {mm.group().lower()}" for mm in METADATA_NAMES.finditer(v))

    for k, v in record.items():
        if k not in ID_FIELDS and k != "question":
            walk(v, k)
    return found


def _leaks(record: dict[str, Any], m: OpaqueMap, originals: Iterable[str]) -> list[str]:
    """Fields of an opaque record that still hold an original IRI or a renamed parameter value."""
    found: list[str] = []
    names = {v for v in originals if v}

    def walk(v: Any, path: str) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{path}.{k}")
        elif isinstance(v, list):
            for x in v:
                walk(x, f"{path}[]")
        elif isinstance(v, str):
            for mm in re.finditer(r"https?://[^\s<>\"]+", v):
                if not mm.group().startswith((m.base,) + W3C_NAMESPACES):
                    found.append(f"{path}: {mm.group()[:80]}")
            if names:
                _, hits = _replace_all(v, {n: n for n in names})
                found.extend(f"{path}: {n[:60]!r}" for n in hits)

    for k, v in record.items():
        if k not in ID_FIELDS:
            walk(v, k)
    return found


def _phrase_regex(template: str, x: str | None, group: str) -> str:
    head, sep, tail = template.partition("{X}")
    if not sep:
        return re.escape(template)
    return re.escape(head) + f"(?P<{group}>{x})" + re.escape(tail)


def _build_substitution(inverse: bool) -> tuple[re.Pattern, list[tuple[str, str]]]:
    """One alternation over every source (or target) phrase, most specific first."""
    rules = []
    for i, (src, dst, x) in enumerate(SUBSTITUTIONS):
        frm, to = (dst, src) if inverse else (src, dst)
        rules.append((len(frm.replace("{X}", "")), i, frm, to, x))
    if inverse:  # longer targets first, so a target inside another is never taken alone
        rules.sort(key=lambda r: (-(r[4] is not None), -r[0], r[1]))
    alternatives = [f"(?P<r{i}>{_phrase_regex(frm, x, f'x{i}')})" for _, i, frm, _, x in rules]
    return re.compile("|".join(alternatives)), [(f"r{i}", to) for _, i, _, to, _ in rules]


_FORWARD, _FORWARD_RULES = _build_substitution(False)
_INVERSE, _INVERSE_RULES = _build_substitution(True)


def substitute(text: str, inverse: bool = False) -> tuple[str, Counter]:
    """Apply the database-name substitution (or its inverse) in one pass; count each rule used."""
    pattern, rules = (_INVERSE, _INVERSE_RULES) if inverse else (_FORWARD, _FORWARD_RULES)
    hits: Counter = Counter()

    def sub(m: re.Match) -> str:
        for name, to in rules:
            if m.group(name) is not None:
                i = int(name[1:])
                hits[SUBSTITUTIONS[i][0]] += 1
                return to.replace("{X}", m.group(f"x{i}")) if "{X}" in to else to
        raise AssertionError("no rule matched")

    return pattern.sub(sub, text), hits


# ---------------------------------------------------------------- records

@dataclass
class RecordReport:
    id: str
    text_round_trip: bool = False
    text_replacements: dict[str, int] = field(default_factory=dict)
    params_left_unchanged: list[str] = field(default_factory=list)
    params_not_in_text: list[str] = field(default_factory=list)
    substitutions: dict[str, int] = field(default_factory=dict)
    source_database_names: int = 0
    metadata_database_names: list[str] = field(default_factory=list)
    prose_markers: list[str] = field(default_factory=list)
    sparql_round_trip: bool = False
    rows_round_trip: bool = False
    store_check: str = "not run"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return (self.error is None and self.text_round_trip and self.sparql_round_trip and self.rows_round_trip
                and self.store_check == "ok")


def map_record(m: OpaqueMap, rec: dict[str, Any], require_neutral: bool = False) -> tuple[dict[str, Any], RecordReport]:
    """Map one question record. With ``require_neutral`` a database name in the
    original question's prose (identifiers such as CHEBI:15377 aside) is an error,
    so the substitution table acts only as a guard."""
    rep = RecordReport(rec["id"])
    if TOKEN.search(rec["question"]):
        raise MappingError("the original question already contains an opaque-shaped token")
    params, pairs, kept = map_params(m, rec.get("params", {}))
    rep.params_left_unchanged = kept
    prose, _ = _replace_all(rec["question"], {k: "{}" for k in pairs})
    source_names = [prose[max(0, mm.start() - 40):mm.end() + 40] for mm in DATABASE_NAMES.finditer(prose)]
    rep.source_database_names = len(source_names)
    if require_neutral and source_names:
        raise MappingError(f"database names in the original question: {source_names}")
    # Names (labels) are replaced where they stand in double quotes, as the
    # templates quote them, so a short name ("CoA", "A") never hits template
    # words; identifiers (accessions, CURIEs, EC numbers) stand bare.
    quoted = {v for k, v in rec.get("params", {}).items() if k in LITERAL_PARAMS - {"accession"}
              and isinstance(v, str) and f'"{v}"' in rec["question"]}
    text_map = {(f'"{k}"' if k in quoted else k): (f'"{v}"' if k in quoted else v) for k, v in pairs.items()}
    text, hits = _replace_all(rec["question"], text_map)
    rep.text_replacements = {text_map[k]: n for k, n in hits.items()}
    rep.params_not_in_text = sorted(k for k, v in rec.get("params", {}).items() if isinstance(v, str)
                                    and v in pairs and v not in hits and f'"{v}"' not in hits)
    _, residual = _replace_all(text, {k: k for k in text_map})
    if require_neutral:
        # Neutral source prose: the table is only a guard. Its targets ("the
        # reaction graph", ...) may occur naturally here, so nothing is
        # substituted or inverted; the name checks before and after stand.
        final, subs, unsubstituted = text, Counter(), text
    else:
        if _INVERSE.search(text):
            raise MappingError("a substitution target already occurs in the question; the substitution "
                               "would not be invertible")
        final, subs = substitute(text)
        unsubstituted, _ = substitute(final, inverse=True)
    left = DATABASE_NAMES.findall(final)
    if left:
        ctx = [final[max(0, mm.start() - 40):mm.end() + 40] for mm in DATABASE_NAMES.finditer(final)]
        raise MappingError(f"database names left after the substitution (extend SUBSTITUTIONS): {ctx}")
    rep.substitutions = dict(subs)
    back, _ = _replace_all(unsubstituted, {v: k for k, v in text_map.items()})
    rep.text_round_trip = unsubstituted == text and back == rec["question"] and not residual
    rep.prose_markers = [w for w in PROSE_MARKERS if w in final]
    text = final

    sparql = map_sparql(m, rec["gold_sparql"])
    rep.sparql_round_trip = normal_form(map_sparql(m, sparql, inverse=True)) == normal_form(rec["gold_sparql"])
    for marker in ("rhea-db", "obolibrary", "uniprot", "chemrof", "geneontology"):
        if marker in sparql:
            raise MappingError(f"opaque gold SPARQL still contains {marker!r}")

    rows = map_rows(m, rec["gold_rows"])
    rep.rows_round_trip = map_rows(m, rows, inverse=True) == rec["gold_rows"]

    out = dict(rec)
    out.update({
        "id": "opaque-" + rec["id"],
        "original_id": rec["id"],
        "graph": "opaque-" + str(rec.get("graph", "")),
        "question": text,
        "gold_sparql": sparql,
        "gold_rows": rows,
        "params": params,
    })
    if "anchors" in rec:
        out["anchors"] = map_anchors(m, rec["anchors"], pairs)
    rep.metadata_database_names = _metadata_names(out)
    leaks = _leaks(out, m, pairs)
    if leaks:
        raise MappingError(f"original values left in the opaque record: {leaks[:10]}")
    return out, rep


def store_check(m: OpaqueMap, client: Any, original: dict[str, Any], mapped: dict[str, Any]) -> str:
    """Run the mapped gold query on the opaque store; its rows mapped back must equal the original gold."""
    from harness.sparql import SparqlError

    try:
        result = client.query(mapped["gold_sparql"])
    except SparqlError as exc:
        return f"error: {exc.kind}: {str(exc)[:300]}"
    if result.truncated:
        return "error: row cap reached"
    back = canonical(m.row(r, inverse=True) for r in result.rows)
    if back != original["gold_rows"]:
        return f"rows differ: {len(back)} rows mapped back, {len(original['gold_rows'])} gold rows"
    if canonical(result.rows) != mapped["gold_rows"]:
        return "rows differ from the offline mapping"
    return "ok"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def question_lines(records: Iterable[dict[str, Any]]) -> Iterable[str]:
    """The lines of the opaque question file: records sorted by id, one JSON object per line."""
    for r in sorted(records, key=lambda r: r["id"]):
        yield json.dumps(r, ensure_ascii=False) + "\n"


def write_questions(path: str, records: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(question_lines(records))


def run(map_path: str, questions: str, out_path: str, report_path: str, endpoint: str | None = None,
        timeout_s: float = 300.0, require_neutral: bool = False) -> dict[str, Any]:
    m = OpaqueMap.load(map_path)
    client = None
    if endpoint:
        from harness.sparql import SparqlClient
        client = SparqlClient(endpoint, timeout_s=timeout_s)
    records = [json.loads(line) for line in open(questions, encoding="utf-8") if line.strip()]
    mapped_records, reports = [], []
    for rec in records:
        try:
            mapped, rep = map_record(m, rec, require_neutral)
        except MappingError as exc:
            rep = RecordReport(rec.get("id", "?"), error=str(exc))
            reports.append(rep)
            continue
        if client is not None:
            rep.store_check = store_check(m, client, rec, mapped)
        mapped_records.append(mapped)
        reports.append(rep)
    write_questions(out_path, mapped_records)
    summary = {
        "questions": len(records),
        "mapped": len(mapped_records),
        "all_ok": all(r.ok for r in reports) and len(mapped_records) == len(records),
        "text_round_trip_ok": sum(r.text_round_trip for r in reports),
        "sparql_round_trip_ok": sum(r.sparql_round_trip for r in reports),
        "rows_round_trip_ok": sum(r.rows_round_trip for r in reports),
        "store_check": dict(Counter(r.store_check.split(":")[0] for r in reports)),
        "questions_with_params_left_unchanged": sum(bool(r.params_left_unchanged) for r in reports),
        "questions_with_prose_markers": dict(Counter(w for r in reports for w in r.prose_markers)),
        "metadata_database_names": {
            "policy": "listed, not renamed: agents see only the question text",
            "occurrences": sum(len(r.metadata_database_names) for r in reports),
            "records": sum(bool(r.metadata_database_names) for r in reports),
            "by_field": dict(Counter(x.split(":")[0] for r in reports for x in r.metadata_database_names)),
        },
        "questions_changed_by_substitution": sum(bool(r.substitutions) for r in reports),
        "questions_with_database_names_in_source": sum(r.source_database_names > 0 for r in reports),
        "require_neutral_source": require_neutral,
        "substitution_rule_uses": dict(Counter(k for r in reports for k, n in r.substitutions.items()
                                               for _ in range(n))),
        "substitution_rule_questions": dict(Counter(k for r in reports for k in r.substitutions)),
        "errors": sum(r.error is not None for r in reports),
        "questions_sha256": _sha256(questions),
        "map_sha256": _sha256(map_path),
        "endpoint": endpoint,
    }
    report = {"summary": summary, "records": [r.__dict__ for r in reports]}
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Map a question file to the opaque store and check it.")
    ap.add_argument("--map", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--endpoint")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--require-neutral-source", action="store_true",
                    help="fail any question whose original prose names a database (for neutrally worded sets)")
    args = ap.parse_args(argv)
    summary = run(args.map, args.questions, args.out, args.report, args.endpoint, args.timeout,
                  args.require_neutral_source)
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
