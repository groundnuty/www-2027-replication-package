"""Scan files for any original name of a held-out map: the independent leak check of the opaque and the
decontaminated inputs (rule 34; design log, criterion 4; External review of b0c3920, item 5).

    uv run python scripts/check_opaque_leaks.py --map ../opaque/heldout/opaque-rhea-142/map.tsv.gz \\
        --input data/sources/opaque-rhea-142/generated/input/prompt-user.txt data/sources/opaque-rhea-142/census-t5.json \\
        --output data/sources/opaque-rhea-142/generated/draw-1.txt

Two kinds of file. An INPUT file (anything built from the store, or shown to a generator or an agent) fails
on any hit: exit status 1. A MODEL-OUTPUT file (what a model wrote back, and what is parsed or rendered from
it) is scanned the same way and its hits are recorded as findings: a model can recall original names from
the structure alone, which is a result, not a leak. Positional files are inputs.

The scan imports nothing from the builders. It reads either map form: the opaque store's TSV (kind I, S or T,
original, tag, opaque) or the decontamination map's JSON (iris, strings, graph_name). Each file is decoded
before it is searched. JSON is parsed and every key and value walked, and a generator prompt's
<census_and_sample> payload likewise, with compact IRIs expanded through the payload's own prefixes and the
sample's Turtle literals read into lexical form, datatype and language tag; other text is read for IRIs,
prefixed names (through the file's own PREFIX or @prefix lines, the W3C prefixes and this scanner's table) and
quoted strings. Every encoding Turtle allows is decoded: all four quoting forms (double and single quotes,
each short or tripled; the single-quoted ones wherever a Turtle text may hold a term), string escapes (\\t, \\', \\uXXXX,
\\UXXXXXXXX, ...), \\u and \\U escapes in IRIs, reserved-character escapes in local names (\\/), and
%-encodings in IRIs. A text that parses as Turtle or TriG (whole, or failing that each fenced block and each
paragraph of it) is also read with rdflib, and every literal and IRI the parser finds is decoded the same way,
whatever its position (a blank-node or collection subject, nested blank nodes, no whitespace). ``--originals-from`` adds the names of natural (unrenamed) input files to the map's, so a
name the map does not hold (a map reused for a changed input) is still searched for. Then, with no length
cutoff:

* original IRIs: every IRI found, and every prefix of it that ends before a delimiter, against the map; and
  the namespace of every original IRI as a substring of the raw text, the decoded text and the IRIs found;
* prefixed names in an original namespace (``CHEBI:15377``, ``up:Protein``), and database names;
* original strings: a data value equal to one; an original of 6 or more characters that holds a letter
  inside any decoded value or text; any other original (shorter, or digits and signs only) equal to a token
  of a data value (a word, or a span between spaces);
* identifiers: the local name of an original IRI that holds a letter and a digit (an accession such as
  P04637, CHEBI_15377), as a token of any decoded value or text. A local name of digits only (a Rhea, taxon
  or PubMed number) written bare cannot be told from a count or a kept value; it is found in identifier
  form: inside an IRI, after a database prefix, or as an original string;
* original typed values (the TSV's T rows, Rhea's integer identifiers): an integer literal with that lexical
  form.

Data values are what the data says: the lexical forms of literals (other than W3C numbers, dates and
booleans) and the label and comment values of a source record. A data value that is itself an opaque term of
the map (a placeholder, a renamed identifier) is searched only for equality: a random opaque number that
contains an original one says nothing about the original. Everything else decoded is fixed text (the
instruction template, the record's format and explanatory fields, render legends, counts), searched for
every kind of name except originals matched as tokens, which in such text are words and counts.

Every hit is reported, with the context it was found in: an original string in "data" (a data value equal to
it), "data-token" (inside, or a token of, a longer data value) or "fixed" (fixed text); every other kind of hit
in "any". An original-string hit that was checked by hand and found not to be a leak (an English word of the
fixed instruction template that is also an original string, such as "filler") may be listed in a review file
(``--reviewed``: {"reviewed": [{"needle": ..., "files": [glob, ...], "classes": ["inputs"], "contexts":
["fixed"], "reason": ...}]}). It is exempt only in a file matching one of its globs, of a listed class (inputs,
model_outputs), and only when every context it was found in is a reviewed one; it is then reported under
"reviewed" with the reason and does not fail the check. The same needle anywhere else, in another class or
context, or as another kind of hit (an identifier, an IRI), still fails. The review file is committed beside
the files it covers, so each exemption is on the record.
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

W3C = ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://www.w3.org/2000/01/rdf-schema#",
       "http://www.w3.org/2002/07/owl#", "http://www.w3.org/2001/XMLSchema#")
XSD = "http://www.w3.org/2001/XMLSchema#"
VALUE_TYPES = frozenset(XSD + t for t in (
    "decimal", "integer", "nonPositiveInteger", "negativeInteger", "long", "int", "short", "byte",
    "nonNegativeInteger", "unsignedLong", "unsignedInt", "unsignedShort", "unsignedByte", "positiveInteger",
    "float", "double", "boolean", "date", "dateTime", "dateTimeStamp", "time", "gYear", "gYearMonth", "gMonth",
    "gMonthDay", "gDay", "duration", "dayTimeDuration", "yearMonthDuration"))
INTEGER_TYPES = frozenset(XSD + t for t in (
    "integer", "nonPositiveInteger", "negativeInteger", "long", "int", "short", "byte", "nonNegativeInteger",
    "unsignedLong", "unsignedInt", "unsignedShort", "unsignedByte", "positiveInteger"))
# The lexical grammars of XSD 1.1 part 2 for the value datatypes (external reviewer 6.1 review of a929c39, item 2): only a
# well-formed lexical form of a number, date or boolean is a value; an ill-typed one ("ATP"^^xsd:integer) is data.
_YR, _MO, _DA = r"-?(?:[1-9]\d{3,}|0\d{3})", r"(?:0[1-9]|1[0-2])", r"(?:0[1-9]|[12]\d|3[01])"
_TZ = r"(?:Z|[+-](?:(?:0\d|1[0-3]):[0-5]\d|14:00))"
_TIME = r"(?:(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?|24:00:00(?:\.0+)?)"
_DUR_DATE, _DUR_TIME = r"(?:\d+Y)?(?:\d+M)?(?:\d+D)?", r"(?:T(?:\d+H)?(?:\d+M)?(?:\d+(?:\.\d+)?S)?)?"
_LEXICAL = {t: re.compile(rx) for t, rx in {
    "decimal": r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)",
    "float": r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?|[+-]?INF|NaN",
    "boolean": r"true|false|1|0",
    "date": rf"(?P<y>{_YR})-(?P<m>{_MO})-(?P<d>{_DA}){_TZ}?",
    "dateTime": rf"(?P<y>{_YR})-(?P<m>{_MO})-(?P<d>{_DA})T{_TIME}{_TZ}?",
    "dateTimeStamp": rf"(?P<y>{_YR})-(?P<m>{_MO})-(?P<d>{_DA})T{_TIME}{_TZ}",
    "time": rf"{_TIME}{_TZ}?",
    "gYear": rf"{_YR}{_TZ}?", "gYearMonth": rf"{_YR}-{_MO}{_TZ}?", "gMonth": rf"--{_MO}{_TZ}?",
    "gMonthDay": rf"--(?P<m>{_MO})-(?P<d>{_DA}){_TZ}?", "gDay": rf"---{_DA}{_TZ}?",
    "duration": rf"-?P{_DUR_DATE}{_DUR_TIME}",
    "dayTimeDuration": rf"-?P(?:\d+D)?{_DUR_TIME}",
    "yearMonthDuration": r"-?P(?:\d+Y)?(?:\d+M)?",
}.items()}
_INTEGER_RANGES = {  # (lowest, highest) value; None: unbounded
    "integer": (None, None), "nonPositiveInteger": (None, 0), "negativeInteger": (None, -1),
    "long": (-2**63, 2**63 - 1), "int": (-2**31, 2**31 - 1), "short": (-2**15, 2**15 - 1), "byte": (-2**7, 2**7 - 1),
    "nonNegativeInteger": (0, None), "unsignedLong": (0, 2**64 - 1), "unsignedInt": (0, 2**32 - 1),
    "unsignedShort": (0, 2**16 - 1), "unsignedByte": (0, 2**8 - 1), "positiveInteger": (1, None),
}
_MONTH_DAYS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def well_formed(lexical: str, datatype: str) -> bool:
    """Whether ``lexical`` is a well-formed lexical form of the XSD value datatype ``datatype`` (after the whitespace
    collapse these datatypes apply): digits, signs and separators in the grammar's places, an integer type's value
    in its range, a day that exists in its month."""
    local = datatype[len(XSD):] if datatype.startswith(XSD) else None
    s = lexical.strip(" \t\n\r")
    if local in _INTEGER_RANGES:
        if not re.fullmatch(r"[+-]?\d+", s):
            return False
        lo, hi = _INTEGER_RANGES[local]
        v = int(s)
        return (lo is None or v >= lo) and (hi is None or v <= hi)
    rx = _LEXICAL.get("float" if local == "double" else local or "")
    m = rx.fullmatch(s) if rx else None
    if m is None:
        return False
    if local in ("duration", "dayTimeDuration", "yearMonthDuration"):
        body = s.lstrip("-")[1:]
        return bool(body) and not body.endswith("T")  # at least one component; a T needs a time component
    groups = m.groupdict()
    if groups.get("d"):
        month, day = int(groups["m"]), int(groups["d"])
        if day > _MONTH_DAYS[month - 1]:
            return False
        if month == 2 and day == 29 and groups.get("y") is not None:
            y = int(groups["y"])
            return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    return True


# Prefixes an original namespace may be written with (this scanner's own table).
KNOWN_PREFIXES = {
    "CHEBI": "http://purl.obolibrary.org/obo/CHEBI_", "GO": "http://purl.obolibrary.org/obo/GO_",
    "obo": "http://purl.obolibrary.org/obo/", "rh": "http://rdf.rhea-db.org/", "up": "http://purl.uniprot.org/core/",
    "uniprot": "http://purl.uniprot.org/uniprot/", "taxon": "http://purl.uniprot.org/taxonomy/",
    "keywords": "http://purl.uniprot.org/keywords/", "location": "http://purl.uniprot.org/locations/",
    "pathway": "http://purl.uniprot.org/pathways/", "ec": "http://purl.uniprot.org/enzyme/",
    "chemrof": "https://w3id.org/chemrof/", "oboInOwl": "http://www.geneontology.org/formats/oboInOwl#",
    "sio": "http://semanticscience.org/resource/", "dcterms": "http://purl.org/dc/terms/",
    "skos": "http://www.w3.org/2004/02/skos/core#", "pubmed": "http://rdf.ncbi.nlm.nih.gov/pubmed/",
    "faldo": "http://biohackathon.org/resource/faldo#", "foaf": "http://xmlns.com/foaf/0.1/",
}
DATABASE_NAMES = re.compile(r"rhea|chebi|uniprot|swiss-?prot|trembl|pubmed|obolibrary|expasy|ncbi", re.I)
W3C_PREFIXES = {"rdf": W3C[0], "rdfs": W3C[1], "owl": W3C[2], "xsd": W3C[3]}
IRI_IN_TEXT = re.compile(r"https?://[^\s<>\"'{}|\\^`]+")
ANGLE_IRI = re.compile(r"<([^<>\s\"]+)>")
PREFIX_LINE = re.compile(r"(?:@prefix|PREFIX)\s+([A-Za-z][\w.-]*)?:\s*<([^<>\s]+)>", re.I)
# A prefixed name; its local part may hold Turtle's reserved-character escapes (\/, \#, ...) and %-encodings.
PLX = r"\\[_~.\-!$&'()*+,;=/?#@%]"
CURIE = re.compile(rf"(?<![\w/.:#-])([A-Za-z][\w.-]*)?:((?:\w|{PLX})(?:[\w.%-]|{PLX})*)")
SUFFIX = r"(\^\^(?:<[^<>\s]+>|(?:[A-Za-z][\w.-]*)?:[\w.%-]+)|@[A-Za-z][\w-]*)?"
# Turtle's four quoting forms, the long ones first so that '''x''' is not read as '' then 'x'.
LONG_DQ = r'"""((?:[^"\\]|\\.|"(?!""))*)"""'
LONG_SQ = r"'''((?:[^'\\]|\\.|'(?!''))*)'''"
SHORT_DQ = r'"((?:[^"\\\n]|\\.)*)"'
SHORT_SQ = r"'((?:[^'\\\n]|\\.)*)'"
ANY_FORM = f"(?:{LONG_DQ}|{LONG_SQ}|{SHORT_DQ}|{SHORT_SQ})"
QUOTED = re.compile(f"(?:{SHORT_DQ}){SUFFIX}")  # a double-quoted literal, read in any text
TURTLE_LITERAL = re.compile(ANY_FORM + SUFFIX)
# In a Turtle text, a literal in any quoting form where a term may stand: after a space, ( [ , ; { or an IRI's closing
# >, and before . ; , ) ] } or the end of the text (TriG's graph braces included).
TURTLE_TERM = re.compile(rf"(?:^|(?<=[\s(\[,;{{>])){ANY_FORM}{SUFFIX}(?=\s*(?:[.;,)\]}}]|\Z))", re.MULTILINE)
TURTLE_OR_QUOTED = re.compile(f"{TURTLE_TERM.pattern}|{QUOTED.pattern}", re.MULTILINE)
# A Turtle text: a prefix or base declaration, a fenced Turtle block, or a line that starts a statement: a subject
# (an IRI, a blank-node label or a prefixed name), then a predicate written as an IRI, a prefixed name or "a"; an
# IRI subject directly followed by an IRI or prefixed-name predicate (Turtle needs no space after an IRI); a
# blank-node subject, [] or a property list "[ predicate", the bracket alone on its line included; or a collection
# "( term". A line starting with a bracket and a word ("[Truncated at ...") is no statement (External review of the
# sources snapshot, item 1: "[] <urn:p> 'ATP' ." was read as prose).
_PRED = r"(?:<[^<>\s]*>|(?:[A-Za-z][\w.-]*)?:\w\S*|(?-i:a)(?=[ \t]))"
TURTLE_TEXT = re.compile(
    r"^[ \t]*(?:@prefix|@base|PREFIX|BASE)\s|```(?:turtle|ttl)"
    rf"|^[ \t]*(?:<[^<>\s]*>|_:\S+|[A-Za-z][\w.-]*:\S*)[ \t]+{_PRED}[ \t]+\S"
    r"|^[ \t]*<[^<>\s]*>[ \t]*(?:<[^<>\s]*>|(?:[A-Za-z][\w.-]*)?:\w)"
    rf"|^[ \t]*\[[ \t]*\][ \t]*{_PRED}|^[ \t]*\[[ \t]*(?:\n[ \t]*)?{_PRED}"
    r"|^[ \t]*\([ \t]*(?:['\"<\[(]|_:|[A-Za-z][\w.-]*:)",
    re.IGNORECASE | re.MULTILINE)
FENCE = re.compile(r"```[A-Za-z]*[ \t]*\n(.*?)```", re.DOTALL)
ESCAPE = re.compile(r"\\(?:u([0-9A-Fa-f]{4})|U([0-9A-Fa-f]{8})|([tbnrf\"'\\/]))")
UCHAR = re.compile(r"\\(?:u([0-9A-Fa-f]{4})|U([0-9A-Fa-f]{8}))")
ECHAR = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f", '"': '"', "'": "'", "\\": "\\", "/": "/"}
WORD = re.compile(r"[A-Za-z0-9_]+")
LONG = 6  # originals at least this long are searched as substrings; shorter ones as whole tokens
DELIMS = "/#_:.-?=&"
TEXT_KEYS = {"labels", "comments", "label", "comment"}  # a source record's data values


@dataclass
class Originals:
    iris: set[str] = field(default_factory=set)
    strings: set[str] = field(default_factory=set)
    typed: set[str] = field(default_factory=set)  # lexical forms of integer identifiers (TSV kind T)
    opaque: set[str] = field(default_factory=set)  # every opaque term (IRIs and lexical forms)
    base: str = "http://example.org/opaque/"
    graph_names: set[str] = field(default_factory=set)

    def namespaces(self) -> set[str]:
        out = set()
        for i in self.iris:
            cut = max(i.rfind("#"), i.rfind("/"))
            ns = i[: cut + 1]
            if len(ns) > len("http://x/") and not ns.startswith(W3C) and not ns.startswith(self.base):
                out.add(ns)
        return out

    def identifiers(self) -> set[str]:
        out = set()
        for i in self.iris:
            local = re.split(r"[/#]", i)[-1]
            for part in {local, *local.split("_")[1:]}:
                if len(part) >= 3 and any(ch.isdigit() for ch in part) and WORD.fullmatch(part):
                    out.add(part)
        return out


def read_map(path: Path) -> Originals:
    raw = gzip.decompress(path.read_bytes()).decode("utf-8")
    o = Originals()
    if raw.lstrip().startswith("{"):
        data = json.loads(raw)
        o.base = data.get("policy", {}).get("base", o.base)
        o.iris = set(data["iris"])
        o.opaque |= set(data["iris"].values())
        for row in data["strings"]:
            o.strings.add(row[0])
            o.opaque.add(row[-1])
        if data.get("graph_name") and data["graph_name"][0]:
            o.graph_names.add(data["graph_name"][0])
        return o
    for line in raw.splitlines():
        if line.startswith("#"):
            if line.startswith("# {"):
                o.base = json.loads(line[2:]).get("opaque_iri_base", o.base)
            continue
        kind, original, _tag, opaque = line.split("\t")
        original, opaque = _unescape(original), _unescape(opaque)
        if kind == "I":
            o.iris.add(original)
        elif kind == "T":
            o.typed.add(original)
        else:
            o.strings.add(original)
        o.opaque.add(opaque)
    return o


W3C_TERMS_PATH = Path(__file__).resolve().parents[1] / "data" / "vocab" / "w3c-terms.json"


def add_originals(o: Originals, paths: list[Path]) -> dict[str, int]:
    """Add the names of natural (unrenamed) input files to ``o``, so a name the map does not know is still searched
    for (External review 61, item 1: a map reused for a changed input). A natural file is decoded as any input: its
    IRIs, other than the terms defined in the W3C vocabularies (rule 36's list) and the W3C namespaces themselves,
    are original IRIs; its data values (literal lexical forms, labels and comments) are original strings; a
    generator prompt's census graph name is a graph name."""
    if not paths:
        return {"iris": 0, "strings": 0, "graph_names": 0}
    defined = set(json.loads(W3C_TERMS_PATH.read_text())["terms"]) | set(W3C)
    before = (len(o.iris), len(o.strings), len(o.graph_names))
    for path in paths:
        raw = path.read_bytes()
        text = (gzip.decompress(raw) if path.suffix == ".gz" else raw).decode("utf-8")
        d = decode(text)
        o.iris |= {i for i in d.iris if i not in defined and not i.startswith(o.base)}
        o.strings |= {v for v in d.data if v.strip()}
        if "<census_and_sample>" in text:
            start = text.index("<census_and_sample>") + len("<census_and_sample>")
            graph = json.loads(text[start:text.index("</census_and_sample>")]).get("census", {}).get("graph")
            if graph:
                o.graph_names.add(graph)
    return {"iris": len(o.iris) - before[0], "strings": len(o.strings) - before[1],
            "graph_names": len(o.graph_names) - before[2]}


def _unescape(v: str) -> str:
    return v.replace("\\t", "\t").replace("\\n", "\n").replace("\\r", "\r").replace("\\\\", "\\")


# --------------------------------------------------------------------------- decoding


@dataclass
class Decoded:
    iris: set[str] = field(default_factory=set)
    data: set[str] = field(default_factory=set)  # data values
    fixed: list[str] = field(default_factory=list)  # everything else decoded
    integers: set[str] = field(default_factory=set)  # lexical forms of integer literals


def _code_point(m: re.Match) -> str | None:
    cp = int(m.group(1) or m.group(2), 16)
    return chr(cp) if cp <= 0x10FFFF and not 0xD800 <= cp <= 0xDFFF else None


def _unescape(s: str) -> str:
    """A Turtle (or JSON) string with its escapes decoded: \\t \\n \\" \\' \\\\ and the rest, \\uXXXX, \\UXXXXXXXX,
    and JSON's \\/."""
    if "\\" not in s:
        return s
    return ESCAPE.sub(lambda m: ECHAR[m.group(3)] if m.group(3) else (_code_point(m) or m.group(0)), s)


def _unescape_iri(iri: str) -> str:
    """An IRI with its Turtle escapes decoded (an IRI may hold only \\uXXXX and \\UXXXXXXXX)."""
    return UCHAR.sub(lambda m: _code_point(m) or m.group(0), iri) if "\\" in iri else iri


def _iri_forms(iri: str) -> set[str]:
    """An IRI as found, with its escapes decoded, and with its %-encodings decoded: a reader recovers an original
    from any of them."""
    out = {iri, _unescape_iri(iri)}
    return out | {unquote(i) for i in out if "%" in i}


def _expand(term: str, prefixes: dict[str, str]) -> str | None:
    if term.startswith("<") and term.endswith(">"):
        return _unescape_iri(term[1:-1])
    m = re.fullmatch(r"([A-Za-z][\w.-]*)?:(\S*)", term)
    if m and (m.group(1) or "") in prefixes:
        local = re.sub(PLX, lambda x: x.group(0)[1:], m.group(2))  # Turtle's reserved-character escapes
        return _unescape_iri(prefixes[m.group(1) or ""]) + local
    return None


def _prefixes(declared: dict[str, str]) -> dict[str, str]:
    """The prefixes a text is read with: its own declarations over the W3C ones and this scanner's table, so that an
    original prefixed name is expanded even where the text does not declare its prefix."""
    return {**W3C_PREFIXES, **KNOWN_PREFIXES, **declared}


def _literal(term: str, prefixes: dict[str, str], d: Decoded) -> bool:
    """A Turtle literal written whole (any of the four quoting forms, then ^^datatype or @tag): decode it into
    ``d``."""
    m = TURTLE_LITERAL.fullmatch(term)
    if not m:
        return False
    body = next((g for g in m.groups()[:4] if g is not None), "")
    lexical, suffix = _unescape(body), m.group(5) or ""
    dt = None
    if suffix.startswith("^^"):
        dt = _expand(suffix[2:], prefixes) or suffix[2:]
        d.iris.add(dt)
    if dt in INTEGER_TYPES:
        d.integers.add(lexical)
    if dt not in VALUE_TYPES or not well_formed(lexical, dt):  # an ill-typed number or date is data
        d.data.add(lexical)
        d.iris |= {m.group(0) for m in IRI_IN_TEXT.finditer(lexical)}  # an IRI inside a literal
    return True


def _parsed_turtle(text: str, prefixes: dict[str, str]):
    """The triples of ``text`` read as Turtle (then as TriG), with the known prefixes declared first and literals kept
    in their written lexical form; None when it does not parse. Independent of the regular expressions above: any
    literal or IRI a Turtle reader finds is found, whatever its position."""
    import logging

    import rdflib

    def parses(graph, fmt: str) -> bool:
        try:
            graph.parse(data=head + text, format=fmt, publicID="urn:x-leak-scan:")
        except Exception:  # rdflib raises parser errors of many types (IndexError among them): not Turtle, and the
            return False  # regular expressions above still read the text
        return True

    head = "".join(f"@prefix {p}: <{ns}> .\n" for p, ns in sorted(prefixes.items())
                   if re.fullmatch(r"(?:[A-Za-z][\w.-]*)?", p))
    normalize, rdflib.NORMALIZE_LITERALS = rdflib.NORMALIZE_LITERALS, False
    level = logging.getLogger("rdflib").level
    logging.getLogger("rdflib").setLevel(logging.CRITICAL)
    try:
        turtle, trig = rdflib.Graph(), rdflib.Dataset()
        if parses(turtle, "turtle"):
            return list(turtle)
        return list(trig.quads()) if parses(trig, "trig") else None
    finally:
        rdflib.NORMALIZE_LITERALS = normalize
        logging.getLogger("rdflib").setLevel(level)


def _turtle_chunks(text: str) -> list[str]:
    """What is tried as Turtle: the whole text; failing that, each fenced block and each paragraph (blank-line
    separated), so a statement inside a model's prose is read too."""
    return [text, *(m.group(1) for m in FENCE.finditer(text)), *re.split(r"\n[ \t]*\n", text)]


def _segments(text: str) -> list[tuple[str, bool]]:
    """The text cut into its fenced blocks (each whole) and the blank-line paragraphs of the rest, each with whether it
    is Turtle (``TURTLE_TEXT``), separators kept so that the pieces joined are the text. A Turtle paragraph that opens
    a long quote runs on to the paragraph that closes it. Each piece is read on its own (external reviewer 6.1 review of a929c39,
    nit): a Turtle paragraph does not make the prose around it Turtle."""
    pieces: list[str] = []
    pos = 0
    for m in [*FENCE.finditer(text), None]:
        part = text[pos:m.start()] if m else text[pos:]
        cut = re.split(r"(\n[ \t]*\n)", part)
        pieces += [p for p in (cut[i] + (cut[i + 1] if i + 1 < len(cut) else "") for i in range(0, len(cut), 2)) if p]
        if m:
            pieces.append(m.group(0))
            pos = m.end()
    out: list[tuple[str, bool]] = []
    for p in pieces:
        last = out[-1] if out else None
        if last and last[1] and (last[0].count('"""') % 2 or last[0].count("'''") % 2):
            out[-1] = (last[0] + p, True)
        else:
            out.append((p, bool(TURTLE_TEXT.search(p))))
    return out


def _read_turtle(text: str, prefixes: dict[str, str], d: Decoded, whole=None) -> None:
    """Every literal and IRI of the Turtle a text holds, decoded into ``d`` as ``_literal`` would. ``whole``: the
    text's own parse, when the caller has it."""
    from rdflib import Literal as RLit
    from rdflib import URIRef

    for i, chunk in enumerate(_turtle_chunks(text)):
        if i and not chunk.strip():
            continue
        triples = whole if i == 0 and whole is not None else _parsed_turtle(chunk, prefixes)
        if triples is None:
            continue
        for t in triples:
            for term in t:
                if isinstance(term, URIRef) and not str(term).startswith("urn:x-leak-scan:"):
                    d.iris.add(str(term))
                elif isinstance(term, RLit):
                    dt = str(term.datatype) if term.datatype else None
                    if dt:
                        d.iris.add(dt)
                    lexical = str(term)
                    if dt in INTEGER_TYPES:
                        d.integers.add(lexical)
                    if dt not in VALUE_TYPES or not well_formed(lexical, dt):
                        d.data.add(lexical)
                        d.iris |= {m.group(0) for m in IRI_IN_TEXT.finditer(lexical)}
        if i == 0:
            return  # the whole text parsed: its parts add nothing


def _string(s: str, prefixes: dict[str, str], d: Decoded, data: bool) -> None:
    if s[:1] in ('"', "'") and _literal(s, prefixes, d):
        return
    if " " not in s and (iri := _expand(s, prefixes)) is not None:
        d.iris.add(iri)
        return
    if re.fullmatch(r"(?:https?|urn):\S+", s):
        d.iris.add(s)
        return
    (d.data.add if data else d.fixed.append)(s)
    for m in IRI_IN_TEXT.finditer(s):  # an IRI inside a longer value
        d.iris.add(m.group(0))


def _walk(x, prefixes: dict[str, str], d: Decoded, key: str | None = None) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            _string(k, prefixes, d, data=False)
            _walk(v, prefixes, d, k)
    elif isinstance(x, list):
        for v in x:
            _walk(v, prefixes, d, key)
    elif isinstance(x, str):
        _string(x, prefixes, d, data=key in TEXT_KEYS)


def decode(text: str) -> Decoded:
    d = Decoded()
    if "<census_and_sample>" in text:  # a generator prompt: the template, then the JSON payload
        start = text.index("<census_and_sample>") + len("<census_and_sample>")
        end = text.index("</census_and_sample>")
        payload = json.loads(text[start:end])
        d.fixed.append(text[:start] + text[end:])
        d.iris |= {m.group(0) for m in IRI_IN_TEXT.finditer(text[:start])}
        _walk(payload, _prefixes(dict(payload.get("census", {}).get("prefixes", {}))), d)
        return _finish(d, text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, (dict, list)):
        _walk(obj, _prefixes({}), d)
        return _finish(d, text)
    # Text: renders, Turtle, a model's reply. In a Turtle text a literal in any quoting form stands where a term may;
    # elsewhere only double quotes make a literal, so English apostrophes and a legend's 'literal' stay fixed text.
    prefixes = _prefixes({(m.group(1) or ""): m.group(2) for m in PREFIX_LINE.finditer(text)})
    d.iris |= {m.group(1) for m in ANGLE_IRI.finditer(text)} | {m.group(0) for m in IRI_IN_TEXT.finditer(text)}
    # A text that parses whole as Turtle is Turtle throughout; otherwise each fenced block and paragraph is read on its
    # own, so a Turtle paragraph does not turn the prose around it into literals (external reviewer 6.1 review of a929c39, nit).
    whole = _parsed_turtle(text, prefixes)
    rests = []
    for seg, turtle in ([(text, True)] if whole is not None else _segments(text)):
        pattern = TURTLE_OR_QUOTED if turtle else QUOTED  # one left-to-right pass: no literal in a literal
        for m in pattern.finditer(seg):
            _literal(m.group(0), prefixes, d)
        rests.append(pattern.sub(" ", seg))
    _read_turtle(text, prefixes, d, whole)  # and whatever a Turtle reader finds, in any position
    rest = "".join(rests)
    for m in CURIE.finditer(rest):
        if (iri := _expand(m.group(0), prefixes)) is not None:
            d.iris.add(iri)
    d.fixed.append(rest)
    return _finish(d, text)


def _finish(d: Decoded, text: str) -> Decoded:
    """Every IRI found also in its decoded forms, and the IRIs of the whole text with its \\u and \\U escapes
    decoded (an escaped IRI outside angle brackets or literals)."""
    unescaped = _unescape_iri(text)
    if unescaped != text:
        d.iris |= {m.group(1) for m in ANGLE_IRI.finditer(unescaped)} | {m.group(0) for m in IRI_IN_TEXT.finditer(unescaped)}
        d.fixed.append(unescaped)
    d.iris = set().union(*(_iri_forms(i) for i in d.iris)) if d.iris else set()
    return d


# --------------------------------------------------------------------------- scanning


def _tokens(s: str) -> set[str]:
    out = set(WORD.findall(s))
    for chunk in s.split():
        out.add(chunk)
        out.add(chunk.strip(".,;:()[]{}\"'"))
    return out


class Scanner:
    def __init__(self, o: Originals):
        self.o = o
        self.namespaces = sorted(o.namespaces())
        nss = set(self.namespaces)
        self.prefixed = sorted(p for p, ns in KNOWN_PREFIXES.items()
                               if ns in nss or any(n.startswith(ns) for n in nss))
        self.alnum_ids = {i for i in o.identifiers() if not i.isdigit()}
        self.long = [s for s in o.strings if len(s) >= LONG and any(ch.isalpha() for ch in s)]
        self.short = set(o.strings) - set(self.long)

    def scan(self, text: str) -> dict[str, list[str]]:
        return {k: v for k, v in self.scan_with_contexts(text)[0].items()}

    def scan_with_contexts(self, text: str) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
        """The hits, and per kind of hit the contexts each was found in. An original string is found in a data value
        equal to it ("data"), inside or as a token of a longer data value ("data-token"), or in fixed text ("fixed");
        every other kind of hit has the context "any" and is never exempted by a review."""
        o, d = self.o, decode(text)
        hits: dict[str, set[str]] = {k: set() for k in (
            "iris", "namespaces", "prefixed_names", "database_names", "graph_names", "strings", "identifiers",
            "typed_values")}
        for iri in d.iris:
            if iri in o.iris:
                hits["iris"].add(iri)
            for i, ch in enumerate(iri):  # an original IRI as the start of a longer one
                if ch in DELIMS and iri[:i] in o.iris:
                    hits["iris"].add(iri[:i])
        decoded = "\n".join([*d.data, *d.fixed])
        everything = "\n".join([text, decoded, *d.iris])
        searched = "\n".join([*(v for v in d.data if v not in o.opaque), *d.fixed])  # opaque terms: equality only
        hits["namespaces"] = {ns for ns in self.namespaces if ns in everything or json.dumps(ns)[1:-1] in text}
        for p in self.prefixed:
            if re.search(rf"(?<![\w/.:#-]){re.escape(p)}:\w", decoded):
                hits["prefixed_names"].add(p)
        hits["database_names"] = {m.group(0).lower() for m in DATABASE_NAMES.finditer(everything)}
        hits["graph_names"] = {g for g in o.graph_names if g in everything}
        # Strings: a data value equal to one; long originals as substrings anywhere decoded; short ones as tokens
        # of data values that are not opaque terms.
        values = [v for v in d.data if v not in o.opaque]
        data_text = "\n".join(values)
        fixed_text = "\n".join(d.fixed)

        def long_in(t: str) -> set[str]:
            grams = {t[i:i + LONG] for i in range(len(t) - LONG + 1)}
            return {s for s in self.long if s[:LONG] in grams and s[-LONG:] in grams and s in t}

        data_tokens: set[str] = set()
        inside: set[str] = set()
        for v in values:
            toks = _tokens(v)
            data_tokens |= toks
            inside |= {s for s in self.short & toks if s != v}
        fixed_tokens = set().union(*(_tokens(f) for f in d.fixed)) if d.fixed else set()
        iri_tokens = set().union(*(_tokens(re.sub(r"[/#:]", " ", i)) for i in d.iris)) if d.iris else set()
        equal = d.data & o.strings
        inside |= {s for s in long_in(data_text) if any(s in v and v != s for v in values)}
        in_fixed = long_in(fixed_text)
        long_split = long_in(searched) - equal - inside - in_fixed  # a long original spanning a data value and fixed text
        hits["strings"] = equal | inside | in_fixed | long_split
        strings = {s: [c for c, where in (("data", equal), ("data-token", inside), ("fixed", in_fixed)) if s in where]
                   for s in hits["strings"]}
        for s in long_split:
            strings[s] = ["data-token", "fixed"]
        hits["identifiers"] = self.alnum_ids & (data_tokens | fixed_tokens | iri_tokens)
        hits["typed_values"] = d.integers & o.typed
        contexts = {k: {x: ["any"] for x in v} for k, v in hits.items() if k != "strings" and v}
        if strings:
            contexts["strings"] = strings
        return {k: sorted(v) for k, v in hits.items() if v}, contexts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--input", type=Path, nargs="*", default=[], help="files whose hits fail the check")
    ap.add_argument("--output", type=Path, nargs="*", default=[], help="model-output files: hits are findings")
    ap.add_argument("--report", type=Path, help="also write the report here")
    ap.add_argument("--reviewed", type=Path, help="hits checked by hand and found not to be leaks (JSON)")
    ap.add_argument("--originals-from", type=Path, nargs="*", default=[],
                    help="natural (unrenamed) inputs whose names are searched for too, beside the map's")
    ap.add_argument("files", type=Path, nargs="*", help="input files (as --input)")
    args = ap.parse_args(argv)
    o = read_map(args.map)
    map_terms = {"iris": len(o.iris), "strings": len(o.strings), "typed": len(o.typed)}
    added = add_originals(o, args.originals_from)
    scanner = Scanner(o)
    reviews = json.loads(args.reviewed.read_text())["reviewed"] if args.reviewed else []
    for r in reviews:
        if (not r.get("files") or not r.get("classes") or not set(r["classes"]) <= {"inputs", "model_outputs"}
                or not r.get("contexts") or not set(r["contexts"]) <= {"data", "data-token", "fixed"}):
            raise SystemExit("a review must name its files, its classes (inputs, model_outputs) and its contexts "
                             f"(data, data-token, fixed): {r}")

    def review_of(path: Path, cls: str, kind: str, needle: str, found_in: list[str]):
        """A review exempts an original-string hit only, in its files and classes, when every context it was found
        in is a reviewed one."""
        if kind != "strings":
            return None
        for r in reviews:
            if r["needle"] == needle and cls in r["classes"] and set(found_in) <= set(r["contexts"]) and any(
                    fnmatch.fnmatch(str(path), g) for g in r["files"]):
                return r["reason"]
        return None
    report = {"map": str(args.map), "map_terms": map_terms,
              "originals_from": {"files": [str(p) for p in args.originals_from], "added": added},
              "reviewed_file": str(args.reviewed) if args.reviewed else None, "inputs": {}, "model_outputs": {}}
    bad = False
    for kind, paths in (("inputs", [*args.input, *args.files]), ("model_outputs", args.output)):
        for path in paths:
            raw = path.read_bytes()
            text = (gzip.decompress(raw) if path.suffix == ".gz" else raw).decode("utf-8")
            found, contexts = scanner.scan_with_contexts(text)
            exempt = {(k, x): why for k, v in found.items() for x in v
                      if (why := review_of(path, kind, k, x, contexts[k][x])) is not None}
            hits = {k: [x for x in v if (k, x) not in exempt] for k, v in found.items()}
            hits = {k: v for k, v in hits.items() if v}
            report[kind][str(path)] = {"hits": {k: v[:50] for k, v in hits.items()},
                                       "counts": {k: len(v) for k, v in hits.items()},
                                       "contexts": {k: {x: contexts[k][x] for x in v[:50]} for k, v in hits.items()},
                                       "reviewed": {x: {"reason": why, "contexts": contexts[k][x]}
                                                    for (k, x), why in sorted(exempt.items())}}
            bad |= kind == "inputs" and bool(hits)
    report["leak"] = bad
    report["model_outputs_with_findings"] = sorted(p for p, r in report["model_outputs"].items() if r["counts"])
    out = json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True)
    if args.report:
        args.report.write_text(out + "\n", encoding="utf-8")
    print(out)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
