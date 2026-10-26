"""The one machine-readable form of an ontology source.

Every source (curated, data-derived census, LLM-generated, and later the
UniProt and a15 ones) is converted into an ``OntologySource``. The renderer
turns it into prompt text and the check compiler turns it into checks, so all
sources reach both on equal terms (proposal §7.3). Only the OWL fragment the
study freezes is represented (§7.2): named classes and their subclass
hierarchy, domain and range as a named class or a union of named classes,
disjointness between named classes, existential restrictions, labels and
comments. Anything else a source states is counted in ``unsupported`` and never
translated.

Census-only facts (observed patterns with their support) live in
``patterns``; other sources leave it empty.

Fields added for the S0.1 rulings 13 and 14 (optional; older JSON without them
loads with their defaults):

* ``PropertyInfo.range_values``: an enumerated data range (``owl:oneOf`` over
  literals) as a list of ``{"lexical": ..., "datatype": ...}``. A value
  satisfies it when it has a listed value's datatype (integer subtypes count as
  integer) and equals that value, compared by value as the scorer does.
* ``PropertyInfo.unsupported_domain`` / ``unsupported_range``: True when a
  conjunct of the property's domain (range) lies outside the fragment (an
  intersection with a restriction or a qualified cardinality) and was left out.
  The supported conjuncts are still listed in ``domain`` / ``range``; the check
  at that position is unknown (reason: unsupported construct) unless a
  supported conjunct flags.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

SCHEMA_VERSION = "ontology-source-v1"
CENSUS_FIELDS = ("type_support", "class_support", "untyped_fillers", "position_support")
# Per-class figures of an earlier census form (08a0296), dropped on reading.
_LEGACY_CLASS_FIELDS = ("member_support", "subclass_support", "type_support")

SourceKind = Literal["curated", "census", "generated"]
# How the data links a node to its classes. Rhea models reactions and
# compounds as classes, so membership is rdfs:subClassOf there.
Membership = Literal["rdf:type", "rdfs:subClassOf"]


@dataclass(frozen=True)
class ClassExpr:
    """A named class (one member) or a union of named classes (several).

    A type falls under the expression if it falls under any member.
    """

    members: tuple[str, ...]

    @staticmethod
    def named(iri: str) -> ClassExpr:
        return ClassExpr((iri,))

    @property
    def is_union(self) -> bool:
        return len(self.members) > 1


@dataclass
class ClassInfo:
    iri: str
    parents: list[str] = field(default_factory=list)  # direct named superclasses
    labels: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)


@dataclass
class PropertyInfo:
    iri: str
    kind: Literal["object", "datatype", "annotation", "unknown"] = "unknown"
    parents: list[str] = field(default_factory=list)  # direct super-properties
    domain: list[ClassExpr] = field(default_factory=list)  # several entries are intersected, as in RDFS
    range: list[ClassExpr] = field(default_factory=list)
    datatype_range: list[str] = field(default_factory=list)  # e.g. xsd:integer, for datatype properties
    labels: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    range_values: list[dict[str, str]] = field(default_factory=list)  # owl:oneOf data range: {"lexical", "datatype"}
    unsupported_domain: bool = False  # a domain conjunct outside the fragment was left out
    unsupported_range: bool = False  # a range conjunct outside the fragment was left out


@dataclass(frozen=True)
class Existential:
    """``cls rdfs:subClassOf [owl:onProperty prop; owl:someValuesFrom filler]``."""

    cls: str
    prop: str
    filler: str


@dataclass(frozen=True)
class Pattern:
    """A census pattern: a predicate seen between a subject class and an object
    class (or a literal datatype), with its support (number of triples).

    ``reading`` says how the classes were read. "membership": both ends are
    classes under the source's membership reading, counted up the subclass
    chain. "literal": the triple pattern as asserted, ``?x predicate
    object_class``, with ``subject_class`` = owl:Thing (any subject); the
    census records these for the membership predicate that is not the
    graph's convention (rdf:type on Rhea), the direct counts rule 20 needs
    beside ``type_support``. Old JSON without the field reads as "membership".
    """

    subject_class: str
    predicate: str
    object_class: str | None  # None when the object is a literal
    object_datatype: str | None
    support: int
    reading: Literal["membership", "literal"] = "membership"


@dataclass
class OntologySource:
    name: str  # e.g. "curated", "census-t5", "generated-1"
    kind: SourceKind
    graph: str  # e.g. "rhea-142"
    membership: Membership
    classes: dict[str, ClassInfo] = field(default_factory=dict)
    properties: dict[str, PropertyInfo] = field(default_factory=dict)
    disjoint: list[tuple[str, str]] = field(default_factory=list)  # unordered pairs, stored sorted
    existentials: list[Existential] = field(default_factory=list)
    patterns: list[Pattern] = field(default_factory=list)
    support_threshold: int | None = None  # census only
    unsupported: dict[str, int] = field(default_factory=dict)  # construct name -> count left out
    provenance: dict = field(default_factory=dict)  # inputs, model, parameters, hashes
    # Census only (check contract v2, rule 20; names fixed in the design log). None in every
    # other source, and then left out of the JSON form.
    type_support: dict[str, int] | None = None  # class -> direct count: subjects of the literal membership triple
    #   (rdf:type; on Rhea, whose convention is rdfs:subClassOf, the literal rdfs:subClassOf triple)
    class_support: dict[str, int] | None = None  # class -> membership count, up the observed subclass chain
    untyped_fillers: dict[str, dict[str, int]] | None = None  # predicate -> {"subject": n, "object": n}
    position_support: dict[str, dict[str, dict[str, int]]] | None = None
    #   predicate -> {"subject": {class: n}, "object": {class or datatype: n}}: the classes observed at each position
    # The classes the source declares (rule 32 (a)) where they differ from ``classes``: a generated ontology's
    # explicit owl:Class / rdfs:Class typings, while ``classes`` holds every IRI in a class position. None (left out of
    # the JSON form) means every listed class is declared, as for the census and the curated source.
    declared_classes: list[str] | None = None
    # Terms the source describes (a label or comment) without declaring them a class or property or using them in a
    # class position (design ruling in the design log Oct 23:02): IRI -> {"labels": [...], "comments": [...]}, rendered
    # in the labels layer only. None (left out of the JSON form) when there are none.
    described: dict[str, dict[str, list[str]]] | None = None
    # Census only: positions whose literal datatypes could not be counted (design ruling in the design log Oct 05:24),
    # predicate -> {"object": "unknown"}, recorded where the position's literal objects reach the threshold (below it
    # no datatype can, so nothing is unknown). Such a position lists no datatype and its property no datatype range;
    # the checker takes it as no constraint. None (left out of the JSON form) when there are none.
    position_datatype: dict[str, dict[str, str]] | None = None
    # Curated sources: the part of its input (published file or store graph) each axiom comes from (design ruling on
    # design log, 20 Oct 08:04; harness.sources.origin), which the checker copies into each flag's evidence. None (left
    # out of the JSON form) where it was not recorded, as in every source built before it.
    origin: dict | None = None

    def to_json(self) -> dict:
        out = asdict(self)
        out["schema_version"] = SCHEMA_VERSION
        out["disjoint"] = sorted(sorted(p) for p in self.disjoint)
        for f in (*CENSUS_FIELDS, "declared_classes", "described", "position_datatype", "origin"):  # census-only
            # fields, declarations, descriptions, unknown datatypes and origins are left out when unset
            if out.get(f) is None:
                out.pop(f, None)
        return out

    @staticmethod
    def from_json(obj: dict) -> OntologySource:
        if obj.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {obj.get('schema_version')!r}")

        def exprs(items):
            return [ClassExpr(tuple(e["members"])) for e in items]

        return OntologySource(
            name=obj["name"],
            kind=obj["kind"],
            graph=obj["graph"],
            membership=obj["membership"],
            classes={
                k: ClassInfo(**{f: x for f, x in v.items() if f not in _LEGACY_CLASS_FIELDS})
                for k, v in obj["classes"].items()
            },
            properties={
                k: PropertyInfo(**{**v, "domain": exprs(v["domain"]), "range": exprs(v["range"])})
                for k, v in obj["properties"].items()
            },
            disjoint=[tuple(p) for p in obj["disjoint"]],
            existentials=[Existential(**e) for e in obj["existentials"]],
            patterns=[Pattern(**p) for p in obj["patterns"]],
            support_threshold=obj.get("support_threshold"),
            unsupported=obj.get("unsupported", {}),
            provenance=obj.get("provenance", {}),
            type_support=obj.get("type_support"),
            class_support=obj.get("class_support"),
            untyped_fillers=obj.get("untyped_fillers"),
            position_support=obj.get("position_support"),
            declared_classes=obj.get("declared_classes"),
            described=obj.get("described"),
            position_datatype=obj.get("position_datatype"),
            origin=obj.get("origin"),
        )

    def digest(self) -> str:
        """sha256 of the canonical JSON form; recorded in every run that uses the source."""
        text = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode()).hexdigest()
