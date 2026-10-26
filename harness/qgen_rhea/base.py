"""Building blocks shared by the question templates.

A template is one of the paper's seven question types. It has one or more
variants; each variant is a concrete pattern: a candidate query that lists the
parameter values to choose from (ordered, so the pool is deterministic), a gold
SPARQL pattern, an English question pattern and the answer columns. Patterns
use ``string.Template`` placeholders (``$name``), which SPARQL does not use.

Modelling facts the patterns rely on (checked against the Rhea 142 store):

- Reactions and compounds are classes. A master (undirected) reaction is
  ``rh:N rdfs:subClassOf rh:Reaction``; its directional and bidirectional
  variants are subclasses of ``rh:DirectionalReaction`` and
  ``rh:BidirectionalReaction``. Reactions also form a hierarchy among
  themselves (``rh:10000 rdfs:subClassOf rh:12020``).
- The Rhea graph links compounds into the ChEBI hierarchy
  (``rh:Compound_1283 rdfs:subClassOf obo:CHEBI_15377``) and participants under
  compounds, so over the union graph even the direct subclasses of a ChEBI
  class include Rhea IRIs. Every ChEBI hierarchy pattern is therefore scoped to
  the ChEBI named graph.
- ChEBI stores positive charges as plain strings ("+1") and zero or negative
  charges as xsd:int; Rhea reactive parts and polymers carry string charges.
  Every numeric comparison casts with ``xsd:integer``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from string import Template
from typing import Any, Callable, ClassVar, Iterable, Mapping

GRAPH_ID = "rhea-142"
ID_PREFIX = "rhea142"
RHEA_GRAPH = "https://sparql.rhea-db.org/rhea"
CHEBI_GRAPH = "http://sparql.rhea-db.org/chebi"
RH = "http://rdf.rhea-db.org/"
CHEBI_IRI_PREFIX = "http://purl.obolibrary.org/obo/CHEBI_"

PREFIXES = (
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
    "PREFIX rh: <http://rdf.rhea-db.org/>\n"
    "PREFIX obo: <http://purl.obolibrary.org/obo/>\n"
    "PREFIX chemrof: <https://w3id.org/chemrof/>\n"
)

MEMBERSHIP_MODES = ("direct", "transitive", "n.a.")
LOOKUP = "lookup"
COMPOSITIONAL = "compositional"
TIERS = (LOOKUP, COMPOSITIONAL)
# What a compositional variant combines (ruling in the design log, comment of 2026-10-18 08:39).
FORMS = ("path", "grouping", "negation", "comparison", "intersection", "distinct_count")
# How a question names an entity: by a label (which must name exactly one entity of
# its kind in the store) or by an identifier.
ANCHOR_KINDS = ("chemical_class", "compound", "reaction", "ec_number")
# A label may be shared with an entity of the other kind (a compound entry named
# like a chemical class; reaction labels share no label with either). The question
# then says which one it means, right after the quoted label (ruling in the design log).
HOMONYM_NOTES = {
    "compound": "(meaning the compound entry, not the chemical class of the same name)",
    "chemical_class": "(meaning the chemical class, not the compound entry of the same name)",
}
HOMONYM_KIND = {"compound": "chemical_class", "chemical_class": "compound"}
# Properties recorded per question (and preferable when sampling):
#   direct_vs_transitive  swapping the membership relation (subClassOf <-> subClassOf+) changes the answer
#   graph_scope           dropping the GRAPH clauses (querying the union graph) changes the answer
DISCRIMINATIONS = ("direct_vs_transitive", "graph_scope")
COLUMN_KINDS = ("iri", "literal", "number")
COLUMN_ORDERS = ("any", "fixed")

# The scorer reads exactly these fields. Every template keeps the defaults:
# no answer is a non-integer aggregate (aggregation uses COUNT and MIN, not
# AVG), so no tolerance needs loosening.
DEFAULT_SCORING: Mapping[str, Any] = {"multiset": False, "rel_tol": 1e-6, "abs_tol": 1e-9, "column_order": "any"}

# Labels longer than this are not used in question text (a few reaction
# equations run to 700+ characters); entities with such labels are skipped.
MAX_LABEL_CHARS = 250

# Question text names no database (ruling in the design log), so the Rhea and opaque
# versions of a question differ only in identifiers and labels. The phrases are
# the source column of the opaque mapper's substitution table (harness/opaque.py
# on feat/opaque); every one of them contains a name the word pattern catches.
DATABASE_PHRASES = (
    "the ChEBI class {X}, as recorded in ChEBI",
    "the ChEBI class {X}",
    "The Rhea generic compound {X}",
    "the Rhea reaction {X}",
    "ChEBI classes, not Rhea compounds",
    "the Rhea reaction hierarchy",
    "the ChEBI subclass hierarchy",
    "the ChEBI hierarchy",
    "the ChEBI class IRI",
    "whose ChEBI class",
    "its ChEBI class",
    "per ChEBI class",
    "a ChEBI class",
    "ChEBI classes",
    "Rhea small-molecule compounds",
    "the Rhea compound IRI",
    "Rhea reactions",
    "does Rhea",
    "recorded in Rhea",
    "recorded in ChEBI",
    "that ChEBI gives",
    "the PubMed IRI",
)
# Whole words, any case: also catches accessions and CURIEs such as RHEA:10000 and CHEBI:15377.
DATABASE_NAMES = re.compile(r"\b(?:rhea|chebi|pubmed|uniprot)\b", re.IGNORECASE)

_SAFE_IRI = re.compile(r'\A[^\s<>"{}|\\^`]+\Z')


class TemplateError(ValueError):
    """A template was given parameters it cannot render."""


def scoring_rules(**overrides: Any) -> dict[str, Any]:
    rules = dict(DEFAULT_SCORING)
    unknown = set(overrides) - set(rules)
    if unknown:
        raise TemplateError(f"unknown scoring fields {sorted(unknown)}")
    rules.update(overrides)
    if rules["column_order"] not in COLUMN_ORDERS:
        raise TemplateError(f"column_order must be one of {COLUMN_ORDERS}")
    return rules


def check_iri(value: str) -> str:
    """Refuse anything that could break out of ``<...>`` in a query."""
    if not isinstance(value, str) or not value.startswith(("http://", "https://")) or not _SAFE_IRI.match(value):
        raise TemplateError(f"not a safe IRI: {value!r}")
    return value


def chebi_curie(iri: str) -> str:
    """``http://purl.obolibrary.org/obo/CHEBI_15377`` -> ``CHEBI:15377``."""
    if not iri.startswith(CHEBI_IRI_PREFIX) or not iri[len(CHEBI_IRI_PREFIX):].isdigit():
        raise TemplateError(f"not a ChEBI class IRI: {iri!r}")
    return "CHEBI:" + iri[len(CHEBI_IRI_PREFIX):]


def usable_label(label: str) -> bool:
    """Whether a label can be quoted in a question (and names no database)."""
    return (
        bool(label)
        and len(label) <= MAX_LABEL_CHARS
        and '"' not in label
        and "\\" not in label
        and "\n" not in label
        and not database_names_in(label)
    )


def database_names_in(text: str) -> list[str]:
    """Database names that a question text contains: the Rhea words above (whole words, any case,
    so also accessions and CURIEs) and the shared list of ``harness.qgen_rhea.text_guard``."""
    from harness.qgen_rhea.text_guard import database_names

    return sorted({m.group(0) for m in DATABASE_NAMES.finditer(text)} | set(database_names(text)))


@dataclass(frozen=True)
class Knobs:
    """Difficulty settings of the compositional tier, recorded in the manifest.

    A compositional variant declares the features it has (``Variant.features``);
    it is used only when each declared feature is allowed here. The numeric
    knobs filter candidates. Generation is deterministic for one setting.
    """

    # Hop counts allowed for path variants (edges from the anchor to the answer).
    path_hops: tuple[int, ...] = (4, 5, 6)
    # Grouped aggregation forms allowed: "having" and/or "order_limit".
    grouping: tuple[str, ...] = ("having", "order_limit")
    # LIMIT of the order_limit form (the top k), with a strict gap below the k-th value.
    top_k: int = 3
    # Depths allowed for negated patterns (edges inside FILTER NOT EXISTS).
    negation_depths: tuple[int, ...] = (3, 4)
    # Number of populations an intersection variant combines.
    populations: int = 2
    # Rows of a compositional list answer.
    min_rows: int = 2
    max_rows: int = 30
    # Members of the population a filter, negation, grouping or intersection acts on.
    min_population: int = 5
    # Structural steps of calibration round 2 (pre-registration change-log entry 6, applied as in the design log,
    # entry 23): 1 raises every compositional variant whose axis it is by one step (compositional_r2.STEPS),
    # 0 keeps the round-1 form. The knobs above keep their round-1 meaning: they decide which round-1
    # variants are active, and the raised form replaces an active variant.
    path_hops_delta: int = 0
    grouping_constraints_delta: int = 0
    negation_depth_delta: int = 0
    populations_delta: int = 0
    # Upper bound of the HAVING constraint the grouping step adds, one value for every grouping variant: the
    # UniProt round-2 value.
    having_max: int = 10

    def __post_init__(self) -> None:
        if not 1 <= self.min_rows <= self.max_rows <= 50:
            raise TemplateError("knobs need 1 <= min_rows <= max_rows <= 50")
        if set(self.grouping) - {"having", "order_limit"}:
            raise TemplateError(f"unknown grouping forms {sorted(set(self.grouping) - {'having', 'order_limit'})}")
        if self.top_k < 1 or self.min_population < 1 or self.populations < 2:
            raise TemplateError("top_k and min_population must be positive, populations at least 2")
        for axis in STRUCTURAL_AXES:
            if getattr(self, f"{axis}_delta") not in (0, 1):
                raise TemplateError(f"{axis}_delta must be 0 or 1")
        if self.having_max < 1:
            raise TemplateError("having_max must be positive")

    def delta(self, axis: str) -> int:
        if axis not in STRUCTURAL_AXES:
            raise TemplateError(f"unknown structural axis {axis!r}")
        return int(getattr(self, f"{axis}_delta"))

    def calibration_round(self) -> int | None:
        """1 when no axis is raised, 2 when every axis is; None for a mixed setting (not a calibration round)."""
        deltas = frozenset(self.delta(a) for a in STRUCTURAL_AXES)
        return {frozenset({0}): 1, frozenset({1}): 2}.get(deltas)

    def allows(self, features: Mapping[str, Any]) -> bool:
        checks = {
            "hops": lambda v: v in self.path_hops,
            "grouping": lambda v: v in self.grouping,
            "negation_depth": lambda v: v in self.negation_depths,
            "populations": lambda v: v == self.populations,
        }
        return all(checks[k](v) for k, v in features.items() if k in checks)

    def to_json(self) -> dict[str, Any]:
        return {
            "path_hops": list(self.path_hops),
            "grouping": list(self.grouping),
            "top_k": self.top_k,
            "negation_depths": list(self.negation_depths),
            "populations": self.populations,
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
            "min_population": self.min_population,
            **{f"{a}_delta": self.delta(a) for a in STRUCTURAL_AXES},
            "having_max": self.having_max,
        }


# Structural difficulty axes of the compositional tier (calibration round 2).
STRUCTURAL_AXES = ("path_hops", "grouping_constraints", "negation_depth", "populations")


@dataclass(frozen=True)
class StepCheck:
    """Whether a raised variant's added step matters for one question, checked on the store.

    ``undo``: gold-like patterns of the question with the step undone (each a
    ``string.Template`` over the question's parameters). ``mode`` "every": the
    gold rows differ from every undo's rows; "some": from at least one.
    ``fan_out``: a pattern whose rows are (near entity, answer item ...): the
    step matters when at least two entities at the near end of the added edge
    contribute answer items and no single one accounts for all of them.
    ``distinct``: a pattern whose rows are (entity, value) for the compared
    entities of a comparison: every entity has one value and the values are
    pairwise distinct (no ties). The compared entities are the variant's
    slots, or (a comparison without slots) the parameters in ``compared``.
    """

    rule: str
    undo: tuple[str, ...] = ()
    mode: str = "every"
    fan_out: str | None = None
    distinct: str | None = None
    compared: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("every", "some"):
            raise TemplateError(f"unknown step-check mode {self.mode!r}")


@dataclass(frozen=True)
class StructuralStep:
    """A compositional variant one step up along its structural axis (calibration round 2)."""

    axis: str
    # The round-1 construct the axis is read from (the precedence rule in compositional_r2.STEP_RULES).
    construct: str
    round_1: int
    round_2: int
    raised: Variant
    note: str = ""

    def __post_init__(self) -> None:
        if self.axis not in STRUCTURAL_AXES:
            raise TemplateError(f"unknown structural axis {self.axis!r}")
        if self.round_2 != self.round_1 + 1:
            raise TemplateError(f"{self.raised.name}: a step raises its axis by exactly one")
        if self.raised.step_check is None:
            raise TemplateError(f"{self.raised.name}: a raised variant checks that its step matters")

    def to_json(self) -> dict[str, Any]:
        return {
            "construct": self.construct,
            "axis": self.axis,
            "round_1": self.round_1,
            "round_2": self.round_2,
            "raised_variant": self.raised.name,
            "step_check": self.raised.step_check.rule,
            "note": self.note,
        }


@dataclass(frozen=True)
class Anchor:
    """An entity the question names: by its label (``label_param``) or by an identifier."""

    kind: str
    entity_param: str = "entity"
    label_param: str | None = "label"
    # For identifier anchors (EC numbers): the parameter holding the identifier.
    id_param: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ANCHOR_KINDS:
            raise TemplateError(f"anchor kind must be one of {ANCHOR_KINDS}, got {self.kind!r}")
        if (self.label_param is None) == (self.id_param is None):
            raise TemplateError("an anchor is named either by a label or by an identifier")

    @property
    def homonym_param(self) -> str | None:
        """Parameter set (True) when the label also names an entity of the other kind."""
        return f"{self.label_param}_homonym" if self.label_param else None

    def describe(self, params: Mapping[str, Any]) -> dict[str, Any]:
        named_by = "label" if self.label_param else "identifier"
        return {
            "kind": self.kind,
            "entity": params[self.entity_param],
            "named_by": named_by,
            "name": params[self.label_param or self.id_param],
            "homonym": bool(self.homonym_param and params.get(self.homonym_param)),
        }


@dataclass(frozen=True)
class EngineProbe:
    """A natural query for a question that an engine quirk can make go wrong.

    After a question is chosen, its probe (rendered with the question's
    parameters) runs on the store. If the probe's rows differ from the gold
    rows, the question is tagged ``engine_sensitive`` with ``reason``, so
    results can be reported for that stratum separately.
    """

    reason: str
    description: str
    pattern: str


@dataclass(frozen=True)
class AnswerColumn:
    name: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in COLUMN_KINDS:
            raise TemplateError(f"column kind must be one of {COLUMN_KINDS}, got {self.kind!r}")

    def to_json(self) -> dict[str, str]:
        return {"name": self.name, "kind": self.kind}


@dataclass(frozen=True)
class Candidate:
    """One parameter choice for a variant.

    ``key`` names the entity the question is about; a template uses each key
    at most once, so two questions never ask about the same entity.
    ``stratum`` is an answer shape known from the candidate query (e.g.
    "single" or "multi" for one or several values); sampling alternates
    between strata so a variant's answers do not all have one shape.
    """

    variant: str
    key: str
    params: dict[str, Any] = field(default_factory=dict)
    stratum: str = ""
    # A hint from the candidate query that a preferred property (see
    # Variant.prefer) is likely; higher-priority candidates are drawn first.
    priority: int = 0
    # Why the candidate may not be picked (e.g. "round_1_bound"), or "" when it may. Such a candidate stays in
    # the pool and is rejected when drawn, so the seeded draw order of every other candidate stays.
    excluded: str = ""


PoolRow = dict[str, Any]
# Turns the rows of a candidate query into parameter dicts. Most variants map
# one row to one dict; a few drop rows or expand one entity into several
# options (e.g. thresholds), which share the entity's key. A "_stratum" entry
# names the answer shape, a "_priority" entry the draw priority and an "_excluded"
# entry why the option may not be picked; none is part of the question's parameters.
ExpandFn = Callable[[list[PoolRow]], "Iterable[dict[str, Any]]"]

# Graph IRIs every pattern can use as $RHEA and $CHEBI.
_GRAPHS = {"RHEA": f"<{RHEA_GRAPH}>", "CHEBI": f"<{CHEBI_GRAPH}>"}


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    candidate_query: str
    gold_pattern: str
    question_pattern: str
    answer_columns: tuple[AnswerColumn, ...]
    key_param: str
    expand: ExpandFn
    hops: int | None = None
    scoring: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_SCORING))
    # Relative share of the template's questions (others make up a shortfall).
    weight: int = 1
    # Natural queries that an engine quirk can break (see EngineProbe).
    engine_probes: tuple[EngineProbe, ...] = ()
    # Properties to prefer when sampling, from DISCRIMINATIONS: candidates that
    # have them all are taken first; the others only when the pool runs out.
    prefer: tuple[str, ...] = ()
    tier: str = LOOKUP
    # The entities the question names, checked against the store after sampling.
    anchors: tuple[Anchor, ...] = ()
    # Membership mode of this variant's gold; None: the template's.
    membership_mode: str | None = None
    # Compositional forms (FORMS) and knob features (hops, grouping,
    # negation_depth, populations) of the variant; see Knobs.allows.
    forms: tuple[str, ...] = ()
    features: Mapping[str, Any] = field(default_factory=dict)
    # The answer is one row (an aggregate, a comparison or a single literal), not a list. Such a
    # variant's answer values are kept diverse (sampling.value_cap per variant and split).
    single_row: bool = False
    # A single-row variant with only two possible answers: exempt from the 25% cap, but each
    # value at most 60% of a stratum.
    two_valued: bool = False
    # SELECT (COUNT(DISTINCT ...) AS ?n): the population the filter, negation,
    # grouping or intersection acts on, before it. It must reach
    # Knobs.min_population and, for a list answer, lose at least one member
    # (the answer's first column has fewer distinct values than ?n).
    population_pattern: str | None = None
    # Alternative readings the question text must rule out when they change
    # the answer (harness.qgen_rhea.readings.Reading objects; templates.py attaches
    # the Rhea readings to every variant).
    readings: tuple = ()
    # For each side variable of the gold (a participation: the named entity's, or
    # the answer or counted compounds'), the phrase of the question that states
    # its scope, as a regular expression on readings.neutral_text (labels as
    # «param» tokens). Every side variable needs one (readings.rhea_readings).
    side_clauses: Mapping[str, str] = field(default_factory=dict)
    # Side variables of named entities whose relative sides a reader could constrain (all on the same side,
    # or not all on the same side), and the phrase that states they are independent (readings.joint_side).
    joint_sides: tuple[str, ...] = ()
    joint_clause: str | None = None
    # A raised (calibration round 2) variant's check that its step matters for each question (StepCheck).
    step_check: Any = None
    # Comparison variants of round 2: the parameter groups of the compared entities in mention order (each
    # group: the entity's IRI parameter, then its label parameter), and which entity wins: "max" or "min" of
    # the values the step check's ``distinct`` pattern gives (e.g. the binding reference: "min").
    # Generation puts the winner in each mention slot as evenly as each (variant, split) allows; the gold
    # does not depend on the mention order.
    slots: tuple[tuple[str, ...], ...] = ()
    winner: str | None = None
    # Sentences the text gets when a parameter flag is set (flag -> sentence, appended in this order): generation
    # sets a reading's ``note_param`` on a question whose answer that reading changes (readings.Reading.note_param).
    notes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.slots == ()) != (self.winner is None) or self.winner not in (None, "max", "min"):
            raise TemplateError(f"{self.name}: a comparison variant declares its slots and its winner (max or min)")
        unknown = set(self.prefer) - set(DISCRIMINATIONS)
        if unknown:
            raise TemplateError(f"{self.name}: unknown preferences {sorted(unknown)}")
        if self.tier not in TIERS:
            raise TemplateError(f"{self.name}: tier must be one of {TIERS}")
        if self.membership_mode is not None and self.membership_mode not in MEMBERSHIP_MODES:
            raise TemplateError(f"{self.name}: membership mode must be one of {MEMBERSHIP_MODES}")
        if set(self.forms) - set(FORMS):
            raise TemplateError(f"{self.name}: unknown forms {sorted(set(self.forms) - set(FORMS))}")
        if self.tier == COMPOSITIONAL and not self.forms:
            raise TemplateError(f"{self.name}: a compositional variant names its forms")

    def candidates(self, rows: list[PoolRow], knobs: "Knobs | None" = None) -> list[Candidate]:
        """Candidates from the candidate query's rows; a "_stratum" entry becomes the stratum.

        Compositional expand functions also take the knobs (default settings when none are given).
        """
        out = []
        # An aggregate over no rows can come back as one row of unbound values (rdflib does this).
        rows = [r for r in rows if any(v is not None for v in r.values())]
        expanded = self.expand(rows, knobs or Knobs()) if self.tier == COMPOSITIONAL else self.expand(rows)
        for p in expanded:
            params = dict(p)
            stratum = str(params.pop("_stratum", ""))
            priority = int(params.pop("_priority", 0))
            excluded = str(params.pop("_excluded", ""))
            out.append(Candidate(self.name, str(params[self.key_param]), params, stratum, priority, excluded))
        return out

    def _prefixes(self) -> str:
        return PREFIXES

    def _graphs(self) -> Mapping[str, str]:
        return _GRAPHS

    def gold_sparql(self, params: Mapping[str, Any]) -> str:
        return PREFIXES + _substitute(self.gold_pattern, {**params, **_GRAPHS})

    def candidate_sparql(self) -> str:
        return PREFIXES + _substitute(self.candidate_query, _GRAPHS)

    def probe_sparql(self, probe: EngineProbe, params: Mapping[str, Any]) -> str:
        return PREFIXES + _substitute(probe.pattern, {**params, **_GRAPHS})

    def population_sparql(self, params: Mapping[str, Any]) -> str | None:
        if self.population_pattern is None:
            return None
        return PREFIXES + _substitute(self.population_pattern, {**params, **_GRAPHS})

    def anchor_list(self, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [a.describe(params) for a in self.anchors]

    def question(self, params: Mapping[str, Any]) -> str:
        """The question text; a label shared with an entity of the other kind gets its note after the first quote."""
        text = _substitute(self.question_pattern, params)
        for anchor in self.anchors:
            if anchor.homonym_param and params.get(anchor.homonym_param):
                quoted = f'"{params[anchor.label_param]}"'
                if quoted not in text or anchor.kind not in HOMONYM_NOTES:
                    raise TemplateError(f"{self.name}: cannot place the note for {quoted}")
                text = text.replace(quoted, f"{quoted} {HOMONYM_NOTES[anchor.kind]}", 1)
        for flag, sentence in self.notes.items():
            if params.get(flag):
                text = f"{text} {sentence}"
        return text


def _substitute(pattern: str, params: Mapping[str, Any]) -> str:
    try:
        return Template(pattern).substitute(params)
    except (KeyError, ValueError) as exc:
        raise TemplateError(f"cannot render pattern: {exc}") from exc


class QuestionTemplate:
    """One of the seven question types. Subclasses set the class attributes."""

    id: ClassVar[str]
    description: ClassVar[str]
    # Whether the gold queries test class membership with direct
    # rdfs:subClassOf, the transitive path rdfs:subClassOf+, or not at all.
    membership_mode: ClassVar[str]
    variants: ClassVar[tuple[Variant, ...]]

    @classmethod
    def variant(cls, name: str) -> Variant:
        for v in cls.variants:
            if v.name == name:
                return v
        raise KeyError(f"{cls.id} has no variant {name!r}")

    @classmethod
    def mode_of(cls, variant: Variant) -> str:
        return variant.membership_mode or cls.membership_mode

    @classmethod
    def tier_variants(cls, tier: str, knobs: "Knobs | None" = None) -> tuple[Variant, ...]:
        """The variants of one tier; with knobs, only the compositional ones they allow.

        A raised variant (it has a step check) was allowed in its round-1 form
        (compositional_r2.resolve_templates), so the knobs do not filter it again.
        """
        out = tuple(v for v in cls.variants if v.tier == tier)
        if knobs is not None and tier == COMPOSITIONAL:
            out = tuple(v for v in out if v.step_check is not None or knobs.allows(v.features))
        return out

    @classmethod
    def to_json(cls) -> dict[str, Any]:
        return {
            "id": cls.id,
            "description": cls.description,
            "membership_mode": cls.membership_mode,
            "variants": [
                {
                    "name": v.name,
                    "tier": v.tier,
                    "description": v.description,
                    "membership_mode": cls.mode_of(v),
                    "hops": v.hops,
                    "forms": list(v.forms),
                    "features": dict(v.features),
                    "weight": v.weight,
                    "answer_columns": [c.to_json() for c in v.answer_columns],
                    "scoring_rules": dict(v.scoring),
                }
                for v in cls.variants
            ],
        }
