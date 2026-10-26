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

import inspect
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
# Question tiers (two-tier profiles): plain lookups, and compositional
# questions (longer paths, grouping, negation over joins, comparisons, set
# operations, distinct counts across joins).
TIERS = ("lookup", "compositional")
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


def sparql_literal(text: str) -> str:
    """A string as a quoted SPARQL literal, with backslashes, quotes and line breaks escaped."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    return f'"{escaped}"'


def usable_label(label: str) -> bool:
    """Whether a label can be quoted in a question."""
    return bool(label) and len(label) <= MAX_LABEL_CHARS and '"' not in label and "\n" not in label


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
class Anchor:
    """An entity a question names by label, and how to check the label is unique.

    ``check_pattern`` is the body of a query that binds ``?e`` to every entity
    of this kind carrying the label; it is rendered with the question's
    parameters, where ``$<param>_lit`` holds the label as a SPARQL literal.
    Generation rejects a candidate whose label matches more or less than one
    entity, and asserts the check again for every question it writes.
    """

    kind: str
    label_params: tuple[str, ...]
    check_pattern: str
    # The parameter holding the entity's IRI: what a later calibration round compares (see Profile.prior_dev_rounds).
    param: str = ""


# Structural difficulty axes of the compositional tier (calibration round 2,
# design ruling on experiments#12): each knob ``<axis>_delta`` raises every
# variant whose axis it is by that many steps (0 or 1).
STRUCTURAL_AXES = ("path_hops", "grouping_constraints", "negation_depth", "populations")


@dataclass(frozen=True)
class StepCheck:
    """Whether a raised variant's added step matters for one question.

    ``undo``: gold-like patterns of the question with the step undone, each
    with a renaming of parameters (pattern parameter -> question parameter).
    ``mode`` "every": the gold rows must differ from every undo's rows (each
    population of the raised kind matters); "some": from at least one (a
    comparison: dropping the winner, or the deciding reference, changes the
    answer). ``fan_out``: a pattern whose rows are (near entity, answer item
    ...); the step matters when no single entity at the near end of the added
    edge accounts for every answer item. ``steps``: optionally, the name of
    the step each undo drops, in the order of ``undo`` (recorded with each
    question's outcome as ``undo_steps``).
    """

    rule: str
    undo: tuple[tuple[str, Mapping[str, str]], ...] = ()
    mode: str = "every"
    fan_out: str | None = None
    steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.steps and len(self.steps) != len(self.undo):
            raise TemplateError(f"step check {self.rule}: {len(self.steps)} step names for {len(self.undo)} undo patterns")


@dataclass(frozen=True)
class StructuralStep:
    """A compositional variant one step up along its structural axis (calibration round 2)."""

    axis: str
    # The round-1 construct the axis is read from (see the precedence rule in the manifest).
    construct: str
    round_1: int
    round_2: int
    raised: Variant
    # Edges joining the anchors and the answer, recounted from the gold text (see the manifest's rule).
    hops_round_1: int
    hops_round_2: int
    note: str = ""

    def __post_init__(self) -> None:
        if self.axis not in STRUCTURAL_AXES:
            raise TemplateError(f"unknown structural axis {self.axis!r}")
        if self.round_2 != self.round_1 + 1:
            raise TemplateError(f"{self.raised.name}: a step raises its axis by exactly one")

    def to_json(self) -> dict[str, Any]:
        return {
            "construct": self.construct,
            "axis": self.axis,
            "round_1": self.round_1,
            "round_2": self.round_2,
            "hops_round_1": self.hops_round_1,
            "hops_round_2": self.hops_round_2,
            "step_check": self.raised.step_check.rule if self.raised.step_check else None,
            "note": self.note,
        }


@dataclass(frozen=True)
class Profile:
    """What a question set needs to know about its graph beyond the templates."""

    graph_id: str
    id_prefix: str
    templates: tuple
    store_identity: Mapping[str, Any]
    # Two tiers (lookup and compositional) with ``lookup_share`` of each
    # template's questions in the lookup tier; records then carry ``tier`` and
    # ``difficulty``, and the split is made within each template and tier.
    two_tier: bool = False
    lookup_share: float = 1.0
    knobs: Mapping[str, Any] = field(default_factory=dict)
    expected_counts: str | None = None
    notes: Mapping[str, Any] = field(default_factory=dict)
    # Reject (and finally refuse) question texts that name a database (text_guard).
    forbid_database_names: bool = False
    # No answer may equal, contain or sit inside another anywhere in the set,
    # not only within one template.
    global_answer_uniqueness: bool = False
    # Structural steps by "template/variant" (compositional variants only);
    # the knobs "<axis>_delta" (0 or 1) choose the round-1 or the raised form.
    steps: Mapping[str, StructuralStep] = field(default_factory=dict)
    # Calibration rounds by the delta knobs they set, and the generator version each round's records carry.
    calibration_rounds: Mapping[int, Mapping[str, int]] = field(default_factory=dict)
    round_versions: Mapping[int, str] = field(default_factory=dict)
    # Round -> (an earlier round, the sha256 of its questions file): no test question of the round may name an anchor
    # entity that a dev question of the same variant named in the earlier round, whose dev results were seen.
    prior_dev_rounds: Mapping[int, tuple[int, str]] = field(default_factory=dict)
    # Round -> a change record for the manifest, emitted when its "applies_to" (seed, per_template) matches the run.
    revisions: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    # The Rhea audit rules of 18 Oct (ported from feat/qgen), off for the sets
    # already generated: value diversity per (variant, split); query times never
    # decide the selection (a time-out fails the run, measured times only in the
    # manifest); a gold that nests GRAPH clauses is refused; scope clauses are
    # matched on the text with anchor labels masked (readings.neutral_text);
    # one-row numeric answers are keyed by the entities they name; the manifest
    # carries the gold-row hash.
    audit_rules: bool = False
    # Store hooks for stores that must not see a GRAPH ?g scan: ``store_count``
    # replaces the per-graph triple count (and ``pinned_triples`` is what it must
    # return); ``prepare(client, seed, knobs)`` runs before sampling and returns
    # notes for the manifest.
    store_count: Callable[[Any], dict[str, int]] | None = None
    pinned_triples: Mapping[str, int] | None = None
    prepare: Callable[[Any, int, Mapping[str, Any]], dict[str, Any]] | None = None
    # Gold confirmation (design ruling (a), design log): every gold is confirmed by its variant's differently
    # written ``confirm_pattern``, run on a client with a far higher time-out, with exact agreement; a gold that
    # disagrees is replaced under the same sampling rules. ``engine_suspect(query)`` names the query shape the
    # store is known to answer wrongly; a gold or confirmation query of that shape fails the run.
    confirm_golds: bool = False
    engine_suspect: Callable[[str], bool] | None = None
    # The generator version records carry outside the calibration rounds (None: the module default).
    version: str | None = None
    # Names besides text_guard.DATABASE_NAMES that no question text may contain (sources, datasets, vocabularies).
    forbidden_names: tuple[str, ...] = ()
    # No anchor entity (any entity a text names by label) appears in two questions of one template, across both tiers.
    unique_anchors_per_template: bool = False
    # Every compositional variant declares a step check, and every compositional question carries its outcome.
    require_step_checks: bool = False
    # Optional (templates, knobs) -> templates: a graph's own variant choices made by its knobs, applied after the
    # structural steps (None: the templates as given).
    resolve_variants: Callable[..., tuple] | None = None
    # Manifest section -> the wording of its "rule" for this graph, replacing the shared wording (empty: shared wording).
    rule_texts: Mapping[str, str] = field(default_factory=dict)

    def tier_counts(self, per_template: int) -> dict[str, int]:
        if not self.two_tier:
            return {"lookup": per_template}
        lookup = int(per_template * self.lookup_share + 0.5)
        return {"lookup": lookup, "compositional": per_template - lookup}

    def calibration_round(self) -> int | None:
        """The round whose delta knobs the profile's knobs match, or None."""
        for n, deltas in sorted(self.calibration_rounds.items()):
            if all(self.knobs.get(k) == v for k, v in deltas.items()):
                return n
        return None

    def prior_dev_round(self) -> tuple[int, str] | None:
        """(earlier round, sha256 of its questions file) whose dev anchors the active round's test questions avoid."""
        return self.prior_dev_rounds.get(self.calibration_round())

    def generator_version(self, default: str) -> str:
        """The version a calibration round's records carry (else the profile's own version, else ``default``)."""
        return self.round_versions.get(self.calibration_round(), self.version or default)

    def delta(self, axis: str) -> int:
        d = self.knobs.get(f"{axis}_delta", 0)
        if d not in (0, 1):
            raise TemplateError(f"{axis}_delta must be 0 or 1, got {d!r}")
        return int(d)


def resolve_templates(templates: Iterable[type[QuestionTemplate]], profile: Profile) -> tuple[type[QuestionTemplate], ...]:
    """The templates with each stepped variant in the form its axis delta asks for.

    Delta 0 keeps the round-1 variant object itself, so a round-1 setting
    generates exactly what it did before the steps existed. A template with
    no stepped variant is returned unchanged.
    """
    out = []
    for t in templates:
        variants = []
        for v in t.variants:
            step = profile.steps.get(f"{t.id}/{v.name}")
            if step is None or v is step.raised:
                variants.append(v)
            elif v.tier != "compositional":
                raise TemplateError(f"{t.id}/{v.name}: only compositional variants take a structural step")
            else:
                variants.append(step.raised if profile.delta(step.axis) else v)
        if all(a is b for a, b in zip(variants, t.variants)):
            out.append(t)
        else:
            out.append(type(t.__name__, (t,), {"variants": tuple(variants)}))
    return tuple(out)


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


PoolRow = dict[str, Any]
# Turns the rows of a candidate query into parameter dicts. Most variants map
# one row to one dict; a few drop rows or expand one entity into several
# options (e.g. thresholds), which share the entity's key. A "_stratum" entry
# names the answer shape and is not part of the question's parameters.
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
    # The fields below default to the Rhea behaviour (one tier, Rhea prefixes
    # and graphs, the template's membership mode).
    prefixes: str | None = None
    graphs: Mapping[str, str] | None = None
    tier: str = "lookup"
    membership_mode: str | None = None
    # The gold pattern under the other reading of membership (direct <->
    # transitive), when a regex swap of rdfs:subClassOf cannot express it.
    membership_alternative: str | None = None
    anchors: tuple[Anchor, ...] = ()
    # Difficulty knobs (Profile.knobs) copied into the question's parameters.
    knob_params: tuple[str, ...] = ()
    # What makes the question hard: hops, anchors, aggregation, negation, set operation.
    difficulty: Mapping[str, Any] = field(default_factory=dict)
    # Alternative readings the question text must rule out when they change
    # the answer (harness.qgen.readings.Reading objects).
    readings: tuple = ()
    # A raised (calibration round 2) variant's check that its step matters for each question (StepCheck).
    step_check: Any = None
    # Comparison variants (round 2): the parameter groups of the compared entities in mention order, and a
    # function (params, gold rows, step outcome) -> index of the winning (or binding) entity. Generation puts
    # the winner in each mention slot as evenly as each (variant, split) allows (seeded permutation).
    slots: tuple[tuple[str, ...], ...] = ()
    winner: Any = None
    # Value diversity (profiles with audit_rules): a single-row variant's answer
    # values are capped per (variant, split); a two-valued one keeps each value
    # at most 60% of a stratum (harness.qgen.sampling.value_cap).
    single_row: bool = False
    two_valued: bool = False
    # The gold written differently (profiles with confirm_golds): the same answer by another route; never the
    # gold's own pattern counted or wrapped.
    confirm_pattern: str | None = None

    def __post_init__(self) -> None:
        unknown = set(self.prefer) - set(DISCRIMINATIONS)
        if unknown:
            raise TemplateError(f"{self.name}: unknown preferences {sorted(unknown)}")
        if self.tier not in TIERS:
            raise TemplateError(f"{self.name}: tier must be one of {TIERS}")
        if self.membership_mode is not None and self.membership_mode not in MEMBERSHIP_MODES:
            raise TemplateError(f"{self.name}: membership mode must be one of {MEMBERSHIP_MODES}")

    def _prefixes(self) -> str:
        return PREFIXES if self.prefixes is None else self.prefixes

    def _graphs(self) -> Mapping[str, str]:
        return _GRAPHS if self.graphs is None else self.graphs

    def candidates(self, rows: list[PoolRow], knobs: Mapping[str, Any] | None = None) -> list[Candidate]:
        """Candidates from the candidate query's rows; a "_stratum" entry becomes the stratum.

        ``knobs`` reach expand functions that take a second argument.
        """
        out = []
        expanded = self.expand(rows, knobs) if knobs is not None and _takes_knobs(self.expand) else self.expand(rows)
        for p in expanded:
            params = dict(p)
            stratum = str(params.pop("_stratum", ""))
            priority = int(params.pop("_priority", 0))
            out.append(Candidate(self.name, str(params[self.key_param]), params, stratum, priority))
        return out

    def gold_sparql(self, params: Mapping[str, Any]) -> str:
        return self._prefixes() + _substitute(self.gold_pattern, {**params, **self._graphs()})

    def candidate_sparql(self, knobs: Mapping[str, Any] | None = None) -> str:
        return self._prefixes() + _substitute(self.candidate_query, {**(knobs or {}), **self._graphs()})

    def probe_sparql(self, probe: EngineProbe, params: Mapping[str, Any]) -> str:
        return self._prefixes() + _substitute(probe.pattern, {**params, **self._graphs()})

    def confirm_sparql(self, params: Mapping[str, Any]) -> str | None:
        if self.confirm_pattern is None:
            return None
        return self._prefixes() + _substitute(self.confirm_pattern, {**params, **self._graphs()})

    def alternative_sparql(self, params: Mapping[str, Any]) -> str | None:
        if self.membership_alternative is None:
            return None
        return self._prefixes() + _substitute(self.membership_alternative, {**params, **self._graphs()})

    def anchor_sparql(self, anchor: Anchor, params: Mapping[str, Any]) -> str:
        literals = {f"{p}_lit": sparql_literal(str(params[p])) for p in anchor.label_params}
        body = _substitute(anchor.check_pattern, {**params, **literals, **self._graphs()})
        return self._prefixes() + "SELECT (COUNT(DISTINCT ?e) AS ?n) WHERE {\n" + body + "\n}"

    def mode(self, template: "type[QuestionTemplate]") -> str:
        return self.membership_mode or template.membership_mode

    def question(self, params: Mapping[str, Any]) -> str:
        return _substitute(self.question_pattern, params)


def _takes_knobs(fn: Callable) -> bool:
    try:
        return len(inspect.signature(fn).parameters) >= 2
    except (TypeError, ValueError):
        return False


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
    def to_json(cls, extended: bool = False) -> dict[str, Any]:
        """The template for the manifest; ``extended`` (two-tier profiles) adds tier, mode and difficulty per variant."""
        return {
            "id": cls.id,
            "description": cls.description,
            "membership_mode": cls.membership_mode,
            "variants": [
                {
                    "name": v.name,
                    "description": v.description,
                    **(
                        {"tier": v.tier, "membership_mode": v.mode(cls), "difficulty": dict(v.difficulty)}
                        if extended
                        else {}
                    ),
                    "hops": v.hops,
                    "weight": v.weight,
                    "answer_columns": [c.to_json() for c in v.answer_columns],
                    "scoring_rules": dict(v.scoring),
                    **({"readings": [r.name for r in v.readings]} if extended else {}),
                }
                for v in cls.variants
            ],
        }
