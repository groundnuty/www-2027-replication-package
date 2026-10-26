"""Registered predictions: for each fault class, which check forms can detect it in principle.

Written before any checker runs on the bank (issue E1.3, criterion 6; proposal §7.6, "the check
forms able to detect it"). Check forms and rule numbers are those of check contract v2 with
amendment 2.1; each row names one fixture in ``tests/fixtures/faults/prediction-fixtures.json``
whose expected conformance labels are written by hand. Number traceability (rule 21) is never in
this table: it is ontology-independent and never attributed to a source (rule 8), and on a bank
item the answer's numbers are in the received rows by construction.
"""

from __future__ import annotations

FAULT_CLASSES = {
    "vocabulary_absent": "invented identifiers (operators invented_*)",
    "type_wrong": "every class, predicate and constant swap (operators swap_*), at d = 2 and d = 4",
    "stale_in_data": "stale terms still in the store: deprecated or obsolete identifiers (operators stale_*); exploratory",
    "meaning": ("the aggregation operators (agg_*); and, as a descriptive stratum outside C2 and outside C4's W, the "
                "type-consistent constant swap (operator type_consistent_constant; design ruling (d) in the design log, "
                "change-log entry 63)"),
}

# fault class, slot, check form, rules, contract fixture, our fixture, which sources can detect it, prediction
REGISTERED_PREDICTIONS: list[dict] = [
    {"fault_class": "vocabulary_absent", "slot": "predicate", "check_form": "vocabulary existence (static)",
     "rules": "25", "contract_fixture": "F2", "fixture": "VA-predicate-vocabulary",
     "sources": "every source (each lists its predicates)",
     "prediction": "invented predicates: recall near 1 for every source"},
    {"fault_class": "vocabulary_absent", "slot": "class", "check_form": "vocabulary existence (static)",
     "rules": "25, 35", "contract_fixture": "F2", "fixture": "VA-class-vocabulary",
     "sources": "every source; shared-hierarchy classes count for all alike",
     "prediction": "invented classes: recall near 1 for every source"},
    {"fault_class": "vocabulary_absent", "slot": "class", "check_form": "census single-pattern emptiness (static)",
     "rules": "20", "contract_fixture": "F23", "fixture": "VA-class-census-empty",
     "sources": "census", "prediction": "the census also flags a class with direct count 0"},
    {"fault_class": "vocabulary_absent", "slot": "constant", "check_form": "none in principle (vocabulary does not apply to data nodes; constant typing is unknown on an untyped node)",
     "rules": "5, 32", "contract_fixture": "F7", "fixture": "VA-constant-untyped",
     "sources": "none", "prediction": "invented constants in a subject or object position are not detected by any ontology check"},
    {"fault_class": "vocabulary_absent", "slot": "constant in a class position", "check_form": "vocabulary existence (static), an IRI absent from the store",
     "rules": "25, 37", "contract_fixture": "F2", "fixture": "VA-constant-class-position",
     "sources": "every source", "prediction": "invented constants in a class position: recall near 1 (rule 37, change-log entry 25)"},
    {"fault_class": "type_wrong", "slot": "predicate", "check_form": "static domain or range against a class the query asserts (no-common-subclass heuristic)",
     "rules": "2, 18, 23", "contract_fixture": "F5", "fixture": "TW-predicate-explicit-type",
     "sources": "sources with domain or range axioms (on Rhea: census only; curated declares none)",
     "prediction": "detected where the source declares the domain or range of the swapped-in predicate"},
    {"fault_class": "type_wrong", "slot": "predicate", "check_form": "fixed-length path: range of one step against the domain of the next",
     "rules": "19", "contract_fixture": "F26", "fixture": "TW-path",
     "sources": "sources with domain and range axioms", "prediction": "detected when both neighbouring steps carry axioms"},
    {"fault_class": "type_wrong", "slot": "constant", "check_form": "constant typing (static, constant in a typed position)",
     "rules": "5", "contract_fixture": "F6", "fixture": "TW-constant-domain",
     "sources": "sources with domain or range axioms", "prediction": "a swapped constant whose classes fall outside the position's domain or range is flagged"},
    {"fault_class": "type_wrong", "slot": "class", "check_form": "static disjointness",
     "rules": "16", "contract_fixture": "F3", "fixture": "TW-class-disjoint",
     "sources": "sources with disjointness axioms (none on Rhea; UniProt's core and generated drafts may have some)",
     "prediction": "detected only where disjointness is declared"},
    {"fault_class": "type_wrong", "slot": "class", "check_form": "static domain against a swapped class (no-common-subclass heuristic)",
     "rules": "18, 23", "contract_fixture": "F5", "fixture": "TW-class-domain",
     "sources": "sources with domain or range axioms", "prediction": "detected where the member's predicates carry a domain or range"},
    {"fault_class": "type_wrong", "slot": "predicate", "check_form": "census join support (low-support heuristic)",
     "rules": "20", "contract_fixture": "F19a", "fixture": "TW-census-join",
     "sources": "census", "prediction": "the census flags joins no listed class supports at both positions; a necessary condition that also over-flags (F19a) and under-flags (F19c)"},
    {"fault_class": "type_wrong", "slot": "constant", "check_form": "footprint domain or range (closed-world reading, annotated open-world accepts)",
     "rules": "10, 16", "contract_fixture": "F9", "fixture": "TW-footprint-closed-world",
     "sources": "sources with domain or range axioms, on answers that return rows",
     "prediction": "answer-changing swaps that still return rows can be caught on the footprint as well"},
    {"fault_class": "stale_in_data", "slot": "constant", "check_form": "none expected (the obsolete entity keeps its type, so constant typing passes)",
     "rules": "5, 32", "contract_fixture": "F6 (passing case)", "fixture": "SD-obsolete-entity",
     "sources": "none", "prediction": "low recall for every source (registered by the design ruling; exploratory)"},
    {"fault_class": "stale_in_data", "slot": "class", "check_form": "vocabulary existence (static), where the deprecated term is outside the source and the shared hierarchy",
     "rules": "25, 35", "contract_fixture": "F2", "fixture": "SD-deprecated-class",
     "sources": "every source that omits deprecated terms",
     "prediction": "low recall for every source (design ruling); note: on Rhea the deprecated ChEBI classes are in no source and not in the shared hierarchy, so this fixture's label is violates"},
    {"fault_class": "meaning", "slot": "aggregation", "check_form": "none in principle",
     "rules": "8, 21", "contract_fixture": "F11", "fixture": "EM-aggregation",
     "sources": "none", "prediction": "wrong aggregation is not detected by any ontology check; recall at the false-flag rate"},
    {"fault_class": "meaning", "operator": "type_consistent_constant", "stratum": "type_consistent (descriptive)",
     "slot": "constant", "check_form": "none in principle (the replacement is another member of the gold constant's own "
     "class, so constant typing, domain, range and the footprint see a member of the class the gold has)",
     "rules": "5, 10, 16", "contract_fixture": "F8 (passing case)", "fixture": "EM-type-consistent-constant",
     "sources": "none",
     "prediction": "about chance for every source: a type check cannot see it, so recall is about the false-flag rate "
                   "(descriptive; outside C2 and outside C4's W)"},
    {"fault_class": "type_wrong", "slot": "predicate, class or constant", "check_form": "none in principle for that source",
     "rules": "-", "contract_fixture": "F22b", "fixture": "TW-axiom-consistent-swap",
     "sources": "none", "prediction": "a swap (type_wrong by operator) that meets every axiom of a source cannot be detected by that source"},
]
