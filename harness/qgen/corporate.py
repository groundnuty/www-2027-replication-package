"""Question templates for the Text2SPARQL 2026 corporate graph (text2sparql-ck26).

Two tiers per template, as for Rhea and UniProt: about a third **lookup** questions and two thirds
**compositional** ones (intersections and unions of populations, set differences, comparisons of entities,
grouping with HAVING or ORDER BY / LIMIT, distinct counts and sums across joins, negation and universal
quantification over joins, paths of two to four edges, explicit signed differences).

Six templates mirror the Rhea and UniProt kinds. The seventh kind there, a join across named graphs, has no
counterpart: the store holds one data graph (``prod-instances``) and the vocabulary graph (``prod-vocab``), and
questions come from the data, never from the vocabulary's axioms.

How anchors are chosen: from instances and links in the data and their counts (suppliers by the products they
supply, countries by the suppliers whose address names them, product managers by their products, categories,
services and bills of materials by their members). Nothing reads class or property axioms (domain, range,
subclass) from the vocabulary graph. Entities are named by a name that is unique among entities of their kind
(checked per question): a hardware product, supplier, product category, service, department or bill of materials
by its ``pv:name``, a person by ``pv:name`` among employees and managers, a country by the country string in
supplier addresses. Products whose name is shared by another product are never named.

Data facts the patterns rely on (checked on the store, recorded in the manifest notes):

- The gold queries use no GRAPH clause; they run over the union default graph, like the challenge's own gold
  queries. Every subject is typed with an instance class (pv:Hardware, pv:Supplier, ...), so the vocabulary
  graph's labels and types never reach an answer.
- Managers are typed only ``pv:Manager``; ``pv:Manager rdfs:subClassOf pv:Employee`` is an axiom of the
  vocabulary, not a type in the data. Questions about employees say that managers are not counted.
- Services are products too: they have product managers, prices, a responsible department and (once) a category.
  Questions say "hardware products"; the services reading is guarded where it changes an answer.
- 23 suppliers give a country in their address but have no ``pv:country`` link (Spain 9, Ukraine 7, ...).
  Countries are read from the address string, and the question says so.
- Compatibility (``pv:compatibleProduct``) is symmetric in the data, so a two-step neighbourhood reaches back to
  its anchor.
- BOM quantities are ``xsd:string``; sums cast them with ``xsd:integer`` (QLever's SUM over the strings is
  unbound). Two BOM parts carry two quantities each; their bills of materials never get a quantity question.
- Prices (``pv:amount``) and reliability indices are decimals that QLever computes with in double precision;
  sums and averages differ from exact decimal arithmetic only in the last digits, inside the scorer's numeric
  tolerance. No gold rounds (the held-out question 32's issue).
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from collections.abc import Iterable
from decimal import Decimal
from itertools import combinations
from typing import Any

from harness.qgen.base import (
    Anchor,
    AnswerColumn,
    EngineProbe,
    PoolRow,
    Profile,
    QuestionTemplate,
    StepCheck,
    TemplateError,
    Variant,
    check_iri,
    sparql_literal,
    usable_label,
)
from harness.qgen.readings import Reading

GRAPH_ID = "text2sparql-ck26"
# Not "ck26": the held-out challenge questions are ck26-NN.
ID_PREFIX = "corporate"
GENERATOR_VERSION = "0.1.0"

PREFIXES = (
    "PREFIX pv: <http://ld.company.org/prod-vocab/>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
)
GRAPHS: dict[str, str] = {}

KNOBS: dict[str, Any] = {
    # Populations named by a set operation or a filter hold between these many members.
    "pop_min": 2,
    "pop_max": 30,
    # Options per anchor for pairs and triples (stable scrambled order), so no anchor floods its pool.
    "max_combo_options": 6,
    # ORDER BY / LIMIT size.
    "top_k": 3,
    # Relative weights of variants within their tier (default 1 each).
    "variant_weights": {},
    # The form of the two comparisons of three (VIABLE_DESIGNS; "winner" is the declared design, design log).
    "comparison_design": "winner",
}

IRI_COL, LIT_COL, NUM_COL = "iri", "literal", "number"

# --------------------------------------------------------------------------
# Anchors: each names one entity by a name unique for its kind.


def _named(kind: str, cls: str, param: str) -> Anchor:
    return Anchor(kind, (f"{param}_name",), f"?e a {cls} ; pv:name ${param}_name_lit .", param)


def hardware_anchor(param: str) -> Anchor:
    return _named("hardware product", "pv:Hardware", param)


def supplier_anchor(param: str) -> Anchor:
    return _named("supplier", "pv:Supplier", param)


def category_anchor(param: str) -> Anchor:
    return _named("product category", "pv:ProductCategory", param)


SERVICE_ANCHOR = _named("service", "pv:Service", "service")
DEPARTMENT_ANCHOR = _named("department", "pv:Department", "department")
BOM_ANCHOR = _named("bill of materials", "pv:BillOfMaterial", "bom")
# Product managers are employees or managers (typed only pv:Manager); the name is unique across both.
PM_ANCHOR = Anchor(
    "person", ("pm_name",), "?e pv:name $pm_name_lit . { ?e a pv:Employee } UNION { ?e a pv:Manager }", "pm"
)
# A country is the string in supplier addresses; its "entity" is that string (country_key = "country:<name>").
COUNTRY_ANCHOR = Anchor(
    "country", ("country",), "?countrySupplier a pv:Supplier ; pv:addressCountry ?e . FILTER(?e = $country_lit)",
    "country_key",
)


def cvariant(**kwargs: Any) -> Variant:
    """A Variant with the corporate prefixes, no GRAPH names and no class-membership test."""
    kwargs.setdefault("membership_mode", "n.a.")
    return Variant(prefixes=PREFIXES, graphs=GRAPHS, **kwargs)


# --------------------------------------------------------------------------
# Alternative readings (harness.qgen.readings), suited to this graph


class PolarReading(Reading):
    """A reading whose clause is matched with quoted labels blanked out, so a name can neither supply nor defeat it."""

    def clause_present(self, text: str) -> bool:
        return super().clause_present(re.sub(r'"[^"]*"', '""', text))


def polar(pos: str, neg: str) -> str:
    """An exact-polarity clause: the text carries ``pos`` and nowhere a negated or reversed form of it (``neg``).

    Sentence-wide, so a second bound or a second mention cannot carry the opposite wording past the clause (code
    reviews of a11aa47: "all of which" matched inside "not all of which", "more than" inside "no more than").
    """
    return rf"^(?!.*(?:{neg}))(?=.*(?:{pos}))"


def _replace_once(old: str, new: str):
    def transform(sparql: str, params: dict) -> str | None:
        return sparql.replace(old, new, 1) if old in sparql else None

    return transform


def _country_by_link(sparql: str, params: dict) -> str | None:
    """Reading: "in the country X" read through the pv:country link of the country resource, not the address string."""
    literal = re.search(r'(\?\w+) pv:addressCountry ("(?:[^"\\]|\\.)*") \.', sparql)
    if literal:
        var, lit = literal.groups()
        new = (
            f"{var} pv:country ?countryLink . "
            f"{{ SELECT DISTINCT ?countryLink WHERE {{ ?countryNamed pv:addressCountry {lit} ; pv:country ?countryLink . }} }}"
        )
        return sparql[: literal.start()] + new + sparql[literal.end():]
    answer = re.search(r"(\?\w+) pv:addressCountry (\?\w+) \.", sparql)
    if answer:
        return sparql[: answer.start()] + f"{answer.group(1)} pv:country {answer.group(2)} ." + sparql[answer.end():]
    return None


R_COUNTRY_LINK = PolarReading(
    "country_link",
    "the country read through the supplier's link to a country resource, not the country named in its address",
    polar(
        r"\bwhose address is in\b|\b(?:have|with) their address in\b|\bare named in the addresses of\b",
        r"\baddress is not in\b|(?:\bnot|n't) have their address\b|\bwithout their address\b|"
        r"\bnot named in the addresses\b|\bnot (?:by|from|through) (?:the|their|its) address",
    ),
    transform=_country_by_link,
)
R_MANAGERS = PolarReading(
    "managers_as_employees",
    "employees also include the people typed only as managers (Manager is a subclass of Employee in the vocabulary)",
    polar(
        r"(?<![\w-])not counting managers\b",
        r"\bnot not counting\b|\bnot counting managers (?:separately|twice)\b|\b(?:also|including|even) (?:counting )?managers\b",
    ),
    transform=_replace_once("?employee a pv:Employee", "VALUES ?personKind { pv:Employee pv:Manager } ?employee a ?personKind"),
)
SERVICES_TRANSFORM = _replace_once(
    "?product a pv:Hardware", "VALUES ?productKind { pv:Hardware pv:Service } ?product a ?productKind"
)


def _services(question_pattern: str) -> Reading:
    """Services counted as products, ruled out by the question's own wording: positive-only, the clause is the
    variant's question form with its quoted names blanked, so any added wording fails it - in any position, with any
    connective (code reviews of bbf63b5 and 7a3454f: "hardware products and repair services", "hardware products, but
    also services" and "services, as well as other ... hardware products" carried a 'hardware products' clause)."""
    blanked = re.sub(r'"\$\w+"', '""', question_pattern)
    if "$" in blanked or "hardware products" not in blanked or re.search(r"servic", blanked, re.IGNORECASE):
        raise ValueError(f"the services clause needs a text with only quoted parameters and no services: {question_pattern}")
    return PolarReading(
        "services_as_products",
        "products also include services, not only hardware products",
        "^" + re.escape(blanked) + "$",
        transform=SERVICES_TRANSFORM,
    )


_PM_COUNT_TEXT = 'How many hardware products have "$pm_name" as their product manager? Give the number.'
_CATEGORY_COUNT_TEXT = 'How many hardware products belong to the product category "$category_name"? Give the number.'
_OUTSIDE_EXPERTISE_TEXT = (
    'Which hardware products whose product manager is "$pm_name" belong to none of the product categories '
    'that are areas of expertise of "$pm_name"? List each product once.'
)

# Suppliers with no hardware products: they pass a negation over their products ("none of the category") or a universal
# ("all of which are eligible") vacuously; the text's "supply hardware products" rules them out.
_PRODUCTLESS_NEG = (
    r"(?:\bnot|n't|\bneed not|\bnot necessarily) supply hardware products\b|\bsupply no hardware products\b|"
    r"\bor (?:supply )?nothing\b|\bwhether or not\b"
)


def _productless(description: str, clause: str, neg: str = _PRODUCTLESS_NEG) -> Reading:
    return PolarReading(
        "productless_suppliers",
        description,
        polar(clause, neg),
        transform=_replace_once("  ?any a pv:Hardware ; pv:hasSupplier ?supplier .\n", ""),
    )


# Exact-polarity clauses of the bounds: the strict or inclusive phrase, and its negated or reversed forms.
AT_LEAST = polar(r"\bat least\b", r"(?:\bnot|n't)(?: \w+)? at least\b|\bno fewer than\b|\bmore than\b")
MORE_THAN = polar(
    r"\bmore than\b", r"(?:\bno|\bnot|n't|\bnothing|\bor)(?: \w+)? more than\b|\bmore than or\b|\bat least\b|\bor more\b"
)
WEIGH_MORE_THAN = polar(
    r"\bweigh more than\b",
    r"(?:\bnot|n't) weigh\b|\b(?:no|or) more than\b|\bmore than or\b|\bas much as\b|\bat least\b",
)
LESS_THAN = polar(r"\bless than\b", r"(?:\bno|\bnot|n't|\bor)(?: \w+)? less than\b|\bless than or\b|\bat most\b|\bor less\b")


def _inclusive(op: str, flipped: str, clause: str) -> Reading:
    return PolarReading(
        "inclusive_bound",
        f"every bound read as {flipped} instead of {op}",
        clause,
        transform=lambda sparql, params: _flip_bounds(sparql, op, flipped),
    )


def _flip_bounds(sparql: str, op: str, flipped: str) -> str | None:
    """Every comparison "<?var or )> <op> <number or ?var>" closing a FILTER/HAVING or before && turned into ``flipped``."""
    pattern = r"(\?\w+|\)) " + re.escape(op) + r" (-?[0-9.]+|\?\w+)(?=\)| &&)"
    out, n = re.subn(pattern, lambda m: f"{m.group(1)} {flipped} {m.group(2)}", sparql)
    return out if n else None


def _without_filter_on_anchor(sparql: str, params: dict) -> str | None:
    m = re.search(r"FILTER\(\?\w+ != <[^>]+>\)\s*", sparql)
    return sparql[: m.start()] + sparql[m.end():] if m else None


R_ANCHOR_SELF = PolarReading(
    "anchor_inclusion",
    "the anchor itself counts among the answers when the path leads back to it",
    polar(r"\bitself is left out\b", r"\bitself is (?:not|never)\b|\bitself (?:is |also )?(?:included|counted|counts)\b"),
    transform=_without_filter_on_anchor,
)
R_DISTINCT = PolarReading(
    "count_with_repeats",
    "every part's supplier counted, a supplier of several parts several times",
    polar(
        r"\bhow many different suppliers\b",
        r"\bnot (?:necessarily )?different\b|\bcounting (?:each|every|repeats)\b|\bwith repeats\b|\bper part\b",
    ),
    transform=_replace_once("COUNT(DISTINCT ?supplier)", "COUNT(?supplier)"),
)


# --------------------------------------------------------------------------
# Shared pieces of candidate queries

# Hardware products whose name no other hardware product carries.
_UNIQUE_HW = """
  {{
    SELECT ?{var} (SAMPLE(?{var}N) AS ?{var}Name) WHERE {{
      ?{var} a pv:Hardware ; pv:name ?{var}N .
      {{ SELECT ?{var}N WHERE {{ ?{var}X a pv:Hardware ; pv:name ?{var}N . }} GROUP BY ?{var}N HAVING (COUNT(DISTINCT ?{var}X) = 1) }}
    }}
    GROUP BY ?{var}
  }}
"""
_UNIQUE_SUPPLIER = """
  {{
    SELECT ?{var} (SAMPLE(?{var}N) AS ?{var}Name) WHERE {{
      ?{var} a pv:Supplier ; pv:name ?{var}N .
      {{ SELECT ?{var}N WHERE {{ ?{var}X a pv:Supplier ; pv:name ?{var}N . }} GROUP BY ?{var}N HAVING (COUNT(DISTINCT ?{var}X) = 1) }}
    }}
    GROUP BY ?{var}
  }}
"""
# Product managers that are typed people with a name (three product managers in the data have neither).
_NAMED_PM = "?{var} pv:name ?{var}Name . {{ ?{var} a pv:Employee }} UNION {{ ?{var} a pv:Manager }}"


def unique_hw(var: str) -> str:
    return _UNIQUE_HW.format(var=var)


def unique_supplier(var: str) -> str:
    return _UNIQUE_SUPPLIER.format(var=var)


def named_pm(var: str) -> str:
    return _NAMED_PM.format(var=var)


# --------------------------------------------------------------------------
# Row -> parameter helpers


def _stable(items: Iterable[Any], salt: str) -> list[Any]:
    """Items in a scrambled order that depends only on their string form and the salt."""
    return sorted(items, key=lambda x: hashlib.sha256(f"{salt}|{x}".encode()).hexdigest())


def _shape(row: PoolRow) -> dict[str, Any]:
    return {"_stratum": "multi" if int(row["n"]) > 1 else "single"} if row.get("n") not in (None, "") else {}


def _country(name: str) -> dict[str, Any]:
    """A country's parameters: its name (in the text), the literal (in the gold) and its anchor key."""
    return {"country": name, "country_lit": sparql_literal(name), "country_key": f"country:{name}"}


def _ok(*labels: Any) -> bool:
    return all(isinstance(x, str) and usable_label(x) for x in labels)


def _one(*specs: tuple[str, str, str]):
    """Expand one row into one parameter dict: specs are (param, row column, "iri" | "label")."""

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        for row in rows:
            out: dict[str, Any] = {}
            labels = []
            for param, col, kind in specs:
                value = row[col]
                if kind == "iri":
                    out[param] = check_iri(value)
                elif kind == "country":
                    labels.append(value)
                    out.update(_country(value))
                else:
                    labels.append(value)
                    out[param] = value
            if _ok(*labels):
                yield {**out, **_shape(row)}

    return expand


def _group(rows: list[PoolRow], key: str) -> OrderedDict[str, list[PoolRow]]:
    out: OrderedDict[str, list[PoolRow]] = OrderedDict()
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


# --------------------------------------------------------------------------
# 1. type_members


class TypeMembers(QuestionTemplate):
    id = "type_members"
    description = (
        "Members of a group: the products of a supplier, the suppliers of a country, the employees of a department or "
        "with an area of expertise; compositional: intersections of two populations and a union of three."
    )
    membership_mode = "n.a."
    variants = (
        cvariant(
            name="supplier_products",
            description="Hardware products of one supplier.",
            candidate_query="""
SELECT ?supplier ?supplierName (COUNT(DISTINCT ?product) AS ?n) WHERE {
""" + unique_supplier("supplier") + """
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
}
GROUP BY ?supplier ?supplierName
ORDER BY ?supplier
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasSupplier <$supplier> .
}""",
            question_pattern='Which hardware products does the supplier "$supplier_name" supply? List each product once.',
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="supplier",
            expand=_one(("supplier", "supplier", "iri"), ("supplier_name", "supplierName", "label")),
            anchors=(supplier_anchor("supplier"),),
            hops=1,
            weight=2,
            difficulty={"hops": 1, "anchors": 1},
        ),
        cvariant(
            name="country_suppliers",
            description="Suppliers whose address names a country (the address string, not the country link).",
            candidate_query="""
SELECT ?country (COUNT(DISTINCT ?supplier) AS ?n) WHERE {
  ?supplier a pv:Supplier ; pv:addressCountry ?country .
}
GROUP BY ?country
HAVING (COUNT(DISTINCT ?supplier) <= 50)
ORDER BY ?country
""",
            gold_pattern="""SELECT DISTINCT ?supplier WHERE {
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
}""",
            question_pattern='Which suppliers have their address in the country "$country"? List each supplier once.',
            readings=(R_COUNTRY_LINK,),
            answer_columns=(AnswerColumn("supplier", IRI_COL),),
            key_param="country_key",
            expand=_one(("country", "country", "country")),
            anchors=(COUNTRY_ANCHOR,),
            hops=1,
            difficulty={"hops": 1, "anchors": 1},
        ),
        cvariant(
            name="department_members",
            description="Employees of one department; people typed only as managers are not employees in the data.",
            candidate_query="""
SELECT ?department ?departmentName (COUNT(DISTINCT ?employee) AS ?n) WHERE {
  ?department a pv:Department ; pv:name ?departmentName .
  ?employee a pv:Employee ; pv:memberOf ?department .
}
GROUP BY ?department ?departmentName
ORDER BY ?department
""",
            gold_pattern="""SELECT DISTINCT ?employee WHERE {
  ?employee a pv:Employee ; pv:memberOf <$department> .
}""",
            question_pattern=(
                'Which employees, not counting managers, are members of the department "$department_name"? List each '
                "employee once."
            ),
            readings=(R_MANAGERS,),
            answer_columns=(AnswerColumn("employee", IRI_COL),),
            key_param="department",
            expand=_one(("department", "department", "iri"), ("department_name", "departmentName", "label")),
            anchors=(DEPARTMENT_ANCHOR,),
            hops=1,
            difficulty={"hops": 1, "anchors": 1},
        ),
        cvariant(
            name="category_experts",
            description="Employees with a product category among their areas of expertise.",
            candidate_query="""
SELECT ?category ?categoryName (COUNT(DISTINCT ?employee) AS ?n) WHERE {
  ?category a pv:ProductCategory ; pv:name ?categoryName .
  ?employee a pv:Employee ; pv:areaOfExpertise ?category .
}
GROUP BY ?category ?categoryName
ORDER BY ?category
""",
            gold_pattern="""SELECT DISTINCT ?employee WHERE {
  ?employee a pv:Employee ; pv:areaOfExpertise <$category> .
}""",
            question_pattern=(
                'Which employees, not counting managers, have the product category "$category_name" among their areas '
                "of expertise? List each employee once."
            ),
            readings=(R_MANAGERS,),
            answer_columns=(AnswerColumn("employee", IRI_COL),),
            key_param="category",
            expand=_one(("category", "category", "iri"), ("category_name", "categoryName", "label")),
            anchors=(category_anchor("category"),),
            hops=1,
            difficulty={"hops": 1, "anchors": 1},
        ),
        cvariant(
            name="category_country_products",
            tier="compositional",
            description="Hardware products of a category whose supplier's address is in a country (intersection).",
            candidate_query="""
SELECT ?category ?categoryName ?country (COUNT(DISTINCT ?product) AS ?n) WHERE {
  ?category a pv:ProductCategory ; pv:name ?categoryName .
  ?product a pv:Hardware ; pv:hasCategory ?category ; pv:hasSupplier ?supplier .
  ?supplier a pv:Supplier ; pv:addressCountry ?country .
}
GROUP BY ?category ?categoryName ?country
HAVING (COUNT(DISTINCT ?product) >= $pop_min && COUNT(DISTINCT ?product) <= $pop_max)
ORDER BY ?category ?country
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasCategory <$category> ; pv:hasSupplier ?supplier .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
}""",
            question_pattern=(
                'Which hardware products belong to the product category "$category_name" and have a supplier whose '
                'address is in the country "$country"? List each product once.'
            ),
            readings=(R_COUNTRY_LINK,),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="category",
            expand=_one(("category", "category", "iri"), ("category_name", "categoryName", "label"), ("country", "country", "country")),
            anchors=(category_anchor("category"), COUNTRY_ANCHOR),
            hops=3,
            difficulty={"hops": 3, "anchors": 2, "set_operation": "intersection", "populations": 2},
            step_check=StepCheck(
                "every_population_matters",
                undo=(
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
}""", {}),
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasCategory <$category> .
}""", {}),
                ),
                steps=("category", "country"),
            ),
        ),
        cvariant(
            name="supplier_trio_union",
            tier="compositional",
            description="Hardware products of any of three suppliers of one country (union of three populations).",
            candidate_query="""
SELECT ?country ?supplier ?supplierName (COUNT(DISTINCT ?product) AS ?n) WHERE {
""" + unique_supplier("supplier") + """
  ?supplier pv:addressCountry ?country .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
}
GROUP BY ?country ?supplier ?supplierName
ORDER BY ?country ?supplier
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  { ?product a pv:Hardware ; pv:hasSupplier <$a> . }
  UNION
  { ?product a pv:Hardware ; pv:hasSupplier <$b> . }
  UNION
  { ?product a pv:Hardware ; pv:hasSupplier <$c> . }
}""",
            question_pattern=(
                'Which hardware products are supplied by the supplier "$a_name", the supplier "$b_name" or the supplier '
                '"$c_name"? List each product once.'
            ),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="a",
            expand=lambda rows, knobs=None: _supplier_trios(rows, knobs),
            anchors=(supplier_anchor("a"), supplier_anchor("b"), supplier_anchor("c")),
            hops=1,
            difficulty={"hops": 1, "anchors": 3, "set_operation": "union", "populations": 3},
            step_check=StepCheck(
                "every_population_matters",
                undo=tuple(
                    (
                        (
                            "SELECT DISTINCT ?product WHERE {\n"
                            f"  {{ ?product a pv:Hardware ; pv:hasSupplier <${x}> . }}\n  UNION\n"
                            f"  {{ ?product a pv:Hardware ; pv:hasSupplier <${y}> . }}\n}}"
                        ),
                        {},
                    )
                    for x, y in (("b", "c"), ("a", "c"), ("a", "b"))
                ),
                steps=("a", "b", "c"),
            ),
        ),
        cvariant(
            name="pm_country_products",
            tier="compositional",
            description="Hardware products of one product manager whose supplier's address is in a country (intersection).",
            candidate_query="""
SELECT ?pm ?pmName ?country (COUNT(DISTINCT ?product) AS ?n) WHERE {
  """ + named_pm("pm") + """
  ?product a pv:Hardware ; pv:hasProductManager ?pm ; pv:hasSupplier ?supplier .
  ?supplier a pv:Supplier ; pv:addressCountry ?country .
}
GROUP BY ?pm ?pmName ?country
HAVING (COUNT(DISTINCT ?product) >= $pop_min && COUNT(DISTINCT ?product) <= $pop_max)
ORDER BY ?pm ?country
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> ; pv:hasSupplier ?supplier .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
}""",
            question_pattern=(
                'Which hardware products have "$pm_name" as their product manager and a supplier whose address is in '
                'the country "$country"? List each product once.'
            ),
            readings=(R_COUNTRY_LINK,),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="pm",
            expand=_one(("pm", "pm", "iri"), ("pm_name", "pmName", "label"), ("country", "country", "country")),
            anchors=(PM_ANCHOR, COUNTRY_ANCHOR),
            hops=3,
            difficulty={"hops": 3, "anchors": 2, "set_operation": "intersection", "populations": 2},
            step_check=StepCheck(
                "every_population_matters",
                undo=(
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
}""", {}),
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> .
}""", {}),
                ),
                steps=("product manager", "country"),
            ),
        ),
        cvariant(
            name="service_category_products",
            tier="compositional",
            description="Hardware products of a category that are eligible for a service (intersection).",
            candidate_query="""
SELECT ?service ?serviceName ?category ?categoryName (COUNT(DISTINCT ?product) AS ?n) WHERE {
  ?service a pv:Service ; pv:name ?serviceName ; pv:eligibleFor ?product .
  ?product a pv:Hardware ; pv:hasCategory ?category .
  ?category a pv:ProductCategory ; pv:name ?categoryName .
}
GROUP BY ?service ?serviceName ?category ?categoryName
HAVING (COUNT(DISTINCT ?product) >= $pop_min && COUNT(DISTINCT ?product) <= $pop_max)
ORDER BY ?service ?category
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  <$service> pv:eligibleFor ?product .
  ?product a pv:Hardware ; pv:hasCategory <$category> .
}""",
            question_pattern=(
                'Which hardware products of the product category "$category_name" are eligible for the service '
                '"$service_name"? List each product once.'
            ),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="service",
            expand=_one(
                ("service", "service", "iri"), ("service_name", "serviceName", "label"),
                ("category", "category", "iri"), ("category_name", "categoryName", "label"),
            ),
            anchors=(SERVICE_ANCHOR, category_anchor("category")),
            hops=2,
            difficulty={"hops": 2, "anchors": 2, "set_operation": "intersection", "populations": 2},
            step_check=StepCheck(
                "every_population_matters",
                undo=(
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasCategory <$category> .
}""", {}),
                    ("""SELECT DISTINCT ?product WHERE {
  <$service> pv:eligibleFor ?product .
  ?product a pv:Hardware .
}""", {}),
                ),
                steps=("service", "category"),
            ),
        ),
    )


def _supplier_trios(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """Triples of named suppliers of one country, together supplying at most pop_max products."""
    cap = int((knobs or KNOBS).get("pop_max", KNOBS["pop_max"]))
    per = int((knobs or KNOBS).get("max_combo_options", KNOBS["max_combo_options"]))
    for country, group in _group(rows, "country").items():
        named = [r for r in group if _ok(r["supplierName"])]
        trios = [t for t in combinations(sorted(named, key=lambda r: r["supplier"]), 3) if sum(int(r["n"]) for r in t) <= cap]
        for t in _stable(trios, f"trio|{country}")[:per]:
            out: dict[str, Any] = {}
            for slot, r in zip("abc", t):
                out[slot] = check_iri(r["supplier"])
                out[f"{slot}_name"] = r["supplierName"]
            yield out


# --------------------------------------------------------------------------
# 2. one_hop_attribute


_ABC = (("a", "a_name"), ("b", "b_name"), ("c", "c_name"))


def _slot_of(params: dict, iri: str) -> int:
    return [params[s[0]] for s in _ABC].index(iri)


def _winner_in_rows(params: dict, rows: list, step: Any) -> int:
    """The winner's slot (the answer's product); with a step outcome, also the declared design's condition (science
    agent's ruling in the design log): dropping the winner, and only the winner, changed the answer."""
    winner = _slot_of(params, rows[0][0]["value"])
    changed = (step or {}).get("undo_changed")
    if changed is not None and changed != [winner == 2, winner == 1, winner == 0]:  # undo order: drop c, b, a
        raise TemplateError(f"the winner (slot {_ABC[winner][0]}) must be the only product whose drop changes the answer: {changed}")
    return winner


def _rank_one_in_rows(params: dict, rows: list, step: Any) -> int:
    """The slot of the product ranked 1 (for balancing mention slots)."""
    (first,) = [r for r in rows if Decimal(r[1]["value"]) == 1]
    return _slot_of(params, first[0]["value"])


def _first_in_rows(params: dict, rows: list, step: Any) -> int:
    return _slot_of(params, rows[0][0]["value"])


def _triple_params(key: str, trio: Iterable[PoolRow]) -> dict[str, Any]:
    out: dict[str, Any] = {"key": key}
    for slot, r in zip("abc", trio):
        out[slot] = check_iri(r["product"])
        out[f"{slot}_name"] = r["productName"]
    return out


def _product_triples(value_col: str):
    """Triples of uniquely named products of one supplier with pairwise distinct values (key: the supplier, not named)."""

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        per = int((knobs or KNOBS).get("max_combo_options", KNOBS["max_combo_options"]))
        for supplier, group in _group(rows, "supplier").items():
            named = sorted((r for r in group if _ok(r["productName"])), key=lambda r: r["product"])
            trios = [t for t in combinations(named, 3) if len({float(r[value_col]) for r in t}) == 3]
            for t in _stable(trios, f"triple|{value_col}|{supplier}")[:per]:
                yield _triple_params(check_iri(supplier), t)

    return expand


def _spread_triples(value_col: str):
    """For each uniquely named product, one triple with the first two others (stable scrambled order) whose values
    differ from it and from each other (key: that product)."""

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        named = sorted((r for r in rows if _ok(r["productName"])), key=lambda r: r["product"])
        for a in named:
            picked = [a]
            for r in _stable(named, f"spread|{value_col}|{a['product']}"):
                if all(float(r[value_col]) != float(x[value_col]) for x in picked):
                    picked.append(r)
                    if len(picked) == 3:
                        yield _triple_params(check_iri(a["product"]), picked)
                        break

    return expand


def _undo_drop_one(gold: str) -> tuple[tuple[str, dict[str, str]], ...]:
    """The comparison gold over two of the three entities: drop c, b, then a (renaming the later ones down)."""
    return ((gold, {"c": "b", "c_name": "b_name"}), (gold, {"b": "c", "b_name": "c_name"}), (gold, {"a": "c", "a_name": "c_name"}))


# --------------------------------------------------------------------------
# Comparisons of three products (one_hop_attribute: price; numeric_filter: reliability index)
#
# In the "winner" form (which of the three is cheapest / most reliable) only the winner's drop can change the answer,
# so dropping a losing product never does (code reviews of a11aa47). The knob comparison_design selects the form;
# "winner" is the declared design (study lead's ruling in the design log; NOTES["declared_design"]): the
# winner must be load-bearing, checked as the only drop that changes the answer, and its mention slot is balanced.
# "ranking" (each product with its rank) and "middle" (the product between the other two) are redesigns, built and
# tested but not used, in which dropping any of the three changes the answer. "gap" (second best minus best) is built only to measure it: dropping
# the worst product never changes it, so it fails the rule by construction and is not a setting.

COMPARISON_DESIGNS = ("winner", "ranking", "middle", "gap")
VIABLE_DESIGNS = ("winner", "ranking", "middle")

_CHEAPEST_GOLD = """SELECT ?product ?amount WHERE {
  VALUES ?product { <$a> <$b> <$c> }
  ?product a pv:Hardware ; pv:price ?price .
  ?price pv:amount ?amount .
}
ORDER BY ?amount
LIMIT 1"""
_RELIABLE_GOLD = """SELECT ?product ?reliability WHERE {
  VALUES ?product { <$a> <$b> <$c> }
  ?product a pv:Hardware ; pv:reliabilityIndex ?reliability .
}
ORDER BY DESC(?reliability)
LIMIT 1"""
# Rank = how many of the three are as good or better (the values are pairwise distinct): 1 for the best.
_PRICE_RANK_GOLD = """SELECT ?product (COUNT(DISTINCT ?other) AS ?rank) WHERE {
  VALUES ?product { <$a> <$b> <$c> }
  VALUES ?other { <$a> <$b> <$c> }
  ?product a pv:Hardware ; pv:price ?price .
  ?price pv:amount ?amount .
  ?other a pv:Hardware ; pv:price ?otherPrice .
  ?otherPrice pv:amount ?otherAmount .
  FILTER(?otherAmount <= ?amount)
}
GROUP BY ?product"""
_RELIABLE_RANK_GOLD = """SELECT ?product (COUNT(DISTINCT ?other) AS ?rank) WHERE {
  VALUES ?product { <$a> <$b> <$c> }
  VALUES ?other { <$a> <$b> <$c> }
  ?product a pv:Hardware ; pv:reliabilityIndex ?reliability .
  ?other a pv:Hardware ; pv:reliabilityIndex ?otherReliability .
  FILTER(?otherReliability >= ?reliability)
}
GROUP BY ?product"""
# Strictly between the lowest and the highest of the three: with two products left there is none.
_PRICE_MIDDLE_GOLD = """SELECT ?product ?amount WHERE {
  {
    SELECT (MIN(?x) AS ?lowest) (MAX(?x) AS ?highest) WHERE {
      VALUES ?compared { <$a> <$b> <$c> }
      ?compared a pv:Hardware ; pv:price ?comparedPrice .
      ?comparedPrice pv:amount ?x .
    }
  }
  VALUES ?product { <$a> <$b> <$c> }
  ?product a pv:Hardware ; pv:price ?price .
  ?price pv:amount ?amount .
  FILTER(?amount > ?lowest && ?amount < ?highest)
}"""
_RELIABLE_MIDDLE_GOLD = """SELECT ?product ?reliability WHERE {
  {
    SELECT (MIN(?x) AS ?lowest) (MAX(?x) AS ?highest) WHERE {
      VALUES ?compared { <$a> <$b> <$c> }
      ?compared a pv:Hardware ; pv:reliabilityIndex ?x .
    }
  }
  VALUES ?product { <$a> <$b> <$c> }
  ?product a pv:Hardware ; pv:reliabilityIndex ?reliability .
  FILTER(?reliability > ?lowest && ?reliability < ?highest)
}"""
_PRICE_GAP_GOLD = """SELECT ?difference WHERE {
  {
    SELECT (MIN(?x) AS ?lowest) WHERE {
      VALUES ?compared { <$a> <$b> <$c> }
      ?compared a pv:Hardware ; pv:price ?comparedPrice .
      ?comparedPrice pv:amount ?x .
    }
  }
  {
    SELECT (MIN(?y) AS ?second) WHERE {
      VALUES ?compared { <$a> <$b> <$c> }
      VALUES ?cheaper { <$a> <$b> <$c> }
      ?compared a pv:Hardware ; pv:price ?comparedPrice .
      ?comparedPrice pv:amount ?y .
      ?cheaper a pv:Hardware ; pv:price ?cheaperPrice .
      ?cheaperPrice pv:amount ?z .
      FILTER(?z < ?y)
    }
  }
  BIND(?second - ?lowest AS ?difference)
}"""
_RELIABLE_GAP_GOLD = """SELECT ?difference WHERE {
  {
    SELECT (MAX(?x) AS ?highest) WHERE {
      VALUES ?compared { <$a> <$b> <$c> }
      ?compared a pv:Hardware ; pv:reliabilityIndex ?x .
    }
  }
  {
    SELECT (MAX(?y) AS ?second) WHERE {
      VALUES ?compared { <$a> <$b> <$c> }
      VALUES ?better { <$a> <$b> <$c> }
      ?compared a pv:Hardware ; pv:reliabilityIndex ?y .
      ?better a pv:Hardware ; pv:reliabilityIndex ?z .
      FILTER(?z > ?y)
    }
  }
  BIND(?highest - ?second AS ?difference)
}"""

_TRIO_ANCHORS = (hardware_anchor("a"), hardware_anchor("b"), hardware_anchor("c"))
_KINDS: dict[str, dict[str, Any]] = {
    "price": {
        "base": {
            "candidate_query": """
SELECT ?supplier ?product ?productName ?amount WHERE {
""" + unique_hw("product") + """
  ?product pv:hasSupplier ?supplier ; pv:price ?price .
  ?price pv:amount ?amount .
}
ORDER BY ?supplier ?product
""",
            "key_param": "key",
            "expand": _product_triples("amount"),
            "anchors": _TRIO_ANCHORS,
            "hops": 2,
        },
        "value": AnswerColumn("amount", NUM_COL),
        "winner": {
            "name": "cheapest_of_three",
            "description": "Which of three hardware products of one supplier has the lowest price (comparison of three).",
            "gold_pattern": _CHEAPEST_GOLD,
            "question_pattern": (
                'Which of the three hardware products "$a_name", "$b_name" and "$c_name" has the lowest price? Give '
                "that product with its price in euros."
            ),
        },
        "ranking": {
            "name": "price_ranking_of_three",
            "description": "The three hardware products of one supplier ranked by price, 1 for the cheapest (every product matters).",
            "gold_pattern": _PRICE_RANK_GOLD,
            "question_pattern": (
                'Rank the three hardware products "$a_name", "$b_name" and "$c_name" by price, from 1 for the cheapest to '
                "3 for the most expensive. Give each product with its rank."
            ),
        },
        "middle": {
            "name": "middle_price_of_three",
            "description": "Which of three hardware products of one supplier has the middle price (every product matters).",
            "gold_pattern": _PRICE_MIDDLE_GOLD,
            "question_pattern": (
                'Which of the three hardware products "$a_name", "$b_name" and "$c_name" has the middle price, higher than '
                "one of the other two and lower than the other? Give that product with its price in euros."
            ),
        },
        "gap": {
            "name": "price_gap_of_three",
            "description": "The second lowest price of three hardware products minus the lowest (measured only).",
            "gold_pattern": _PRICE_GAP_GOLD,
            "question_pattern": (
                'What is the price in euros of the second cheapest of the three hardware products "$a_name", "$b_name" '
                'and "$c_name" minus the price of the cheapest of them? Give the number.'
            ),
        },
        "rank_flip": ("FILTER(?otherAmount <= ?amount)", "FILTER(?otherAmount >= ?amount)"),
    },
    "reliability": {
        "base": {
            "candidate_query": """
SELECT ?product ?productName ?reliability WHERE {
""" + unique_hw("product") + """
  ?product pv:reliabilityIndex ?reliability .
}
ORDER BY ?product
""",
            "key_param": "key",
            "expand": _spread_triples("reliability"),
            "anchors": _TRIO_ANCHORS,
            "hops": 1,
        },
        "value": AnswerColumn("reliability", NUM_COL),
        "winner": {
            "name": "most_reliable_of_three",
            "description": (
                "Which of three hardware products has the highest reliability index (comparison; the index is one value "
                "per supplier country, so the three come from suppliers in three countries)."
            ),
            "gold_pattern": _RELIABLE_GOLD,
            "question_pattern": (
                'Which of the three hardware products "$a_name", "$b_name" and "$c_name" has the highest reliability '
                "index? Give that product with its reliability index."
            ),
        },
        "ranking": {
            "name": "reliability_ranking_of_three",
            "description": "Three hardware products ranked by reliability index, 1 for the highest (every product matters).",
            "gold_pattern": _RELIABLE_RANK_GOLD,
            "question_pattern": (
                'Rank the three hardware products "$a_name", "$b_name" and "$c_name" by reliability index, from 1 for the '
                "highest index to 3 for the lowest. Give each product with its rank."
            ),
        },
        "middle": {
            "name": "middle_reliability_of_three",
            "description": "Which of three hardware products has the middle reliability index (every product matters).",
            "gold_pattern": _RELIABLE_MIDDLE_GOLD,
            "question_pattern": (
                'Which of the three hardware products "$a_name", "$b_name" and "$c_name" has the middle reliability index, '
                "higher than one of the other two and lower than the other? Give that product with its reliability index."
            ),
        },
        "gap": {
            "name": "reliability_gap_of_three",
            "description": "The highest reliability index of three hardware products minus the second highest (measured only).",
            "gold_pattern": _RELIABLE_GAP_GOLD,
            "question_pattern": (
                'What is the reliability index of the most reliable of the three hardware products "$a_name", "$b_name" '
                'and "$c_name" minus that of the second most reliable? Give the number.'
            ),
        },
        "rank_flip": ("FILTER(?otherReliability >= ?reliability)", "FILTER(?otherReliability <= ?reliability)"),
    },
}

R_RANK_DIRECTION = (
    "rank_direction",
    "ranks counted from the other end (1 for the most expensive, or for the lowest index)",
    polar(
        r"\bfrom 1 for the (?:cheapest|highest index)\b",
        r"\bfrom 1 for the (?:most expensive|lowest index)\b|\bfrom 3 for the (?:cheapest|highest)\b",
    ),
)


def _comparison_variant(kind: str, design: str) -> Variant:
    spec = _KINDS[kind]
    text = spec[design]
    gold = text["gold_pattern"]
    steps = ("c", "b", "a")
    common = dict(tier="compositional", **spec["base"], **text)
    if design == "winner":
        return cvariant(
            **common,
            answer_columns=(AnswerColumn("product", IRI_COL), spec["value"]),
            difficulty={"hops": spec["base"]["hops"], "anchors": 3, "comparison": True, "aggregate": "order_limit", "populations": 3},
            slots=_ABC,
            winner=_winner_in_rows,
            step_check=StepCheck("winner_matters", undo=_undo_drop_one(gold), mode="some", steps=steps),
        )
    every = StepCheck("every_product_matters", undo=_undo_drop_one(gold), mode="every", steps=steps)
    difficulty = {"hops": spec["base"]["hops"], "anchors": 3, "comparison": True, "populations": 3}
    if design == "ranking":
        return cvariant(
            **common,
            answer_columns=(AnswerColumn("product", IRI_COL), AnswerColumn("rank", NUM_COL)),
            difficulty={**difficulty, "aggregate": "ranking"},
            readings=(PolarReading(*R_RANK_DIRECTION, transform=_replace_once(*spec["rank_flip"])),),
            slots=_ABC,
            winner=_rank_one_in_rows,
            step_check=every,
        )
    if design == "middle":
        return cvariant(
            **common,
            answer_columns=(AnswerColumn("product", IRI_COL), spec["value"]),
            difficulty={**difficulty, "aggregate": "between"},
            slots=_ABC,
            winner=_first_in_rows,
            step_check=every,
        )
    return cvariant(
        **common,
        answer_columns=(AnswerColumn("difference", NUM_COL),),
        difficulty={**difficulty, "aggregate": "order_statistic_difference"},
        step_check=every,
    )


COMPARISON_VARIANTS: dict[tuple[str, str], Variant] = {
    (kind, design): _comparison_variant(kind, design) for kind in _KINDS for design in COMPARISON_DESIGNS
}


def comparison_variant(kind: str, design: str) -> Variant:
    """The comparison of three products ("price" or "reliability") in one design (COMPARISON_DESIGNS)."""
    return COMPARISON_VARIANTS[(kind, design)]


def resolve_comparisons(templates: Iterable[type[QuestionTemplate]], knobs: Any) -> Any:
    """The templates with both comparisons of three in the design the knob comparison_design names ("winner": the
    templates unchanged, today's questions)."""
    design = (knobs or {}).get("comparison_design", "winner")
    if design not in VIABLE_DESIGNS:
        raise ValueError(f"comparison_design must be one of {VIABLE_DESIGNS}, got {design!r}")
    if design == "winner":
        return templates
    swap = {COMPARISON_VARIANTS[(k, "winner")].name: COMPARISON_VARIANTS[(k, design)] for k in _KINDS}
    out = []
    for t in templates:
        if any(v.name in swap for v in t.variants):
            out.append(type(t.__name__, (t,), {"variants": tuple(swap.get(v.name, v) for v in t.variants)}))
        else:
            out.append(t)
    return tuple(out)

_SHARED_PAIRS = """
SELECT ?a ?aName ?b ?bName (COUNT(DISTINCT ?x) AS ?n) WHERE {
""" + unique_hw("a") + unique_hw("b") + """
  ?x a pv:Hardware ; pv:compatibleProduct ?a , ?b .
  FILTER(STR(?a) < STR(?b))
  FILTER NOT EXISTS { ?a pv:compatibleProduct ?b . }
}
GROUP BY ?a ?aName ?b ?bName
ORDER BY ?a ?b
"""


def _pairs(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """Pairs of uniquely named products, at most max_combo_options per first product (stable scrambled order)."""
    per = int((knobs or KNOBS).get("max_combo_options", KNOBS["max_combo_options"]))
    for a, group in _group(rows, "a").items():
        for r in _stable(group, f"pair|{a}")[:per]:
            if _ok(r["aName"], r["bName"]):
                yield {"a": check_iri(r["a"]), "a_name": r["aName"], "b": check_iri(r["b"]), "b_name": r["bName"]}


def _ordered_pairs(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """Both orders of each pair, the first-named product being the one whose neighbours are filtered."""
    for p in _pairs(rows, knobs):
        yield p
        yield {"a": p["b"], "a_name": p["b_name"], "b": p["a"], "b_name": p["a_name"]}


class OneHopAttribute(QuestionTemplate):
    id = "one_hop_attribute"
    description = (
        "An attribute one hop from an entity (a product's price and categories, a supplier's locality); "
        "compositional: products compatible with both or with one but not the other of two products, and the "
        "cheapest of three products."
    )
    membership_mode = "n.a."
    variants = (
        cvariant(
            name="product_price",
            description="The price in euros of one hardware product.",
            candidate_query="""
SELECT ?product ?productName WHERE {
""" + unique_hw("product") + """
  ?product pv:price/pv:amount ?amount .
}
ORDER BY ?product
""",
            gold_pattern="""SELECT ?amount WHERE {
  <$product> a pv:Hardware ; pv:price ?price .
  ?price pv:amount ?amount .
}""",
            question_pattern='What is the price in euros of the hardware product "$product_name"? Give the amount as a number.',
            answer_columns=(AnswerColumn("amount", NUM_COL),),
            key_param="product",
            expand=_one(("product", "product", "iri"), ("product_name", "productName", "label")),
            anchors=(hardware_anchor("product"),),
            hops=2,
            difficulty={"hops": 2, "anchors": 1},
        ),
        cvariant(
            name="product_categories",
            description="The product categories of one hardware product.",
            candidate_query="""
SELECT ?product ?productName (COUNT(DISTINCT ?category) AS ?n) WHERE {
""" + unique_hw("product") + """
  ?product pv:hasCategory ?category .
}
GROUP BY ?product ?productName
ORDER BY ?product
""",
            gold_pattern="""SELECT DISTINCT ?category WHERE {
  <$product> a pv:Hardware ; pv:hasCategory ?category .
}""",
            question_pattern='Which product categories does the hardware product "$product_name" belong to? List each category once.',
            answer_columns=(AnswerColumn("category", IRI_COL),),
            key_param="product",
            expand=_one(("product", "product", "iri"), ("product_name", "productName", "label")),
            anchors=(hardware_anchor("product"),),
            hops=1,
            difficulty={"hops": 1, "anchors": 1},
        ),
        cvariant(
            name="supplier_locality",
            description="The locality in one supplier's address.",
            candidate_query="""
SELECT ?supplier ?supplierName WHERE {
""" + unique_supplier("supplier") + """
  ?supplier pv:addressLocality ?locality .
}
ORDER BY ?supplier
""",
            gold_pattern="""SELECT ?locality WHERE {
  <$supplier> a pv:Supplier ; pv:addressLocality ?locality .
}""",
            question_pattern='Which locality is given in the address of the supplier "$supplier_name"? Give the locality\'s name.',
            answer_columns=(AnswerColumn("locality", LIT_COL),),
            key_param="supplier",
            expand=_one(("supplier", "supplier", "iri"), ("supplier_name", "supplierName", "label")),
            anchors=(supplier_anchor("supplier"),),
            hops=1,
            difficulty={"hops": 1, "anchors": 1},
        ),
        cvariant(
            name="shared_compatible",
            tier="compositional",
            description="Hardware products compatible with both of two products (intersection of two neighbourhoods).",
            candidate_query=_SHARED_PAIRS,
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:compatibleProduct <$a> ; pv:compatibleProduct <$b> .
}""",
            question_pattern=(
                'Which hardware products are compatible with both the hardware product "$a_name" and the hardware '
                'product "$b_name"? List each product once.'
            ),
            readings=(
                PolarReading(
                    "either_not_both",
                    "compatible with either of the two products, not with both",
                    polar(
                        r"\bcompatible with both\b",
                        r"\bnot (?:compatible with |with )?both\b|\beither\b|\bat least one of\b|\bone or (?:both|the other)\b",
                    ),
                    pattern="""SELECT DISTINCT ?product WHERE {
  { ?product a pv:Hardware ; pv:compatibleProduct <$a> . } UNION { ?product a pv:Hardware ; pv:compatibleProduct <$b> . }
}""",
                ),
            ),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="a",
            expand=_pairs,
            anchors=(hardware_anchor("a"), hardware_anchor("b")),
            hops=2,
            difficulty={"hops": 2, "anchors": 2, "set_operation": "intersection", "populations": 2},
            step_check=StepCheck(
                "every_population_matters",
                undo=(
                    ("""SELECT DISTINCT ?product WHERE { ?product a pv:Hardware ; pv:compatibleProduct <$b> . }""", {}),
                    ("""SELECT DISTINCT ?product WHERE { ?product a pv:Hardware ; pv:compatibleProduct <$a> . }""", {}),
                ),
                steps=("a", "b"),
            ),
        ),
        cvariant(
            name="compatible_not_with",
            tier="compositional",
            description=(
                "Hardware products compatible with one product but not with another (difference of two "
                "neighbourhoods; the two are not compatible with each other)."
            ),
            candidate_query=_SHARED_PAIRS,
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:compatibleProduct <$a> .
  FILTER NOT EXISTS { ?product pv:compatibleProduct <$b> . }
}""",
            question_pattern=(
                'Which hardware products are compatible with the hardware product "$a_name" but not with the hardware '
                'product "$b_name"? List each product once.'
            ),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="a",
            expand=_ordered_pairs,
            anchors=(hardware_anchor("a"), hardware_anchor("b")),
            hops=2,
            difficulty={"hops": 2, "anchors": 2, "set_operation": "difference", "negation": 1, "populations": 2},
            step_check=StepCheck(
                "every_population_matters",
                undo=(
                    ("""SELECT DISTINCT ?product WHERE { ?product a pv:Hardware ; pv:compatibleProduct <$a> . }""", {}),
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware .
  FILTER NOT EXISTS { ?product pv:compatibleProduct <$b> . }
}""", {}),
                ),
                steps=("not compatible with b", "compatible with a"),
            ),
        ),
        # The comparison of three: its form is the comparison_design setting (see COMPARISON_DESIGNS).
        COMPARISON_VARIANTS[("price", "winner")],
    )


# --------------------------------------------------------------------------
# 3. multi_hop_path


class MultiHopPath(QuestionTemplate):
    id = "multi_hop_path"
    description = (
        "Entities two or more edges from an anchor: suppliers of compatible products or of a manager's products; "
        "compositional: suppliers of products compatible with a supplier's products, suppliers of a bill of "
        "materials' parts, two compatibility steps, and the countries of a product manager's suppliers."
    )
    membership_mode = "n.a."
    variants = (
        cvariant(
            name="compatible_suppliers",
            description="Suppliers of the hardware products compatible with one product.",
            candidate_query="""
SELECT ?product ?productName (COUNT(DISTINCT ?supplier) AS ?n) WHERE {
""" + unique_hw("product") + """
  ?product pv:compatibleProduct ?other .
  ?other a pv:Hardware ; pv:hasSupplier ?supplier .
}
GROUP BY ?product ?productName
ORDER BY ?product
""",
            gold_pattern="""SELECT DISTINCT ?supplier WHERE {
  <$product> pv:compatibleProduct ?other .
  ?other a pv:Hardware ; pv:hasSupplier ?supplier .
}""",
            question_pattern=(
                'Which suppliers supply the hardware products that are compatible with the hardware product '
                '"$product_name"? List each supplier once.'
            ),
            answer_columns=(AnswerColumn("supplier", IRI_COL),),
            key_param="product",
            expand=_one(("product", "product", "iri"), ("product_name", "productName", "label")),
            anchors=(hardware_anchor("product"),),
            hops=2,
            difficulty={"hops": 2, "anchors": 1},
        ),
        cvariant(
            name="pm_products_suppliers",
            description="Suppliers of the hardware products of one product manager.",
            candidate_query="""
SELECT ?pm ?pmName (COUNT(DISTINCT ?supplier) AS ?n) WHERE {
  """ + named_pm("pm") + """
  ?product a pv:Hardware ; pv:hasProductManager ?pm ; pv:hasSupplier ?supplier .
}
GROUP BY ?pm ?pmName
HAVING (COUNT(DISTINCT ?supplier) <= 50)
ORDER BY ?pm
""",
            gold_pattern="""SELECT DISTINCT ?supplier WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> ; pv:hasSupplier ?supplier .
}""",
            question_pattern=(
                'Which suppliers supply the hardware products whose product manager is "$pm_name"? List each supplier once.'
            ),
            answer_columns=(AnswerColumn("supplier", IRI_COL),),
            key_param="pm",
            expand=_one(("pm", "pm", "iri"), ("pm_name", "pmName", "label")),
            anchors=(PM_ANCHOR,),
            hops=2,
            difficulty={"hops": 2, "anchors": 1},
        ),
        cvariant(
            name="supplier_compatible_suppliers",
            tier="compositional",
            description=(
                "Other suppliers of hardware products compatible with a product of one supplier (three edges; the "
                "anchor supplier left out)."
            ),
            candidate_query="""
SELECT ?supplier ?supplierName (COUNT(DISTINCT ?otherSupplier) AS ?n) WHERE {
""" + unique_supplier("supplier") + """
  ?own a pv:Hardware ; pv:hasSupplier ?supplier ; pv:compatibleProduct ?other .
  ?other a pv:Hardware ; pv:hasSupplier ?otherSupplier .
  FILTER(?otherSupplier != ?supplier)
}
GROUP BY ?supplier ?supplierName
HAVING (COUNT(DISTINCT ?otherSupplier) >= $pop_min && COUNT(DISTINCT ?otherSupplier) <= $pop_max)
ORDER BY ?supplier
""",
            gold_pattern="""SELECT DISTINCT ?otherSupplier WHERE {
  ?own a pv:Hardware ; pv:hasSupplier <$supplier> ; pv:compatibleProduct ?other .
  ?other a pv:Hardware ; pv:hasSupplier ?otherSupplier .
  FILTER(?otherSupplier != <$supplier>)
}""",
            question_pattern=(
                'Which suppliers supply hardware products that are compatible with a hardware product supplied by the '
                'supplier "$supplier_name"? The supplier "$supplier_name" itself is left out. List each supplier once.'
            ),
            readings=(R_ANCHOR_SELF,),
            answer_columns=(AnswerColumn("supplier", IRI_COL),),
            key_param="supplier",
            expand=_one(("supplier", "supplier", "iri"), ("supplier_name", "supplierName", "label")),
            anchors=(supplier_anchor("supplier"),),
            hops=3,
            difficulty={"hops": 3, "anchors": 1},
            step_check=StepCheck(
                "path_fans_out",
                fan_out="""SELECT ?own ?otherSupplier WHERE {
  ?own a pv:Hardware ; pv:hasSupplier <$supplier> ; pv:compatibleProduct ?other .
  ?other a pv:Hardware ; pv:hasSupplier ?otherSupplier .
  FILTER(?otherSupplier != <$supplier>)
}""",
            ),
        ),
        cvariant(
            name="bom_part_suppliers",
            tier="compositional",
            description="Suppliers of the hardware products listed as parts of one bill of materials (three edges).",
            candidate_query="""
SELECT ?bom ?bomName (COUNT(DISTINCT ?supplier) AS ?n) WHERE {
  ?bom a pv:BillOfMaterial ; pv:name ?bomName ; pv:hasBomPart ?part .
  ?part pv:hasPart ?product .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
}
GROUP BY ?bom ?bomName
HAVING (COUNT(DISTINCT ?supplier) <= 50)
ORDER BY ?bom
""",
            gold_pattern="""SELECT DISTINCT ?supplier WHERE {
  <$bom> pv:hasBomPart ?part .
  ?part pv:hasPart ?product .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
}""",
            question_pattern=(
                'Which suppliers supply the hardware products listed as parts in the bill of materials "$bom_name"? '
                "List each supplier once."
            ),
            answer_columns=(AnswerColumn("supplier", IRI_COL),),
            key_param="bom",
            expand=_one(("bom", "bom", "iri"), ("bom_name", "bomName", "label")),
            anchors=(BOM_ANCHOR,),
            hops=3,
            difficulty={"hops": 3, "anchors": 1},
            step_check=StepCheck(
                "path_fans_out",
                fan_out="""SELECT ?product ?supplier WHERE {
  <$bom> pv:hasBomPart ?part .
  ?part pv:hasPart ?product .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
}""",
            ),
        ),
        cvariant(
            name="two_step_compatible",
            tier="compositional",
            description="Hardware products two compatibility steps from one product (the product itself left out).",
            candidate_query="""
SELECT ?product ?productName (COUNT(DISTINCT ?reached) AS ?n) WHERE {
""" + unique_hw("product") + """
  ?product pv:compatibleProduct ?step .
  ?step pv:compatibleProduct ?reached .
  FILTER(?reached != ?product)
}
GROUP BY ?product ?productName
HAVING (COUNT(DISTINCT ?reached) >= $pop_min && COUNT(DISTINCT ?reached) <= $pop_max)
ORDER BY ?product
""",
            gold_pattern="""SELECT DISTINCT ?reached WHERE {
  <$product> pv:compatibleProduct ?step .
  ?step a pv:Hardware ; pv:compatibleProduct ?reached .
  ?reached a pv:Hardware .
  FILTER(?reached != <$product>)
}""",
            question_pattern=(
                'Which hardware products are compatible with some hardware product that is itself compatible with the '
                'hardware product "$product_name"? The product "$product_name" itself is left out; products directly '
                "compatible with it count when they are reached this way. List each product once."
            ),
            readings=(
                R_ANCHOR_SELF,
                PolarReading(
                    "direct_neighbours_excluded",
                    "the products directly compatible with the anchor left out",
                    polar(
                        r"\bdirectly compatible with it count\b",
                        r"\bnot directly compatible\b|"
                        r"\bdirectly compatible with it (?:do not|don't|does not|never|are left out|are excluded)\b",
                    ),
                    transform=lambda sparql, params: sparql.replace(
                        "FILTER(?reached != <" + params["product"] + ">)",
                        "FILTER(?reached != <" + params["product"] + ">) "
                        "FILTER NOT EXISTS { <" + params["product"] + "> pv:compatibleProduct ?reached . }",
                    ),
                ),
            ),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="product",
            expand=_one(("product", "product", "iri"), ("product_name", "productName", "label")),
            anchors=(hardware_anchor("product"),),
            hops=2,
            difficulty={"hops": 2, "anchors": 1, "path": "two steps of one relation"},
            step_check=StepCheck(
                "path_fans_out",
                fan_out="""SELECT ?step ?reached WHERE {
  <$product> pv:compatibleProduct ?step .
  ?step a pv:Hardware ; pv:compatibleProduct ?reached .
  ?reached a pv:Hardware .
  FILTER(?reached != <$product>)
}""",
            ),
        ),
        cvariant(
            name="pm_supplier_countries",
            tier="compositional",
            description="Countries in the addresses of the suppliers of one product manager's hardware products (three edges).",
            candidate_query="""
SELECT ?pm ?pmName (COUNT(DISTINCT ?country) AS ?n) WHERE {
  """ + named_pm("pm") + """
  ?product a pv:Hardware ; pv:hasProductManager ?pm ; pv:hasSupplier ?supplier .
  ?supplier pv:addressCountry ?country .
}
GROUP BY ?pm ?pmName
HAVING (COUNT(DISTINCT ?country) >= $pop_min && COUNT(DISTINCT ?country) <= 50)
ORDER BY ?pm
""",
            gold_pattern="""SELECT DISTINCT ?country WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> ; pv:hasSupplier ?supplier .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry ?country .
}""",
            question_pattern=(
                'Which countries are named in the addresses of the suppliers of the hardware products whose product '
                'manager is "$pm_name"? Give each country name once.'
            ),
            readings=(R_COUNTRY_LINK,),
            answer_columns=(AnswerColumn("country", LIT_COL),),
            key_param="pm",
            expand=_one(("pm", "pm", "iri"), ("pm_name", "pmName", "label")),
            anchors=(PM_ANCHOR,),
            hops=3,
            difficulty={"hops": 3, "anchors": 1},
            step_check=StepCheck(
                "path_fans_out",
                fan_out="""SELECT ?product ?country WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> ; pv:hasSupplier ?supplier .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry ?country .
}""",
            ),
        ),
    )


# --------------------------------------------------------------------------
# 4. aggregation


def _having_thresholds(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """Per country, every "at least n products" that keeps a proper, non-empty subset of its suppliers with products."""
    for country, group in _group(rows, "country").items():
        if not _ok(country):
            continue
        counts = [int(r["n"]) for r in group]
        for n in sorted(set(counts)):
            kept = sum(1 for c in counts if c >= n)
            if 1 <= kept < len(counts) and kept <= 50:
                yield {**_country(country), "n": str(n)}


def _top_without_tie(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """Countries with more than top_k products whose top_k-th and next prices differ (no tie at the cut)."""
    k = int((knobs or KNOBS).get("top_k", KNOBS["top_k"]))
    for country, group in _group(rows, "country").items():
        amounts = sorted((float(r["amount"]) for r in group), reverse=True)
        if _ok(country) and len(amounts) > k and amounts[k - 1] > amounts[k]:
            yield _country(country)


_QUANTITY_PROBE = EngineProbe(
    "qlever_sum_over_strings",
    "BOM quantities are xsd:string; QLever's SUM over them without a cast is unbound (the gold casts with xsd:integer)",
    """SELECT (SUM(?quantity) AS ?total) WHERE {
  <$bom> pv:hasBomPart ?part .
  ?part pv:quantity ?quantity .
}""",
)


class Aggregation(QuestionTemplate):
    id = "aggregation"
    description = (
        "Counts and sums over a group (a supplier's total price, a manager's or category's product count); "
        "compositional: grouping with HAVING, ORDER BY / LIMIT without ties, distinct counts and sums across a "
        "bill of materials, and an average over an intersection."
    )
    membership_mode = "n.a."
    variants = (
        cvariant(
            name="supplier_total_price",
            description="The total price of one supplier's hardware products.",
            candidate_query="""
SELECT ?supplier ?supplierName (COUNT(DISTINCT ?product) AS ?n) WHERE {
""" + unique_supplier("supplier") + """
  ?product a pv:Hardware ; pv:hasSupplier ?supplier ; pv:price/pv:amount ?amount .
}
GROUP BY ?supplier ?supplierName
HAVING (COUNT(DISTINCT ?product) >= 2)
ORDER BY ?supplier
""",
            gold_pattern="""SELECT (SUM(?amount) AS ?total) WHERE {
  ?product a pv:Hardware ; pv:hasSupplier <$supplier> ; pv:price ?price .
  ?price pv:amount ?amount .
}""",
            question_pattern=(
                'What is the total price in euros of all hardware products supplied by the supplier "$supplier_name", '
                "adding up one price per product? Give the total as a number."
            ),
            answer_columns=(AnswerColumn("total", NUM_COL),),
            key_param="supplier",
            expand=_one(("supplier", "supplier", "iri"), ("supplier_name", "supplierName", "label")),
            anchors=(supplier_anchor("supplier"),),
            hops=2,
            difficulty={"hops": 2, "anchors": 1, "aggregate": "sum"},
        ),
        cvariant(
            name="pm_product_count",
            description="How many hardware products one person manages (services, which also have managers, not counted).",
            candidate_query="""
SELECT ?pm ?pmName WHERE {
  """ + named_pm("pm") + """
  ?product a pv:Hardware ; pv:hasProductManager ?pm .
}
GROUP BY ?pm ?pmName
ORDER BY ?pm
""",
            gold_pattern="""SELECT (COUNT(DISTINCT ?product) AS ?products) WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> .
}""",
            question_pattern=_PM_COUNT_TEXT,
            readings=(_services(_PM_COUNT_TEXT),),
            answer_columns=(AnswerColumn("products", NUM_COL),),
            key_param="pm",
            expand=_one(("pm", "pm", "iri"), ("pm_name", "pmName", "label")),
            anchors=(PM_ANCHOR,),
            hops=1,
            difficulty={"hops": 1, "anchors": 1, "aggregate": "count"},
        ),
        cvariant(
            name="category_product_count",
            description="How many hardware products belong to one product category.",
            candidate_query="""
SELECT ?category ?categoryName WHERE {
  ?category a pv:ProductCategory ; pv:name ?categoryName .
  ?product a pv:Hardware ; pv:hasCategory ?category .
}
GROUP BY ?category ?categoryName
ORDER BY ?category
""",
            gold_pattern="""SELECT (COUNT(DISTINCT ?product) AS ?products) WHERE {
  ?product a pv:Hardware ; pv:hasCategory <$category> .
}""",
            question_pattern=_CATEGORY_COUNT_TEXT,
            readings=(_services(_CATEGORY_COUNT_TEXT),),
            answer_columns=(AnswerColumn("products", NUM_COL),),
            key_param="category",
            expand=_one(("category", "category", "iri"), ("category_name", "categoryName", "label")),
            anchors=(category_anchor("category"),),
            hops=1,
            difficulty={"hops": 1, "anchors": 1, "aggregate": "count"},
        ),
        cvariant(
            name="country_suppliers_having",
            tier="compositional",
            description="Suppliers of a country supplying at least n hardware products each (GROUP BY with HAVING).",
            candidate_query="""
SELECT ?country ?supplier (COUNT(DISTINCT ?product) AS ?n) WHERE {
  ?supplier a pv:Supplier ; pv:addressCountry ?country .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
}
GROUP BY ?country ?supplier
ORDER BY ?country ?supplier
""",
            gold_pattern="""SELECT ?supplier WHERE {
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
}
GROUP BY ?supplier
HAVING (COUNT(DISTINCT ?product) >= $n)""",
            question_pattern=(
                'Which suppliers with their address in the country "$country" supply at least $n hardware products each? '
                "List each supplier once."
            ),
            readings=(R_COUNTRY_LINK, _inclusive(">=", ">", AT_LEAST)),
            answer_columns=(AnswerColumn("supplier", IRI_COL),),
            key_param="country_key",
            expand=_having_thresholds,
            anchors=(COUNTRY_ANCHOR,),
            hops=2,
            difficulty={"hops": 2, "anchors": 1, "aggregate": "group_having", "grouping_constraints": 1},
            step_check=StepCheck(
                "every_constraint_matters",
                undo=(
                    ("""SELECT DISTINCT ?supplier WHERE {
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
}""", {}),
                    ("""SELECT ?supplier WHERE {
  ?supplier a pv:Supplier .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
}
GROUP BY ?supplier
HAVING (COUNT(DISTINCT ?product) >= $n)""", {}),
                ),
                steps=("at least n products", "country"),
            ),
        ),
        cvariant(
            name="top_priced_in_country",
            tier="compositional",
            description=(
                "The top_k most expensive hardware products whose supplier's address is in a country (ORDER BY / LIMIT; "
                "no tie at the cut)."
            ),
            candidate_query="""
SELECT ?country ?product ?amount WHERE {
  ?supplier a pv:Supplier ; pv:addressCountry ?country .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier ; pv:price/pv:amount ?amount .
}
ORDER BY ?country ?product
""",
            gold_pattern="""SELECT ?product WHERE {
  ?product a pv:Hardware ; pv:hasSupplier ?supplier ; pv:price ?price .
  ?price pv:amount ?amount .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
}
ORDER BY DESC(?amount)
LIMIT 3""",
            question_pattern=(
                'Which three hardware products with a supplier whose address is in the country "$country" have the '
                "highest prices? List the three products."
            ),
            readings=(R_COUNTRY_LINK,),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="country_key",
            expand=_top_without_tie,
            anchors=(COUNTRY_ANCHOR,),
            hops=3,
            difficulty={"hops": 3, "anchors": 1, "aggregate": "order_limit", "grouping_constraints": 1},
            step_check=StepCheck(
                "every_constraint_matters",
                undo=(
                    ("""SELECT ?product WHERE {
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
}""", {}),
                    ("""SELECT ?product WHERE {
  ?product a pv:Hardware ; pv:price ?price .
  ?price pv:amount ?amount .
}
ORDER BY DESC(?amount)
LIMIT 3""", {}),
                ),
                steps=("top three by price", "country"),
            ),
        ),
        cvariant(
            name="bom_distinct_suppliers",
            tier="compositional",
            description="How many different suppliers supply the parts of one bill of materials (distinct count across a join).",
            candidate_query="""
SELECT ?bom ?bomName (COUNT(DISTINCT ?part) AS ?parts) (COUNT(DISTINCT ?supplier) AS ?suppliers) WHERE {
  ?bom a pv:BillOfMaterial ; pv:name ?bomName ; pv:hasBomPart ?part .
  ?part pv:hasPart ?product .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
}
GROUP BY ?bom ?bomName
ORDER BY ?bom
""",
            gold_pattern="""SELECT (COUNT(DISTINCT ?supplier) AS ?suppliers) WHERE {
  <$bom> pv:hasBomPart ?part .
  ?part pv:hasPart ?product .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
}""",
            question_pattern=(
                'How many different suppliers supply the hardware products listed as parts in the bill of materials '
                '"$bom_name"? Give the number.'
            ),
            readings=(R_DISTINCT,),
            answer_columns=(AnswerColumn("suppliers", NUM_COL),),
            key_param="bom",
            expand=_one(("bom", "bom", "iri"), ("bom_name", "bomName", "label")),
            anchors=(BOM_ANCHOR,),
            hops=3,
            difficulty={"hops": 3, "anchors": 1, "aggregate": "count_distinct_join"},
            step_check=StepCheck(
                "join_matters",
                undo=(
                    ("""SELECT (COUNT(DISTINCT ?product) AS ?suppliers) WHERE {
  <$bom> pv:hasBomPart ?part .
  ?part pv:hasPart ?product .
  ?product a pv:Hardware .
}""", {}),
                ),
                steps=("suppliers, not the parts' products, counted",),
            ),
        ),
        cvariant(
            name="bom_total_quantity",
            tier="compositional",
            description=(
                "The total quantity of parts in one bill of materials (sum of string quantities, cast); bills with a "
                "part that has two quantities are left out."
            ),
            candidate_query="""
SELECT ?bom ?bomName (COUNT(DISTINCT ?part) AS ?n) WHERE {
  ?bom a pv:BillOfMaterial ; pv:name ?bomName ; pv:hasBomPart ?part .
  ?part pv:quantity ?quantity .
  FILTER NOT EXISTS {
    ?bom pv:hasBomPart ?doubled .
    ?doubled pv:quantity ?q1 , ?q2 .
    FILTER(?q1 != ?q2)
  }
}
GROUP BY ?bom ?bomName
ORDER BY ?bom
""",
            gold_pattern="""SELECT (SUM(xsd:integer(?quantity)) AS ?total) WHERE {
  <$bom> pv:hasBomPart ?part .
  ?part pv:quantity ?quantity .
}""",
            question_pattern=(
                'What is the total quantity of parts in the bill of materials "$bom_name", adding up the quantities of '
                "all the parts it lists? Give the number."
            ),
            readings=(
                PolarReading(
                    "parts_not_quantities",
                    "the number of parts listed, not the sum of their quantities",
                    polar(
                        r"\badding up the quantities\b",
                        r"\b(?:not|without|rather than|instead of) adding up the quantities\b|\bcounting the parts\b|"
                        r"\bnumber of parts\b",
                    ),
                    pattern="""SELECT (COUNT(DISTINCT ?part) AS ?total) WHERE {
  <$bom> pv:hasBomPart ?part .
}""",
                ),
            ),
            answer_columns=(AnswerColumn("total", NUM_COL),),
            key_param="bom",
            expand=_one(("bom", "bom", "iri"), ("bom_name", "bomName", "label")),
            anchors=(BOM_ANCHOR,),
            hops=2,
            difficulty={"hops": 2, "anchors": 1, "aggregate": "sum"},
            engine_probes=(_QUANTITY_PROBE,),
            step_check=StepCheck(
                "quantities_matter",
                undo=(("""SELECT (COUNT(DISTINCT ?part) AS ?total) WHERE {
  <$bom> pv:hasBomPart ?part .
  ?part pv:quantity ?quantity .
}""", {}),),
                steps=("quantities summed, not parts counted",),
            ),
        ),
        cvariant(
            name="category_country_average_price",
            tier="compositional",
            description="The average price of a category's hardware products whose supplier's address is in a country.",
            candidate_query=TypeMembers.variant("category_country_products").candidate_query,
            gold_pattern="""SELECT (AVG(?amount) AS ?average) WHERE {
  ?product a pv:Hardware ; pv:hasCategory <$category> ; pv:hasSupplier ?supplier ; pv:price ?price .
  ?price pv:amount ?amount .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
}""",
            question_pattern=(
                'What is the average price in euros of the hardware products that belong to the product category '
                '"$category_name" and have a supplier whose address is in the country "$country"? Give the average as '
                "a number, without rounding."
            ),
            readings=(R_COUNTRY_LINK,),
            answer_columns=(AnswerColumn("average", NUM_COL),),
            key_param="category",
            expand=_one(("category", "category", "iri"), ("category_name", "categoryName", "label"), ("country", "country", "country")),
            anchors=(category_anchor("category"), COUNTRY_ANCHOR),
            hops=4,
            difficulty={"hops": 4, "anchors": 2, "aggregate": "avg", "set_operation": "intersection", "populations": 2},
            step_check=StepCheck(
                "every_population_matters",
                undo=(
                    ("""SELECT (AVG(?amount) AS ?average) WHERE {
  ?product a pv:Hardware ; pv:hasSupplier ?supplier ; pv:price ?price .
  ?price pv:amount ?amount .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
}""", {}),
                    ("""SELECT (AVG(?amount) AS ?average) WHERE {
  ?product a pv:Hardware ; pv:hasCategory <$category> ; pv:price ?price .
  ?price pv:amount ?amount .
}""", {}),
                ),
                steps=("category", "country"),
            ),
        ),
    )


# --------------------------------------------------------------------------
# 5. numeric_filter


def _thresholds(group_key: str, value_col: str, op: str, extra: tuple[tuple[str, str, str], ...]):
    """Per group, every threshold taken from its own values that keeps a proper, non-empty subset of at most 50."""

    def keep(x: float, t: float) -> bool:
        return x > t if op == ">" else x < t

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        for group in _group(rows, group_key).values():
            first = group[0]
            labels = [first[col] for _, col, kind in extra if kind == "label"]
            if not _ok(*labels):
                continue
            values = OrderedDict((r[value_col], float(r[value_col])) for r in sorted(group, key=lambda r: float(r[value_col])))
            nums = [float(r[value_col]) for r in group]
            for lexical, t in values.items():
                kept = sum(1 for x in nums if keep(x, t))
                if 1 <= kept < len(nums) and kept <= 50:
                    out = {p: (check_iri(first[c]) if kind == "iri" else first[c]) for p, c, kind in extra}
                    yield {**out, "threshold": lexical}

    return expand


def _two_bounds(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """Per country, (height, width) bounds from its products' values where each bound and the country matter."""
    per = int((knobs or KNOBS).get("max_combo_options", KNOBS["max_combo_options"]))
    everything = [(int(r["height"]), int(r["width"])) for r in rows]
    for country, group in _group(rows, "country").items():
        if not _ok(country):
            continue
        dims = [(int(r["height"]), int(r["width"])) for r in group]
        options = []
        for h in sorted({d[0] for d in dims}):
            for w in sorted({d[1] for d in dims}):
                both = sum(1 for x, y in dims if x > h and y > w)
                only_h = sum(1 for x, y in dims if x > h)
                only_w = sum(1 for x, y in dims if y > w)
                if 1 <= both <= 50 and only_h > both and only_w > both:
                    options.append((h, w))
        taken = 0
        for h, w in _stable(options, f"bounds|{country}"):
            if taken >= per:
                break
            # The country matters too: other countries' products pass both bounds.
            if sum(1 for x, y in everything if x > h and y > w) > sum(1 for x, y in dims if x > h and y > w):
                taken += 1
                yield {**_country(country), "height": str(h), "width": str(w)}


def _references(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """(supplier, reference product of another supplier) where the reference's weight splits the supplier's products."""
    per = int((knobs or KNOBS).get("max_combo_options", KNOBS["max_combo_options"]))
    refs = [r for r in rows if r.get("ref")]
    own = [r for r in rows if r.get("supplier")]
    for supplier, group in _group(own, "supplier").items():
        name = group[0]["supplierName"]
        if not _ok(name):
            continue
        weights = [float(r["weight"]) for r in group]
        fits = [
            r for r in refs
            if r["refSupplier"] != supplier and _ok(r["refName"])
            and 1 <= sum(1 for w in weights if w > float(r["refWeight"])) < len(weights)
        ]
        for r in _stable(fits, f"ref|{supplier}")[:per]:
            yield {"supplier": check_iri(supplier), "supplier_name": name, "ref": check_iri(r["ref"]), "ref_name": r["refName"]}


def _height_pairs(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """Pairs of uniquely named products of one supplier with different heights; which comes first is scrambled."""
    per = int((knobs or KNOBS).get("max_combo_options", KNOBS["max_combo_options"]))
    for supplier, group in _group(rows, "supplier").items():
        named = sorted((r for r in group if _ok(r["productName"])), key=lambda r: r["product"])
        pairs = [(x, y) for x, y in combinations(named, 2) if x["height"] != y["height"]]
        for x, y in _stable(pairs, f"heights|{supplier}")[:per]:
            if hashlib.sha256(f"order|{x['product']}|{y['product']}".encode()).digest()[0] % 2:
                x, y = y, x
            yield {"a": check_iri(x["product"]), "a_name": x["productName"], "b": check_iri(y["product"]), "b_name": y["productName"]}


class NumericFilter(QuestionTemplate):
    id = "numeric_filter"
    description = (
        "Numeric comparisons: a supplier's products above a weight, a manager's products below a price; "
        "compositional: two bounds within a country, heavier than a reference product, the most reliable of three, "
        "and an explicit signed height difference."
    )
    membership_mode = "n.a."
    variants = (
        cvariant(
            name="supplier_heavy_products",
            description="A supplier's hardware products weighing more than a threshold (thresholds from its own weights).",
            candidate_query="""
SELECT ?supplier ?supplierName ?weight WHERE {
""" + unique_supplier("supplier") + """
  ?product a pv:Hardware ; pv:hasSupplier ?supplier ; pv:weight_g ?weight .
}
ORDER BY ?supplier ?product
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasSupplier <$supplier> ; pv:weight_g ?weight .
  FILTER(?weight > $threshold)
}""",
            question_pattern=(
                'Which hardware products supplied by the supplier "$supplier_name" weigh more than $threshold grams? '
                "List each product once."
            ),
            readings=(_inclusive(">", ">=", MORE_THAN),),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="supplier",
            expand=_thresholds("supplier", "weight", ">", (("supplier", "supplier", "iri"), ("supplier_name", "supplierName", "label"))),
            anchors=(supplier_anchor("supplier"),),
            hops=1,
            difficulty={"hops": 1, "anchors": 1, "filter": "numeric"},
        ),
        cvariant(
            name="pm_cheap_products",
            description="A person's hardware products priced below a threshold (thresholds from their own prices).",
            candidate_query="""
SELECT ?pm ?pmName ?amount WHERE {
  """ + named_pm("pm") + """
  ?product a pv:Hardware ; pv:hasProductManager ?pm ; pv:price/pv:amount ?amount .
}
ORDER BY ?pm ?product
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> ; pv:price ?price .
  ?price pv:amount ?amount .
  FILTER(?amount < $threshold)
}""",
            question_pattern=(
                'Which hardware products whose product manager is "$pm_name" cost less than $threshold euros? List each '
                "product once."
            ),
            readings=(_inclusive("<", "<=", LESS_THAN),),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="pm",
            expand=_thresholds("pm", "amount", "<", (("pm", "pm", "iri"), ("pm_name", "pmName", "label"))),
            anchors=(PM_ANCHOR,),
            hops=2,
            difficulty={"hops": 2, "anchors": 1, "filter": "numeric"},
        ),
        cvariant(
            name="taller_and_wider",
            tier="compositional",
            description="Hardware products of a country's suppliers above both a height and a width (two numeric constraints).",
            candidate_query="""
SELECT ?country ?product ?height ?width WHERE {
  ?supplier a pv:Supplier ; pv:addressCountry ?country .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier ; pv:height_mm ?height ; pv:width_mm ?width .
}
ORDER BY ?country ?product
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasSupplier ?supplier ; pv:height_mm ?height ; pv:width_mm ?width .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
  FILTER(?height > $height && ?width > $width)
}""",
            question_pattern=(
                'Which hardware products with a supplier whose address is in the country "$country" are more than '
                "$height millimetres high and more than $width millimetres wide? List each product once."
            ),
            readings=(R_COUNTRY_LINK, _inclusive(">", ">=", MORE_THAN)),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="country_key",
            expand=_two_bounds,
            anchors=(COUNTRY_ANCHOR,),
            hops=2,
            difficulty={"hops": 2, "anchors": 1, "filter": "numeric", "constraints": 2},
            step_check=StepCheck(
                "every_constraint_matters",
                undo=(
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasSupplier ?supplier ; pv:width_mm ?width .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
  FILTER(?width > $width)
}""", {}),
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasSupplier ?supplier ; pv:height_mm ?height .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
  FILTER(?height > $height)
}""", {}),
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:height_mm ?height ; pv:width_mm ?width .
  FILTER(?height > $height && ?width > $width)
}""", {}),
                ),
                steps=("height bound", "width bound", "country"),
            ),
        ),
        cvariant(
            name="heavier_than_reference",
            tier="compositional",
            description="A supplier's hardware products heavier than a reference product of another supplier.",
            candidate_query="""
SELECT ?supplier ?supplierName ?weight ?ref ?refName ?refWeight ?refSupplier WHERE {
  {
""" + unique_supplier("supplier") + """
    ?product a pv:Hardware ; pv:hasSupplier ?supplier ; pv:weight_g ?weight .
  }
  UNION
  {
""" + unique_hw("ref") + """
    ?ref pv:weight_g ?refWeight ; pv:hasSupplier ?refSupplier .
  }
}
ORDER BY ?supplier ?ref
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  <$ref> a pv:Hardware ; pv:weight_g ?refWeight .
  ?product a pv:Hardware ; pv:hasSupplier <$supplier> ; pv:weight_g ?weight .
  FILTER(?weight > ?refWeight)
}""",
            question_pattern=(
                'Which hardware products supplied by the supplier "$supplier_name" weigh more than the hardware product '
                '"$ref_name"? List each product once.'
            ),
            readings=(_inclusive(">", ">=", WEIGH_MORE_THAN),),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="supplier",
            expand=_references,
            anchors=(supplier_anchor("supplier"), hardware_anchor("ref")),
            hops=2,
            difficulty={"hops": 2, "anchors": 2, "comparison": True, "filter": "numeric"},
            step_check=StepCheck(
                "every_constraint_matters",
                undo=(
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasSupplier <$supplier> .
}""", {}),
                    ("""SELECT DISTINCT ?product WHERE {
  <$ref> a pv:Hardware ; pv:weight_g ?refWeight .
  ?product a pv:Hardware ; pv:weight_g ?weight .
  FILTER(?weight > ?refWeight)
}""", {}),
                ),
                steps=("heavier than the reference", "supplier"),
            ),
        ),
        COMPARISON_VARIANTS[("reliability", "winner")],
        cvariant(
            name="height_difference",
            tier="compositional",
            description="The height of one hardware product minus the height of another (explicit signed difference).",
            candidate_query="""
SELECT ?supplier ?product ?productName ?height WHERE {
""" + unique_hw("product") + """
  ?product pv:hasSupplier ?supplier ; pv:height_mm ?height .
}
ORDER BY ?supplier ?product
""",
            gold_pattern="""SELECT ?difference WHERE {
  <$a> a pv:Hardware ; pv:height_mm ?heightA .
  <$b> a pv:Hardware ; pv:height_mm ?heightB .
  BIND(?heightA - ?heightB AS ?difference)
}""",
            question_pattern=(
                'What is the height in millimetres of the hardware product "$a_name" minus the height in millimetres of '
                'the hardware product "$b_name"? The result may be zero or negative. Give the number.'
            ),
            answer_columns=(AnswerColumn("difference", NUM_COL),),
            key_param="a",
            expand=_height_pairs,
            anchors=(hardware_anchor("a"), hardware_anchor("b")),
            hops=1,
            difficulty={"hops": 1, "anchors": 2, "comparison": True, "signed": True},
            step_check=StepCheck(
                "both_terms_matter",
                undo=(
                    ("SELECT ?difference WHERE { <$a> a pv:Hardware ; pv:height_mm ?difference . }", {}),
                    ("SELECT ?difference WHERE { <$b> a pv:Hardware ; pv:height_mm ?heightB . BIND(0 - ?heightB AS ?difference) }", {}),
                ),
                steps=("minus b's height", "a's height"),
            ),
        ),
    )


# --------------------------------------------------------------------------
# 6. absence_negation


def _a_not_b(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """(person, category A, category B) where A-not-B is non-empty and both the negation and A itself matter."""
    per = int((knobs or KNOBS).get("max_combo_options", KNOBS["max_combo_options"]))
    names = {r["category"]: r["categoryName"] for r in rows}
    for pm_iri, group in _group(rows, "pm").items():
        pm_name = group[0]["pmName"]
        if not _ok(pm_name):
            continue
        cats: dict[str, set[str]] = OrderedDict()
        for r in group:
            cats.setdefault(r["product"], set()).add(r["category"])
        options = []
        for a in sorted(set(names) & {c for cs in cats.values() for c in cs}):
            for b in sorted(names):
                if a == b or not _ok(names[a], names[b]):
                    continue
                gold = {p for p, cs in cats.items() if a in cs and b not in cs}
                in_a = {p for p, cs in cats.items() if a in cs}
                not_b = {p for p, cs in cats.items() if b not in cs}
                if 1 <= len(gold) <= 50 and in_a != gold and not_b != gold:
                    options.append((a, b))
        for a, b in _stable(options, f"anotb|{pm_iri}")[:per]:
            yield {
                "pm": check_iri(pm_iri), "pm_name": pm_name,
                "cat_a": check_iri(a), "cat_a_name": names[a], "cat_b": check_iri(b), "cat_b_name": names[b],
            }


def _all_eligible(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """(country, service) with 1..pop_max suppliers whose products are all eligible and one whose products are partly so."""
    cap = int((knobs or KNOBS).get("pop_max", KNOBS["pop_max"]))
    groups: OrderedDict[tuple[str, str], list[PoolRow]] = OrderedDict()
    for r in rows:
        groups.setdefault((r["country"], r["service"]), []).append(r)
    for (country, service), group in groups.items():
        full = sum(1 for r in group if int(r["covered"]) == int(r["total"]))
        partial = sum(1 for r in group if 0 < int(r["covered"]) < int(r["total"]))
        if 1 <= full <= cap and partial and _ok(country, group[0]["serviceName"]):
            yield {**_country(country), "service": check_iri(service), "service_name": group[0]["serviceName"]}


_NESTED_SCOPE_PROBE = EngineProbe(
    "qlever_nested_exists_scope",
    "the natural query binds the service once at the top (VALUES) and uses it two FILTER NOT EXISTS levels down; "
    "QLever leaves it unbound there, so the inner test reads 'eligible for some service' and the query returns the "
    "suppliers all of whose products are eligible for any service, not only for the one named (on the store, the "
    "probe's rows equal that reading's rows for every tagged question)",
    """SELECT DISTINCT ?supplier WHERE {
  VALUES ?service { <$service> }
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
  ?any a pv:Hardware ; pv:hasSupplier ?supplier .
  FILTER NOT EXISTS {
    ?product a pv:Hardware ; pv:hasSupplier ?supplier .
    FILTER NOT EXISTS { ?service pv:eligibleFor ?product . }
  }
}""",
)


class AbsenceNegation(QuestionTemplate):
    id = "absence_negation"
    description = (
        "Absence: a supplier's products without a service, a manager's products without a compatible product, a "
        "country's products without a reliability index; compositional: suppliers lacking a category, products "
        "outside every area of a manager's expertise, suppliers all of whose products are eligible for a service, "
        "and a category difference."
    )
    membership_mode = "n.a."
    variants = (
        cvariant(
            name="supplier_products_without_service",
            description="A supplier's hardware products eligible for no service.",
            candidate_query="""
SELECT ?supplier ?supplierName (COUNT(DISTINCT ?product) AS ?n) WHERE {
""" + unique_supplier("supplier") + """
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
  FILTER NOT EXISTS { ?service pv:eligibleFor ?product . }
  FILTER EXISTS { ?covered a pv:Hardware ; pv:hasSupplier ?supplier . ?anyService pv:eligibleFor ?covered . }
}
GROUP BY ?supplier ?supplierName
ORDER BY ?supplier
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasSupplier <$supplier> .
  FILTER NOT EXISTS { ?service a pv:Service ; pv:eligibleFor ?product . }
}""",
            question_pattern=(
                'Which hardware products supplied by the supplier "$supplier_name" are not eligible for any service? '
                "List each product once."
            ),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="supplier",
            expand=_one(("supplier", "supplier", "iri"), ("supplier_name", "supplierName", "label")),
            anchors=(supplier_anchor("supplier"),),
            hops=2,
            difficulty={"hops": 2, "anchors": 1, "negation": 1},
        ),
        cvariant(
            name="pm_products_without_compatible",
            description="A person's hardware products with no compatible product recorded.",
            candidate_query="""
SELECT ?pm ?pmName (COUNT(DISTINCT ?product) AS ?n) WHERE {
  """ + named_pm("pm") + """
  ?product a pv:Hardware ; pv:hasProductManager ?pm .
  FILTER NOT EXISTS { ?product pv:compatibleProduct ?other . }
}
GROUP BY ?pm ?pmName
ORDER BY ?pm
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> .
  FILTER NOT EXISTS { ?product pv:compatibleProduct ?other . }
}""",
            question_pattern=(
                'Which hardware products whose product manager is "$pm_name" have no compatible product recorded? List '
                "each product once."
            ),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="pm",
            expand=_one(("pm", "pm", "iri"), ("pm_name", "pmName", "label")),
            anchors=(PM_ANCHOR,),
            hops=2,
            difficulty={"hops": 2, "anchors": 1, "negation": 1},
        ),
        cvariant(
            name="country_products_without_reliability",
            description="Hardware products of a country's suppliers with no reliability index recorded.",
            candidate_query="""
SELECT ?country (COUNT(DISTINCT ?product) AS ?n) WHERE {
  ?supplier a pv:Supplier ; pv:addressCountry ?country .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
  FILTER NOT EXISTS { ?product pv:reliabilityIndex ?r . }
}
GROUP BY ?country
HAVING (COUNT(DISTINCT ?product) <= 50)
ORDER BY ?country
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
  FILTER NOT EXISTS { ?product pv:reliabilityIndex ?reliability . }
}""",
            question_pattern=(
                'Which hardware products with a supplier whose address is in the country "$country" have no reliability '
                "index recorded? List each product once."
            ),
            readings=(R_COUNTRY_LINK,),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="country_key",
            expand=_one(("country", "country", "country")),
            anchors=(COUNTRY_ANCHOR,),
            hops=2,
            difficulty={"hops": 2, "anchors": 1, "negation": 1},
        ),
        cvariant(
            name="country_suppliers_without_category",
            tier="compositional",
            description=(
                "Suppliers of a country that supply hardware products but none of a category (negation over a join; "
                "suppliers with no products at all are not counted)."
            ),
            candidate_query="""
SELECT ?country ?category ?categoryName (COUNT(DISTINCT ?supplier) AS ?n) WHERE {
  ?category a pv:ProductCategory ; pv:name ?categoryName .
  ?supplier a pv:Supplier ; pv:addressCountry ?country .
  ?any a pv:Hardware ; pv:hasSupplier ?supplier .
  FILTER NOT EXISTS { ?product a pv:Hardware ; pv:hasSupplier ?supplier ; pv:hasCategory ?category . }
  FILTER EXISTS {
    ?other a pv:Supplier ; pv:addressCountry ?country .
    ?inCategory a pv:Hardware ; pv:hasSupplier ?other ; pv:hasCategory ?category .
  }
}
GROUP BY ?country ?category ?categoryName
HAVING (COUNT(DISTINCT ?supplier) >= $pop_min && COUNT(DISTINCT ?supplier) <= $pop_max)
ORDER BY ?country ?category
""",
            gold_pattern="""SELECT DISTINCT ?supplier WHERE {
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
  ?any a pv:Hardware ; pv:hasSupplier ?supplier .
  FILTER NOT EXISTS { ?product a pv:Hardware ; pv:hasSupplier ?supplier ; pv:hasCategory <$category> . }
}""",
            question_pattern=(
                'Which suppliers with their address in the country "$country" supply hardware products, but none of the '
                'product category "$category_name"? List each supplier once.'
            ),
            readings=(
                R_COUNTRY_LINK,
                _productless(
                    "suppliers that supply nothing at all also count as supplying nothing of the category",
                    r"\bsupply hardware products, but none\b",
                ),
            ),
            answer_columns=(AnswerColumn("supplier", IRI_COL),),
            key_param="country_key",
            expand=_one(("country", "country", "country"), ("category", "category", "iri"), ("category_name", "categoryName", "label")),
            anchors=(COUNTRY_ANCHOR, category_anchor("category")),
            hops=3,
            difficulty={"hops": 3, "anchors": 2, "negation": 1, "populations": 2},
            step_check=StepCheck(
                "every_population_matters",
                undo=(
                    ("""SELECT DISTINCT ?supplier WHERE {
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
  ?any a pv:Hardware ; pv:hasSupplier ?supplier .
}""", {}),
                    ("""SELECT DISTINCT ?supplier WHERE {
  ?supplier a pv:Supplier .
  ?any a pv:Hardware ; pv:hasSupplier ?supplier .
  FILTER NOT EXISTS { ?product a pv:Hardware ; pv:hasSupplier ?supplier ; pv:hasCategory <$category> . }
}""", {}),
                ),
                steps=("none of the category", "country"),
            ),
        ),
        cvariant(
            name="products_outside_expertise",
            tier="compositional",
            description=(
                "A person's hardware products in none of the categories that are that person's areas of expertise "
                "(negation over a join)."
            ),
            candidate_query="""
SELECT ?pm ?pmName (COUNT(DISTINCT ?product) AS ?n) WHERE {
  """ + named_pm("pm") + """
  ?product a pv:Hardware ; pv:hasProductManager ?pm .
  FILTER NOT EXISTS { ?product pv:hasCategory ?c . ?pm pv:areaOfExpertise ?c . }
  FILTER EXISTS { ?covered a pv:Hardware ; pv:hasProductManager ?pm ; pv:hasCategory ?ce . ?pm pv:areaOfExpertise ?ce . }
}
GROUP BY ?pm ?pmName
HAVING (COUNT(DISTINCT ?product) <= 50)
ORDER BY ?pm
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> .
  FILTER NOT EXISTS { ?product pv:hasCategory ?category . <$pm> pv:areaOfExpertise ?category . }
}""",
            question_pattern=_OUTSIDE_EXPERTISE_TEXT,
            readings=(
                PolarReading(
                    "some_category_outside",
                    "products with at least one category outside the expertise, not products with none inside it",
                    polar(
                        r"\bbelong to none of the product categories that are areas of expertise\b",
                        r"(?:\bnot|n't) belong to none\b|\bthat are not areas of expertise\b|"
                        r"\bsome (?:of the )?product categor(?:y|ies)\b|\bat least one product category\b",
                    ),
                    pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> ; pv:hasCategory ?category .
  FILTER NOT EXISTS { <$pm> pv:areaOfExpertise ?category . }
}""",
                ),
                # A service with no category is in none of the expertise categories (design log: declared).
                _services(_OUTSIDE_EXPERTISE_TEXT),
            ),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="pm",
            expand=_one(("pm", "pm", "iri"), ("pm_name", "pmName", "label")),
            anchors=(PM_ANCHOR,),
            hops=3,
            difficulty={"hops": 3, "anchors": 1, "negation": 1},
            step_check=StepCheck(
                "every_constraint_matters",
                undo=(
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> .
}""", {}),
                    # The product-manager condition dropped: every hardware product outside that person's expertise.
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware .
  FILTER NOT EXISTS { ?product pv:hasCategory ?category . <$pm> pv:areaOfExpertise ?category . }
}""", {}),
                ),
                steps=("outside the expertise", "product manager"),
            ),
        ),
        cvariant(
            name="suppliers_all_eligible",
            tier="compositional",
            description=(
                "Suppliers of a country that supply hardware products, every one of them eligible for a service "
                "(universal quantification: a negation nested in a negation)."
            ),
            # Counted per supplier by aggregation: QLever does not see a variable of the outer query two
            # FILTER NOT EXISTS levels down (engine probe qlever_nested_exists_scope), so the pool avoids nesting.
            candidate_query="""
SELECT ?country ?service ?serviceName ?supplier (COUNT(DISTINCT ?product) AS ?total) (COUNT(DISTINCT ?eligible) AS ?covered) WHERE {
  ?service a pv:Service ; pv:name ?serviceName .
  ?supplier a pv:Supplier ; pv:addressCountry ?country .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
  OPTIONAL { ?service pv:eligibleFor ?product . BIND(?product AS ?eligible) }
}
GROUP BY ?country ?service ?serviceName ?supplier
ORDER BY ?country ?service ?supplier
""",
            gold_pattern="""SELECT DISTINCT ?supplier WHERE {
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
  ?any a pv:Hardware ; pv:hasSupplier ?supplier .
  FILTER NOT EXISTS {
    ?product a pv:Hardware ; pv:hasSupplier ?supplier .
    FILTER NOT EXISTS { <$service> pv:eligibleFor ?product . }
  }
}""",
            question_pattern=(
                'Which suppliers with their address in the country "$country" supply hardware products, all of which '
                'are eligible for the service "$service_name"? List each supplier once.'
            ),
            readings=(
                R_COUNTRY_LINK,
                PolarReading(
                    "some_not_all",
                    "suppliers with at least one eligible product, not only those whose products are all eligible",
                    polar(
                        r"\bproducts, all of which are eligible for the service\b",
                        r"\bnot all of which\b|\ball of which are (?:not |in)eligible\b|"
                        r"\b(?:none|some|any|one|at least one) of which\b",
                    ),
                    pattern="""SELECT DISTINCT ?supplier WHERE {
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
  ?product a pv:Hardware ; pv:hasSupplier ?supplier .
  <$service> pv:eligibleFor ?product .
}""",
                ),
                # Design log: declared (a supplier with no hardware products has none that is not eligible).
                _productless(
                    "suppliers that supply no hardware products also count, having no product that is not eligible",
                    r"\bsupply hardware products, all of which\b",
                    _PRODUCTLESS_NEG + r"|\bif any\b",
                ),
            ),
            answer_columns=(AnswerColumn("supplier", IRI_COL),),
            key_param="country_key",
            expand=lambda rows, knobs=None: _all_eligible(rows, knobs),
            anchors=(COUNTRY_ANCHOR, SERVICE_ANCHOR),
            engine_probes=(_NESTED_SCOPE_PROBE,),
            hops=3,
            difficulty={"hops": 3, "anchors": 2, "negation": 2, "quantifier": "universal"},
            step_check=StepCheck(
                "every_constraint_matters",
                undo=(
                    ("""SELECT DISTINCT ?supplier WHERE {
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
  ?any a pv:Hardware ; pv:hasSupplier ?supplier .
}""", {}),
                    ("""SELECT DISTINCT ?supplier WHERE {
  ?supplier a pv:Supplier .
  ?supplier pv:addressCountry $country_lit .
  ?any a pv:Hardware ; pv:hasSupplier ?supplier .
  FILTER NOT EXISTS {
    ?product a pv:Hardware ; pv:hasSupplier ?supplier .
    <$service> pv:eligibleFor ?product .
  }
}""", {}),
                    # The country dropped (code review of a11aa47): suppliers anywhere whose products are all eligible.
                    ("""SELECT DISTINCT ?supplier WHERE {
  ?supplier a pv:Supplier .
  ?any a pv:Hardware ; pv:hasSupplier ?supplier .
  FILTER NOT EXISTS {
    ?product a pv:Hardware ; pv:hasSupplier ?supplier .
    FILTER NOT EXISTS { <$service> pv:eligibleFor ?product . }
  }
}""", {}),
                ),
                steps=("all eligible for the service", "inner negation", "country"),
            ),
        ),
        cvariant(
            name="category_a_not_b",
            tier="compositional",
            description="A person's hardware products in one category but not in another (set difference).",
            candidate_query="""
SELECT ?pm ?pmName ?product ?category ?categoryName WHERE {
  """ + named_pm("pm") + """
  ?product a pv:Hardware ; pv:hasProductManager ?pm ; pv:hasCategory ?category .
  ?category a pv:ProductCategory ; pv:name ?categoryName .
}
ORDER BY ?pm ?product ?category
""",
            gold_pattern="""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> ; pv:hasCategory <$cat_a> .
  FILTER NOT EXISTS { ?product pv:hasCategory <$cat_b> . }
}""",
            question_pattern=(
                'Which hardware products whose product manager is "$pm_name" belong to the product category '
                '"$cat_a_name" but not to the product category "$cat_b_name"? List each product once.'
            ),
            answer_columns=(AnswerColumn("product", IRI_COL),),
            key_param="pm",
            expand=_a_not_b,
            anchors=(PM_ANCHOR, category_anchor("cat_a"), category_anchor("cat_b")),
            hops=2,
            difficulty={"hops": 2, "anchors": 3, "set_operation": "difference", "negation": 1, "populations": 3},
            step_check=StepCheck(
                "every_population_matters",
                undo=(
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> ; pv:hasCategory <$cat_a> .
}""", {}),
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasProductManager <$pm> .
  FILTER NOT EXISTS { ?product pv:hasCategory <$cat_b> . }
}""", {}),
                    # The product manager dropped (code review of a11aa47): the difference over every hardware product.
                    ("""SELECT DISTINCT ?product WHERE {
  ?product a pv:Hardware ; pv:hasCategory <$cat_a> .
  FILTER NOT EXISTS { ?product pv:hasCategory <$cat_b> . }
}""", {}),
                ),
                steps=("not in category b", "in category a", "product manager"),
            ),
        ),
    )


TEMPLATES = (TypeMembers, OneHopAttribute, MultiHopPath, Aggregation, NumericFilter, AbsenceNegation)
TEMPLATES_BY_ID = {t.id: t for t in TEMPLATES}

FORBIDDEN_NAMES = (
    "Text2SPARQL", "DBpedia", "AKSW", "eccenca", "company.org", "ld.company.org", "prod-vocab", "prod-instances",
    "Wikidata", "CK26", "QLever", "SPARQL", "RDF", "OWL",
)

NOTES = {
    "anchors": (
        "Anchors come from instances and links in the prod-instances graph and their counts (suppliers by products, "
        "countries by suppliers whose address names them, product managers by products, categories, services and bills "
        "of materials by members), never from the prod-vocab axioms. Each is named by a name unique for its kind "
        "(checked per question): hardware product, supplier, product category, service, department and bill of "
        "materials by pv:name; a person by pv:name among employees and managers; a country by the address string."
    ),
    "one_anchor_per_template": (
        "No anchor entity appears in two questions of one template (both tiers, both splits); see anchor_reuse."
    ),
    "prior_dev_anchors": (
        "This set has had no calibration round, so no earlier dev anchors exist to avoid. The rule (no test item names "
        "an anchor of a calibration dev item of the same variant) applies to its calibration: a later round must pin "
        "this set's questions_sha256 in the profile's prior_dev_rounds, and its test items then avoid this set's dev "
        "anchors of the same variant. Within this set the one-anchor-per-template rule already keeps every test item "
        "off every dev anchor of its template."
    ),
    "union_graph": (
        "Golds use no GRAPH clause and run over the union default graph (as the challenge's own gold queries do); every "
        "subject is typed with an instance class, so the vocabulary graph's labels and types never reach an answer. "
        "graph_scope discrimination is therefore not applicable."
    ),
    "managers": (
        "Managers are typed only pv:Manager; Manager rdfs:subClassOf Employee is a vocabulary axiom, not data. "
        "Questions about employees say 'not counting managers' (reading managers_as_employees)."
    ),
    "services": (
        "Services are products too (product managers, prices, responsible departments, one category); questions say "
        "'hardware products', and the reading services_as_products is declared on the two count lookups "
        "(pm_product_count, category_product_count) and on products_outside_expertise. Run on every gold of the set "
        "that types its answer as hardware (232 questions), it changes two answers, both declared with the clause "
        "present: a category count and one products_outside_expertise question (corporate-absence_negation-030)."
    ),
    "folded_readings": (
        "Two readings found by the step-check audit of a11aa47 are declared by the study lead's ruling on issue "
        "20: services_as_products on products_outside_expertise (changes 1 of 8 answers, "
        "corporate-absence_negation-030) and productless_suppliers on suppliers_all_eligible (suppliers with no "
        "hardware products pass 'all of which are eligible' vacuously; changes 1 of 8 answers, "
        "corporate-absence_negation-035); every text of both variants carries the clause. Declaring them added one key "
        "to the alternative_readings field of those 16 records; no text, gold or other field changed (seed unchanged)."
    ),
    "unnamed_product_managers": (
        "Three product managers have no type and no name in the data; they are never anchors, and golds that read a "
        "product manager by name or type leave them out."
    ),
    "countries": (
        "23 suppliers give a country in their address but have no pv:country link (Spain 9, Ukraine 7, Belgium 3, "
        "North Macedonia 2, Malaysia 1, Palestine 1); countries are read from the address string and questions say so "
        "(reading country_link)."
    ),
    "compatibility": "pv:compatibleProduct is symmetric in the data (no one-way pair), so two steps lead back to the anchor.",
    "reliability": (
        "pv:reliabilityIndex is one value per supplier country: every product of a supplier in that country carries it "
        "(220 suppliers) or none of a supplier's products has one (26 suppliers); the reliability comparison therefore "
        "names three products whose suppliers are in three countries."
    ),
    "product_names": (
        "Hardware product names are built from the names of their categories (\"Inductor Switch\" is in Inductor and "
        "Switch), so category questions can partly be answered from a product's name; the gold reads the category links."
    ),
    "nested_negation": (
        "QLever does not pass a variable bound in the outer query two FILTER NOT EXISTS levels down. A query with VALUES "
        "?service at the top and ?service in the inner FILTER NOT EXISTS returns the suppliers all of whose products "
        "are eligible for some service, any service: measured on the store for the 8 universal questions, its rows "
        "equal that reading's rows each time, from 2 of 2 suppliers with products (Cuba, Mozambique) to 23 of 32 "
        "(United States, corporate-absence_negation-035), where the golds have 1 each. The universal gold names the "
        "service by its IRI, and the natural form is an engine probe (qlever_nested_exists_scope)."
    ),
    "bom_quantities": (
        "pv:quantity values are xsd:string; golds cast with xsd:integer, and QLever's SUM without the cast is unbound "
        "(engine probe qlever_sum_over_strings). Two BOM parts (in two bills of materials) carry two quantities each; "
        "their bills never get a quantity question."
    ),
    "decimals": (
        "QLever stores pv:amount and pv:reliabilityIndex as doubles, so sums and averages differ from exact decimal "
        "arithmetic in the last digits only, inside the scorer's tolerance (rel 1e-6); no gold rounds, so the held-out "
        "question 32 issue (ROUND over doubles) cannot arise."
    ),
    "comparisons_of_three": (
        "With comparison_design 'winner' (the default and the declared design, see declared_design), "
        "cheapest_of_three and most_reliable_of_three ask which of three products is best, so only the winner's drop "
        "can change the answer. The settings 'ranking' (each product with its rank) and 'middle' (the product between "
        "the other two) redesign both so that dropping any of the three changes the answer; they are not used."
    ),
    "declared_design": {
        "comparisons_of_three": {
            "design": "winner",
            "variants": ["numeric_filter/most_reliable_of_three", "one_hop_attribute/cheapest_of_three"],
            "ruling": (
                "study lead, design log: keep 'winner' (the comparison_design default) for the "
                "three-product comparisons; the winner and the runner-up that binds are load-bearing (dropping the "
                "winner changes the answer; the losing products are present so that the comparison is among three, "
                "and the runner-up binds the winner's margin)"
            ),
            "load_bearing": (
                "as checked on the store for every question (step check winner_matters, mode 'some', undo order: drop "
                "c, b, a): the three compared values are pairwise distinct, so the winner beats the runner-up "
                "strictly; the gold over the other two products with the winner dropped gives other rows (that answer is "
                "the runner-up, by the gold's ORDER BY / LIMIT 1 over the two left; this is not checked separately); "
                "the gold with the runner-up dropped and the gold with the third "
                "product dropped each give the gold's rows. Exactly this pattern is required, and any other fails "
                "generation. The runner-up is therefore not load-bearing in the drop-one sense: its own drop leaves "
                "the answer unchanged, and it binds only as the answer the winner's drop gives. No size of the "
                "winner's margin is checked. The winner's mention slot is balanced within each (variant, split)."
            ),
        },
        "paths": {
            "design": "fan_out",
            "variants": [
                "multi_hop_path/bom_part_suppliers", "multi_hop_path/pm_supplier_countries",
                "multi_hop_path/supplier_compatible_suppliers", "multi_hop_path/two_step_compatible",
            ],
            "step_check": (
                "per question, a fan-out query lists (entity at the anchor end of the path, answer item); the path "
                "matters when at least two such entities lead to answer items and no single one of them accounts for "
                "every answer item (rejection 'step_not_load_bearing' otherwise). No undo drops a hop."
            ),
            "precedent": (
                "the round-2 path (hop) variants of the other graphs use the same fan-out check (StepCheck.fan_out): "
                "UniProt alone (feat/qgen-uniprot 0b247dd, harness/qgen/uniprot.py: organism_reactions, "
                "organism_distinct_reactions, proteins_reacting_with_compound, disease_genes); Rhea alone "
                "(feat/qgen 18e0d54, harness/qgen/compositional_r2.py: ec_compounds, microspecies_reactions, "
                "reaction_count_below_class) or with one undo that is the variant's round-1 gold, the question before "
                "its hop step (compounds_of_class, compound_reaction_ec_numbers, opposite_side_compounds, "
                "partner_count, descendants_by_charge). No graph drops each hop by an undo; this set has no round 1, "
                "so there is no round-1 gold to undo to."
            ),
        },
    },
    "ties": (
        "ORDER BY / LIMIT and comparison questions are drawn only where no tie decides the answer (top_k-th and next "
        "value differ; compared values pairwise distinct), the issue of held-out questions 29, 46 and 50."
    ),
    "no_cross_graph_join": (
        "Rhea and UniProt have a seventh template joining named data graphs; this store has one data graph and the "
        "vocabulary graph, and questions never read the vocabulary, so the kind has no counterpart here."
    ),
}

# The wording of shared manifest rules for this graph (the shared wording describes the biology graphs).
RULE_TEXTS = {
    "step_checks": (
        "every compositional question's steps must matter, checked on the store when it is drawn (rejection "
        "'step_not_load_bearing'): populations and constraints - every population, bound, negation and top-k cut the "
        "question names is a step, and the gold with each one dropped in turn must change the rows (undo patterns, all "
        "must change; each named in undo_steps); scope conditions that matter for some questions only (the anchor left "
        "out, directly compatible products counted, suppliers that supply hardware products) are alternative readings "
        "instead, whose clause the text must carry; comparisons of three - the compared values are pairwise distinct, "
        "and either dropping any of the three changes the answer (mode 'every') or, in the winner form, dropping the "
        "winner and only the winner does (mode 'some', the declared design by the study lead's ruling in the design log; "
        "see profile.notes.declared_design); "
        "declared_steps gives each variant's mode and steps; signed differences - each of the two terms alone gives "
        "another value; aggregates over a bill of materials - counting the parts' products instead of their suppliers, "
        "or the parts instead of summing their quantities, gives another value; paths - the answer comes through at "
        "least two entities at the anchor end of the path and no single one of them accounts for every answer item "
        "(fan-out)"
    ),
    "alternative_readings": (
        "for every question each alternative reading suited to this graph (the country read through the supplier's "
        "country link rather than its address, managers counted as employees, services counted as products, a bound "
        "read inclusively or strictly, the anchor counted among its own answers, a supplier counted once per part, "
        "either product rather than both, the number of parts rather than their quantities, suppliers with no products, "
        "a product with some category outside an expertise, some rather than all products eligible) is run on the "
        "store; if it changes the answer, the question text must carry the clause that rules it out, matched with its "
        "polarity: a negated or reversed wording anywhere in the text fails the clause, and quoted names are blanked "
        "before matching; the services clause is positive-only, the variant's question form exactly (names blanked), "
        "so any added wording fails it. Otherwise generation fails"
    ),
    "discrimination": (
        "direct_vs_transitive and graph_scope do not apply to this graph: no question reads a class hierarchy, and the "
        "golds use no GRAPH clause (they run over the union default graph); both are null for every question"
    ),
}

# Fields that name the benchmark or its store ("text2sparql-ck26", "Text2SPARQL 2026 corporate graph"), so no agent may
# see them: the record's graph and the manifest's store. The runner shows the agent a fixed system prompt and, as the
# user message, the question text alone (harness/agent/prompts.user_message, fed by scripts/run_pilot.attempt_question,
# which copies id, base_question_id, template, split and the text; feat/e0.1-skeleton d1a84f7); the checker loop's
# feedback renders check results only (feat/checker-loop 89e1c2d, harness/checkloop/feedback.py).
AGENT_HIDDEN_FIELDS = {"record": ("graph",), "manifest": ("store",)}

PROFILE = Profile(
    graph_id=GRAPH_ID,
    id_prefix=ID_PREFIX,
    templates=TEMPLATES,
    store_identity={
        "engine": "QLever 9bebc60",
        "image": "qlever-9bebc60c51.sif",
        "store": "text2sparql-ck26 (Text2SPARQL 2026 corporate graph, AKSW/text2sparql.aksw.org at 935cb5a)",
    },
    two_tier=True,
    lookup_share=1 / 3,
    knobs=KNOBS,
    notes=NOTES,
    forbid_database_names=True,
    # No answer equals or nests in another anywhere in the set (as UniProt): no dev/test leak across templates either.
    global_answer_uniqueness=True,
    version=GENERATOR_VERSION,
    forbidden_names=FORBIDDEN_NAMES,
    unique_anchors_per_template=True,
    require_step_checks=True,
    resolve_variants=resolve_comparisons,
    rule_texts=RULE_TEXTS,
)

__all__ = ["GRAPH_ID", "ID_PREFIX", "PROFILE", "TEMPLATES", "TEMPLATES_BY_ID"]
