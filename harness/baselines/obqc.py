"""OBQC as published (design log, AC4; design ruling 3 of change-log entry 59): the
ontology-based query checker of Allemang and Sequeda, "Increasing the LLM Accuracy for
Question Answering: Ontologies to the Rescue!", arXiv:2405.11706v1 (``obqc`` in the
proposal), as a checker arm of our harness.

The paper's checker (section 3) turns the query's basic graph patterns into RDF, with
every variable replaced by a resource of a reserved namespace (``qq:``), builds an RDFLib
conjunctive graph of two named graphs, ``:query`` and ``:ontology``, and runs one SPARQL
query per rule; each result row fills the rule's English template. All eight rules run
here as the paper prints them (``PRINTED``), on the same kind of graph:

* rules 1-6 (section 3.1: Domain, Range, Double Range, Double Domain, Domain Range,
  Incorrect Property) read the query's triples; rules 7-8 (section 3.2: IRI Output,
  Subject Output) read them with the SELECT clause's variables marked ``?v a qq:Variable``;
* ``:ontology`` holds the curated source's axioms the rules read (property types, domains,
  ranges; a union domain or range is a blank node, which ``ISIRI`` drops, as in the paper);
* ``rdfs:subClassOf*`` reads the default graph, which holds the CURATED source's own
  subclass axioms and nothing else (the ruling; the paper does not say which graph the
  path reads): one way (the query's class must lie at or below the declared one) and
  closed world (classes not linked by subclass axioms are incompatible), as printed;
* static only: no store is read (the footprint is off), so a constant's classes come only
  from type triples written in the query.

Three edits to the printed text, each documented in
results/judge-pilot-20261019/obqc-deviations.md and checked by the tests:
``SELECT *`` replaces each head, so every template variable is bound (the Range template
names ``{o}``, which its head omits; without DISTINCT the rows are the same solutions);
the IRI Output rule's second block reads ``GRAPH :ontology``, not ``GRAPH :query`` (the
prose, "if a predicate has a specified range", and the rule's 18.40% share of invocations
in Table 3 rule out the printed graph, where no range is ever stated; the printed reading
is kept as ``iri_output_graph="query"``); and the Subject Output rule skips the marker
triple itself (as printed, ``?s ?p ?o`` matches ``?s a qq:Variable`` and flags every
selected variable).

Rules 1-6 have a pass and a not-applicable reading too, for the contract's four-way
status: a rule whose published query finds nothing passes when its pattern applied (the
same query without its final ``FILTER NOT EXISTS``), else is not applicable. Rules 7-8
have no filter: no row is not applicable. Every rule's row flags and drives a retry, as
every rule feeds OBQC's repair loop (the ontology-independent Subject Output included).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from typing import Any

from rdflib import BNode, Dataset, Literal, URIRef, Variable
from rdflib.namespace import OWL, RDF, RDFS

from harness.checkloop.feedback import OBQC_FORMAT_ID, obqc_format_sha256
from harness.checks import numbers
from harness.checks.compiler import Output, number_traceability
from harness.checks.outcomes import CheckResult, Outcome, OutputReport
from harness.checks.query import QueryAnalysis, analyse
from harness.sources.model import ClassExpr, OntologySource
from harness.sparql_grammar import graph_query, prepare_query

OBQC_VERSION = "obqc/1"
PAPER = "Allemang and Sequeda (2024), arXiv:2405.11706v1, sections 3.1-3.2"
QQ = "urn:obqc:qq#"  # the reserved namespace; REPLACE(STR(?v), "^.*[/#]", "") gives back the variable's name
QUERY_GRAPH = URIRef("urn:obqc:query")
ONTOLOGY_GRAPH = URIRef("urn:obqc:ontology")
PROLOGUE = ("PREFIX : <urn:obqc:>\nPREFIX qq: <urn:obqc:qq#>\n"
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n")

# The rule queries as printed in arXiv:2405.11706v1 (HTML and PDF agree), verbatim but for the
# indentation of the first line.
PRINTED: dict[str, str] = {
    "obqc.domain": """SELECT ?p ?domain ?s ?class WHERE {
    GRAPH :query{
      ?s ?p ?o .
      ?s a ?class .
    }
    GRAPH :ontology{
      ?p rdfs:domain ?domain.
      FILTER (ISIRI (?domain))
    }
    FILTER NOT EXISTS {
        ?class rdfs:subClassOf* ?domain .
    }
}
""",
    "obqc.range": """SELECT ?p ?range ?s ?class  WHERE {
    GRAPH :query{
        ?s ?p ?o .
        ?o a ?class .
    }
    GRAPH :ontology{
      ?p rdfs:range ?range. FILTER (ISIRI (?range))
    }
    FILTER NOT EXISTS {?class rdfs:subClassOf* ?range .}
}
""",
    "obqc.double_range": """SELECT ?p ?rangep ?q ?rangeq WHERE {
    GRAPH :query {
      ?s1  ?p ?o .
      ?s2  ?q ?o .
    }
    GRAPH :ontology{
      ?p rdfs:range ?rangep.
      ?q rdfs:range ?rangeq.
      FILTER (ISIRI (?rangeq)) FILTER (ISIRI (?rangep))
    }
    FILTER NOT EXISTS {{
      { ?rangep rdfs:subClassOf* ?rangeq .}
      UNION
      { ?rangeq rdfs:subClassOf* ?rangep .}
}}}
""",
    "obqc.double_domain": """SELECT ?p ?domp ?q ?domq WHERE {
    GRAPH :query{
      ?s ?p ?o1 .
      ?s ?q ?o2 .
    }
    GRAPH :ontology{
      ?p rdfs:domain ?domp.
      ?q rdfs:domain ?domq.
      FILTER (ISIRI (?domq)) FILTER (ISIRI (?domp))
    }
    FILTER NOT EXISTS {{
      { ?domp rdfs:subClassOf* ?domq .}
      UNION
      { ?domq rdfs:subClassOf* ?domp .}
}}}
""",
    "obqc.domain_range": """SELECT ?p ?rangep ?q ?domq
WHERE {
    GRAPH :query
    {
      ?s ?p ?o  .
       ?o  ?q ?o2 .
    }
    GRAPH :ontology
    {
      ?p rdfs:range ?rangep .
      ?q rdfs:domain ?domq.
      FILTER (ISIRI (?domq))
      FILTER (ISIRI (?rangep))
    }
    FILTER NOT EXISTS
  {
    {
      { ?rangep rdfs:subClassOf* ?domq .}
      UNION
      { ?domq rdfs:subClassOf* ?rangep .}
      }
    }
}
""",
    "obqc.incorrect_property": """SELECT ?p WHERE {
    GRAPH :query {
    ?s ?p ?o .
    FILTER NOT EXISTS {
      VALUES ?ns {
          "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        "http://www.w3.org/2002/07/owl#"
        "http://www.w3.org/2000/01/rdf-schema#"
          "http://www.w3.org/2004/02/skos/core#"
      }
      FILTER(STRSTARTS(STR(?p), ?ns))
    }}
  FILTER NOT EXISTS {
    GRAPH :ontology {
      ?p a ?type
}}}
""",
    "obqc.iri_output": """SELECT  ?varname WHERE {
    GRAPH :query {
      ?s ?p ?o .
      ?o a qq:Variable .
      BIND  (REPLACE(STR(?o), "^.*[/#]", "") as ?varname)
    }
    GRAPH :query {
        ?p rdfs:range ?range.
        FILTER (ISIRI(?range))
}}
""",
    "obqc.subject_output": """SELECT  ?varname WHERE {
    GRAPH :query {
      ?s ?p ?o .
      BIND  (REPLACE(STR(?s), "^.*[/#]", "") as ?varname)
      ?s a qq:Variable .
}}
""",
}
RULE_IDS = tuple(PRINTED)
# Paper section, name, and the explanation template as printed (sections 3.1.1-3.2); the Domain
# template's {dom} is the query's ?domain.
RULES: dict[str, dict[str, Any]] = {
    "obqc.domain": {"section": "3.1.1", "name": "Domain Rule", "template":
                    "The property {p} has domain {dom}, but its subject {s} is a {class}, which isn’t a subclass "
                    "of {dom}", "vars": ("p", "domain", "s", "class")},
    "obqc.range": {"section": "3.1.2", "name": "Range Rule", "template":
                   "The property {p} has range {range}, but its object {o} is a {class}, which isn’t a subclass "
                   "of {range}", "vars": ("p", "range", "o", "class")},
    "obqc.double_range": {"section": "3.1.3", "name": "Double Range Rule", "template":
                          "The property {p} has range {rangep}, and {q} has range {rangeq}, and these are "
                          "incompatible.", "vars": ("p", "rangep", "q", "rangeq")},
    "obqc.double_domain": {"section": "3.1.4", "name": "Double Domain Rule", "template":
                           "The property {p} has domain {domp}, and {q} has domain {domq}, and these are "
                           "incompatible.", "vars": ("p", "domp", "q", "domq")},
    "obqc.domain_range": {"section": "3.1.5", "name": "Domain Range Rule", "template":
                          "The property {p} has range {rangep}, and {q} has domain {domq}, and these are incompatible "
                          "with the query.", "vars": ("p", "rangep", "q", "domq")},
    "obqc.incorrect_property": {"section": "3.1.6", "name": "Incorrect Property", "template":
                                "The property {p} isn’t defined in the ontology. Please only use properties from "
                                "the ontology, or from a standard source like rdf:, rdfs:, owl:, or skos:",
                                "vars": ("p",)},
    "obqc.iri_output": {"section": "3.2", "name": "IRI Output Rule", "template":
                        "Your selected variable {varname} is an IRI; your output should be something human readable, "
                        "an ID or a label.", "vars": ("varname",)},
    "obqc.subject_output": {"section": "3.2", "name": "Subject Output Rule", "template":
                            "Your selected variable ?{varname} is an IRI (the subject of a triple is always an IRI). "
                            "Your output should be something human readable, an ID or a label.", "vars": ("varname",)},
}
SELECT_RULES = frozenset({"obqc.iri_output", "obqc.subject_output"})
# The named arms (design ruling in the design log's follow-up round): OBQC as published is the primary
# baseline; the WHERE-clause six (rules 1-6) are reported beside it as our variant.
OBQC_ARMS: dict[str, tuple[str, ...]] = {"obqc": RULE_IDS, "obqc-where": RULE_IDS[:6]}
_MARKER_FILTER = "      FILTER (!sameTerm(?o, qq:Variable))\n"
_KIND_TYPE = {"object": OWL.ObjectProperty, "datatype": OWL.DatatypeProperty, "annotation": OWL.AnnotationProperty,
              "unknown": RDF.Property}


def executed_texts(iri_output_graph: str = "ontology") -> dict[str, str]:
    """Each rule's query as run: the prologue and the printed text with the documented edits."""
    if iri_output_graph not in ("ontology", "query"):
        raise ValueError(f"iri_output_graph is 'ontology' or 'query' (as printed), not {iri_output_graph!r}")
    out = {}
    for cid, text in PRINTED.items():
        body = re.sub(r"\ASELECT\s.*?WHERE", "SELECT * WHERE", text, count=1, flags=re.DOTALL)
        if cid == "obqc.iri_output" and iri_output_graph == "ontology":
            first = body.index("GRAPH :query {")
            second = body.index("GRAPH :query {", first + 1)
            body = body[:second] + "GRAPH :ontology {" + body[second + len("GRAPH :query {"):]
        if cid == "obqc.subject_output":
            anchor = "      ?s ?p ?o .\n"
            body = body.replace(anchor, anchor + _MARKER_FILTER, 1)
        out[cid] = PROLOGUE + body
    return out


def _without_last_filter_not_exists(text: str) -> str:
    """The query without its last FILTER NOT EXISTS at the top level of the WHERE group: the
    rule's pattern without its violation test (the paper's strings hold no braces)."""
    depth, last = 0, None
    for m in re.finditer(r"[{}]|FILTER NOT EXISTS", text):
        if m.group(0) == "{":
            depth += 1
        elif m.group(0) == "}":
            depth -= 1
        elif depth == 1:
            last = m.start()
    if last is None:
        raise ValueError("no top-level FILTER NOT EXISTS")
    open_at = text.index("{", last)
    depth = 0
    for i in range(open_at, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[:last] + text[i + 1:]
    raise ValueError("unbalanced braces")


def _show(term: Any) -> str:
    """A term as the feedback shows it: a qq: resource as its variable, an IRI in angle
    brackets, a literal (the rules' ?varname) as its text."""
    if isinstance(term, URIRef):
        return "?" + term[len(QQ):] if term.startswith(QQ) else f"<{term}>"
    if isinstance(term, Literal):
        return str(term)
    return term.n3() if term is not None else ""


def _node(term: Any) -> Any:
    return URIRef(QQ + str(term)) if isinstance(term, Variable) else term


def query_triples(qa: QueryAnalysis) -> list[tuple]:
    """The query's triple patterns as RDF, variables as qq: resources. Every group counts,
    flat: the required part, OPTIONAL and negated parts, and the patterns our fragment sets
    aside (UNION branches, subqueries, EXISTS), as the paper extracts every BGP and ignores
    the logic around it. Paths are read as the fragment reads them (rule 19): a sequence is a
    chain through fresh nodes, an inverse swaps its ends, rdf:type/rdfs:subClassOf* is a type
    triple; an alternative gives one triple per alternative (flat, as a UNION would), and a
    negated property set none (it names what the pattern excludes)."""
    out = []
    for g in qa.groups:
        for e in g.edges:
            if e.negated:
                continue
            preds = [URIRef(p) for p in e.preds] or [_node(e.pvar)]
            out += [(_node(e.s), p, _node(e.o)) for p in preds]
    return sorted(set(out))


class ObqcCompiler:
    """The OBQC rule set over one curated source, in the interface the checker arms call
    (``source``, ``hierarchy``, ``bound``, ``check``); ``configuration`` and
    ``feedback_format`` enter the arm's identity and its checker settings."""

    def __init__(self, source: OntologySource, *, rules: Sequence[str] = RULE_IDS, iri_output_graph: str = "ontology"):
        if source.kind != "curated":
            raise ValueError(f"OBQC reads the curated ontology (the ruling), not a {source.kind} source")
        unknown = [r for r in rules if r not in PRINTED]
        if unknown:
            raise ValueError(f"unknown OBQC rule(s) {unknown}; known: {list(RULE_IDS)}")
        self.source = source
        self.hierarchy = None  # no shared hierarchy: rdfs:subClassOf* reads the source's own axioms
        self.bound = None  # no retrieval: static only
        self.rules = tuple(r for r in RULE_IDS if r in rules)
        self.iri_output_graph = iri_output_graph
        self.texts = executed_texts(iri_output_graph)
        self._violation = {cid: prepare_query(self.texts[cid]) for cid in self.rules}
        self._applies = {cid: prepare_query(_without_last_filter_not_exists(self.texts[cid]))
                         for cid in self.rules if cid not in SELECT_RULES}

    def configuration(self) -> dict[str, Any]:
        texts = {cid: self.texts[cid] for cid in self.rules}
        digest = hashlib.sha256(json.dumps(texts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {"rule_set": "obqc", "version": OBQC_VERSION, "paper": PAPER, "rules": list(self.rules),
                "iri_output_graph": self.iri_output_graph, "queries_sha256": digest,
                "subclass_hierarchy": "the curated source's own rdfs:subClassOf axioms", "footprint": False}

    def feedback_format(self) -> dict[str, str]:
        return {"id": OBQC_FORMAT_ID, "sha256": obqc_format_sha256()}

    # -- the graphs -------------------------------------------------------------------------

    def ontology_triples(self, iris: Iterable[URIRef] | None = None) -> tuple[list[tuple], set[URIRef]]:
        """(the :ontology triples about ``iris``, or about every term when None; the classes
        their domains and ranges name). Only what the rules read: each property's type,
        domains and ranges (a union, an enumeration or a dropped conjunct is a blank node), and
        each class's type, read by the Incorrect Property rule's ``?p a ?type``."""
        src = self.source
        terms = [URIRef(t) for t in (list(src.properties) + list(src.classes))] if iris is None else iris
        out: list[tuple] = []
        named: set[URIRef] = set()

        def expr(e: ClassExpr) -> Any:
            if e.is_union:
                return BNode()
            node = URIRef(e.members[0])
            named.add(node)
            return node

        for t in terms:
            info = src.properties.get(str(t))
            if info is not None:
                out.append((t, RDF.type, _KIND_TYPE.get(info.kind, RDF.Property)))
                out += [(t, RDFS.domain, expr(d)) for d in info.domain]
                out += [(t, RDFS.range, expr(r)) for r in info.range]
                out += [(t, RDFS.range, URIRef(d)) for d in info.datatype_range]
                if info.range_values:
                    out.append((t, RDFS.range, BNode()))
                if info.unsupported_domain:
                    out.append((t, RDFS.domain, BNode()))
                if info.unsupported_range:
                    out.append((t, RDFS.range, BNode()))
            if str(t) in src.classes:
                out.append((t, RDF.type, OWL.Class))
        return out, named

    def subclass_triples(self, classes: Iterable[URIRef] | None = None) -> list[tuple]:
        """The source's subclass axioms on the way up from ``classes`` (all of them when None):
        everything ``rdfs:subClassOf*`` can follow from a class it starts at."""
        src = self.source.classes
        if classes is None:
            return [(URIRef(c), RDFS.subClassOf, URIRef(p)) for c, info in src.items() for p in info.parents]
        out, seen, todo = [], set(), [str(c) for c in classes]
        while todo:
            c = todo.pop()
            if c in seen:
                continue
            seen.add(c)
            for p in src[c].parents if c in src else ():
                out.append((URIRef(c), RDFS.subClassOf, URIRef(p)))
                todo.append(p)
        return out

    def _dataset(self, triples: list[tuple], selected: Sequence[str], slice_ontology: bool) -> Dataset:
        ds = Dataset(default_union=False)
        q = ds.graph(QUERY_GRAPH)
        for t in triples:
            q.add(t)
        for v in selected:  # section 3.2: the SELECT clause's variables, marked in the query graph
            q.add((URIRef(QQ + v), RDF.type, URIRef(QQ + "Variable")))
        iris = {x for t in triples for x in t if isinstance(x, URIRef) and not x.startswith(QQ)}
        onto, named = self.ontology_triples(iris if slice_ontology else None)
        o = ds.graph(ONTOLOGY_GRAPH)
        for t in onto:
            o.add(t)
        for t in self.subclass_triples(iris | named if slice_ontology else None):
            ds.default_graph.add(t)
        return ds

    def _bindings(self, qa: QueryAnalysis, slice_ontology: bool = True) -> dict[str, tuple[list[dict], int | None]]:
        """Per rule: (its flagged bindings, the template variables only, distinct and sorted,
        each with its number of solutions; the number of distinct applicable bindings, None for
        rules 7-8). The ontology is sliced to the terms of the query and their superclasses,
        which every rule pattern starts from, so the slice gives what the whole would."""
        triples = query_triples(qa)
        where = self._dataset(triples, (), slice_ontology)
        select = self._dataset(triples, qa.projected, slice_ontology) if SELECT_RULES & set(self.rules) else None
        out: dict[str, tuple[list[dict], int | None]] = {}
        for cid in self.rules:
            ds = select if cid in SELECT_RULES else where
            names = RULES[cid]["vars"]
            counts: dict[tuple, int] = {}
            for row in graph_query(ds, self._violation[cid]):
                key = tuple(_show(row[n]) for n in names)
                counts[key] = counts.get(key, 0) + 1
            flags = [{**dict(zip(names, k)), "solutions": n} for k, n in sorted(counts.items())]
            applies = None
            if cid in self._applies:
                applies = len({tuple(_show(row[n]) for n in names)
                               for row in graph_query(ds, self._applies[cid])})
            out[cid] = (flags, applies)
        return out

    # -- one output ---------------------------------------------------------------------------

    def check(self, output: Output, client: Any = None, type_map: Any = None,
              qa: QueryAnalysis | None = None) -> OutputReport:
        qa = qa or analyse(output.query)
        name = self.source.name
        results: list[CheckResult] = []
        tooling: list[str] = []
        evidence: dict[str, Any] = {"budget_used": 0}
        if qa.error:  # as the compiler reads it (design log): parsed by nobody, or a tooling limit
            evidence["parse_error"] = qa.error
            executed = output.store_accepted if output.store_accepted is not None else (
                output.result_rows is not None or output.trajectory_rows is not None)
            reason = "parser_mismatch" if executed else "query_parse_error"
            if executed:
                tooling.append("parser_mismatch")
            results += [CheckResult(cid, name, Outcome.UNKNOWN, reason) for cid in self.rules]
        else:
            evidence["obqc"] = {"query_triples": len(query_triples(qa)), "selected": list(qa.projected)}
            for cid, (flags, applies) in self._bindings(qa).items():
                results += self._results(cid, flags, applies)
        results.append(number_traceability(name, output))
        missing = None
        if output.result_rows is not None:
            missing = sum(1 for row in output.answer if not numbers.row_among(row, output.result_rows))
        return OutputReport(source=name, role=output.role, results=results, negation=qa.negation,
                            unsupported_constructs=list(qa.unsupported), unsupported_coverage={}, evidence=evidence,
                            checked_query=output.query, checked_query_status=output.checked_status,
                            answer_rows_not_in_checked_result=missing, tooling_limits=tooling)

    def _results(self, cid: str, flags: list[dict], applies: int | None) -> list[CheckResult]:
        name = self.source.name
        rule = RULES[cid]
        if flags:
            out = []
            for f in flags:
                binding = {k: f[k] for k in rule["vars"]}
                fill = {**binding, "dom": binding.get("domain", "")}
                key = " ".join(f"{k}={binding[k]}" for k in rule["vars"])
                out.append(CheckResult(cid, name, Outcome.FLAG, "obqc_violation", key,
                                       detail=rule["template"].format(**fill),
                                       evidence={"binding": binding, "solutions": f["solutions"],
                                                 "section": rule["section"]}))
            return out
        if cid == "obqc.incorrect_property":
            return [CheckResult(cid, name, Outcome.PASS, "holds" if applies else "no_terms")]
        if applies is None:
            return [CheckResult(cid, name, Outcome.NOT_APPLICABLE, "no_binding")]
        if applies:
            return [CheckResult(cid, name, Outcome.PASS, "holds", evidence={"applicable": applies})]
        return [CheckResult(cid, name, Outcome.NOT_APPLICABLE, "no_axiom")]


def obqc_compilers(source: OntologySource) -> dict[str, ObqcCompiler]:
    """Both named OBQC arms over one curated source (OBQC_ARMS), for the replay matrix and the bank."""
    return {name: ObqcCompiler(source, rules=rules) for name, rules in OBQC_ARMS.items()}
