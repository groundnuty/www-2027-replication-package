#!/usr/bin/env python3
"""Build the shared list of W3C built-in terms (check contract rule 36).

Rule 36 (as narrowed in change-log entry 25) exempts from vocabulary existence
only the terms DEFINED in the published RDF, RDFS, OWL 2 and XSD vocabularies,
not whole namespaces. The compiler and the fault-bank labeller both read the
list this script writes, so the two cannot drift; the labeller reads the data
file, never this code.

RDF, RDFS and OWL terms are the subjects in their own namespace in the W3C
vocabulary files kept beside the output (data/vocab/w3c/, fetched from
www.w3.org with Accept: text/turtle; their sha256 is recorded). XSD has no
RDF vocabulary file, so its terms are the XML Schema datatypes that RDF 1.1
Concepts (W3C Recommendation, 25 February 2014, section 5.1) lists as usable
in RDF, plus xsd:anySimpleType / anyType named there as unsuitable but still
defined.

    uv run python scripts/build_w3c_terms.py   # writes data/vocab/w3c-terms.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rdflib import Graph

ROOT = Path(__file__).resolve().parent.parent
VOCAB = ROOT / "data" / "vocab"
FILES = {
    "rdf": ("rdf.ttl", "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "https://www.w3.org/1999/02/22-rdf-syntax-ns.ttl"),
    "rdfs": ("rdfs.ttl", "http://www.w3.org/2000/01/rdf-schema#", "https://www.w3.org/2000/01/rdf-schema.ttl"),
    "owl": ("owl.ttl", "http://www.w3.org/2002/07/owl#", "https://www.w3.org/2002/07/owl.ttl"),
}
XSD = "http://www.w3.org/2001/XMLSchema#"
# RDF 1.1 Concepts, section 5.1 (https://www.w3.org/TR/rdf11-concepts/#xsd-datatypes).
XSD_TERMS = [
    "string", "boolean", "decimal", "integer", "double", "float", "date", "time", "dateTime", "dateTimeStamp",
    "gYear", "gMonth", "gDay", "gYearMonth", "gMonthDay", "duration", "yearMonthDuration", "dayTimeDuration",
    "byte", "short", "int", "long", "unsignedByte", "unsignedShort", "unsignedInt", "unsignedLong",
    "positiveInteger", "nonNegativeInteger", "negativeInteger", "nonPositiveInteger", "hexBinary", "base64Binary",
    "anyURI", "language", "normalizedString", "token", "NMTOKEN", "Name", "NCName",
    # listed in the same section as not suitable for use in RDF, but defined by XML Schema
    "QName", "ENTITY", "ID", "IDREF", "NOTATION", "IDREFS", "ENTITIES", "NMTOKENS", "anySimpleType", "anyType",
]


def build() -> dict:
    sources, terms = {}, set()
    for key, (name, ns, url) in FILES.items():
        path = VOCAB / "w3c" / name
        data = path.read_bytes()
        g = Graph().parse(data=data, format="turtle")
        found = {str(s) for s in g.subjects() if str(s).startswith(ns) and str(s) != ns}
        terms |= found
        sources[key] = {"url": url, "file": f"w3c/{name}", "sha256": hashlib.sha256(data).hexdigest(),
                        "namespace": ns, "terms": len(found)}
    terms |= {XSD + t for t in XSD_TERMS}
    sources["xsd"] = {"spec": "https://www.w3.org/TR/rdf11-concepts/#xsd-datatypes", "namespace": XSD,
                      "terms": len(XSD_TERMS)}
    return {"schema": "w3c-terms/1", "rule": "check contract rule 36 (change-log entry 25)",
            "sources": sources, "terms": sorted(terms)}


def main() -> int:
    out = VOCAB / "w3c-terms.json"
    out.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
