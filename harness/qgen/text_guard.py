"""Question-text guard: no database names.

A question names concepts ("keyword", "Gene Ontology term", "EC number",
"subcellular location"), never the databases that hold them, so the text does
not point at a source or a schema. ``database_names(text)`` lists the names
found (whole words, case-insensitive, plus the upper-case abbreviation "GO");
profiles with ``forbid_database_names`` reject candidates whose text has any
and fail generation if a written question still does.
"""

from __future__ import annotations

import re

DATABASE_NAMES = (
    "Rhea", "ChEBI", "UniProt", "UniProtKB", "Swiss-Prot", "SwissProt", "TrEMBL", "UniRef", "UniParc",
    "PubMed", "MEDLINE", "PDB", "wwPDB", "PDBe", "InterPro", "Pfam", "PROSITE", "SMART", "KEGG", "MetaCyc",
    "BioCyc", "EcoCyc", "Reactome", "IntAct", "STRING", "Ensembl", "RefSeq", "EMBL", "GenBank", "DDBJ", "NCBI",
    "ExPASy", "MACiE", "IUBMB", "BRENDA", "OMIM", "Orphanet", "QuickGO", "AmiGO",
)
_NAMES = re.compile(r"(?<![\w-])(" + "|".join(re.escape(n) for n in DATABASE_NAMES) + r")(?![\w-])", re.IGNORECASE)
_GO = re.compile(r"(?<![\w-])GO(?![\w-])")


# Names a private graph adds at run time (its project, organisation, sources and
# tools), read from a private file and never written into code.
_PRIVATE: list[re.Pattern] = []


def register_private_names(names: list[str]) -> None:
    """Add names (whole words, case-insensitive) that question texts must not contain."""
    _PRIVATE.clear()
    clean = sorted({n.strip() for n in names if n.strip()})
    if clean:
        _PRIVATE.append(re.compile(r"(?<![\w-])(" + "|".join(re.escape(n) for n in clean) + r")(?![\w-])", re.IGNORECASE))


def database_names(text: str, extra: tuple[str, ...] = ()) -> list[str]:
    """Database names in ``text`` (empty when clean); ``extra`` adds a profile's own names, matched the same way."""
    found = [m.group(1) for m in _NAMES.finditer(text)]
    found += [m.group(0) for m in _GO.finditer(text)]
    for rx in _PRIVATE:
        found += [m.group(1) for m in rx.finditer(text)]
    if extra:
        own = re.compile(r"(?<![\w-])(" + "|".join(re.escape(n) for n in extra) + r")(?![\w-])", re.IGNORECASE)
        found += [m.group(1) for m in own.finditer(text)]
    return found
