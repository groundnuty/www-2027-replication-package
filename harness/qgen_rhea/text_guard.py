"""Question-text guard: no database names.

A question names concepts ("keyword", "Gene Ontology term", "EC number",
"subcellular location"), never the databases that hold them, so the text does
not point at a source or a schema. ``database_names(text)`` lists the names
found (whole words, case-insensitive, plus the upper-case abbreviation "GO");
profiles with ``forbid_database_names`` reject candidates whose text has any
and fail generation if a written question still does.

``record_database_names(record)`` checks every agent-visible field of a
question record (AGENT_VISIBLE_FIELDS), identifiers (answer-column names,
scoring-rule keys) split into words first.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from typing import Any

DATABASE_NAMES = (
    "Rhea", "ChEBI", "UniProt", "UniProtKB", "Swiss-Prot", "SwissProt", "TrEMBL", "UniRef", "UniParc",
    "PubMed", "MEDLINE", "PDB", "wwPDB", "PDBe", "InterPro", "Pfam", "PROSITE", "SMART", "KEGG", "MetaCyc",
    "BioCyc", "EcoCyc", "Reactome", "IntAct", "STRING", "Ensembl", "RefSeq", "EMBL", "GenBank", "DDBJ", "NCBI",
    "ExPASy", "MACiE", "IUBMB", "BRENDA", "OMIM", "Orphanet", "QuickGO", "AmiGO",
)
_NAMES = re.compile(r"(?<![\w-])(" + "|".join(re.escape(n) for n in DATABASE_NAMES) + r")(?![\w-])", re.IGNORECASE)
_GO = re.compile(r"(?<![\w-])GO(?![\w-])")


def database_names(text: str) -> list[str]:
    """Database names in ``text`` (empty when clean)."""
    found = [m.group(1) for m in _NAMES.finditer(text)]
    found += [m.group(0) for m in _GO.finditer(text)]
    return found


# The record fields an agent can see. The runner renders only "question", as the user message
# (harness/agent/prompts.user_message, fed by scripts/run_pilot.attempt_question, feat/e0.1-skeleton 3211632);
# the checker loop (feat/checker-loop d0ecf78) matches with "scoring_rules" without rendering them. The column
# names and scoring rules are guarded too, so no later prompt or checker message can leak a name through them.
AGENT_VISIBLE_FIELDS = ("question", "answer_columns", "scoring_rules")
_SEPARATORS = re.compile(r"[_\-\s.]+")
# Case changes inside a word: "uniprot|Accession", "PDB|Id", "Ch|EBI" (a name is then found as a run of the parts).
_CASE_CHANGE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def identifier_database_names(name: str, names: Callable[[str], list[str]] | None = None) -> list[str]:
    """Database names in an identifier ("chebi_label", "uniprotAccession", "PubMedCount", "go_term").

    The identifier is cut at "_", "-", "." and spaces and at case changes, and every run of consecutive pieces,
    across the separators too, is checked as a word, written solid and with hyphens, so a name written in mixed
    case ("PubMed", "ChEBI", "UniProtKB"), split by separators ("pub_med", "Ch_EBI") or hyphenated
    ("Swiss-Prot_accession") is found whole (external reviews of 01eeb51 and b50e77d); a lower-case "go" counts
    here, since identifiers are lower case.
    """
    names = names or database_names
    found: list[str] = []
    pieces = [p for part in _SEPARATORS.split(name) for p in _CASE_CHANGE.split(part) if p]
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces) + 1):
            for run in dict.fromkeys(("".join(pieces[i:j]), "-".join(pieces[i:j]))):
                hits = names(run) or ([run] if run.lower() == "go" else [])
                found += [h for h in hits if h not in found]
    return found


def _strings(value: Any, path: str) -> Iterator[tuple[str, str, bool]]:
    """(path, string, is_identifier) for the keys and string values in a JSON value."""
    if isinstance(value, str):
        yield path, value, False
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield f"{path}.{key}", str(key), True
            yield from _strings(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _strings(item, f"{path}[{i}]")


def record_database_names(record: Mapping[str, Any], names: Callable[[str], list[str]] | None = None) -> dict[str, list[str]]:
    """Database names in the agent-visible fields of a question record, by field path (empty when clean)."""
    names = names or database_names
    out: dict[str, list[str]] = {}
    text = record.get("question")
    if isinstance(text, str) and (found := names(text)):
        out["question"] = found
    for i, column in enumerate(record.get("answer_columns") or ()):
        name = column.get("name") if isinstance(column, Mapping) else None
        if isinstance(name, str) and (found := identifier_database_names(name, names)):
            out[f"answer_columns[{i}].name"] = found
    for path, value, is_identifier in _strings(record.get("scoring_rules") or {}, "scoring_rules"):
        found = identifier_database_names(value, names) if is_identifier else names(value)
        if found:
            out[path] = sorted(set(out.get(path, [])) | set(found))
    return out
