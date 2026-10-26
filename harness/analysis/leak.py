"""The leak exclusion sensitivity of C1 (pre-registration §5 "Prompt renders", change-log entry 91): a descriptive report.

The rule, pre-stated: if a test question has a gold answer value (not an anchor) verbatim in one compared arm's prompt
only, C1 is also reported excluding those questions, the set computed at analysis time from the frozen prompts and
gold. The compared arms are C1's, the curated and the census prompt; the prompts are read only from the config's files
by their frozen sha256 (``c1_leak_prompts``), and the gold from the question records the run analysed (the frozen
question set, by its hash).

**Matching** is the leak audit's (feat/freeze-check results/leak-audit-20261019/leak_audit.py, the audit behind entry
91's counts), its functions carried over unchanged and checked against it (tests/test_analysis_leak.py):
- IRIs are compared as sets, never as substrings: from a prompt every full IRI (``<...>`` or bare ``http(s)://...``,
  trailing ``.,;:)`` stripped) and every CURIE whose prefix the prompt declares (``PREFIX p: <ns>``) is expanded; the
  conventional CURIEs ``CHEBI:``, ``GO:`` and ``RHEA:`` are expanded too when the prompt does not declare them.
- Gold literals are normalised as the scorer does: strings by their exact lexical form (case counts), numbers by value
  (no leading ``+``, trailing fractional zeros dropped, leading zeros kept), every other datatype by its stripped
  lexical form; a string, date or other literal hits when non-word characters (or the text's ends) bound it.
- The trivial-value filter: a value is ignored when it is a boolean, shorter than 4 characters, or a pure number with
  at most 3 digits; IRIs are never filtered.

A question is in the set when, after the trivial-value filter, a gold value of it is in one compared prompt and none
is in the other (the audit's paired a-only and b-only questions, measure gold_f). Its base question leaves C1a and C1b, which are recomputed
on the remaining questions with the registered code (prereg.c1a, prereg.c1b on the frozen cluster map's resampling);
the report gives the estimates, the raw p's and the 95% intervals, never a claim: it is in no registry (calibrated,
grouping or multiplicity) and the confirmatory family does not read it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harness.analysis.metrics import ArmKey, Unit, question_vectors
from harness.analysis.prereg import (
    PLACEHOLDER,
    TESTS,
    ClusterMap,
    PreregConfig,
    PreregError,
    Vectors,
    _interval,
    bounded,
    c1a,
    c1b,
)
from harness.analysis.stats import QuestionIndex

# --- the leak audit's matching rules (leak_audit.py, unchanged) ------------------------------------------------------

XSD = "http://www.w3.org/2001/XMLSchema#"
NUMERIC_DATATYPES = {XSD + t for t in (
    "integer", "int", "long", "short", "byte", "decimal", "float", "double", "nonNegativeInteger",
    "positiveInteger", "nonPositiveInteger", "negativeInteger", "unsignedInt", "unsignedLong",
    "unsignedShort", "unsignedByte")}
TEMPORAL_DATATYPES = {XSD + "date", XSD + "dateTime"}
CONVENTIONAL_PREFIXES = {
    "CHEBI": "http://purl.obolibrary.org/obo/CHEBI_",
    "GO": "http://purl.obolibrary.org/obo/GO_",
    "RHEA": "http://rdf.rhea-db.org/",
}
NUM_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")
PREFIX_RE = re.compile(r"PREFIX\s+([A-Za-z][\w.-]*)?:\s*<([^<>\s]+)>", re.IGNORECASE)
FULL_IRI_RE = re.compile(r"<(https?://[^<>\s\"]+)>|(?<![\w<])(https?://[^\s<>\"'(),;|]+)")
CURIE_RE = re.compile(r"(?<![\w/#:.<-])([A-Za-z][\w-]*(?:\.[\w-]+)*)?:(\w[\w.\-%]*)")
NUMBER_TOKEN_RE = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w]|\.\d)")


def canon_number(lexical: str) -> str:
    """The canonical form numbers are compared by: no '+', no trailing fractional zeros."""
    s = lexical.strip().lstrip("+")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def trivial(kind: str, lexical: str) -> bool:
    """The trivial-value filter (see the module docstring). IRIs are never trivial."""
    if kind == "iri":
        return False
    s = lexical.strip()
    if kind == "boolean" or s.lower() in ("true", "false"):
        return True
    if len(s) < 4:
        return True
    return bool(NUM_RE.fullmatch(s)) and sum(ch.isdigit() for ch in s) <= 3


def cell_kind(cell: dict[str, Any]) -> str:
    if cell.get("type") == "iri":
        return "iri"
    if cell.get("type") == "bnode":
        return "bnode"
    dt = cell.get("datatype")
    if cell.get("lang") or dt in (None, XSD + "string"):
        return "string"
    if dt in NUMERIC_DATATYPES:
        return "number"
    if dt == XSD + "boolean":
        return "boolean"
    if dt in TEMPORAL_DATATYPES:
        return "temporal"
    return "other"


def gold_values(q: dict[str, Any]) -> list[tuple[str, str]]:
    """(kind, lexical) of every bound cell in every gold row, deduplicated per question."""
    out = []
    for row in q["gold_rows"]:
        for cell in row:
            if cell is None:
                continue
            kind = cell_kind(cell)
            if kind == "bnode":
                continue
            out.append((kind, cell["value"]))
    return sorted(set(out))


def declared_prefixes(text: str) -> dict[str, str]:
    return {(p or ""): ns for p, ns in PREFIX_RE.findall(text)}


def extract_iris(text: str, prefixes: dict[str, str] | None = None) -> set[str]:
    """Every full IRI in the text and every CURIE expanded by the text's own (or the conventional) prefixes."""
    declared = declared_prefixes(text) if prefixes is None else prefixes
    table = {**CONVENTIONAL_PREFIXES, **declared}
    iris = set()
    for bracketed, bare in FULL_IRI_RE.findall(text):
        iri = bracketed or bare.rstrip(".,;:)]}")
        iris.add(iri)
    for prefix, local in CURIE_RE.findall(text):
        local = local.rstrip(".")
        if prefix in table and local:
            iris.add(table[prefix] + local)
    return iris


def boundary_pattern(value: str, ignore_case: bool) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)", re.IGNORECASE if ignore_case else 0)


class Text:
    """What a prompt holds: its IRIs (as a set), its number tokens (canonical), and its literals by boundary search."""

    def __init__(self, name: str, raw: str, prefixes: dict[str, str] | None = None) -> None:
        self.name = name
        self.raw = raw
        self.iris = frozenset(extract_iris(raw, prefixes))
        self.numbers = frozenset(canon_number(t) for t in NUMBER_TOKEN_RE.findall(raw))
        self._cache: dict[tuple[str, bool], bool] = {}

    def _search(self, value: str, ignore_case: bool) -> bool:
        key = (value, ignore_case)
        hit = self._cache.get(key)
        if hit is None:
            hit = self._cache[key] = bool(value) and boundary_pattern(value, ignore_case).search(self.raw) is not None
        return hit

    def has_iri(self, iri: str) -> bool:
        return iri in self.iris

    def has_literal(self, kind: str, lexical: str) -> bool:
        if kind == "number":
            return canon_number(lexical) in self.numbers
        value = lexical if kind == "string" else lexical.strip()
        return self._search(value, ignore_case=False)


# --- the rule ---------------------------------------------------------------------------------------------------------


def gold_hits(q: dict[str, Any], text: Text) -> set[tuple[str, str]]:
    """The question's gold values in the text after the trivial-value filter (the audit's gold_f)."""
    return {(k, v) for k, v in gold_values(q) if (text.has_iri(v) if k == "iri" else text.has_literal(k, v))
            and not trivial(k, v)}


def _base(qid: str, q: Mapping[str, Any]) -> str:
    return q.get("base_question_id") or qid


def leaked_questions(questions: Mapping[str, dict[str, Any]], prompts: Mapping[str, str],
                     bases: Sequence[str]) -> dict[str, Any]:
    """The analysed questions (those of the base questions ``bases``) with a gold answer value verbatim in one of the
    two compared prompts only (``prompts``: exactly two, cohort -> text): their base questions, and per prompt how many
    analysed questions it holds a gold value of."""
    if len(prompts) != 2:
        raise PreregError(f"the leak rule compares two prompts (C1's), got {sorted(prompts)}")
    texts = {cohort: Text(cohort, raw) for cohort, raw in prompts.items()}
    hit: dict[str, set[str]] = {cohort: set() for cohort in texts}
    analysed = set(bases)
    considered = sorted(qid for qid, q in questions.items() if _base(qid, q) in analysed)
    for qid in considered:
        for cohort, text in texts.items():
            if gold_hits(questions[qid], text):
                hit[cohort].add(qid)
    a, b = (hit[c] for c in texts)
    return {"bases": sorted({_base(q, questions[q]) for q in a ^ b}), "hits": {c: len(h) for c, h in hit.items()},
            "n_questions": len(considered)}


def frozen_prompts(cfg: PreregConfig, root: Path, *, required: bool) -> dict[str, str] | None:
    """C1's two compared prompts (the curated and the census prompt cohort), read from the config's files under
    ``root`` by their frozen sha256. None when the config names none and ``required`` is false (a test run); refused
    while any entry is [FILL], when the entries are not exactly the two cohorts, or when a file's sha256 differs."""
    spec = cfg.c1_leak_prompts
    if not spec:
        if required:
            raise PreregError("the frozen prompts of C1's compared arms are not in the config (c1_leak_prompts): the "
                              "leak exclusion (prereg §5, entry 91) is computed from them at analysis time")
        return None
    cohorts = {cfg.curated_prompt_cohort, cfg.census_prompt_cohort}
    if set(spec) != cohorts:
        raise PreregError(f"c1_leak_prompts must name exactly C1's prompt cohorts {sorted(cohorts)}, got {sorted(spec)}")
    out = {}
    for cohort in (cfg.curated_prompt_cohort, cfg.census_prompt_cohort):
        entry = spec[cohort]
        if not isinstance(entry, Mapping) or PLACEHOLDER in (entry.get("file"), entry.get("sha256")):
            raise PreregError(f"c1_leak_prompts[{cohort!r}] is not frozen (its file and sha256 [FILL])")
        raw = (Path(root) / entry["file"]).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != entry["sha256"]:
            raise PreregError(f"{entry['file']}: sha256 {actual} is not the frozen {cohort} prompt's {entry['sha256']}")
        out[cohort] = raw.decode("utf-8")
    return out


def _components(contrast: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"label": c["label"], "estimate": c["estimate"], "se": c["se"],
             "ci95": {t: _interval(c["_s"], 0.05, "two", t) for t in TESTS}} for c in contrast["components"]]


def c1_excluding(arms: Mapping[ArmKey, Sequence[Unit]], index: QuestionIndex, cfg: PreregConfig, clusters: ClusterMap,
                 excluded: Sequence[str]) -> dict[str, Any]:
    """C1a and C1b (the registered code, on the cluster map's resampling drawn from the registered seed) on the index
    without the ``excluded`` base questions: estimates, raw p's and 95% intervals per component. Descriptive."""
    gone = set(excluded)
    keep = [i for i, q in enumerate(index.questions) if q not in gone]
    reduced = QuestionIndex(tuple(index.questions[i] for i in keep), tuple(index.strata[i] for i in keep))
    keys = [cfg.prompt(m, c) for m in cfg.prompt_models() for c in (cfg.curated_prompt_cohort, cfg.census_prompt_cohort)]
    V = Vectors({k: question_vectors([u for u in arms[k] if u.cluster not in gone], reduced) for k in keys}, {}, {})
    K = clusters.resampling(reduced, cfg)
    out = {}
    for contrast in (c1a(V, K, cfg), c1b(V, K, cfg)):
        out[contrast["id"]] = {"sided": contrast["sided"], "p_test": contrast["p_test"],
                               **({"direction": contrast["direction"]} if "direction" in contrast else {}),
                               "components": _components(contrast)}
    return {"n_questions": len(reduced), "clusters": K.summary(), **bounded(out)}


def report(arms: Mapping[ArmKey, Sequence[Unit]], index: QuestionIndex, cfg: PreregConfig,
           questions: Mapping[str, dict[str, Any]] | None, prompts: Mapping[str, str] | None,
           clusters: ClusterMap) -> dict[str, Any]:
    """The leak exclusion's report on the analysed questions (``index``): how many each compared prompt holds a gold
    value of, the excluded base questions (one compared prompt only), and C1 recomputed without them when there are
    any."""
    rule = ("pre-stated (prereg §5 'Prompt renders', change-log entry 91): a test question with a gold answer value "
            "(not an anchor) verbatim in one compared arm's prompt only leaves C1 in this second, descriptive report; "
            "matched by the leak audit's rules, the trivial-value filter applied")
    if prompts is None or questions is None:
        return {"descriptive": True, "rule": rule, "computed": False,
                "reason": "a test run without the config's frozen prompts or the question records"}
    found = leaked_questions(questions, prompts, index.questions)
    out = {"descriptive": True, "rule": rule, "computed": True,
           "prompts": {c: dict(cfg.c1_leak_prompts[c]) for c in prompts}, "n_questions": found["n_questions"],
           "questions_with_a_gold_value_in_the_prompt": found["hits"], "excluded": len(found["bases"]),
           "excluded_questions": found["bases"]}
    if not found["bases"]:
        return {**out, "contrasts": None,
                "reading": "no question has a gold answer value in one compared prompt only: C1 is reported once"}
    return {**out, "contrasts": c1_excluding(arms, index, cfg, clusters, found["bases"])}
